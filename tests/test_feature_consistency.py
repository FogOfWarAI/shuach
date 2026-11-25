# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Regression tests for consistency between contract, validation and encoders.

These lock down findings from an independent review: the two timing paths must
agree, augmentation must preserve declared semantics, and declarations that a
model cannot honour must be rejected at construction time.
"""

import json
import unittest

import numpy as np

from gamematch.features import (
    SCHEMAS,
    FeatureEncoder,
    FeatureSchema,
    FeatureSpec,
    Modality,
    Orientation,
    OutputSpec,
    Scope,
    sample_inputs,
    validate_ids,
)

try:
    import torch

    from gamematch.features import make_model_from_schema, swap_sides

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - base install
    TORCH_AVAILABLE = False


class TimingConsistencyTests(unittest.TestCase):
    def test_at_and_validate_agree_on_legacy_feature_name_key(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=3, seed=0)
        times = np.asarray(batch.available_at["history_available_at"]).copy()
        times[0, 0, 0, 0] = float(np.asarray(batch.prediction_time).reshape(-1)[0])
        batch.available_at = {"history": times}
        # validate accepts the legacy key
        with self.assertRaises(ValueError):
            batch.validate()
        # at() resolves the same key and excludes the same field
        view = schema.at(
            batch.prediction_time, batch.available_at,
            batch=batch.batch_size, masks=batch.masks,
        )
        self.assertIn("history", view.excluded)

    def test_scalar_prediction_time_applies_to_every_row(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=3, seed=0, prediction_time=100.0)
        times = np.full((3, 2, 6, 10), 50.0)
        times[2, 0, 0, 0] = 200.0        # leak in the LAST row only
        batch.available_at = {"history_available_at": times}
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            batch.validate()

    def test_masked_padding_does_not_count_as_leak(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=3, seed=0, prediction_time=100.0)
        pt = float(np.asarray(batch.prediction_time).reshape(-1)[0])
        times = np.full((3, 2, 6, 10), pt)          # padding tied with t
        times[np.asarray(batch.masks["history"], dtype=bool)] = pt - 86400.0
        batch.available_at = {"history_available_at": times}
        batch.validate()                            # observed steps are earlier
        view = schema.at(
            pt, batch.available_at, batch=batch.batch_size, masks=batch.masks,
        )
        self.assertNotIn("history", view.excluded)

    def test_reference_encoder_respects_at(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=3, seed=0, prediction_time=100.0)
        safe = FeatureEncoder(schema, dim=6, seed=1)
        safe_tokens = np.asarray(safe(batch).tokens).copy()
        pt = float(np.asarray(batch.prediction_time).reshape(-1)[0])
        batch.available_at["history_available_at"] = np.full((3, 2, 6, 10), pt + 1.0)
        hidden = np.asarray(safe(batch).tokens).copy()
        self.assertFalse(np.allclose(safe_tokens, hidden))

    def test_at_requires_prediction_time_and_rejects_unknown_keys(self):
        schema = SCHEMAS["overwatch_history"]
        with self.assertRaises(ValueError):
            schema.at(None, {})
        with self.assertRaisesRegex(ValueError, "unknown feature"):
            schema.at(100.0, {"not_a_field": np.zeros((2, 2))})


class DeclarationConsistencyTests(unittest.TestCase):
    def test_unimplemented_reductions_rejected(self):
        for reduce in ("max", "last"):
            with self.assertRaisesRegex(ValueError, "reduce"):
                FeatureSpec(
                    "x", Modality.SEQUENCE, Scope.PLAYER, 3,
                    steps=4, mask=True, reduce=reduce,
                )

    def test_json_roundtrip_is_idempotent(self):
        for schema in SCHEMAS.values():
            first = FeatureSchema.from_dict(schema.to_dict())
            second = FeatureSchema.from_dict(first.to_dict())
            self.assertEqual(first.to_dict(), second.to_dict(), schema.name)

    def test_multiclass_dim_inferred_from_labels(self):
        payload = SCHEMAS["overwatch_history"].to_dict()
        payload["task"].pop("dim")
        restored = FeatureSchema.from_dict(payload)
        self.assertEqual(restored.task.dim, len(restored.task.labels))

    def test_antisymmetric_multiclass_receives_side_information(self):
        schema = SCHEMAS["overwatch_history"]
        self.assertEqual(schema.task.resolved_orientation, Orientation.ANTISYMMETRIC)
        self.assertIsNone(schema.task.orientation)  # serialization stays raw


@unittest.skipUnless(TORCH_AVAILABLE, "requires the neural extra")
class AugmentationAndHeadTests(unittest.TestCase):
    def test_wdl_swap_changes_logits_and_reverses_labels(self):
        schema = SCHEMAS["overwatch_history"]
        model = make_model_from_schema(schema, dim=8).eval()
        batch = sample_inputs(schema, batch=5, seed=2)
        swapped = swap_sides(schema, batch)
        with torch.no_grad():
            a, b = model(batch), model(swapped)
        self.assertFalse(torch.allclose(a, b))
        self.assertTrue(np.allclose(
            np.asarray(batch.targets) + np.asarray(swapped.targets),
            schema.task.dim - 1,
        ))

    def test_swap_does_not_reverse_time_axis_of_match_sequence(self):
        schema = FeatureSchema(
            "timed_match", 2, 3,
            (FeatureSpec("clock", Modality.SEQUENCE, Scope.MATCH, 1,
                         steps=2, mask=True),),
            OutputSpec("win", "binary", orientation=Orientation.SYMMETRIC),
        )
        batch = sample_inputs(schema, batch=2, seed=0)
        field = np.asarray(batch.fields["clock"]).reshape(2, 2)
        field[:] = [[10, 20], [30, 40]]
        batch.fields["clock"] = field
        swapped = swap_sides(schema, batch)
        self.assertTrue(np.array_equal(
            np.asarray(swapped.fields["clock"]).reshape(2, 2), field))

    def test_regression_dim_gt_one_head_and_targets_match(self):
        schema = FeatureSchema(
            "multi_reg", 1, 1,
            (FeatureSpec("x", Modality.DENSE, Scope.MATCH, 3),),
            OutputSpec("values", "regression", dim=3, link="identity",
                       orientation=Orientation.NONE),
        )
        model = make_model_from_schema(schema, dim=6)
        batch = sample_inputs(schema, batch=4, seed=0)
        batch.validate()
        logits = model(batch)
        targets = model.targets(batch)
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(targets.shape), (4, 3))
        self.assertTrue(torch.isfinite(model.criterion()(logits, targets)))

    def test_one_hot_targets_are_accepted(self):
        schema = SCHEMAS["overwatch_history"]
        model = make_model_from_schema(schema, dim=6)
        batch = sample_inputs(schema, batch=4, seed=0)
        classes = np.random.default_rng(0).integers(0, schema.task.dim, size=batch.batch_size)
        batch.targets = np.eye(schema.task.dim)[classes]
        batch.validate()
        targets = model.targets(batch)
        self.assertEqual(tuple(targets.shape), (4,))
        self.assertTrue(torch.isfinite(
            model.criterion()(model(batch), targets)))

    def test_team_dense_has_no_dead_slot_parameter(self):
        schema = FeatureSchema(
            "team_only", 2, 1,
            (FeatureSpec("team_form", Modality.DENSE, Scope.TEAM, 4),),
            OutputSpec("win", "binary"),
        )
        model = make_model_from_schema(schema, dim=6)
        self.assertNotIn("team_form", dict(model.encoder.slot_embeddings))



class VocabularyValidationTests(unittest.TestCase):
    """Data-time vocabulary checks must use real participants, not padding."""

    def test_padding_does_not_consume_a_vocabulary_row(self):
        schema = SCHEMAS["hero_binary"]         # vocab_size=114
        ids = np.zeros((4, 2, 5), dtype=int)
        ids[0, 0, :3] = [1, 2, 3]                # side 0 has 3 real players
        ids[0, 1, :2] = [4, 5]                   # side 1 has 2 (unequal teams)
        ids[1, 0, :5] = [6, 7, 8, 9, 10]
        ids[1, 1, :5] = [11, 12, 13, 14, 15]
        report = validate_ids(schema, {"hero_id": ids})
        self.assertTrue(report["ok"])
        self.assertEqual(report["fields"]["hero_id"]["required_rows"], 16)
        self.assertEqual(report["fields"]["hero_id"]["headroom"], 114 - 16)

    def test_out_of_vocabulary_id_reported_by_name(self):
        schema = SCHEMAS["hero_binary"]
        ids = np.array([[[1, 2, 999, 0, 0], [3, 4, 5, 6, 7]]])
        report = validate_ids(schema, {"hero_id": ids}, strict=False)
        self.assertFalse(report["ok"])
        self.assertEqual(report["fields"]["hero_id"]["out_of_vocabulary"], [999])
        with self.assertRaisesRegex(ValueError, "999"):
            validate_ids(schema, {"hero_id": ids})

    def test_validity_mask_excludes_non_participants(self):
        schema = SCHEMAS["hero_binary"]
        ids = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
        mask = np.ones((1, 2, 5), dtype=bool)
        mask[0, 1, 3:] = False                   # side 1 has only 3 players
        ids[0, 1, 3:] = 999                      # padding carries a junk ID
        report = validate_ids(schema, {"hero_id": ids}, masks={"hero_id": mask})
        self.assertTrue(report["ok"])
        self.assertNotIn(999, report["fields"]["hero_id"]["observed_ids"])

    def test_exact_capacity_with_unknown_bucket(self):
        schema = SCHEMAS["hero_binary"]
        ids = np.arange(1, 114).reshape(1, 1, 113)[..., None]
        ids = np.pad(ids, ((0, 0), (0, 1), (0, 0), (0, 0)))
        report = validate_ids(schema, {"hero_id": ids})
        self.assertEqual(report["fields"]["hero_id"]["required_rows"], 114)
        self.assertEqual(report["fields"]["hero_id"]["headroom"], 0)

    def test_unequal_masks_are_rejected(self):
        schema = SCHEMAS["hero_binary"]
        ids = np.ones((2, 2, 5), dtype=int)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            validate_ids(schema, {"hero_id": ids}, masks={"hero_id": np.ones((2, 2))})


class ReproducibilityTests(unittest.TestCase):
    """A run artifact must identify the feature contract that produced it."""

    def test_content_hash_is_stable_and_sensitive(self):
        from dataclasses import replace
        schema = SCHEMAS["hero_binary"]
        self.assertEqual(schema.content_hash(), FeatureSchema.from_dict(schema.to_dict()).content_hash())
        changed = replace(schema, features=(
            replace(schema.features[0], vocab_size=200),
        ))
        self.assertNotEqual(schema.content_hash(), changed.content_hash())

    def test_provenance_reports_undeclared_sources(self):
        from dataclasses import replace
        schema = SCHEMAS["hero_binary"]
        declared = replace(schema, features=(replace(schema.features[0], source="uci-dota2:roster"),))
        prov = declared.provenance()
        self.assertEqual(prov["undeclared_sources"], [])
        self.assertEqual(prov["fields"]["hero_id"]["source"], "uci-dota2:roster")
        self.assertEqual(prov["content_hash"], declared.content_hash())
        self.assertEqual(SCHEMAS["hero_binary"].provenance()["undeclared_sources"], ["hero_id"])

    def test_schema_hash_survives_json_roundtrip(self):
        for schema in SCHEMAS.values():
            with self.subTest(schema=schema.name):
                self.assertEqual(
                    schema.content_hash(),
                    FeatureSchema.from_dict(json.loads(schema.to_json())).content_hash(),
                )


if __name__ == "__main__":
    unittest.main()


class ComparabilityTests(unittest.TestCase):
    """A model comparison must declare the input all models share."""

    def setUp(self):
        from gamematch.comparability import assert_comparable, input_contract

        self.contract = input_contract
        self.assert_comparable = assert_comparable

    class _Vocab:
        size = 115
        values = [1, 2, 3, 4, 5]

    def _contract(self, **overrides):
        base = dict(
            dataset="uci-dota2", split_fingerprint="abc123",
            vocabulary=self._Vocab(), schema_hash="hash1",
        )
        base.update(overrides)
        return self.contract(**base)

    def test_shared_contract_accepted(self):
        c = self.assert_comparable(
            {"logistic": self._contract(), "mlp": self._contract()},
            ["logistic", "mlp"],
        )
        self.assertEqual(c["models"], ["logistic", "mlp"])
        self.assertEqual(c["vocabulary_size"], 115)

    def test_vocabulary_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "vocabulary_size"):
            self.assert_comparable(
                {"a": self._contract(), "b": self._contract(vocabulary=50)},
                ["a", "b"],
            )

    def test_split_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "split_fingerprint"):
            self.assert_comparable(
                {"a": self._contract(), "b": self._contract(split_fingerprint="other")},
                ["a", "b"],
            )

    def test_missing_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing input contract"):
            self.assert_comparable({"a": self._contract()}, ["a", "b"])

    def test_vocabulary_values_are_hashed_not_embedded(self):
        c = self._contract()
        self.assertEqual(len(c["vocabulary_values_sha256"]), 64)


class SchemaTrainConfigTests(unittest.TestCase):
    """Optimization config is validated but cannot change schema semantics."""

    def setUp(self):
        from gamematch.schema_config import SchemaTrainConfig, load_schema_train_config

        self.cfg_cls = SchemaTrainConfig
        self.load = load_schema_train_config

    def test_load_example_and_roundtrip(self):
        cfg = self.load("examples/schema-train.json")
        self.assertEqual(cfg.dim, 16)
        self.assertEqual(cfg.to_dict(), self.cfg_cls(**cfg.to_dict()).to_dict())

    def test_overrides_applied_after_load(self):
        cfg = self.load("examples/schema-train.json", overrides={"dim": 8, "seed": 3})
        self.assertEqual((cfg.dim, cfg.seed), (8, 3))

    def test_invalid_values_rejected_before_side_effects(self):
        for bad in ({"epochs": 0}, {"dim": 1}, {"learning_rate": -0.1},
                    {"device": "tpu"}, {"seed": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.cfg_cls(**bad).validate()

    def test_unknown_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.load({"structure": "ordered"})       # schema semantics forbidden

    def test_accepts_bare_config_dict(self):
        cfg = self.load({"epochs": 3, "dim": 4})
        self.assertEqual((cfg.epochs, cfg.dim), (3, 4))


class SideLayoutTests(unittest.TestCase):
    """Unequal and multi-team layouts must be declared, not silently assumed."""

    def test_default_layouts_remain_exchangeable(self):
        for schema in SCHEMAS.values():
            if schema.sides == 2:
                self.assertTrue(schema.exchangeable_sides, schema.name)
            self.assertEqual(len(schema.side_sizes), schema.sides)

    def test_unequal_teams_declared(self):
        from dataclasses import replace
        schema = replace(
            SCHEMAS["hero_binary"], players=3, sides=2,
            players_per_side=(3, 2),
        )
        self.assertEqual(schema.side_sizes, (3, 2))
        self.assertFalse(schema.exchangeable_sides)
        self.assertEqual(
            FeatureSchema.from_dict(schema.to_dict()).players_per_side, (3, 2)
        )

    def test_invalid_side_layouts_rejected(self):
        from dataclasses import replace
        base = SCHEMAS["hero_binary"]
        for kwargs in (
            {"players_per_side": (3,)},             # wrong count
            {"players_per_side": (0, 5)},          # nonpositive
            {"players_per_side": (4, 4)},          # max != players
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(base, **kwargs)

    def test_multi_team_layout_is_not_exchangeable(self):
        from dataclasses import replace
        schema = replace(SCHEMAS["hero_binary"], sides=3, players=5,
                         players_per_side=(5, 5, 5))
        self.assertFalse(schema.exchangeable_sides)
        self.assertEqual(len(schema.side_sizes), 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires the neural extra")
    def test_unequal_sides_disable_signed_pooling_and_swap(self):
        from dataclasses import replace

        from gamematch.features import make_model_from_schema, swap_sides

        schema = replace(SCHEMAS["hero_binary"], players=3, players_per_side=(3, 2))
        model = make_model_from_schema(schema, dim=6)
        self.assertFalse(model.encoder.uses_signed_sides)
        self.assertTrue(model.head.bias is not None)
        batch = sample_inputs(schema, batch=2, seed=0)
        with self.assertRaisesRegex(ValueError, "exchangeable"):
            swap_sides(schema, batch)
