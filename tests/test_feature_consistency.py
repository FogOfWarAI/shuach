# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Regression tests for consistency between contract, validation and encoders.

These lock down findings from an independent review: the two timing paths must
agree, augmentation must preserve declared semantics, and declarations that a
model cannot honour must be rejected at construction time.
"""

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


if __name__ == "__main__":
    unittest.main()
