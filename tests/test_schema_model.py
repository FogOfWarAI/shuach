# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Trainable schema bridge: gradients, head semantics and catalog coverage."""

import unittest

import numpy as np

from gamematch.features import (
    SCHEMAS,
    FeatureBatch,
    FeatureSchema,
    FeatureSpec,
    Modality,
    Orientation,
    OutputSpec,
    Scope,
    Structure,
    sample_inputs,
)

try:
    import torch

    from gamematch.features import SchemaModel, make_head, make_model_from_schema

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on base install
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "requires the neural extra")
class SchemaModelTests(unittest.TestCase):
    def test_gradients_reach_every_declared_feature(self):
        schema = SCHEMAS["overwatch_history"]
        model = make_model_from_schema(schema, dim=8, seed=0)
        batch = sample_inputs(schema, batch=4, seed=3)
        logits = model(batch)
        logits.sum().backward()

        named = dict(model.encoder.embeddings) | dict(model.encoder.projections)
        self.assertEqual(set(named), {f.name for f in schema.features})
        for spec in schema.features:
            layer = named[spec.name]
            weight = layer.weight if hasattr(layer, "weight") else None
            if weight is not None:
                self.assertIsNotNone(weight.grad, spec.name)
                self.assertTrue(torch.isfinite(weight.grad).all())

    def test_head_matches_output_spec(self):
        cases = [
            (OutputSpec("win", "binary"), 6, (3, 1)),
            (OutputSpec("wdl", "multiclass", labels=("l", "d", "w"), dim=3), 6, (3, 3)),
            (OutputSpec("grade", "ordinal", labels=("a", "b", "c", "d"), dim=4), 6, (3, 4)),
            (OutputSpec("mmr", "regression"), 6, (3, 1)),
        ]
        for spec, dim, expected in cases:
            head = make_head(spec, dim)
            out = head(torch.zeros(3, dim))
            self.assertEqual(tuple(out.shape), expected, spec.kind)

    def test_predict_proba_respects_declared_link(self):
        binary = make_model_from_schema(SCHEMAS["hero_binary"], dim=8)
        probs = binary.predict_proba(sample_inputs(SCHEMAS["hero_binary"], batch=5, seed=1))
        self.assertEqual(tuple(probs.shape), (5, 2))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(5), atol=1e-5))

        multi = make_model_from_schema(SCHEMAS["overwatch_history"], dim=8)
        probs = multi.predict_proba(sample_inputs(SCHEMAS["overwatch_history"], batch=5, seed=1))
        self.assertEqual(tuple(probs.shape), (5, 3))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(5), atol=1e-5))

    def test_all_catalog_schemas_build_and_forward(self):
        for name, schema in SCHEMAS.items():
            if schema.task.kind == "assignment":
                continue
            model = make_model_from_schema(schema, dim=6, seed=0)
            batch = sample_inputs(schema, batch=3, seed=7)
            out = model(batch)
            self.assertTrue(torch.isfinite(out).all(), name)
            self.assertEqual(out.shape[0], 3, name)

    def test_timed_match_level_field_encodes_and_blocks_when_too_late(self):
        schema = make_labeled_schema()
        model = SchemaModel(schema, dim=4)
        batch = sample_inputs(schema, batch=3, seed=0)
        batch.validate()
        inputs = model.encoder(batch)
        self.assertEqual(tuple(inputs.tokens.shape), (3, schema.sides, schema.players, 4))
        self.assertEqual(len(model.encoder.visible_fields(batch)), 2)
        key = schema.features[1].available_at
        batch.available_at[key] = np.full(
            np.asarray(batch.available_at[key]).shape,
            float(np.asarray(batch.prediction_time).reshape(-1)[0]) + 1.0,
        )
        self.assertEqual(len(model.encoder.visible_fields(batch)), 1)

    def test_custom_schema_needs_only_declarations(self):
        schema = FeatureSchema(
            name="new_track",
            sides=2,
            players=4,
            task=OutputSpec("win", "binary"),
            features=(
                FeatureSpec("hero", Modality.COEFFICIENT, Scope.PLAYER, 1,
                            steps=1, vocab_size=12),
                FeatureSpec("team_form", Modality.DENSE, Scope.TEAM, 5),
            ),
        )
        model = make_model_from_schema(schema, dim=8)
        batch = sample_inputs(schema, batch=6, seed=2)
        batch.validate()  # raises on failure; a returned report means passed
        out = model(batch)
        self.assertTrue(torch.isfinite(out).all())


class SampleInputsTests(unittest.TestCase):
    def test_any_schema_yields_a_valid_batch(self):
        for name, schema in SCHEMAS.items():
            batch = sample_inputs(schema, batch=3, seed=0)
            self.assertIsInstance(batch, FeatureBatch)
            self.assertEqual(batch.batch_size, 3, name)
            batch.validate()  # raises on any violation; no PRE/POST binary remains

    def test_sample_is_reproducible(self):
        schema = SCHEMAS["hero_binary"]
        a = sample_inputs(schema, batch=4, seed=11)
        b = sample_inputs(schema, batch=4, seed=11)
        for key in a.fields:
            self.assertTrue(np.array_equal(a.fields[key], b.fields[key]), key)


def make_labeled_schema() -> FeatureSchema:
    """Schema with a timed MATCH-level field (still an input, not a label)."""
    return FeatureSchema(
        name="labeled",
        sides=2,
        players=3,
        task=OutputSpec("win", "binary"),
        features=(
            FeatureSpec("hero", Modality.COEFFICIENT, Scope.PLAYER, 1,
                        steps=1, vocab_size=10),
            FeatureSpec("context_timeline", Modality.SEQUENCE, Scope.MATCH, 2,
                        steps=4, mask=True, available_at="context_available_at",
                        structure=Structure.ORDERED),
        ),
    )


@unittest.skipUnless(TORCH_AVAILABLE, "requires the neural extra")
class DerivedBehaviourTests(unittest.TestCase):
    """Convenience comes from derivation: one declaration drives many behaviours."""

    def test_every_modality_scope_cell_builds_and_forwards(self):
        cells = [
            (Modality.DENSE, Scope.MATCH), (Modality.DENSE, Scope.TEAM),
            (Modality.DENSE, Scope.PLAYER), (Modality.COEFFICIENT, Scope.MATCH),
            (Modality.COEFFICIENT, Scope.TEAM), (Modality.COEFFICIENT, Scope.PLAYER),
            (Modality.SEQUENCE, Scope.MATCH), (Modality.SEQUENCE, Scope.TEAM),
            (Modality.SEQUENCE, Scope.PLAYER),
        ]
        for modality, scope in cells:
            with self.subTest(cell=f"{modality.value}/{scope.value}"):
                kwargs = {}
                if modality is Modality.COEFFICIENT:
                    kwargs["vocab_size"] = 8
                if modality is Modality.SEQUENCE:
                    kwargs.update(steps=4, mask=True)
                spec = FeatureSpec("f", modality, scope, 3, **kwargs)
                schema = FeatureSchema("cell", 2, 4, (spec,), OutputSpec("win", "binary"))
                batch = sample_inputs(schema, batch=3, seed=0)
                batch.validate()
                model = make_model_from_schema(schema, dim=6)
                out = model(batch)
                self.assertTrue(torch.isfinite(out).all())

    def test_structure_changes_pooling_and_ordered_adds_slot_parameters(self):
        base = SCHEMAS["dota_future_rating"]
        ordered = base.features[0].resolved_structure
        self.assertEqual(ordered, Structure.ORDERED)
        model = make_model_from_schema(base, dim=6)
        self.assertIn("snapshots", dict(model.encoder.slot_embeddings))
        batch = sample_inputs(base, batch=3, seed=0)
        logits = model(batch)
        self.assertTrue(torch.isfinite(logits).all())

    def test_antisymmetric_orientation_flips_logits_on_side_swap(self):
        schema = SCHEMAS["hero_binary"]
        self.assertEqual(schema.task.resolved_orientation, Orientation.ANTISYMMETRIC)
        model = make_model_from_schema(schema, dim=6).eval()
        batch = sample_inputs(schema, batch=4, seed=0)
        swapped = model.augmented_batch(batch)
        with torch.no_grad():
            a = model(batch)
            b = model(swapped)
        self.assertTrue(torch.allclose(a, -b, atol=1e-5))
        # labels must flip with the sides
        self.assertTrue(np.allclose(
            np.asarray(batch.targets) + np.asarray(swapped.targets), 1.0))

    def test_criterion_is_derived_from_output_spec(self):
        cases = [
            (SCHEMAS["hero_binary"], torch.nn.BCEWithLogitsLoss),
            (SCHEMAS["overwatch_history"], torch.nn.CrossEntropyLoss),
            (SCHEMAS["dota_future_rating"], torch.nn.MSELoss),
        ]
        for schema, expected in cases:
            model = make_model_from_schema(schema, dim=6)
            self.assertIsInstance(model.criterion(), expected, schema.name)
            batch = sample_inputs(schema, batch=4, seed=0)
            target = model.targets(batch)
            loss = model.criterion()(model(batch), target)
            self.assertTrue(torch.isfinite(loss), schema.name)

    def test_at_time_controls_which_fields_reach_the_encoder(self):
        schema = SCHEMAS["overwatch_history"]
        model = make_model_from_schema(schema, dim=6)
        clean = sample_inputs(schema, batch=3, seed=0)
        self.assertEqual(len(model.encoder.visible_fields(clean)), len(schema.features))
        leaked = clean
        leaked.available_at["history_available_at"] = np.full(
            np.asarray(leaked.available_at["history_available_at"]).shape,
            float(np.asarray(leaked.prediction_time).reshape(-1)[0]) + 1.0,
        )
        self.assertNotIn("history", [f.name for f in model.encoder.visible_fields(leaked)])


if __name__ == "__main__":
    unittest.main()
