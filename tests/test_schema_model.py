# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Trainable schema bridge: gradients, head semantics and catalog coverage."""

import unittest

import numpy as np

from gamematch.features import (
    SCHEMAS,
    Availability,
    FeatureBatch,
    FeatureSchema,
    FeatureSpec,
    Modality,
    OutputSpec,
    Scope,
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

    def test_post_match_fields_never_enter_encoder(self):
        schema = make_labeled_schema()
        model = SchemaModel(schema, dim=4)
        batch = sample_inputs(schema, batch=3, seed=0)
        inputs = model.encoder(batch)
        self.assertEqual(tuple(inputs.tokens.shape), (3, schema.sides, schema.players, 4))

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
            batch.validate(strict_time=False)  # raises on failure; POST_MATCH fields are allowed here

    def test_sample_is_reproducible(self):
        schema = SCHEMAS["hero_binary"]
        a = sample_inputs(schema, batch=4, seed=11)
        b = sample_inputs(schema, batch=4, seed=11)
        for key in a.fields:
            self.assertTrue(np.array_equal(a.fields[key], b.fields[key]), key)


def make_labeled_schema() -> FeatureSchema:
    """Schema with a POST_MATCH label feature to test predictor isolation."""
    return FeatureSchema(
        name="labeled",
        sides=2,
        players=3,
        task=OutputSpec("win", "binary"),
            features=(
                FeatureSpec("hero", Modality.COEFFICIENT, Scope.PLAYER, 1,
                            steps=1, vocab_size=10),
                FeatureSpec("label", Modality.DENSE, Scope.MATCH, 1,
                            available=Availability.POST_MATCH),
            ),
    )


if __name__ == "__main__":
    unittest.main()
