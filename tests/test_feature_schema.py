# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Shuach contributors

"""Executable feature contracts: layout, timing, serialization and catalog wiring."""

import json
import unittest

import numpy as np

from gamematch.catalog import PROFILES, feature_schema_for
from gamematch.features import (
    SCHEMAS,
    Availability,
    FeatureBatch,
    FeatureEncoder,
    FeatureSchema,
    FeatureSpec,
    Modality,
    Scope,
    predictor_schema,
)


def sample_batch(schema, batch=3, seed=0):
    """Build natural-shaped inputs for ANY schema from its own declarations."""
    rng = np.random.default_rng(seed)
    raw = {}
    for spec in schema.features:
        shape = schema.natural_shape(spec, batch)
        if spec.modality is Modality.COEFFICIENT:
            raw[spec.name] = rng.integers(1, spec.vocab_size, size=shape)
        else:
            raw[spec.name] = rng.normal(size=shape)
        if spec.mask:
            raw[f"{spec.name}__valid"] = rng.random(shape[:-1]) < 0.85
    return raw


class CatalogWiringTests(unittest.TestCase):
    def test_every_profile_has_an_executable_schema(self):
        self.assertEqual(len(PROFILES), 6)
        for name, profile in PROFILES.items():
            schema = feature_schema_for(name)
            self.assertEqual(schema.name, profile.feature_schema, name)
            self.assertTrue(schema.features, name)
            self.assertIsInstance(schema.cells(), dict)

    def test_unknown_profile_still_raises(self):
        with self.assertRaises(ValueError):
            feature_schema_for("not-a-profile")

    def test_profile_without_schema_raises(self):
        profile = next(iter(PROFILES.values()))
        from dataclasses import replace

        with self.assertRaises(ValueError):
            _ = replace(profile, feature_schema=None).executable_schema


class CanonicalLayoutTests(unittest.TestCase):
    def test_all_schemas_validate_and_encode_to_one_interface(self):
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                batch = FeatureBatch.from_natural(schema, **sample_batch(schema))
                post = any(f.available is Availability.POST_MATCH for f in schema.features)
                report = batch.validate(strict_time=not post)
                self.assertEqual(report["features"], len(schema.features))
                for spec in schema.features:
                    self.assertEqual(
                        tuple(batch.fields[spec.name].shape),
                        schema.canonical_shape(spec, batch.batch_size),
                        spec.name,
                    )
                encoded = FeatureEncoder(schema, dim=8, seed=0)(batch)
                self.assertEqual(
                    tuple(encoded.tokens.shape),
                    (batch.batch_size, schema.sides, schema.players, 8),
                )
                self.assertEqual(tuple(encoded.team_context.shape)[-1], 8)

    def test_scope_collapses_axes_instead_of_changing_rank(self):
        schema = SCHEMAS["overwatch_history"]
        batch = FeatureBatch.from_natural(schema, **sample_batch(schema))
        shapes = {f.name: tuple(batch.fields[f.name].shape) for f in schema.features}
        self.assertEqual(shapes["history"][1:], (2, 6, 10, 5))
        self.assertEqual(shapes["static"][1:], (2, 6, 1, 13))
        self.assertEqual(shapes["target_mode"][1:], (1, 1, 1, 1))

    def test_adding_one_feature_needs_no_other_change(self):
        from dataclasses import replace

        base = SCHEMAS["overwatch_history"]
        extended = replace(
            base,
            name="extended",
            features=base.features + (
                FeatureSpec("party_cohesion", Modality.DENSE, Scope.TEAM, 2, group="party"),
            ),
        )
        raw = sample_batch(base)
        raw["party_cohesion"] = np.zeros((3, 2, 2), dtype=float)
        batch = FeatureBatch.from_natural(extended, **raw)
        batch.validate()
        self.assertEqual(
            tuple(batch.fields["party_cohesion"].shape), (3, 2, 1, 1, 2)
        )


class ContractFailureTests(unittest.TestCase):
    def test_temporal_leakage_is_rejected(self):
        schema = SCHEMAS["overwatch_history"]
        raw = sample_batch(schema)
        times = np.full((3, 2, 6, 10), 50.0)
        times[0, 0, 0, -1] = 100.0
        batch = FeatureBatch.from_natural(
            schema,
            prediction_time=np.array([100.0] * 3),
            available_at={"history": times},
            **raw,
        )
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            batch.validate()

    def test_clean_timing_passes(self):
        schema = SCHEMAS["overwatch_history"]
        raw = sample_batch(schema)
        batch = FeatureBatch.from_natural(
            schema,
            prediction_time=np.array([100.0] * 3),
            available_at={"history": np.full((3, 2, 6, 10), 50.0)},
            **raw,
        )
        self.assertTrue(batch.validate()["temporal_evidence_checked"])

    def test_post_match_feature_blocked_from_predictor(self):
        schema = SCHEMAS["dota_future_rating"]
        batch = FeatureBatch.from_natural(schema, **sample_batch(schema))
        with self.assertRaisesRegex(ValueError, "POST_MATCH"):
            batch.validate()
        clean = predictor_schema(schema)
        self.assertEqual(len(clean.features), len(schema.features) - 1)
        self.assertNotIn("snapshots", [f.name for f in clean.features])

    def test_bad_natural_shape_names_the_expected_shape(self):
        schema = SCHEMAS["hero_binary"]
        raw = {"hero_id": np.ones((3, 2, 4), dtype=int)}
        with self.assertRaisesRegex(ValueError, "expected natural shape"):
            FeatureBatch.from_natural(schema, **raw)


class SerializationTests(unittest.TestCase):
    def test_json_roundtrip_preserves_cells_and_timing(self):
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                restored = FeatureSchema.from_dict(json.loads(schema.to_json()))
                self.assertEqual(restored.cells(), schema.cells())
                self.assertEqual(restored.task.kind, schema.task.kind)
                self.assertEqual(
                    [f.available for f in restored.features],
                    [f.available for f in schema.features],
                )

    def test_invalid_spec_rejected_at_declaration_time(self):
        with self.assertRaises(ValueError):
            FeatureSpec("bad", Modality.SEQUENCE, Scope.PLAYER, 5, steps=1)
        with self.assertRaises(ValueError):
            FeatureSpec("bad", Modality.COEFFICIENT, Scope.PLAYER, 1, vocab_size=0)


if __name__ == "__main__":
    unittest.main()
