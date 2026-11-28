# SPDX-License-Identifier: Apache-2.0
"""Decision-level regression tests for declarative match scoring."""

import unittest
from dataclasses import replace

import numpy as np

from gamematch.features import (
    SCHEMAS,
    FeatureBatch,
    FeatureEncoder,
    FeatureSchema,
    FeatureSpec,
    Modality,
    OutputSpec,
    Scope,
    Structure,
    sample_inputs,
)
from gamematch.matching.search import minimize_assignments

try:
    import torch

    from gamematch.features import make_model_from_schema
except ImportError:
    torch = None


def team_field(name, **kwargs):
    return FeatureSpec(name, Modality.DENSE, Scope.TEAM, 1, **kwargs)


def timed_schema():
    return FeatureSchema("rank", 2, 1, (team_field("base"), team_field("form", available_at="ts")),
                         OutputSpec("left_win", "binary"))


def candidates(schema, indices):
    # The third candidate's form becomes available after the lobby is scored.
    base, form, stamp = np.array([[2, -2, 1], [0.2, 1, 1], [5, 0, 11]])[indices].T

    def sides(value):
        return np.stack((value, np.zeros_like(value)), axis=1)

    return FeatureBatch.from_natural(schema, base=sides(base), form=sides(form),
                                     prediction_time=10, available_at={"ts": sides(stamp)})


class BatchContractTests(unittest.TestCase):
    def test_shared_timestamp_validates_all_fields(self):
        fields = tuple(team_field(n, available_at="clock") for n in ("before", "after"))
        values = np.array([[1, 2]])
        for order in (fields, fields[::-1]):
            schema = FeatureSchema("shared", 2, 1, order)
            batch = FeatureBatch.from_natural(schema, before=values, after=values,
                                              prediction_time=3, available_at={"clock": values})
            batch.validate()
            self.assertCountEqual(schema.at(3, batch.available_at).field_names, ("before", "after"))
            self.assertCountEqual(FeatureEncoder(schema).visible_fields(batch), order)

    def test_reference_encoding_is_independent_of_other_candidates(self):
        schema = timed_schema()
        encoder = FeatureEncoder(schema, dim=3)
        together = encoder(candidates(schema, [2, 0, 1])).team_context
        for row, i in enumerate((2, 0, 1)):
            np.testing.assert_allclose(together[row], encoder(candidates(schema, [i])).team_context[0])
        with self.assertRaisesRegex(ValueError, "not strictly earlier"):
            candidates(schema, [0, 1, 2]).validate()

    def test_malformed_time_evidence_is_rejected(self):
        for ts, pt in ((np.ones((2, 1)), 10), (np.full((2, 2), np.nan), 10),
                       (np.ones((2, 2)), [10, 10, 10])):
            schema = timed_schema()
            batch = candidates(schema, [0, 1])
            batch.prediction_time, batch.available_at = pt, {"ts": ts}
            with self.assertRaises(ValueError):
                batch.validate()
            with self.assertRaises(ValueError):
                FeatureEncoder(schema)(batch)

    def test_absent_clock_never_exposes_timed_values(self):
        schema = timed_schema()
        batch = candidates(schema, [0, 1])
        batch.prediction_time = None
        encoder = FeatureEncoder(schema)
        self.assertEqual([f.name for f in encoder.visible_fields(batch)], ["base"])
        with self.assertRaisesRegex(ValueError, "prediction_time"):
            batch.validate()
        self.assertFalse(batch.validate(strict_time=False)["temporal_evidence_checked"])

    def test_padded_asymmetric_slots_cannot_contribute_ids_or_timestamps(self):
        base = SCHEMAS["hero_binary"]
        schema = replace(base, players=2, players_per_side=(1, 2),
                         features=(replace(base.features[0], available_at="ts"),))
        batch = FeatureBatch.from_natural(schema, hero_id=[[[7, 9999], [8, 9]]],
                                          prediction_time=10, available_at={"ts": [[[1, np.nan], [1, 1]]]})
        batch.validate()
        reference = FeatureEncoder(schema)
        a = reference(batch).team_context
        if torch is not None:
            model = make_model_from_schema(schema)
            expected = model(batch)
        batch.fields["hero_id"][0, 0, 1] = 0
        np.testing.assert_allclose(reference(batch).team_context, a)
        if torch is not None:
            torch.testing.assert_close(model(batch), expected)

    def test_contract_rejects_different_task_on_public_entry_points(self):
        schema = SCHEMAS["overwatch_history"]
        foreign = replace(schema, task=replace(schema.task, labels=("win", "draw", "loss")))
        batch = sample_inputs(foreign, seed=5)
        calls = [schema.validate, FeatureEncoder(schema)]
        if torch is not None:
            model = make_model_from_schema(schema)
            calls += [model, model.predict_proba, model.targets, model.augmented_batch]
        for call in calls:
            with self.subTest(call=type(call).__name__), self.assertRaisesRegex(ValueError, "contract"):
                call(batch)
        schema.validate(sample_inputs(FeatureSchema.from_dict(schema.to_dict())))

    def test_wrong_field_name_cannot_remove_a_time_constraint(self):
        declaration = timed_schema().to_dict()
        field = declaration["features"][1]
        field["availble_at"] = field.pop("available_at")
        with self.assertRaisesRegex(ValueError, "availble_at"):
            FeatureSchema.from_dict(declaration)

    def test_different_fitted_vocabularies_are_incompatible_even_at_equal_size(self):
        base = SCHEMAS["hero_binary"]
        def fitted(digest):
            return replace(base, features=(replace(base.features[0], vocabulary_values_sha256=digest),))

        schema, wrong = fitted("a" * 64), fitted("b" * 64)
        restored = FeatureSchema.from_dict(schema.to_dict())
        schema.validate(sample_inputs(restored))
        for other in (wrong, base):
            with self.assertRaisesRegex(ValueError, "contract"):
                schema.validate(sample_inputs(other))

    def test_reference_role_gate_distinguishes_roles_and_shares_repeated_slots(self):
        base = SCHEMAS["hero_binary"]
        slots = ("captain", "flex", "flex", "flex", "flex")
        schema = replace(base, role_slots=(slots, slots))
        self.assertEqual(FeatureSchema.from_dict(schema.to_dict()), schema)
        rosters = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                            [[2, 1, 3, 4, 5], [6, 7, 8, 9, 10]],
                            [[1, 3, 2, 4, 5], [6, 7, 8, 9, 10]]])
        batch = FeatureBatch.from_natural(schema, hero_id=rosters)
        teams = FeatureEncoder(schema, dim=6)(batch).team_context
        self.assertFalse(np.allclose(teams[0], teams[1]))
        np.testing.assert_allclose(teams[0], teams[2])
        if torch is not None:
            model = make_model_from_schema(schema, dim=6)
            scores = model(batch)
            self.assertFalse(torch.allclose(scores[0], scores[1]))
            torch.testing.assert_close(scores[0], scores[2])


@unittest.skipIf(torch is None, "requires the neural extra")
class MatchScoringTests(unittest.TestCase):
    def test_partial_history_and_empty_history_keep_the_player_in_the_team(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=2, seed=17, prediction_time=10)
        batch.masks["history"][:] = True
        batch.masks["history"][0, 0, 0] = False
        batch.available_at["history_available_at"][1, 1, 1, 0] = 10
        model = make_model_from_schema(schema)
        before = model(batch).detach()
        batch.fields["history"][1, 1, 1, 0] += 100
        torch.testing.assert_close(before, model(batch))
        self.assertTrue(model.encoder(batch).valid[0, 0, 0])
        batch.fields["static"][0, 0, 0] += 100
        self.assertFalse(torch.allclose(before[0], model(batch)[0]))
        batch.fields["history"][1, 1, 1, 1] += 100
        self.assertFalse(torch.allclose(before[1], model(batch)[1]))

    def test_search_result_and_probabilities_do_not_depend_on_batching(self):
        schema = timed_schema()
        model = make_model_from_schema(schema, dim=2).eval()
        with torch.no_grad():
            for layer in model.encoder.projections.values():
                layer.weight.copy_(torch.tensor([[1.], [0.]]))
            model.head.weight.copy_(torch.tensor([[1., 0.]]))

        def predict(indices):
            return model.predict_proba(candidates(schema, indices))[:, 1].detach().numpy()

        np.testing.assert_allclose(predict([2, 0, 1]), [predict([i])[0] for i in (2, 0, 1)])
        for size in (1, 2, 3):
            result = minimize_assignments([0, 1, 2], predict, lambda p: abs(p - .5), size)
            self.assertEqual(result["assignment"], 0)

    def test_equal_teams_can_learn_mode_dependent_draw_rates(self):
        torch.manual_seed(7)
        schema = SCHEMAS["overwatch_rating"]
        states = np.tile([25., 25 / 3], (2, 2, 6, 1))
        batch = FeatureBatch.from_natural(schema, rating_state=states,
                                          calibration_terms=[[0, 0, 0, 0], [0, 0, 1, 0]])
        model = make_model_from_schema(schema, dim=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        # Same team strength, two different empirical draw rates.
        labels = torch.tensor([[.45, .1, .45], [.1, .8, .1]])
        original = model.predict_proba(batch).detach()
        for _ in range(80):
            optimizer.zero_grad()
            loss = -(labels * model(batch).log_softmax(-1)).sum(-1).mean()
            loss.backward()
            optimizer.step()
        probs = model.predict_proba(batch).detach()
        torch.testing.assert_close(probs[:, 0], probs[:, 2])
        self.assertLess(((probs - labels) ** 2).mean(), ((original - labels) ** 2).mean())
        self.assertGreater(probs[1, 1], probs[0, 1])

    def test_wdl_side_swap_reverses_probabilities_and_one_hot_targets(self):
        schema = SCHEMAS["overwatch_history"]
        batch = sample_inputs(schema, batch=3, seed=17)
        batch.targets = np.eye(3)
        model = make_model_from_schema(schema)
        swapped = model.augmented_batch(batch)
        torch.testing.assert_close(model.predict_proba(swapped), model.predict_proba(batch).flip([1]))
        np.testing.assert_array_equal(swapped.targets, batch.targets[:, ::-1])

    def test_rating_mode_is_preserved_while_signed_calibration_channels_reverse(self):
        schema = SCHEMAS["overwatch_rating"]
        batch = sample_inputs(schema, batch=2, seed=6)
        model = make_model_from_schema(schema)
        swapped = model.augmented_batch(batch)
        expected = batch.fields["calibration_terms"] * [-1, 1, 1, -1]
        np.testing.assert_allclose(swapped.fields["calibration_terms"], expected)
        torch.testing.assert_close(model(swapped), model(batch).flip([-1]))

    def test_restored_weights_require_the_same_contract(self):
        schema = SCHEMAS["overwatch_history"]
        model = make_model_from_schema(schema)
        restored = make_model_from_schema(FeatureSchema.from_dict(schema.to_dict()))
        restored.load_state_dict(model.state_dict())
        batch = sample_inputs(schema, seed=9)
        torch.testing.assert_close(restored(batch), model(batch))
        foreign = replace(schema, task=replace(schema.task, labels=("win", "draw", "loss")))
        with self.assertRaisesRegex(ValueError, "contract"):
            make_model_from_schema(foreign).load_state_dict(model.state_dict())

    def test_asymmetric_team_order_reaches_prediction(self):
        for sizes, names in (((1, 4), None), ((2, 2), ("attack", "defend"))):
            schema = FeatureSchema("sided", 2, max(sizes), (team_field("strength"),),
                                   players_per_side=sizes, side_roles=names)
            self.assertEqual(schema, FeatureSchema.from_dict(schema.to_dict()))
            batch = FeatureBatch.from_natural(schema, strength=[[10., 1.], [1., 10.]])
            torch.manual_seed(3)
            model = make_model_from_schema(schema, dim=4)
            out = model(batch).detach()
            self.assertFalse(torch.allclose(out[0], out[1]))
            with self.assertRaisesRegex(ValueError, "exchangeable"):
                model.augmented_batch(batch)

    def test_role_swap_changes_score_only_for_ordered_player_slots(self):
        base = SCHEMAS["hero_binary"]
        heroes = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                           [[2, 1, 3, 4, 5], [6, 7, 8, 9, 10]]])
        for structure in (Structure.ORDERED, Structure.SET):
            schema = replace(base, features=(replace(base.features[0], structure=structure),))
            batch = FeatureBatch.from_natural(schema, hero_id=heroes)
            torch.manual_seed(3)
            model = make_model_from_schema(schema, dim=8)
            scores = model(batch)
            self.assertEqual(torch.allclose(scores[0], scores[1]), structure is Structure.SET)
            torch.testing.assert_close(model(model.augmented_batch(batch)), -scores)
