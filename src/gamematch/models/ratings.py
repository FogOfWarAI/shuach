# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Outcome-only rating adapters. Library defaults are recorded, not tuned on test."""

import math
from collections import defaultdict
from functools import partial

import numpy as np
from scipy.special import expit, logit, ndtr


class TeamElo:
    """Mean-team Elo; each member receives K*(result-expectation).

    This convention changes a team's mean by K*error. It is explicitly distinct
    from sum-team Elo or variants that divide the member update by team size.
    """

    def __init__(self, k=32.0, scale=400.0):
        self.k, self.scale = k, scale
        self.ratings = defaultdict(float)

    def features(self, teams):
        difference = np.mean([self.ratings[p] for p in teams[0]]) - np.mean(
            [self.ratings[p] for p in teams[1]]
        )
        return float(difference * math.log(10) / self.scale)

    def update(self, teams, outcome):
        error = outcome / 2 - expit(self.features(teams))
        for side, team in enumerate(teams):
            for player in team:
                self.ratings[player] += (1 if side == 0 else -1) * self.k * error


class TrueSkillRating:
    """Track each player's mean and uncertainty with the TrueSkill library.

    features returns a left-side win score on the log-odds scale. update accepts
    the two teams and the observed result (0 for a left loss, 1 for a draw,
    2 for a left win). The caller controls when each result becomes available.
    """

    def __init__(self, draw_probability=0.02):
        import trueskill

        self.environment = trueskill.TrueSkill(draw_probability=draw_probability)
        self.ratings = {}

    def _teams(self, teams):
        for team in teams:
            for player in team:
                if player not in self.ratings:
                    self.ratings[player] = self.environment.create_rating()
        return [[self.ratings[p] for p in team] for team in teams]

    def features(self, teams):
        groups = self._teams(teams)
        delta = sum(r.mu for r in groups[0]) - sum(r.mu for r in groups[1])
        variance = sum(r.sigma**2 + self.environment.beta**2 for team in groups for r in team)
        return float(logit(np.clip(ndtr(delta / math.sqrt(variance)), 1e-8, 1 - 1e-8)))

    def update(self, teams, outcome):
        ranks = [1, 0] if outcome == 0 else [0, 0] if outcome == 1 else [0, 1]
        updated = self.environment.rate(self._teams(teams), ranks=ranks)
        for players, ratings in zip(teams, updated):
            self.ratings.update(zip(players, ratings))


class OpenSkillRating:
    """Use an OpenSkill team model with a common delayed-feedback replay interface.

    algorithm selects PlackettLuce (default), BradleyTerryFull or ThurstoneMostellerFull.
    features returns the left side's win score on the log-odds scale. The rating
    runner fits a separate validation calibrator to obtain loss/draw/win probabilities.
    """

    def __init__(self, algorithm="PlackettLuce"):
        from openskill.models import BradleyTerryFull, PlackettLuce, ThurstoneMostellerFull

        algorithms = {
            cls.__name__: cls for cls in (PlackettLuce, BradleyTerryFull, ThurstoneMostellerFull)
        }
        if algorithm not in algorithms:
            raise ValueError(f"Unknown OpenSkill team model: {algorithm}")
        self.environment = algorithms[algorithm]()
        self.ratings = {}

    def _teams(self, teams):
        for team in teams:
            for player in team:
                if player not in self.ratings:
                    self.ratings[player] = self.environment.rating()
        return [[self.ratings[p] for p in team] for team in teams]

    def features(self, teams):
        probability = self.environment.predict_win(self._teams(teams))[0]
        return float(logit(np.clip(probability, 1e-8, 1 - 1e-8)))

    def update(self, teams, outcome):
        """Apply one observed result to the player IDs on the left and right teams.

        The event adapter encodes loss, draw and win as 0, 1 and 2 from the
        left side's perspective. The OpenSkill backend expects one rank per team.
        """
        ranks = [1, 0] if outcome == 0 else [0, 0] if outcome == 1 else [0, 1]
        updated = self.environment.rate(self._teams(teams), ranks=ranks)
        for players, ratings in zip(teams, updated):
            self.ratings.update(zip(players, ratings))


# Existing configs which omit models continue to run the original comparison.
DEFAULT_RATING_MODELS = ("elo", "trueskill", "openskill")
RATING_MODELS = {
    "elo": TeamElo,
    "trueskill": TrueSkillRating,
    "openskill": OpenSkillRating,
    "openskill_bt_full": partial(OpenSkillRating, "BradleyTerryFull"),
    "openskill_tm_full": partial(OpenSkillRating, "ThurstoneMostellerFull"),
}
