# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import json
import tempfile
import unittest
from pathlib import Path

from gamematch.asymmetric_runner import run_asymmetric
from gamematch.decision_runner import run_decisions
from gamematch.history_runner import run_history
from gamematch.runner import run_benchmark
from gamematch.skill_runner import run_skill


class ConfigurationTests(unittest.TestCase):
    def test_invalid_matrix_fails_before_data_loading_or_output_creation(self):
        workflows = [
            (run_benchmark, "pilot", "logistic"),
            (run_history, "temporal_completed_player_history", "history_deepsets"),
            (run_asymmetric, "temporal_predator_outcome", "prior"),
            (run_skill, "temporal_future_observed_rating", "current"),
            (run_decisions, "synthetic_independent_decisions", "logistic"),
        ]
        invalid = [
            {"seeds": []},
            {"seeds": [0, 0]},
            {"seeds": [True]},
            {"seeds": [-1]},
            {"seeds": [2**32]},
            {"models": []},
            {"models": "logistic"},
            {"max_train_rows": 0},
            {"max_eval_rows": -1},
            {"max_eval_rows": 1.5},
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, output = root / "config.json", root / "output"
            for run, protocol, model in workflows:
                for changed in invalid:
                    values = {
                        "protocol": protocol,
                        "models": [model],
                        "seeds": [0],
                        "datasets": ["uci-dota2"],
                        "mechanisms": ["additive"],
                        **changed,
                    }
                    config.write_text(json.dumps(values))
                    with self.subTest(run=run.__name__, changed=changed):
                        with self.assertRaises(ValueError):
                            run(config, root, output)
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
