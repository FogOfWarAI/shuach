# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""All Logistic entry points must reject capped fits before publishing a checkpoint."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from gamematch.asymmetric_runner import run_asymmetric
from gamematch.data.asymmetric import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from gamematch.models.tabular import require_logistic_convergence, train_tabular
from gamematch.tuning.engine import fit


def capped_model(limit):
    return Mock(max_iter=limit, n_iter_=np.array([limit]))


class LogisticConvergenceTests(unittest.TestCase):
    def test_guard_uses_each_solver_limit_and_all_iteration_counts(self):
        for limit in [1000, 2000]:
            with self.subTest(limit=limit):
                model = SimpleNamespace(max_iter=limit, n_iter_=np.array([12, 18]))
                self.assertEqual(require_logistic_convergence(model), 18)
                for iterations in [limit, limit + 1]:
                    model.n_iter_ = np.array([12, iterations])
                    with self.assertRaisesRegex(RuntimeError, "iteration cap"):
                        require_logistic_convergence(model)

    def test_hero_and_synthetic_adapter_rejects_before_checkpoint(self):
        rows = SimpleNamespace(
            rosters=np.tile(np.arange(1, 11).reshape(1, 2, 5), (2, 1, 1)),
            labels=np.array([0, 1]),
        )
        vocabulary = SimpleNamespace(size=11, transform=lambda values: values)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("gamematch.models.tabular.LogisticRegression", return_value=capped_model(1000)),
            patch("gamematch.models.tabular.joblib.dump") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "iteration cap"):
                train_tabular(
                    "logistic",
                    {part: rows for part in ["train", "valid", "test"]},
                    vocabulary,
                    SimpleNamespace(logistic_c=1.0),
                    0,
                    Path(temporary),
                )
            save.assert_not_called()

    def test_hpo_rejects_before_checkpoint_or_metrics(self):
        part = SimpleNamespace(
            tabular=np.zeros((2, 2)),
            target=np.array([0, 1]),
            row_id=np.array(["fixture-0", "fixture-1"]),
            inputs={},
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("gamematch.tuning.engine.LogisticRegression", return_value=capped_model(2000)),
            patch("gamematch.tuning.engine.set_seed"),
            patch("gamematch.tuning.engine.joblib.dump") as save,
        ):
            with self.assertRaisesRegex(RuntimeError, "iteration cap"):
                fit(
                    "binary",
                    "logistic",
                    {},
                    part,
                    part,
                    {"tabular": True, "C": 1.0, "threads": 1},
                    0,
                    temporary,
                )
            save.assert_not_called()
            self.assertFalse((Path(temporary) / "metrics.json").exists())

    def test_dbd_rejects_capped_fit_and_marks_run_failed(self):
        count = 30
        features = pd.DataFrame({name: np.zeros(count) for name in NUMERIC_FEATURES})
        for name in CATEGORICAL_FEATURES:
            features[name] = "fixture"
        rows = np.arange(count)
        data = SimpleNamespace(
            features=features,
            outcome=rows % 5,
            row_ids=rows,
            player_ids=rows,
            time=rows,
            available_at=rows,
            manifest={"fixture": True},
        )
        masks = {name: rows // 10 == index for index, name in enumerate(["train", "valid", "test"])}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("gamematch.asymmetric_runner.load_asymmetric", return_value=data),
            patch(
                "gamematch.asymmetric_runner.history_split_masks", return_value=(masks, [10, 20])
            ),
            patch(
                "gamematch.asymmetric_runner.LogisticRegression", return_value=capped_model(1000)
            ),
            patch("gamematch.asymmetric_runner.set_seed"),
            patch("gamematch.asymmetric_runner.joblib.dump") as save,
        ):
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "protocol": "temporal_predator_outcome",
                        "datasets": ["dbd-2022"],
                        "models": ["logistic"],
                        "seeds": [0],
                        "training": {"threads": 1},
                    }
                )
            )
            output = root / "run"
            with self.assertRaisesRegex(RuntimeError, "iteration cap"):
                run_asymmetric(config, root, output)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "failed")
            self.assertFalse(list(output.glob("*/seed-*/metrics.json")))
            self.assertTrue(
                all(Path(call.args[1]).name != "checkpoint.joblib" for call in save.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
