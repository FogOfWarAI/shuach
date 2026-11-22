# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gamematch.history_runner import cached_history
from gamematch.models import make_model
from gamematch.models.heroes import NEURALAC_VARIANTS


class AblationContractTests(unittest.TestCase):
    def test_neuralac_variants_train_only_active_components(self):
        x = torch.arange(1, 21).reshape(2, 2, 5)
        for name in NEURALAC_VARIANTS:
            model = make_model(name, 30)
            y = model(x)
            self.assertEqual(y.shape, (2,))
            y.sum().backward()
            for parameter in model.parameters():
                if not parameter.requires_grad:
                    self.assertIsNone(parameter.grad)
            self.assertEqual(
                model.cooperation_attention.bias is not None, name == "neuralac_author_bias"
            )

    def test_target_only_history_withholds_cross_mode_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "events.jsonl"
            teams = {"blue": list("abcdef"), "red": list("ghijkl")}
            events = [
                {
                    "time": i * 100,
                    "duration": 10,
                    "competitive": bool(i % 2),
                    "result": "WIN",
                    "teams": teams,
                }
                for i in range(8)
            ]
            source.write_text("\n".join(json.dumps(e) for e in events))
            config = {
                "source": "events.jsonl",
                "history_length": 4,
                "same_domain_only": True,
                "target_domain_features_only": True,
            }
            first, metadata = cached_history(root, config)
            self.assertIn("target-domain", metadata["feature_scope"])
            # Changing outcomes of mode 1 cannot alter mode-0 input histories.
            for event in events:
                if event["competitive"]:
                    event["result"] = "LOSS"
            source.write_text("\n".join(json.dumps(e) for e in events))
            second, _ = cached_history(root, config)
            mask = first.domain == 0
            np.testing.assert_array_equal(first.history[mask], second.history[mask])
            np.testing.assert_array_equal(first.static[mask], second.static[mask])
            self.assertTrue(np.all(first.static[..., :4] == 0))
            self.assertTrue(np.all(first.static[mask, ..., 8:12] == 0))
            with self.assertRaises(ValueError):
                cached_history(root, config | {"same_domain_only": False})


if __name__ == "__main__":
    unittest.main()
