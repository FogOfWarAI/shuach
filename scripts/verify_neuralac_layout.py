# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compare the current NeuralAC function/gradients with an archived own layout."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gamematch.data.heroes import file_hash
from gamematch.models.neuralac import NeuralAC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("previous_neuralac_layout", args.previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dtype = torch.float64 if args.device == "cpu" else torch.float32
    cases = []
    for seed in range(5):
        for attention in [False, True]:
            torch.manual_seed(seed)
            previous = module.NeuralAC(117, attention=attention).to(args.device, dtype).eval()
            current = NeuralAC(117, attention=attention).to(args.device, dtype).eval()
            current.load_state_dict(previous.state_dict())
            rng = np.random.default_rng(seed)
            values = np.stack([rng.choice(116, 10, replace=False) + 1 for _ in range(64)])
            rosters = torch.as_tensor(values.reshape(-1, 2, 5), device=args.device)
            a, b = previous(rosters).sigmoid(), current(rosters).sigmoid()
            a.sum().backward()
            b.sum().backward()
            error = float((a - b).abs().max().detach().cpu())
            gradient = max(
                float((p.grad - q.grad).abs().max().cpu())
                for p, q in zip(previous.parameters(), current.parameters())
                if p.grad is not None
            )
            if error > (1e-12 if args.device == "cpu" else 5e-5):
                raise AssertionError("Layout changed probabilities beyond declared tolerance")
            if gradient > (1e-10 if args.device == "cpu" else 5e-4):
                raise AssertionError("Layout changed gradients beyond declared tolerance")
            cases.append(
                {
                    "seed": seed,
                    "attention": attention,
                    "max_probability_error": error,
                    "max_gradient_error": gradient,
                }
            )
    payload = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "previous_sha256": file_hash(args.previous),
        "current_sha256": file_hash(Path("src/gamematch/models/neuralac.py")),
        "script_sha256": file_hash(Path(__file__)),
        "scope": "Five initializations, attention on/off, 64 valid rosters each, same parameters and evaluation-mode probabilities plus all active parameter gradients. Dropout disabled; this does not claim identical stochastic training trajectories across layouts/devices. Primary experimental checkpoints remain frozen.",
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        args.device,
        len(cases),
        "function/gradient cases passed",
        max(c["max_gradient_error"] for c in cases),
    )


if __name__ == "__main__":
    main()
