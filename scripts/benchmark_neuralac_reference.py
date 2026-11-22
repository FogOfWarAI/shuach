# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Matched local microbenchmark against separately cached author equations.

Run after other training jobs finish for uncontended measurements. This is not
an end-to-end framework ranking or a comparison of original published scores.
"""

import argparse
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from verify_neuralac_reference import MAPPING

from gamematch.data.heroes import file_hash
from gamematch.models.neuralac import NeuralAC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--replicates", type=int, default=30)
    p.add_argument("--concurrent-training", action="store_true")
    p.add_argument("--previous-implementation", type=Path)
    p.add_argument("--output", type=Path, default=Path("reports/neuralac-performance.json"))
    args = p.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    torch.set_num_threads(args.threads)
    reference = Path("datasets/extracted/NAC-main/NAC.py")
    spec = importlib.util.spec_from_file_location("reference_neuralac", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    own = NeuralAC(117).eval()
    author = module.NAC(117, hidden_dim=20, device=torch.device(args.device)).eval()
    target = author.state_dict()
    for left, right in MAPPING.items():
        target[right] = own.state_dict()[left].clone()
    for name in ["Coop.attenM.W.bias", "Comp.attenM.W.bias"]:
        target[name].zero_()
    author.load_state_dict(target)
    author.Coop.attenM.W.bias.requires_grad_(False)
    author.Comp.attenM.W.bias.requires_grad_(False)
    own.to(args.device)
    author.to(args.device)
    previous = None
    if args.previous_implementation is not None:
        previous_spec = importlib.util.spec_from_file_location(
            "previous_neuralac", args.previous_implementation
        )
        previous_module = importlib.util.module_from_spec(previous_spec)
        previous_spec.loader.exec_module(previous_module)
        previous = previous_module.NeuralAC(117).eval().to(args.device)
        previous.load_state_dict(own.state_dict())
    rng = np.random.default_rng(0)
    order_rng = np.random.default_rng(20261114)
    rows = []

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    for batch in [256, 1024, 2048]:
        x = np.stack([rng.choice(116, 10, replace=False) + 1 for _ in range(batch)])
        with torch.no_grad():
            a = own(torch.from_numpy(x.reshape(-1, 2, 5)).to(args.device)).sigmoid()
            b = author(x)
        difference = float((a - b).abs().max().detach().cpu())
        if difference > 5e-5:
            raise AssertionError("Implementations do not agree sufficiently for timing")
        previous_difference = None
        if previous is not None:
            with torch.no_grad():
                old = previous(torch.from_numpy(x.reshape(-1, 2, 5)).to(args.device)).sigmoid()
            previous_difference = float((a - old).abs().max().detach().cpu())
            if previous_difference > 5e-5:
                raise AssertionError("Layout optimization changed the evaluated function")
        for backward in [False, True]:
            implementations = {
                "author_zero_attention_bias": author,
                "gamematch_vectorized": own,
            }
            if previous is not None:
                implementations["gamematch_previous_layout"] = previous
            samples = {name: [] for name in implementations}

            def operation(name):
                model = implementations[name]
                model.zero_grad(set_to_none=True)
                if name.startswith("author"):
                    value = model(x)
                else:
                    value = model(torch.from_numpy(x.reshape(-1, 2, 5)).to(args.device)).sigmoid()
                if backward:
                    value.mean().backward()

            context = torch.enable_grad() if backward else torch.no_grad()
            with context:
                for name in implementations:
                    for _ in range(5):
                        operation(name)
                    sync()
                orders = []
                for _ in range(args.replicates):
                    order = order_rng.permutation(list(implementations)).tolist()
                    orders.append(order)
                    for name in order:
                        sync()
                        start = time.perf_counter()
                        operation(name)
                        sync()
                        samples[name].append(time.perf_counter() - start)
            for name in implementations:
                rows.append(
                    {
                        "implementation": name,
                        "batch_size": batch,
                        "mode": "forward_backward" if backward else "inference",
                        "median_seconds": float(np.median(samples[name])),
                        "p10_seconds": float(np.quantile(samples[name], 0.1)),
                        "p90_seconds": float(np.quantile(samples[name], 0.9)),
                        "seconds": samples[name],
                        "paired_invocation_orders": orders,
                        "max_probability_difference": difference,
                        "max_previous_probability_difference": previous_difference,
                    }
                )
                print(name, batch, backward, rows[-1]["median_seconds"], flush=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": file_hash(Path(__file__)),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "device": args.device,
        "threads": args.threads,
        "dtype": "float32",
        "warmup": 5,
        "concurrent_training": args.concurrent_training,
        "reference_sha256": file_hash(reference),
        "implementation_sha256": file_hash(Path("src/gamematch/models/neuralac.py")),
        "previous_implementation": str(args.previous_implementation)
        if previous is not None
        else None,
        "previous_implementation_sha256": file_hash(args.previous_implementation)
        if previous is not None
        else None,
        "scope": "same weights, valid rosters, probability output and disabled dropout; numpy-to-device conversion included, no optimizer step; author attention biases zero and frozen to match paper equations",
        "order_policy": "Paired replicates with deterministically randomized implementation order; seed 20261114",
        "limitations": "single local host with background non-training activity not controlled; not original-score reproduction, training throughput under dropout, or cross-framework superiority",
        "measurements": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
