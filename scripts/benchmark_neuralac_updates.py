# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compare old/new local NeuralAC optimizer-update throughput on fixed fake data.

These engineering measurements are not model-evaluation jobs, epoch timings,
score replications or evidence of identical stochastic training trajectories.
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
from torch.nn import functional as F

from gamematch.data.heroes import file_hash
from gamematch.models.neuralac import NeuralAC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()
    if args.replicates < 1 or args.updates < 1:
        raise ValueError("Positive replicate and update counts required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("old_neuralac_layout", args.previous)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = {"previous": module.NeuralAC, "optimized": NeuralAC}
    rng = np.random.default_rng(20261114)
    order_rng = np.random.default_rng(20261115)

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    records = []
    for batch in (256, 1024, 2048):
        # Same ten distinct hero IDs, input bank and binary targets for both layouts.
        bank = np.stack([rng.choice(116, 10, replace=False) + 1 for _ in range(batch * 8)])
        bank = bank.reshape(8, batch, 2, 5)
        labels = rng.integers(0, 2, size=(8, batch)).astype("float32")
        for replicate in range(args.replicates):
            torch.manual_seed(replicate)
            initial = NeuralAC(117, dropout=0.2).state_dict()
            order = order_rng.permutation(list(classes)).tolist()
            for name in order:
                torch.manual_seed(replicate)
                model = classes[name](117, dropout=0.2).to(args.device).train()
                model.load_state_dict(initial)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                def update(index, model, optimizer):
                    x = torch.from_numpy(bank[index % 8]).to(args.device)
                    y = torch.from_numpy(labels[index % 8]).to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(model(x), y)
                    loss.backward()
                    optimizer.step()
                    return loss

                for step in range(5):
                    update(step, model, optimizer)
                sync()
                started = time.perf_counter()
                for step in range(args.updates):
                    final_loss = update(step + 5, model, optimizer)
                sync()
                elapsed = time.perf_counter() - started
                loss_value = float(final_loss.detach().cpu())
                assert np.isfinite(loss_value)
                assert all(torch.isfinite(p).all().item() for p in model.parameters())
                records.append(
                    {
                        "implementation": name,
                        "batch_size": batch,
                        "replicate": replicate,
                        "paired_order": order,
                        "updates": args.updates,
                        "seconds": elapsed,
                        "seconds_per_update": elapsed / args.updates,
                        "examples_per_second": batch * args.updates / elapsed,
                        "last_stochastic_loss": loss_value,
                        "all_parameters_finite": True,
                    }
                )
                print(args.device, batch, replicate, name, round(elapsed, 3), flush=True)
                del optimizer, model
    summaries = []
    for batch in (256, 1024, 2048):
        medians = {}
        for name in classes:
            values = [
                row["seconds_per_update"]
                for row in records
                if row["batch_size"] == batch and row["implementation"] == name
            ]
            medians[name] = float(np.median(values))
        summaries.append(
            {
                "batch_size": batch,
                "median_seconds_per_update": medians,
                "previous_over_optimized_time_ratio": medians["previous"] / medians["optimized"],
            }
        )
    payload = {
        "status": "completed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_threads": 4,
        "dtype": "float32",
        "dropout": 0.2,
        "optimizer": {"name": "Adam", "lr": 0.001},
        "warmup_updates": 5,
        "replicates": args.replicates,
        "timed_updates_per_replicate": args.updates,
        "script_sha256": file_hash(Path(__file__)),
        "previous_sha256": file_hash(args.previous),
        "current_sha256": file_hash(Path("src/gamematch/models/neuralac.py")),
        "scope": "Fixed eight-batch random roster/label bank, shared initialization per paired block, active dropout, BCEWithLogits, zero_grad/backward/Adam updates and NumPy-to-device conversion; randomized old/new block order; device synchronized at block boundaries; no concurrent model training.",
        "limitations": "Single local host; no data-loading, validation, checkpoint, or epoch scheduling time; not production throughput or a published-score experiment. Tensor layouts may consume dropout RNG differently, so stochastic losses and weight trajectories need not match. Timing evidence does not authorize substituting checkpoints or claims of faster convergence.",
        "summaries": summaries,
        "measurements": records,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
