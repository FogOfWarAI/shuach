# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Measure actual CPU/MPS training steps; no dataset or external service is used."""

import json
import platform
import time
from pathlib import Path

import torch

from gamematch.models import make_model


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = {
        "macos": platform.mac_ver()[0],
        "torch": str(torch.__version__),
        "mps_available": torch.backends.mps.is_available(),
        "threads": 4,
        "measurements": [],
    }
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for name in ["linear", "fm", "mlp", "deepsets", "neuralac"]:
        for batch_size in [512, 2048]:
            for device in devices:
                torch.manual_seed(0)
                model = make_model(name, 150).to(device)
                x = torch.randint(1, 150, (batch_size, 2, 5)).to(device)
                y = torch.randint(0, 2, (batch_size,)).float().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                row = {"model": name, "batch_size": batch_size, "device": device}
                try:
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).backward()
                        optimizer.step()
                    synchronize(device)
                    started = time.perf_counter()
                    for _ in range(10):
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
                        loss.backward()
                        optimizer.step()
                    synchronize(device)
                    row.update(
                        {
                            "seconds_per_step": (time.perf_counter() - started) / 10,
                            "finite_loss": bool(torch.isfinite(loss).cpu()),
                        }
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                report["measurements"].append(row)
                print(json.dumps(row), flush=True)
                del model, x, y, optimizer
                if device == "mps":
                    torch.mps.empty_cache()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/device-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
