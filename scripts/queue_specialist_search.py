# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Finite dependency gate for the already authorized specialist experiment."""

import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
models = ["static_deepsets", "history_deepsets", "owo_transformer", "owo", "dawn"]
last = None
while True:
    states = {}
    for name in models:
        path = root / "runs/hpo-v1/overwatch" / name / "status.json"
        states[name] = json.loads(path.read_text())["status"] if path.exists() else "pending"
    if states != last:
        print(json.dumps({"waiting_for_first_round": states}), flush=True)
        last = states
    if all(state == "completed" for state in states.values()):
        break
    if any(state == "failed" for state in states.values()):
        raise RuntimeError(
            "First-round specialist dependency failed; inspect its status before continuing"
        )
    time.sleep(20)
os.execve(
    sys.executable,
    [
        sys.executable,
        str(root / "scripts/run_local.py"),
        "--python",
        str(root / "scripts/run_specialist_search.py"),
    ],
    os.environ.copy(),
)
