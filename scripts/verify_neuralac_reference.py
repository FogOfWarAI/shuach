# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 GameMatch contributors

"""Compare independently written equations with a separately downloaded reference.

The external author's implementation is not copied into the software distribution.
Only its two attention biases are zeroed to match the bias-free paper equations.
Dropout is disabled for deterministic function/gradient comparison.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from gamematch.data.heroes import file_hash
from gamematch.models.neuralac import NeuralAC

MAPPING = {
    "individual.weight": "BT.skill.weight",
    "cooperation.weight": "Coop.embedding.weight",
    "attack.weight": "Comp.blade.weight",
    "defense.weight": "Comp.chest.weight",
    "cooperation_attention.weight": "Coop.attenM.W.weight",
    "competition_attention.weight": "Comp.attenM.W.weight",
    "cooperation_network.0.weight": "Coop.MLP.0.weight",
    "cooperation_network.0.bias": "Coop.MLP.0.bias",
    "cooperation_network.3.weight": "Coop.MLP.3.weight",
    "cooperation_network.3.bias": "Coop.MLP.3.bias",
    "competition_network.0.weight": "Comp.MLP.0.weight",
    "competition_network.0.bias": "Comp.MLP.0.bias",
    "competition_network.3.weight": "Comp.MLP.3.weight",
    "competition_network.3.bias": "Comp.MLP.3.bias",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, default=Path("datasets/extracted/NAC-main/NAC.py")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/neuralac-reference-check.json")
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("external_neuralac", args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    results = []
    for attention in [False, True]:
        for seed in range(5):
            torch.manual_seed(seed)
            own = NeuralAC(117, attention=attention).double().eval()
            author = module.NAC(117, hidden_dim=20, need_att=attention).double().eval()
            source = own.state_dict()
            target = author.state_dict()
            for left, right in MAPPING.items():
                target[right] = source[left].clone()
            for name in ["Coop.attenM.W.bias", "Comp.attenM.W.bias"]:
                target[name].zero_()
            author.load_state_dict(target)
            rng = np.random.default_rng(seed)
            x = np.stack([rng.choice(116, 10, replace=False) + 1 for _ in range(32)])
            ours = own(torch.from_numpy(x.reshape(-1, 2, 5))).sigmoid()
            theirs = author(x)
            forward_error = float((ours - theirs).abs().max().detach())
            weights = torch.linspace(0.1, 1.0, len(x), dtype=torch.float64)
            (ours * weights).sum().backward()
            (theirs * weights).sum().backward()
            own_params, author_params = (
                dict(own.named_parameters()),
                dict(author.named_parameters()),
            )
            gradient_error = 0.0
            for left, right in MAPPING.items():
                a, b = own_params[left].grad, author_params[right].grad
                if a is None and b is None:
                    continue
                if a is None or b is None:
                    raise AssertionError(f"Gradient participation differs: {left}")
                gradient_error = max(gradient_error, float((a - b).abs().max()))
            if max(forward_error, gradient_error) > 1e-10:
                raise AssertionError("Reference equivalence failed")
            results.append(
                {
                    "seed": seed,
                    "attention": attention,
                    "max_probability_error": forward_error,
                    "max_gradient_error": gradient_error,
                }
            )
    payload = {
        "status": "passed",
        "reference_path": str(args.reference),
        "reference_sha256": file_hash(args.reference),
        "project_implementation_sha256": file_hash(Path("src/gamematch/models/neuralac.py")),
        "scope": "CPU float64, 32 valid random rosters per seed, five initializations, attention on and off; all shared parameter gradients",
        "reference_adjustment": "Author attention biases set to zero to match paper equations; both models in evaluation mode with autograd enabled",
        "not_established": "Original experimental protocol or published score reproduction; training dropout random-number streams differ",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": len(results),
                "max_probability_error": max(r["max_probability_error"] for r in results),
                "max_gradient_error": max(r["max_gradient_error"] for r in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
