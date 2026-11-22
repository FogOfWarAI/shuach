# Getting started

## Install and check the version

Use Python 3.10 or later. Start at the repository root or the unpacked source
directory containing `pyproject.toml` and `examples/`. On Windows PowerShell,
activate the new environment with `.\.venv\Scripts\Activate.ps1`. On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gamematch --version
gamematch doctor
```

The version command prints `gamematch 0.1.0rc5.dev3`. If it prints a different
version, run `python -m gamematch --version` in the activated environment and
check the package location with `python -m pip show gamematch-research`.
If you have only the wheel, install its local filename with pip and try
`gamematch lobby-demo --seed 0` from any directory. The source distribution
also includes the example scripts and JSON configs used below.

| To use | Install from the source directory |
| --- | --- |
| CPU quickstart, logistic regression, PyTorch models or assignment search | `python -m pip install -e .` |
| LightGBM benchmarks | `python -m pip install -e '.[tabular]'` |
| TrueSkill, OpenSkill or Dota reference reconstruction | `python -m pip install -e '.[rating]'` |
| LoL Parquet preparation | `python -m pip install -e '.[parquet]'` |
| Research plotting | `python -m pip install -e '.[research]'` |
| Contributor checks, including optional model tests | `python -m pip install -e '.[tabular,rating,dev]'` |

## Complete a run without downloading data

```bash
python examples/constrained_assignment.py
gamematch decision-benchmark --config examples/synthetic-smoke.json --output runs/example
gamematch report runs/example
```

The synthetic example trains logistic regression and an MLP, saves checkpoints, and evaluates independent generated lobbies with several probability transformations and solvers. It uses CPU and does not download data. Its small sample verifies usage, not research effectiveness. Inspect `runs/example/manifest.json`, each model's `metrics.json` and `decision-metrics.json`, and the saved prediction archives.

Open `runs/example/report/report.md` for the two trained models and the
assignment results. In `runs/example/report/summary.json`, this configuration
should give `model_jobs_present == model_jobs_declared == 2` and
`decision_rows_present == decision_rows_declared == 18` (two scorers, three
probability transforms, three solvers). To compare configurations, give each
a new run directory. See the [command reference](cli.md) for report destinations
and the [matching API](matching-api.md) to supply your own scorer.

For optional LightGBM on macOS, a compatible OpenMP runtime is needed. The current workspace's `scripts/run_local.py` locates PyTorch's bundled runtime and adds its directory only to a child process. On a fresh machine use the dependency's documented installation procedure; the library itself does not change system settings. TrueSkill and OpenSkill are optional rating dependencies.

## A public-data example

```bash
gamematch datasets
gamematch fetch-data uci-dota2 --data-root datasets
gamematch benchmark --config examples/uci-smoke.json --output runs/uci-example
```

The UCI download is about 1.8 MB. This CPU example uses a capped development partition and at most ten epochs; it is a usage check, not the full research result. The official test file is excluded from the `pilot` protocol. The complete reference configuration is in `benchmarks/local-full-heroes.json`.

After the example finishes, reload the trained model and connect it to exhaustive composition search:

```bash
python examples/hero_composition_search.py --artifact runs/uci-example/uci-dota2/seed-0-mlp
```

The script selects ten training-observed hero IDs (or accepts `--heroes` followed by ten IDs), evaluates all 252 ordered 5v5 character partitions, and prints the minimum predicted imbalance. It applies the saved vocabulary and checkpoint without retraining. These are hypothetical **hero compositions**, not human account assignments; the printed probability is a model estimate with no independent observed outcome for the selected composition. The separate synthetic decision example supplies a hidden evaluator when measured decision regret is needed.

`fetch-data` also supports `neuralac-dota2018`, `neuralac-lol`, `overwatch`, `dbd-2022`, `dbd-2023`, and `dota2-matches-v2`. It verifies the pinned size/SHA-256, extracts only the files required by adapters, and records provenance under `datasets/acquisition/`. Existing altered files raise an error. Downloads do not bypass authentication, accept changed publisher files silently, or resume partial HTTP transfers. Data have separate source licenses; DBD is noncommercial and the NeuralAC data license is not independently clarified. Raw data are not bundled with the package.

The Dota snapshot track additionally requires the rating extra and a derived reference:

```bash
gamematch fetch-data dota2-matches-v2 --data-root datasets
python scripts/prepare_dota_reference.py --data-root datasets
```

This reconstruction processes a large historical prefix and takes longer than the example. A previous reference created by a different builder is deliberately not overwritten. Use `--output` for a separate audit build, or execute the archived `reference-builder.py` to reproduce that exact historical artifact. The skill runner expects its reference under `datasets/derived/dota2-reference-v2/`; see the provenance and target limitations before treating it as a skill label.

## Search with your own scorer

```python
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments

ratings = {"Ana": 8, "Bo": 7, "Cy": 3, "Di": 2}

def team_totals(batch):
    return [
        tuple(sum(ratings[p] for p in team) for team in assignment.teams)
        for assignment in batch
    ]

result = minimize_assignments(
    constrained_assignments(ratings, (2, 2), parties=[("Ana", "Di")]),
    team_totals,
    objective=lambda totals: [abs(left - right) for left, right in totals],
)
assert result["assignment"].teams == (("Ana", "Di"), ("Bo", "Cy"))
assert result["cost"] == 0 and result["candidate_count"] == 2
```

Ana and Di stay together. Both teams total 10, so the minimum rating gap is zero. With ordered sides, there are two feasible assignments. The search scores both and returns the first in a tie. Larger lobbies can be slow because every feasible assignment is scored. The [matching API](matching-api.md) shows how to score in batches and how to use predicted win probabilities.

For a 1v4 outcome distribution, a different objective may be appropriate, for example deviation of expected predator eliminations from a chosen target. Such a target is a modeling choice, not a universal definition of fairness. The public DBD adapters do not identify all four prey, so they cannot supply validated reassignment predictions.

Run `python examples/asymmetric_assignment.py --target-eliminations 2` for a data-free 1v4 example. It keeps a two-person party together, supplies a five-class distribution from explicit toy role abilities, and minimizes deviation of expected eliminations from the chosen target. Changing the target changes the modeling objective. The probabilities are illustrative assumptions, not a trained DBD model or independent evidence of real-player fairness.

## Local verification

Install the contributor extras in the table above first.

```bash
python -m coverage run --source=gamematch -m unittest discover -s tests -v
python -m coverage report
ruff check src tests
python -m build
```

Tests generate their own small fixtures and do not require private data, downloads or a GPU. Full benchmark configurations explicitly name MPS where selected; replace these settings with CPU in a new configuration on other hardware. The runner refuses an unavailable device rather than silently changing the experiment.
