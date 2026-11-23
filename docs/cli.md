# Command-line reference

Run `gamematch --version` to identify the installed build. `python -m gamematch`
uses the same commands through the active Python interpreter. Run
`gamematch COMMAND --help` for accepted flags and defaults.

## Inspect the environment and choose inputs

| Command | What it returns |
| --- | --- |
| `doctor` | JSON with Python, PyTorch, device availability and import status for LightGBM, TrueSkill and OpenSkill. A missing optional dependency is reported in the JSON. |
| `datasets` | JSON registry of dataset keys, data terms, source locations, byte sizes and hashes. |
| `models [--task TASK]` | Model records and separate base/variant counts. Task filters are `outcome_prediction` and `future_observed_rating`. |
| `input-profiles` | JSON list of input schemas, target definitions and available adapters. |
| `check-models --profile PROFILE --models MODEL [MODEL ...]` | Per-model compatibility and execution route, plus `input_compatible` and `native_runner_ready`. |
| `fetch-data DATASET [--data-root DIR]` | Downloads or verifies the cached asset under `DIR` (default `datasets`), extracts required inputs and records a receipt at `DIR/acquisition/DATASET.json`. |
| `inspect --dataset DATASET [--workspace DIR]` | Reads a prepared static hero dataset under `DIR/datasets` using the pilot split, then prints its manifest. |

```bash
gamematch check-models --profile league-history --models owo dawn
gamematch outcome-benchmark --config examples/uci-smoke.json --dry-run
```

In the first example both models accept the inputs. OwO uses an external driver
and DAWN has a model API, so `native_runner_ready` is false. The command exits
successfully when the requested inputs are compatible. A known incompatible
pair prints its JSON explanation and exits with status 2. Unknown identifiers
also exit with status 2. In the second example the program validates the
declared native model/input combination and supplied training ranges without
opening a dataset. Hardware and actual data are checked when running the job.

## Run a benchmark

Training commands take `--config FILE`, `--output DIR` and optional
`--workspace DIR`. Relative config and output paths are resolved from the
current directory. The workspace defaults to the current directory and supplies
the base for the adapter's inputs. Use a fresh output for a changed experiment.

| Command | Task or protocol | Example config supplied in the source distribution |
| --- | --- | --- |
| `outcome-benchmark` | Dispatches the four native outcome tracks according to the config. Accepts `--dry-run` without `--output`. | `examples/uci-smoke.json` |
| `benchmark` | Hero-roster binary outcomes, pilot or final split | `examples/uci-smoke.json` |
| `history-benchmark` | Player-history loss/draw/win prediction | `benchmarks/local-overwatch-history.json` |
| `rating-benchmark` | Delayed-feedback rating replay and outcome calibration | `benchmarks/local-overwatch-ratings.json` |
| `asymmetric-benchmark` | Five elimination-count probabilities | `benchmarks/local-asymmetric.json` |
| `skill-benchmark` | Future observed-rating regression | `benchmarks/local-skill.json` |
| `decision-benchmark` | Synthetic outcomes and assignment regret. Requires the `neural` extra. | `examples/synthetic-smoke.json` |
| `feature-schema` | Print the executable input/output contract for an input profile (`--profile NAME`). Base install only. | — |
| `lobby-approx` | Anytime local-swap assignment for lobbies too large to enumerate (`--players`, `--time-budget`). Base install only. | — |

The two smoke configs select CPU. Check `training.device` and any
`model_training` overrides in the larger local configs before starting.
Training commands may print progress before the final path or JSON object.
The hero runner writes `report.md`; the other native runners write
`results.json`. The unified command prints the summary path in its final
`report` field.

## Evaluate assignments and inspect results

```bash
gamematch lobby-demo --seed 0
gamematch lobby-approx --players 20 --time-budget 1
gamematch feature-schema --profile overwatch-history
python -m pip install -e '.[neural]'
gamematch decision-benchmark --config examples/synthetic-smoke.json --output runs/first-decision
gamematch report runs/first-decision
```

`lobby-approx` reports `candidate_count` against `exact_candidate_count` so an
approximate answer is never mistaken for an enumerated optimum. A completed
decision run writes `decision-confidence.json`; `report` adds its intervals to
`report.md`. Both work in a base install; only the training commands need
PyTorch.

`lobby-demo` works from a wheel install alone. Its JSON includes
`candidate_count` (126 for the ten-player, exchangeable-side lobby),
the predicted win probability and regret against the hidden synthetic evaluator.
The latter two commands use the config from the source tree.

`report RUN [--output DIR]` reads a run manifest and saved metrics and prints
the path to the generated Markdown report. The default destination is
`RUN/report/`, alongside a machine-readable `summary.json`. The summary
records missing model jobs and decision rows. Aggregate means require all
declared seeds. A report aggregates existing results without fitting a model.

For model inputs, return shapes and artifact layout, see the
[Python API](api.md) and [predictor reference](inference-api.md).
