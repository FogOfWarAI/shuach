# Python API guide

With a trained hero predictor, two lists of five **hero IDs** are enough to score a lineup. For a **player rating**, you also need persistent player IDs and the times when previous results became available. [中文 API 指南](api-zh.md)

| What you have | What you want | Entry point |
| --- | --- | --- |
| A small lobby, party or role restrictions, and a way to score a team | The feasible split with the smallest cost | [Generate and score assignments](matching-api.md) |
| Two completed hero lineups and match outcomes | Train a left-side win predictor | `gamematch.outcomes.run_outcomes` |
| A trained run and new observations | Predict with the saved vocabulary, preprocessing and calibration | [Choose a predictor](inference-api.md) |
| Time-ordered player matches with results | Replay an online rating system | [Rating adapters](#rating-adapters) |
| A new dataset or a different neural architecture | Inspect record fields, shapes and factory signatures | [Research API reference](research-api.md) |

## Try a four-player lobby

Ana, Bo, Cy and Di have ratings of 8, 7, 3 and 2. Split them into two teams of
two with the same total rating. Swapping the teams leaves this cost unchanged,
so there are only three distinct partitions to check.

```python
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments

ratings = {"Ana": 8, "Bo": 7, "Cy": 3, "Di": 2}

def totals(batch):
    return [
        tuple(sum(ratings[p] for p in team) for team in assignment.teams)
        for assignment in batch
    ]

result = minimize_assignments(
    constrained_assignments(ratings, sizes=(2, 2), exchangeable_sides=True),
    score_candidates=totals,
    objective=lambda scores: [abs(left - right) for left, right in scores],
)
assert result["assignment"].teams == (("Ana", "Di"), ("Bo", "Cy"))
assert result["cost"] == 0 and result["candidate_count"] == 3
```

Both teams total 10. `score_candidates` receives a list of assignments. The
objective must return one finite real cost for each item in that list. Equal
costs select the first candidate in enumeration order. An empty feasible set
raises `ValueError`. Add party and role rules using the [matching reference](matching-api.md).

## Check whether a model can run on your data

```python
from gamematch.catalog import check_models
from gamematch.outcomes import plan_outcomes

config = {
    "protocol": "pilot",
    "datasets": ["uci-dota2"],
    "models": ["signed_hero", "set_transformer"],
    "seeds": [0],
}
plan = plan_outcomes(config)
assert plan["profile"]["name"] == "hero-roster" and plan["native_runner_ready"]

check = check_models(profile="league-history", models=["owo", "dawn"])
assert [row["execution"] for row in check["models"]] == ["external_driver", "model_api_only"]
```

- `check_models(profile, models)` checks registered names and returns a record
  per model. Read `input_compatible` for the input contract and `execution`
  for the available entry point. Unknown or duplicate IDs raise `ValueError`.
- `plan_outcomes(config)` reads an in-memory configuration. An incompatible
  native request raises `ValueError` before data access.
- `run_outcomes(config_path, workspace, output)` reads a **JSON file** and runs
  the native adapter. Use the workspace that contains your data and a fresh
  output directory. The returned `Path` points to `report.md` for hero models
  or `results.json` for history, asymmetry and rating tracks.

To train the two hero models above, follow the [worked model comparison](model-additions.md).
For a JSON listing of every registered ID use `gamematch models`.

## Train and reload a hero predictor

| Step | Python entry point | Inputs and result |
| --- | --- | --- |
| Prepare data | `gamematch.data.heroes.prepare` | Dataset key and data-root `Path`. Returns train/valid/test records, a vocabulary fitted on training heroes, and a manifest. |
| Build a network yourself | `gamematch.models.make_model` | Model ID and vocabulary size. Pass encoded int64 rosters shaped `[matches, 2, 5]`. The network returns one logit per match. Apply sigmoid once to get a probability. |
| Predict from a trained run | `gamematch.inference.HeroPredictor` | Pass the model's seed directory to the constructor. Its `predict(rosters)` accepts **raw integer hero IDs** in the same shape and returns probabilities directly, one per row. |

Keep the model directory in the run tree so the predictor can find the fitted
vocabulary. In `p[i]`, a value of 0.7 means the model estimates a 70% chance
that the **first** side in row `i` wins. The [predictor guide](inference-api.md)
includes the import, artifact path and a complete example for a saved fit.

## Rating adapters

`gamematch.models.ratings` exposes `TeamElo()`, `TrueSkillRating()` and
`OpenSkillRating(algorithm="PlackettLuce")`. Install the `rating` extra for the
latter two. The other accepted OpenSkill algorithm names are `BradleyTerryFull`
and `ThurstoneMostellerFull`. Use their [catalog IDs](model-additions.md) in JSON configs.

| Operation | Meaning |
| --- | --- |
| `model.features((left, right))` | Returns a scalar win score on a log-odds scale using the current player states. |
| `model.update((left, right), outcome)` | Updates player states. Encode a left loss as 0, a draw as 1 and a left win as 2. |
| `outcome-benchmark` with `temporal_delayed_feedback` | Delivers each outcome after it becomes available and fits a validation calibrator. Saved probability columns are left loss, draw, win. |

Calling `update` directly gives the caller control of the timeline. For a
prediction at time `t`, only deliver earlier outcomes with `available_at < t`.

## Read a run and diagnose a failed call

`gamematch.reporting.create_run_report(run_directory, output=None)` returns the
path to the generated Markdown report. The adjacent `summary.json` contains
declared and completed job counts. An aggregate mean appears when all required
seeds are present. [Artifact and replay guide](reproduction.md)

| Symptom | What to check |
| --- | --- |
| The model exists but the input profile is incompatible | Inspect `check_models(...)["models"]` for the reason. An Overwatch player-history model needs recorded histories. A hero draft supplies only character IDs. |
| A checkpoint file or vocabulary cannot be found | Point the predictor at the seed/model directory, with its original parents and preprocessing files. |
| Reusing a run raises a fingerprint error | Compare source, data and configuration with the run manifest. Choose a new directory for a changed experiment. |
| A metric rejects the prediction array | Binary metrics expect `[N]` positive-class probabilities. Categorical metrics expect `[N,C]` rows that sum to one. [Metric definitions](research-api.md#evaluation) |

## Exploratory hyperparameter search

The tuning engine consumes transformed partitions. For `fit`, `reload_model` and
`predict_raw` signatures see the [research API](research-api.md#exploratory-hyperparameter-search).
For selection budgets, seeds and recovery, see the [search protocol](hyperparameter-search.md).
