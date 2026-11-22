# Outcome models, inputs and execution

OwO and DAWN are outcome predictors, alongside roster and tabular predictors.
Their history input contract is not a different target task. Model architecture,
available information, executable adapter and empirical evidence are separate.
Version `0.1.0rc5.dev0` adds a machine-readable catalog and an outcome entry point.
The frozen rc4 experiments and package remain unchanged in the parent workspace.

```bash
gamematch models
gamematch models --task outcome_prediction
gamematch input-profiles
gamematch check-models --profile overwatch-history --models history_deepsets owo dawn
gamematch check-models --profile league-history --models owo dawn
gamematch outcome-benchmark --config examples/uci-smoke.json --dry-run
gamematch outcome-benchmark --config examples/uci-smoke.json --output runs/uci-new
```

The current catalog contains **32 IDs**: 28 base model or baseline entries and four named NeuralAC variants. A model such as MLP has one ID even when it serves multiple tasks. For the four entries added in dev3, see the [roster and rating examples](model-additions.md).

## Compatibility and execution are distinct

Input profile | Target | Outcome estimators | Execution
--- | --- | --- | ---
Hero rosters | Binary loss/win | Prior, Linear, Signed Hero, Logistic, LightGBM, FM, MLP, Deep Sets, Set Transformer, NeuralAC and its four variants | Unified native runner
Overwatch history | Loss/draw/win | Static/History Deep Sets, OwO Transformer, OwO, DAWN | Unified native runner; all three classes retained
LoL player history | Binary loss/win | History Deep Sets, OwO; aggregated Logistic/LightGBM | Separate frozen research drivers
LoL player history | Binary loss/win | Static Deep Sets, OwO Transformer, DAWN | General model API; full dataset driver and results not yet integrated
DBD context | Ordered 0–4 eliminations | Prior, History Frequency, Logistic, LightGBM, MLP, Ordinal MLP | Unified native runner
Overwatch event replay | Loss/draw/win | Elo, TrueSkill, OpenSkill Plackett–Luce, Bradley–Terry full and Thurstone–Mosteller full | Unified native runner with delayed state updates

The future-rating models belong to regression and remain under `skill-benchmark`.
Their observed-rating target must not become a match-outcome label or ACC score.
`check-models` returns both `input_compatible` and `native_runner_ready`; a successful
input check does not imply that a full training driver or empirical result exists.
The check reads metadata only. Native runners subsequently validate actual data,
split and output identities. Requests for missing adapters fail before loading
data or creating run directories. No features are silently invented or discarded.

The unified command dispatches the existing `pilot`/`final`,
`temporal_completed_player_history`, `temporal_predator_outcome` and
`temporal_delayed_feedback` protocols. Native configurations, artifact layouts,
selection procedures and class semantics remain unchanged. Existing task-specific
commands still work. Use new output directories: development code is not a valid
replacement for a frozen run's archived source.

## History models without game-specific class assumptions

```python
from gamematch.models.outcomes import HistoryOutcomeSpec, build_history_outcome_model

spec = HistoryOutcomeSpec(
    history_features=25,
    static_features=6,
    team_sizes=(5, 5),
    history_length=10,
    outcome_labels=("loss", "win"),
    domains=1,
)
model = build_history_outcome_model("dawn", spec, width=16, heads=2, layers=1)
# logits = model(history, static, history_valid, domain)
# logits[:, i] corresponds to spec.outcome_labels[i]; use cross entropy.
```

DAWN needs an explicit domain vocabulary and target-domain vector. A constant
zero domain in a single-mode dataset is valid computation, but cannot establish
cross-mode transfer. Binary 5v5 and three-class 6v6 use the same construction API.
The current tensor encoder requires two equal-size teams. A 1v4 request fails;
padding to 4v4 without a player mask would change the modeled task. A hero-only
file cannot supply persistent player histories without a new declared adapter.

Defaults use unordered players and team/view pooling, an explicit adaptation.
They do not replace the original global-readout reference or a selected HPO
configuration. Restore exact saved parameters and configuration for inference.

`validate_history_batch` checks shapes, masks, numeric values, target/domain
indices and, when both timestamp tensors are supplied, strict feedback
availability (`history_available_at < prediction_time`). Call it at the data
boundary to avoid repeated GPU synchronization. It explicitly reports whether
time evidence was supplied; it cannot certify raw-source truth, train-only
normalization or partition independence. Empty player histories are supported.

`python examples/history_outcomes.py` exercises one CPU training update and exact
checkpoint reload for Deep Sets, OwO and DAWN on generated binary and three-class
inputs. These fixtures verify software behavior and are not efficacy experiments.

## Comparisons and paper terminology

Compare architectures under the same input information, split, target definition
and declared selection/calibration budgets. Compare additional history fields in
separate information-set contrasts. A model's higher score with additional fields
does not isolate an architecture effect. Results from different tasks must not
share an overall ACC ranking.

Suggested manuscript wording: “Outcome prediction includes roster-based, player-
history-based and domain-conditioned estimators. Model compatibility is defined
by the input contract; executable adapter coverage is reported separately. OwO
and DAWN are public-data outcome-model adaptations, not isolated task categories.”
