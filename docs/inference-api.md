# Predict with a saved fit

The four predictors in `gamematch.inference` load the output of a native benchmark. Create the predictor once and call `predict` for each new batch. Give the constructor the **seed/model directory** and keep it in the directory tree written by the runner. Several predictors also read preprocessing state from the parent directory.

## Score a hero lineup

From an installed source checkout, the UCI example trains a small Logistic model on CPU. The first command obtains the dataset and checks its hash. The benchmark also trains an MLP.

```sh
gamematch fetch-data uci-dota2 --data-root datasets
gamematch benchmark --config examples/uci-smoke.json --output runs/uci-example
```

Run this Python example in the same directory. The outer dimension holds matches, the next dimension selects the side, and the last dimension contains the five raw hero IDs for that side.

```python
import numpy as np
from gamematch.inference import HeroPredictor

predictor = HeroPredictor("runs/uci-example/uci-dota2/seed-0-logistic")
rosters = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]], dtype=np.int64)
left_win = predictor.predict(rosters)
assert left_win.shape == (1,) and 0 <= left_win[0] <= 1
print(f"Left-side win probability: {left_win[0]:.3f}")
```

The same predictor can score candidate compositions. The example reads ten observed IDs from the training vocabulary and returns the assignment whose predicted probability is closest to 0.5.

```sh
python examples/hero_composition_search.py --artifact runs/uci-example/uci-dota2/seed-0-logistic
```

## Choose a predictor

Cite the model stored in your checkpoint. A predictor can load several architectures. Find the trained model ID in the run configuration, then look up its source in [model papers and BibTeX](papers.md).

Here `RUN` is the output directory passed to the relevant benchmark and `N` is the number of examples. Supply at least one example to each of these four interfaces. The optional `batch_size` is a positive Python integer. It limits the neural forward pass. Tabular models may preprocess and score the whole input at once. The three constructors that take `device` accept `"cpu"` (default), `"cuda"` and `"mps"`. Requesting an unavailable accelerator raises `ValueError`.

| Constructor and artifact location | Arguments to `predict` | Returned NumPy array |
| --- | --- | --- |
| `HeroPredictor(directory, device="cpu", batch_size=1024)` at `RUN/DATASET/seed-SEED-MODEL` | `rosters` of integer hero IDs, shape `[N,2,5]`. The saved vocabulary maps unseen IDs to index zero. | `[N]` probability that the first side wins |
| `HistoryPredictor(directory, device="cpu", batch_size=128)` at `RUN/seed-SEED-MODEL` | `history`, `static`, `valid`, `domain` from the Overwatch history adapter, before standardization | `[N,3]` calibrated probabilities in left-side loss, draw, win order |
| `AsymmetricPredictor(directory, batch_size=1024)` at `RUN/DATASET/seed-SEED-MODEL` | `features`, a pre-match DataFrame from `load_asymmetric(...).features` | `[N,5]` calibrated probabilities for 0, 1, 2, 3 and 4 eliminations |
| `SkillPredictor(directory, device="cpu", batch_size=256)` at `RUN/seed-SEED-MODEL` | `snapshots`, `valid`, `lineups`, `current_mu` from the skill adapter | `[N]` estimated OpenSkill posterior mean after five further observed games |

**Overwatch histories.** Use `build_histories` with the history length and `same_domain_only` setting recorded in the run. `history[N,2,P,T,H]` contains right-aligned, completed observations and `valid[N,2,P,T]` is a boolean mask. `static[N,2,P,S]` holds the player summaries and `domain[N]` is 0 for noncompetitive and 1 for competitive. Match the checkpoint's player count and feature widths. The predictor applies the saved training normalizer, restores zero padding and applies the validation temperature. For target-domain-only fits it also masks the excluded static fields. The caller supplies the sequences under the original feedback timing contract.

**Dead by Daylight.** Use the same pre-event feature columns as the dataset adapter. The five columns `previous_kill_fraction_0` through `previous_kill_fraction_4` must form a probability distribution for each row of the history-frequency baseline. Learned models read the fitted `preprocessor.joblib` from the dataset directory. Logistic and LightGBM retain a column for every count. A count absent during fitting starts with zero probability before clipping and calibration. For a probability matrix `p`, the expected elimination count for each row is `p @ np.arange(5)`.

**Future ratings.** `snapshots[N,12,6]` contains the skill adapter's log-transformed performance and rating features before standardization. `valid[N,12]` must be boolean with at least one observed step per player. Each integer roster in `lineups[N,2,5]` places the focal player's team first. Supply the current rating means as a finite vector `current_mu[N]`. Training defines the timing of the post-game inputs and the five-game target. See the [data contracts](data-contracts.md).

## Replay indexed League of Legends histories

`gamematch.indexed_inference.IndexedHistoryPredictor(run_directory, data_directory, device="cpu", threads=2)` accepts a selected seed directory and the original indexed feature store. It checks the saved protocol, model source, selected artifacts and feature-store hashes. The `threads` argument sets PyTorch's process-wide intra-op thread count.

| Method | Result |
| --- | --- |
| `predict_logits(rows, batch_size=256)` | Raw logits `[N,2]` in team 100 loss, win order |
| `predict_proba(rows, batch_size=256, calibrated=True)` | Probabilities `[N,2]` in that order. Set `calibrated=False` to apply softmax at unit temperature. |

Pass a one-dimensional integer vector of **zero-based store positions**. Queries preserve order and repetitions. An empty integer vector yields shape `[0,2]`. Out-of-range indices or invalid batch sizes raise `ValueError`. The [indexed-history guide](richer-history.md) gives a complete call with the research artifact paths.

## Artifact compatibility

Load trusted fits. The tabular models and fitted preprocessors use joblib serialization, which can execute code during loading. The native raw-input predictors expect `metrics.json`, their checkpoint, and the associated training artifacts described above. The runners' resume checks and the indexed predictor's hash checks apply to their respective workflows. For tuned fits, use the [search API](api.md#exploratory-hyperparameter-search) with its own transformed partition and selected configuration.
