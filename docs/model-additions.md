# Compare the added roster and rating models

Use the roster models when the ten hero picks are known. Use the rating models
when the same people appear in a sequence of matches and you know when each
result becomes available. All four IDs work through `outcome-benchmark`.

| Model ID | What it learns or updates | Input profile |
| --- | --- | --- |
| `signed_hero` | One learned effect per hero. Add the left team's effects, subtract the right team's effects, then add a learned side intercept. An unseen hero contributes zero. | `hero-roster` |
| `set_transformer` | Self-attention within each team, followed by pooling with a learned query. The readout combines the two teams in side order. Reordering heroes within a team preserves its prediction. | `hero-roster` |
| `openskill_bt_full` | OpenSkill's Bradley–Terry full-pairing update for player skills and their uncertainties. | `overwatch-ratings` |
| `openskill_tm_full` | OpenSkill's Thurstone–Mosteller full-pairing update for the same player records. | `overwatch-ratings` |

## Train on UCI, then score a new lineup

Install the base package and start in the source directory. The dry run checks
the requested model and dataset combination. The next two commands obtain the
data and train using a capped CPU pilot with one seed.

```bash
gamematch outcome-benchmark --config examples/uci-roster-models.json --dry-run
gamematch fetch-data uci-dota2 --data-root datasets
gamematch outcome-benchmark --config examples/uci-roster-models.json --output runs/roster-example
```

Run the following in the same working directory after training. The raw hero
IDs do not need manual reindexing. Each predictor reads the training vocabulary
from its parent directory.

```python
from pathlib import Path
import numpy as np
from gamematch.inference import HeroPredictor

dataset_dir = Path("runs/roster-example/uci-dota2")
rosters = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
for name in ("signed_hero", "set_transformer"):
    predictor = HeroPredictor(dataset_dir / f"seed-0-{name}")
    p = predictor.predict(rosters)
    assert p.shape == (1,) and 0 <= p[0] <= 1
    print(f"{name}: left-team win probability {p[0]:.3f}")
```

For the shipped configuration, the example lineup produced 0.497 for `signed_hero` and 0.539 for `set_transformer` on the checked CPU run. These are predictions for the single example. Check the run's `report.md` for evaluation across the held-out matches. `signed_hero` shares each
hero's coefficient across sides. The existing `linear` model can fit different
coefficients on each side. For `set_transformer`, `embedding_dim` is the width
per head (two heads), and `hidden_dim` sets the feed-forward width. GameMatch
uses the existing training loss, split and validation checkpoint selection.

## Replay player ratings on Overwatch

This source example uses all eligible events from the downloaded file. Install
the rating extra, then run the three OpenSkill models on the same event order.

```bash
python -m pip install -e '.[rating]'
gamematch fetch-data overwatch --data-root datasets
gamematch outcome-benchmark --config examples/overwatch-rating-models.json --output runs/ratings-example
gamematch report runs/ratings-example
```

The report includes four rows of model results: the validation-calibrated
prior, Plackett–Luce (`openskill`), Bradley–Terry and Thurstone–Mosteller.
The example pools domains. Each model directory contains `predictions.npz` with
three probability columns in **left loss, draw, left win** order. The rating
models also save player means and uncertainties in `rating-state.json`. A tie
is delivered using equal ranks. The runner fits calibration on validation
events and keeps it fixed during test replay.

## Sources and scope of the update

- **Signed hero effects.** The public Dota 2 Draft AI project reports a
  draft-only linear candidate with signed per-hero contributions and a failed
  temporal readiness check. GameMatch independently implements the shared
  additive form in PyTorch and uses its own datasets, optimizer and protocol.
  The upstream study's results belong to its own data. [Project and study](https://github.com/jbaptiste-q/dota2-draft-win-prediction)
- **Set Transformer.** The original method uses attention to encode and pool
  unordered sets. GameMatch implements a compact two-team adaptation using
  PyTorch, a shared per-team encoder and one pooling seed. [ICML paper](https://proceedings.mlr.press/v97/lee19d.html) · [Authors' implementation](https://github.com/juho-lee/set_transformer)
- **Rating backends.** Both adapters call the optional OpenSkill package
  directly. The original `openskill` ID still selects Plackett–Luce. Two
  additional IDs expose distinct update rules already provided by the upstream
  library. [OpenSkill model API](https://openskill.me/en/v6.0.0/api/openskill/models/index.html) · [JOSS software paper](https://joss.theoj.org/papers/10.21105/joss.05901)
- **A recent model with different data requirements.** Guan and Tung's
  September 2026 LoL preprint models professional-team strength, recent
  same-side results and first-pick information. The static hero adapter
  lacks persistent professional-team IDs and first-pick history. That
  forecasting protocol would need a separate adapter. [Preprint](https://arxiv.org/abs/2609.08060)
