# Richer player histories (experimental)

`gamematch.data.rich_history.build_rich_histories(events, history_length=10)`
returns the existing `HistoryData` contract with eight sequence fields and 46
static fields. It is an opt-in extension; the original history builder and the
running, frozen searches retain their original feature definitions.

The first five sequence fields and first thirteen static fields are exactly the
original fields. Three additional sequence channels describe the prior match's
pre-match team Elo, opponent Elo, and observed result minus its pre-match
expectation. Elo starts at 1500 with a fixed K of 32 and updates only when an
outcome becomes available. This is a declared feature generator, not an estimate
of an official game rating.

The 33 added static channels comprise:

- Four statistics over each of the last 3/5/10/20 completed games: observed
  count, mean result, result standard deviation, mean log duration.
- Exponentially weighted result, signed streak length, recent-versus-earlier
  result difference, and smoothed three-class result entropy.
- Player, team and opponent Elo, expected result and update count.
- Previously observed teammate/opponent fractions and log encounter counts.
- Completed observations in the last 1/7/30 days and an explicit history flag.

Pair familiarity is shared exposure, not a causal synergy estimate. Counts
describe the available sample, not total games played. No KDA, hero, role, rank
or party field is invented for Overwatch.

All states use `available_at < prediction_time`. The current outcome cannot
change the current feature vector. Within-team permutations only permute the
player axis. Tests cover overlap, tied availability, target/future outcome
perturbation, side exchange, pair familiarity, and expiring activity windows.
Train-only normalization remains the caller's responsibility.

## Validation-only Logistic probe

The four information sets are original (5 sequence / 13 static fields), richer
sequence only (8 / 13), richer static only (5 / 46), and both (8 / 46). All use
the same mean/std team aggregation, train-fitted scaler and three Logistic
regularization candidates. The completed `reports/richer-history/probe-v2/`
experiment retains twelve fitted pipelines, validation predictions, split IDs,
selection records, source snapshots and runtime identity. Its `results.json`
reports validation NLL, accuracy, balanced accuracy, macro-F1 and predicted
class counts; `verification.json` records independent checks and frozen-source
resumption. Checkpoint reload errors are zero and the twelve candidate metrics
agree with the earlier exploratory table. Reusing validation labels for both
regularization selection and reporting makes these exploratory results.

The original `audit_rich_history.py` builds the feature cache and earlier probe.
To record a new probe, choose a fresh output directory:

```bash
GAMEMATCH_PROBE_OUTPUT=reports/richer-history/my-probe \
  .venv/bin/python scripts/run_local.py --python scripts/run_rich_feature_probe.py
```

The probe does not train OwO/DAWN or compute test scores. Resume an existing
probe with its archived driver and source snapshot, as described in the
[reproduction guide](reproduction.md); current source is not a substitute for
its frozen identity. These research artifacts are not bundled with the package.

## Completed reference neural comparison

The frozen `runs/rich-history-v1/protocol.json` declares **30 fits**: original
versus fully enriched inputs, crossed with global versus team readout for OwO
and DAWN (eight cells), plus History Deep Sets on the two input sets (two cells),
each with seeds 11, 22 and 33. It uses width 16, two heads, one layer, dropout 0.1, a 12-epoch
cap and validation stopping under a fixed configuration. Cells run in seed-major
order without outcome-based pruning. Input width and readout change parameter
counts even though optimizer and epoch budgets are shared.

All thirty fits have completed. The complete neural comparison export
contains two neural input sets; the sequence-only and static-only ablations
above are Logistic probes. Complete declared seed sets are required for comparisons.
The enhanced tensors have also
passed CPU forward/backward checks with History Deep Sets, Transformer, OwO
and DAWN; the standalone Transformer is not one of these thirty fits.

## Compact LoL histories and their audit

The new LoL candidate is stored under `datasets/raw/leagueData` in the research
workspace. `scripts/audit_league_data.py` verifies downloaded hashes and audits
identity, complete rosters, duplicate labels and prospective history coverage.
`scripts/prepare_league_history.py` has additionally constructed a compact
541,413,076-byte NumPy store under `datasets/derived/leagueData-v1/`, covering
288,601 retained 5v5 matches. Its purged chronological partitions contain
201,782 training, 43,055 validation and 43,291 test matches.

`gamematch.data.indexed_history.IndexedHistoryStore` expands referenced completed
observations in batches: history `[B,2,5,10,25]`, static `[B,2,5,6]`, and a valid
history mask. Twenty-four stored history components encode prior outcome,
duration, sixteen performance quantities and six historical-role indicators;
the twenty-fifth is recency relative to the target match. The target is the
first side's binary win outcome. Player identities are
lookup keys only. Current draft, role, performance and the sampled rank snapshot
are excluded. The builder checks duplicate consistency for its selected team,
outcome, role and sixteen performance fields, not every source column.

`reports/richer-history/league-independent-check.json` separates two levels of
verification. Across all stored arrays, 8,448,803 nonmissing history references
pass strict feedback-time, completion-order and right-alignment checks. Train-only
normalization was independently recomputed over 5,030,182 history entries and
2,017,820 static player appearances; maximum moment differences are below
`5.3e-12`. Direct source-PUUID correspondence and the 24 stored history components
were checked for seven sampled current matches and 222 historical references,
not for every raw record. The sample spans all three partitions and has zero
maximum stored-performance discrepancy. These checks do not establish source
authenticity or exhaustively verify same-player history completeness against
every raw appearance.

`scripts/run_league_baselines.py` compares outcome-history versus
performance-history Logistic/LightGBM inputs. All twelve validation candidates
and four selected test evaluations have completed locally. The independently
audited export is under `reports/richer-history/league-comparison/`.
`scripts/run_league_neural.py` separately runs History Deep Sets and a team-readout
OwO adaptation with explicit two-class heads, both information sets, and seeds
11/22/33. All twelve fits and eight aggregate model/information cells are complete
as of 15 November 2026. The exporter withholds incomplete three-seed cells.
Neural configuration is fixed, whereas each tabular cell
selects among three validation candidates; search budgets are unequal.
The new data remain an exploratory extension, separate from the seven reference
collections.

The producing source bytes are archived with the compact store. After later
source changes, verify the existing store without rebuilding or rewriting it:

```bash
python scripts/prepare_league_history.py --verify
```

A deliberately changed builder requires a new directory, for example
`python scripts/prepare_league_history.py --output datasets/derived/leagueData-v2`.
Do not replace the store used by a frozen experiment. Each experiment archives
its own driver and imported library. To resume after source changes, use the
archived driver with `GAMEMATCH_WORKSPACE` set to the checkout and `PYTHONPATH`
set to that experiment's `source-snapshot` directory. Protocol or runtime
differences cause an explicit error; saved selected fits are not silently
retrained.

`gamematch.indexed_inference.IndexedHistoryPredictor` reloads a selected binary
model, the exact hashed store, normalization and validation temperature. It
checks the producing architecture files before loading weights and expands
only the requested rows. For example, in the research checkout:

```python
import numpy as np
from gamematch.indexed_inference import IndexedHistoryPredictor

predictor = IndexedHistoryPredictor(
    "runs/league-neural-v1/outcome_history/history_deepsets/seed-11",
    "datasets/derived/leagueData-v1",
)
probabilities = predictor.predict_proba(np.array([100, 200]), batch_size=37)
# Columns: team100 loss, team100 win. These are store row indices, not player IDs.
```

This is replay against the original store, not a live matchmaking service or
an API for arbitrary new feature schemas. `scripts/verify_indexed_predictors.py`
independently checks up to 128 test rows for every completed predictor using
batch size 37, and records pending fits explicitly. Its `--require-complete`
flag fails if any declared fit is missing.

Reading the candidate's Parquet records requires the optional `parquet` extra;
the locally exercised PyArrow version is 21.0.0. In a source checkout, install it
with `python -m pip install -e '.[parquet]'`, then run
`python scripts/audit_league_data.py` after obtaining the fixed-source manifest
and all twelve verified shards. These data files are separate research-cache
prerequisites and are not included in the software distribution.

Base installation, `build_rich_histories`, `IndexedHistoryStore`, the Overwatch
feature probe and neural experiment driver, and the shard-manifest tests do not
require PyArrow. The prepared-store baseline driver additionally requires the
`tabular` extra for LightGBM, but does not read Parquet. The
manifest validator runs before the script imports PyArrow; its tests use small
non-Parquet fixtures and do not parse Parquet records. The source distribution
includes the audit scripts needed by these tests.

Native match IDs are absent; the adapter uses declared reconstructed contexts
from start time, duration and version, then requires coherent full rosters.
Neither an internally consistent store nor sampled source checks authenticate
those identifiers. Its single ranked mode does not independently test CHAMP's
original cross-mode premise or the effect of newly assembled teams.
