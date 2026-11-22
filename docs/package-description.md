# Shuach · 耍起

Shuach is the public project name. Install this distribution from source and use the `gamematch` Python package and command-line tool.

GameMatch is a Python research framework for competitive-team outcome prediction,
delayed player ratings, asymmetric outcomes, future observed-rating proxies, and
independently evaluated synthetic assignment decisions.

The local development version provides seven verified public-data acquisition
entries, PyTorch and tabular models, explicit OwO/DAWN/MMR-Net adaptations,
train-only transforms, time-aware evaluation, saved predictors, and constrained
batched exact search. It records source/data identities, split identifiers,
checkpoints and per-example predictions for independent audits.

The model catalog separates target task, input contract and executable adapter.
OwO and DAWN are outcome predictors with configurable history-model class heads.
`gamematch check-models` checks compatibility, and `gamematch outcome-benchmark`
dispatches compatible native outcome protocols without changing their semantics.

Optional player-history extensions provide delayed Overwatch summaries and
individual-performance histories from a separate public LoL collection. Compact
observation storage expands only requested batches; an indexed inference API
reuses audited saved models. Preparation and controlled comparison commands are
documented in `docs/richer-history.md` in the source distribution.

## First run

GameMatch requires Python 3.10 or later. In an activated virtual environment,
install the local wheel with `python -m pip install /path/to/package.whl`
(substitute the actual filename), or run `python -m pip install .` from the
unpacked source distribution. Check the installation and run a data-free example
from any directory:

```bash
gamematch doctor
gamematch lobby-demo --seed 0
```

The demo selects a partition of ten synthetic players and reports the
chosen teams, estimated win probability and regret against a known simulation.
For a complete train-and-report example, use the source archive. It includes
the `examples/` configurations, API reference, data contracts, model cards
and reproduction guide under `docs/`. Start with `docs/getting-started.md`
or the Chinese quickstart at `docs/start-zh.md`. For Python calls, read
`docs/api.md` (English) or `docs/api-zh.md` (中文). The examples in
`docs/model-additions.md` train two hero lineup models and replay three
OpenSkill backends on real match histories.

Project-authored code uses Apache-2.0. Raw datasets must be acquired separately
under each provider's terms. The public-data models adapt published architectures.
Evaluation on logged teams measures prediction performance. The assignment
experiments use an independent synthetic evaluator.
