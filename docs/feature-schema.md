# Unified feature contracts

Shuach historically shipped seven parallel, hard-coded tensor contracts. The
table in [data contracts](data-contracts.md) described them in prose and
`catalog.py` stored `input_schema` as a label string; neither drove validation,
encoding or model construction. `gamematch.features` turns that table into an
executable contract. It depends only on NumPy.

## Four orthogonal axes

| Axis | Values | Why it exists |
| --- | --- | --- |
| Modality | `COEFFICIENT`, `DENSE`, `SEQUENCE` | ID-keyed sparse coefficients, fixed-width floats, and masked ordered time steps need different encoders. |
| Scope | `MATCH`, `TEAM`, `PLAYER` | The individual/team split needs a third level for map, mode, domain, lobby and prediction time. |
| Side | `sides = 1` or `2` | Two-sided matchmaking and single-sided tasks (predator, focal player) must never silently mix. |
| Availability | `PRE_MATCH`, `POST_MATCH`, `available_at` | Leakage prevention becomes a failing assertion instead of a paragraph of prose. |

## One canonical layout

Every field is lifted to `[B, S, N, T, C]`: batch, sides, players per side,
steps, channels. Scope decides which axes collapse to 1; only `SEQUENCE` has
`T > 1`. A `COEFFICIENT` field carries integer indices plus floating weights;
index `0` is the unknown/padding token.

| Scope / modality | Natural input | Canonical |
| --- | --- | --- |
| MATCH / dense | `[B, C]` | `[B,1,1,1,C]` |
| TEAM / dense | `[B,S,C]` | `[B,S,1,1,C]` |
| PLAYER / dense | `[B,S,N,C]` | `[B,S,N,1,C]` |
| MATCH / sequence | `[B,T,C]` | `[B,1,1,T,C]` |
| TEAM / sequence | `[B,S,T,C]` | `[B,S,1,T,C]` |
| PLAYER / sequence | `[B,S,N,T,C]` | `[B,S,N,T,C]` |

Callers supply natural-shaped arrays; `FeatureBatch.from_natural` does the
lifting and `validate()` checks shapes, dtypes, finiteness, mask alignment,
coefficient vocabulary bounds, post-match isolation and strict
`available_at < prediction_time`.

## Declare a track

```python
from gamematch.features import (
    FeatureSchema, FeatureSpec, OutputSpec, Modality, Scope, Availability,
)

schema = FeatureSchema(
    name="my_track",
    sides=2,
    players=5,
    task=OutputSpec("left_win", "binary"),
    features=(
        FeatureSpec("hero_id", Modality.COEFFICIENT, Scope.PLAYER, 1,
                    vocab_size=114, reduce="sum", group="roster"),
        FeatureSpec("history", Modality.SEQUENCE, Scope.PLAYER, 5,
                    steps=10, mask=True, available_at="history_available_at"),
        FeatureSpec("static", Modality.DENSE, Scope.PLAYER, 13),
        FeatureSpec("map_id", Modality.COEFFICIENT, Scope.MATCH, 1,
                    vocab_size=16),
    ),
)
print(schema.to_json())
```

Adding a feature is one `FeatureSpec` and one array — no adapter, model
signature or runner edits. Use `schema.cells()` to see which
(modality, scope) cells are occupied; `MATCH × SEQUENCE` and most `TEAM × *`
cells are still open, which is where queue context and party statistics belong.

## Inspect from the command line

```
gamematch feature-schema --profile overwatch-history
gamematch input-profiles
```

Every `InputProfile` now carries a `feature_schema` key and an
`executable_schema` property. `catalog.feature_schema_for(name)` is the public
entry point. `check_models()` output still uses the original string labels, so
existing scripts keep working.

```python
from gamematch.features import FeatureBatch, FeatureEncoder, predictor_schema

batch = FeatureBatch.from_natural(schema, hero_id=..., history=...,
                                 history__valid=..., static=..., map_id=...)
report = batch.validate()
inputs = FeatureEncoder(schema, dim=32)(batch)
# inputs.tokens [B,S,N,D], inputs.valid [B,S,N],
# inputs.match_context [B,D], inputs.team_context [B,S,D]
```

`predictor_schema(schema)` drops every `POST_MATCH` field — for example the
Dota snapshot features, which the source dataset explicitly declares as
post-game and therefore invalid for pre-match prediction.

## Output contracts

`OutputSpec` declares `binary`, `multiclass`, `ordinal`, `regression` and
`assignment` heads with an ordered label vocabulary, output dimension and link
function. This replaces the scattered `outcome_labels`, float labels,
five-column elimination counts, standardized rating residuals and assignment
dictionaries with one declarative object.

## Migration status and limits

- The contract layer is complete and wired to the catalog. **The existing
  runners still consume their native tensors**; `FeatureEncoder` is a
  deterministic reference encoder, not yet a trained `torch.nn.Module`. Migrate
  one model at a time with a regression comparison before switching a track.
- `SCHEMAS` covers all six catalog profiles plus `synthetic_world` and
  `decision_assignment`. Widths mirror the current adapters (5/13 history
  channels, 114 heroes); changing an adapter requires updating its spec.
- `MATCH × SEQUENCE` and `TEAM × SEQUENCE` are unoccupied cells, not
  unsupported cases — the layout and validator handle them.
- Synthetic decision intervals are exploratory marginal intervals. They do not
  fix the deeper limitation that oracle regret and training labels come from
  the same hidden generator.
