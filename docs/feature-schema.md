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
| Timing | `available_at` + `schema.at(t)` | One predicate (`available_at < prediction_time`) replaces a fragile PRE/POST tag; "post-game" is a property of the prediction time, not of a field. |
| Structure | `SET`, `ORDERED`, `SINGLE` | Declares the inductive bias so pooling and side-swap augmentation are derived, not hand-coded. |

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
    FeatureSchema, FeatureSpec, OutputSpec, Modality, Scope, Structure,
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

`schema.at(prediction_time, available_at)` lists fields observable across
the whole supplied batch. Both encoders apply the same strict cutoff to
each candidate, player and step individually. A late history step is masked
only at its own position. The player's static features still contribute to
the team. With no timestamp evidence, scoring omits the timed field.
`batch.validate()` rejects late observed steps when auditing source data.
A post-game forecast (Dota
future rating) sets a later prediction time and keeps its snapshot inputs; a
pre-match prediction excludes them. Labels live in `OutputSpec`, never among
the features. `predictor_schema(schema)` is a legacy alias that keeps all
fields unless a prediction time is supplied.

## Output contracts

`OutputSpec` declares `binary`, `multiclass`, `ordinal`, `regression` and
`assignment` heads with an ordered label vocabulary, output dimension and link
function. This replaces the scattered `outcome_labels`, float labels,
five-column elimination counts, standardized rating residuals and assignment
dictionaries with one declarative object.

For the array shapes, mask rules, timing arguments and error messages on the input side, see [Supplying inputs](inputs.md).

## Train a model straight from a schema

With the `neural` extra installed, a declared contract becomes a trainable
model in three lines. `SchemaModel` applies the same four-axis algebra as the
reference encoder, but with learnable tables and projections, and builds the
output head from `OutputSpec`:

```python
from gamematch.features import SCHEMAS, make_model_from_schema, sample_inputs

schema = SCHEMAS["overwatch_history"]       # or your own FeatureSchema
model = make_model_from_schema(schema, dim=16)
batch = sample_inputs(schema, batch=32)     # or FeatureBatch.from_natural(...)
logits = model(batch)                        # [B, 1] or [B, num_labels]
probs = model.predict_proba(batch)           # link applied for inference
```

`synopsis`: a runnable version with a training loop is in
`examples/schema_model.py` — it works with no dataset and covers every
catalog profile:

```bash
python examples/schema_model.py --profile hero-roster --steps 20
```

The contract stays PyTorch-free: `gamematch.features` imports on a base
install, and only `SchemaModel` / `SchemaEncoder` / `make_model_from_schema` /
`make_head` require the `neural` extra.

## Generate sample data from any contract

`sample_inputs(schema, batch, seed=...)` builds a valid `FeatureBatch` from a
schema alone — natural shapes, masks, coefficient slots and strict
`available_at < prediction_time` are all filled consistently. Use it for smoke
tests, tutorials, or to validate a new adapter before wiring real data. Swap
it for `FeatureBatch.from_natural(schema, ...)` once your loader is ready.

## Migration status and limits

- The contract layer and a trainable bridge are complete and wired to the
  catalog. The `SchemaModel` path is ready for new tracks and demos. The
  **existing benchmark runners still consume their native tensors**; they have
  not been switched to `SchemaModel`. Migrate one track at a time with a
  regression comparison before treating results as equivalent.
- `SCHEMAS` covers all six catalog profiles plus `synthetic_world` and
  `decision_assignment`. Widths mirror the current adapters (5/13 history
  channels, 114 heroes); changing an adapter requires updating its spec.
- All nine (modality × scope) cells are exercised by tests, including
  `MATCH × SEQUENCE`; natural-shaped masks are lifted to canonical `[B,S,N,T]`.
- For exchangeable teams, binary logits change sign under side swap.
  Ordered class probabilities reverse, including when using one-hot targets
  for augmentation. The draw logit uses the team mean and absolute difference
  and can learn a mode-specific rate even when both teams have equal strength.
- Set `players_per_side=(1, 4)` for unequal teams, or
  `side_roles=("attack", "defend")` for distinct sides of equal size.
  Directional scoring preserves side order. To assign player roles pass
  `role_slots=(("captain", "flex", "flex"), ("captain", "flex", "flex"))`
  on a 3v3 schema. Arrange each side's player inputs in that order. Repeated
  role names share weights. An explicitly ordered static player field with
  unnamed slots uses the position index. Ordered sequences use time positions.
- The model, encoder, target conversion and side swap check the batch contract.
  When fitting a coefficient vocabulary, supply its ordered mapping fingerprint
  in `FeatureSpec.vocabulary_values_sha256` on both the adapter and the model
  schema. The native input contract exposes the digest under the same name.
  With the digest omitted, only the declared vocabulary size is available.
- New `SchemaModel.state_dict()` checkpoints include the contract hash.
  Use the original code and declaration for existing checkpoints without this
  fingerprint. The revised draw head, directional pooling and position gates
  change the architecture or interpretation of weights and require retraining
  before comparing results with the revised model.
- Synthetic decision intervals are exploratory marginal intervals. They do not
  fix the deeper limitation that oracle regret and training labels come from
  the same hidden generator.
