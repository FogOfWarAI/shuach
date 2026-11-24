# Supplying inputs

This guide is about the **input side** of the [unified feature contract](feature-schema.md): what array to hand over, in what shape, and how the framework lifts it into the canonical layout. For the semantics of each real dataset (which columns exist and when they become available), see [Data and evidence contracts](data-contracts.md).

The short version: declare a `FeatureSchema`, pass natural-shaped arrays to `FeatureBatch.from_natural`, call `validate()`, then either encode (reference / trainable) or inspect. You never hand-write the `[B,S,N,T,C]` tensor yourself.

## The four axes in one table

Every input is placed by four independent declarations on its `FeatureSpec`:

| Axis | Values | What it controls |
| --- | --- | --- |
| `modality` | `COEFFICIENT`, `DENSE`, `SEQUENCE` | How channels are interpreted and encoded |
| `scope` | `MATCH`, `TEAM`, `PLAYER` | Which of the side / player axes are present |
| side count | `schema.sides` = 1 or 2 | Whether a side axis exists at all |
| `structure` | `SET`, `ORDERED`, `SINGLE` | Inductive bias: pooling and side-swap symmetry |
| `available_at` | timestamp key, or `None` | The single timing predicate (see below) |

There is **no PRE/POST binary**. Availability has exactly one predicate:
`available_at < prediction_time`, evaluated by `schema.at(t)`. `available_at=None`
means a static field with no temporal constraint. Labels are declared by
`OutputSpec` and never appear among the features.

The canonical shape is always `[B, S, N, T, C]` — batch, sides, players per side, time steps, channels. Axes that do not apply collapse to 1:

- `scope=MATCH` → `S=1, N=1`
- `scope=TEAM` → `N=1`, `S=schema.sides`
- `scope=PLAYER` → `S=schema.sides, N=schema.players`
- `modality=SEQUENCE` → `T=spec.steps`; otherwise `T=1`

`schema.canonical_shape(spec, B)` returns the exact 5-D target, and `schema.natural_shape(spec, B)` returns the shape you are expected to pass in.

## Natural shape by cell

`schema.natural_shape(spec, B)` always includes the channel axis, even when `C=1`. `from_natural` additionally accepts a trailing `C=1` axis omitted, so both forms below work. `B` is your batch size, `S` is `schema.sides`, `N` is `schema.players`, `T` is `spec.steps`, `C` is `spec.width`.

| Scope | Modality | Natural array you pass | Lifted to |
| --- | --- | --- | --- |
| MATCH | COEFFICIENT / DENSE | `[B, C]` (or `[B]` when `C=1`) | `[B,1,1,T,C]` |
| TEAM | COEFFICIENT / DENSE | `[B, S, C]` (or `[B,S]` when `C=1`) | `[B,S,1,T,C]` |
| PLAYER | COEFFICIENT / DENSE | `[B, S, N, C]` (or `[B,S,N]` when `C=1`) | `[B,S,N,T,C]` |
| any | SEQUENCE | same as its scope with a `T` axis added: `[B,S,N,T,C]` etc. | `[B,S,N,T,C]` |

Worked examples from the built-in schemas:

```python
import numpy as np
from gamematch.features import SCHEMAS

schema = SCHEMAS["hero_binary"]           # sides=2, players=5, one COEFFICIENT/PLAYER field
spec = schema.features[0]                  # hero_id: width=1, vocab=114
print(schema.natural_shape(spec, 4))       # (4, 2, 5, 1)   -- C axis included
print(schema.canonical_shape(spec, 4))     # (4, 2, 5, 1, 1)
# from_natural also accepts (4, 2, 5): a trailing C=1 axis may be omitted

schema = SCHEMAS["overwatch_history"]      # SEQUENCE/PLAYER + DENSE/PLAYER + COEFFICIENT/MATCH
for spec in schema.features:
    print(spec.name, schema.natural_shape(spec, 4))
# history       (4, 2, 6, 10, 5)
# static        (4, 2, 6, 13)
# target_mode   (4, 1)             <- MATCH scope, C=1 (pass (4,) if you prefer)
```

## The three modalities

### COEFFICIENT — discrete IDs with optional weights

Use for hero IDs, participant IDs, mode IDs, any categorical code. `vocab_size` is required; index `0` is reserved as padding/unknown and contributes nothing. Pass either the ID array or a `(ids, weights)` tuple:

```python
from gamematch.features import FeatureBatch

ids = np.array([                       # [B, S, N] = [2, 2, 5] for two 5v5 rosters
    [[3, 7, 11, 0, 0], [2, 5, 9, 13, 17]],
    [[1, 4, 8, 12, 15], [6, 10, 14, 18, 21]],
])
weights = np.where(ids > 0, 1.0, 0.0)  # optional; defaults to all ones
batch = FeatureBatch.from_natural(schema, hero_id=(ids, weights))
```

`reduce` controls aggregation across the slot/time axes: `sum` (default), `mean`, or `max`. The validator rejects indices outside `[0, vocab_size)` and non-finite weights.

### DENSE — fixed-width real vectors

Use for ratings, per-player summaries, numeric context. Pass a float array whose last axis is exactly `spec.width`:

```python
ratings = np.random.default_rng(0).normal(size=(B, 2, 6, 2))  # rating_state: width=2
```

No normalization is applied by the contract: scaling belongs in your adapter or in a train-fitted transform. Keep the scale consistent between training and inference.

### SEQUENCE — time steps with a validity mask

A sequence field must declare `steps >= 2` and `mask=True`. Supply the array plus a boolean mask under the key `f"{name}__valid"`:

```python
schema = SCHEMAS["overwatch_history"]      # history: [B,2,6,10,5], mask key history__valid
rng = np.random.default_rng(0)
B = 4
raw = {
    "history": rng.normal(size=(B, 2, 6, 10, 5)),
    "history__valid": rng.random((B, 2, 6, 10)) < 0.85,
    "static": rng.normal(size=(B, 2, 6, 13)),
    "target_mode": rng.integers(0, 2, size=(B, 1)),
}
batch = FeatureBatch.from_natural(schema, **raw)
```

Masked steps are ignored by the encoder and by pooling. Sequences are typically right-aligned with missing steps masked — keep that convention consistent. The reference encoder reduces a sequence by masked mean; `reduce=last` is available for declarations that need it.

## Timing: one predicate, not a binary tag

"Post-game" describes **when you predict**, not a kind of field. A post-game
forecast (Dota future rating) legitimately uses post-game snapshots as inputs;
a pre-match prediction does not. Both are expressed by the same rule:

```python
view = schema.at(prediction_time, available_at)
print(view.field_names)      # fields observable at t
print(view.excluded)         # field -> count of violating steps
visible_schema = view.schema # reduced schema, ready to build a model
```

- `available_at=None` on a spec → no temporal constraint, always visible.
- `available_at="some_key"` → the batch must provide that timestamp array, and
  every observed step must satisfy `available_at < prediction_time` (strict).
- `predictor_schema(schema)` is a legacy alias; with no arguments it keeps all
  fields. Pass `prediction_time` to get the same evaluation as `at(t)`.

At batch level:

```python
batch = FeatureBatch.from_natural(
    schema,
    prediction_time=prediction_time,                       # scalar UTC seconds, or [B]
    available_at={"history_available_at": times},           # key = spec.available_at
    **raw,
)
batch.validate()   # asserts strict < for every observed step
```

The key of `available_at` is the **name declared on the spec** (for
`overwatch_history` that is `history_available_at`, not `history`). An older
form keyed by the feature name is still accepted. Equality is not enough:
feedback tied exactly to the prediction time is rejected.

Targets are separate: pass them as `targets=`, derived from `OutputSpec` by
`sample_inputs`. They are never encoded and never enter `ModelInputs`.

## Structure and orientation

`structure` declares the inductive bias of a scope's member axis; it is
inferred when omitted (`MATCH` → `SINGLE`, `TEAM` → `ORDERED`, `PLAYER` → `SET`).

| `structure` | Meaning | Effect in `SchemaModel` |
| --- | --- | --- |
| `SET` | exchangeable members (hero roster) | permutation-invariant masked-mean pooling |
| `ORDERED` | ordered slots (timeline, side order) | learnable slot embeddings before pooling |
| `SINGLE` | one entity (map, mode) | no pooling |

`OutputSpec.orientation` declares what swapping the two sides does to the
label. `ANTISYMMETRIC` (win/loss) uses a signed side difference and a bias-free
head, so `f(swap(x)) == -f(x)`, and `model.augmented_batch(batch)` flips labels
accordingly. `SYMMETRIC` and `NONE` use permutation-invariant pooling. For
three-class WDL, augmentation reverses the class order rather than negating
logits.

`model.criterion()` and `model.targets(batch)` derive the loss and target shape
from `OutputSpec` — no hand-written `BCEWithLogits` / `CrossEntropy` switch.

## The fastest path: generate a batch from a schema

`sample_inputs` builds a complete, valid batch from declarations alone — useful for smoke tests, tutorials, and checking a new adapter before wiring real data:

```python
from gamematch.features import sample_inputs

batch = sample_inputs(SCHEMAS["overwatch_history"], batch=16, seed=0)
report = batch.validate()      # raises on any violation
```

Swap it for `FeatureBatch.from_natural(schema, ...)` once your loader exists. Every array is shaped from the schema, so a spec change automatically changes the sample — no shape constants to keep in sync.

## From arrays to a trainable model

The trainable bridge consumes the same batch; no reshaping in between:

```python
from gamematch.features import make_model_from_schema

model = make_model_from_schema(SCHEMAS["overwatch_history"], dim=16)
logits = model(batch)                  # head built from schema.task
probs = model.predict_proba(batch)     # declared link applied
```

Gradients reach every declared feature. See [Unified feature contracts](feature-schema.md#train-a-model-straight-from-a-schema) and `examples/schema_model.py` for a full training loop.

## Built-in schema reference

| Schema key | sides × players | Fields (modality / scope) | Task |
| --- | --- | --- | --- |
| `hero_binary` | 2 × 5 | `hero_id` (coef/player, vocab 114) | binary |
| `overwatch_rating` | 2 × 6 | `rating_state` (dense/player, w2), `calibration_terms` (dense/match, w4) | multiclass, 3 labels |
| `overwatch_history` | 2 × 6 | `history` (seq/player 10×5), `static` (dense/player 13), `target_mode` (coef/match, vocab 2) | multiclass, 3 labels |
| `league_history` | 2 × 5 | `history` (seq/player 10×5), `static` (dense/player 13) | binary |
| `dbd_predator` | 1 × 1 | `environment_id` (coef/match, vocab 16), `avatar_id` (coef/match, vocab 32), `history_numeric` (dense/match, w9) | ordinal, 5 labels |
| `dota_future_rating` | 1 × 1 | `snapshots` (seq/player 12×6, ORDERED, timed), `lineups` (coef/team 5, vocab 114), `current_mean` (dense/match) | regression |
| `synthetic_world` | 2 × 5 | `participant_id` (coef/player, vocab 101) | binary |
| `decision_assignment` | 2 × 5 | `lobby` (coef/match, w10, vocab 101) | assignment |

Inspect any of them exactly as your code sees it:

```bash
gamematch feature-schema --profile overwatch-history
```

Catalog profile names differ from schema keys (`hero-roster` → `hero_binary`, and so on); `catalog.feature_schema_for(name)` and `catalog.feature_schema_for` in the CLI resolve the mapping. Table widths mirror the current adapters — changing an adapter means updating its spec too.

## Common errors

| Message | Cause | Fix |
| --- | --- | --- |
| `expected natural shape ..., got ...` | Wrong array rank or axis order | Compare with `schema.natural_shape(spec, B)`; a trailing `C=1` axis may be omitted |
| `Missing feature array` | A declared field was not supplied | Pass every `spec.name`, plus `name__valid` for masked fields |
| `...__valid expects natural mask shape ...` | Mask given in the wrong rank | Mask shape is the field's natural shape minus the channel axis, e.g. `[B,T]` for MATCH × SEQUENCE |
| `available_at refers to unknown feature` | Timestamp key matches no spec | Use `spec.available_at` as the key, not an arbitrary name |
| `feedback at ... is not strictly earlier` | Feedback tied to or after prediction time | Shift timestamps back, or set a later `prediction_time` for a post-game task; `<` is strict |
| `index ... outside vocabulary` | Coefficient ID ≥ `vocab_size` | Use `0` for unknown/padding; raise `vocab_size` only if the data really needs it |
| `steps > 1 is only valid for SEQUENCE` | Non-sequence field declares multiple steps | Move the time axis into a `SEQUENCE` spec with `mask=True` |
| `SchemaModel ... requires the neural extra` | Trainable bridge used on a base install | `python -m pip install -e '.[neural]'` |

`validate()` returns a report on success (track, shapes, cells, timing evidence) and raises `ValueError` listing every problem on failure. Call it once at the adapter boundary — it is cheap compared to training.
