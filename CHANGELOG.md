# Change log

## Unreleased — data-time vocabulary checks and documented layer boundary

- Add `validate_ids()` / `vocabulary_requirements()` to check raw participant
  IDs against declared coefficient vocabularies once at the adapter boundary.
  Padding and masked-out slots are excluded, so unequal teams padded to a
  common slot count are validated on real participants only. A mid-training
  `nn.Embedding` `IndexError` now becomes a named adapter error.
- Document the boundary between the decision layer (unequal teams, role slots,
  party constraints, budgeted search) and the input layer (fixed N per side,
  padding). Explain that `Orientation.ANTISYMMETRIC` is only meaningful for
  two exchangeable sides.
- Add 5 vocabulary validation tests. 288 tests pass.

## Unreleased — consistency review and framework design guide

- Align timing evaluation across `schema.at()` and `FeatureBatch.validate()`:
  shared key resolution (declared name or legacy feature name), scalar
  prediction-time broadcast, per-row alignment and observed-step masking.
  The reference encoder now applies the same visibility rule as the trainable
  bridge, and pools only valid players.
- Make declared behaviour honest: multi-class WDL now receives a side-sensitive
  context with reversed labels under swap; ordered-class augmentation reverses
  any label width; `swap_sides` flips only sided fields and never a MATCH time
  axis; `regression` with `dim>1` and one-hot class targets are handled; TEAM
  dense fields no longer create dead slot parameters; unimplemented `reduce`
  values (`max`, `last`) are rejected at declaration.
- JSON keeps raw declarations so round-trips are idempotent; `from_dict`
  infers `dim` from labels when omitted. Add 14 consistency regression tests.
- Add `docs/feature-framework-design.md` with an architecture diagram
  (`docs/assets/feature-framework.svg`) explaining the flow from declaration
  to model. 283 tests pass; coverage stays at 90%.

## Unreleased — time evaluation, structure semantics, derived behaviour

- Replace the PRE/POST match binary with one predicate:
  ``FeatureSchema.at(t)`` evaluates ``available_at < prediction_time`` and
  returns the observable fields. Post-game inputs (Dota snapshots) stay legal
  inputs for a post-game prediction time instead of being deleted by
  ``predictor_schema``. Labels are declared only by ``OutputSpec``.
- Add ``Structure`` (set / ordered / single) so pooling is derived from the
  inductive bias: ordered sequences get learnable slot embeddings, and
  ``OutputSpec.orientation`` drives signed side pooling plus bias-free heads
  for binary win/loss and side-swap augmentation with label reversal.
- Derive the loss and target shape from ``OutputSpec`` via ``model.criterion()``
  / ``model.targets()``; examples no longer contain a loss switch.
- Remove the declared-but-unused ``transform`` field. Old JSON configs with
  ``available`` / ``transform`` keys still load; the keys are ignored.
- Fix natural-shaped mask lifting so all nine modality x scope cells work,
  including MATCH x SEQUENCE. 269 tests pass; coverage stays at 90%.

## Unreleased — unified contracts, temporal assertions, decision intervals

- Bridge declarative schemas to trainable models: `SchemaEncoder` / `SchemaModel`
  / `make_model_from_schema` apply the four-axis algebra with learnable tables
  and build the head from `OutputSpec`. Lazy exports keep the base install
  PyTorch-free. `sample_inputs` builds a valid batch from any schema alone.
  `examples/schema_model.py` trains and predicts with no dataset; covers every
  catalog profile. This connects contract -> model -> prediction; the existing
  benchmark runners still consume their native tensors.
- Add `gamematch.features`: declarative `FeatureSpec` / `FeatureSchema` /
  `OutputSpec` contracts over four axes (modality, scope, side, availability),
  one canonical `[B,S,N,T,C]` layout, boundary validation, a reference encoder
  and JSON serialization. Every `InputProfile` now resolves to an executable
  schema; inspect one with `gamematch feature-schema --profile NAME`. See
  [unified feature contracts](docs/feature-schema.md).
- Enforce `available_at < prediction_time` on the real history runner path.
  `audit_history_timing` checks every observed step before standardization and
  applies the full `validate_history_batch` contract to a bounded sample; the
  result is recorded in the run manifest.
- Report decision quality with paired seed / lobby-cluster bootstrap intervals
  (`decision-confidence.json`, plus a section in `report.md`), including
  exact-minus-random contrasts. Point-estimate regret is no longer the only
  decision evidence.
- Add anytime assignment search: `minimize_assignments_budgeted` (time and
  candidate budgets, explicit `anytime_budget_reached` status) and
  `local_swap_balance` (batched cross-side swaps for lobbies too large to
  enumerate). `gamematch lobby-approx` demonstrates a 20-player case.
- Move PyTorch to the optional `neural` extra. A base install can now run
  `--version`, `doctor`, `feature-schema`, `lobby-demo` and `lobby-approx`
  without downloading CUDA wheels; `models/__init__.py` re-exports lazily.
  CI verifies the base install before installing `.[neural,tabular,rating,dev]`.
  The `neural` extra requires `torch>=2.3` (2.2.x cannot exchange tensors with
  NumPy 2.x), and CI asserts the resolved torch / NumPy pair before the suite.
- Slim the documentation set: `docs/release-readiness.md` is now a build and
  verification procedure, local validation logs and status placeholders are
  removed, and research exporters write to `reports/`. Two single-purpose
  export scripts were dropped from the source distribution.

## 0.1.0rc5.dev3 — roster models, rating rules and reader guides, 2026-11-21

- Train and reload `signed_hero` and `set_transformer` under the existing hero-roster protocol. The first model learns an additive effect per hero. The second learns interactions within each team.
- Select `openskill_bt_full` and `openskill_tm_full` alongside the original Plackett–Luce backend. Rating configurations that omit the model list retain their previous defaults.
- Follow [worked model comparisons](docs/model-additions.md) on UCI and Overwatch. The [Chinese quickstart](docs/start-zh.md), [中文 API 指南](docs/api-zh.md), [English API guide](docs/api.md) and [writing guide](docs/writing-guide.md) include examples and expected outputs.

## 0.1.0rc5.dev2 — public API and source distribution, 2026-11-20

- Added `gamematch --version`, a base-install CPU quickstart, a CLI reference,
  worked examples for constrained assignment and saved predictors, and
  contributor and private-reporting instructions.
- Check supplied training ranges in outcome planning, including selected
  hero models' overrides. Empty fitted hero vocabularies map every ID to zero.
- Preserve complete assignment enumeration with single-use role iterables.
  Match a string restriction as one role, reject complex solver scores, and
  support a valid optimum whose value is `None`.
- Support reversed NumPy views in history and rating inference and validate
  inputs used for probability calibration and the elimination-frequency baseline.
  DBD tabular fits retain the five elimination-count columns when the training
  sample omits a class. The same class mapping is used after reload.
- Derive package metadata from the importable version and exercise the console
  entry point and distribution inspector in the prepared CI workflow.

## 0.1.0rc5.dev1 — native artifact integrity, 2026-11-18

- Verify completed native runs against SHA-256 inventories before rewriting shared preparation files. Bind inventories to the experiment fingerprint, fit location and immutable manifest fields, including history normalizers and hero vocabulary. Inventory checkpoints, metrics, predictions and shared asymmetric preprocessing. Interrupted fits without a completion inventory restart through the normal runner.
- Write JSON through unique temporary files and atomic replacement, preserving the previous file on failure.
- Reject noninteger training counts and nonfinite optimizer settings before loading data or creating outputs, including model-specific hero overrides.
- Validate acquisition provenance destinations before downloading and write receipts atomically.
- Diagnose actual optional dependency loading, with actionable LightGBM installation and OpenMP errors. Create the distribution inspector's report directory in a fresh source installation.
- Gather requested indexed-history fields before expansion and normalization. Generated-data checks preserve returned arrays exactly while measuring lower temporary allocation for outcome-only features.

## 0.1.0rc5.dev0 — explicit outcome compatibility, 2026-11-15

- Added a catalog separating model task, input schema, native runner coverage and paper-reproduction scope. OwO and DAWN are outcome predictors; all 24 base IDs and four NeuralAC variants are represented without counting seeds or hyperparameters as models.
- Added `models`, `input-profiles`, `check-models` and `outcome-benchmark` commands. Invalid model/input requests fail before data loading or output creation; compatible protocols reuse existing training and artifact implementations.
- Added an explicit history outcome model API with configurable ordered class labels, equal team sizes and domain vocabulary. Added boundary checks for shapes, masks, class/domain indices and supplied strict feedback timing.
- Added binary 5v5 and three-class 6v6 training/reload examples. LoL DAWN is model-API support, not a completed integrated dataset experiment. Original frozen runs and rc4 distribution artifacts are preserved separately.

## 0.1.0rc4 — protected baseline recovery, 2026-11-14

- Check all locked LoL baseline candidate artifacts and selections before fitting or writing status. Missing locked evidence now fails without overwriting checkpoints; unfinished, unselected fits can still restart.
- Verify saved Logistic pipeline parameters against the declared regularization, seed and iteration budget, in addition to artifact hashes and prediction replay.
- Added seven regression tests for locked evidence, interrupted recovery, estimator settings and LightGBM compatibility. Existing frozen research artifacts retain their original source versions.

## 0.1.0rc3 — local richer-history candidate, 2026-11-14

- Added strictly delivered Overwatch history summaries, prior strength context and co-appearance features, retaining the original input fields for controlled comparisons.
- Added optional LoL individual-performance history preparation, hash-verified compact observation storage, batch-only expansion and indexed saved-model inference. Raw Parquet reading uses a separate optional dependency.
- Added frozen information-set/readout experiment drivers and complete-seed exporters with source, data, checkpoint, calibration and metric checks. Research results remain exploratory; the new LoL resource has a separate entry point from the seven-data acquisition registry.
- Rejected malformed categorical/binary targets and mismatched prediction shapes before fitting or scoring.
- Added schema-v2 search recovery checks binding inputs, transforms, seed/model identities, candidate inventories and saved artifacts. Completed resumes audit predictions and metrics without retraining; earlier experiment snapshots remain unchanged.
- Added an explicit CPU/MPS/CUDA override for neural hyperparameter-search candidates, with availability checks and immutable device declarations; omitted overrides retain the historical local plan.
- Expanded semantic, integration and installed-package checks across local dependency environments. Platform execution evidence is recorded separately from prepared CI configuration.

## 0.1.0rc2 — local tuning candidate, 2026-11-14

- Added bounded validation-only hyperparameter search with frozen spaces, source snapshots, two-seed confirmation, five-seed final evaluation and checkpoint replay checks.
- Added activation dropout to hero, ordinal and history feedforward networks while preserving reference checkpoint parameter names and default predictions.
- Added explicitly adapted team/view pooling and pre-normalization options for unordered-roster OwO/DAWN; reference configurations retain their previous defaults.
- Added probability MAE/RMSE to binary and categorical evaluation, with explicit class-axis scaling and separate task-specific outcome-score reports.
- Hardened search resumption so changed data are rejected even when jobs are already completed, without overwriting the saved preprocessor on rejection.
- Added archived-validation replay of search promotions and selections, complete-seed report gates and before/after research reporting. Expanded experiments are exploratory follow-ups after prior test access.
- Added the explicit task input/target inventory, complete-task progress reports, and validation/test row and label checks against reconstructed inputs.
- Keyed future-rating snapshot caches by all relevant source identities, preserving prior caches; rejected invalid match IDs/player slots before indexing and verified unchanged existing feature arrays.
- Unified Logistic iteration-cap rejection across reference and tuning entry points, and made tuning integrity checks remain active under optimized Python.
- Pruned repeated-role candidate permutations before construction while preserving candidate/tie order; removed repeated full domain-array conversion during history inference.

## 0.1.0rc1 — local release candidate, 2026-11-14

- Added delayed player rating/history, asymmetric predator outcomes, future observed-rating regression and independent synthetic decision tracks.
- Added NeuralAC, explicit OwO/DAWN/MMR-Net adaptations, tabular/rating baselines and component diagnostics.
- Added data availability and target-time purging, training-only transforms, complete per-example artifacts and independent metric/source audits.
- Added party/role/asymmetric candidate generation and batched exact search with explicit objective functions.
- Added API/model/protocol documentation, generated-fixture integration tests, coverage and Linux/macOS CI configuration.
- Added hash-pinned public-data acquisition, saved predictor interfaces, validation-only binary calibration, and executable CPU examples.
- Corrected Unicode archive serialization and canonical path handling across CLI/Python entry points.
- Fixed UTC-second conversion across pandas nanosecond/microsecond storage defaults; clean dependency installation detected the incompatibility.
- Hardened comparison identity checks across seeds, rejected silently rounded source fields, and made incomplete role constraints fail explicitly. Small constrained spaces are checked against independent exhaustive permutation enumeration.
- Made paired per-example losses reject fractional class labels, invalid probability vectors and regression broadcasting before scoring, so malformed external predictions cannot produce plausible comparison intervals.
- Added dependency/interpreter/platform identity to run-reuse checks so completed and resumed model jobs cannot silently mix software environments.
- Made the report command support every benchmark track in a dedicated output directory, preserve original artifacts, reject duplicate jobs and withhold aggregate means for incomplete seed sets. The legacy hero reporter rejects other protocols before writing.
- Packed NeuralAC batch/side and interaction axes before dense kernels, preserving parameter names and the evaluated function while avoiding high-rank MPS overhead. Archived the previous layout and measured both against the author implementation; retained the original trained experiment artifacts.
- Added an executable 1v4 five-outcome objective example with party preservation, and a systematic exploratory comparison under common calibration policies.

## 0.0.1 — development prototype

Static hero adapters, initial PyTorch baselines, development pilot and fixed-lobby demonstration. Frozen source snapshots in early and primary runs retain this version for provenance.
