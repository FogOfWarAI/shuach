# Reproducing and auditing experiments

The full local configurations are in `benchmarks/`. `requirements-local-lock.txt` records executed versions, not a portable resolver lock. Copy and change the device explicitly for another host. Exact floating-point identity across Torch versions, hardware or MPS/CPU is not promised.

## Artifacts

Each run directory contains a manifest, resolved configuration and frozen `source-snapshot/gamematch`. Task outputs add exact split IDs, source SHA-256 values and train-fitted transformations. Each model records metrics and a checkpoint plus per-example predictions. NumPy archives use numeric or Unicode arrays and load with `allow_pickle=False`. Torch checkpoints are loaded with `weights_only=True`; joblib artifacts should come from a trusted producer.

For a long job, run from an installed wheel in a dedicated environment or use
an independent source copy on `PYTHONPATH`. An editable installation reads from
the working tree. Editing that tree changes its fingerprint and can cause a
subsequent resume to reject the directory. To replay an older experiment, use
its frozen package, resolved configuration and recorded dependency versions.
The native runners include Python, operating-system family, architecture and
relevant dependency versions in the reuse identity. A changed fingerprint
requires a fresh output directory. Completed fits with valid completion records
can be reused. Interrupted fits restart training. Historical snapshots retain
the recovery behavior of the source version saved with the run.

The initial DBD archives contained object-string ID arrays. A recorded serialization migration converted only IDs to Unicode and verified unchanged numeric predictions; original archives were retained under `.cache/serialization-backups/`. `runs/asymmetric-v1/serialization-migration.json` records hashes. This is separate from a model or metric change.

The current future-rating adapter stores snapshots under `snapshot-cache/<identity>/` within the reference data directory. Its identity includes both reference examples and rosters, raw performance data, the builder and snapshot times. Older `snapshots.npz` and its manifest remain untouched. A roster/duration update cannot reuse a stale feature cache, and invalid fractional, negative or out-of-range match IDs fail before indexing. For the existing research inputs, `audit_tuning_data.py` verifies every old/new array is identical before accepting the strengthened provenance alongside frozen manifests; it does not silently replace their identity. The current rebuild and original cache also have identical archive hashes.

## Independent metric audit

```bash
python scripts/audit_results.py runs/asymmetric-v1 runs/skill-v1 --output reports/audit.json
python scripts/audit_decisions.py runs/decisions-v1
python scripts/audit_splits.py runs/asymmetric-v1 runs/skill-v1
python scripts/audit_calibration.py --run runs/hero-calibration-v1
python scripts/diagnose_calibration.py
python scripts/verify_neuralac_reference.py
python scripts/analyze_experiments.py
```

The first command recomputes NLL/accuracy/Brier or regression MAE/RMSE/bias directly from stored predictions, checks row identity uniqueness, and verifies frozen source fingerprints. It does not prove causal validity or audit every possible preprocessing mistake. The split auditor independently checks saved group/time partitions, availability boundaries and the membership of every completed prediction file. The reference comparison needs the separately downloaded author repository; it is not bundled in the distribution. The decision auditor independently reconstructs the pair-sum oracle and every saved assignment metric. Primary lobby pools have zero overlap with any factual partition; the same five seed values are shared across additive/cooperative mechanisms.

The calibration auditor requires the completed follow-up. It checks original validation membership, unchanged primary checkpoint and prediction hashes, unchanged test identities/labels, the saved validation objective and approximate bounded first-order optimality, then independently replays each saved slope/intercept. It does not independently refit the underlying validation predictor. The metric auditor rejects nonfinite values, invalid probability ranges and misaligned shapes before recomputation; deliberately corrupt artifacts exercise these failure paths.

The calibration diagnostic reads all 105 existing affine fits, retaining each model and seed, and records parameter ranges, mean probabilities and unchanged-ranking checks. It does not refit or identify a causal side effect. Marginal probability agreement is not sufficient for calibration; a constant predictor's slope and intercept are not separately identifiable.

`analyze_experiments.py` is a workspace report script with explicit run names. It marks incomplete seed sets and excludes them from paired comparisons. Summary tables use mean and sample standard deviation across seeds, never the NLL of an unreported probability ensemble. Deterministic models may have zero seed variation; duplicated runs are not independent model evidence.

## Uncertainty and interpretation

Paired comparisons align exact row IDs and labels. The bootstrap resamples paired training seeds, then evaluation clusters, computing a row-weighted mean loss difference. Hero comparisons cluster canonical roster groups; Overwatch uses calendar days; DBD reports day and predator-account sensitivities; future-rating results report account and match sensitivities. These dimensions address different dependencies and do not jointly remove player dependence, shared matches and time drift. Eighteen Overwatch test days and less than one day of Dota future-rating coverage limit generalization.

Intervals are exploratory, marginal percentile intervals with no multiple-comparison adjustment. They should accompany effect size and assumptions, not select a leaderboard winner. A small neural improvement whose interval crosses zero is reported as unresolved under this experiment. Distinct datasets and outcome spaces are not pooled into one universal score.

These intervals condition on the recorded training/validation samples and fitted predictors. Models and calibrators are not refitted inside bootstrap replicates. Resampling optimization seeds and evaluation clusters does not quantify uncertainty over alternative training datasets or calibration histories. Where validation is used for both checkpoint selection and calibration, that shared validation use is part of the recorded protocol rather than an independent calibration holdout.

Synthetic decision results use five independently drawn hidden worlds per mechanism. World seeds also control training and labels; 500 lobbies per world do not become 2,500 independent world replicates. Raw, temperature and offset transformations are diagnostics, not tuned methods. All candidates are scored once for the factorial experiment; candidate-score budgets are search comparisons, not measured production latency.

## Quality checks

Run the test suite and static checks as described in [Building and verifying a distribution](release-readiness.md). Any claimed implementation speedup must use the same inputs, model semantics, numeric precision and an explicitly reported measurement setup. Concurrent training jobs include contention in their durations, and device probes are microbenchmarks rather than production latency measurements.

Reuse saved checkpoints with the dependency versions recorded in their manifests. Tests of supported installations and tests of cross-version artifact migration are different claims.

## Research exports

Install the optional `research` dependencies to export figures and complete-seed tables. `scripts/export_research_artifacts.py` consumes `reports/analysis/results.json` and records the input hash. Run `analyze_cohorts.py`, `analyze_calibration_comparisons.py` and `analyze_experiments.py` first. The calibration comparison requires a completed calibration manifest; its 36 comparisons were specified after inspecting calibration summaries and are labeled exploratory, not an additional model-training benchmark.

`scripts/benchmark_neuralac_updates.py --device mps --previous PATH --output PATH` provides a separate old/new local update-loop benchmark with dropout, BCE loss and Adam. It uses a fixed synthetic input bank and five randomized paired blocks of 100 timed updates, synchronizing at block boundaries. It excludes loading, validation and checkpoints, does not compare the author implementation, and does not promise matching stochastic trajectories. These engineering update blocks are separate from model-quality experiment counts.
