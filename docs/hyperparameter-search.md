# Validation-only hyperparameter search

Run the bounded local matrix with:

```bash
.venv/bin/python scripts/run_local.py --python -m gamematch.tuning.search --workspace . --output runs/hpo-new
```

Use `--datasets uci-dota2` or `--models mlp deepsets` to run subsets of the same frozen matrix. Each output root freezes all candidate configurations, source and runtime identities before training. Resume requires identical identities; changed spaces require a new output root.

This research driver expects the prepared datasets and reference-run split artifacts described in the reproduction guide. For CPU execution on Linux or another host, explicitly select the device:

```bash
python -m gamematch.tuning.search --workspace . --output runs/hpo-cpu --device cpu
```

`--device cpu`, `--device mps` or `--device cuda` overrides neural candidates; tabular candidates remain CPU estimators with unchanged configurations. An unavailable requested backend fails before creating the run. Omitting the option preserves the historical local plan, which mixes CPU and Apple MPS. The explicit choice is part of the frozen protocol and must remain the same on resume. When splitting neural and tabular jobs into separate output roots, pass the same device declaration to both so their complete matrix protocols agree. Backend availability is checked; individual PyTorch operator support still depends on the chosen model and installed version.

The matrix covers three hero datasets, both DBD periods, Overwatch, future observed rating, and two synthetic mechanisms. It includes eight configurations per model and dataset, with one original-style configuration and seven fixed alternatives. Neural candidates vary activation dropout (0, .1, .2, .35, .5), learning rate (.0003–.003), weight decay (0–.01), Adam/AdamW, plateau scheduling, width, embedding dimension, depth where supported, and batch size. Regression also varies MSE, Huber and MAE training losses. FM has no activation dropout. Tabular comparators receive eight configurations for logistic C, ridge alpha, or LightGBM leaf count, learning rate, leaf support and L2 regularization.

All candidates use full primary training data. Epoch caps are 60 for DBD/rating, 30 for Overwatch, 80 for ordinary hero networks, 150 for real-data NeuralAC and 200 for synthetic NeuralAC; early stopping uses eight non-improving validation epochs. These are bounded compute budgets, not a claim of convergence or exhaustive optimization.

Stage one uses seed 100. The top three neural configurations are rerun with seed 101; selection minimizes their two-seed mean validation score. Deterministic real-data tabular estimators use one selection seed. Final neural configurations are independently trained with seeds 0–4; deterministic real-data tabular estimators are fitted once. Synthetic screening uses world 0, confirmation uses world 1, and final evaluation retains worlds 0–4. Validation labels select synthetic scorers; hidden oracle regret never selects a configuration.

The fitter accepts only train and validation partitions. Classification checkpoints and configurations minimize raw validation NLL. Future-rating checkpoints and configurations minimize validation MAE in original rating units; this differs from the old MSE checkpoint criterion and is reported as part of the new protocol. A hashed `selection.json` is persisted before any final test evaluation. DBD and Overwatch temperature calibration uses validation only after selection, matching the previous probability evaluation convention. Reusing validation for stopping, search and calibration can itself overfit; independent future confirmation is still needed.

Primary partition IDs/targets are audited, train-only transforms are retained, and every final checkpoint is reloaded and compared on validation predictions. Search artifacts contain validation predictions and no test predictions. Final artifacts include all metrics, predictions, training curves, selected configuration and source identity. Original runs and their recorded results remain unchanged.

New runs use schema v2. Data identities include input arrays, split metadata and the fitted preprocessor; existing identity files are verified without rewriting them. Fit records hash checkpoints and saved validation predictions, and neural records also hash training curves. Resume checks model/configuration/seed identities and recomputes validation scores before replaying selection. A locked selection binds the complete candidate/seed inventory. Completed jobs still verify all final seeds, predictions, calibration and metrics; normal completed resume does not repeat training or neural inference. Hashes detect accidental drift within a trusted local archive; they are not signatures proving external provenance. The separate checkpoint replay audit supplies an independent inference check.

Categorical targets must be integer arrays in the declared class range, binary targets must be zero or one, and regression targets and scaling must be finite. The generic history tuning task currently requires three loss/draw/win classes. The optional binary LoL history experiment uses its dedicated runner and explicit two-class head, described in [richer-history.md](richer-history.md).

This is an exploratory follow-up: original test scores had already been seen before the search was designed. The new test results are not an untouched confirmatory experiment, and no protocol change is made to force a deep network to outperform a simpler model.

## Archived first round and specialist follow-up

The already-started `hpo-v1` and `hpo-v1-baselines` roots retain an earlier frozen implementation. Resume those experiments by placing their `source-snapshot` on `PYTHONPATH`, not by importing the evolving editable checkout. Those archives are not silently upgraded to schema v2. They do not contain the later completed-job data and artifact checks; independent current audits replay validation choices and compare retained split identities. Newly created runs use the stronger checks above.

The specialist driver waits for the five first-round Overwatch models, then records its own new source/driver snapshot. It gives each model 16 four-epoch candidates, six twelve-epoch candidates, and three 48-epoch candidates repeated on seeds 200/201, followed by five final fits. Every rung starts training from its declared seed rather than continuing the shorter checkpoint. The first-round selected configuration is retained through promotion. Trial counts are equal; parameter counts, optimizer steps and GPU seconds are not. Random dimensions include dropout, optimizer, weight decay, learning rate, width, heads, layers, batch size and normalization; declared large trials use width 128/256, eight heads and three layers. Deep Sets ignores attention-only dimensions.

The `pooling="team"` attention variant keeps four ordered team/view means instead of one global mean; DAWN keeps its domain token separate. This is a new adaptation for unordered teams, not an exact CUPID/CHAMP reproduction. The other choice retains the reference global readout. No industrial economy-weighted loss can be reconstructed from the public outcome-only collection. Promotions and choices use raw validation NLL; final temperature-calibrated NLL is reported alongside raw NLL.

To resume an archived specialist driver after checkout changes, set `GAMEMATCH_WORKSPACE` to the original workspace and `PYTHONPATH` to the specialist `source-snapshot`, and execute its `source-driver.py`. Outputs reject mismatched source, driver or dependency identities. Data and library dependencies must still be available locally.

`scripts/audit_tuning_integrity.py` verifies the frozen source, re-scores archived validation predictions, replays shortlists/promotions and checks each locked choice against final configurations. `export_tuning_results.py` and `export_specialist_results.py` require complete declared seed sets before adding a model to their aggregate results. Concurrent wall times are cost records, not isolated throughput benchmarks.

The broad exporter also writes `progress.md` and `partial-results.json` for tasks whose entire searched model set has finished, while other tasks remain pending. These files are explicitly incomplete and do not satisfy the final-delivery gate. `verify_tuned_predictors.py` independently restores completed final checkpoints on CPU, checks their original configuration identities, and compares up to 64 saved validation and test predictions using batch size 37. Only inference device and batching change; calibration parameters are reused, not refitted. Its report must be refreshed after the remaining fits complete.
