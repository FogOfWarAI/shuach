# Research API reference

Use this reference when implementing a data adapter, changing a network, comparing saved predictions or running a validation search. For a first complete example, start with the [Python API guide](api.md).

Shapes use `B` for batch size, `P` for players per side, `T` for history length and `V` for vocabulary size.

| Task | Jump to |
| --- | --- |
| Prepare a roster or build time-ordered player histories | [Static records](#static-records-gamematchdataheroes) · [Team events and history](#team-events-and-history) |
| Choose a network and understand its output | [Outcome and sequence models](#outcome-and-sequence-models) |
| Calculate metrics or aggregate a saved run | [Evaluation](#evaluation) · [Experiment runners](#experiment-runners) |
| Fit or reload a candidate for model selection | [Hyperparameter search](#exploratory-hyperparameter-search) |

## Public files: `gamematch.data.acquisition`

`REGISTRY` describes seven supported collection keys, source URLs, separate data licenses, exact file sizes and SHA-256 values. `fetch_dataset(name, data_root)` downloads a missing asset, verifies cached content, and extracts only required members into the adapter's layout. It returns and saves a provenance record. Changed existing files, webpages, oversized responses, failed hashes and escaping destinations raise errors; no author code is executed. This is an explicit acquisition call, never an automatic import-time download. See [getting started](getting-started.md) for CLI usage and data terms.

## Static records: `gamematch.data.heroes`

`HeroRecords(rosters, labels, row_ids, groups)` stores integer rosters `[N,2,5]`, binary labels `[N]`, unique row IDs and canonical roster groups. `.take(indices)` returns a corresponding subset; `len(records)` counts matches.

`HeroVocabulary().fit(training_rosters)` learns observed identifiers. `.transform(rosters)` maps known identifiers to `1..V-1`, unseen identifiers to `0`; `.size` includes that unknown token. Model-specific unknown-token behavior must be checked before cold-start use; the benchmark reports unknown counts and is not a verified new-hero evaluation.

`prepare(dataset, data_root, protocol='pilot', split_seed=0, max_train_rows=None, max_eval_rows=None)` returns `(splits, vocabulary, manifest)`. Supported dataset keys are `uci-dota2`, `neuralac-dota2018`, and `neuralac-lol`. `splits` has `train`, `valid`, `test` `HeroRecords`. Final UCI evaluation uses the official test file and purges intersecting train/validation roster groups. `pilot` uses an inner holdout and never opens that official test file. Row caps are development options and must be disclosed.

`group_ids(rosters)` canonicalizes within-team permutations and side mirrors for split grouping only. Side order remains unchanged in model inputs. `file_hash(path)` computes streaming SHA-256.

## Team events and history

`gamematch.data.events.MatchEvent(row_id, time, available_at, teams, outcome, domain, duration=None)` is an immutable event. Teams are two nonempty disjoint tuples of unique participants. Outcome values are `0=left loss`, `1=draw`, `2=left win`. Time and availability are finite seconds, availability cannot precede prediction, and an observed duration must be positive. The dataclass does not authenticate token identity or timestamp meaning.

`load_overwatch(path, feedback_delay=0)` returns `(events, manifest)`, filtering duplicate, incomplete and repeated-player records. It assumes reported time is a start proxy and sets availability to `time + duration + delay`. `temporal_boundaries(events, train_fraction=.7, valid_fraction=.15)` returns ordered boundaries and rejects collapsed partitions.

`gamematch.data.history.build_histories(events, history_length=10, same_domain_only=False)` requires sorted, equally sized teams and binary competitive domains. It returns `HistoryData` with `history [N,2,P,T,5]`, `static [N,2,P,13]`, boolean `valid [N,2,P,T]`, labels, domains, row IDs and times. Histories are right-aligned; masked entries are zero. It updates a previous outcome only if `available_at < current_time`.

`history_split_masks(data, train_fraction=.7, valid_fraction=.15)` returns boolean masks and boundaries, purging unfinished train/validation feedback. `standardize_histories(data, train)` mutates features using the training mask and returns the fitted normalizer. `HistoryData.save(path)` and `.load(path)` use non-object NumPy archives.

`gamematch.data.asymmetric.load_asymmetric(dataset, root, feedback_delay=0)` loads `dbd-2022` or `dbd-2023` and returns typed arrays, pre-event features and provenance. `gamematch.data.skill.load_skill(workspace)` returns `(SkillData, match_metadata)` from verified derived Dota files; `skill_splits(data, match_metadata)` additionally purges future targets at temporal boundaries. See [data contracts](data-contracts.md) for target meaning.

## Outcome and sequence models

`gamematch.models.make_model(name, vocab_size, embedding_dim=20, hidden_dim=64, dropout=None, interaction_hidden=50)` returns a PyTorch module for `linear`, `signed_hero`, `fm`, `mlp`, `deepsets`, `set_transformer`, `neuralac` or its named ablations. Input is a `LongTensor [B,2,5]`; output is binary logits `[B]`. Apply sigmoid to obtain probabilities. Tabular `prior`, `logistic` and `lightgbm` are runner choices, not neural factory names. `dropout=None` retains the reference defaults (0.2 for NeuralAC, zero for MLP/Deep Sets); an explicit value configures activation dropout. The tuning engine supplies the selected value; the original reference runner retains its established defaults.

`NeuralAC(vocab_size, embedding_dim=20, interaction_hidden=50, dropout=.2, cooperation=True, competition=True, attention=True, attention_bias=False)` accepts 5v5 only. Disabling a component freezes its unused parameters. `attention_bias=False` follows the paper; the author-bias variant follows that detail of the released implementation. A full comparison still depends on data splits and training settings.

`gamematch.models.omninet.OmniEncoder(width=32, heads=4, layers=2, dropout=.1, omni=True)` maps `[B,T,width]` to the same shape; `valid [B,T]` masks padding. `omni=False` disables cross-layer meta-attention. Empty masked sequences return zero representations.

`OwO(history_features, static_features, team_size=5, history_length=10, width=32, heads=4, layers=2, dropout=.1, classes=2, domains=0, domain_features=0, context_tokens=2, ordered_slots=True, omni=True, pooling="global", norm_first=False)` accepts history/static/mask tensors and optional `domain [B]` plus continuous domain attributes. It returns class logits `[B,classes]`. Positive `domains` enables DAWN-style domain context. Ordered slots preserve position sensitivity. Unordered `pooling="global"` averages all player/view tokens; `pooling="team"` separately averages players within each side/view and retains the four ordered means. Team pooling requires unordered slots and is an explicit adaptation. `norm_first` configures pre- versus post-normalization. `make_history_model(name, history_features, static_features, team_size, steps, config)` selects `static_deepsets`, `history_deepsets`, `owo_transformer`, `owo` or `dawn` with the declared three-class public-data contract.

For the two history Deep Sets variants, set `config["deepsets_dropout"]` explicitly. Its default is zero to preserve the original reference runner; the attention models instead use `config["dropout"]`. The HPO engine maps its selected `dropout` to the Deep Sets key. Passing only the attention-model key directly to the factory does not enable Deep Sets dropout.

`MMRNet(features, hero_vocab, steps=12, width=160, heads=10, layers=6, dropout=.1, omni=True)` consumes snapshots `[B,steps,features]`, lineup IDs `[B,2,P]` and valid mask `[B,steps]`. Output is a scalar `[B]`; the experiment runner interprets it as a standardized future-rating residual. `make_skill_model(name, features, hero_vocab, config)` chooses `mlp`, `gru`, `transformer` or `mmrnet`; baseline and original-paper defaults are not silently interchangeable.

`gamematch.models.ordinal.OrdinalMLP(features, hidden=64, classes=5, dropout=0.0)` provides a monotone cumulative-link outcome head and returns normalized log probabilities. The asymmetric runner converts them to five class probabilities.

## Evaluation

`binary_metrics(labels, probabilities, bins=15)` requires binary labels and one finite probability in `[0,1]` per row. It returns NLL, Brier, accuracy, AUC (or `None` for one-class labels), prevalence and equal-width ECE. `multiclass_metrics(labels, probabilities)` requires integer labels and a normalized `[N,C]` simplex; it returns NLL, accuracy, summed multiclass Brier and class counts. These proper scores use documented numerical clipping; do not compare their scales across unrelated outcome spaces.

Both functions also return `probability_mae` and `probability_rmse`. Binary errors use the positive-class probability against a 0/1 target; categorical errors average over all `N*C` one-hot components. Thus binary RMSE squared equals binary Brier, whereas categorical RMSE squared equals summed categorical Brier divided by `C`. Probability MAE is descriptive, not a proper score for selecting probability forecasts. These errors differ from ordinal expected-count error or an expected win/draw/loss score error. Continuous rating regression does not have LogLoss without a declared predictive density.

`evaluation.comparison.load_predictions(path)` loads a non-object archive, sorts unique row IDs and rejects duplicates. `paired_losses(first_paths, second_paths, metric='log_loss')` aligns rows, labels and supplied identity/cluster/time metadata both within model pairs and across seeds, returning `[seeds,examples]` differences and a reference record. The same row identifiers with changed labels or cluster membership are rejected. Regression metrics are `absolute_error` and `squared_error`. `paired_cluster_bootstrap(differences, clusters, replicates=2000, seed=0, alpha=.05, metric='log_loss')` resamples paired seeds and evaluation clusters, reporting the per-example mean difference and a percentile interval. Cluster dimensions are sensitivity choices; independence is an assumption, not inferred from IDs.

`evaluation.calibration.fit_binary_calibration(logits, validation_labels, mode="temperature")` fits positive-slope binary calibration by convex validation NLL minimization. `mode="affine"` also fits an intercept. Both require finite logits and both binary validation classes; fitted bounds and validation loss are recorded. `apply_binary_calibration(logits, parameters)` uses the saved transform without labels. Temperature preserves the zero-logit boundary; both preserve ranking apart from floating-point saturation. Callers must keep calibration labels separate from the final test.

## Experiment runners

Each entry accepts three paths. `config_path` names the JSON file, `workspace`
is the base for adapter data and caches, and `output` is the destination run
directory. The hero runner returns the absolute path to `report.md`; the other
five return the path to `results.json` in the run directory. A changed run
fingerprint raises `ValueError` on reuse.

Module/function | Configuration protocol | Task
--- | --- | ---
`runner.run_benchmark` | `pilot` or `final` | Static hero outcomes
`rating_runner.run_ratings` | `temporal_delayed_feedback` | Online ratings plus WDL calibration
`history_runner.run_history` | `temporal_completed_player_history` | Pre-event player history networks
`asymmetric_runner.run_asymmetric` | `temporal_predator_outcome` | Five predator outcomes
`skill_runner.run_skill` | `temporal_future_observed_rating` | Future rating-state regression
`decision_runner.run_decisions` | `synthetic_independent_decisions` | Known-mechanism prediction and decisions

`runner.TrainConfig` describes hero/synthetic optimizer and device settings. Task-specific JSON fields are illustrated in `benchmarks/`. `reporting.create_run_report(run_directory, output=None)` summarizes all six task protocols into a dedicated directory (default `RUN/report`) without changing manifests, results or model artifacts. Only complete declared seed sets receive aggregate means; synthetic solver diagnostics remain separate from trained-model jobs. It reads saved metrics rather than independently recomputing them. `runner.render_report` is a legacy hero-only helper and rejects other protocols before writing. The current runners are separate task entry points, not a universal arbitrary-dataset trainer.

## Exploratory hyperparameter search

`python -m gamematch.tuning.search --workspace WORKSPACE --output NEW_RUN` runs the bounded research matrix described in [the search protocol](hyperparameter-search.md). It requires the declared local data and primary split/reference artifacts; it is not an automatic dataset downloader. Optional `--datasets` and `--models` select subsets. The output freezes candidate spaces, source and environment identity. Use a fresh output for changed code or settings; resume archived experiments with their own source snapshot.

`gamematch.tuning.engine.fit(task, name, spec, train, valid, cfg, seed, directory)` accepts explicit training and validation `Partition` objects. It saves a validation checkpoint, predictions and curves. `reload_model(directory, task, name, spec, cfg)` restores that checkpoint. `predict_raw(model, part, task, cfg)` expects an **already transformed** partition. Full search calls `evaluate(model, bundle, cfg, directory)` after locking a validation-selected configuration. The specialist driver and report scripts in the source archive operate on the research-artifact layout.
