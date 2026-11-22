# Model implementation cards

All experiment conclusions are limited to the supplied inputs, declared training/search budgets and splits. The table below describes the retained fixed-configuration reference; subsequent validation-only search is documented separately. Architecture adaptation, numerical equivalence and reproduction of an original paper's results are separate claims.

OwO and DAWN are outcome predictors. The distinction below describes their input
contracts, not exclusion from a general outcome-prediction task. The executable
[model catalog and compatibility API](outcome-models.md) distinguish architecture
support from available dataset runners and completed empirical evidence.

Family | Implementation and input | Selection/calibration | Scope
--- | --- | --- | ---
Prior | Training outcome frequencies; domain-aware calibration baseline for Overwatch | Validation calibration where declared | Required difficulty/selection reference
Linear | PyTorch linear logit from two-side hero counts | Validation stopping | Separate implementation entry from sklearn Logistic, not a new algorithm
Signed Hero | One weight per hero shared across sides, plus a side intercept | Validation NLL checkpoint | An additive, interpretable roster baseline |
Logistic regression | sklearn, two-side hero counts or pre-event DBD features | Fixed C, convergence checked | Convex reference; repeated seeds are deterministic
LightGBM | Library adapter, same task features | Validation early stopping and fixed tree budget | Not a deep model; CPU/OpenMP
FM / MLP / Deep Sets | Independent PyTorch implementations | Validation NLL checkpoint | Same hero information; within-team invariant
Set Transformer | Shared per-team self-attention and learned-query pooling | Validation NLL checkpoint | Compact two-roster adaptation of attention over unordered sets |
NeuralAC | Individual, cooperation, competition and bilinear attention | 50-epoch cap in primary hero run | Paper-equation implementation with independent numerical author-code check
Elo / TrueSkill / OpenSkill | Mean-team Elo and the installed rating libraries | Delayed replay with validation calibration | Raw score and calibrated loss/draw/win probabilities |
Static/history Deep Sets | Player static features, optionally mean pooled prior sequence | Validation CE then scalar temperature | Common information baseline for OwO/DAWN
OwO / DAWN | Temporal, scalar-feature spatial and player-permutation encoders; DAWN context injection | Same public history inputs and training budget | Explicit adaptation, not original CUPID/CHAMP deployment
Ordinal MLP | Learned monotone cumulative-logit thresholds | Validation NLL and temperature | Ordered 0–4 DBD counts preserved
Current / mean-delta / ridge | Future-rating baselines with explicit current-state reference | Train-fitted mean/ridge | Distinguish predicting future state from rediscovering current state
Snapshot MLP / GRU / Transformer / MMR-Net | Completed-game feature snapshots and hero lineups | Validation MSE; predict residual around current mean | Public observed-rating proxy, not original QuickSkill novice labels

The three OpenSkill IDs choose Plackett–Luce, Bradley–Terry full or Thurstone–Mosteller full. For architecture details, sources and commands, see the [roster and rating guide](model-additions.md).

## NeuralAC fidelity and ablations

The independent implementation follows [Gu et al., AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16528). The downloaded [author code](https://github.com/bigdata-ustc/NAC) is retained separately and excluded from the package. `scripts/verify_neuralac_reference.py` maps shared weights, zeros the two author attention biases, disables dropout, and compares probabilities plus all shared parameter gradients for ten cases (five seeds, attention enabled/disabled). The checked CPU float64 errors are near machine precision. This does not imply identical dropout streams or published benchmark reproduction.

The main implementation defaults to bias-free bilinear attention as written in the paper; `neuralac_author_bias` enables the released code's biases. `neuralac_no_cooperation`, `neuralac_no_competition` and `neuralac_no_attention` expose component sensitivity. They are evaluated with unchanged optimizer/split settings on UCI in an explicitly exploratory run specified after primary test scores were available. All variants are retained; no test-based winner is selected as a revised primary model. Group splits, batch size and validation stopping differ from the authors' original protocol.

<!-- bib: neuralac -->

## OwO and DAWN adaptations

Sources are [CUPID](https://doi.org/10.1145/3686978), [CHAMP](https://arxiv.org/abs/2609.04870), and their cited OmniNet design. `OmniEncoder` runs Transformer layers, attends over their stacked tokens, concatenates per-token depth outputs, projects them back to width and adds a normalized residual. This explicit dense reduction resolves an implementation detail not fully specified in the matchmaking papers; it is not the authors' released network code or a claim to all OmniNet variants.

The public run uses width 32, four heads, two layers, dropout .1 and ten completed prior outcomes per player. Temporal features contain outcome, duration, mode, recency and target-mode agreement. Static features contain global/per-mode counts, smoothed outcomes, recency and target mode. All history models receive target mode, including OwO and Deep Sets; DAWN adds learned domain context inside the encoders. `owo_transformer` removes cross-layer meta-attention at the same depth and width, not equal parameter count.

The original MOBA designs model position-sensitive 5v5 slots and richer industrial observations. The Overwatch adapter has no role slots, so it uses 6v6 side/view embeddings and pooled output with within-team permutation invariance. Its task is three-class loss/draw/win. Industrial KDA, economy, party and rank-tier fields are unavailable. Results on this input cannot establish that DAWN's original cross-mode system is ineffective or superior.

The specialist follow-up additionally declares `pooling="team"` and `norm_first=True/False` choices. Team pooling averages players within each side and temporal/spatial view, preserving four ordered representations; DAWN's domain token remains separate. This variant is tested for within-team permutation invariance and team-assignment sensitivity. It is a separately named structural adaptation, so a winning variant must not be reported as the unchanged reference OwO/DAWN. The [search protocol](hyperparameter-search.md) records budgets and exploratory limitations.

<!-- bib: cupid champ omninet -->

## MMR-Net adaptation

[QuickSkill](https://doi.org/10.1145/3511808.3557070) motivates predicting future rating from sequential early-game performance. The public implementation uses twelve three-minute snapshots and six continuous features, separate lineup embedding and temporal positions. A dense projection replaces industrial discretized feature embeddings. The reference run uses width 32, four heads and two layers; the class exposes larger original-style defaults separately.

The model predicts a standardized residual relative to the current observed rating. Baseline models receive the same snapshots and lineups where applicable; current-rating persistence is always included. The target is rebuilt OpenSkill state after five additional observations, with temporal purging. It is neither actual MMR nor a verified novice cohort. The small, selected test cohort makes uncertainty and scope more important than a small raw MAE difference.

<!-- bib: quickskill -->

## Conditional and unavailable reproductions

OptMatch requires verified player-character histories and an implementation that can be mapped to the original method. The supplied PTOM anonymous repository previously returned an access error. Teammate autoencoder variants remain candidates for future work.

<!-- bib: optmatch ptom -->

[Download the model bibliography](models.bib).
