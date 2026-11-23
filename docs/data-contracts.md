# Data and evidence contracts

An adapter declares what a record identifies, when its fields become available, and which outcome is observed. A dataset being readable is not enough to make every model or decision task valid.

## Task inputs and prediction targets

This table describes implemented inputs, rather than every column available in a downloaded file. `B` is batch size. The executable version of this table is the [unified feature contract](feature-schema.md). Seven real collections map to five prediction tracks; synthetic prediction and assignment evaluation are additional experiment stages.

Task / collections | Prediction time | Model input | Target / output
--- | --- | --- | ---
Hero outcome: UCI Dota2, NeuralAC Dota2018 and LoL | Both hero rosters are known; these exports lack event times | Two ordered sides of five hero IDs, shape `[B,2,5]`; tabular/FM/MLP models use two side-count vectors, embedding models use the IDs | Binary left-side win label and one predicted win probability
Online player ratings: Overwatch | Before the next recorded match, after strictly earlier feedback | Two six-player rosters index stored Elo/TrueSkill/OpenSkill states; previously completed results update the states. WDL calibration uses rating score `s`, `abs(s)`, target mode `d`, and `s*d` | Left-side loss/draw/win probabilities, preserving all three classes
Player-history outcomes: Overwatch | Same pre-event availability contract | Per player: ten completed history steps with five fields, thirteen static summaries, validity mask; target mode. Shapes `[B,2,6,10,5]`, `[B,2,6,13]`, `[B,2,6,10]` | Three-class loss/draw/win probabilities for the observed teams
Asymmetric predator outcome: DBD 2022/2023 | Map and predator avatar are known, before the match | Two categorical map/avatar fields plus nine completed-history numeric features. Train-fitted one-hot/scaling gives 51/66 columns in the primary respective splits | Number of prey eliminated, `0..4`; five probabilities and derived expected count
Future observed rating: Dota2 Matches v2 | After the focal match completes | Twelve three-minute snapshots, six features per step, valid-step mask, current rating and the two five-hero lineups; focal player's team first | OpenSkill posterior mean after five additional observed completed games; network predicts a standardized residual around current mean
Synthetic outcome learning: additive/cooperative worlds | Proposed two-side roster is known | Two five-participant ID sets, `[B,2,5]`, represented by the same count/embedding interfaces as hero models | Sampled Bernoulli win label; model estimates left-side win probability
Synthetic assignment evaluation | Before choosing teams from an unseen lobby | Ten participant IDs, feasible team sizes/constraints and a frozen scorer; 252 ordered 5v5 candidates for exhaustive search | A chosen assignment minimizing `abs(predicted_p-.5)`; independent hidden probabilities supply true imbalance and decision regret, not training labels for the solver

### Exact Overwatch features

An opt-in [richer-history extension](richer-history.md) adds delivered-feedback
rating context, rolling summaries and observed pair familiarity (8 sequence /
46 static channels). It is currently an experimental feature probe; the frozen
primary and first-round search results below retain the original 5/13 fields.

Each history step contains: the player's earlier result score (loss/draw/win = `0/.5/1`), `log1p` earlier duration in minutes, earlier mode, `log1p` days since that feedback became available, and whether the earlier mode equals the target mode. Histories are right-aligned and missing steps are masked.

The thirteen static fields are four summaries for each of three scopes (all modes, noncompetitive, competitive), plus target mode. The four summaries are `log1p` completed count, smoothed mean result score, smoothed draw frequency, and `log1p` feedback recency. The legacy field name `smoothed_win_fraction` **includes half credit for draws**: `(wins + .5*draws + 1)/(n+2)`. It is not a strict count of wins divided by matches.

Static Deep Sets uses these static fields and history coverage; its five history-average channels are zero. History Deep Sets also uses the mean of the five history channels. OwO/Transformer/DAWN use the ordered valid sequences and static feature tokens; DAWN additionally injects the target-mode representation inside its encoders. Player IDs organize historical lookup and split diagnostics, rather than acting as learned player-ID embeddings in these networks. Current-match duration, result, KDA, gold, role/party and rank-tier data are not supplied as predictors.

### Exact DBD features

Categorical inputs are `environment_id` and `avatar_id`. The nine numeric inputs are `log1p_completed`, smoothed `mean_previous_kills`, `last_previous_kills`, `log1p_days_since_feedback`, and five smoothed frequencies `previous_kill_fraction_0` through `_4`. The predator account identifies which completed history to retrieve; it is not one-hot encoded as an input. Unobserved history uses declared smoothing/prior values. Current-match behavior and the source's full-period `total_xp` are excluded. Four prey identities are unavailable.

### Exact future-rating features and target

Snapshots at minutes 3, 6, ..., 36 contain `log1p(gold)`, `log1p(last hits)`, `log1p(experience)`, current OpenSkill mean, current OpenSkill uncertainty, and `log1p(observed match count)`. The last three quantities are repeated across valid snapshot positions and are available after the focal match's rating update. End-of-match masking removes later snapshot positions. Shapes are `[B,12,6]`, `[B,12]`, and hero IDs `[B,2,5]`; the current mean is also the residual reference. Current-state persistence uses only that mean, and mean-delta adds a train-fitted constant.

Let the current posterior mean be `mu_t` and the mean after five additional observed games be `mu_future`. Training uses `z = (mu_future - mu_t - train_delta_mean)/train_delta_sd`, and prediction restores `mu_hat = mu_t + train_delta_mean + train_delta_sd*z_hat`. This is a post-game forecast of an algorithmic state; the current game's economic snapshots would not be valid pre-match inputs. Account/match IDs and availability timestamps support construction, grouping and purging, not identity embeddings. This target is not true MMR, a verified novice label or the outcome of a newly reassigned game.

## Identities and outcomes

Track | Entity | Observed outcome | Supported use
--- | --- | --- | ---
Hero rosters | Hero/character, not person | Left-side binary win | Static outcome prediction
Overwatch events | Anonymized player token | Left loss/draw/win | Ratings and pre-event history prediction
DBD predator | Predator account and avatar/map | 0–4 prey eliminated | Ordinal/multiclass outcome prediction
Dota snapshots | Nonzero encoded account, current hero lineup | Later observed OpenSkill state | Post-game future-state regression
Synthetic worlds | Generated persistent participant | Sampled Bernoulli result and hidden probability | Independent assignment evaluation

Public hero exports contain no persistent player identity. DBD does not identify the four prey. Neither supports reconstructing a human matchmaking queue. Token stability in Overwatch is not independently authenticated. In Dota Matches, account ID `0` is anonymous and must never become a single global player. The reference builder gives each anonymous occurrence a fresh state and excludes it as a target account.

## Time and splitting

All timestamps used internally are UTC seconds. Event replay uses a strict `<` availability condition: feedback tied with the next event's start is not available to that prediction. Overlapping matches remain pending. The neural history adapter masks missing history rather than reading a current outcome.

The Overwatch export has inconsistent prose metadata: actual retained records are 6v6 and include draws. The adapter follows inspected fields, records every exclusion, and preserves draws. Its reported `time` is treated as a start proxy; adding recorded duration and an optional delay is an explicit assumption. A 24-hour delayed-rating sensitivity is included, but cannot verify the original timestamp definition.

DBD timestamps are starts; duration determines when a previous outcome can enter history. Current-game speed, chase, damage, score and duration are excluded as predictors. The provided `total_xp` aggregates the observation period, so it is not a safe pre-match feature. The adapter rebuilds completed counts, frequencies and recency instead.

Hero splits group identical lineups, within-team permutations and side mirrors together. UCI keeps the official test set and removes train/validation group overlap. This controls roster memorization, not time shift: hero records lack event timestamps, and their human players may recur across partitions. Test vocabulary is not fitted in advance. Temporal tasks instead partition by chronological cutoffs, purging training and validation labels that have not become available by the next boundary. Continuing players may appear in multiple time partitions by design.

## Dota future-rating proxy

The downloaded `player_ratings.csv` is not used. The supposedly earlier history overlaps the first 360 main matches. The reference builder uses only historical starts at least 24 hours before the earliest main start; missing historical durations make this embargo an assumption. Seven historical matches with inconsistent paired outcomes are rejected. Main matches are processed by completion time. Unknown/duplicate heroes and invalid rosters are filtered.

The rebuilt rating is a default OpenSkill Plackett–Luce posterior, not true ability or the game's MMR. Inputs become usable after a match completes: twelve three-minute snapshots of gold, last hits and experience (log transformed), current reference mean/uncertainty, observed match count, and two hero lineups. Snapshots later than the recorded match end are masked. The target is that account's reference mean after five additional observed matches; target availability is stored separately and purged at train/validation boundaries.

The focal player's team is placed first in the lineup tensor. The current networks pool heroes within each side; the lineup pathway does not separately identify that player's own hero or role. Individual performance snapshots identify the focal observation. Richer personal-hero/role inputs would constitute a new model/input comparison, not an unnoticed change to this benchmark.

Only returning accounts with sufficient future observations receive labels. This creates right-censoring and selection bias. The strict main split leaves 25,172 training, 899 validation and 942 test examples, with only 21 limited-observed-history test examples. Test coverage is less than a day. `low_history` describes observation coverage, not verified account novelty. No novice effectiveness or long-horizon generalization claim follows from this experiment.

## Decision evidence

Historical test labels evaluate the observed teams only. A solver's own predicted objective is not independent evidence that the chosen new teams are fair. The synthetic benchmark draws training labels from a hidden mechanism, keeps independent lobbies out of training, and uses the hidden probabilities only after candidate scoring. It reports prediction NLL/AUC, predicted solver gap, true imbalance and regret against the hidden-mechanism optimum. The experiment is synthetic intervention evidence, not an online human trial.

For ordered 5v5 candidates, minimizing `abs(p-.5)` reflects one competitive-balance objective. Draws, asymmetric games, role satisfaction, waiting times and queue constraints can require different objectives. The generic search API requires the caller to specify that choice explicitly.
