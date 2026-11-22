# Relationship to existing software

This comparison describes documented workflows as of 14 November 2026. It is not an exhaustive feature audit or a cross-package performance ranking. The projects solve overlapping but different problems; their extensibility can support uses beyond the examples below.

Project | Documented workflow | Relationship to GameMatch
--- | --- | ---
[skillbench](https://github.com/mgm52/skillbench), from [Skill Issues](https://arxiv.org/abs/2410.02831) | Rating emulators, match-acquisition functions and a simulator that selects historical CS:GO matchups; reported experiments use a random 50/50 split | This is a direct software precedent for separating prediction and acquisition. GameMatch adds explicit feedback-availability contracts, multiple outcome tasks and independent hidden-mechanism assignment evaluation; these contributions need assessment beyond generic modularity.
[RecBole](https://recbole.io/docs/) | Unified recommendation data, models, training and evaluation | GameMatch supplies competitive-roster and event-availability contracts. Recommender datasets and retrieval metrics are not substituted for opponent outcomes.
[OpeNTF](https://github.com/fani-lab/OpeNTF) | Neural team formation/recommendation, including temporal, graph and fairness work | GameMatch evaluates outcomes between opposing teams and separates prediction from assignment regret. Expert-team recommendation is a distinct task.
[Open Match](https://open-match.dev/site/docs/guides/matchmaker/) | Ticket pools, custom Match Functions and services that return matches/assignments | GameMatch is an offline research component. A future deployment could expose a trained scorer through a Match Function, but no such service is implemented here.
[OpenSkill](https://openskill.me/en/stable/) | Multiplayer probabilistic rating algorithms and rating prediction/update APIs | GameMatch uses the existing library and adds delayed-feedback replay, common validation calibration and comparisons with neural histories.
[PyEPO](https://khalil-research.github.io/PyEPO/) | End-to-end predict-then-optimize learning with optimization models and solver backends | GameMatch currently trains outcome predictors separately and evaluates constrained assignments in independent hidden worlds. It does not implement PyEPO's differentiable optimization training methods.

These sources support the workflow descriptions. The connections and integration possibilities in the last column are our architectural interpretation, not integrations claimed by those projects.

Skillbench's data loader also supports a chronological split; the random split describes its paper's experiment, not the absence of a time-split API. Its simulator chooses among observed team pairs, updates from a remaining recorded outcome and evaluates non-draw accuracy. This differs from evaluating newly partitioned rosters against a hidden synthetic outcome mechanism. Reference inspection and the later module probe use commit `4749a9a5b3d852a753ec82d320809bc2c82831f0`.

## Executed Skillbench interface probe

<!-- bib: skillissues -->

The local audit (`reports/skillbench-runtime-probe.json` in the full research workspace) executes unmodified pinned source modules and the author simulator on generated fixtures, using Python 3.10 and locally available dependencies. It is not a pip installation of the author's dependency lock, a CS:GO benchmark reproduction, or a speed comparison. Python 3.10 preserves the set-sampling API used by the pinned `TeamPair` implementation. The original source and research audit are cached separately and excluded from GameMatch distributions.

- With the same TrueSkill 0.4.5 parameters, draw probability 0.1 and `mpmath` backend, 200 generated 5v5 events give maximum probability error `2.67e-15` and rating-state error `3.56e-14`. GameMatch's default draw probability is 0.02; default configurations are therefore not silently equated.
- On the same 252 ordered team partitions, maximizing Skillbench's `LikeliestDrawAF` is equivalent to minimizing `abs(p-.5)` when the two win probabilities are complementary. Both interfaces attain the same optimum within `1.06e-15`; the selected candidate has zero gap under the author objective. This verifies correspondence for this specific objective, not all acquisition functions.
- The author simulator successfully consumes 12 generated match records. Its in-sample score is retained only as a smoke check and is not presented as predictive performance.
- On 80 overlapping generated events, an explicit external feedback queue around the author emulator matches GameMatch's delayed replay within `1.23e-15`. Updating immediately instead changes some probabilities by up to 0.205. This illustrates a protocol difference, not an error in Skillbench's distinct historical-acquisition task.

The script is `scripts/verify_skillbench_reference.py`. The cached dependency/source paths and interpreter are recorded in the report. An independent installer/user reproduction and matched runtime/memory measurements remain open.

The recorded local invocation is:

```bash
PYTHONHASHSEED=0 PYTHONPATH=.cache/skillbench-probe-deps \
  MPLCONFIGDIR=.cache/matplotlib \
  .cache/python310-env/bin/python scripts/verify_skillbench_reference.py
```

It requires the separately cached fixed-source manifest and modules; the script verifies those hashes before importing them and never invokes the author's downloader. These research-cache prerequisites are not included in the source distribution.

## Concrete contribution to evaluate

The implemented contribution is a coherent competitive-game research workflow: seven adapter contracts; delayed outcome and future-target availability; explicit public-data model adaptations; reloadable predictors; role/party/asymmetric candidate interfaces; independent synthetic assignment evaluation; and artifact-level metric, split and source audits. Generic modularity, the use of PyTorch, a CLI, and the familiar distinction between prediction and optimization are not new algorithms.

The most direct matched implementation comparison is NeuralAC against its separately cached author code. Numerical equivalence is checked before timing; dropout is disabled and attention biases are zeroed/frozen to match the paper equations. The separate search comparison holds candidates, scores, objective and tie rule fixed while changing candidate materialization. These measurements address particular implementation costs. They cannot establish that GameMatch is universally faster, more scalable, or more mature than the libraries above.

## Priorities for future integration

An Open Match example should preserve ticket identity, party membership, feasible roles and prediction-time availability, then measure queue/service latency separately from offline inference. A PyEPO extension should first define what feedback supplies optimization costs and how counterfactual evaluation remains independent. A recommendation/team-formation extension needs a new typed task and appropriate labels rather than reusing binary victory outcomes under a different name. These are future directions, not completed features.
