# Model papers and BibTeX

[Download all BibTeX entries](models.bib). Expand a citation to copy it or follow the publication link. Cite the methods used in your experiment alongside the Shuach source revision and dataset. The [implementation cards](model-cards.md) document changes to inputs, architecture and evaluation.

## Hero rosters

**NeuralAC** models cooperation and competition between heroes. **Deep Sets** and **Set Transformer** motivate the permutation-invariant team encoders; the latter uses a shared encoder and one pooling query per side in Shuach. **Factorization Machines** is the source for the pairwise interaction baseline.

The signed-hero baseline uses one weight per hero across both sides. The Dota 2 Draft AI software study describes this signed additive formulation.

<!-- bib: neuralac deepsets settransformer fm draftai -->

## Player histories and future ratings

**CUPID** introduces OwO; **CHAMP** extends it with DAWN's cross-domain context. **QuickSkill** motivates MMR-Net. Shuach adapts these methods to public histories and future observed rating targets. **OmniNet**, **Transformer** and **GRU** are architectural sources.

<!-- bib: cupid champ quickskill omninet transformer gru -->

## Player rating systems

The team Elo adapter aggregates player ratings. The TrueSkill and OpenSkill adapters use their respective libraries. Cite the **OpenSkill software paper** for the library and **Weng & Lin** for the approximate online ranking methods behind its update rules.

<!-- bib: elo trueskill openskill weng2011 -->

## Prediction baselines and calibration

These are methodological references for logistic regression, ridge regression, LightGBM, backpropagation, dropout, cumulative-link ordinal regression and temperature scaling. The neural ordinal head and its parameterization are documented in the [research API](research-api.md#outcome-and-sequence-models).

<!-- bib: logistic ridge lightgbm backprop dropout ordinal temperature -->

## Related work

OptMatch, PTOM and the professional League of Legends forecasting study are discussed for comparison. The model cards and [roster guide](model-additions.md) explain the additional data those studies require. *Skill Issues* accompanies the [Skillbench comparison](related-software.md#executed-skillbench-interface-probe).

<!-- bib: optmatch ptom guantung skillissues -->
