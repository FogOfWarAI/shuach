# Shuach · 耍起

Machine learning for balanced multiplayer matchmaking. Estimate who is likely to win, then use that prediction to decide how a lobby should split into teams.

[Run your first match](getting-started.md){ .md-button .md-button--primary }
[Browse the Python API](reference/index.md){ .md-button }

<div class="match-example" markdown="1">

**One lobby, two teams**

Ava (6) and Bo (4) can captain a team. Cy (3) and Di (1) fill the remaining slots. Pairing Ava with Di and Bo with Cy gives each team one captain and a total rating of seven.

</div>

[See the constraints, scorer and executable example](matching-api.md#work-through-a-four-player-lobby)

## Follow a match from data to decision

| What do you want to do? | Start here |
| --- | --- |
| Estimate player skill using only results available at match time | [Player ratings](api.md#rating-adapters) · [Data contracts](data-contracts.md) |
| Estimate a lineup’s win probability | [Choose a model](outcome-models.md) · [Reload a trained predictor](inference-api.md) |
| Keep parties together and fill required roles | [Generate feasible teams](matching-api.md#generate-candidates) · [Score assignments in batches](matching-api.md#score-candidates-in-batches) |
| Declare new input features or output heads | [Unified feature contracts](feature-schema.md) |
| Build the arrays a schema expects | [Supplying inputs](inputs.md) · [Data contracts](data-contracts.md) |
| Check prediction quality and compare team assignments | [Reproduce an experiment](reproduction.md) · [Evaluation API](research-api.md#evaluation) |

## Look up an interface

The [Python API reference](reference/index.md) lists functions, classes and methods by module. For data shapes, supported inputs and worked calls, use the [Python guide](api.md). For terminal commands and output files, use the [CLI reference](cli.md).

[Model papers and BibTeX](papers.md) · [中文入门](start-zh.md) · [中文 API 指南](api-zh.md)
