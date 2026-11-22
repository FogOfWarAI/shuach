# GameMatch Python API 指南

如果你有一个大厅，想把人分成两队，从[分队示例](start-zh.md#给四个人分队)开始。如果你已经选好两队英雄，想算左队胜率，用下表的 `HeroPredictor`。玩家评级还需要同一玩家跨场次的身份和赛果反馈时间。下文中的「左队」均指输入中的第一队。

## 先检查模型能否使用手头的数据

查询模型出处时，可在英文 API 的类或函数说明下展开 **BibTeX**，也可按模型查阅[论文与引用](papers.md)，或[下载完整的 BibTeX 文件](models.bib)。

这段代码只读模型目录，无需下载数据。返回的每一行都说明一个模型能否接收该类输入，以及是否已有训练入口。

```python
from gamematch.catalog import check_models

check = check_models("hero-roster", ["signed_hero", "set_transformer"])
for row in check["models"]:
    assert row["input_compatible"] and row["execution"] == "native_runner"
assert check["native_runner_ready"]
```

| 函数 | 传什么 | 得到什么 |
| --- | --- | --- |
| `gamematch.catalog.check_models(profile, models)` | 输入类型名和一组模型 ID | 按传入顺序返回各模型的兼容性、执行方式和原因。未知或重复 ID 会引发 `ValueError`。 |
| `gamematch.outcomes.plan_outcomes(config)` | 从 JSON 解码后的字典，含协议和该协议要求的模型、数据集或种子 | 返回原生训练计划。模型与协议不兼容时会在读数据前引发 `ValueError`。 |
| `gamematch.outcomes.run_outcomes(config_path, workspace, output)` | 配置文件路径、包含数据的工作目录、输出目录 | 训练结束后返回绝对路径。阵容任务返回 `report.md`，玩家历史、非对称赛果及评级任务返回 `results.json`。 |

全新的实验请选一个新的输出目录。续跑已有实验时，代码、数据和配置需要与运行记录匹配。[可直接执行的训练配置和命令](model-additions.md)

## 用已训练的模型预测

以下类从 `gamematch.inference` 导入。将训练产生的模型目录传给构造函数，保留上级目录中的词表或预处理文件。创建一次预测器，就可以多次调用 `predict`。表中 `N` 为本批样本数，至少为 1。

| 预测器 | `predict` 接收什么 | 如何读返回的 NumPy 数组 |
| --- | --- | --- |
| `HeroPredictor` | 整数英雄 ID，形状 `[N, 2, 5]`；每局依次列出左右队各五个英雄 | `[N]`。例如 `p[0] == 0.7` 表示模型估计第一局左队有 70% 的胜算。 |
| `HistoryPredictor` | Overwatch 适配器产生的历史、静态特征、有效步掩码和模式。输入使用标准化前的数值 | `[N, 3]`。每行从左到右是左队负、平、胜的校准概率。 |
| `AsymmetricPredictor` | DBD 适配器产生的赛前特征 DataFrame | `[N, 5]`。第 `k` 列是恰好消灭 `k` 人的概率，`k` 为 0 到 4。 |
| `SkillPredictor` | 对局快照、有效步掩码、英雄阵容和当前评级均值 | `[N]`。每项是再完成五场观测对局后的预测 OpenSkill 均值，单位为评级分。 |

如果要对一局英雄阵容打分，把数据写成 `[[[1,2,3,4,5], [6,7,8,9,10]]]`。预测器接收原始 ID 并返回概率。直接调用底层网络时则要先用训练词表编码，输出的 logit 需经过一次 sigmoid 才是概率。可以运行[训练与预测示例](model-additions.md)核对这两步。其他预测器的完整签名、设备选项和目录布局见[预测器参考](inference-api.md)。

## 给候选分队打分

`gamematch.matching.constraints.constrained_assignments(players, sizes, ...)` 逐个生成可行分队。玩家 ID 必须唯一，左右队人数之和必须等于玩家数。用 `parties` 指定不能拆散的玩家，用 `roles` 和 `allowed_roles` 指定角色槽位及玩家能担任的角色。

`gamematch.matching.search.minimize_assignments(candidates, score_candidates, objective, batch_size=256)` 每次把一批候选交给评分函数。目标函数应按原顺序为每个候选返回一个有限数值，数值越小越好。结果的 `assignment.teams` 是选中的两队，`cost` 是目标值，`candidate_count` 是实际检查的候选数。同分时保留枚举顺序最早的一项。[带角色限制的完整示例](matching-api.md)

## 调用没有得到预期结果时

| 现象 | 检查方法 |
| --- | --- |
| 有两队五个英雄，预测却报告形状不对 | 一局的形状应是 `(1, 2, 5)`，最外层是批次。 |
| 重载英雄模型找不到文件 | 传入类似 `runs/roster-example/uci-dota2/seed-0-signed_hero` 的目录，并保留其上一级 `uci-dota2/data-manifest.json`，训练词表在这个文件里。 |
| 分队搜索提示没有可行解 | 检查某个固定组排是否大于两边容量，或某个必须填的角色是否无人能担任。生成器可以返回零项，最小化函数收到空候选集时会引发 `ValueError`。 |
| 想了解每个参数或底层张量形状 | 见[英文 API 指南](api.md)、[训练与研究接口](research-api.md)和[数据契约](data-contracts.md)。 |
