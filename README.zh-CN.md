# Shuach · 耍起

[English](README.md) · 面向多人游戏的匹配学习与分队实验框架。项目源自 GameMatch，Python 导入名和命令行入口沿用 `gamematch`。

GameMatch 是面向竞技游戏的 Python 研究库。给定一个小大厅，它可以找出满足组排和角色要求、评分最接近的两队。给定历史对局，它可以训练模型，估计新阵容的胜率或逐场更新玩家评级。当前开发版为 **0.1.0rc5.dev3**。

[中文入门](docs/start-zh.md) · [中文 API 指南](docs/api-zh.md) · [Python API in English](docs/api.md) · [命令行参考](docs/cli.md) · [文档目录](docs/README.md)

## 快速使用

在包含本版 `pyproject.toml` 的目录用 Python 3.10+ 安装。以下示例只需要基础依赖，使用 CPU 和生成的数据。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gamematch --version
python examples/constrained_assignment.py
gamematch decision-benchmark --config examples/synthetic-smoke.json --output runs/example
gamematch report runs/example
```

`gamematch --version` 应显示 `gamematch 0.1.0rc5.dev3`。第一个示例输出左右两队和每个槽位的角色。打开 `runs/example/report/report.md` 查看两个模型的预测结果和分队表现。同目录的 `summary.json` 应显示 2 个已完成模型任务、18 条决策结果。

要使用真实数据，可先检查 UCI 示例的模型和数据协议，再下载并训练：

```bash
gamematch outcome-benchmark --config examples/uci-smoke.json --dry-run
gamematch fetch-data uci-dota2 --data-root datasets
gamematch benchmark --config examples/uci-smoke.json --output runs/uci-example
```

示例使用 CPU；完整本机配置另有 MPS 设备设置。Mac 上 LightGBM 的 OpenMP 加载方式见[本机指南](docs/quickstart.md)。下载器不执行作者仓库代码；原始数据按各自许可管理，不包含在软件发行包中。

## 选择数据和模型

有两队各五个英雄 ID，可以用 `signed_hero` 估计单个英雄对胜率的贡献，或用 `set_transformer` 学习同队英雄之间的关系。有稳定的玩家 ID 和每场结果的可用时间，则可以逐场回放 OpenSkill 评级。[新增模型的训练与预测示例](docs/model-additions.md)

`gamematch models` 列出 28 个基础模型／基线 ID 和 4 个 NeuralAC 变体。`gamematch check-models` 检查手头的字段能否支持所选模型。原始对局、固定划分、验证集选模和检查点怎样关联，可参阅[赛果模型指南](docs/outcome-models.md)与[复现指南](docs/reproduction.md)。

模块 | 当前范围
--- | ---
数据 | UCI Dota2、NeuralAC Dota2018／LoL、Overwatch、DBD 两套数据、Dota2 Matches v2
预测 | 带符号的英雄效应模型、Logistic、LightGBM、FM、MLP、Deep Sets、Set Transformer、NeuralAC、序数模型及序列网络
评级 | Elo、TrueSkill、OpenSkill 的 Plackett–Luce、Bradley–Terry 与 Thurstone–Mosteller
论文模型适配 | OwO／DAWN 用于基于玩家历史的赛果预测；MMR-Net 用于未来观测评级；明确记录工业特征和任务差异
决策 | 不等大小两队、不可拆分组排、角色槽位、显式目标函数、流式精确搜索
评估 | 概率适当评分、校准、回归误差、配对不确定性、独立合成机制下的求解差距与决策遗憾
复核 | 指标重算、划分与时间边界检查、检查点重载、源码／数据身份核验
调参 | 冻结候选空间、验证集选参、两种子复核、五种子重训及 OwO／DAWN 专项搜索

英雄 ID 与玩家 ID、历史结果预测与新组队效果、观测评分代理与真实 MMR，各自有不同的数据要求。相关边界是接口和实验协议的一部分。公开数据上的架构适配不等于 CUPID、CHAMP 或 QuickSkill 工业原结果复现。

## 每个任务输入什么、预测什么

任务 | 实际输入特征 | 预测目标
--- | --- | ---
英雄阵容赛果（UCI Dota2、Dota2018、LoL） | 两队各 5 个英雄 ID；转换为两侧计数或英雄嵌入 | 左队获胜概率，标签为 0/1
玩家在线评级（Overwatch） | 两队各 6 个玩家的当前评级状态；只由此前已反馈的赛果更新 | 左队负／平／胜三类概率
玩家历史赛果（Overwatch） | 每人最近 10 场已完成历史的 5 个特征、13 个静态汇总、有效历史掩码和当前模式 | 左队负／平／胜三类概率
丰富历史赛果（Overwatch，可选） | 每人最近 10 场已完成历史的 8 个特征、46 个静态汇总；包含反馈后可用的评级、表现残差与共同参赛历史 | 左队负／平／胜三类概率
个人表现历史（LoL leagueData，可选） | 每人最近 10 场已完成历史的 25 个特征及 6 个静态汇总；对照仅保留 3 个赛果／时间字段 | team 100 获胜概率，标签为 0/1
非对称赛果（DBD 2022/2023） | 地图、捕食者角色，以及该玩家此前的击杀数、频率、场数和反馈间隔等 9 个数值特征 | 消灭人数 0–4 的五类概率及期望人数
未来观测评级（Dota2 Matches v2） | 本场结束后可见的 12 个时间步 × 6 个特征、掩码、当前 OpenSkill 状态和两队英雄 | 再完成 5 场已观测比赛后的 OpenSkill 均值
合成赛果学习 | 两队各 5 个生成的参与者 ID | 左队获胜概率；训练标签由隐藏机制采样
合成组队优化 | 候选大厅的 10 个参与者、组队约束和已训练预测器 | 使预测胜率最接近 0.5 的组队；用独立隐藏机制评价遗憾

Overwatch 的深度模型用玩家 ID 查历史，不训练玩家 ID 嵌入。DBD 缺少四名被追捕者的身份，因此当前任务是赛果预测。未来评级使用本场完整信息，是赛后预测，目标也不是游戏真实 MMR。具体字段、张量形状、可用时间和标签变换见[数据契约](docs/data-contracts.md#task-inputs-and-prediction-targets)。

## 文档和开发

- 读示例、查接口或贡献新页面，可以从[文档目录](docs/README.md)和[文档写作指南](docs/writing-guide.md)进入。
- [与其他软件的关系](docs/related-software.md)：RecBole、OpeNTF、Open Match、OpenSkill、PyEPO 的互补范围。
- [发布准备情况](docs/release-readiness.md)：本机证据、跨平台验证和公共使用方面的剩余工作。
- [调参协议](docs/hyperparameter-search.md)：候选空间、指标定义、选模记录与冻结源码续跑。
- [丰富历史指南](docs/richer-history.md)：新增输入、紧凑存储、独立数据核验和检查点推理。
- [贡献指南](CONTRIBUTING.md)、[协作规范](CODE_OF_CONDUCT.md)、[安全问题反馈](SECURITY.md)、[更新记录](CHANGELOG.md)、[第三方说明](THIRD_PARTY.md)。

```bash
python -m coverage run --source=gamematch -m unittest discover -s tests
python -m coverage report
ruff check src tests scripts examples
```

运行上述开发检查前，请先安装 `python -m pip install -e '.[tabular,rating,dev]'`。本版的安装包验证和跨平台记录见[发布准备情况](docs/release-readiness.md)。源代码采用 [Apache-2.0](LICENSE)，数据和依赖保留各自许可。

## 完整研究工作区

完整研究工作区的根目录另有 `research-index.md`、历史实验输出和文献缓存。源码发行包包含软件文档、示例、配置和测试；wheel 包含可导入的库与 `gamematch` 命令。需要复跑旧研究结果时，使用该实验保存的源码快照和配置。
