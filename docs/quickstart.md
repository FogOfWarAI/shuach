# GameMatch 本机运行指南

这份指南用于开发源码目录和 macOS 本地实验。请在仓库根目录创建虚拟环境后运行下列命令。

## 安装与检查

`requirements-local-lock.txt` 记录历史实验依赖版本。要运行全部测试，可用下面的命令安装开发依赖。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[tabular,rating,dev]'
.venv/bin/python -m gamematch --version
.venv/bin/python scripts/run_local.py doctor
.venv/bin/python scripts/run_local.py --python -m unittest discover -s tests -v
```

`doctor` 同时检查 LightGBM、TrueSkill 和 OpenSkill 能否实际导入，并报告版本。`not_installed` 表示未安装，`import_error` 附带加载失败的原因。

本机 LightGBM 需要 LLVM OpenMP。源码目录中的 `scripts/run_local.py` 可在 PyTorch 提供 libomp 时为子进程添加搜索路径。旧版 PyTorch 可能没有该库，需按 LightGBM 的安装说明配置兼容的 OpenMP；macOS 上对应 libomp。通过 wheel 安装时可直接执行 `gamematch doctor` 检查环境。

## 实验命令

在项目根目录执行，使用新的输出目录：

```bash
.venv/bin/python scripts/run_local.py benchmark --config benchmarks/local-pilot.json --output runs/my-pilot
.venv/bin/python scripts/run_local.py benchmark --config benchmarks/local-full-heroes.json --output runs/my-heroes
.venv/bin/python -m gamematch rating-benchmark --config benchmarks/local-overwatch-ratings.json --output runs/my-ratings
.venv/bin/python -m gamematch history-benchmark --config benchmarks/local-overwatch-history.json --output runs/my-history
.venv/bin/python scripts/run_local.py asymmetric-benchmark --config benchmarks/local-asymmetric.json --output runs/my-asymmetric
.venv/bin/python -m gamematch decision-benchmark --config benchmarks/local-decisions.json --output runs/my-decisions
.venv/bin/python -m gamematch skill-benchmark --config benchmarks/local-skill.json --output runs/my-skill
```

完整配置按模型选择 CPU 或本机 MPS；例如原始英雄主实验的 NeuralAC 使用 CPU，FM、MLP 和 Deep Sets 使用 MPS。后续的 NeuralAC MPS 性能优化没有替换原始训练记录。没有 MPS 时复制配置，同时修改 `training.device` 和 `model_training` 中的设备覆盖项，再使用新输出目录。框架不会将失败的 GPU 作业静默算作完成的 CPU 结果。

如需未来评分实验，先执行 `gamematch fetch-data dota2-matches-v2 --data-root datasets` 和 `python scripts/prepare_dota_reference.py --data-root datasets`。若数据已准备在研究工作区，可用 `--workspace` 指向该目录，另选新的 `--output`。历史构建源码可在当时派生目录的 `reference-builder.py` 找到。

公开数据可通过 `python -m gamematch datasets` 查看，使用 `python -m gamematch fetch-data uci-dota2 --data-root datasets` 下载并核验。各数据集保留各自许可，下载过程不执行作者仓库代码。

## 协议与边界

- 英雄轨道只用两队英雄，不能做玩家纵向评级。训练集拟合词表；队内置换和双方交换共享划分组。NAC 约 80/10/10；UCI 保留官方测试并清除训练／验证中的重叠组。pilot 不打开官方测试。
- Overwatch 按时间约 70/15/15 划分，仅在 `time + duration + delay < current_time` 后更新结果。原始时间戳含义仍有显式假设，并运行延迟一天的敏感性对照。验证集校准；测试时网络与校准器冻结，先前已完成测试比赛可更新历史。
- OwO/DAWN 使用公开赛果、模式与历史，缺少工业 KDA、经济、角色和组排信息。实验采用无位置的 6v6、三类 WDL 和较小网络，属于架构与数据适配。
- DBD 保留 0–4 击杀分布，只使用地图、角色与已完成捕猎者历史。缺少四名逃生者身份，不能称为完整 1v4 重组队实验。
- 决策轨道由独立合成机制评价未发生组队；真实概率不交给模型或搜索评分器。这不是真人线上效果证据。

未来评级轨道使用完成比赛后的 12 个三分钟快照，预测再经过 5 场已观测比赛的 OpenSkill 状态。它不是实际 MMR 或真实技能，样本只覆盖可观察到后续比赛的回访玩家。训练和验证按目标可用时间清除边界泄漏。

## 输出与恢复

每次运行保存 manifest、配置、源码快照、数据来源与精确划分。每个模型保存最佳检查点、训练日志、校准参数、逐例概率和指标。神经网络为 .pt，表格模型和校准器为 .joblib；只加载自己生成或信任的序列化文件。

当前版本要求源码、配置、数据和运行环境身份一致，才能复用已完成模型；未完成模型从头重跑，暂不恢复中断 epoch 的优化器状态。身份变化需使用新输出目录。本轮启动代码冻结在 `.cache/frozen-*` 并通过 PYTHONPATH 运行，允许训练期间继续开发。

执行 `.venv/bin/python -m gamematch report runs/my-heroes` 可为任一正式轨道生成独立摘要，默认写入 `runs/my-heroes/report/`，不覆盖原始结果。缺少已声明种子的组合显示为 pending；可用 `--output reports/my-heroes` 指定专用报告目录。

训练时间包含验证与数据传送。并发运行的墙钟时间不作为独占硬件的速度排名。

## 约束接口

`gamematch.matching.constraints.constrained_assignments` 支持不等大小两队、不可拆分 party、角色槽位和允许角色集合。仅在明确声明可交换时去掉阵营镜像；不等队伍大小不能声明可交换。穷举适合小 lobby，完整角色排列可能迅速增长。

`.venv/bin/python examples/asymmetric_assignment.py --target-eliminations 2` 展示 1v4 的五类结果目标与组排约束；使用明确的合成角色能力，不是 DBD 重组队实验。

数据许可按来源分别管理。软件安装包不包含原始记录、论文 PDF 或第三方项目缓存。
