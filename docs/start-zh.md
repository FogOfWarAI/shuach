# 第一次使用 Shuach

如果你有一组候选玩家，可以直接调用分队搜索。如果你有历史对局，可以训练赛果模型，再用它给新的阵容打分。下面先做一次无需数据、无需训练的分队。

## 安装后先检查版本

使用 Python 3.10 或更高版本，在仓库根目录或解压后包含 `pyproject.toml` 的目录执行以下命令。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gamematch --version
```

上述激活命令适用于 macOS 和 Linux。使用 Windows PowerShell 时，激活命令是 `.\.venv\Scripts\Activate.ps1`。当前开发版显示 `gamematch 0.1.0rc5.dev3`。只有 wheel 时，在虚拟环境里用 pip 安装本地的 `.whl` 文件即可；下面的 Python 示例也可直接运行。

## 给四个人分队

小林、小周、小何、小陈的评分分别是 8、7、3、2。下面枚举所有两人对两人的方案，目标是让两队总分尽可能接近。

```python
from gamematch.matching.constraints import constrained_assignments
from gamematch.matching.search import minimize_assignments

ratings = {"小林": 8, "小周": 7, "小何": 3, "小陈": 2}

def rating_gaps(batch):
    gaps = []
    for assignment in batch:
        left, right = assignment.teams
        left_total = sum(ratings[name] for name in left)
        right_total = sum(ratings[name] for name in right)
        gaps.append(abs(left_total - right_total))
    return gaps

result = minimize_assignments(
    constrained_assignments(ratings, (2, 2), exchangeable_sides=True),
    score_candidates=rating_gaps,
    objective=lambda gaps: gaps,
)
print(result["assignment"].teams)
assert result["cost"] == 0 and result["candidate_count"] == 3
```

输出是 `(('小林', '小陈'), ('小周', '小何'))`。两队分别是 8＋2 和 7＋3，各得 10 分。`cost` 是传入的目标函数值，这里的单位是评分差。由于交换阵营不影响这个目标，`exchangeable_sides=True` 只需检查三种分队。如果主客场或阵营会影响评分，保留默认值 `False`，让求解器检查两个朝向。[组排与角色约束示例](matching-api.md)

## 换成真实英雄阵容

假设你有两队各五个**英雄 ID**，希望预测左队获胜的概率。从源码目录运行下面的命令。第一条校验本地缓存或下载 UCI 数据，第二条用 CPU 训练两个模型。

```bash
gamematch fetch-data uci-dota2 --data-root datasets
gamematch outcome-benchmark --config examples/uci-roster-models.json --output runs/roster-example
gamematch report runs/roster-example
```

打开 `runs/roster-example/report/report.md`，确认两个模型都有结果。要给新阵容打分，把 `runs/roster-example/uci-dota2/seed-0-signed_hero` 传给 `HeroPredictor`，再调用 `predict`。一个样本写作 `[[[1,2,3,4,5], [6,7,8,9,10]]]`，最外层表示这批只有一局。返回的第一项是左队胜率。静态阵容数据没有玩家身份，不能用它更新个人评级。[完整预测代码](model-additions.md)

## 按手头的数据找接口

| 你手里有什么 | 想得到什么 | 接下来读什么 |
| --- | --- | --- |
| 大厅、固定组排、角色槽位 | 一种满足约束的分队 | [组队 API](matching-api.md) |
| Python 项目里的对局或阵容 | 查找参数、返回值、数据形状和常见错误 | [中文 API 指南](api-zh.md) |
| 已完成的对局与玩家 ID | 逐场更新评级，估计负／平／胜概率 | [评级模型与示例](model-additions.md) |
| 已训练模型和新样本 | 每局的预测结果 | [模型重载与输入格式](inference-api.md) |
| 新的数据或模型实现 | 接入框架并保留可复核的结果 | [研究接口](research-api.md) · [贡献指南](../CONTRIBUTING.md) |
| 一段准备提交的文档或 API 说明 | 明确该写哪些细节、怎样验证示例 | [文档写作说明](writing-guide.md) |
