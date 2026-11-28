# 从阵容到评估

假设你在开赛前拿到两支五人队的英雄阵容，想预测左队获胜的概率。仓库中的 `hero-roster` 示例覆盖了从组装输入到评估预测的全过程。在仓库根目录、已激活的 Python 环境中运行：

```sh
python -m pip install -e ".[neural]"
python examples/schema_model.py --profile hero-roster --steps 20 --seed 0
```

### 1. 输入：每条记录代表什么？

每条样本是一场比赛，阵容按「比赛、队伍、玩家」排列。两支队伍的先后顺序要与胜负标签一致。示例使用 `sample_inputs` 生成两批数据；接入真实数据时，用 `FeatureBatch.from_natural` 传入相同结构的英雄编号和赛果。每个字段的数组形状、编号和权重约定见[输入指南](inputs.md)。

### 2. 特征处理：预测时能看见什么？

`hero_id` 是玩家级别的离散特征。编码器查找英雄向量并聚合成队伍表示。若要加入历史战绩，先为字段声明可用时间；一项战绩只有在预测时刻之前已经可用，才可以参与评分。新增字段时要同时说明它属于玩家、队伍还是整场比赛。[特征契约](feature-schema.md)

### 3. 模型：如何得到胜率？

`make_model_from_schema` 根据声明构建编码器和预测头。训练时 `model(batch)` 返回原始分数；推理时 `predict_proba(batch)` 返回两列概率。本例的标签顺序是 `loss, win`，因此第二列表示左队胜率。要复现论文里的具体算法，请按数据集选择对应的模型和实验配置。[模型指南](outcome-models.md)

### 4. 任务：什么算预测正确？

任务在建模前由 `gamematch.tasks.OutputSpec` 声明。本例目标是左队是否获胜，损失函数使用带 logits 的二元交叉熵。如果比赛允许平局，应声明负、平、胜三个有序类别并保留平局样本。若目标是给定大厅选出最均衡的分队，还需要规定组排、角色等约束，以及比较候选方案的目标。[分队接口](matching-api.md)

### 5. 评估：输出数字能说明什么？

示例在另一批生成样本上打印样本数、对数损失和 Brier 分数。两项指标越小越好；对数损失对自信但错误的预测惩罚更大。生成数据适合检查训练与评估是否接通。衡量真实赛果预测质量需要独立测试集，并遵循相应基准的时间或分组切分。

比较分队方案还要看决策遗憾，即所选方案与已知最优方案的真实代价之差。使用 `gamematch decision-benchmark --config examples/synthetic-smoke.json --output runs/example` 可以在已知生成机制的大厅上查看这类结果。[复现实验与审计](reproduction.md)
