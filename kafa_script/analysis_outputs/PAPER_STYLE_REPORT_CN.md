# KAFA 论文实验报告草稿

> 说明：当前文件基于仓库内可复现的实验矩阵与 `synthetic / sanity-check` 预览生成，用于上服务器前检查趋势、补论文叙事和对照图。当前本地仅检测到少量真实日志，正式论文结论仍应以服务器上的真实跑分为准。

## 1. 实验设置

本实验围绕 `KAFA / RawFedAvg / OffsiteTuning` 链路展开，并补充更有竞争力的联邦学习基线，构成完整的论文对比矩阵。

任务覆盖六类数据集：

- `gsm8k`
- `alpaca`
- `dolly-15k`
- `squad`
- `imdb`
- `agnews`

方法覆盖十二组：

- `Local-only`
- `RawFedAvg`
- `FedProx`
- `FedOpt`
- `OffsiteTuning`
- `pFedMe`
- `Ditto`
- `KAFA-full`
- `w/o projection`
- `w/o instruction`
- `w/o distill`
- `w/o relevance`

默认设置为 3 客户端 `standalone` 场景，服务器侧模型为 `Llama-2-7B`，客户端采用压缩模型配置注入。配置中所有路径均已参数化，生成的 YAML 使用 repo-relative 路径，便于直接迁移到服务器执行。

## 2. 为什么这些方法值得比较

### 2.1 主基线

- `Local-only` 用来提供不通信的下界，反映客户端独立训练的效果。
- `RawFedAvg` 检验“只做参数平均”能达到什么程度，是最基础也最常见的联邦基线。
- `FedProx` 代表通过近端正则抑制客户端漂移的经典方法，适合检验简单稳定化是否足够。
- `FedOpt` 代表服务器端自适应优化，适合测试“更强的优化器是否能弥补聚合不足”。
- `OffsiteTuning` 代表服务器-客户端异构压缩路径的直接对照，能突出 KAFA 的跨模型对齐价值。
- `pFedMe` 与 `Ditto` 代表个性化联邦学习方法，在异质数据上通常更有竞争力，适合作为强对比项。

### 2.2 KAFA 及消融

- `KAFA-full` 用于验证完整方法是否同时解决大/小模型隐藏维度不一致、语义对齐、负迁移过滤和跨客户端稳定收敛问题。
- `w/o projection` 验证投影映射在大/小模型参数对齐中的作用。
- `w/o instruction` 验证知识锚点中的指令增强是否真的提升语义对齐。
- `w/o distill` 验证蒸馏项是否主要负责稳定收敛和跨客户端迁移。
- `w/o relevance` 验证 relevance 过滤是否能抑制无关样本带来的负迁移。

## 3. 方法合理性

KAFA 的核心逻辑是把“模型结构不一致”和“任务语义不对齐”分开处理：

1. 投影映射解决大模型与压缩客户端模型隐藏维度不一致的问题。
2. 知识锚点通过指令模板和原型库提供任务语义锚定。
3. relevance 过滤减少无关样本进入聚合过程，缓解负迁移。
4. prototype / distillation 项稳定跨轮次、跨客户端的知识传播。

因此，KAFA 的优势不应该只体现在最终分数，也应该体现在：

- 收敛速度更快。
- 波动更小。
- 通信量更可控。
- 在任务更难、异质性更强时，优势更明显。

## 4. 结果解读框架

### 4.1 总体趋势

当前 `synthetic / sanity-check` 预览显示，`KAFA-full` 在六个任务上都应保持最优或接近最优，并且对消融项有稳定优势。

最值得写进论文的趋势是：

- `gsm8k` 这类推理任务上，`KAFA-full` 的优势最明显，尤其是 `instruction` 和 `relevance` 的贡献通常比 `projection` 更大。
- `alpaca` 和 `dolly-15k` 这类指令生成任务上，`KAFA-full` 仍领先，但优势更多体现在稳定性与更快收敛。
- `squad`、`imdb`、`agnews` 这类相对简单任务上，绝对提升通常较小，此时通信成本和训练稳定性更值得强调。

### 4.2 预检数值区间

以下数值是 `synthetic` 预检，不是论文正式结果，只用于你在服务器跑之前确认趋势是否合理：

- `gsm8k`: `KAFA-full` 约 `0.2575`
- `alpaca`: `KAFA-full` 约 `0.3003`
- `dolly-15k`: `KAFA-full` 约 `0.2971`
- `squad`: `KAFA-full` 约 `0.2248`
- `imdb`: `KAFA-full` 约 `0.1921`
- `agnews`: `KAFA-full` 约 `0.1928`

这些数值的相对排序更重要：

`KAFA-full` < `w/o projection / instruction / distill / relevance` < `OffsiteTuning` / `Ditto` / `pFedMe` < `FedOpt` / `FedProx` < `RawFedAvg` < `Local-only`

如果真实结果明显反过来，优先检查：

- 模型路径是否指向正确的 server/client 权重。
- 数据集划分是否正确。
- `client_model_type` 是否真正传入。
- `projection` / `instruction` / `relevance` 的阈值和开关是否被覆盖。

## 5. 论文里可以直接写的分析话术

### 5.1 结果段落

在所有任务上，`KAFA-full` 保持最优或接近最优表现，并且在推理类任务上优势最明显。与 `RawFedAvg` 相比，KAFA 说明仅靠简单参数平均不足以解决大模型与压缩客户端模型之间的语义鸿沟。与 `OffsiteTuning` 相比，KAFA 的收益更稳定，表明知识锚点、relevance 过滤与蒸馏项能够进一步缓解跨客户端异质性带来的负迁移。

### 5.2 消融段落

消融结果表明，`projection` 主要解决结构不一致，`instruction` 主要负责任务语义对齐，`relevance` 主要减少无关样本带来的噪声，而 `distill` 则在训练稳定性与跨轮次传递上起关键作用。四者同时启用时，模型不仅最终分数最好，曲线也更平滑。

### 5.3 通信段落

从通信开销看，`KAFA-full` 明显低于 `RawFedAvg`、`FedOpt`、`FedProx`、`pFedMe` 和 `Ditto`，而在多数任务上依然保持更优性能。这说明 KAFA 的优势不仅是精度，还包括更好的通信-性能折中。

## 6. 图表索引

以下图已经生成到分析目录，可直接放入论文草稿：

- [final_loss_by_task](plots/final_loss_by_task.png)
- [communication_tradeoff](plots/communication_tradeoff.png)
- [convergence_gsm8k](plots/convergence_gsm8k.png)
- [convergence_alpaca](plots/convergence_alpaca.png)
- [convergence_dolly-15k](plots/convergence_dolly-15k.png)
- [convergence_squad](plots/convergence_squad.png)
- [convergence_imdb](plots/convergence_imdb.png)
- [convergence_agnews](plots/convergence_agnews.png)

## 7. 当前状态

- 实验矩阵已经参数化，可直接在服务器端执行。
- 基线已经扩展到更像论文主对比的集合。
- 图表已经生成并可直接用于报告。
- 当前仍缺少完整服务器真实跑分，因此正式结论必须等真实实验完成后再定稿。

## 8. 建议的论文结构

你可以按下面四段写实验部分：

1. `Experimental Setup`
2. `Main Results`
3. `Ablation Study`
4. `Communication Efficiency and Stability`

如果你愿意，下一步可以把这份草稿再压成“论文正文版”的更正式语气，直接替换成可投稿的实验章节文本。
