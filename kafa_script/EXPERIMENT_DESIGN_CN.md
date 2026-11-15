# KAFA 实验设计与指标说明

## 1. 问题背景

KAFA 面向的是异构大模型联邦微调场景：服务器端拥有能力更强但成本更高的大模型，客户端只能承载较小模型或压缩模型；同时客户端数据私有、分布异质，并可能混入与目标任务无关的数据。直接 FedAvg/LoRA 聚合会遇到三个问题：

- 模型结构不一致：服务器大模型 LoRA 与客户端小模型 LoRA 维度不同，不能直接聚合。
- 知识迁移不稳定：客户端本地数据有限且 non-IID，单纯平均容易把任务无关更新也聚进去。
- 通信与算力受限：全量模型上传下载不可接受，需要只传 PEFT 参数或更小的锚点信息。

KAFA 的核心假设是：服务器大模型可以把目标任务知识压缩为“知识锚点”，包括任务指令、原型库和软标签；客户端用这些锚点筛选相关样本、对齐中间表示并进行蒸馏；服务器再通过投影矩阵完成大/小 LoRA 的双向映射。

## 2. 算法要证明的主张

1. 有效性：KAFA-full 在目标任务最终性能上优于 Local-only、RawFedAvg、FedProx、FedOpt、OffsiteTuning、pFedMe、Ditto。
2. 效率：KAFA 在接近或超过 baseline 性能的同时，通信量低于直接传输客户端 LoRA 的方法。
3. 鲁棒性：在 non-IID 与任务无关样本注入下，相关性过滤与原型对齐能减少无关数据对训练的污染。
4. 组件必要性：去掉 projection、instruction、distill、relevance 任一组件后，性能或稳定性应下降。

## 3. 需要补充的数据集

建议按任务类型分层，而不是只跑 GSM8K：

- reasoning：GSM8K。主打数学推理，重点看 exact match、loss、PPL、收敛速度。
- instruction generation：Alpaca、Dolly-15k。主打指令跟随，重点看 loss、PPL、BLEU/ROUGE 或人工/LLM judge 扩展。
- extraction QA：SQuAD。主打抽取式问答，重点看 exact match、F1、loss。
- classification：IMDB、AGNews。主打分类鲁棒性，重点看 accuracy、macro-F1、loss。
- optional out-of-domain noise：用 IMDB 注入 GSM8K/Alpaca 客户端，或用 AGNews 注入 SQuAD，测试 relevance filter。

每个任务至少跑 3 个 seed：42、43、44。正式论文表格只使用 `evidence_level=verified_real` 的运行。

## 4. 实验矩阵

主实验方法：

- Local-only：客户端各自训练，不通信。
- RawFedAvg：普通联邦平均。
- FedProx：检查 proximal 正则是否足以稳定 non-IID。
- FedOpt：检查服务器优化器收益。
- OffsiteTuning：对比服务器 emulator / layer dropping 路线。
- pFedMe、Ditto：个性化联邦学习 baseline。
- KAFA-full：完整方法。

消融实验：

- w/o projection：不用大/小 LoRA 投影，验证结构异构映射的贡献。
- w/o instruction：不做指令锚点增强，验证任务语义引导。
- w/o distill：不使用软标签蒸馏，验证输出分布知识。
- w/o relevance：不过滤任务无关样本，验证鲁棒性。

敏感性实验：

- prototype 数量：50、100、150、300。
- relevance threshold：0.3、0.5、0.7。
- non-IID alpha：0.1、0.5、1.0。
- irrelevant ratio：0%、10%、30%。
- client model：TinyLlama/更小模型，与服务器 Llama-2-7B 的异构程度。

## 5. 指标体系

主性能：

- final_test_avg_loss：最终测试平均 loss，越低越好。
- final_test_acc / final_test_exact_match：任务可解释性能，越高越好。
- final_test_perplexity：由 avg loss 推导或日志直接读取，越低越好。
- best_test_avg_loss：训练过程中最优测试 loss。

收敛与稳定性：

- test_loss_auc：测试 loss 曲线面积，越低表示整体收敛更快更稳。
- round_to_best_test_avg_loss：达到最优 loss 的轮数。
- final_generalization_gap：val_avg_loss - test_avg_loss，用来检查 split 偏差或过拟合。
- seed std：3 个 seed 的标准差，越小越稳定。

效率：

- total_download_mb、total_upload_mb、total_comm_mb。
- loss_per_comm_gb：单位通信成本下的最终 loss。
- comm_mb_per_round、comm_mb_cumulative：用于方法理论通信量对照。

KAFA 专属诊断，后续应从 trainer 日志或结构化 JSON 里补齐：

- relevant_sample_ratio：被判定为任务相关的样本比例。
- irrelevant_sample_ratio：被过滤样本比例。
- new_prototypes_added：增量原型数量。
- prototype_bank_size：原型库大小。
- anchor_loss、proto_loss、distill_loss：组件损失曲线。
- projection_update_norm：上/下投影后的 LoRA 更新范数。

## 6. 数据使用与作图

当前工具链应遵循三层数据：

- `real_runs.csv` / `real_curves.csv`：从日志解析出来的数据。
- `evidence_level=verified_real`：有 `exp_print.log`、`system_metrics.log` 或 raw eval 的真实运行，可进论文表格。
- `evidence_level=log_only_unverified`：只有 `eval_results.log`，需要人工确认来源；不能直接作为论文结论。
- `synthetic_expectation_*.csv`：仅用于预检查图表样式和趋势，不作为实验结果。

推荐图表：

- 主表：任务 x 方法的 final_test_avg_loss、accuracy/EM、PPL、通信量、seed std。
- 收敛曲线：每个任务画 test_avg_loss mean ± std。
- 消融柱状图：KAFA-full 与四个 ablation 的性能下降。
- 通信-性能散点图：x 轴 cumulative communication，y 轴 final loss 或 accuracy。
- 鲁棒性曲线：irrelevant ratio 增大时性能变化。
- 敏感性热力图：prototype 数量 x threshold 对 final loss 的影响。

## 7. 预期结果

合理预期不是“KAFA 所有场景都碾压”，而是：

- GSM8K/SQuAD 这类需要任务结构和中间语义对齐的任务，KAFA-full 优势最明显。
- Alpaca/Dolly 上 KAFA 的主要优势可能体现为更快收敛和更低方差，而不是最终 loss 大幅下降。
- IMDB/AGNews 这种简单分类任务，绝对提升可能较小，但通信效率和抗噪性应更好。
- w/o relevance 在注入无关数据时应明显退化；如果不退化，说明 relevance filter 没有真正参与训练或注入比例太低。
- w/o projection 若优于 full，优先检查 client/server hidden size、LoRA key 映射和 projection 矩阵是否正确。

## 8. 最终结论模板

如果实验符合预期，可以得出：

KAFA 通过知识锚点把服务器大模型的任务知识转化为客户端可用的轻量监督信号，并通过 LoRA 投影解决大/小模型结构异构问题。在 non-IID 和任务无关数据存在时，原型相关性过滤与软标签蒸馏能提升本地更新质量；在通信受限条件下，KAFA 能以更低通信量获得更好的收敛与最终性能。

如果实验不符合预期，优先排查：

- 真实数据是否足够，是否误用了 synthetic 或 log-only 数据。
- KAFA trainer 是否确实接收了 knowledge_anchor。
- prototype 维度是否与 client hidden state 对齐。
- projection_mapper 是否真的参与上/下投影。
- eval metrics 是否对任务合适，例如 GSM8K 不能只看 loss，必须补 exact match。
