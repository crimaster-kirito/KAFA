# LLM评估指标增强总结

## 已完成的工作

### 1. 核心指标模块 (`federatedscope/llm/metrics/`)

创建了完整的LLM评估指标模块，包含以下指标：

#### 基础指标（无需额外依赖）
- **Perplexity (困惑度)** - `perplexity`, `ppl`
  - 语言模型核心评估指标
  - 计算公式：PPL = exp(avg_loss)
  - 值越小越好

- **Token Accuracy (词元准确率)** - `token_acc`, `token_accuracy`
  - 预测token与真实token的匹配率
  - 自动忽略padding token (-100)
  - 值越大越好

- **Exact Match (精确匹配)** - `exact_match`, `em`
  - 用于问答任务
  - 不区分大小写
  - 值越大越好

#### 高级指标（需要安装依赖）
- **BLEU Score** - `bleu`
  - 机器翻译/文本生成指标
  - 安装：`pip install nltk`
  - 值越大越好

- **ROUGE Score** - `rouge`, `rouge_l`
  - 文本摘要指标
  - 安装：`pip install rouge`
  - 值越大越好

- **BERTScore** - `bertscore`
  - 基于BERT的语义相似度
  - 安装：`pip install bert-score`
  - 值越大越好

### 2. 增强版训练器 (`federatedscope/llm/trainer/llm_trainer_with_metrics.py`)

创建了 `LLMTrainerWithMetrics` 类，特点：
- 继承自原 `LLMTrainer`
- 自动计算 perplexity 和 token_acc
- 支持在配置中指定任意指标组合
- 自动记录和输出所有指标

### 3. 配置文件示例 (`federatedscope/llm/config/llm_metrics_example.yaml`)

提供了完整的配置示例，展示如何使用新指标。

### 4. 文档和示例

- **README.md**: 详细的使用文档
- **example_usage.py**: 可运行的示例代码
- **本总结文档**: 快速参考

## 使用方法

### 快速开始

1. **在配置文件中指定指标**:
```yaml
trainer:
  type: llmtrainer_metrics  # 使用增强版训练器

eval:
  metrics: ['loss', 'perplexity', 'token_acc']  # 指定要计算的指标
```

2. **安装额外依赖**（如需要BLEU/ROUGE等）:
```bash
pip install nltk rouge bert-score
python -c "import nltk; nltk.download('punkt')"
```

3. **运行训练**:
```bash
python federatedscope/main.py --cfg your_config.yaml
```

### 指标选择建议

| 任务类型 | 推荐指标 |
|---------|---------|
| 语言模型预训练 | perplexity, token_acc |
| 机器翻译 | perplexity, bleu |
| 文本摘要 | perplexity, rouge |
| 问答系统 | perplexity, exact_match, token_acc |
| 代码生成 | perplexity, exact_match |
| 通用文本生成 | perplexity, bleu, rouge, bertscore |

### 完整指标配置示例

```yaml
eval:
  metrics: ['loss', 'perplexity', 'token_acc', 'bleu', 'rouge', 'exact_match', 'bertscore']
  best_res_update_round_wise_key: val_perplexity  # 用于选择最佳模型的指标
```

## 代码修改详情

### 修改的文件

1. **`federatedscope/core/monitors/metric_calculator.py`**
   - 添加了LLM指标相关导入
   - 实现了 `eval_perplexity`, `eval_token_accuracy`, `eval_bleu`, `eval_rouge`, `eval_exact_match` 函数
   - 在 `SUPPORT_METRICS` 字典中注册了所有新指标

### 新增的文件

1. **`federatedscope/llm/metrics/__init__.py`** - 指标模块初始化
2. **`federatedscope/llm/metrics/llm_metrics.py`** - 指标实现
3. **`federatedscope/llm/trainer/llm_trainer_with_metrics.py`** - 增强版训练器
4. **`federatedscope/llm/config/llm_metrics_example.yaml`** - 配置示例
5. **`federatedscope/llm/metrics/README.md`** - 详细文档
6. **`federatedscope/llm/metrics/example_usage.py`** - 使用示例

## 论文参考

这些指标的选择基于以下论文的最佳实践：

1. **Perplexity**: 语言模型标准评估指标
2. **BLEU**: Papineni et al. (2002) - BLEU: a Method for Automatic Evaluation of Machine Translation
3. **ROUGE**: Lin (2004) - ROUGE: A Package for Automatic Evaluation of Summaries
4. **BERTScore**: Zhang et al. (2020) - BERTScore: Evaluating Text Generation with BERT

## 下一步建议

1. **验证安装**:
   ```bash
   cd federatedscope/llm/metrics
   python example_usage.py
   ```

2. **测试配置**:
   使用提供的 `llm_metrics_example.yaml` 运行一个小实验验证指标是否正常工作

3. **自定义指标**:
   参考 `llm_metrics.py` 中的实现，可以方便地添加自己的评估指标

4. **集成到工作流**:
   将新指标集成到你的实验脚本和结果分析流程中

## 常见问题

**Q: 为什么perplexity值很大？**
A: 在训练初期 perplexity 可能较高（>100），随着训练进行应该逐渐降低。典型值在10-100之间。

**Q: token_acc应该多少算好？**
A: 取决于任务难度。语言模型通常 > 0.3 就算不错，问答任务通常 > 0.5。

**Q: 如何只使用部分指标？**
A: 在配置中指定你需要的指标即可，例如：`metrics: ['loss', 'perplexity']`

**Q: 这些指标会增加多少计算开销？**
A: perplexity和token_acc几乎没有额外开销。BLEU/ROUGE/BERTScore会增加一些CPU计算时间。
