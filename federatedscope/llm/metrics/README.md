# LLM Evaluation Metrics

This module provides comprehensive evaluation metrics for Large Language Models (LLMs) in FederatedScope.

## Supported Metrics

### 1. Perplexity (`perplexity`, `ppl`)
- **Description**: Standard metric for language models measuring prediction uncertainty
- **Range**: [1, +∞), lower is better
- **Formula**: PPL = exp(cross_entropy_loss)
- **Use case**: General language modeling evaluation

### 2. Token Accuracy (`token_acc`, `token_accuracy`)
- **Description**: Percentage of correctly predicted tokens
- **Range**: [0, 1], higher is better
- **Use case**: Basic LM evaluation, next-token prediction

### 3. BLEU Score (`bleu`)
- **Description**: N-gram precision-based metric for text generation
- **Range**: [0, 1], higher is better
- **Dependencies**: `pip install nltk`
- **Use case**: Machine translation, text generation

### 4. ROUGE Score (`rouge`, `rouge_l`)
- **Description**: Recall-oriented metric based on n-gram overlap
- **Range**: [0, 1], higher is better
- **Dependencies**: `pip install rouge`
- **Use case**: Text summarization, document generation

### 5. Exact Match (`exact_match`, `em`)
- **Description**: Percentage of predictions that exactly match references
- **Range**: [0, 1], higher is better
- **Use case**: Question answering, code generation

### 6. BERTScore (`bertscore`)
- **Description**: Semantic similarity using BERT embeddings
- **Range**: [0, 1], higher is better
- **Dependencies**: `pip install bert-score`
- **Use case**: Semantic evaluation of generated text

### 7. GSM8K Exact Match (`gsm8k_exact_match_accuracy`, `gsm8k_em`)
- **Description**: Numeric final-answer exact match for GSM8K
- **Range**: [0, 1], higher is better
- **Use case**: Grade-school math reasoning evaluation

### 8. KAFA Dirty-Filter Metrics
- **Metrics**: `dirty_filter_recall`, `dirty_filter_precision`, `clean_retention_rate`
- **Range**: [0, 1], higher is better
- **Use case**: Dirty-supervision filtering and clean-data preservation

### 9. Stability / NaN Rate (`nan_rate`, `stability_nan_rate`)
- **Description**: Fraction of non-finite batches/samples observed during training
- **Range**: [0, 1], lower is better
- **Use case**: Training stability monitoring

## Quick Start

### 1. Install Dependencies

```bash
# Basic metrics (perplexity, token_acc, exact_match) - no extra dependencies

# For BLEU
pip install nltk
python -c "import nltk; nltk.download('punkt')"

# For ROUGE
pip install rouge

# For BERTScore
pip install bert-score
```

### 2. Update Configuration

```yaml
trainer:
  type: llmtrainer_metrics  # Use enhanced trainer

eval:
  metrics: ['loss', 'perplexity', 'token_acc']  # Add your metrics
```

### 3. Run Training

```bash
python federatedscope/main.py --cfg your_config.yaml
```

## Configuration Examples

### Basic Setup (Perplexity + Token Accuracy)
```yaml
trainer:
  type: llmtrainer_metrics
eval:
  metrics: ['loss', 'perplexity', 'token_acc']
```

### Translation Task (BLEU + Perplexity)
```yaml
trainer:
  type: llmtrainer_metrics
eval:
  metrics: ['loss', 'perplexity', 'bleu']
```

### Summarization Task (ROUGE + Perplexity)
```yaml
trainer:
  type: llmtrainer_metrics
eval:
  metrics: ['loss', 'perplexity', 'rouge']
```

### QA Task (Exact Match + Perplexity)
```yaml
trainer:
  type: llmtrainer_metrics
eval:
  metrics: ['loss', 'perplexity', 'exact_match']
```

### Full Metrics (All Available)
```yaml
trainer:
  type: llmtrainer_metrics
eval:
  metrics: ['loss', 'perplexity', 'token_acc', 'bleu', 'rouge', 'exact_match', 'gsm8k_exact_match_accuracy', 'dirty_filter_recall', 'dirty_filter_precision', 'clean_retention_rate', 'nan_rate', 'bertscore']
```

## Metric Selection Guide

| Task Type | Recommended Metrics |
|-----------|-------------------|
| Language Modeling | perplexity, token_acc |
| Machine Translation | perplexity, bleu |
| Text Summarization | perplexity, rouge |
| Question Answering | perplexity, exact_match, token_acc |
| Code Generation | perplexity, exact_match |
| Text Generation | perplexity, bleu, rouge, bertscore |

## Implementation Details

### Perplexity Calculation
```python
perplexity = exp(avg_loss)
```
- Uses the cross-entropy loss from model outputs
- Clamped to avoid overflow (max 20.0)

### Token Accuracy
```python
token_acc = correct_tokens / total_valid_tokens
```
- Ignores padding tokens (-100 and 0)
- Calculated at each batch and accumulated

### BLEU/ROUGE/Exact Match
- These metrics require `generated_text` and `reference_text` in context
- For generation tasks, implement custom hooks to populate these fields
- See `examples/` for generation task implementations

## Custom Metrics

To add your own metric:

1. Create a metric function in `llm_metrics.py`:
```python
def eval_my_metric(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    # Your calculation
    return score
```

2. Register it in `register_llm_metrics()`:
```python
register_metric('my_metric', eval_my_metric, larger_is_better=True)
```

3. Use in config:
```yaml
eval:
  metrics: ['loss', 'my_metric']
```

## Troubleshooting

### Import Errors
- **nltk not found**: `pip install nltk && python -c "import nltk; nltk.download('punkt')"`
- **rouge not found**: `pip install rouge`
- **bert_score not found**: `pip install bert-score`

### Memory Issues
- BERTScore requires significant GPU memory
- Consider reducing batch size or using CPU for evaluation

### Metric Values
- Perplexity should typically be in range [1, 1000] for well-trained models
- Token accuracy > 0.5 is usually reasonable
- BLEU/ROUGE scores depend heavily on task difficulty

## References

1. **Perplexity**: [Wikipedia - Perplexity](https://en.wikipedia.org/wiki/Perplexity)
2. **BLEU**: [Papineni et al., 2002](https://www.aclweb.org/anthology/P02-1040.pdf)
3. **ROUGE**: [Lin, 2004](https://www.aclweb.org/anthology/W04-1013.pdf)
4. **BERTScore**: [Zhang et al., 2020](https://arxiv.org/abs/1904.09675)

## Contributing

To add new metrics:
1. Implement the metric function in `llm_metrics.py`
2. Add documentation in this README
3. Update the example config
4. Submit a PR with tests
