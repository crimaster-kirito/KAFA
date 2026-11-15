"""
Example script demonstrating LLM metrics usage.

This script shows how to:
1. Calculate perplexity from model outputs
2. Compute token-level accuracy
3. Use BLEU and ROUGE scores
4. Integrate with FederatedScope training

Run with:
    python example_usage.py
"""

import torch
import torch.nn.functional as F
import numpy as np

# Import metrics
try:
    from llm_metrics import (
        eval_perplexity,
        eval_token_accuracy,
        eval_bleu,
        eval_rouge,
        eval_exact_match,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: llm_metrics not available. Run from correct directory.")


def example_perplexity():
    """Example: Calculate perplexity from logits and labels."""
    print("\n=== Example: Perplexity Calculation ===")
    
    # Simulate model outputs
    batch_size = 4
    seq_len = 10
    vocab_size = 50000
    
    # Random logits (model outputs before softmax)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Ground truth labels (token IDs)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='mean'
    )
    
    # Calculate perplexity
    perplexity = torch.exp(loss)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")
    print(f"Interpretation: Model is as confused as if it had {perplexity.item():.1f} equally likely choices")


def example_token_accuracy():
    """Example: Calculate token-level accuracy."""
    print("\n=== Example: Token Accuracy ===")
    
    batch_size = 4
    seq_len = 10
    vocab_size = 50000
    
    # Model predictions
    logits = torch.randn(batch_size, seq_len, vocab_size)
    predictions = torch.argmax(logits, dim=-1)
    
    # Ground truth (with some padding tokens)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, 5:] = -100  # Padding
    labels[1, 7:] = -100  # Padding
    
    # Calculate accuracy
    mask = labels != -100
    correct = ((predictions == labels) & mask).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    
    print(f"Correct tokens: {correct}/{total}")
    print(f"Token Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def example_bleu():
    """Example: Calculate BLEU score."""
    print("\n=== Example: BLEU Score ===")
    
    if not METRICS_AVAILABLE:
        print("Metrics not available. Skipping.")
        return
    
    # Example translations
    generated = [
        "The cat sat on the mat",
        "A quick brown fox jumps",
        "Machine translation is difficult"
    ]
    
    references = [
        "The cat sat on the mat",
        "The quick brown fox jumps over",
        "Machine translation can be challenging"
    ]
    
    # Create mock context
    class MockCtx:
        pass
    
    ctx = MockCtx()
    ctx.generated_text = generated
    ctx.reference_text = references
    
    bleu_score = eval_bleu(ctx)
    print(f"BLEU Score: {bleu_score:.4f}")
    print("Note: Perfect match = 1.0, no overlap = 0.0")


def example_rouge():
    """Example: Calculate ROUGE score."""
    print("\n=== Example: ROUGE Score ===")
    
    if not METRICS_AVAILABLE:
        print("Metrics not available. Skipping.")
        return
    
    # Example summaries
    generated = [
        "The weather is nice today",
        "Machine learning is a subset of AI",
        "Federated learning enables collaborative training"
    ]
    
    references = [
        "Today the weather is very nice",
        "Machine learning is part of artificial intelligence",
        "Federated learning allows collaborative model training"
    ]
    
    class MockCtx:
        pass
    
    ctx = MockCtx()
    ctx.generated_text = generated
    ctx.reference_text = references
    
    rouge_score = eval_rouge(ctx)
    print(f"ROUGE-L Score: {rouge_score:.4f}")
    print("Note: Measures longest common subsequence")


def example_exact_match():
    """Example: Calculate exact match score."""
    print("\n=== Example: Exact Match ===")
    
    if not METRICS_AVAILABLE:
        print("Metrics not available. Skipping.")
        return
    
    # Example QA pairs
    predictions = [
        "Paris",
        "42",
        "Machine Learning",
        "wrong answer"
    ]
    
    references = [
        "paris",  # Case insensitive match
        "42",
        "machine learning",  # Case insensitive match
        "correct answer"
    ]
    
    class MockCtx:
        pass
    
    ctx = MockCtx()
    ctx.generated_text = predictions
    ctx.reference_text = references
    
    em_score = eval_exact_match(ctx)
    print(f"Exact Match Score: {em_score:.4f}")
    print(f"Correct: {int(em_score * len(predictions))}/{len(predictions)}")


def show_integration_example():
    """Show how to integrate with FederatedScope config."""
    print("\n=== Integration Example ===")
    print("""
To use these metrics in your FederatedScope experiment:

1. Update your YAML config:

```yaml
trainer:
  type: llmtrainer_metrics

eval:
  metrics: ['loss', 'perplexity', 'token_acc', 'bleu', 'rouge']
  
# Optional: track best model by specific metric
eval:
  best_res_update_round_wise_key: val_perplexity
```

2. Run training:

```bash
python federatedscope/main.py --cfg your_config.yaml
```

3. Metrics will be automatically:
   - Calculated during evaluation
   - Logged to console
   - Saved to results
   - Used for early stopping (if configured)
""")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM Metrics Examples")
    print("=" * 60)
    
    example_perplexity()
    example_token_accuracy()
    example_bleu()
    example_rouge()
    example_exact_match()
    show_integration_example()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install nltk rouge bert-score")
    print("2. Download nltk data: python -c \"import nltk; nltk.download('punkt')\"")
    print("3. Update your config to use llmtrainer_metrics")
    print("4. Run your experiment with new metrics!")


if __name__ == "__main__":
    main()
