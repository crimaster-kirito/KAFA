"""
LLM-specific evaluation metrics for FederatedScope.

This module provides additional metrics for language model evaluation:
- Perplexity (PPL)
- Token-level Accuracy
- BLEU Score
- ROUGE Score
- Exact Match
- BERTScore
"""

import logging

logger = logging.getLogger(__name__)

# Import all metric functions
from .llm_metrics import (
    eval_perplexity,
    eval_token_accuracy,
    eval_bleu,
    eval_rouge,
    eval_exact_match,
    eval_gsm8k_exact_match,
    eval_dirty_filter_recall,
    eval_dirty_filter_precision,
    eval_clean_retention_rate,
    eval_nan_rate,
    eval_bertscore,
    extract_gsm8k_answer,
    normalize_numeric_answer,
    register_llm_metrics,
)

__all__ = [
    'eval_perplexity',
    'eval_token_accuracy',
    'eval_bleu',
    'eval_rouge',
    'eval_exact_match',
    'eval_gsm8k_exact_match',
    'eval_dirty_filter_recall',
    'eval_dirty_filter_precision',
    'eval_clean_retention_rate',
    'eval_nan_rate',
    'eval_bertscore',
    'extract_gsm8k_answer',
    'normalize_numeric_answer',
    'register_llm_metrics',
]

# Auto-register metrics when module is imported
register_llm_metrics()
