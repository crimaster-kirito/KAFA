"""
LLM-specific evaluation metrics implementation.

Based on common practices in NLP/LLM research:
- Perplexity: Standard language model metric (lower is better)
- Token Accuracy: Basic LM metric (higher is better)
- BLEU: Machine translation metric (higher is better)
- ROUGE: Summarization metric (higher is better)
- Exact Match: QA metric (higher is better)
- BERTScore: Semantic similarity metric (higher is better)
"""

import logging
import re
import numpy as np

from federatedscope.register import register_metric

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

sentence_bleu = None
SmoothingFunction = None
word_tokenize = None
Rouge = None
bert_score = None
NLTK_AVAILABLE = False
ROUGE_AVAILABLE = False
BERTSCORE_AVAILABLE = False


def _ensure_nltk_available():
    global sentence_bleu, SmoothingFunction, word_tokenize, NLTK_AVAILABLE
    if sentence_bleu is not None:
        return True
    try:
        from nltk.translate.bleu_score import sentence_bleu as _sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction as _smooth
        from nltk.tokenize import word_tokenize as _word_tokenize
        sentence_bleu = _sentence_bleu
        SmoothingFunction = _smooth
        word_tokenize = _word_tokenize
        NLTK_AVAILABLE = True
        return True
    except Exception as exc:
        NLTK_AVAILABLE = False
        logger.warning("nltk not available for BLEU: %s", exc)
        return False


def _ensure_rouge_available():
    global Rouge, ROUGE_AVAILABLE
    if Rouge is not None:
        return True
    try:
        from rouge import Rouge as _Rouge
        Rouge = _Rouge
        ROUGE_AVAILABLE = True
        return True
    except Exception as exc:
        ROUGE_AVAILABLE = False
        logger.warning("rouge not available for ROUGE-L: %s", exc)
        return False


def _ensure_bertscore_available():
    global bert_score, BERTSCORE_AVAILABLE
    if bert_score is not None:
        return True
    try:
        from bert_score import score as _bert_score
        bert_score = _bert_score
        BERTSCORE_AVAILABLE = True
        return True
    except Exception as exc:
        BERTSCORE_AVAILABLE = False
        logger.warning("bert_score not available: %s", exc)
        return False

GSM8K_ANSWER_RE = re.compile(
    r"(?:####|The answer is)\s*(-?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _get_stat(ctx, *names, default=0.0):
    stats = getattr(ctx, 'ka_filter_stats', None)
    if stats is None:
        stats = getattr(ctx, 'dirty_filter_stats', None)
    if stats is None:
        stats = getattr(ctx, 'stats', None)
    if isinstance(stats, dict):
        for name in names:
            if name in stats:
                return _to_float(stats[name], default)
    for name in names:
        if hasattr(ctx, name):
            return _to_float(getattr(ctx, name), default)
    return default


def normalize_numeric_answer(answer):
    """Normalize GSM8K-style numeric answers for exact-match scoring."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    text = text.replace(',', '')
    try:
        number = float(text)
    except Exception:
        return text.lower()
    if number.is_integer():
        return str(int(number))
    return f'{number:.10g}'


def extract_gsm8k_answer(text):
    """Extract the final numeric answer from GSM8K model output/reference."""
    if text is None:
        return None
    text = str(text)
    match = GSM8K_ANSWER_RE.search(text)
    if match:
        return normalize_numeric_answer(match.group(1))
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])


# ==================== Metric Functions ====================

def eval_perplexity(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate perplexity for language model.
    
    Perplexity measures how well a probability model predicts a sample.
    Lower is better.
    
    Formula: PPL = exp(cross_entropy_loss)
    
    Args:
        ctx: Context object containing loss information
        
    Returns:
        float: Perplexity value
    """
    if hasattr(ctx, 'loss_batch_total') and hasattr(ctx, 'num_samples'):
        if ctx.num_samples > 0:
            avg_loss = ctx.loss_batch_total / ctx.num_samples
            # Clamp to avoid overflow
            avg_loss = min(avg_loss, 20.0)
            return float(np.exp(avg_loss))
    
    # Fallback: calculate from logits and labels
    if y_prob is not None and y_true is not None and TORCH_AVAILABLE:
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.detach().cpu()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu()
        
        # Reshape for cross entropy
        if y_prob.dim() == 3:
            batch_size, seq_len, vocab_size = y_prob.shape
            y_prob = y_prob.view(-1, vocab_size)
            y_true = y_true.view(-1)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(y_prob, y_true, reduction='mean', ignore_index=-100)
        return float(torch.exp(loss))
    
    return 0.0


def eval_token_accuracy(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate token-level accuracy for language model.
    
    Compares predicted tokens with ground truth labels.
    Ignores padding tokens (usually -100 or 0).
    
    Args:
        ctx: Context object
        y_true: Ground truth token IDs
        y_pred: Predicted token IDs
        y_prob: Model logits/probabilities
        
    Returns:
        float: Token accuracy (0-1)
    """
    if y_prob is None:
        return 0.0
    
    if TORCH_AVAILABLE:
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
    
    # Get predicted tokens
    if y_prob.ndim >= 2:
        y_pred = np.argmax(y_prob, axis=-1)
    else:
        y_pred = y_prob
    
    if y_true is None:
        return 0.0
    
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    if len(y_true_flat) != len(y_pred_flat):
        return 0.0
    
    # Create mask for valid tokens (ignore padding -100 and 0)
    mask = (y_true_flat != -100) & (y_true_flat != 0)
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (y_true_flat[mask] == y_pred_flat[mask]).sum()
    total = mask.sum()
    
    return float(correct) / float(total)


def eval_bleu(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate BLEU score for text generation.
    
    BLEU (Bilingual Evaluation Understudy) measures the similarity 
    between generated text and reference text using n-gram precision.
    
    Args:
        ctx: Context object with generated_text and reference_text
        
    Returns:
        float: BLEU score (0-1)
    """
    if not _ensure_nltk_available():
        logger.warning("nltk not available. Skipping BLEU calculation.")
        return 0.0
    
    # Try to get from context
    generated = getattr(ctx, 'generated_text', None)
    references = getattr(ctx, 'reference_text', None)
    
    if generated is None or references is None:
        return 0.0
    
    if not isinstance(generated, list):
        generated = [generated]
    if not isinstance(references, list):
        references = [references]
    
    if len(generated) == 0 or len(references) == 0:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    scores = []
    
    for gen, ref in zip(generated, references):
        try:
            gen_tokens = word_tokenize(str(gen).lower())
            ref_tokens = word_tokenize(str(ref).lower())
            
            if len(gen_tokens) == 0:
                scores.append(0.0)
                continue
            
            score = sentence_bleu([ref_tokens], gen_tokens, 
                                  smoothing_function=smoothing)
            scores.append(score)
        except Exception as e:
            logger.debug(f"Error calculating BLEU: {e}")
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def eval_rouge(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate ROUGE-L score for text generation.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap of n-grams between generated and reference text.
    
    Args:
        ctx: Context object with generated_text and reference_text
        
    Returns:
        float: ROUGE-L F1 score (0-1)
    """
    if not _ensure_rouge_available():
        logger.warning("rouge not available. Skipping ROUGE calculation.")
        return 0.0
    
    generated = getattr(ctx, 'generated_text', None)
    references = getattr(ctx, 'reference_text', None)
    
    if generated is None or references is None:
        return 0.0
    
    if not isinstance(generated, list):
        generated = [generated]
    if not isinstance(references, list):
        references = [references]
    
    if len(generated) == 0 or len(references) == 0:
        return 0.0
    
    try:
        rouge = Rouge()
        scores = []
        
        for gen, ref in zip(generated, references):
            try:
                gen_str = str(gen).strip()
                ref_str = str(ref).strip()
                
                if len(gen_str) == 0 or len(ref_str) == 0:
                    scores.append(0.0)
                    continue
                
                score = rouge.get_scores(gen_str, ref_str)[0]['rouge-l']['f']
                scores.append(score)
            except Exception as e:
                logger.debug(f"Error calculating ROUGE for single sample: {e}")
                scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        logger.warning(f"Error calculating ROUGE: {e}")
        return 0.0


def eval_exact_match(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate exact match score for question answering.
    
    Measures the percentage of predictions that exactly match the reference.
    
    Args:
        ctx: Context object with generated_text and reference_text
        
    Returns:
        float: Exact match ratio (0-1)
    """
    generated = getattr(ctx, 'generated_text', None)
    references = getattr(ctx, 'reference_text', None)
    
    if generated is None or references is None:
        return 0.0
    
    if not isinstance(generated, list):
        generated = [generated]
    if not isinstance(references, list):
        references = [references]
    
    if len(generated) == 0 or len(references) == 0:
        return 0.0
    
    matches = []
    for gen, ref in zip(generated, references):
        # Normalize: lowercase and strip whitespace
        gen_norm = str(gen).strip().lower()
        ref_norm = str(ref).strip().lower()
        matches.append(1.0 if gen_norm == ref_norm else 0.0)
    
    return float(np.mean(matches)) if matches else 0.0


def eval_gsm8k_exact_match(ctx, y_true=None, y_pred=None, y_prob=None,
                           **kwargs):
    """
    Calculate GSM8K final-answer exact match accuracy.

    This extracts the last/final numeric answer from generated outputs and
    references, so both "#### 42" and "The answer is 42" are handled.
    """
    generated = (
        getattr(ctx, 'generated_text', None)
        or getattr(ctx, 'predictions', None)
        or getattr(ctx, 'model_answers', None)
        or getattr(ctx, 'gsm8k_predictions', None))
    references = (
        getattr(ctx, 'reference_text', None)
        or getattr(ctx, 'references', None)
        or getattr(ctx, 'answers', None)
        or getattr(ctx, 'gsm8k_references', None))

    generated = _as_list(generated)
    references = _as_list(references)
    if not generated or not references:
        return 0.0

    matches = []
    for gen, ref in zip(generated, references):
        pred_answer = extract_gsm8k_answer(gen)
        ref_answer = extract_gsm8k_answer(ref)
        if pred_answer is None or ref_answer is None:
            matches.append(0.0)
        else:
            matches.append(1.0 if pred_answer == ref_answer else 0.0)
    return float(np.mean(matches)) if matches else 0.0


def eval_dirty_filter_recall(ctx, y_true=None, y_pred=None, y_prob=None,
                             **kwargs):
    dirty_seen = _get_stat(ctx, 'dirty_samples_seen', 'dirty_seen')
    dirty_filtered = _get_stat(ctx, 'dirty_samples_filtered',
                               'dirty_filtered')
    return dirty_filtered / dirty_seen if dirty_seen > 0 else 0.0


def eval_dirty_filter_precision(ctx, y_true=None, y_pred=None, y_prob=None,
                                **kwargs):
    dirty_filtered = _get_stat(ctx, 'dirty_samples_filtered',
                               'dirty_filtered')
    clean_filtered = _get_stat(ctx, 'clean_samples_filtered',
                               'clean_filtered',
                               'clean_samples_mistakenly_filtered')
    denom = dirty_filtered + clean_filtered
    return dirty_filtered / denom if denom > 0 else 0.0


def eval_clean_retention_rate(ctx, y_true=None, y_pred=None, y_prob=None,
                              **kwargs):
    total = _get_stat(ctx, 'total_samples', 'samples')
    dirty_seen = _get_stat(ctx, 'dirty_samples_seen', 'dirty_seen')
    clean_filtered = _get_stat(ctx, 'clean_samples_filtered',
                               'clean_filtered',
                               'clean_samples_mistakenly_filtered')
    clean_seen = max(0.0, total - dirty_seen)
    return ((clean_seen - clean_filtered) / clean_seen
            if clean_seen > 0 else 0.0)


def eval_nan_rate(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    if hasattr(ctx, 'nan_rate'):
        return _to_float(getattr(ctx, 'nan_rate'))
    nan_batches = _get_stat(ctx, 'nan_batches', 'nonfinite_batches')
    forward_batches = _get_stat(ctx, 'forward_batches', 'total_batches')
    if forward_batches > 0:
        return nan_batches / forward_batches
    nonfinite = _get_stat(ctx, 'nonfinite_samples')
    total = _get_stat(ctx, 'total_samples', 'samples')
    return nonfinite / total if total > 0 else 0.0


def eval_bertscore(ctx, y_true=None, y_pred=None, y_prob=None, **kwargs):
    """
    Calculate BERTScore for text generation.
    
    BERTScore uses contextual embeddings from BERT to compute semantic
    similarity between generated and reference text.
    
    Args:
        ctx: Context object with generated_text and reference_text
        
    Returns:
        float: BERTScore F1 (0-1)
    """
    if not _ensure_bertscore_available():
        logger.warning("bert_score not available. Skipping BERTScore calculation.")
        return 0.0
    
    generated = getattr(ctx, 'generated_text', None)
    references = getattr(ctx, 'reference_text', None)
    
    if generated is None or references is None:
        return 0.0
    
    if not isinstance(generated, list):
        generated = [generated]
    if not isinstance(references, list):
        references = [references]
    
    if len(generated) == 0 or len(references) == 0:
        return 0.0
    
    try:
        # Convert to strings
        generated_strs = [str(g) for g in generated]
        references_strs = [str(r) for r in references]
        
        # Calculate BERTScore
        P, R, F1 = bert_score(generated_strs, references_strs, 
                              lang='en', verbose=False)
        
        return float(F1.mean())
    except Exception as e:
        logger.warning(f"Error calculating BERTScore: {e}")
        return 0.0


# ==================== Metric Registration ====================

def register_llm_metrics():
    """
    Register all LLM metrics with FederatedScope.
    
    This function should be called during initialization.
    """
    metrics_to_register = [
        ('perplexity', eval_perplexity, False),
        ('ppl', eval_perplexity, False),
        ('token_acc', eval_token_accuracy, True),
        ('token_accuracy', eval_token_accuracy, True),
        ('bleu', eval_bleu, True),
        ('rouge', eval_rouge, True),
        ('rouge_l', eval_rouge, True),
        ('exact_match', eval_exact_match, True),
        ('em', eval_exact_match, True),
        ('gsm8k_exact_match', eval_gsm8k_exact_match, True),
        ('gsm8k_exact_match_accuracy', eval_gsm8k_exact_match, True),
        ('gsm8k_em', eval_gsm8k_exact_match, True),
        ('gsm8k_accuracy', eval_gsm8k_exact_match, True),
        ('dirty_filter_recall', eval_dirty_filter_recall, True),
        ('dirty_filter_precision', eval_dirty_filter_precision, True),
        ('clean_retention_rate', eval_clean_retention_rate, True),
        ('nan_rate', eval_nan_rate, False),
        ('stability_nan_rate', eval_nan_rate, False),
        ('bertscore', eval_bertscore, True),
    ]
    
    def _build_metric_register(name, metric_func, larger_is_better):
        def _register(types):
            if name in types:
                return name, metric_func, larger_is_better
            return None

        return _register

    for metric_name, func, larger_is_better in metrics_to_register:
        try:
            register_metric(
                metric_name,
                _build_metric_register(metric_name, func, larger_is_better))
            logger.debug(f"Registered metric: {metric_name}")
        except Exception as e:
            logger.warning(f"Failed to register metric {metric_name}: {e}")
    
    logger.info(f"Registered {len(metrics_to_register)} LLM metrics")
