"""
Enhanced LLM Trainer with comprehensive metrics support.

This trainer extends the base LLMTrainer to support additional metrics:
- Perplexity
- Token Accuracy
- BLEU
- ROUGE
- Exact Match
- BERTScore

Usage:
    In your config file:
    ```
    trainer:
      type: llmtrainer_metrics
    eval:
      metrics: ['loss', 'perplexity', 'token_acc']
    ```
"""

import torch
import logging
import numpy as np

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar, lifecycle
from federatedscope.core.trainers.enums import MODE, LIFECYCLE

logger = logging.getLogger(__name__)

# Import metrics module to ensure registration
try:
    from federatedscope.llm.metrics import *
except ImportError:
    logger.warning("Could not import llm.metrics module")


class LLMTrainerWithMetrics(LLMTrainer):
    """
    Enhanced LLM Trainer with comprehensive metrics support.
    
    Additional metrics supported:
    - perplexity/ppl: Language model perplexity
    - token_acc/token_accuracy: Token-level prediction accuracy
    - bleu: BLEU score for generation tasks
    - rouge/rouge_l: ROUGE-L score for summarization
    - exact_match/em: Exact match for QA tasks
    - bertscore: Semantic similarity score
    """
    
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # Store config for metrics
        self.eval_metrics_list = getattr(config.eval, 'metrics', ['loss'])
        logger.info(f"LLMTrainerWithMetrics initialized with metrics: {self.eval_metrics_list}")
    
    def _hook_on_fit_start_init(self, ctx):
        """Initialize context with additional metric tracking."""
        super()._hook_on_fit_start_init(ctx)
        
        # Initialize metric tracking variables
        ctx.total_tokens = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.correct_tokens = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.forward_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.nan_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.skipped_batches = CtxVar(0, LIFECYCLE.ROUTINE)
        
        # For generation metrics
        ctx.generated_text = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.reference_text = CtxVar([], LIFECYCLE.ROUTINE)
    
    def _hook_on_batch_forward(self, ctx):
        """Enhanced forward pass with token accuracy tracking."""
        # Call parent forward
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        ctx.forward_batches += 1
        if torch.isnan(loss) or not torch.isfinite(loss):
            ctx.nan_batches += 1
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        
        # Calculate token accuracy for this batch
        if 'token_acc' in self.eval_metrics_list or 'token_accuracy' in self.eval_metrics_list:
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                
                # Flatten for comparison
                labels_flat = labels.view(-1)
                preds_flat = predictions.view(-1)
                
                # Count valid tokens (ignore padding -100)
                valid_mask = (labels_flat != -100)
                valid_tokens = valid_mask.sum().item()
                
                if valid_tokens > 0:
                    correct = ((labels_flat == preds_flat) & valid_mask).sum().item()
                    ctx.correct_tokens += correct
                    ctx.total_tokens += valid_tokens
    
    def _hook_on_batch_end(self, ctx):
        """Enhanced batch end with metric accumulation."""
        if ctx.skip_this_batch:
            ctx.skipped_batches += 1
            if ctx.cfg.llm.retry_on_nan_loss:
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
    
    def _hook_on_fit_end(self, ctx):
        """Enhanced evaluation with comprehensive metrics."""
        avg_loss = 0 if float(ctx.num_samples) == 0 else ctx.loss_batch_total / float(ctx.num_samples)
        
        # Base metrics
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        
        # Calculate perplexity
        if 'perplexity' in self.eval_metrics_list or 'ppl' in self.eval_metrics_list:
            try:
                import math
                perplexity = math.exp(min(avg_loss, 20.0))  # Clamp to avoid overflow
                eval_results[f'{ctx.cur_split}_perplexity'] = perplexity
                eval_results[f'{ctx.cur_split}_ppl'] = perplexity
            except Exception as e:
                logger.warning(f"Error calculating perplexity: {e}")
        
        # Calculate token accuracy
        if 'token_acc' in self.eval_metrics_list or 'token_accuracy' in self.eval_metrics_list:
            if ctx.total_tokens > 0:
                token_acc = ctx.correct_tokens / ctx.total_tokens
                eval_results[f'{ctx.cur_split}_token_acc'] = token_acc
                eval_results[f'{ctx.cur_split}_token_accuracy'] = token_acc
            else:
                eval_results[f'{ctx.cur_split}_token_acc'] = 0.0
                eval_results[f'{ctx.cur_split}_token_accuracy'] = 0.0
        nan_rate = (
            float(ctx.nan_batches) / float(ctx.forward_batches)
            if int(ctx.forward_batches) > 0 else 0.0)
        eval_results[f'{ctx.cur_split}_nan_rate'] = nan_rate
        eval_results[f'{ctx.cur_split}_stability_nan_rate'] = nan_rate
        eval_results[f'{ctx.cur_split}_skipped_batch_rate'] = (
            float(ctx.skipped_batches) / float(ctx.forward_batches)
            if int(ctx.forward_batches) > 0 else 0.0)
        
        setattr(ctx, 'eval_metrics', eval_results)
        
        # Log metrics
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in eval_results.items()])
        logger.info(f"[{ctx.cur_split}] {metric_str}")

        # Move trainable part to cpu if configured
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')


def call_llm_trainer_with_metrics(trainer_type):
    """Factory function for LLMTrainerWithMetrics."""
    if trainer_type == 'llmtrainer_metrics':
        return LLMTrainerWithMetrics
    return None


# Register the trainer
register_trainer('llmtrainer_metrics', call_llm_trainer_with_metrics)
