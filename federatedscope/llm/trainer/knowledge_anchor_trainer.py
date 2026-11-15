"""
Knowledge Anchor Trainer for Client
?1. ?
2. 
3. ?4. 
5. bank
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
import random
import re
from typing import Dict, List, Optional

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE

logger = logging.getLogger(__name__)

FINAL_ANSWER_RE = re.compile(r"(?:####|The answer is)\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^<>]+)=(-?\d+(?:\.\d+)?)>>")
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'have', 'he', 'her', 'his', 'how', 'in', 'is', 'it', 'many', 'much',
    'of', 'on', 'or', 'she', 'so', 'the', 'then', 'to', 'total', 'was',
    'what', 'will', 'with'
}


class KnowledgeAnchorClientTrainer(LLMTrainer):
    """
    
    DistilLlamaLMoRA
    """
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # GPU
        logger.info(f"Trainer?- Device: {device}")
        if hasattr(model, 'parameters') and len(list(model.parameters())) > 0:
            model_device = next(model.parameters()).device
            logger.info(f": {model_device}")
        else:
            logger.info(f": N/A")
        if torch.cuda.is_available():
            logger.info(f"CUDA - GPU: {torch.cuda.device_count()}")
            if isinstance(device, str) and 'cuda' in device:
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                logger.info(f"GPU {gpu_id} - : {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA is not available; training will use CPU.")
        
        # 
        self.knowledge_anchor = None
        self.incremental_bank = []
        self.task_relevance_threshold = getattr(
            config.llm.knowledge_anchor, 'task_relevance_threshold', 0.5
        )
        self.prototype_alignment_weight = getattr(
            config.llm.knowledge_anchor, 'prototype_alignment_weight', 0.5  # =0.5
        )
        self.intermediate_layer = getattr(
            config.llm.knowledge_anchor, 'intermediate_layer', 16  # l=16
        )
        self.use_instruction_augmentation = getattr(
            config.llm.knowledge_anchor, 'use_instruction_augmentation', True
        )
        self.max_prototypes = getattr(
            config.llm.knowledge_anchor, 'max_prototypes', 100
        )
        # FedKA
        self.top_k_prototypes = getattr(
            config.llm.knowledge_anchor, 'top_k_prototypes', 5  # K_p=5
        )
        self.prototype_temperature = getattr(
            config.llm.knowledge_anchor, 'prototype_temperature', 0.1  # =0.1
        )
        self.distill_weight = getattr(
            config.llm.knowledge_anchor, 'distill_weight', 1.0  # =1.0
        )
        self.soft_label_temperature = getattr(
            config.llm.knowledge_anchor, 'soft_label_temperature', 4.0
        )
        self.enable_mse_relevance = getattr(
            config.llm.knowledge_anchor, 'enable_mse_relevance', True
        )
        self.relevance_mse_multiplier = getattr(
            config.llm.knowledge_anchor, 'relevance_mse_multiplier', 1.5
        )
        self.incremental_mse_multiplier = getattr(
            config.llm.knowledge_anchor, 'incremental_mse_multiplier', 0.8
        )
        self.simple_client_filter = bool(
            getattr(config.llm.knowledge_anchor, 'simple_client_filter',
                    False))
        self.client_prototype_warmup_samples = max(
            0,
            int(
                getattr(config.llm.knowledge_anchor,
                        'client_prototype_warmup_samples', 64)))
        self.client_prototype_max_size = max(
            1,
            int(
                getattr(config.llm.knowledge_anchor,
                        'client_prototype_max_size', self.max_prototypes)))
        self.relevance_filter_drop_ratio = min(
            0.95,
            max(
                0.0,
                float(
                    getattr(config.llm.knowledge_anchor,
                            'relevance_filter_drop_ratio', 0.1))))
        self.relevance_filter_min_history = max(
            1,
            int(
                getattr(config.llm.knowledge_anchor,
                        'relevance_filter_min_history', 64)))
        self.use_soft_label_distill = bool(
            getattr(config.llm.knowledge_anchor, 'use_soft_label_distill',
                    True))
        self.min_keep_per_batch = max(
            0,
            int(
                getattr(config.llm.knowledge_anchor, 'min_keep_per_batch',
                        0)))
        self.max_drop_ratio = float(
            getattr(config.llm.knowledge_anchor, 'max_drop_ratio', 1.0))
        self.max_drop_ratio = min(1.0, max(0.0, self.max_drop_ratio))
        self.log_relevance_stats = bool(
            getattr(config.llm.knowledge_anchor, 'log_relevance_stats', True))
        self.enable_dirty_supervision_filter = getattr(
            config.llm.knowledge_anchor, 'enable_dirty_supervision_filter',
            True)
        self.dirty_consistency_threshold = getattr(
            config.llm.knowledge_anchor, 'dirty_consistency_threshold', 0.55)
        self._dimension_warned = False
        self._vocab_mismatch_warned = False
        self._nonfinite_warned = False
        self.client_prototypes = []
        self.relevance_score_history = []
        
        # 
        self.stats = {
            'relevant_samples': 0,
            'irrelevant_samples': 0,
            'new_prototypes_added': 0,
            'dirty_samples_seen': 0,
            'dirty_samples_filtered': 0,
            'dirty_samples_missed': 0,
            'clean_samples_filtered': 0,
            'filtered_samples': 0,
            'inconsistent_supervision_samples': 0,
            'nonfinite_samples': 0,
            'nonfinite_batches': 0,
            'client_prototypes_added': 0
        }

    def set_knowledge_anchor(self, knowledge_anchor: Dict):
        """"""
        self.knowledge_anchor = knowledge_anchor
        logger.info(
            "Client received knowledge anchor: %s instructions, %s prototypes",
            len(knowledge_anchor.get('instructions', [])),
            knowledge_anchor.get('prototype_bank',
                                 {}).get('num_prototypes', 0))

    def augment_input_with_instruction(self, input_text: str) -> str:
        """
        
        
        Args:
            input_text: 
            
        Returns:
            str: 
        """
        if not self.use_instruction_augmentation or not self.knowledge_anchor:
            return input_text
        
        instructions = self.knowledge_anchor.get('instructions', [])
        if not instructions:
            return input_text
        
        # ?        instruction = random.choice(instructions)
        
        # FedKASEP]
        # SEP]okenizerLlamatoken?        sep_token = " [SEP] "  # okenizerep_token
        augmented_text = f"{instruction}{sep_token}{input_text}"
        
        return augmented_text

    def _get_prototypes_tensor(self, device: torch.device,
                               hidden_dim: Optional[int] = None):
        if not self.knowledge_anchor:
            return None
        prototype_bank = self.knowledge_anchor.get('prototype_bank', {})
        prototypes = prototype_bank.get('prototypes', None)
        prototypes_large = prototype_bank.get('prototypes_large', None)
        if hidden_dim is not None and prototypes_large is not None:
            try:
                proto_large_dim = int(prototypes_large.shape[-1])
                if int(hidden_dim) == proto_large_dim:
                    prototypes = prototypes_large
            except Exception:
                pass
        if prototypes is None:
            return None
        if isinstance(prototypes, np.ndarray):
            prototypes = torch.from_numpy(prototypes)
        if not isinstance(prototypes, torch.Tensor):
            return None
        return prototypes.to(device=device, dtype=torch.float32)

    def _align_hidden_and_prototypes(self, hidden_state: torch.Tensor,
                                     prototypes: torch.Tensor):
        if prototypes is None:
            return hidden_state.float(), None
        hidden_state = hidden_state.float()
        prototypes = prototypes.float()
        if hidden_state.shape[-1] == prototypes.shape[-1]:
            return hidden_state, prototypes

        min_dim = min(hidden_state.shape[-1], prototypes.shape[-1])
        if not self._dimension_warned:
            logger.warning(
                "Hidden dim %s != prototype dim %s; clipping to %s.",
                hidden_state.shape[-1], prototypes.shape[-1], min_dim)
            self._dimension_warned = True
        return hidden_state[..., :min_dim], prototypes[..., :min_dim]

    def dynamic_prototype_matching(self,
                                   hidden_state: torch.Tensor,
                                   prototypes: torch.Tensor):
        """
        FedKA?        top-K_p=5?0.1?        
        Args:
            hidden_state: ?[hidden_dim]
            
        Returns:
            tuple: (top_k_indices, top_k_weights, max_similarity) - top-K
        """
        if prototypes is None:
            return None, None, 0.0

        hidden_state, prototypes = self._align_hidden_and_prototypes(
            hidden_state, prototypes)
        hidden_norm = F.normalize(hidden_state.float(), dim=-1)
        proto_norm = F.normalize(prototypes.float(), dim=-1)
        similarities = torch.matmul(proto_norm, hidden_norm)
        max_similarity = float(similarities.max().item())

        # top-K_p
        top_k = min(self.top_k_prototypes, int(similarities.shape[0]))
        if top_k <= 0:
            return None, None, max_similarity
        top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k)
        top_k_weights = F.softmax(
            top_k_similarities / self.prototype_temperature, dim=0)

        # =0.1?        top_k_weights = F.softmax(top_k_similarities / self.prototype_temperature,

        return top_k_indices, top_k_weights, max_similarity
    
    def check_task_relevance(self,
                             hidden_state: torch.Tensor,
                             prototypes: torch.Tensor,
                             avg_prototype_distance: float):
        """
        ?        
        Args:
            hidden_state: ?[hidden_dim]
            
        Returns:
            tuple: (is_relevant, matched_prototype_idx) - 
        """
        if prototypes is None:
            return False, 0.0, 0.0, None

        hidden_state, prototypes = self._align_hidden_and_prototypes(
            hidden_state, prototypes)
        mse_distances = torch.mean((prototypes - hidden_state.unsqueeze(0))**2,
                                   dim=-1)
        mse_distances = torch.nan_to_num(
            mse_distances, nan=float('inf'), posinf=float('inf'),
            neginf=float('inf'))
        min_mse = float(mse_distances.min().item())

        _, _, max_similarity = self.dynamic_prototype_matching(
            hidden_state, prototypes)

        if self.enable_mse_relevance and avg_prototype_distance > 0:
            threshold = self.relevance_mse_multiplier * avg_prototype_distance
            is_relevant = min_mse <= threshold
        else:
            is_relevant = max_similarity >= self.task_relevance_threshold

        return is_relevant, min_mse, max_similarity, mse_distances

    def _apply_instruction_prefix(self, input_ids: torch.Tensor,
                                  labels: torch.Tensor,
                                  attention_mask: torch.Tensor):
        if not self.use_instruction_augmentation or not self.knowledge_anchor:
            return input_ids, labels, attention_mask

        instructions = self.knowledge_anchor.get('instructions', [])
        if not instructions:
            return input_ids, labels, attention_mask

        instruction = random.choice(instructions)
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else ' [SEP] '
        prefix_text = f"{instruction}{sep_token}"
        prefix_ids = self.tokenizer(prefix_text,
                                    add_special_tokens=False,
                                    return_tensors='pt')['input_ids'][0].to(
                                        input_ids.device)

        if prefix_ids.numel() == 0:
            return input_ids, labels, attention_mask

        seq_len = input_ids.shape[1]
        prefix_len = int(prefix_ids.numel())
        bs = input_ids.shape[0]
        new_input_ids = []
        new_labels = []
        new_attention = []

        for b_idx in range(bs):
            attn_row = attention_mask[b_idx] if attention_mask is not None else \
                torch.ones(seq_len, dtype=torch.long, device=input_ids.device)
            merged_input = torch.cat([prefix_ids, input_ids[b_idx]], dim=0)
            merged_attn = torch.cat([
                torch.ones(prefix_len, dtype=attn_row.dtype, device=attn_row.device),
                attn_row
            ],
                                    dim=0)
            merged_label = torch.cat([
                torch.full((prefix_len,),
                           -100,
                           dtype=labels[b_idx].dtype,
                           device=labels[b_idx].device), labels[b_idx]
            ],
                                     dim=0)

            new_input_ids.append(merged_input[-seq_len:])
            new_attention.append(merged_attn[-seq_len:])
            new_labels.append(merged_label[-seq_len:])

        return torch.stack(new_input_ids), torch.stack(new_labels), torch.stack(
            new_attention)

    @staticmethod
    def _compute_avg_pairwise_mse(prototypes: torch.Tensor) -> float:
        if prototypes is None or prototypes.shape[0] <= 1:
            return 0.0
        diffs = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
        mse = (diffs * diffs).mean(dim=-1)
        mask = ~torch.eye(mse.shape[0], dtype=torch.bool, device=mse.device)
        vals = mse[mask]
        return float(vals.mean().item()) if vals.numel() > 0 else 0.0

    @staticmethod
    def _batch_text(batch, key, batch_size):
        values = batch.get(key, None)
        if values is None:
            return [''] * batch_size
        return [str(value or '') for value in values]

    @staticmethod
    def _tokens(text: str):
        return re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", str(text).lower())

    @staticmethod
    def _safe_eval_expression(expr: str):
        if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", expr):
            return None
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None

    @staticmethod
    def _arithmetic_consistency(answer: str) -> float:
        matches = list(CALC_RE.finditer(str(answer)))
        final_match = FINAL_ANSWER_RE.search(str(answer))
        if not matches:
            return 0.85 if final_match else 0.0

        correct = 0
        declared_values = []
        for match in matches:
            expr_value = KnowledgeAnchorClientTrainer._safe_eval_expression(
                match.group(1))
            declared = float(match.group(2))
            declared_values.append(declared)
            if expr_value is not None and abs(expr_value - declared) <= 1e-6:
                correct += 1

        if not final_match:
            return 0.0
        calc_score = correct / len(matches)
        final_value = float(final_match.group(1))
        final_score = 1.0 if declared_values and abs(
            final_value - declared_values[-1]) <= 1e-6 else 0.0
        if final_score == 0.0:
            return min(0.35, 0.25 * calc_score)
        return 0.65 * calc_score + 0.35 * final_score

    @staticmethod
    def _numeric_grounding(question: str, answer: str) -> float:
        q_nums = set(re.findall(r"-?\d+(?:\.\d+)?", str(question)))
        a_nums = set(re.findall(r"-?\d+(?:\.\d+)?", str(answer)))
        if not q_nums:
            return 0.7
        return len(q_nums & a_nums) / len(q_nums)

    @classmethod
    def _lexical_grounding(cls, question: str, answer: str) -> float:
        q_tokens = {
            tok
            for tok in cls._tokens(question)
            if len(tok) > 2 and tok not in STOPWORDS and not tok.replace(
                '.', '', 1).isdigit()
        }
        a_tokens = {
            tok
            for tok in cls._tokens(answer)
            if len(tok) > 2 and tok not in STOPWORDS and not tok.replace(
                '.', '', 1).isdigit()
        }
        if not q_tokens:
            return 0.7
        return len(q_tokens & a_tokens) / len(q_tokens)

    @classmethod
    def supervision_consistency_score(cls, question: str, answer: str) -> float:
        arithmetic = cls._arithmetic_consistency(answer)
        grounding = (
            0.65 * cls._numeric_grounding(question, answer) +
            0.35 * cls._lexical_grounding(question, answer))
        return 0.55 * arithmetic + 0.45 * grounding

    @staticmethod
    def _masked_lm_loss(logits, labels, keep_mask):
        keep_mask = torch.tensor(keep_mask,
                                 device=labels.device,
                                 dtype=torch.bool)
        if keep_mask.numel() == 0 or not bool(keep_mask.any().item()):
            return torch.tensor(0.0, device=labels.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none').view(shift_labels.shape)
        valid_tokens = shift_labels.ne(-100)
        per_sample_loss = token_loss.sum(dim=1) / valid_tokens.sum(
            dim=1).clamp(min=1)
        return per_sample_loss[keep_mask].mean()

    def add_to_incremental_bank(self,
                                hidden_state: torch.Tensor,
                                sample_info: Optional[Dict] = None):
        """
        ank
        ?        
        Args:
            hidden_state: ?            sample_info: 
        """
        bank_entry = {
            'prototype': hidden_state.detach().cpu(),
            'info': sample_info or {}
        }
        self.incremental_bank.append(bank_entry)
        self.stats['new_prototypes_added'] += 1
        logger.info("Added prototype to incremental bank (%s/%s).",
                    len(self.incremental_bank), self.max_prototypes)

    def get_incremental_bank(self) -> List[Dict]:
        """bank"""
        return self.incremental_bank

    def clear_incremental_bank(self):
        """Clear incremental bank."""
        self.incremental_bank = []

    def _get_client_prototypes_tensor(self, device: torch.device):
        if not self.client_prototypes:
            return None
        return torch.stack(self.client_prototypes).to(device=device,
                                                      dtype=torch.float32)

    def _add_client_prototype(self, hidden_state: torch.Tensor):
        if len(self.client_prototypes) >= self.client_prototype_max_size:
            return
        if hidden_state is None or not torch.isfinite(hidden_state).all():
            return
        proto = F.normalize(hidden_state.detach().float(), dim=-1).cpu()
        self.client_prototypes.append(proto)
        self.stats['client_prototypes_added'] += 1

    def _simple_client_space_forward(self, ctx, outputs, hidden_states, labels,
                                     attention_mask, dirty_flags):
        """Client-space ranking filter: no server-hidden MSE or KL coupling."""
        logits = outputs.logits
        proto_tensor = self._get_client_prototypes_tensor(ctx.device)
        question_texts = self._batch_text(ctx.data_batch, 'instruction',
                                          hidden_states.shape[0])
        input_texts = self._batch_text(ctx.data_batch, 'input',
                                       hidden_states.shape[0])
        output_texts = self._batch_text(ctx.data_batch, 'output',
                                        hidden_states.shape[0])
        question_texts = [
            question if question.strip() else input_texts[idx]
            for idx, question in enumerate(question_texts)
        ]

        relevant_mask = []
        scores = []
        pending_prototypes = []
        score_threshold = None
        if len(self.relevance_score_history) >= self.relevance_filter_min_history:
            score_threshold = float(
                np.percentile(
                    np.asarray(self.relevance_score_history,
                               dtype=np.float64),
                    self.relevance_filter_drop_ratio * 100.0))

        for i in range(hidden_states.shape[0]):
            hidden_i = hidden_states[i].detach().float()
            is_dirty = bool(dirty_flags[i]) if i < len(dirty_flags) else False
            if is_dirty:
                self.stats['dirty_samples_seen'] += 1

            consistency_score = 1.0
            if self.enable_dirty_supervision_filter and output_texts[i]:
                consistency_score = self.supervision_consistency_score(
                    question_texts[i], output_texts[i])
            is_inconsistent = (
                self.enable_dirty_supervision_filter and
                consistency_score < self.dirty_consistency_threshold)

            is_finite = bool(torch.isfinite(hidden_i).all().item())
            if not is_finite:
                self.stats['nonfinite_samples'] += 1

            score = 1.0
            has_reference = proto_tensor is not None and proto_tensor.numel() > 0
            if is_finite and has_reference:
                hidden_norm = F.normalize(hidden_i, dim=-1)
                score = float(torch.matmul(proto_tensor, hidden_norm).max().item())
                if np.isfinite(score):
                    self.relevance_score_history.append(score)
                    if len(self.relevance_score_history) > 5000:
                        self.relevance_score_history = self.relevance_score_history[-5000:]
                else:
                    is_finite = False
                    self.stats['nonfinite_samples'] += 1

            warming_up = len(self.client_prototypes) < \
                self.client_prototype_warmup_samples
            passes_rank = True if score_threshold is None else \
                score >= score_threshold
            keep_sample = bool(is_finite and not is_inconsistent and
                               (warming_up or passes_rank))
            relevant_mask.append(keep_sample)
            scores.append(score)

            clean_for_proto = is_finite and not is_dirty and not is_inconsistent
            if clean_for_proto:
                pending_prototypes.append(hidden_i)

            if keep_sample:
                if is_dirty:
                    self.stats['dirty_samples_missed'] += 1
                self.stats['relevant_samples'] += 1
            else:
                self.stats['filtered_samples'] += 1
                if is_inconsistent:
                    self.stats['inconsistent_supervision_samples'] += 1
                else:
                    self.stats['irrelevant_samples'] += 1
                if is_dirty:
                    self.stats['dirty_samples_filtered'] += 1
                else:
                    self.stats['clean_samples_filtered'] += 1

        for hidden_i in pending_prototypes:
            self._add_client_prototype(hidden_i)

        kept = int(sum(relevant_mask))
        if self.log_relevance_stats and scores:
            finite_scores = np.asarray([s for s in scores if np.isfinite(s)],
                                       dtype=np.float64)
            if finite_scores.size > 0:
                logger.info(
                    "Simple filter stats: score p10=%.6f p50=%.6f p90=%.6f "
                    "threshold=%s keep=%s/%s client_prototypes=%s",
                    float(np.percentile(finite_scores, 10)),
                    float(np.percentile(finite_scores, 50)),
                    float(np.percentile(finite_scores, 90)),
                    "None" if score_threshold is None else f"{score_threshold:.6f}",
                    kept, len(relevant_mask), len(self.client_prototypes))
            else:
                logger.warning(
                    "Simple filter stats contain no finite scores; keep=%s/%s",
                    kept, len(relevant_mask))

        if kept == 0:
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                    LIFECYCLE.BATCH)
        else:
            standard_loss = self._masked_lm_loss(logits, labels,
                                                 relevant_mask)
            if not torch.isfinite(standard_loss):
                logger.warning("Skip batch: non-finite simple-filter loss.")
                if hasattr(ctx, 'nan_batches'):
                    ctx.nan_batches += 1
                self.stats['nonfinite_batches'] += 1
                ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                        LIFECYCLE.BATCH)
            else:
                ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
                ctx.loss_batch = CtxVar(standard_loss, LIFECYCLE.BATCH)

        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

    def _hook_on_batch_forward(self, ctx):
        """Forward hook with anchor relevance and dirty supervision filtering."""
        if not hasattr(self, '_device_verified'):
            logger.info(f": {ctx.device}")
            if torch.cuda.is_available() and 'cuda' in str(ctx.device):
                logger.info(f"?GPU - : {ctx.device}")
            else:
                logger.warning(f"?CPU - : {ctx.device} (?")
            self._device_verified = True
        
        # batch
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(ctx.device)
        else:
            attention_mask = torch.ones_like(input_ids, device=ctx.device)
        dirty_flags = ctx.data_batch.get('_is_dirty', None)
        if dirty_flags is not None:
            dirty_flags = dirty_flags.detach().cpu().bool().tolist()
        else:
            dirty_flags = [False] * input_ids.shape[0]

        input_ids, labels, attention_mask = self._apply_instruction_prefix(
            input_ids, labels, attention_mask)
        
        # 
        outputs = ctx.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        if hasattr(ctx, 'forward_batches'):
            ctx.forward_batches += 1
        if not torch.isfinite(outputs.loss):
            if hasattr(ctx, 'nan_batches'):
                ctx.nan_batches += 1
            self.stats['nonfinite_batches'] += 1
            self.stats['nonfinite_samples'] += int(input_ids.shape[0])
            logger.warning("Skip batch: non-finite model loss.")
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                    LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return
        
        # idden_states
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            logger.error("idden_statesoutput_hidden_states=True")
            # hidden_statesoss
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            return
        
        num_layers = len(outputs.hidden_states)
        if self.intermediate_layer < 0:
            layer_idx = num_layers + self.intermediate_layer
        else:
            layer_idx = self.intermediate_layer
        
        if layer_idx < 0 or layer_idx >= num_layers:
            logger.warning("Layer index %s out of range [0, %s]; using %s.",
                           self.intermediate_layer, num_layers - 1,
                           num_layers // 2)
            layer_idx = num_layers // 2
        
        hidden_states = outputs.hidden_states[layer_idx]
        
        # 
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states * mask_expanded
            hidden_states = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            hidden_states = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # hidden_states?        del outputs.hidden_states
        if hasattr(outputs, 'past_key_values'):
            del outputs.past_key_values

        if ctx.cur_mode not in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return

        if self.simple_client_filter:
            self._simple_client_space_forward(ctx, outputs, hidden_states,
                                              labels, attention_mask,
                                              dirty_flags)
            return
        
        standard_loss = outputs.loss
        anchor_loss = torch.tensor(0.0, device=ctx.device)
        
        # atch
        relevant_mask = []
        batch_relevant_count = 0
        batch_irrelevant_count = 0
        batch_new_prototype_count = 0
        
        # ?        prototype_bank = {}
        prototypes = None
        soft_labels = None
        if self.knowledge_anchor:
            prototype_bank = self.knowledge_anchor.get('prototype_bank', {})
            prototypes = self._get_prototypes_tensor(
                ctx.device, hidden_dim=hidden_states.shape[-1])
            soft_labels = prototype_bank.get('soft_labels', None)

        if prototypes is None or prototypes.shape[0] == 0:
            # 
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return
        if not torch.isfinite(prototypes).all():
            logger.warning(
                "Prototype bank contains non-finite values; skip this batch.")
            if hasattr(ctx, 'nan_batches'):
                ctx.nan_batches += 1
            self.stats['nonfinite_batches'] += 1
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                    LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return

        avg_prototype_distance = float(
            prototype_bank.get('avg_prototype_distance', 0.0)
            or self._compute_avg_pairwise_mse(prototypes))
        question_texts = self._batch_text(ctx.data_batch, 'instruction',
                                          hidden_states.shape[0])
        input_texts = self._batch_text(ctx.data_batch, 'input',
                                       hidden_states.shape[0])
        output_texts = self._batch_text(ctx.data_batch, 'output',
                                        hidden_states.shape[0])
        question_texts = [
            question if question.strip() else input_texts[idx]
            for idx, question in enumerate(question_texts)
        ]
        
        relevance_records = []
        filtered_reasons = []
        for i in range(hidden_states.shape[0]):
            hidden_match_i, proto_match = self._align_hidden_and_prototypes(
                hidden_states[i], prototypes)
            hidden_state_i = hidden_states[i]
            is_nonfinite = (
                proto_match is None or
                not torch.isfinite(hidden_match_i.detach()).all() or
                not torch.isfinite(proto_match).all())
            if is_nonfinite:
                if not self._nonfinite_warned:
                    logger.warning(
                        "Non-finite hidden/prototype detected; affected samples "
                        "will be filtered and will not be rescued by safeguard.")
                    self._nonfinite_warned = True
                is_relevant, min_mse, max_similarity = False, float('inf'), float('-inf')
                self.stats['nonfinite_samples'] += 1
            else:
                is_relevant, min_mse, max_similarity, _ = self.check_task_relevance(
                    hidden_match_i.detach(), proto_match, avg_prototype_distance)
                if not np.isfinite(min_mse) or not np.isfinite(max_similarity):
                    is_relevant = False
                    self.stats['nonfinite_samples'] += 1
            relevance_records.append((min_mse, max_similarity))
            is_dirty = bool(dirty_flags[i]) if i < len(dirty_flags) else False
            if is_dirty:
                self.stats['dirty_samples_seen'] += 1
            consistency_score = 1.0
            if self.enable_dirty_supervision_filter and output_texts[i]:
                consistency_score = self.supervision_consistency_score(
                    question_texts[i], output_texts[i])
            is_inconsistent = (
                is_relevant and self.enable_dirty_supervision_filter and
                consistency_score < self.dirty_consistency_threshold)
            keep_sample = bool(is_relevant and not is_inconsistent)
            relevant_mask.append(keep_sample)
            filtered_reasons.append({
                'is_dirty': is_dirty,
                'is_inconsistent': is_inconsistent,
            })
            
            if keep_sample:
                if is_dirty:
                    self.stats['dirty_samples_missed'] += 1
                batch_relevant_count += 1
                self.stats['relevant_samples'] += 1
                
                # top-K_p
                top_k_indices, top_k_weights, _ = self.dynamic_prototype_matching(
                    hidden_match_i.detach(), proto_match)
                
                if top_k_indices is not None and top_k_weights is not None and proto_match is not None:
                    proto_loss = torch.tensor(0.0, device=ctx.device)
                    # ?                    proto_loss = torch.tensor(0.0, device=ctx.device)
                    distill_loss = torch.tensor(0.0, device=ctx.device)
                    
                    for k_idx, weight in zip(top_k_indices, top_k_weights):
                        k_idx = int(k_idx.item()) if isinstance(
                            k_idx, torch.Tensor) else int(k_idx)
                        weight = float(weight.item()) if isinstance(
                            weight, torch.Tensor) else float(weight)
                        target_prototype = proto_match[k_idx]
                        proto_loss += weight * F.mse_loss(
                            hidden_match_i,
                            target_prototype,
                            reduction='mean')
                        
                        if (self.use_soft_label_distill and soft_labels is not None
                                and k_idx < len(soft_labels)
                                and hasattr(outputs, 'logits')):
                            # ogitsoken
                            if attention_mask is not None:
                                seq_length = attention_mask[i].sum().item() - 1
                                sample_logits = outputs.logits[i, seq_length, :]  # [vocab_size]
                            else:
                                sample_logits = outputs.logits[i, -1, :]  # [vocab_size]
                            
                            # 
                            student_logits = F.log_softmax(
                                sample_logits.float() /
                                self.soft_label_temperature,
                                dim=0)
                            teacher_logits = soft_labels[k_idx]
                            if isinstance(teacher_logits, np.ndarray):
                                teacher_logits = torch.from_numpy(teacher_logits)
                            if not isinstance(teacher_logits, torch.Tensor):
                                teacher_logits = torch.tensor(teacher_logits)
                            teacher_logits = teacher_logits.to(
                                device=ctx.device, dtype=torch.float32)

                            # logits
                            if torch.all(teacher_logits >= 0) and abs(
                                    float(teacher_logits.sum().item()) -
                                    1.0) < 1e-2:
                                teacher_prob = teacher_logits / teacher_logits.sum(
                                ).clamp(min=1e-12)
                            else:
                                teacher_prob = F.softmax(
                                    teacher_logits /
                                    self.soft_label_temperature,
                                    dim=0)

                            # Handle tokenizer/vocab mismatch between
                            # server-side soft labels and client model logits.
                            if student_logits.shape[0] != teacher_prob.shape[0]:
                                common_dim = min(student_logits.shape[0],
                                                 teacher_prob.shape[0])
                                if not self._vocab_mismatch_warned:
                                    logger.warning(
                                        "Vocab mismatch for distillation "
                                        "(student=%s, teacher=%s); aligning to %s.",
                                        student_logits.shape[0],
                                        teacher_prob.shape[0], common_dim)
                                    self._vocab_mismatch_warned = True
                                student_logits = student_logits[:common_dim]
                                teacher_prob = teacher_prob[:common_dim]
                                teacher_prob = teacher_prob / teacher_prob.sum(
                                ).clamp(min=1e-12)
                            
                            # KL
                            kl_loss = F.kl_div(student_logits,
                                               teacher_prob,
                                               reduction='batchmean')
                            distill_loss += weight * kl_loss
                    
                    # _anchor = L_proto +  * L_distill (=1.0)
                    sample_anchor_loss = proto_loss + self.distill_weight * distill_loss
                    anchor_loss += sample_anchor_loss

                    if avg_prototype_distance > 0 and min_mse >= \
                            self.incremental_mse_multiplier * avg_prototype_distance:
                        if len(self.incremental_bank) < self.max_prototypes:
                            batch_new_prototype_count += 1
                            self.add_to_incremental_bank(
                                hidden_match_i.detach(),
                                sample_info={
                                    'type': 'weak_match',
                                    'min_mse': min_mse,
                                    'avg_prototype_distance':
                                    avg_prototype_distance
                                })
                    
                    logger.debug(f" {i}: op-K={len(top_k_indices)}, ={sample_anchor_loss.item():.4f}")
                else:
                    # bank
                    batch_new_prototype_count += 1
                    if len(self.incremental_bank) < self.max_prototypes:
                        self.add_to_incremental_bank(hidden_match_i.detach(), sample_info={'type': 'relevant_no_prototype'})
                        logger.info("Sample %s added to incremental bank (%s/%s).",
                                    i, len(self.incremental_bank),
                                    self.max_prototypes)
                    else:
                        logger.warning("Sample %s matched no prototype but bank is full.",
                                       i)
            else:
                #  - bank?                batch_irrelevant_count += 1
                batch_irrelevant_count += 1
                self.stats['filtered_samples'] += 1
                if is_inconsistent:
                    self.stats['inconsistent_supervision_samples'] += 1
                else:
                    self.stats['irrelevant_samples'] += 1
                if is_dirty:
                    self.stats['dirty_samples_filtered'] += 1
                else:
                    self.stats['clean_samples_filtered'] += 1
                logger.debug(
                    "Sample %s filtered: relevant=%s consistency=%.3f inconsistent=%s",
                    i, is_relevant, consistency_score, is_inconsistent)

        # Safeguard: avoid all samples being filtered in a batch.
        # Re-activate top candidates by smallest MSE when needed.
        batch_size = len(relevant_mask)
        keep_count = int(sum(relevant_mask))
        min_keep = min(self.min_keep_per_batch, batch_size)
        min_keep_by_drop = int(np.ceil((1.0 - self.max_drop_ratio) *
                                       batch_size))
        min_keep = max(min_keep, min_keep_by_drop)
        if keep_count < min_keep and relevance_records:
            need = min_keep - keep_count
            candidates = sorted(
                (idx for idx, keep in enumerate(relevant_mask)
                 if not keep and np.isfinite(relevance_records[idx][0])
                 and np.isfinite(relevance_records[idx][1])),
                key=lambda idx: (
                    np.isfinite(relevance_records[idx][0]) is False,
                    relevance_records[idx][0]))[:need]
            if need > 0 and not candidates:
                logger.warning(
                    "Safeguard skipped: no finite candidate is available "
                    "(batch=%s, keep=%s).", batch_size, keep_count)
            for idx in candidates:
                relevant_mask[idx] = True
                self.stats['relevant_samples'] += 1
                self.stats['filtered_samples'] = max(
                    0, self.stats['filtered_samples'] - 1)
                reason = filtered_reasons[idx]
                if reason['is_inconsistent']:
                    self.stats['inconsistent_supervision_samples'] = max(
                        0, self.stats['inconsistent_supervision_samples'] - 1)
                else:
                    self.stats['irrelevant_samples'] = max(
                        0, self.stats['irrelevant_samples'] - 1)
                if reason['is_dirty']:
                    self.stats['dirty_samples_filtered'] = max(
                        0, self.stats['dirty_samples_filtered'] - 1)
                else:
                    self.stats['clean_samples_filtered'] = max(
                        0, self.stats['clean_samples_filtered'] - 1)
                batch_relevant_count += 1
                batch_irrelevant_count = max(0, batch_irrelevant_count - 1)
            logger.warning(
                "Safeguard activated: kept %s extra samples (batch=%s, keep=%s).",
                len(candidates), batch_size, int(sum(relevant_mask)))

        if self.log_relevance_stats and relevance_records:
            mins = np.array([x[0] for x in relevance_records], dtype=np.float64)
            sims = np.array([x[1] for x in relevance_records], dtype=np.float64)
            mins_finite = mins[np.isfinite(mins)]
            sims_finite = sims[np.isfinite(sims)]
            mse_threshold = self.relevance_mse_multiplier * avg_prototype_distance \
                if self.enable_mse_relevance and avg_prototype_distance > 0 else None
            if mins_finite.size == 0 or sims_finite.size == 0:
                logger.warning(
                    "Relevance stats contain non-finite values only; "
                    "threshold_mse=%s avg_proto_dist=%.6f keep=%s/%s",
                    "None" if mse_threshold is None else f"{mse_threshold:.6f}",
                    float(avg_prototype_distance),
                    int(sum(relevant_mask)),
                    len(relevant_mask))
            else:
                logger.info(
                    "Relevance stats: min_mse p50=%.6f p90=%.6f p99=%.6f max_sim p50=%.6f "
                    "p90=%.6f threshold_mse=%s avg_proto_dist=%.6f keep=%s/%s",
                    float(np.percentile(mins_finite, 50)),
                    float(np.percentile(mins_finite, 90)),
                    float(np.percentile(mins_finite, 99)),
                    float(np.percentile(sims_finite, 50)),
                    float(np.percentile(sims_finite, 90)),
                    "None" if mse_threshold is None else f"{mse_threshold:.6f}",
                    float(avg_prototype_distance),
                    int(sum(relevant_mask)),
                    len(relevant_mask))
        
        # 
        if batch_relevant_count > 0:
            anchor_loss = anchor_loss / batch_relevant_count
        
        # 
        if sum(relevant_mask) > 0:
            standard_loss = self._masked_lm_loss(outputs.logits, labels,
                                                 relevant_mask)
            relevant_mask_tensor = torch.ones(1, device=ctx.device)
            # lossmask
            standard_loss = standard_loss * relevant_mask_tensor.mean()  # ?        
        # L = L_CE +  * L_anchor (=0.5)
        total_loss = standard_loss + self.prototype_alignment_weight * anchor_loss
        if not torch.isfinite(total_loss):
            logger.warning("Skip batch: non-finite total loss.")
            if hasattr(ctx, 'nan_batches'):
                ctx.nan_batches += 1
            self.stats['nonfinite_batches'] += 1
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(torch.tensor(0.0, device=ctx.device),
                                    LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return
        
        if batch_relevant_count > 0 or batch_irrelevant_count > 0:
            logger.debug(
                "Batch stats: kept=%s, filtered=%s, new_prototypes=%s, bank=%s",
                batch_relevant_count, batch_irrelevant_count,
                batch_new_prototype_count, len(self.incremental_bank))
        
        # batch
        if sum(relevant_mask) == 0:
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.debug("Skip batch: all %s samples were filtered.",
                         len(relevant_mask))
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            if batch_irrelevant_count > 0:
                logger.debug("Batch kept %s samples and filtered %s samples.",
                             batch_relevant_count, batch_irrelevant_count)
        
        ctx.loss_batch = CtxVar(total_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)

    def _current_filter_metrics(self, ctx):
        total_samples = self.stats['relevant_samples'] + self.stats[
            'filtered_samples']
        dirty_seen = self.stats['dirty_samples_seen']
        dirty_filtered = self.stats['dirty_samples_filtered']
        clean_filtered = self.stats['clean_samples_filtered']
        clean_seen = max(0, total_samples - dirty_seen)
        filtered_total = dirty_filtered + clean_filtered
        dirty_recall = dirty_filtered / dirty_seen if dirty_seen > 0 else 0.0
        dirty_precision = (
            dirty_filtered / filtered_total if filtered_total > 0 else 0.0)
        clean_retention = (
            (clean_seen - clean_filtered) / clean_seen
            if clean_seen > 0 else 0.0)
        sample_nan_rate = (
            self.stats['nonfinite_samples'] / total_samples
            if total_samples > 0 else 0.0)
        batch_nan_rate = (
            float(ctx.nan_batches) / float(ctx.forward_batches)
            if hasattr(ctx, 'forward_batches') and int(ctx.forward_batches) > 0
            else 0.0)
        nan_rate = max(sample_nan_rate, batch_nan_rate)
        return {
            'dirty_filter_recall': float(dirty_recall),
            'dirty_filter_precision': float(dirty_precision),
            'clean_retention_rate': float(clean_retention),
            'dirty_samples_seen': float(dirty_seen),
            'dirty_samples_filtered': float(dirty_filtered),
            'dirty_samples_missed': float(self.stats['dirty_samples_missed']),
            'clean_samples_filtered': float(clean_filtered),
            'filter_rate': float(self.stats['filtered_samples'] /
                                 total_samples)
            if total_samples > 0 else 0.0,
            'nan_rate': float(nan_rate),
            'stability_nan_rate': float(nan_rate),
            'nonfinite_samples': float(self.stats['nonfinite_samples']),
            'nonfinite_batches': float(self.stats['nonfinite_batches']),
        }

    def _hook_on_fit_end(self, ctx):
        """hook"""
        super()._hook_on_fit_end(ctx)
        filter_metrics = self._current_filter_metrics(ctx)
        if hasattr(ctx, 'eval_metrics') and isinstance(ctx.eval_metrics, dict):
            ctx.ka_filter_stats = filter_metrics
            for name, value in filter_metrics.items():
                ctx.eval_metrics[f'{ctx.cur_split}_{name}'] = value
        
        # 
        total_samples = self.stats['relevant_samples'] + self.stats[
            'filtered_samples']
        if total_samples > 0:
            relevant_ratio = self.stats['relevant_samples'] / total_samples * 100
            filtered_ratio = self.stats['filtered_samples'] / total_samples * 100
            logger.info("=" * 80)
            logger.info("Knowledge-anchor client training statistics")
            logger.info("=" * 80)
            logger.info("Total samples: %s", total_samples)
            logger.info("Kept samples: %s (%.1f%%)",
                        self.stats['relevant_samples'], relevant_ratio)
            logger.info(
                "Filtered samples: %s (%.1f%%), low_relevance=%s, inconsistent_supervision=%s",
                self.stats['filtered_samples'], filtered_ratio,
                self.stats['irrelevant_samples'],
                self.stats['inconsistent_supervision_samples'])
            logger.info("New prototypes: %s (bank: %s/%s)",
                        self.stats['new_prototypes_added'],
                        len(self.incremental_bank), self.max_prototypes)
            if self.simple_client_filter:
                logger.info("Client-space prototypes: %s added, active=%s",
                            self.stats['client_prototypes_added'],
                            len(self.client_prototypes))
            if self.stats['nonfinite_samples'] > 0:
                logger.info("Non-finite samples: %s",
                            self.stats['nonfinite_samples'])
            dirty_seen = self.stats['dirty_samples_seen']
            if dirty_seen > 0:
                dirty_filtered = self.stats['dirty_samples_filtered']
                dirty_recall = dirty_filtered / dirty_seen * 100
                logger.info(
                    "Dirty supervision: seen=%s, filtered=%s, missed=%s, "
                    "filter_recall=%.1f%%", dirty_seen, dirty_filtered,
                    self.stats['dirty_samples_missed'], dirty_recall)
                logger.info(
                    "Clean samples filtered: %s",
                    self.stats['clean_samples_filtered'])
                logger.info(
                    "Dirty filter metrics: recall=%.4f, precision=%.4f, "
                    "clean_retention=%.4f, nan_rate=%.4f",
                    filter_metrics['dirty_filter_recall'],
                    filter_metrics['dirty_filter_precision'],
                    filter_metrics['clean_retention_rate'],
                    filter_metrics['nan_rate'])
            logger.info("=" * 80)
        else:
            logger.warning("No samples were processed by the anchor trainer.")
        
        # GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def call_knowledge_anchor_client_trainer(trainer_type):
    if trainer_type == 'knowledge_anchor_client':
        return KnowledgeAnchorClientTrainer
    return None


register_trainer('knowledge_anchor_client', call_knowledge_anchor_client_trainer)


