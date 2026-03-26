"""
Knowledge Anchor Trainer for Client
客户端知识锚点微调器：
1. 接收服务器下发的知识锚点（指令+原型库）
2. 使用指令进行输入增强
3. 在中层进行原型对齐
4. 判断任务相关性，过滤无关数据
5. 维护增量bank
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
import random
from typing import Dict, List, Optional

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE

logger = logging.getLogger(__name__)


class KnowledgeAnchorClientTrainer(LLMTrainer):
    """
    客户端知识锚点微调器
    使用DistilLlama等小型LLM，配合LoRA进行微调
    """
    def __init__(self, model, data, device, config, only_for_eval=False, monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # 验证GPU使用
        logger.info(f"Trainer初始化 - Device: {device}")
        if hasattr(model, 'parameters') and len(list(model.parameters())) > 0:
            model_device = next(model.parameters()).device
            logger.info(f"模型设备: {model_device}")
        else:
            logger.info(f"模型设备: N/A")
        if torch.cuda.is_available():
            logger.info(f"CUDA可用 - GPU数量: {torch.cuda.device_count()}")
            if isinstance(device, str) and 'cuda' in device:
                gpu_id = int(device.split(':')[1]) if ':' in device else 0
                logger.info(f"使用GPU {gpu_id} - 内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("CUDA不可用，将使用CPU训练（会很慢）")
        
        # 知识锚点相关配置
        self.knowledge_anchor = None  # 从服务器接收的知识锚点
        self.incremental_bank = []  # 增量原型bank
        self.task_relevance_threshold = getattr(
            config.llm.knowledge_anchor, 'task_relevance_threshold', 0.5
        )
        self.prototype_alignment_weight = getattr(
            config.llm.knowledge_anchor, 'prototype_alignment_weight', 0.5  # λ=0.5
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
        # FedKA方法参数
        self.top_k_prototypes = getattr(
            config.llm.knowledge_anchor, 'top_k_prototypes', 5  # K_p=5
        )
        self.prototype_temperature = getattr(
            config.llm.knowledge_anchor, 'prototype_temperature', 0.1  # τ=0.1
        )
        self.distill_weight = getattr(
            config.llm.knowledge_anchor, 'distill_weight', 1.0  # γ=1.0
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
        self._dimension_warned = False
        
        # 统计信息
        self.stats = {
            'relevant_samples': 0,
            'irrelevant_samples': 0,
            'new_prototypes_added': 0
        }

    def set_knowledge_anchor(self, knowledge_anchor: Dict):
        """设置知识锚点（从服务器接收）"""
        self.knowledge_anchor = knowledge_anchor
        logger.info(f"客户端接收到知识锚点: {len(knowledge_anchor.get('instructions', []))} 个指令, "
                   f"{knowledge_anchor.get('prototype_bank', {}).get('num_prototypes', 0)} 个原型")

    def augment_input_with_instruction(self, input_text: str) -> str:
        """
        使用指令进行输入增强
        
        Args:
            input_text: 原始输入文本
            
        Returns:
            str: 增强后的文本
        """
        if not self.use_instruction_augmentation or not self.knowledge_anchor:
            return input_text
        
        instructions = self.knowledge_anchor.get('instructions', [])
        if not instructions:
            return input_text
        
        # 随机选择一个指令模板
        instruction = random.choice(instructions)
        
        # 组合指令和输入（FedKA方法：使用[SEP]分隔符）
        # 注意：实际应用中，[SEP]可能需要根据tokenizer调整（如Llama使用特殊token）
        sep_token = " [SEP] "  # 或使用tokenizer的sep_token
        augmented_text = f"{instruction}{sep_token}{input_text}"
        
        return augmented_text

    def _get_prototypes_tensor(self, device: torch.device):
        if not self.knowledge_anchor:
            return None
        prototype_bank = self.knowledge_anchor.get('prototype_bank', {})
        prototypes = prototype_bank.get('prototypes', None)
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
            return hidden_state, None
        if hidden_state.shape[-1] == prototypes.shape[-1]:
            return hidden_state, prototypes

        min_dim = min(hidden_state.shape[-1], prototypes.shape[-1])
        if not self._dimension_warned:
            logger.warning(
                f"hidden维度({hidden_state.shape[-1]})与原型维度({prototypes.shape[-1]})不一致，"
                f"将自动裁剪到{min_dim}维")
            self._dimension_warned = True
        return hidden_state[..., :min_dim], prototypes[..., :min_dim]

    def dynamic_prototype_matching(self,
                                   hidden_state: torch.Tensor,
                                   prototypes: torch.Tensor):
        """
        动态原型匹配（FedKA方法）
        计算与所有原型的余弦相似度，选取top-K_p=5个最相似原型，计算温度软权重（τ=0.1）
        
        Args:
            hidden_state: 样本的隐藏状态 [hidden_dim]
            
        Returns:
            tuple: (top_k_indices, top_k_weights, max_similarity) - top-K原型索引、软权重、最大相似度
        """
        if prototypes is None:
            return None, None, 0.0

        hidden_state, prototypes = self._align_hidden_and_prototypes(
            hidden_state, prototypes)
        hidden_norm = F.normalize(hidden_state, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        similarities = torch.matmul(proto_norm, hidden_norm)
        max_similarity = float(similarities.max().item())

        # 选取top-K_p个最相似原型
        top_k = min(self.top_k_prototypes, int(similarities.shape[0]))
        if top_k <= 0:
            return None, None, max_similarity
        top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k)

        # 计算温度软权重（τ=0.1）
        top_k_weights = F.softmax(top_k_similarities / self.prototype_temperature,
                                  dim=0)

        return top_k_indices, top_k_weights, max_similarity
    
    def check_task_relevance(self,
                             hidden_state: torch.Tensor,
                             prototypes: torch.Tensor,
                             avg_prototype_distance: float):
        """
        检查任务相关性（基于最大相似度）
        
        Args:
            hidden_state: 样本的隐藏状态 [hidden_dim]
            
        Returns:
            tuple: (is_relevant, matched_prototype_idx) - 是否相关，匹配的原型索引
        """
        if prototypes is None:
            return False, 0.0, 0.0, None

        hidden_state, prototypes = self._align_hidden_and_prototypes(
            hidden_state, prototypes)
        mse_distances = torch.mean((prototypes - hidden_state.unsqueeze(0))**2,
                                   dim=-1)
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

    def add_to_incremental_bank(self,
                                hidden_state: torch.Tensor,
                                sample_info: Optional[Dict] = None):
        """
        将新原型添加到增量bank
        注意：只有任务相关但无匹配原型的样本才会被添加
        
        Args:
            hidden_state: 样本的隐藏状态
            sample_info: 样本信息（可选）
        """
        bank_entry = {
            'prototype': hidden_state.detach().cpu(),
            'info': sample_info or {}
        }
        self.incremental_bank.append(bank_entry)
        self.stats['new_prototypes_added'] += 1
        logger.info(f"✓ 新增原型已添加到增量bank（总数: {len(self.incremental_bank)}/{self.max_prototypes}）")

    def get_incremental_bank(self) -> List[Dict]:
        """获取增量bank（用于上传到服务器）"""
        return self.incremental_bank

    def clear_incremental_bank(self):
        """清空增量bank（上传后清空）"""
        self.incremental_bank = []

    def _hook_on_batch_forward(self, ctx):
        """
        前向传播hook：处理输入增强、原型对齐、任务相关性判断
        """
        # 验证设备（仅在第一个batch）
        if not hasattr(self, '_device_verified'):
            logger.info(f"训练设备: {ctx.device}")
            if torch.cuda.is_available() and 'cuda' in str(ctx.device):
                logger.info(f"✓ 使用GPU训练 - 设备: {ctx.device}")
            else:
                logger.warning(f"⚠ 使用CPU训练 - 设备: {ctx.device} (会很慢)")
            self._device_verified = True
        
        # 获取batch数据
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(ctx.device)
        else:
            attention_mask = torch.ones_like(input_ids, device=ctx.device)

        # 指令增强（token级前缀拼接）
        input_ids, labels, attention_mask = self._apply_instruction_prefix(
            input_ids, labels, attention_mask)
        
        # 前向传播
        outputs = ctx.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 检查hidden_states是否存在
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            logger.error("模型未返回hidden_states，请确保模型支持output_hidden_states=True")
            # 如果无法获取hidden_states，使用标准loss
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            return
        
        # 获取层数并验证索引
        num_layers = len(outputs.hidden_states)
        # 支持负数索引（从末尾计算）
        if self.intermediate_layer < 0:
            layer_idx = num_layers + self.intermediate_layer
        else:
            layer_idx = self.intermediate_layer
        
        # 边界检查
        if layer_idx < 0 or layer_idx >= num_layers:
            logger.warning(f"层索引 {self.intermediate_layer} (实际索引 {layer_idx}) 超出范围 [0, {num_layers-1}]，"
                         f"使用中间层 {num_layers // 2}")
            layer_idx = num_layers // 2
        
        # 提取中间层隐藏状态用于原型对齐
        hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        
        # 对序列维度求平均
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states * mask_expanded
            hidden_states = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            hidden_states = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # 清理不需要的hidden_states以节省内存
        del outputs.hidden_states
        if hasattr(outputs, 'past_key_values'):
            del outputs.past_key_values
        
        # 任务相关性判断和结构化锚点对齐损失（FedKA方法）
        standard_loss = outputs.loss
        anchor_loss = torch.tensor(0.0, device=ctx.device)
        
        # 对batch中的每个样本进行判断
        relevant_mask = []
        batch_relevant_count = 0
        batch_irrelevant_count = 0
        batch_new_prototype_count = 0
        
        # 获取原型库和软标签
        prototype_bank = {}
        prototypes = None
        soft_labels = None
        if self.knowledge_anchor:
            prototype_bank = self.knowledge_anchor.get('prototype_bank', {})
            prototypes = self._get_prototypes_tensor(ctx.device)
            soft_labels = prototype_bank.get('soft_labels', None)

        if prototypes is None or prototypes.shape[0] == 0:
            # 没有可用原型库时，退化为标准训练
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            ctx.loss_batch = CtxVar(outputs.loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)
            return

        avg_prototype_distance = float(
            prototype_bank.get('avg_prototype_distance', 0.0)
            or self._compute_avg_pairwise_mse(prototypes))
        
        for i in range(hidden_states.shape[0]):
            hidden_state_i = hidden_states[i]  # 保持计算图用于梯度
            hidden_match_i, proto_match = self._align_hidden_and_prototypes(
                hidden_state_i, prototypes)
            is_relevant, min_mse, _, _ = self.check_task_relevance(
                hidden_match_i.detach(), proto_match, avg_prototype_distance)
            relevant_mask.append(is_relevant)
            
            if is_relevant:
                batch_relevant_count += 1
                self.stats['relevant_samples'] += 1
                
                # 动态原型匹配：获取top-K_p个原型及其软权重
                top_k_indices, top_k_weights, _ = self.dynamic_prototype_matching(
                    hidden_match_i.detach(), proto_match)
                
                if top_k_indices is not None and top_k_weights is not None and proto_match is not None:
                    # 结构化锚点对齐损失
                    proto_loss = torch.tensor(0.0, device=ctx.device)
                    distill_loss = torch.tensor(0.0, device=ctx.device)
                    
                    for k_idx, weight in zip(top_k_indices, top_k_weights):
                        k_idx = int(k_idx.item()) if isinstance(
                            k_idx, torch.Tensor) else int(k_idx)
                        weight = float(weight.item()) if isinstance(
                            weight, torch.Tensor) else float(weight)
                        target_prototype = proto_match[k_idx]
                        
                        # 对比拉近损失（简化版：使用MSE）
                        proto_loss += weight * F.mse_loss(
                            hidden_match_i,
                            target_prototype,
                            reduction='mean'
                        )
                        
                        # 加权软标签蒸馏损失（如果有软标签）
                        if soft_labels is not None and hasattr(outputs, 'logits'):
                            # 获取样本的logits（最后一个token或平均）
                            if attention_mask is not None:
                                seq_length = attention_mask[i].sum().item() - 1
                                sample_logits = outputs.logits[i, seq_length, :]  # [vocab_size]
                            else:
                                sample_logits = outputs.logits[i, -1, :]  # [vocab_size]
                            
                            # 温度软化
                            student_logits = F.log_softmax(
                                sample_logits / self.soft_label_temperature,
                                dim=0)
                            teacher_logits = soft_labels[k_idx]
                            if isinstance(teacher_logits, np.ndarray):
                                teacher_logits = torch.from_numpy(teacher_logits)
                            if not isinstance(teacher_logits, torch.Tensor):
                                teacher_logits = torch.tensor(teacher_logits)
                            teacher_logits = teacher_logits.to(
                                device=ctx.device, dtype=torch.float32)

                            # 兼容两种格式：概率分布或原始logits
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
                            
                            # KL散度
                            kl_loss = F.kl_div(student_logits,
                                               teacher_prob,
                                               reduction='batchmean')
                            distill_loss += weight * kl_loss
                    
                    # 融合锚点损失：L_anchor = L_proto + γ * L_distill (γ=1.0)
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
                    
                    logger.debug(f"样本 {i}: 任务相关，top-K={len(top_k_indices)}, 锚点损失={sample_anchor_loss.item():.4f}")
                else:
                    # 相关但没有匹配的原型，添加到增量bank
                    batch_new_prototype_count += 1
                    if len(self.incremental_bank) < self.max_prototypes:
                        self.add_to_incremental_bank(hidden_match_i.detach(), sample_info={'type': 'relevant_no_prototype'})
                        logger.info(f"样本 {i}: 任务相关但无匹配原型，添加到增量bank（当前bank大小: {len(self.incremental_bank)}/{self.max_prototypes}）")
                    else:
                        logger.warning(f"样本 {i}: 任务相关但无匹配原型，增量bank已满，无法添加")
            else:
                # 任务无关样本 - 不添加到增量bank，直接排除
                batch_irrelevant_count += 1
                self.stats['irrelevant_samples'] += 1
                logger.debug(f"样本 {i}: 任务不相关，将被排除（相似度低于阈值 {self.task_relevance_threshold}）")
        
        # 计算平均锚点损失
        if batch_relevant_count > 0:
            anchor_loss = anchor_loss / batch_relevant_count
        
        # 只对相关样本计算损失
        if sum(relevant_mask) > 0:
            # 对相关样本的loss进行mask
            relevant_mask_tensor = torch.tensor(relevant_mask, device=ctx.device, dtype=torch.float32)
            standard_loss = standard_loss * relevant_mask_tensor.mean()  # 简化处理
        
        # 总损失：L = L_CE + λ * L_anchor (λ=0.5)
        total_loss = standard_loss + self.prototype_alignment_weight * anchor_loss
        
        # 记录batch级别的统计信息（只在debug模式下输出，减少日志）
        if batch_relevant_count > 0 or batch_irrelevant_count > 0:
            logger.debug(f"Batch统计: 相关={batch_relevant_count}, 无关={batch_irrelevant_count}, "
                       f"新增原型={batch_new_prototype_count}, 增量bank大小={len(self.incremental_bank)}")
        
        # 如果所有样本都不相关，跳过这个batch
        if sum(relevant_mask) == 0:
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.debug(f"⚠ 跳过batch：所有{len(relevant_mask)}个样本任务不相关（将被排除）")
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)
            if batch_irrelevant_count > 0:
                logger.debug(f"✓ Batch处理: {batch_relevant_count}个相关样本继续训练，{batch_irrelevant_count}个无关样本被排除")
        
        ctx.loss_batch = CtxVar(total_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(outputs.logits, LIFECYCLE.BATCH)

    def _hook_on_fit_end(self, ctx):
        """微调结束hook：打印统计信息并清理内存"""
        super()._hook_on_fit_end(ctx)
        
        # 详细统计信息
        total_samples = self.stats['relevant_samples'] + self.stats['irrelevant_samples']
        if total_samples > 0:
            relevant_ratio = self.stats['relevant_samples'] / total_samples * 100
            irrelevant_ratio = self.stats['irrelevant_samples'] / total_samples * 100
            logger.info("=" * 80)
            logger.info("客户端训练统计信息")
            logger.info("=" * 80)
            logger.info(f"总样本数: {total_samples}")
            logger.info(f"  ✓ 任务相关样本（GSM8K）: {self.stats['relevant_samples']} ({relevant_ratio:.1f}%) - 已用于训练")
            logger.info(f"  ✗ 任务无关样本（IMDB）: {self.stats['irrelevant_samples']} ({irrelevant_ratio:.1f}%) - 已被排除")
            logger.info(f"  📦 新增原型数: {self.stats['new_prototypes_added']} (增量bank大小: {len(self.incremental_bank)}/{self.max_prototypes})")
            logger.info("=" * 80)
        else:
            logger.warning("未处理任何样本，统计信息为空")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def call_knowledge_anchor_client_trainer(trainer_type):
    if trainer_type == 'knowledge_anchor_client':
        return KnowledgeAnchorClientTrainer
    return None


register_trainer('knowledge_anchor_client', call_knowledge_anchor_client_trainer)
