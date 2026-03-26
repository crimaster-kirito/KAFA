"""
Knowledge Anchor Generator for Server
生成知识锚点的两个部分：
1. 指令（Instructions）：通过任务描述自动推断生成多样化Alpaca风格模板
2. 原型库（Prototype Bank）：提取公共数据集在LLM中层的隐藏状态，K-means++聚类聚合，包含软标签关联

实现FedKA方法（第四章）：
- 任务描述自动推断：固定种子分层采样，三次独立采样生成候选，融合为单一描述
- 多样指令模板生成：K=8个Alpaca风格模板
- Prototype Bank构建：K-means++聚类，M=150个原型，软标签关联（T=4.0）

注意：这是基于预训练模型的微调场景，不是从头训练
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from transformers import AutoTokenizer
import logging
from typing import List, Dict, Optional
import random

logger = logging.getLogger(__name__)


class KnowledgeAnchorGenerator:
    """
    知识锚点生成器
    """
    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: 服务器端LLM模型（Llama-7B）
            tokenizer: 对应的tokenizer
            config: 配置对象
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # 配置参数（根据FedKA论文设置）
        self.num_instructions = getattr(config.llm.knowledge_anchor, 'num_instructions', 8)  # K=8
        self.num_prototypes = getattr(config.llm.knowledge_anchor, 'num_prototypes', 150)  # M=150
        self.intermediate_layer = getattr(config.llm.knowledge_anchor, 'intermediate_layer', 16)  # l=16
        self.generation_temperature = getattr(config.llm.knowledge_anchor, 'generation_temperature', 0.7)
        self.max_new_tokens = getattr(config.llm.knowledge_anchor, 'max_new_tokens', 256)
        self.soft_label_temperature = getattr(config.llm.knowledge_anchor, 'soft_label_temperature', 4.0)  # T=4.0
        self.task_description_seed = getattr(config.llm.knowledge_anchor, 'task_description_seed', 42)  # seed=42
        self.task_description_samples = getattr(config.llm.knowledge_anchor, 'task_description_samples', 100)  # 100条指令

    def infer_task_description(self, public_dataset) -> str:
        """
        任务描述自动推断（FedKA方法）
        1. 固定随机种子（seed=42）从公共数据集分层采样100条指令
        2. 进行三次独立采样，生成三个候选任务描述
        3. 使用大模型融合三个候选为单一描述
        
        Args:
            public_dataset: 服务器端的公共数据集
            
        Returns:
            str: 融合后的任务描述
        """
        logger.info("=" * 80)
        logger.info("开始任务描述自动推断（FedKA方法）...")
        logger.info("=" * 80)
        
        # 固定随机种子
        np.random.seed(self.task_description_seed)
        random.seed(self.task_description_seed)
        
        # 提取指令文本
        instructions_texts = []
        for sample in public_dataset:
            if isinstance(sample, dict):
                text = sample.get('instruction', sample.get('input', sample.get('question', '')))
            else:
                text = str(sample)
            if text and len(text.strip()) > 0:
                instructions_texts.append(text.strip())
        
        if len(instructions_texts) == 0:
            logger.warning("无法从数据集中提取指令文本，使用默认任务描述")
            return "General language understanding and generation task"
        
        # 分层采样：如果有类别标签，按比例采样；否则随机采样
        sample_size = min(self.task_description_samples, len(instructions_texts))
        
        # 三次独立采样，生成三个候选
        candidates = []
        for i in range(3):
            # 重新设置种子以确保可重复性
            np.random.seed(self.task_description_seed + i)
            sampled_indices = np.random.choice(
                len(instructions_texts),
                size=sample_size,
                replace=False
            )
            sampled_texts = [instructions_texts[idx] for idx in sampled_indices]
            candidates.append(sampled_texts)
            logger.info(f"候选 {i+1}: 采样了 {len(sampled_texts)} 条指令")
        
        # 生成三个候选任务描述
        candidate_descriptions = []
        for i, candidate_texts in enumerate(candidates):
            # 取前几条作为代表
            representative_texts = candidate_texts[:10]
            prompt = f"""Based on the following sample instructions, generate a concise and comprehensive task description that captures the common characteristics and goals of these tasks.

Sample instructions:
{chr(10).join([f"- {text[:100]}" for text in representative_texts[:5]])}

Task description:"""
            
            description = self._generate_text(prompt, max_new_tokens=128)
            if description:
                candidate_descriptions.append(description.strip())
                logger.info(f"候选描述 {i+1}: {description[:150]}...")
            else:
                # 如果生成失败，使用简单描述
                candidate_descriptions.append(f"Task based on: {representative_texts[0][:100]}")
        
        # 融合三个候选为单一描述
        if len(candidate_descriptions) > 0:
            fusion_prompt = f"""Merge the following three task descriptions into one most accurate and comprehensive description.

Description 1: {candidate_descriptions[0]}
Description 2: {candidate_descriptions[1] if len(candidate_descriptions) > 1 else candidate_descriptions[0]}
Description 3: {candidate_descriptions[2] if len(candidate_descriptions) > 2 else candidate_descriptions[0]}

Merged task description:"""
            
            final_description = self._generate_text(fusion_prompt, max_new_tokens=256)
            if final_description:
                logger.info(f"✓ 融合后的任务描述: {final_description[:200]}...")
                return final_description.strip()
        
        # 如果融合失败，使用第一个候选
        logger.warning("任务描述融合失败，使用第一个候选")
        return candidate_descriptions[0] if candidate_descriptions else "General language understanding and generation task"

    def generate_instructions(self, public_dataset, task_description: str = None) -> List[str]:
        """
        生成指令模板，用于客户端输入增强（FedKA方法）
        使用融合后的任务描述生成K=8个多样化Alpaca风格指令模板
        
        Args:
            public_dataset: 服务器端的公共数据集
            task_description: 任务描述（如果为None，则自动推断）
            
        Returns:
            List[str]: 生成的指令模板列表
        """
        self.model.eval()
        instructions = []
        
        logger.info("=" * 80)
        logger.info("开始生成任务指令（使用服务器端LLM）...")
        logger.info("=" * 80)
        
        # 如果任务描述为空，自动推断
        if task_description is None:
            task_description = self.infer_task_description(public_dataset)
        
        # 使用融合后的任务描述生成K=8个与公共数据集风格一致的指令模板
        # 从公共数据集中采样一些样本，分析其风格特征
        logger.info(f"分析公共数据集风格并生成 {self.num_instructions} 个指令模板...")
        logger.info(f"任务描述: {task_description[:200]}...")
        
        # 从公共数据集中采样一些样本作为风格参考
        sample_size = min(20, len(public_dataset))
        sample_indices = np.random.choice(len(public_dataset), size=sample_size, replace=False)
        style_samples = []
        for idx in sample_indices:
            sample = public_dataset[idx]
            if isinstance(sample, dict):
                # 提取输入文本（根据数据集格式）
                text = sample.get('instruction', sample.get('input', sample.get('question', '')))
                if text:
                    style_samples.append(text[:200])  # 截断以节省token
            else:
                style_samples.append(str(sample)[:200])
        
        style_examples = "\n".join([f"- {s}" for s in style_samples[:5]])  # 只取前5个作为示例
        
        # 生成与公共数据集风格一致的指令模板
        prompt = f"""Based on the following task description and sample inputs from the public dataset, generate {self.num_instructions} diverse instruction templates that match the style and format of the public dataset. Each template should be concise and clear, suitable for guiding the model to complete the task. The templates should follow the same style as the sample inputs.

Task description: {task_description}

Sample inputs from public dataset (for style reference):
{style_examples}

Generate {self.num_instructions} instruction templates that match the dataset style (one per line, numbered 1-{self.num_instructions}):"""
        
        logger.info("正在使用LLM生成指令模板...")
        generated_text = self._generate_text(prompt, max_new_tokens=512)
        logger.info(f"LLM生成的原始文本:\n{generated_text[:500]}...")
        
        # 解析生成的文本，提取指令
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 移除编号和符号
            instruction = line.lstrip('0123456789.-*: ').strip()
            if len(instruction) > 10:  # 过滤太短的
                # 确保是Alpaca风格（通常以动词开头）
                instructions.append(instruction)
                logger.info(f"  ✓ 提取的指令 {len(instructions)}: {instruction}")
                if len(instructions) >= self.num_instructions:
                    break
        
        # 如果生成不足，使用默认模板
        if len(instructions) < self.num_instructions:
            default_templates = [
                "Please complete the following task:",
                "Answer the question based on the given context:",
                "Generate a response for the following input:",
                "Solve the following problem:",
                "Provide a detailed explanation for:"
            ]
            logger.info(f"\n生成的指令不足 {self.num_instructions} 个，补充默认模板...")
            for i, template in enumerate(default_templates[:self.num_instructions - len(instructions)]):
                instructions.append(template)
                logger.info(f"  ✓ 添加默认模板 {len(instructions)}: {template}")
        
        # 输出所有生成的指令
        logger.info("\n" + "=" * 80)
        logger.info(f"任务指令生成完成！共生成 {len(instructions)} 个指令:")
        logger.info("=" * 80)
        for i, instruction in enumerate(instructions[:self.num_instructions], 1):
            logger.info(f"  [{i:2d}] {instruction}")
        logger.info("=" * 80)
        
        return instructions[:self.num_instructions]

    def _generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """使用模型生成文本"""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
            
        try:
            logger.debug(f"  使用LLM生成文本 (max_tokens={max_new_tokens}, temp={self.generation_temperature})...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.generation_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除原始prompt部分
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            if generated_text:
                logger.debug(f"  LLM生成成功，长度: {len(generated_text)} 字符")
            else:
                logger.warning(f"  LLM生成结果为空")
            
            return generated_text
        except Exception as e:
            logger.warning(f"  文本生成失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return ""

    def extract_prototype_bank(self, public_dataset, dataloader) -> Dict:
        """
        提取原型库：从公共数据集中提取中间层隐藏状态，使用K-means聚类
        
        Args:
            public_dataset: 公共数据集
            dataloader: 数据加载器
            
        Returns:
            Dict: 包含原型中心、聚类标签等信息
        """
        self.model.eval()
        all_hidden_states = []
        all_texts = []
        
        logger.info("开始提取中间层隐藏状态...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 准备输入
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 获取中间层输出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # 检查hidden_states是否存在
                if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                    logger.error("模型未返回hidden_states，请确保模型支持output_hidden_states=True")
                    raise ValueError("模型不支持输出hidden_states")
                
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
                
                # 提取指定层的隐藏状态
                hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
                logger.debug(f"使用第 {layer_idx} 层（共 {num_layers} 层）的隐藏状态")
                
                # 对序列维度求平均（或使用[CLS] token）
                if attention_mask is not None:
                    # 使用attention mask进行加权平均
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    hidden_states = hidden_states * mask_expanded
                    hidden_states = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    hidden_states = hidden_states.mean(dim=1)  # [batch, hidden_dim]
                
                all_hidden_states.append(hidden_states.cpu())
                
                # 保存文本（用于调试）
                if batch_idx < 5:  # 只保存前几个batch的文本
                    texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    all_texts.extend(texts)
                
                if batch_idx % 10 == 0:
                    logger.info(f"已处理 {batch_idx + 1} 个batch")
        
        # 合并所有隐藏状态
        all_hidden_states = torch.cat(all_hidden_states, dim=0)  # [num_samples, hidden_dim]
        logger.info(f"提取了 {all_hidden_states.shape[0]} 个样本的隐藏状态，维度: {all_hidden_states.shape[1]}")
        
        # K-means++聚类（FedKA方法）
        logger.info(f"开始K-means++聚类，目标聚类数: {self.num_prototypes}")
        hidden_np = all_hidden_states.numpy()
        
        # 如果样本数少于聚类数，调整聚类数
        actual_num_prototypes = min(self.num_prototypes, len(hidden_np))
        
        # 使用K-means++初始化
        centers, _ = kmeans_plusplus(hidden_np, n_clusters=actual_num_prototypes, random_state=42)
        kmeans = KMeans(n_clusters=actual_num_prototypes, init=centers, n_init=1, random_state=42)
        cluster_labels = kmeans.fit_predict(hidden_np)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        logger.info(f"K-means++聚类完成，生成了 {actual_num_prototypes} 个原型")
        
        # 计算每个聚类的样本数
        cluster_sizes = np.bincount(cluster_labels)
        
        # 软标签关联（FedKA方法）：对每个簇计算温度软化后的平均概率分布
        logger.info("开始计算软标签关联...")
        all_logits = []  # 需要收集logits用于软标签计算
        
        # 重新前向传播以获取logits（用于软标签）
        self.model.eval()
        soft_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits  # [batch, seq_len, vocab_size]
                    # 取最后一个token的logits（或平均）
                    if attention_mask is not None:
                        # 找到每个序列的最后一个有效token
                        seq_lengths = attention_mask.sum(dim=1) - 1
                        batch_logits = []
                        for i, length in enumerate(seq_lengths):
                            batch_logits.append(logits[i, length, :])
                        batch_logits = torch.stack(batch_logits)
                    else:
                        batch_logits = logits[:, -1, :]  # 最后一个token
                    
                    all_logits.append(batch_logits.cpu())
                
                if batch_idx % 10 == 0:
                    logger.info(f"已处理 {batch_idx + 1} 个batch用于软标签计算")
        
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)  # [num_samples, vocab_size]
            logger.info(f"收集了 {all_logits.shape[0]} 个样本的logits，词汇表大小: {all_logits.shape[1]}")
            
            # 对每个簇计算温度软化后的平均概率分布
            for m in range(actual_num_prototypes):
                cluster_mask = (cluster_labels == m)
                cluster_logits = all_logits[cluster_mask]  # [cluster_size, vocab_size]
                
                if len(cluster_logits) > 0:
                    # 温度软化
                    softmax_logits = F.softmax(cluster_logits / self.soft_label_temperature, dim=1)
                    # 平均得到软标签
                    soft_label = softmax_logits.mean(dim=0)  # [vocab_size]
                    soft_labels.append(soft_label.numpy())
                else:
                    # 空簇，使用均匀分布
                    vocab_size = all_logits.shape[1]
                    soft_labels.append(np.ones(vocab_size) / vocab_size)
            
            soft_labels = np.array(soft_labels)  # [num_prototypes, vocab_size]
            logger.info(f"✓ 软标签计算完成，形状: {soft_labels.shape}")
        else:
            logger.warning("无法获取logits，跳过软标签计算")
            soft_labels = None
        
        avg_proto_dist = self._compute_avg_pairwise_mse(cluster_centers)

        prototype_bank = {
            'prototypes': cluster_centers,  # [num_prototypes, hidden_dim]
            'cluster_labels': cluster_labels,  # [num_samples]
            'cluster_sizes': cluster_sizes,  # [num_prototypes]
            'soft_labels': soft_labels,  # [num_prototypes, vocab_size] 或 None
            'num_prototypes': actual_num_prototypes,
            'hidden_dim': all_hidden_states.shape[1],
            'avg_prototype_distance': avg_proto_dist
        }
        
        return prototype_bank

    def generate_knowledge_anchor(self, public_dataset, dataloader, task_description: str = None) -> Dict:
        """
        生成完整的知识锚点
        
        Args:
            public_dataset: 服务器端公共数据集
            dataloader: 数据加载器
            task_description: 任务描述（可选）
            
        Returns:
            Dict: 包含指令和原型库的知识锚点
        """
        logger.info("开始生成知识锚点...")
        
        # 1. 生成指令
        instructions = self.generate_instructions(public_dataset, task_description)
        
        # 2. 提取原型库
        prototype_bank = self.extract_prototype_bank(public_dataset, dataloader)
        
        knowledge_anchor = {
            'instructions': instructions,
            'prototype_bank': prototype_bank,
            'version': 1.0  # 用于版本控制
        }
        
        logger.info("知识锚点生成完成")
        return knowledge_anchor

    @staticmethod
    def _compute_avg_pairwise_mse(prototypes: torch.Tensor) -> float:
        """Compute mean pairwise MSE distance among prototypes."""
        if prototypes is None or len(prototypes) <= 1:
            return 0.0
        if not isinstance(prototypes, torch.Tensor):
            prototypes = torch.tensor(prototypes, dtype=torch.float32)
        proto = prototypes.float()
        diff = proto.unsqueeze(1) - proto.unsqueeze(0)
        mse = (diff * diff).mean(dim=-1)
        k = mse.shape[0]
        mask = ~torch.eye(k, dtype=torch.bool, device=mse.device)
        vals = mse[mask]
        if vals.numel() == 0:
            return 0.0
        return float(vals.mean().item())
