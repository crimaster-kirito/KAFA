"""
Knowledge Anchor Aggregator for Server
服务器端知识锚点聚合器：
1. 聚合客户端上传的LoRA参数（需要从小模型映射到大模型）
2. 整合客户端上传的增量bank
3. 更新知识锚点（原型库）

注意：这是基于预训练模型的微调场景，使用LoRA等参数高效微调方法

新增功能：服务器端投影LoRA
- 服务器维护全局大LoRA（hidden_size=4096）
- 通过投影矩阵P/Q实现大LoRA ↔ 小LoRA的双向转换
- 所有投影逻辑集中在服务器端，客户端无需修改
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import logging
from typing import Dict, List, Optional

from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.llm.misc.lora_mapper import LoRAMapper, create_lora_mapper
from federatedscope.llm.misc.projection_mapper import ProjectionMapper, create_projection_mapper

logger = logging.getLogger(__name__)


class KnowledgeAnchorAggregator(Aggregator):
    """
    知识锚点聚合器
    处理LoRA聚合和增量bank整合
    """
    def __init__(self, model=None, device='cpu', config=None, small_model=None):
        """
        Args:
            model: 服务器端大模型（Llama-7B）
            device: 设备
            config: 配置对象
            small_model: 客户端小模型（用于LoRA映射，可选）
        """
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        
        # 投影映射器（服务器端投影LoRA）
        self.projection_mapper = None
        use_projection = getattr(config.llm.knowledge_anchor, 'use_projection_lora', True)
        if use_projection and model is not None:
            try:
                # 从配置或模型获取hidden_size
                small_hidden = getattr(config.llm.knowledge_anchor, 'small_hidden', 2048)
                large_hidden = getattr(config.llm.knowledge_anchor, 'large_hidden', None)
                if large_hidden is None:
                    # 尝试从模型config获取
                    if hasattr(model, 'config'):
                        large_hidden = getattr(model.config, 'hidden_size', 4096)
                    else:
                        large_hidden = 4096
                
                rank = getattr(config.train, 'lora_rank', 8)
                learnable = getattr(config.llm.knowledge_anchor, 'projection_learnable', True)
                q_strategy = getattr(config.llm.knowledge_anchor, 'q_strategy', 'transpose')
                
                self.projection_mapper = create_projection_mapper(
                    small_hidden=small_hidden,
                    large_hidden=large_hidden,
                    rank=rank,
                    learnable=learnable,
                    q_strategy=q_strategy,
                    device=device
                )
                logger.info(f"✓ 已初始化投影映射器: small_hidden={small_hidden}, "
                           f"large_hidden={large_hidden}, rank={rank}")
            except Exception as e:
                logger.warning(f"创建投影映射器失败: {e}，将使用传统LoRA映射")
                self.projection_mapper = None
        
        # LoRA映射器（传统方法，作为回退）
        self.lora_mapper = None
        if self.projection_mapper is None and small_model is not None and model is not None:
            try:
                self.lora_mapper = create_lora_mapper(
                    small_model, 
                    model, 
                    mapping_strategy=getattr(config.llm.knowledge_anchor, 'lora_mapping_strategy', 'interpolate')
                )
            except Exception as e:
                logger.warning(f"创建LoRA映射器失败: {e}")
        
        # 全局大LoRA state_dict（服务器维护）
        self.global_large_lora = None
        
        # 知识锚点相关
        self.knowledge_anchor = None
        self.incremental_bank_buffer = []  # 临时存储客户端上传的增量bank

    def set_knowledge_anchor(self, knowledge_anchor: Dict):
        """设置知识锚点"""
        self.knowledge_anchor = knowledge_anchor
    
    def get_global_large_lora(self) -> Dict[str, torch.Tensor]:
        """
        获取服务器全局大LoRA state_dict
        
        Returns:
            Dict: 大模型LoRA state_dict
        """
        if self.global_large_lora is not None:
            return self.global_large_lora
        
        # 从PEFT模型获取LoRA参数
        if self.model is not None:
            try:
                from peft import PeftModel
                if isinstance(self.model, PeftModel):
                    # 获取PEFT state_dict（只包含LoRA参数）
                    self.global_large_lora = self.model.get_peft_state_dict()
                    logger.info(f"✓ 从PEFT模型获取全局大LoRA，包含 {len(self.global_large_lora)} 个参数")
                else:
                    # 如果不是PEFT模型，尝试从state_dict中提取LoRA参数
                    full_state_dict = self.model.state_dict()
                    self.global_large_lora = self._extract_lora_params(full_state_dict)
                    logger.info(f"✓ 从模型state_dict提取LoRA参数，包含 {len(self.global_large_lora)} 个参数")
            except Exception as e:
                logger.warning(f"获取全局大LoRA失败: {e}")
                self.global_large_lora = {}
        
        return self.global_large_lora if self.global_large_lora is not None else {}
    
    def broadcast_small_lora(self) -> Dict[str, torch.Tensor]:
        """
        广播小LoRA（下投影：大LoRA → 小LoRA）
        
        在服务器broadcast前调用，将全局大LoRA投影为小LoRA并返回
        
        Returns:
            Dict: 投影后的小模型LoRA state_dict（用于broadcast给客户端）
        """
        if self.projection_mapper is None:
            logger.warning("投影映射器未初始化，无法进行下投影")
            return {}
        
        # 获取全局大LoRA
        large_lora = self.get_global_large_lora()
        if not large_lora:
            logger.warning("全局大LoRA为空，无法进行下投影")
            return {}
        
        # 下投影：大LoRA → 小LoRA
        small_lora = self.projection_mapper.down_project(large_lora)
        logger.info(f"✓ 下投影完成: 大LoRA({len(large_lora)}个参数) -> 小LoRA({len(small_lora)}个参数)")
        
        return small_lora

    def aggregate(self, agg_info):
        """
        聚合客户端更新
        
        Args:
            agg_info: 包含客户端反馈的字典
                - client_feedback: List of (sample_size, model_params, incremental_bank)
                - recover_fun: 恢复函数（用于秘密共享，可选）
                
        Returns:
            Dict: 聚合后的模型参数
        """
        models = agg_info["client_feedback"]
        recover_fun = agg_info.get('recover_fun', None)
        
        # 分离LoRA参数和增量bank
        lora_updates = []
        incremental_banks = []
        client_weights = []
        
        for i, feedback in enumerate(models):
            if isinstance(feedback, tuple):
                if len(feedback) == 3:
                    # (sample_size, model_params, incremental_bank)
                    sample_size, model_params, incremental_bank = feedback
                elif len(feedback) == 2:
                    # (sample_size, model_params)
                    sample_size, model_params = feedback
                    incremental_bank = None
                else:
                    raise ValueError(f"Unexpected feedback format: {len(feedback)} elements")
            else:
                sample_size = 1
                model_params = feedback
                incremental_bank = None
            
            # 提取LoRA参数
            lora_params = self._extract_lora_params(model_params)
            
            # 如果需要映射，进行映射
            if self.lora_mapper is not None:
                try:
                    mapped_lora = self.lora_mapper.map_lora_weights(lora_params)
                    lora_updates.append((sample_size, mapped_lora))
                except Exception as e:
                    logger.warning(f"LoRA映射失败，使用原始参数: {e}")
                    lora_updates.append((sample_size, lora_params))
            else:
                lora_updates.append((sample_size, lora_params))
            
            # 收集增量bank
            if incremental_bank is not None:
                incremental_banks.append(incremental_bank)
            
            client_weights.append(sample_size)
        
        # 聚合LoRA参数（此时是小LoRA）
        aggregated_small_lora = self._aggregate_loras(lora_updates, client_weights, recover_fun)
        
        # 整合增量bank
        if incremental_banks:
            self._integrate_incremental_banks(incremental_banks)
        
        # 如果使用投影映射，进行上投影并更新全局大LoRA
        if self.projection_mapper is not None and aggregated_small_lora:
            try:
                # 上投影：小LoRA → 大LoRA
                large_lora_delta = self.projection_mapper.up_project(aggregated_small_lora)
                logger.info(f"✓ 上投影完成: 小LoRA({len(aggregated_small_lora)}个参数) -> "
                           f"大LoRA增量({len(large_lora_delta)}个参数)")
                
                # 更新全局大LoRA
                self._update_global_large_lora(large_lora_delta)
                
                # 返回更新后的大LoRA（用于更新服务器模型）
                return self.get_global_large_lora()
            except Exception as e:
                logger.warning(f"上投影失败: {e}，返回聚合后的小LoRA")
                return aggregated_small_lora
        else:
            # 不使用投影映射，直接返回聚合结果
            return aggregated_small_lora

    def _extract_lora_params(self, model_params: Dict) -> Dict:
        """提取LoRA参数"""
        lora_params = {}
        for key, value in model_params.items():
            if 'lora' in key.lower() or 'adapter' in key.lower():
                lora_params[key] = value
        return lora_params

    def _aggregate_loras(self, lora_updates: List, client_weights: List, recover_fun=None) -> Dict:
        """
        聚合LoRA参数（加权平均）
        
        Args:
            lora_updates: List of (sample_size, lora_params)
            client_weights: 客户端权重列表
            recover_fun: 恢复函数（可选）
            
        Returns:
            Dict: 聚合后的LoRA参数
        """
        total_weight = sum(client_weights)
        
        # 获取所有key
        all_keys = set()
        for _, lora_params in lora_updates:
            all_keys.update(lora_params.keys())
        
        # 加权聚合
        aggregated = {}
        for key in all_keys:
            weighted_sum = None
            for i, (sample_size, lora_params) in enumerate(lora_updates):
                if key in lora_params:
                    weight = client_weights[i] / total_weight if total_weight > 0 else 1.0 / len(lora_updates)
                    param = param2tensor(lora_params[key])
                    
                    if recover_fun:
                        param = recover_fun(param)
                    
                    if weighted_sum is None:
                        weighted_sum = param * weight
                    else:
                        weighted_sum += param * weight
            
            if weighted_sum is not None:
                aggregated[key] = weighted_sum
        
        return aggregated

    def _integrate_incremental_banks(self, incremental_banks: List[List[Dict]]):
        """
        整合客户端上传的增量bank
        
        Args:
            incremental_banks: List of incremental banks from clients
        """
        # 收集所有新原型
        all_new_prototypes = []
        for bank in incremental_banks:
            if isinstance(bank, list):
                all_new_prototypes.extend(bank)
        
        if not all_new_prototypes:
            return
        
        logger.info(f"收到 {len(all_new_prototypes)} 个新原型，开始整合...")
        
        # 提取原型向量
        prototype_vectors = []
        for entry in all_new_prototypes:
            if isinstance(entry, dict) and 'prototype' in entry:
                prototype_vectors.append(entry['prototype'])
            elif isinstance(entry, torch.Tensor):
                prototype_vectors.append(entry)
        
        if not prototype_vectors:
            return
        
        # 转换为tensor
        prototype_tensor = torch.stack(prototype_vectors)  # [num_new, hidden_dim]
        
        # 与现有原型库合并
        if self.knowledge_anchor and 'prototype_bank' in self.knowledge_anchor:
            existing_prototypes = self.knowledge_anchor['prototype_bank'].get('prototypes', None)
            
            if existing_prototypes is not None:
                # 合并原型
                combined_prototypes = torch.cat([existing_prototypes, prototype_tensor], dim=0)
                
                # 重新聚类（可选：如果原型数量过多，可以重新聚类）
                max_prototypes = getattr(self.cfg.llm.knowledge_anchor, 'max_prototypes', 100)
                if len(combined_prototypes) > max_prototypes:
                    logger.info(f"原型数量超过{max_prototypes}，进行重新聚类...")
                    combined_prototypes = self._recluster_prototypes(combined_prototypes, max_prototypes)
                
                # 更新知识锚点
                self.knowledge_anchor['prototype_bank']['prototypes'] = combined_prototypes
                self.knowledge_anchor['prototype_bank']['num_prototypes'] = len(combined_prototypes)
                
                logger.info(f"原型库已更新，当前原型数: {len(combined_prototypes)}")
            else:
                # 如果没有现有原型，直接使用新原型
                self.knowledge_anchor['prototype_bank']['prototypes'] = prototype_tensor
                self.knowledge_anchor['prototype_bank']['num_prototypes'] = len(prototype_tensor)

    def _recluster_prototypes(self, prototypes: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        重新聚类原型（当原型数量过多时）
        
        Args:
            prototypes: 原型tensor [num_prototypes, hidden_dim]
            num_clusters: 目标聚类数
            
        Returns:
            torch.Tensor: 聚类后的原型中心 [num_clusters, hidden_dim]
        """
        prototypes_np = prototypes.numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(prototypes_np)
        
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return cluster_centers

    def _update_global_large_lora(self, large_lora_delta: Dict[str, torch.Tensor]):
        """
        更新全局大LoRA（累加增量）
        
        Args:
            large_lora_delta: 大LoRA增量（从上投影得到）
        """
        # 获取当前全局大LoRA
        current_large_lora = self.get_global_large_lora()
        
        # 更新策略：累加增量（可以添加学习率或动量）
        lr = getattr(self.cfg.llm.knowledge_anchor, 'lora_update_lr', 1.0)
        
        for key, delta in large_lora_delta.items():
            if key in current_large_lora:
                # 累加增量
                current_large_lora[key] = current_large_lora[key] + lr * delta
            else:
                # 新参数，直接添加
                current_large_lora[key] = lr * delta
        
        # 更新全局大LoRA
        self.global_large_lora = current_large_lora
        
        # 如果模型是PEFT模型，同步更新模型参数
        if self.model is not None:
            try:
                from peft import PeftModel
                if isinstance(self.model, PeftModel):
                    # 只更新LoRA参数，不影响base model
                    self.model.load_state_dict(current_large_lora, strict=False)
                    logger.debug("✓ 已同步更新PEFT模型的LoRA参数")
            except Exception as e:
                logger.warning(f"同步更新模型参数失败: {e}")
    
    def update(self, model_parameters):
        """
        更新模型参数
        
        如果使用投影映射，model_parameters应该是大LoRA state_dict
        否则，直接加载到模型
        """
        if self.model is not None:
            if self.projection_mapper is not None:
                # 使用投影映射时，更新全局大LoRA
                if isinstance(model_parameters, dict):
                    # 检查是否是大LoRA参数
                    is_large_lora = any('lora' in k.lower() for k in model_parameters.keys())
                    if is_large_lora:
                        self.global_large_lora = model_parameters
                        # 同步到PEFT模型
                        try:
                            from peft import PeftModel
                            if isinstance(self.model, PeftModel):
                                self.model.load_state_dict(model_parameters, strict=False)
                        except:
                            pass
                    else:
                        # 不是LoRA参数，正常加载
                        self.model.load_state_dict(model_parameters, strict=False)
                else:
                    self.model.load_state_dict(model_parameters, strict=False)
            else:
                # 不使用投影映射，正常加载
                self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        """保存模型"""
        assert self.model is not None
        
        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        """加载模型"""
        assert self.model is not None
        
        import os
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt.get('cur_round', -1)
        else:
            raise ValueError(f"The file {path} does NOT exist")
