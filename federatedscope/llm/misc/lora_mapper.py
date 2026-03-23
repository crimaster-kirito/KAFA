"""
LoRA Mapper: 小模型LoRA映射到大模型LoRA
将客户端小模型（如DistilLlama）的LoRA参数映射到服务器大模型（Llama-7B）的LoRA层
"""
import torch
import torch.nn as nn
import copy
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class LoRAMapper:
    """
    LoRA参数映射器：小模型 -> 大模型
    支持多种映射策略：
    1. 直接映射（如果层名匹配）
    2. 多到一映射（多个小模型层映射到一个大模型层）
    3. 插值映射（基于层索引的插值）
    """
    def __init__(self, small_model_config, large_model_config, mapping_strategy='interpolate'):
        """
        Args:
            small_model_config: 小模型配置（用于确定层数）
            large_model_config: 大模型配置（用于确定层数）
            mapping_strategy: 映射策略 ('direct', 'many_to_one', 'interpolate')
        """
        self.small_model_config = small_model_config
        self.large_model_config = large_model_config
        self.mapping_strategy = mapping_strategy
        
        # 获取层数
        self.small_num_layers = getattr(small_model_config, 'num_hidden_layers', 
                                       getattr(small_model_config, 'n_layer', 12))
        self.large_num_layers = getattr(large_model_config, 'num_hidden_layers',
                                       getattr(large_model_config, 'n_layer', 32))
        
        # 构建映射关系
        self.layer_mapping = self._build_layer_mapping()
        
    def _build_layer_mapping(self) -> Dict[int, int]:
        """
        构建层映射关系
        
        Returns:
            Dict: {small_layer_idx: large_layer_idx}
        """
        mapping = {}
        
        if self.mapping_strategy == 'direct':
            # 直接映射：小模型的前N层映射到大模型的前N层
            for i in range(min(self.small_num_layers, self.large_num_layers)):
                mapping[i] = i
                
        elif self.mapping_strategy == 'many_to_one':
            # 多到一映射：多个小模型层映射到一个大模型层
            ratio = self.large_num_layers / self.small_num_layers
            for i in range(self.small_num_layers):
                large_idx = int(i * ratio)
                mapping[i] = min(large_idx, self.large_num_layers - 1)
                
        elif self.mapping_strategy == 'interpolate':
            # 插值映射：均匀分布映射
            for i in range(self.small_num_layers):
                # 将小模型层均匀映射到大模型层
                large_idx = int(i * (self.large_num_layers - 1) / (self.small_num_layers - 1)) \
                    if self.small_num_layers > 1 else 0
                mapping[i] = min(large_idx, self.large_num_layers - 1)
        else:
            raise ValueError(f"Unknown mapping strategy: {self.mapping_strategy}")
        
        logger.info(f"构建层映射: {self.mapping_strategy}, "
                   f"小模型{self.small_num_layers}层 -> 大模型{self.large_num_layers}层")
        return mapping

    def map_lora_weights(self, small_lora_state_dict: Dict, 
                        large_lora_state_dict: Optional[Dict] = None) -> Dict:
        """
        将小模型LoRA权重映射到大模型LoRA权重
        
        Args:
            small_lora_state_dict: 小模型LoRA state_dict
            large_lora_state_dict: 大模型LoRA state_dict（可选，用于初始化）
            
        Returns:
            Dict: 映射后的大模型LoRA state_dict
        """
        large_lora = {}
        
        if large_lora_state_dict is not None:
            # 如果提供了大模型LoRA，先复制
            large_lora = copy.deepcopy(large_lora_state_dict)
        
        # 提取小模型LoRA参数
        small_lora_params = self._extract_lora_params(small_lora_state_dict)
        
        # 按层映射
        for small_layer_idx, large_layer_idx in self.layer_mapping.items():
            # 查找小模型该层的LoRA参数
            small_layer_lora = self._get_layer_lora(small_lora_params, small_layer_idx)
            
            if small_layer_lora:
                # 映射到大模型层
                large_layer_lora = self._map_layer_lora(
                    small_layer_lora, 
                    small_layer_idx, 
                    large_layer_idx
                )
                
                # 更新大模型LoRA字典
                for key, value in large_layer_lora.items():
                    large_lora[key] = value
        
        return large_lora

    def _extract_lora_params(self, state_dict: Dict) -> Dict:
        """提取LoRA参数（过滤非LoRA参数）"""
        lora_params = {}
        for key, value in state_dict.items():
            if 'lora' in key.lower() or 'adapter' in key.lower():
                lora_params[key] = value
        return lora_params

    def _get_layer_lora(self, lora_params: Dict, layer_idx: int) -> Dict:
        """获取指定层的LoRA参数"""
        layer_lora = {}
        for key, value in lora_params.items():
            # 查找包含层索引的key（如 "layers.5.lora_A"）
            if f'.{layer_idx}.' in key or f'.{layer_idx}_' in key:
                layer_lora[key] = value
        return layer_lora

    def _map_layer_lora(self, small_layer_lora: Dict, 
                        small_layer_idx: int, 
                        large_layer_idx: int) -> Dict:
        """
        映射单层LoRA参数
        
        Args:
            small_layer_lora: 小模型层的LoRA参数
            small_layer_idx: 小模型层索引
            large_layer_idx: 大模型层索引
            
        Returns:
            Dict: 映射后的大模型层LoRA参数
        """
        large_layer_lora = {}
        
        for key, value in small_layer_lora.items():
            # 替换层索引
            new_key = key.replace(f'.{small_layer_idx}.', f'.{large_layer_idx}.')
            new_key = new_key.replace(f'.{small_layer_idx}_', f'.{large_layer_idx}_')
            
            # 如果维度不匹配，需要调整（这里简化处理，实际可能需要更复杂的映射）
            if isinstance(value, torch.Tensor):
                # 如果维度匹配，直接使用
                large_layer_lora[new_key] = value
            else:
                large_layer_lora[new_key] = value
        
        return large_layer_lora

    def aggregate_mapped_loras(self, mapped_loras: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """
        聚合多个映射后的LoRA参数
        
        Args:
            mapped_loras: 多个映射后的LoRA state_dict列表
            weights: 权重列表（可选，默认均匀权重）
            
        Returns:
            Dict: 聚合后的LoRA state_dict
        """
        if weights is None:
            weights = [1.0 / len(mapped_loras)] * len(mapped_loras)
        
        # 获取所有key
        all_keys = set()
        for lora_dict in mapped_loras:
            all_keys.update(lora_dict.keys())
        
        # 加权聚合
        aggregated = {}
        for key in all_keys:
            values = []
            for i, lora_dict in enumerate(mapped_loras):
                if key in lora_dict:
                    values.append(lora_dict[key] * weights[i])
            
            if values:
                aggregated[key] = sum(values)
        
        return aggregated


def create_lora_mapper(small_model, large_model, mapping_strategy='interpolate'):
    """
    创建LoRA映射器的便捷函数
    
    Args:
        small_model: 小模型（用于获取config）
        large_model: 大模型（用于获取config）
        mapping_strategy: 映射策略
        
    Returns:
        LoRAMapper实例
    """
    small_config = small_model.config if hasattr(small_model, 'config') else None
    large_config = large_model.config if hasattr(large_model, 'config') else None
    
    if small_config is None or large_config is None:
        raise ValueError("模型必须具有config属性")
    
    return LoRAMapper(small_config, large_config, mapping_strategy)

