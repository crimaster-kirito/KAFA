"""
ProjectionMapper: 服务器端投影LoRA映射器
实现大LoRA ↔ 小LoRA的双向投影，所有投影逻辑集中在服务器端

核心功能：
1. 下投影（down_project）：大LoRA → 小LoRA（用于broadcast）
2. 上投影（up_project）：小LoRA → 大LoRA（用于aggregate）

假设：
- 服务器：大模型（hidden_size=4096，例如Llama-2-7B）
- 客户端：小模型（hidden_size=2048，例如TinyLlama）
- LoRA rank=8，target_modules=["q_proj", "v_proj"]
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProjectionMapper(nn.Module):
    """
    投影映射器：实现大LoRA ↔ 小LoRA的双向投影
    
    投影公式：
    - 下投影（大→小）：
      small_lora_A = P @ large_lora_A          # (rank, small_hidden) = (small_hidden, large_hidden) @ (rank, large_hidden)
      small_lora_B = large_lora_B @ P.t()       # (small_hidden, rank) = (large_hidden, rank) @ (large_hidden, small_hidden)
    
    - 上投影（小→大）：
      large_lora_A = Q @ small_lora_A          # (rank, large_hidden) = (large_hidden, small_hidden) @ (rank, small_hidden)
      large_lora_B = small_lora_B @ Q.t()       # (large_hidden, rank) = (small_hidden, rank) @ (small_hidden, large_hidden)
    
    其中 Q 可以是 P 的转置、伪逆，或独立可学习参数
    """
    def __init__(self, 
                 small_hidden: int = 2048,
                 large_hidden: int = 4096,
                 rank: int = 8,
                 learnable: bool = True,
                 q_strategy: str = 'transpose'):
        """
        Args:
            small_hidden: 小模型hidden_size
            large_hidden: 大模型hidden_size
            rank: LoRA rank
            learnable: 是否将P设为可学习参数（默认True）
            q_strategy: Q矩阵的生成策略
                - 'transpose': Q = P.t()（默认，推荐）
                - 'pseudo_inverse': Q = pinverse(P)
                - 'learnable': Q作为独立可学习参数
        """
        super(ProjectionMapper, self).__init__()
        self.small_hidden = small_hidden
        self.large_hidden = large_hidden
        self.rank = rank
        self.q_strategy = q_strategy
        
        # 投影矩阵P：small_hidden × large_hidden
        if learnable:
            # 可学习参数，使用Xavier初始化
            self.P = nn.Parameter(torch.randn(small_hidden, large_hidden) * 0.02)
        else:
            # 固定矩阵，使用随机初始化
            self.register_buffer('P', torch.randn(small_hidden, large_hidden) * 0.02)
        
        # 投影矩阵Q：根据策略设置
        if q_strategy == 'learnable':
            self.Q = nn.Parameter(torch.randn(large_hidden, small_hidden) * 0.02)
        elif q_strategy == 'pseudo_inverse':
            # Q将在forward时计算，这里不注册参数
            self.Q = None
        else:  # 'transpose'
            # Q = P.t()，不需要单独存储
            self.Q = None
        
        logger.info(f"初始化ProjectionMapper: small_hidden={small_hidden}, "
                   f"large_hidden={large_hidden}, rank={rank}, "
                   f"learnable={learnable}, q_strategy={q_strategy}")
    
    def get_Q(self) -> torch.Tensor:
        """获取Q矩阵"""
        if self.q_strategy == 'transpose':
            return self.P.t()
        elif self.q_strategy == 'pseudo_inverse':
            # 计算伪逆
            return torch.pinverse(self.P)
        elif self.q_strategy == 'learnable':
            return self.Q
        else:
            raise ValueError(f"Unknown q_strategy: {self.q_strategy}")
    
    def down_project(self, large_lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        下投影：大LoRA → 小LoRA
        
        Args:
            large_lora_state_dict: 大模型LoRA state_dict
                - 格式：{"base_model.model.model.layers.{i}.self_attn.{module}.lora_A.default.weight": tensor, ...}
                - lora_A shape: (rank, large_hidden)
                - lora_B shape: (large_hidden, rank)
        
        Returns:
            Dict: 投影后的小模型LoRA state_dict
                - lora_A shape: (rank, small_hidden)
                - lora_B shape: (small_hidden, rank)
        """
        small_lora_state_dict = {}
        
        for key, value in large_lora_state_dict.items():
            if not isinstance(value, torch.Tensor):
                # 非tensor参数直接复制
                small_lora_state_dict[key] = value
                continue
            
            # 判断是lora_A还是lora_B
            if 'lora_A' in key:
                # lora_A: (rank, large_hidden) -> (rank, small_hidden)
                # small_lora_A = P @ large_lora_A
                # shape: (small_hidden, large_hidden) @ (rank, large_hidden).t() = (small_hidden, rank)
                # 需要转置：large_lora_A.t() @ P.t() = (rank, small_hidden)
                if value.dim() == 2:
                    # value shape: (rank, large_hidden)
                    projected = value @ self.P.t()  # (rank, large_hidden) @ (large_hidden, small_hidden) = (rank, small_hidden)
                else:
                    # 处理其他维度情况
                    projected = torch.matmul(value, self.P.t())
                small_lora_state_dict[key] = projected
                
            elif 'lora_B' in key:
                # lora_B: (large_hidden, rank) -> (small_hidden, rank)
                # small_lora_B = large_lora_B @ P.t()
                # shape: (large_hidden, rank) @ (large_hidden, small_hidden) = (small_hidden, rank)
                if value.dim() == 2:
                    # value shape: (large_hidden, rank)
                    projected = self.P @ value  # (small_hidden, large_hidden) @ (large_hidden, rank) = (small_hidden, rank)
                else:
                    projected = torch.matmul(self.P, value)
                small_lora_state_dict[key] = projected
            else:
                # 其他参数（如lora_embedding等）直接复制
                small_lora_state_dict[key] = value
        
        logger.debug(f"下投影完成: {len(large_lora_state_dict)} -> {len(small_lora_state_dict)} 个参数")
        return small_lora_state_dict
    
    def up_project(self, small_lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        上投影：小LoRA → 大LoRA
        
        Args:
            small_lora_state_dict: 小模型LoRA state_dict
                - lora_A shape: (rank, small_hidden)
                - lora_B shape: (small_hidden, rank)
        
        Returns:
            Dict: 投影后的大模型LoRA state_dict
                - lora_A shape: (rank, large_hidden)
                - lora_B shape: (large_hidden, rank)
        """
        large_lora_state_dict = {}
        Q = self.get_Q()  # (large_hidden, small_hidden)
        
        for key, value in small_lora_state_dict.items():
            if not isinstance(value, torch.Tensor):
                # 非tensor参数直接复制
                large_lora_state_dict[key] = value
                continue
            
            # 判断是lora_A还是lora_B
            if 'lora_A' in key:
                # lora_A: (rank, small_hidden) -> (rank, large_hidden)
                # large_lora_A = Q @ small_lora_A
                # shape: (large_hidden, small_hidden) @ (rank, small_hidden).t() = (large_hidden, rank)
                # 需要转置：small_lora_A.t() @ Q.t() = (rank, large_hidden)
                if value.dim() == 2:
                    # value shape: (rank, small_hidden)
                    projected = value @ Q.t()  # (rank, small_hidden) @ (small_hidden, large_hidden) = (rank, large_hidden)
                else:
                    projected = torch.matmul(value, Q.t())
                large_lora_state_dict[key] = projected
                
            elif 'lora_B' in key:
                # lora_B: (small_hidden, rank) -> (large_hidden, rank)
                # large_lora_B = small_lora_B @ Q.t()
                # shape: (small_hidden, rank) @ (small_hidden, large_hidden) = (large_hidden, rank)
                if value.dim() == 2:
                    # value shape: (small_hidden, rank)
                    projected = Q @ value  # (large_hidden, small_hidden) @ (small_hidden, rank) = (large_hidden, rank)
                else:
                    projected = torch.matmul(Q, value)
                large_lora_state_dict[key] = projected
            else:
                # 其他参数直接复制
                large_lora_state_dict[key] = value
        
        logger.debug(f"上投影完成: {len(small_lora_state_dict)} -> {len(large_lora_state_dict)} 个参数")
        return large_lora_state_dict
    
    def forward(self, x):
        """前向传播（用于训练投影矩阵）"""
        # 如果需要对投影矩阵进行端到端训练，可以实现此方法
        # 当前主要用于推理，所以forward方法可以留空或实现特定训练逻辑
        return x


def create_projection_mapper(small_hidden: int = 2048,
                             large_hidden: int = 4096,
                             rank: int = 8,
                             learnable: bool = True,
                             q_strategy: str = 'transpose',
                             device: str = 'cpu') -> ProjectionMapper:
    """
    创建投影映射器的便捷函数
    
    Args:
        small_hidden: 小模型hidden_size
        large_hidden: 大模型hidden_size
        rank: LoRA rank
        learnable: 是否可学习
        q_strategy: Q矩阵策略
        device: 设备
    
    Returns:
        ProjectionMapper实例
    """
    mapper = ProjectionMapper(
        small_hidden=small_hidden,
        large_hidden=large_hidden,
        rank=rank,
        learnable=learnable,
        q_strategy=q_strategy
    )
    mapper = mapper.to(device)
    return mapper

