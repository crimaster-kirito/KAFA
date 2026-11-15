"""
ProjectionMapper: 服务器端投影LoRA映射器
实现大LoRA ↔ 小LoRA的双向投影，所有投影逻辑集中在服务器端

核心功能：
1. 下投影（down_project）：大LoRA → 小LoRA（用于broadcast）
2. 上投影（up_project）：小LoRA → 大LoRA（用于aggregate）

GQA / MHA：
- q_proj 的 lora_B 输出维为 hidden_size，使用 P（small_hidden × large_hidden）
- v_proj / k_proj 在 GQA 下输出维为 num_kv_heads * head_dim，常与 hidden 不同；
  此时使用独立的 P_kv（small_kv_out × large_kv_out），避免错误的维度投影。
"""
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def infer_kv_out_dim(model_config: Any) -> int:
    """
    从 HuggingFace 风格 config 推断单头注意力中 K/V 线性层输出维度
    （num_key_value_heads * head_dim）。
    """
    hidden = getattr(model_config, 'hidden_size', None)
    n_heads = getattr(model_config, 'num_attention_heads', None)
    if hidden is None or not n_heads:
        return int(hidden) if hidden is not None else 0
    head_dim = getattr(model_config, 'head_dim', None)
    if head_dim is None:
        head_dim = hidden // n_heads
    n_kv = getattr(model_config, 'num_key_value_heads', n_heads)
    return int(n_kv * int(head_dim))


def _is_kv_proj_lora_key(key: str) -> bool:
    return '.v_proj.' in key or '.k_proj.' in key


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
                 q_strategy: str = 'transpose',
                 large_kv_out: Optional[int] = None,
                 small_kv_out: Optional[int] = None):
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
            large_kv_out: 大模型 K/V 投影输出维；默认等于 large_hidden（MHA）
            small_kv_out: 小模型 K/V 投影输出维；默认等于 small_hidden（MHA）
        """
        super(ProjectionMapper, self).__init__()
        self.small_hidden = small_hidden
        self.large_hidden = large_hidden
        self.rank = rank
        self.q_strategy = q_strategy
        self.large_kv_out = int(
            large_kv_out) if large_kv_out is not None else large_hidden
        self.small_kv_out = int(
            small_kv_out) if small_kv_out is not None else small_hidden
        self.use_kv_lora_proj = (self.large_kv_out != self.large_hidden) or (
            self.small_kv_out != self.small_hidden)

        # 投影矩阵P：small_hidden × large_hidden
        if learnable:
            # 可学习参数，使用Xavier初始化
            self.P = nn.Parameter(torch.randn(small_hidden, large_hidden) * 0.02)
        else:
            # 固定矩阵，使用随机初始化
            self.register_buffer('P', torch.randn(small_hidden, large_hidden) * 0.02)

        # v_proj / k_proj 的 lora_B：large_kv_out → small_kv_out（GQA 时与 hidden 不同）
        if self.use_kv_lora_proj:
            if learnable:
                self.P_kv = nn.Parameter(
                    torch.randn(self.small_kv_out, self.large_kv_out) * 0.02)
            else:
                self.register_buffer(
                    'P_kv',
                    torch.randn(self.small_kv_out, self.large_kv_out) * 0.02)
        else:
            self.register_parameter('P_kv', None)

        # 投影矩阵Q：根据策略设置
        if q_strategy == 'learnable':
            self.Q = nn.Parameter(torch.randn(large_hidden, small_hidden) * 0.02)
            if self.use_kv_lora_proj:
                self.Q_kv = nn.Parameter(
                    torch.randn(self.large_kv_out, self.small_kv_out) * 0.02)
            else:
                self.register_parameter('Q_kv', None)
        elif q_strategy == 'pseudo_inverse':
            # Q将在forward时计算，这里不注册参数
            self.Q = None
            self.Q_kv = None
        else:  # 'transpose'
            # Q = P.t()，不需要单独存储
            self.Q = None
            self.Q_kv = None

        logger.info(
            f"初始化ProjectionMapper: small_hidden={small_hidden}, "
            f"large_hidden={large_hidden}, rank={rank}, "
            f"learnable={learnable}, q_strategy={q_strategy}, "
            f"large_kv_out={self.large_kv_out}, small_kv_out={self.small_kv_out}, "
            f"use_kv_lora_proj={self.use_kv_lora_proj}")
    
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

    def get_Q_kv(self) -> Optional[torch.Tensor]:
        """K/V lora_B 上投影矩阵 (large_kv_out × small_kv_out)"""
        if not self.use_kv_lora_proj or self.P_kv is None:
            return None
        if self.q_strategy == 'transpose':
            return self.P_kv.t()
        if self.q_strategy == 'pseudo_inverse':
            return torch.pinverse(self.P_kv)
        if self.q_strategy == 'learnable':
            return self.Q_kv
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
                # lora_A: (rank, in_features)；注意力线性层 in_features == hidden_size
                in_dim = value.shape[-1]
                if in_dim != self.large_hidden:
                    logger.warning(
                        f"下投影 lora_A 输入维 {in_dim} != large_hidden={self.large_hidden} "
                        f"(key={key})，仍按 P 投影，若形状不匹配将失败")
                if value.dim() == 2:
                    projected = value @ self.P.t()
                else:
                    projected = torch.matmul(value, self.P.t())
                small_lora_state_dict[key] = projected

            elif 'lora_B' in key:
                # lora_B: (out_features, rank)；GQA 下 v/k 的 out 为 kv_out
                if value.dim() != 2:
                    projected = torch.matmul(self.P, value)
                elif (self.use_kv_lora_proj and self.P_kv is not None
                      and _is_kv_proj_lora_key(key)):
                    out_d = value.shape[0]
                    if out_d != self.large_kv_out:
                        logger.warning(
                            f"下投影 K/V lora_B 行维 {out_d} != large_kv_out="
                            f"{self.large_kv_out} (key={key})，跳过 P_kv，尝试 P")
                        projected = self.P @ value
                    else:
                        projected = self.P_kv @ value
                else:
                    projected = self.P @ value
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
        Q_kv = self.get_Q_kv()

        for key, value in small_lora_state_dict.items():
            if not isinstance(value, torch.Tensor):
                # 非tensor参数直接复制
                large_lora_state_dict[key] = value
                continue

            # 判断是lora_A还是lora_B
            if 'lora_A' in key:
                in_dim = value.shape[-1]
                if in_dim != self.small_hidden:
                    logger.warning(
                        f"上投影 lora_A 输入维 {in_dim} != small_hidden="
                        f"{self.small_hidden} (key={key})")
                if value.dim() == 2:
                    projected = value @ Q.t()
                else:
                    projected = torch.matmul(value, Q.t())
                large_lora_state_dict[key] = projected

            elif 'lora_B' in key:
                if value.dim() != 2:
                    projected = torch.matmul(Q, value)
                elif (self.use_kv_lora_proj and Q_kv is not None
                      and _is_kv_proj_lora_key(key)):
                    out_d = value.shape[0]
                    if out_d != self.small_kv_out:
                        logger.warning(
                            f"上投影 K/V lora_B 行维 {out_d} != small_kv_out="
                            f"{self.small_kv_out} (key={key})，回退 Q")
                        projected = Q @ value
                    else:
                        projected = Q_kv @ value
                else:
                    projected = Q @ value
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
                             device: str = 'cpu',
                             large_kv_out: Optional[int] = None,
                             small_kv_out: Optional[int] = None) -> ProjectionMapper:
    """
    创建投影映射器的便捷函数
    
    Args:
        small_hidden: 小模型hidden_size
        large_hidden: 大模型hidden_size
        rank: LoRA rank
        learnable: 是否可学习
        q_strategy: Q矩阵策略
        device: 设备
        large_kv_out: 大模型 K/V 线性层输出维；默认 large_hidden
        small_kv_out: 小模型 K/V 线性层输出维；默认 small_hidden
    
    Returns:
        ProjectionMapper实例
    """
    mapper = ProjectionMapper(
        small_hidden=small_hidden,
        large_hidden=large_hidden,
        rank=rank,
        learnable=learnable,
        q_strategy=q_strategy,
        large_kv_out=large_kv_out,
        small_kv_out=small_kv_out)
    mapper = mapper.to(device)
    return mapper

