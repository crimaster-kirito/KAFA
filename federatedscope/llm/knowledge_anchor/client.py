"""
Knowledge Anchor Client Worker
客户端实现：
1. 接收服务器下发的知识锚点
2. 使用小模型（DistilLlama）进行微调
3. 上传增量bank和LoRA参数
"""
import os
import logging
import torch
from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.model_builder import get_model

logger = logging.getLogger(__name__)


class KnowledgeAnchorClient(Client):
    """
    知识锚点客户端
    使用小模型（如DistilLlama）进行微调
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        # 客户端使用小模型
        # 如果配置中指定了客户端模型，使用它；否则使用服务器模型（不推荐）
        client_model_type = getattr(config.llm.knowledge_anchor, 'client_model_type', None)
        
        if client_model_type and client_model_type.strip():
            logger.info("=" * 80)
            logger.info(f"客户端{ID}开始加载小模型...")
            logger.info(f"配置的模型路径: {client_model_type}")
            
            # 解析模型路径
            model_path_parts = client_model_type.split('@')
            model_path = model_path_parts[0]
            model_hub = model_path_parts[1] if len(model_path_parts) > 1 else 'huggingface_llm'
            
            # 检查路径是否存在（支持绝对路径和相对路径）
            abs_model_path = os.path.abspath(model_path)
            if os.path.exists(model_path):
                logger.info(f"✓ 检测到本地模型路径: {model_path}")
                logger.info(f"  绝对路径: {abs_model_path}")
            elif os.path.exists(abs_model_path):
                logger.info(f"✓ 检测到本地模型路径（绝对路径）: {abs_model_path}")
            else:
                logger.warning(f"✗ 模型路径不存在: {model_path}")
                logger.warning(f"  尝试的绝对路径: {abs_model_path}")
                logger.warning(f"  将尝试从 {model_hub} 加载（可能需要下载）")
            
            # 更新配置以使用小模型
            config.defrost()
            original_model_type = config.model.type
            config.model.type = client_model_type
            config.freeze()
            
            # 构建小模型
            logger.info(f"正在加载模型 (hub: {model_hub})...")
            try:
                from federatedscope.core.auxiliaries.model_builder import get_model
                import traceback
                client_model = get_model(config, data, backend=config.backend)
                logger.info(f"✓ 客户端{ID}成功加载小模型")
                logger.info(f"  模型类型: {type(client_model)}")
                # 尝试获取模型参数数量
                try:
                    num_params = sum(p.numel() for p in client_model.parameters())
                    logger.info(f"  模型参数量: {num_params / 1e6:.2f}M")
                except:
                    pass
                logger.info("=" * 80)
            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"✗ 客户端{ID}加载小模型失败!")
                logger.error(f"  错误类型: {type(e).__name__}")
                logger.error(f"  错误信息: {str(e)}")
                logger.error(f"  模型路径: {model_path}")
                logger.error(f"  模型hub: {model_hub}")
                logger.error("详细错误堆栈:")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("=" * 80)
                logger.warning(f"客户端{ID}将回退使用服务器模型")
                client_model = model
            # 恢复原始配置（避免影响其他客户端）
            config.defrost()
            config.model.type = original_model_type
            config.freeze()
        else:
            logger.warning(f"客户端{ID}未指定小模型类型，将使用服务器模型（不推荐）")
            client_model = model
        
        super(KnowledgeAnchorClient, self).__init__(
            ID, server_id, state, config, data, client_model, device, strategy, **kwargs)
        
        # 确保trainer设置了知识锚点（如果有）
        self.knowledge_anchor = None

    def _register_default_handlers(self):
        """注册消息处理器"""
        super()._register_default_handlers()
        # 注册知识锚点消息处理器（使用register_handlers，不依赖comm_manager）
        self.register_handlers('knowledge_anchor',
                              self.callback_funcs_for_knowledge_anchor,
                              [None])

    def callback_funcs_for_knowledge_anchor(self, message: Message):
        """
        接收服务器下发的知识锚点
        """
        self.knowledge_anchor = message.content
        logger.info(f"客户端{self.ID}接收到知识锚点")
        
        # 设置trainer的知识锚点
        if hasattr(self.trainer, 'set_knowledge_anchor'):
            self.trainer.set_knowledge_anchor(self.knowledge_anchor)
            logger.info(f"客户端{self.ID}已设置知识锚点到trainer")

    def callback_funcs_for_model_para(self, message: Message):
        """
        重写模型参数接收处理，在训练前设置知识锚点
        """
        # 先调用父类方法接收模型参数
        result = super().callback_funcs_for_model_para(message)
        
        # 如果收到了知识锚点，确保trainer已设置
        if self.knowledge_anchor and hasattr(self.trainer, 'set_knowledge_anchor'):
            self.trainer.set_knowledge_anchor(self.knowledge_anchor)
        
        return result

    def callback_funcs_for_model_para_after_train(self, message: Message):
        """
        重写训练后的模型参数上传，包含增量bank
        """
        # 调用父类的训练逻辑
        sample_size, model_para_all, results = self.trainer.train()
        
        # 获取增量bank
        incremental_bank = []
        if hasattr(self.trainer, 'get_incremental_bank'):
            incremental_bank = self.trainer.get_incremental_bank()
            # 清空增量bank（已上传）
            if hasattr(self.trainer, 'clear_incremental_bank'):
                self.trainer.clear_incremental_bank()
        
        # 上传模型参数和增量bank
        sender = message.sender
        timestamp = message.timestamp
        
        # 准备上传内容：(sample_size, model_para, incremental_bank)
        content = (sample_size, model_para_all, incremental_bank)
        
        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=self._gen_timestamp(
                    init_timestamp=timestamp,
                    instance_number=sample_size),
                content=content
            )
        )
        
        logger.info(f"客户端{self.ID}上传模型参数和增量bank（{len(incremental_bank)}个新原型）")

