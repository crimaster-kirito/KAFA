"""
Knowledge Anchor Server Worker
服务器端实现：
1. 生成知识锚点（指令+原型库）
2. 在服务器端使用40%数据微调Llama-7B
3. 每轮下发知识锚点给客户端
4. 接收客户端上传的增量bank和LoRA参数
5. 聚合并更新知识锚点
"""
import logging
import torch
from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
from federatedscope.llm.misc.knowledge_anchor_generator import KnowledgeAnchorGenerator
from federatedscope.llm.dataloader.dataloader import get_tokenizer

logger = logging.getLogger(__name__)


class KnowledgeAnchorServer(Server):
    """
    知识锚点服务器
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(KnowledgeAnchorServer, self).__init__(
            ID, state, config, data, model, client_num, total_round_num, device, strategy, **kwargs)
        
        # 初始化知识锚点生成器
        try:
            model_name, _ = config.model.type.split('@')
        except:
            model_name = config.model.type
        tokenizer, _ = get_tokenizer(model_name, config.data.root, config.llm.tok_len)
        self.anchor_generator = KnowledgeAnchorGenerator(model, tokenizer, config)
        
        # 知识锚点（每轮更新）
        self.knowledge_anchor = None
        
        # 服务器端trainer（用于在服务器数据上微调）
        if data and len(data.get('train', [])) > 0:
            self.server_trainer = get_trainer(
                model=model,
                data=data,
                device=device,
                config=config,
                only_for_eval=False,
                monitor=self._monitor
            )
        else:
            self.server_trainer = None
            logger.warning("服务器没有训练数据，将跳过服务器端微调")
        
        # 设置聚合器的知识锚点（确保aggregator已初始化）
        if hasattr(self, 'aggregator') and hasattr(self.aggregator, 'set_knowledge_anchor'):
            # 初始时还没有知识锚点，会在第一轮生成后设置
            pass

    def _register_default_handlers(self):
        """注册消息处理器"""
        super()._register_default_handlers()
        # 注册知识锚点相关的消息处理器
        # 使用register_handlers而不是comm_manager.register_handler，因为comm_manager可能还未初始化
        self.register_handlers('knowledge_anchor',
                               self.callback_funcs_for_knowledge_anchor)
        self.register_handlers('incremental_bank',
                               self.callback_funcs_for_incremental_bank)

    def callback_funcs_for_knowledge_anchor(self, message: Message):
        """处理客户端对知识锚点的请求（如果需要）"""
        # 如果客户端请求知识锚点，发送当前的知识锚点
        pass

    def callback_funcs_for_incremental_bank(self, message: Message):
        """处理客户端上传的增量bank"""
        sender = message.sender
        incremental_bank = message.content
        
        # 将增量bank传递给聚合器处理
        if hasattr(self.aggregator, '_integrate_incremental_banks'):
            self.aggregator._integrate_incremental_banks([incremental_bank])
            logger.info(f"收到客户端{sender}的增量bank，包含{len(incremental_bank)}个新原型")

    def generate_and_send_knowledge_anchor(self):
        """
        生成知识锚点并发送给客户端
        在每轮训练开始前调用
        """
        # 获取服务器端训练数据
        if not self.data or not isinstance(self.data, dict) or 'train' not in self.data:
            logger.warning("服务器没有训练数据，无法生成知识锚点")
            return None
        
        train_data = self.data.get('train', [])
        if not train_data or len(train_data) == 0:
            logger.warning("服务器训练数据为空，无法生成知识锚点")
            return None
        
        # 创建数据加载器
        from federatedscope.core.auxiliaries.dataloader_builder import get_dataloader
        train_loader = get_dataloader(
            train_data,
            self._cfg.dataloader.batch_size,
            self._cfg.dataloader.drop_last if hasattr(self._cfg.dataloader, 'drop_last') else False
        )
        
        # 生成知识锚点
        logger.info("\n" + "=" * 80)
        logger.info("服务器开始生成知识锚点...")
        logger.info("=" * 80)
        self.knowledge_anchor = self.anchor_generator.generate_knowledge_anchor(
            public_dataset=train_data,
            dataloader=train_loader,
            task_description=None  # 可以从配置中获取
        )
        
        # 输出生成的知识锚点摘要
        if self.knowledge_anchor:
            instructions = self.knowledge_anchor.get('instructions', [])
            prototype_bank = self.knowledge_anchor.get('prototype_bank', {})
            num_prototypes = prototype_bank.get('num_prototypes', 0)
            
            logger.info("\n" + "=" * 80)
            logger.info("知识锚点生成摘要:")
            logger.info(f"  - 指令数量: {len(instructions)}")
            logger.info(f"  - 原型数量: {num_prototypes}")
            logger.info("=" * 80)
        
        # 设置聚合器的知识锚点
        if hasattr(self.aggregator, 'set_knowledge_anchor'):
            self.aggregator.set_knowledge_anchor(self.knowledge_anchor)
        
        # 发送知识锚点给所有客户端
        logger.info(f"\n服务器发送知识锚点给客户端...")
        # 获取所有客户端ID
        # 优先使用comm_manager的neighbors（如果已初始化），否则使用配置的client_num
        if hasattr(self, 'comm_manager') and hasattr(self.comm_manager, 'neighbors') and len(self.comm_manager.neighbors) > 0:
            client_ids = list(self.comm_manager.neighbors.keys())
        else:
            # 如果neighbors还未初始化，使用配置的client_num
            client_ids = list(range(1, self.client_num + 1))
            logger.warning(f"comm_manager.neighbors未初始化，使用配置的client_num={self.client_num}")
        
        if len(client_ids) == 0:
            logger.warning("没有可用的客户端，跳过知识锚点发送")
            return self.knowledge_anchor
        
        logger.info(f"向客户端 {client_ids} 发送知识锚点")
        self.comm_manager.send(
            Message(
                msg_type='knowledge_anchor',
                sender=self.ID,
                receiver=client_ids,
                timestamp=self.state,  # 使用state作为timestamp
                state=self.state,
                content=self.knowledge_anchor
            )
        )
        
        return self.knowledge_anchor

    def train_on_server_data(self):
        """
        在服务器端数据上微调模型
        """
        if self.server_trainer is None:
            return
        
        logger.info("服务器开始在本地数据上微调...")
        num_samples, _, results = self.server_trainer.train()
        logger.info(f"服务器微调完成，处理了{num_samples}个样本")
        
        # 记录服务器端损失
        if results and 'train_loss' in results:
            logger.info(f"服务器训练损失: {results['train_loss']:.4f}")

    def callback_funcs_model_para(self, message: Message):
        """
        重写模型参数接收处理，支持接收增量bank
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        
        # 检查content是否包含增量bank
        if isinstance(content, tuple) and len(content) >= 3:
            # (sample_size, model_para, incremental_bank)
            sample_size, model_para, incremental_bank = content[0], content[1], content[2]
        elif isinstance(content, tuple) and len(content) == 2:
            # (sample_size, model_para)
            sample_size, model_para = content
            incremental_bank = None
        else:
            sample_size = content[0] if isinstance(content, tuple) else 1
            model_para = content[1] if isinstance(content, tuple) else content
            incremental_bank = None
        
        # 处理增量bank
        if incremental_bank is not None:
            self.callback_funcs_for_incremental_bank(
                Message(
                    msg_type='incremental_bank',
                    sender=sender,
                    receiver=[self.ID],
                    timestamp=timestamp,
                    state=round,
                    content=incremental_bank
                )
            )
        
        # 调用父类方法处理模型参数
        # 创建标准格式的message
        standard_message = Message(
            msg_type='model_para',
            sender=sender,
            receiver=[self.ID],
            timestamp=timestamp,
            state=round,
            content=(sample_size, model_para)
        )
        
        return super().callback_funcs_model_para(standard_message)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        重写broadcast方法，使用投影后的小LoRA
        
        如果使用投影映射，将全局大LoRA投影为小LoRA后直接发送给客户端
        不修改服务器模型，直接替换发送的model_para内容
        """
        # 检查是否使用投影映射
        use_projection = (hasattr(self, 'aggregator') and 
                         hasattr(self.aggregator, 'projection_mapper') and 
                         self.aggregator.projection_mapper is not None)
        
        if use_projection:
            # 获取投影后的小LoRA
            try:
                small_lora = self.aggregator.broadcast_small_lora()
                if small_lora:
                    logger.info(f"使用投影后的小LoRA进行broadcast（{len(small_lora)}个参数）")
                    
                    # 直接使用小LoRA进行broadcast，不修改服务器模型
                    # 需要手动实现broadcast逻辑，直接发送小LoRA
                    if filter_unseen_clients:
                        self.sampler.change_state(self.unseen_clients_id, 'unseen')
                    
                    if sample_client_num > 0:
                        receiver = self.sampler.sample(size=sample_client_num)
                    else:
                        receiver = list(self.comm_manager.neighbors.keys())
                        if msg_type == 'model_para':
                            self.sampler.change_state(receiver, 'working')
                    
                    # 噪声注入（如果需要）
                    if self._noise_injector is not None and msg_type == 'model_para':
                        # 对于投影后的小LoRA，跳过噪声注入（因为不是完整模型）
                        pass
                    
                    # 量化（如果需要）
                    model_para = small_lora
                    if msg_type == 'model_para' and self._cfg.quantization.method == 'uniform':
                        from federatedscope.core.compression import symmetric_uniform_quantization
                        nbits = self._cfg.quantization.nbits
                        model_para = symmetric_uniform_quantization(model_para, nbits)
                    
                    # 发送消息
                    rnd = self.state - 1 if msg_type == 'evaluate' else self.state
                    # 使用state作为timestamp（与父类保持一致）
                    timestamp = self.state if hasattr(self, 'state') else rnd
                    self.comm_manager.send(
                        Message(msg_type=msg_type,
                                sender=self.ID,
                                receiver=receiver,
                                state=min(rnd, self.total_round_num),
                                timestamp=timestamp,
                                content=model_para))
                    
                    if self._cfg.federate.online_aggr:
                        for idx in range(self.model_num):
                            self.aggregators[idx].reset()
                    
                    if filter_unseen_clients:
                        self.sampler.change_state(self.unseen_clients_id, 'seen')
                    
                    return
            except Exception as e:
                logger.warning(f"获取投影后的小LoRA失败: {e}，使用原始broadcast方法")
        
        # 不使用投影映射或获取失败，使用原始broadcast方法
        return super().broadcast_model_para(
            msg_type=msg_type,
            sample_client_num=sample_client_num,
            filter_unseen_clients=filter_unseen_clients
        )
    
    def check_and_move_on(self, check_eval_result=False, min_received_num=None):
        """
        重写移动逻辑，在聚合前生成并发送知识锚点
        """
        # 如果是新的一轮，先生成并发送知识锚点
        if self.state == 0 or (self.state > 0 and self.state not in self.msg_buffer['train']):
            # 生成知识锚点
            self.generate_and_send_knowledge_anchor()
            
            # 在服务器端数据上微调（可选）
            if hasattr(self._cfg.llm.knowledge_anchor, 'train_on_server') and \
               self._cfg.llm.knowledge_anchor.train_on_server:
                self.train_on_server_data()
        
        # 调用父类方法
        return super().check_and_move_on(check_eval_result, min_received_num)

