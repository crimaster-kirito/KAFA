from federatedscope.llm.trainer.trainer import LLMTrainer  # 假设基类
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F


class KnowledgeAnchorTrainer(LLMTrainer):
    def __init__(self, model, data, device, config, **kwargs):
        super().__init__(model, data, device, config, **kwargs)

        # 启用 LoRA
        peft_config = LoraConfig(r=config.train.lora_rank, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        self.model = get_peft_model(self.model, peft_config)

        # 接收服务器下发的知识锚点（假设以软标签或特征形式）
        self.global_anchor = None  # 在每轮开始时从消息更新

    def train_one_round(self, global_anchor=None):
        if global_anchor is not None:
            self.global_anchor = global_anchor

        # 本地训练：标准损失 + 锚点蒸馏损失
        for batch in self.dataloader:
            outputs = self.model(**batch)
            standard_loss = outputs.loss

            # 锚点蒸馏损失（示例：KL divergence on softened logits）
            if self.global_anchor is not None:
                anchor_logits = self.global_anchor['soft_labels'][batch_indices]  # 假设锚点提供软标签
                distill_loss = F.kl_div(F.log_softmax(outputs.logits, dim=-1),
                                        F.softmax(anchor_logits / self.config.train.temp, dim=-1),
                                        reduction='batchmean') * (self.config.train.temp ** 2)
                loss = standard_loss + self.config.train.anchor_lambda * distill_loss
            else:
                loss = standard_loss

            loss.backward()
            self.optimizer.step()

        # 计算并上传偏差向量（而非全 LoRA 参数）
        local_state = self.model.state_dict()
        global_state = self.ctx.global_model_state  # 从服务器接收的全局状态
        deviation = {k: local_state[k] - global_state[k] for k in local_state if 'lora' in k}

        # 添加差分隐私噪声（FederatedScope 已支持 DP-SGD，可额外在偏差上加噪）
        for k in deviation:
            deviation[k] += torch.normal(0, self.config.privacy.noise_std, deviation[k].shape).to(self.device)

        return deviation  # 上传给服务器
