# 知识锚点算法逻辑检查清单

## 📋 检查顺序和重点

按照以下顺序系统检查，确保实现与设计一致。

---

## 阶段1：配置和注册检查 ⚙️

### 1.1 配置文件检查
**文件**: `fedbiot_script/knowledge_anchor_llama2.yaml`

**检查点**:
- [ ] `federate.method: "knowledgeanchor"` 是否正确
- [ ] `data.splitter: 'knowledge_anchor'` 是否正确
- [ ] `trainer.type: knowledge_anchor_client` 是否正确
- [ ] `llm.knowledge_anchor.client_model_type` 是否指向本地小模型路径
- [ ] 数据分配比例：40%服务器、40%客户端、20%测试
- [ ] 服务器和客户端模型路径是否正确

**关键配置项**:
```yaml
federate:
  method: "knowledgeanchor"  # 必须匹配
  client_num: 3

data:
  splitter: 'knowledge_anchor'  # 必须匹配
  splits: [0.4, 0.4, 0.2]

trainer:
  type: knowledge_anchor_client  # 必须匹配

llm.knowledge_anchor:
  client_model_type: '/path/to/tinyllama@huggingface_llm'
```

---

### 1.2 组件注册检查
**文件**: 
- `federatedscope/core/configs/constants.py`
- `federatedscope/llm/knowledge_anchor/__init__.py`
- `federatedscope/core/auxiliaries/worker_builder.py`
- `federatedscope/core/auxiliaries/trainer_builder.py`
- `federatedscope/core/auxiliaries/aggregator_builder.py`

**检查点**:
- [ ] `constants.py` 中 `AGGREGATOR_TYPE["knowledgeanchor"] = "knowledge_anchor"` 是否存在
- [ ] `__init__.py` 中 worker 是否正确注册
- [ ] `worker_builder.py` 中是否正确导入 knowledge_anchor 模块
- [ ] `trainer_builder.py` 中是否正确导入 knowledge_anchor_trainer
- [ ] `aggregator_builder.py` 中是否正确映射到 KnowledgeAnchorAggregator

---

## 阶段2：数据分配检查 📊

### 2.1 数据分配器
**文件**: `federatedscope/core/splitters/generic/knowledge_anchor_splitter.py`

**检查点**:
- [ ] 是否正确实现 40/40/20 分配
- [ ] 服务器数据是否占 40%
- [ ] 客户端数据总和是否占 40%（3个客户端）
- [ ] 测试数据是否占 20%
- [ ] non-IID 分配是否正确（Dirichlet分布）
- [ ] 任务不相关数据注入逻辑是否正确
- [ ] 返回格式是否正确：`[server_data, client1_data, client2_data, client3_data, test_data]`

**关键逻辑检查**:
```python
# 检查点1: 数据比例
server_ratio = 0.4
client_ratio = 0.4  # 总和
test_ratio = 0.2

# 检查点2: non-IID分配
# 使用Dirichlet分布，alpha越小越non-IID

# 检查点3: 任务不相关数据
# 从测试集中抽取一部分注入到某个客户端
```

---

## 阶段3：服务器端流程检查 🖥️

### 3.1 知识锚点生成器
**文件**: `federatedscope/llm/misc/knowledge_anchor_generator.py`

**检查点**:
- [ ] `generate_instructions()` 是否正确生成指令
  - [ ] 是否使用服务器LLM采样公共数据集
  - [ ] 生成的指令数量是否符合配置
  - [ ] 指令格式是否正确（可用于输入增强）
- [ ] `extract_prototype_bank()` 是否正确提取原型库
  - [ ] 是否提取中间层隐藏状态（`intermediate_layer`）
  - [ ] 是否对序列维度求平均
  - [ ] K-means聚类是否正确
  - [ ] 原型数量是否符合配置
  - [ ] 边界检查是否正确（层索引超出范围时的处理）
- [ ] `generate_knowledge_anchor()` 是否组合指令和原型库

**关键逻辑检查**:
```python
# 检查点1: 指令生成
instructions = self.generate_instructions(public_dataset, task_description)
# 应该返回 List[str]

# 检查点2: 原型库提取
prototype_bank = self.extract_prototype_bank(public_dataset, dataloader)
# 应该返回 Dict，包含:
# - 'prototypes': np.ndarray  # [num_prototypes, hidden_dim]
# - 'num_prototypes': int
# - 'intermediate_layer': int

# 检查点3: 层索引边界检查
# 如果 intermediate_layer 超出范围，应该使用中间层
```

---

### 3.2 服务器Worker
**文件**: `federatedscope/llm/knowledge_anchor/server.py`

**检查点**:
- [ ] `__init__()` 是否正确初始化
  - [ ] 是否正确创建 KnowledgeAnchorGenerator
  - [ ] 是否正确创建服务器端trainer（如果有数据）
- [ ] `generate_and_send_knowledge_anchor()` 是否正确
  - [ ] 是否在每轮开始时调用
  - [ ] 是否正确生成知识锚点
  - [ ] 是否正确发送给所有客户端
  - [ ] 是否设置到聚合器
- [ ] `train_on_server_data()` 是否正确
  - [ ] 是否在服务器数据上微调
  - [ ] 是否记录训练损失
- [ ] `check_and_move_on()` 重写是否正确
  - [ ] 是否在新轮开始时生成知识锚点
  - [ ] 是否调用服务器端训练
- [ ] `callback_funcs_model_para()` 是否正确处理增量bank
  - [ ] 是否从消息中提取增量bank
  - [ ] 是否传递给聚合器

**关键流程检查**:
```python
# 流程1: 初始化
server = KnowledgeAnchorServer(...)
# 应该创建 anchor_generator 和 server_trainer

# 流程2: 每轮开始
server.check_and_move_on()
# 应该:
# 1. 生成知识锚点
# 2. 发送给客户端
# 3. 在服务器数据上微调（如果启用）

# 流程3: 接收客户端更新
server.callback_funcs_model_para(message)
# 应该提取 (sample_size, model_para, incremental_bank)
```

---

### 3.3 服务器聚合器
**文件**: `federatedscope/core/aggregators/knowledge_anchor_aggregator.py`

**检查点**:
- [ ] `aggregate()` 是否正确聚合
  - [ ] 是否聚合LoRA参数（使用映射器）
  - [ ] 是否整合增量bank
  - [ ] 是否更新知识锚点
- [ ] `_aggregate_lora_params()` 是否正确
  - [ ] 是否使用LoRA映射器（如果需要）
  - [ ] 聚合策略是否正确（加权平均）
- [ ] `_integrate_prototype_banks()` 是否正确
  - [ ] 是否合并增量bank
  - [ ] 是否重新聚类（如果超过max_prototypes）
  - [ ] 是否更新全局原型库
- [ ] `set_knowledge_anchor()` 是否正确设置初始知识锚点

**关键逻辑检查**:
```python
# 检查点1: LoRA聚合
# 如果客户端使用小模型，需要映射到大模型
if need_mapping:
    mapped_lora = lora_mapper.map_lora_state_dict(...)
aggregated_lora = aggregate_mapped_lora(...)

# 检查点2: 增量bank整合
# 合并所有客户端的增量bank
all_incremental = [bank1, bank2, bank3]
if total_prototypes > max_prototypes:
    # 重新聚类
    reclustered = kmeans_cluster(...)
```

---

## 阶段4：客户端流程检查 💻

### 4.1 客户端Trainer
**文件**: `federatedscope/llm/trainer/knowledge_anchor_trainer.py`

**检查点**:
- [ ] `set_knowledge_anchor()` 是否正确设置知识锚点
- [ ] `augment_input_with_instruction()` 是否正确
  - [ ] 是否使用指令模板增强输入
  - [ ] 是否处理占位符（`{input}`, `{instruction}`）
- [ ] `check_task_relevance()` 是否正确
  - [ ] 是否计算余弦相似度
  - [ ] 是否与阈值比较
  - [ ] 是否返回匹配的原型索引
- [ ] `_hook_on_batch_forward()` 是否正确
  - [ ] 是否提取中间层隐藏状态
  - [ ] 是否对每个样本判断任务相关性
  - [ ] 是否计算对齐损失
  - [ ] 是否管理增量bank
  - [ ] 是否跳过不相关样本
- [ ] `add_to_incremental_bank()` 是否正确
  - [ ] 是否添加新原型到增量bank
  - [ ] 格式是否正确（包含hidden_state等）
- [ ] `get_incremental_bank()` 和 `clear_incremental_bank()` 是否正确

**关键逻辑检查**:
```python
# 检查点1: 输入增强（应该在数据预处理阶段，但trainer中可能有）
augmented_text = augment_input_with_instruction(input_text)

# 检查点2: 前向传播
outputs = model(...)
hidden_states = outputs.hidden_states[intermediate_layer]
# 对序列维度求平均
hidden_state = hidden_states.mean(dim=1)  # [batch, hidden_dim]

# 检查点3: 任务相关性判断
for each sample:
    is_relevant, matched_idx = check_task_relevance(hidden_state)
    if is_relevant:
        if matched_idx is not None:
            # 有对应原型，计算对齐损失
            alignment_loss += cosine_loss(hidden_state, prototype)
        else:
            # 无对应原型，加入增量bank
            add_to_incremental_bank(hidden_state)
    else:
        # 跳过该样本
        skip_this_sample = True

# 检查点4: 损失计算
total_loss = standard_loss + alignment_weight * alignment_loss
```

---

### 4.2 客户端Worker
**文件**: `federatedscope/llm/knowledge_anchor/client.py`

**检查点**:
- [ ] `__init__()` 是否正确
  - [ ] 是否从配置加载小模型
  - [ ] 是否正确处理本地模型路径
  - [ ] 是否有错误处理和回退机制
- [ ] `callback_funcs_for_knowledge_anchor()` 是否正确
  - [ ] 是否接收知识锚点
  - [ ] 是否设置到trainer
- [ ] `callback_funcs_for_model_para()` 是否正确
  - [ ] 是否在训练前设置知识锚点
- [ ] `callback_funcs_for_model_para_after_train()` 是否正确
  - [ ] 是否获取增量bank
  - [ ] 是否上传模型参数和增量bank
  - [ ] 是否清空增量bank

**关键流程检查**:
```python
# 流程1: 初始化
client = KnowledgeAnchorClient(...)
# 应该:
# 1. 从配置读取 client_model_type
# 2. 加载小模型（本地路径）
# 3. 创建trainer

# 流程2: 接收知识锚点
client.callback_funcs_for_knowledge_anchor(message)
# 应该:
# 1. 保存知识锚点
# 2. 设置到trainer

# 流程3: 训练和上传
client.callback_funcs_for_model_para_after_train(message)
# 应该:
# 1. 训练模型
# 2. 获取增量bank
# 3. 上传 (sample_size, model_para, incremental_bank)
```

---

## 阶段5：通信流程检查 📡

### 5.1 消息传递流程
**检查点**:
- [ ] 服务器 → 客户端：知识锚点
  - [ ] 消息类型：`'knowledge_anchor'`
  - [ ] 发送时机：每轮开始时
  - [ ] 接收处理：`client.callback_funcs_for_knowledge_anchor()`
- [ ] 客户端 → 服务器：模型参数和增量bank
  - [ ] 消息类型：`'model_para'`
  - [ ] 消息内容：`(sample_size, model_para, incremental_bank)`
  - [ ] 接收处理：`server.callback_funcs_model_para()`
- [ ] 服务器 → 客户端：聚合后的模型参数
  - [ ] 消息类型：`'model_para'`
  - [ ] 发送时机：聚合后
  - [ ] 接收处理：`client.callback_funcs_for_model_para()`

**关键流程检查**:
```
Round 0:
  服务器: 生成知识锚点 → 发送给客户端
  客户端: 接收知识锚点 → 设置到trainer
  客户端: 训练 → 上传 (model_para, incremental_bank)
  服务器: 接收 → 聚合 → 更新知识锚点 → 发送新模型参数

Round 1:
  服务器: 使用更新后的知识锚点 → 发送给客户端
  ...
```

---

## 阶段6：关键逻辑漏洞检查 🔍

### 6.1 数据流检查
- [ ] 输入增强是否在正确时机执行
  - ⚠️ **潜在问题**: 输入增强应该在tokenization之前，但当前可能在trainer中处理
  - [ ] 检查数据预处理流程
- [ ] 任务不相关数据是否被正确跳过
  - [ ] 检查 `skip_this_batch` 是否正确设置
  - [ ] 检查损失计算是否排除不相关样本
- [ ] 增量bank是否在正确时机清空
  - [ ] 上传后是否清空
  - [ ] 是否避免重复上传

### 6.2 模型一致性检查
- [ ] 服务器和客户端模型架构是否匹配
  - [ ] 中间层索引是否一致
  - [ ] 隐藏状态维度是否匹配
- [ ] LoRA映射是否正确
  - [ ] 小模型层数到大模型层数的映射
  - [ ] 参数维度是否匹配
- [ ] 原型库维度是否匹配
  - [ ] 服务器生成的原型维度
  - [ ] 客户端提取的隐藏状态维度
  - [ ] 增量bank中的原型维度

### 6.3 边界情况检查
- [ ] 知识锚点为None时的处理
  - [ ] 客户端是否优雅降级
- [ ] 原型库为空时的处理
  - [ ] 是否所有样本都加入增量bank
- [ ] 所有样本都不相关时的处理
  - [ ] 是否跳过整个batch
  - [ ] 损失计算是否正确
- [ ] 增量bank为空时的处理
  - [ ] 是否上传空列表
  - [ ] 服务器是否正确处理

### 6.4 内存和性能检查
- [ ] 隐藏状态是否正确释放
  - [ ] 是否使用 `del outputs.hidden_states`
- [ ] GPU内存管理
  - [ ] 是否使用 `torch.cuda.empty_cache()`
  - [ ] 是否使用 `gc.collect()`
- [ ] 原型库大小控制
  - [ ] 是否在超过max_prototypes时重新聚类

---

## 阶段7：端到端流程验证 ✅

### 7.1 完整流程检查
按照以下流程逐步验证：

```
1. 初始化阶段
   ✓ 数据分配：40/40/20
   ✓ 服务器加载Llama-7B
   ✓ 客户端加载TinyLlama
   ✓ 创建trainer和aggregator

2. Round 0 开始
   ✓ 服务器生成知识锚点
   ✓ 服务器发送知识锚点给客户端
   ✓ 客户端接收并设置知识锚点
   ✓ 服务器在本地数据上微调

3. Round 0 训练
   ✓ 客户端接收模型参数
   ✓ 客户端使用知识锚点训练
     - 输入增强
     - 原型对齐
     - 任务相关性判断
     - 增量bank管理
   ✓ 客户端上传 (model_para, incremental_bank)

4. Round 0 聚合
   ✓ 服务器接收客户端更新
   ✓ 服务器聚合LoRA参数（使用映射器）
   ✓ 服务器整合增量bank
   ✓ 服务器更新知识锚点
   ✓ 服务器发送新模型参数给客户端

5. Round 1+ 循环
   ✓ 重复步骤2-4
   ✓ 知识锚点逐步更新
   ✓ 损失逐步下降
```

---

## 阶段8：配置一致性检查 🔧

### 8.1 配置项检查清单
- [ ] `intermediate_layer` 配置
  - [ ] 服务器和客户端是否使用相同的层索引
  - [ ] 层索引是否在有效范围内
- [ ] `task_relevance_threshold` 配置
  - [ ] 阈值是否合理（0-1之间）
- [ ] `prototype_alignment_weight` 配置
  - [ ] 权重是否合理（通常0.1-0.5）
- [ ] `num_prototypes` 和 `max_prototypes` 配置
  - [ ] `max_prototypes` 应该 >= `num_prototypes`

---

## 快速检查命令 🚀

### 检查关键文件是否存在
```bash
# 检查所有关键文件
ls -la federatedscope/core/splitters/generic/knowledge_anchor_splitter.py
ls -la federatedscope/core/aggregators/knowledge_anchor_aggregator.py
ls -la federatedscope/llm/misc/knowledge_anchor_generator.py
ls -la federatedscope/llm/trainer/knowledge_anchor_trainer.py
ls -la federatedscope/llm/knowledge_anchor/server.py
ls -la federatedscope/llm/knowledge_anchor/client.py
ls -la federatedscope/llm/misc/lora_mapper.py
```

### 检查注册是否正确
```bash
# 检查常量映射
grep -n "knowledgeanchor" federatedscope/core/configs/constants.py

# 检查worker注册
grep -n "knowledge_anchor" federatedscope/llm/knowledge_anchor/__init__.py

# 检查trainer注册
grep -n "knowledge_anchor_client" federatedscope/llm/trainer/knowledge_anchor_trainer.py
```

---

## 常见问题检查 ⚠️

### 问题1: 知识锚点未正确传递
- [ ] 检查消息类型是否正确注册
- [ ] 检查消息处理器是否正确注册
- [ ] 检查消息发送和接收时机

### 问题2: 增量bank未正确上传
- [ ] 检查 `get_incremental_bank()` 是否被调用
- [ ] 检查消息内容格式是否正确
- [ ] 检查服务器是否正确解析

### 问题3: LoRA映射失败
- [ ] 检查模型配置是否匹配
- [ ] 检查映射策略是否正确
- [ ] 检查参数维度是否匹配

### 问题4: 任务相关性判断不准确
- [ ] 检查余弦相似度计算
- [ ] 检查阈值设置
- [ ] 检查原型库维度

---

## 检查完成后的验证 🎯

完成所有检查后，运行以下测试：

1. **单元测试**: 测试每个组件的核心功能
2. **集成测试**: 测试组件之间的交互
3. **端到端测试**: 运行完整的一轮训练
4. **日志检查**: 检查关键日志输出是否正确

---

## 检查优先级 ⭐

### 高优先级（必须检查）
1. 数据分配逻辑
2. 知识锚点生成和传递
3. 任务相关性判断
4. LoRA参数聚合

### 中优先级（重要检查）
1. 输入增强实现
2. 原型对齐损失
3. 增量bank管理
4. 服务器端训练

### 低优先级（可选检查）
1. 内存优化
2. 错误处理
3. 日志输出
4. 性能优化

---

**检查完成后，记录发现的问题和修复方案！**

