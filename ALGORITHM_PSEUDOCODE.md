# 知识锚点联邦学习算法伪代码（Knowledge Anchor Federated Learning）

## 算法概述

本算法实现了**服务器端投影LoRA联邦蒸馏**方案，结合**知识锚点（Knowledge Anchor）**机制，用于大模型（服务器）和小模型（客户端）之间的联邦学习。

### 核心组件

- **服务器**：大模型（Llama-2-7B，hidden_size=4096）+ 全局大LoRA
- **客户端**：小模型（TinyLlama，hidden_size=2048）+ 小LoRA
- **投影映射器**：实现大LoRA ↔ 小LoRA的双向投影（所有投影逻辑在服务器端）
- **知识锚点**：指令（Instructions）+ 原型库（Prototype Bank）

---

## 完整算法流程

### 初始化阶段

```
算法：Knowledge Anchor FL 初始化
输入：
  - 服务器模型：M_server (Llama-2-7B, hidden_size=4096)
  - 客户端模型：M_client (TinyLlama, hidden_size=2048)
  - 数据：D = {D_server, D_client_1, D_client_2, D_client_3}
  - LoRA配置：rank=8, target_modules=["q_proj", "v_proj"]

1. 服务器初始化：
   1.1 加载大模型 M_server
   1.2 初始化大LoRA：L_large = PEFT(M_server, rank=8)
   1.3 初始化投影映射器：
       P = ProjectionMapper(small_hidden=2048, large_hidden=4096, rank=8)
       Q = P.t()  // 或伪逆，或独立可学习参数
   1.4 初始化知识锚点生成器：Generator(M_server, tokenizer)
   1.5 全局大LoRA状态：global_large_lora = L_large.get_peft_state_dict()

2. 客户端初始化（每个客户端）：
   2.1 加载小模型 M_client
   2.2 初始化小LoRA：L_small = PEFT(M_client, rank=8)
   2.3 初始化知识锚点trainer：Trainer(M_client, L_small)
```

---

### 每轮训练流程（Round r）

```
算法：Knowledge Anchor FL 训练轮次
For round r = 0 to T-1:
    
    // ========== 服务器端：生成知识锚点 ==========
    3. 服务器生成知识锚点：
       3.1 从服务器数据 D_server 生成知识锚点：
           KA = Generator.generate_knowledge_anchor(D_server)
           KA = {
               instructions: [I_1, I_2, ..., I_N],  // N个指令
               prototype_bank: {
                   prototypes: P ∈ R^(K×d),  // K个原型，每个d维
                   soft_labels: S ∈ R^(K×V), // K个软标签，V是词汇表大小
                   num_prototypes: K
               }
           }
       3.2 发送知识锚点给所有客户端：
           Send(KA) → {Client_1, Client_2, Client_3}
    
    4. 服务器端微调（可选）：
       If train_on_server:
           M_server ← Train(M_server, D_server, L_large)
           global_large_lora ← L_large.get_peft_state_dict()
    
    // ========== 服务器端：下投影并广播 ==========
    5. 服务器下投影（大LoRA → 小LoRA）：
       5.1 获取全局大LoRA：
           large_lora = global_large_lora  // shape: {lora_A: (8, 4096), lora_B: (4096, 8)}
       5.2 下投影：
           For each LoRA parameter in large_lora:
               If 'lora_A' in key:
                   small_lora_A = large_lora_A @ P.t()  
                   // (8, 4096) @ (4096, 2048) = (8, 2048)
               If 'lora_B' in key:
                   small_lora_B = P @ large_lora_B  
                   // (2048, 4096) @ (4096, 8) = (2048, 8)
       5.3 广播小LoRA给所有客户端：
           Send(small_lora) → {Client_1, Client_2, Client_3}
    
    // ========== 客户端：本地训练 ==========
    6. 客户端训练（并行，每个客户端独立）：
       For each client i in {1, 2, 3}:
           6.1 接收知识锚点：KA_i = Receive(KA)
           6.2 接收小LoRA：small_lora_i = Receive(small_lora)
           6.3 加载小LoRA到本地模型：
               M_client_i.load_state_dict(small_lora_i)
           6.4 初始化增量bank：incremental_bank_i = []
           6.5 本地训练：
               For each batch (x, y) in D_client_i:
                   // 前向传播
                   outputs = M_client_i(x, output_hidden_states=True)
                   hidden_states = outputs.hidden_states[layer_idx]  // 中间层
                   h = mean(hidden_states, dim=1)  // [batch, hidden_dim]
                   
                   // 任务相关性判断
                   For each sample j in batch:
                       h_j = h[j]
                       // 计算与原型库的相似度
                       similarities = cosine_similarity(h_j, KA_i.prototypes)
                       max_sim = max(similarities)
                       
                       If max_sim >= threshold:  // 任务相关
                           // 动态原型匹配（FedKA方法）
                           top_k_indices, top_k_weights = TopK(similarities, K_p=5)
                           
                           // 结构化锚点对齐损失
                           L_proto = Σ(weight_k * MSE(h_j, prototype_k))  // k in top_k
                           
                           // 软标签蒸馏损失（如果有）
                           L_distill = Σ(weight_k * KL(student_logits, soft_label_k))
                           
                           // 锚点损失
                           L_anchor = L_proto + γ * L_distill  // γ=1.0
                           
                           // 标准交叉熵损失
                           L_CE = CrossEntropy(outputs.logits, y)
                           
                           // 总损失
                           L_total = L_CE + λ * L_anchor  // λ=0.5
                           
                           // 反向传播
                           Backward(L_total)
                           
                           relevant_samples++
                       Else:  // 任务无关
                           // 排除该样本，不参与训练
                           irrelevant_samples++
                           
                           // 可选：添加到增量bank（如果相似度接近阈值）
                           If max_sim >= threshold * 0.8:
                               incremental_bank_i.append(h_j)
                   
                   // 如果所有样本都无关，跳过该batch
                   If all samples irrelevant:
                       Skip batch
               
               6.6 获取训练后的LoRA参数：
                   small_lora_update_i = M_client_i.get_trainable_state_dict()
               6.7 上传给服务器：
                   Send(sample_size_i, small_lora_update_i, incremental_bank_i) → Server
    
    // ========== 服务器端：聚合和上投影 ==========
    7. 服务器聚合：
       7.1 收集所有客户端更新：
           updates = [(sample_size_1, small_lora_1, bank_1),
                      (sample_size_2, small_lora_2, bank_2),
                      (sample_size_3, small_lora_3, bank_3)]
       7.2 聚合小LoRA更新（FedAvg）：
           total_weight = Σ(sample_size_i)
           aggregated_small_lora = Σ(sample_size_i / total_weight * small_lora_i)
       7.3 整合增量bank：
           all_new_prototypes = Concat([bank_1, bank_2, bank_3])
           If len(all_new_prototypes) > max_prototypes:
               all_new_prototypes = KMeans_Clustering(all_new_prototypes, max_prototypes)
           KA.prototype_bank.prototypes = Concat(KA.prototype_bank.prototypes, all_new_prototypes)
       7.4 上投影（小LoRA → 大LoRA）：
           For each LoRA parameter in aggregated_small_lora:
               If 'lora_A' in key:
                   large_lora_A_delta = small_lora_A @ Q.t()  
                   // (8, 2048) @ (2048, 4096) = (8, 4096)
               If 'lora_B' in key:
                   large_lora_B_delta = Q @ small_lora_B  
                   // (4096, 2048) @ (2048, 8) = (4096, 8)
       7.5 更新全局大LoRA：
           global_large_lora += lr * large_lora_delta  // lr=1.0（可配置）
           M_server.load_state_dict(global_large_lora, strict=False)
    
    // ========== 下一轮 ==========
    8. r = r + 1
```

---

## 关键算法细节

### 1. 投影映射公式

```
下投影（Broadcast）：
  small_lora_A = large_lora_A @ P.t()     // (rank, large_hidden) → (rank, small_hidden)
  small_lora_B = P @ large_lora_B         // (large_hidden, rank) → (small_hidden, rank)

上投影（Aggregate）：
  large_lora_A = small_lora_A @ Q.t()     // (rank, small_hidden) → (rank, large_hidden)
  large_lora_B = Q @ small_lora_B         // (small_hidden, rank) → (large_hidden, rank)

其中：
  P: (small_hidden × large_hidden) = (2048 × 4096)
  Q: (large_hidden × small_hidden) = (4096 × 2048)
  默认：Q = P.t()
```

### 2. 任务相关性判断

```
算法：CheckTaskRelevance
输入：
  - h: 样本的隐藏状态向量 [hidden_dim]
  - prototypes: 原型库 [K, hidden_dim]
  - threshold: 相似度阈值 (默认0.5)

1. 计算余弦相似度：
   similarities = cosine_similarity(h, prototypes)  // [K]

2. 最大相似度：
   max_sim = max(similarities)

3. 判断：
   If max_sim >= threshold:
       return (True, max_sim)  // 任务相关
   Else:
       return (False, max_sim)  // 任务无关
```

### 3. 动态原型匹配（FedKA方法）

```
算法：DynamicPrototypeMatching
输入：
  - h: 样本隐藏状态 [hidden_dim]
  - prototypes: 原型库 [K, hidden_dim]
  - K_p: top-K原型数 (默认5)
  - τ: 温度参数 (默认0.1)

1. 计算相似度：
   similarities = cosine_similarity(h, prototypes)  // [K]

2. 选择top-K_p：
   top_k_indices = TopK(similarities, K_p)  // 索引
   top_k_similarities = similarities[top_k_indices]

3. 计算软权重（温度缩放softmax）：
   weights = softmax(top_k_similarities / τ)  // [K_p]

4. 返回：
   return (top_k_indices, weights, max(similarities))
```

### 4. 结构化锚点对齐损失

```
算法：ComputeAnchorLoss
输入：
  - h: 样本隐藏状态 [hidden_dim]
  - top_k_indices: top-K原型索引 [K_p]
  - top_k_weights: 软权重 [K_p]
  - prototypes: 原型库 [K, hidden_dim]
  - soft_labels: 软标签 [K, vocab_size] (可选)
  - outputs: 模型输出（包含logits）

1. 原型对齐损失：
   L_proto = 0
   For k_idx, weight in zip(top_k_indices, top_k_weights):
       prototype_k = prototypes[k_idx]
       L_proto += weight * MSE(h, prototype_k)

2. 软标签蒸馏损失（如果有）：
   L_distill = 0
   If soft_labels is not None:
       For k_idx, weight in zip(top_k_indices, top_k_weights):
           student_logits = outputs.logits[-1, :]  // 最后一个token的logits
           teacher_logits = soft_labels[k_idx]
           
           // 温度软化
           student_soft = log_softmax(student_logits / τ)
           teacher_soft = softmax(teacher_logits / τ)
           
           // KL散度
           L_distill += weight * KL(student_soft, teacher_soft)

3. 总锚点损失：
   L_anchor = L_proto + γ * L_distill  // γ=1.0

4. 返回 L_anchor
```

### 5. 知识锚点生成

```
算法：GenerateKnowledgeAnchor
输入：
  - M_server: 服务器大模型
  - D_server: 服务器数据
  - num_instructions: 指令数量
  - num_prototypes: 原型数量

1. 生成指令：
   instructions = []
   For i = 1 to num_instructions:
       // 从数据中提取代表性样本
       sample = SelectRepresentativeSample(D_server)
       // 使用模型生成指令（或从样本中提取）
       instruction = GenerateInstruction(M_server, sample)
       instructions.append(instruction)

2. 生成原型库：
   prototypes = []
   soft_labels = []
   
   For each sample in D_server:
       outputs = M_server(sample, output_hidden_states=True)
       h = outputs.hidden_states[layer_idx].mean(dim=1)  // [hidden_dim]
       logits = outputs.logits[-1, :]  // [vocab_size]
       prototypes.append(h)
       soft_labels.append(logits)
   
   // 聚类得到K个原型
   prototypes, soft_labels = KMeans_Clustering(prototypes, num_prototypes, soft_labels)

3. 返回：
   return {
       instructions: instructions,
       prototype_bank: {
           prototypes: prototypes,  // [K, hidden_dim]
           soft_labels: soft_labels,  // [K, vocab_size]
           num_prototypes: K
       }
   }
```

---

## 数据流图

```
Round r:

服务器端：
  ┌─────────────────────────────────────────┐
  │ 1. 生成知识锚点 KA                      │
  │    KA = {instructions, prototype_bank}  │
  └──────────────┬──────────────────────────┘
                 │
                 ├─→ 发送KA给所有客户端
                 │
  ┌──────────────▼──────────────────────────┐
  │ 2. 获取全局大LoRA                       │
  │    large_lora = global_large_lora       │
  └──────────────┬──────────────────────────┘
                 │
                 ├─→ 下投影：P @ large_lora
                 │
  ┌──────────────▼──────────────────────────┐
  │ 3. 生成小LoRA                           │
  │    small_lora = down_project(large_lora)│
  └──────────────┬──────────────────────────┘
                 │
                 ├─→ 广播small_lora给所有客户端
                 │
                 │
客户端端（并行）：
  ┌──────────────▼──────────────────────────┐
  │ Client i:                               │
  │ 4. 接收 KA_i, small_lora_i              │
  │ 5. 加载 small_lora_i 到 M_client_i      │
  │ 6. 本地训练：                            │
  │    - 任务相关性判断                      │
  │    - 原型对齐损失                        │
  │    - 反向传播更新 small_lora_i          │
  │ 7. 上传：                               │
  │    (sample_size, small_lora_update_i,   │
  │     incremental_bank_i)                 │
  └──────────────┬──────────────────────────┘
                 │
                 │
服务器端：
  ┌──────────────▼──────────────────────────┐
  │ 8. 聚合所有客户端更新                   │
  │    aggregated_small_lora = FedAvg(...)   │
  └──────────────┬──────────────────────────┘
                 │
                 ├─→ 上投影：Q @ aggregated_small_lora
                 │
  ┌──────────────▼──────────────────────────┐
  │ 9. 更新全局大LoRA                       │
  │    global_large_lora += lr * large_delta│
  └──────────────────────────────────────────┘
```

---

## 关键特性

### 1. 服务器端投影LoRA
- ✅ **所有投影逻辑在服务器端**：客户端无需知道投影过程
- ✅ **通信高效**：只传输小LoRA参数（约1/4参数量）
- ✅ **客户端极简**：客户端只需像普通LoRA一样训练和上传

### 2. 知识锚点机制
- ✅ **任务相关性判断**：自动识别并排除任务无关数据
- ✅ **原型对齐**：通过原型库实现知识传递
- ✅ **增量学习**：动态扩展原型库

### 3. FedKA方法
- ✅ **动态原型匹配**：每个样本匹配top-K_p个原型
- ✅ **软权重**：使用温度缩放softmax计算权重
- ✅ **结构化损失**：L_anchor = L_proto + γ * L_distill

---

## 配置参数

```yaml
# 投影映射配置
llm.knowledge_anchor:
  use_projection_lora: true
  small_hidden: 2048
  large_hidden: 4096
  projection_learnable: true
  q_strategy: 'transpose'  # 'transpose', 'pseudo_inverse', 'learnable'
  lora_update_lr: 1.0

# 知识锚点配置
llm.knowledge_anchor:
  num_instructions: 10
  num_prototypes: 50
  intermediate_layer: 16
  task_relevance_threshold: 0.5
  prototype_alignment_weight: 0.5  # λ
  top_k_prototypes: 5  # K_p
  prototype_temperature: 0.1  # τ
  distill_weight: 1.0  # γ
```

---

## 算法优势

1. **通信高效**：只传输小LoRA参数，通信量减少约75%
2. **客户端轻量**：客户端无需投影计算，保持原有LoRA训练逻辑
3. **知识传递**：通过知识锚点实现服务器到客户端的知识传递
4. **异构数据鲁棒**：自动识别并排除任务无关数据
5. **增量学习**：动态扩展原型库，适应新任务

