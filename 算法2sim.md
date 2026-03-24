# 知识锚点联邦学习算法伪代码（Knowledge Anchor Federated Learning）

## 算法概述

本算法借助**服务器端投影LoRA的异构联邦学习**方案，结合提出的**知识锚点（Knowledge Anchor）**机制，实现了在联邦条件下闭源大模型微调的问题。

### 核心组件

- **服务器**：大模型（Llama-2-7B，hidden_size=4096）+ 全局大LoRA
- **客户端**：小模型（TinyLlama，hidden_size=2048）+ 小LoRA
- **投影映射器**：实现大LoRA ↔ 小LoRA的双向投影（所有投影逻辑在服务器端）
- **知识锚点**：指令（Instructions）+ 原型库（Prototype Bank）

---

## 完整算法流程

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
       Q = P⁻¹
   1.4 初始化知识锚点生成器：Generator(M_server, tokenizer)
   1.5 全局大LoRA状态：global_large_lora = L_large.get_peft_state_dict()
   1.6 生成指令：instructions = GenerateInstruction(M_server, D_server)

2. 客户端初始化：
   2.1 加载小模型 M_client
   2.2 初始化小LoRA：L_small = PEFT(M_client, rank=8)
   
算法：Knowledge Anchor FL 训练轮次
每轮训练流程（Round r）
For round r = 0 to T-1:
    // ==================== 服务器端 ====================
    3. 生成知识锚点：
    prototypes, soft_labels = GeneratePrototypeBank(M_server, D_server)
    KA = {
               instructions: [I_1, I_2, ..., I_N],  // N个指令
               prototype_bank: {
                   prototypes: P ∈ R^(K×d),  // K个原型，每个d维
                   soft_labels: S ∈ R^(K×V), // K个软标签，V是词汇表大小
                   num_prototypes: K
                   avg_prototypes = mean(MSE distances between all pairs in prototype_bank)
               }
           }
    4. 微调：
    M_server ← FineTuning(M_server, D_server, L_large)
    global_large_lora ← L_large.get_peft_state_dict()
    5. 下投影（大LoRA → 小LoRA）：
    small_lora_A = large_lora_A @ P.t()     
    small_lora_B = P @ large_lora_B    
    
    Send(KA, small_lora) → {Client_1, Client_2, Client_3}
    
    // ==================== 客户端 ====================
    For each client i in {1, 2, 3}:
    	KA_i, small_lora_i = Receive(KA, small_lora)
        M_client_i.load_state_dict(small_lora_i)
        incremental_bank_i = []
        6. 本地训练：
        For each batch (x, y) in D_client_i:
        	x' = I_i +[SEP] + x
            // 前向传播
            outputs = M_client_i(x', output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  // 中间层
            h = mean(hidden_states, dim=1)  // [batch, hidden_dim]
                   
            // 任务相关性判断
            For each sample j in batch:
            	h_j = h[j]
                // 计算与原型库的相似度
                similarities = cosine_similarity(h_j, KA_i.prototypes)
                max_sim = max(similarities)
                mse_distances_j = mean_squared_error(h_j, KA_i.prototypes, axis=-1)
                
                If mse_distances_j > 1.5 * avg_prototypes:  // 任务无关
                	// 排除该样本，不参与训练
                    irrelevant_samples++   
                Else:  // 任务相关    
                    If mse_distances_j >= 0.8 * avg_prototypes:
                    	incremental_bank_i.append(h_j)       
            
                	// 动态原型匹配（FedKA方法）
                	top_k_indices, top_k_weights = TopK(similarities, K_p=5)
                           
                	// 结构化锚点对齐损失
                	L_proto = Σ(weight_k * MSE(h_j, prototype_k))  // k in top_k
                           
                	// 软标签蒸馏损失
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
        
        small_lora_delta_i = {
            key: trained_param - received_param 
            for key, trained_param in M_client_i.get_trainable_state_dict().items()
            for received_param in received_small_lora.items() if key in received_small_lora
        }

	Send(sample_size_i, small_lora_update_i, incremental_bank_i) → Server
    
    // ==================== 服务器端 ====================
    7. 聚合：
       7.1 收集所有客户端更新：
           updates = [(sample_size_i, small_lora_delta_i, incremental_bank_i) for i in clients]
       7.2 聚合小LoRA更新（FedAvg）：
           total_weight = Σ(sample_size_i)
           aggregated_small_lora = Σ(sample_size_i / total_weight * small_lora_i)
       7.3 上投影（小LoRA → 大LoRA）：
       	   large_lora_A_delta = aggregated_small_lora_delta_A @ Q.t()  
           large_lora_B_delta = Q @ aggregated_small_lora_delta_B
       7.4 更新全局大LoRA：
           global_large_lora += large_lora_delta
           M_server.load_state_dict(global_large_lora, strict=False)
       7.5 整合增量bank：
           all_new_prototypes = Concat([bank_1, bank_2, bank_3])
           If len(all_new_prototypes) > max_prototypes:
               all_new_prototypes = KMeans_Clustering(all_new_prototypes, max_prototypes)
           KA.prototype_bank.prototypes = Concat(KA.prototype_bank.prototypes, all_new_prototypes)
    
    // ==================== 下一轮 ====================
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
  默认：Q = P⁻¹
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
