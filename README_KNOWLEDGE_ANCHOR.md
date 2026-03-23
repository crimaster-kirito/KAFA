# 知识锚点算法实现说明

## 概述

本实现提供了一个完整的知识锚点（Knowledge Anchor）联邦学习算法框架，用于在异构联邦学习场景下微调大语言模型。

## 核心特性

### 1. 数据分配策略
- **40% 服务器数据**：用于服务器端LLM微调和知识锚点生成
- **40% 客户端数据**：分配给3个客户端，支持non-IID分布
- **20% 测试数据**：用于评估模型性能
- 支持异构分布和任务不相关数据的注入

### 2. 知识锚点组成

#### 指令（Instructions）
- 服务器端LLM通过采样公共数据集生成指令模板
- 用于客户端输入增强，引导数据向公共语义空间对齐

#### 原型库（Prototype Bank）
- 提取公共数据集在LLM中间层的隐藏状态
- 使用K-means聚类聚合形成原型
- 用于客户端任务相关性判断和原型对齐

### 3. 客户端微调流程

1. **接收知识锚点**：从服务器接收指令和原型库
2. **输入增强**：使用指令模板增强私有数据输入
3. **原型对齐**：在中间层与原型库进行对齐
4. **任务相关性判断**：
   - 相关且有对应原型 → 继续微调
   - 相关但无对应原型 → 加入增量bank
   - 无关 → 跳过该数据
5. **上传更新**：上传增量bank和LoRA参数

### 4. LoRA映射

- 支持小模型（如DistilLlama）LoRA到大模型（Llama-7B）LoRA的映射
- 映射策略：
  - `direct`: 直接映射
  - `many_to_one`: 多到一映射
  - `interpolate`: 插值映射（默认）

## 文件结构

```
federatedscope/
├── core/
│   ├── splitters/generic/
│   │   └── knowledge_anchor_splitter.py      # 数据分配器
│   └── aggregators/
│       └── knowledge_anchor_aggregator.py    # 服务器聚合器
├── llm/
│   ├── misc/
│   │   ├── knowledge_anchor_generator.py     # 知识锚点生成器
│   │   └── lora_mapper.py                     # LoRA映射器
│   └── trainer/
│       └── knowledge_anchor_trainer.py        # 客户端trainer
└── core/configs/
    └── cfg_llm.py                             # 配置文件（已更新）

fedbiot_script/
└── knowledge_anchor_llama2.yaml               # 示例配置文件
```

## 使用方法

### 1. 配置文件

使用 `fedbiot_script/knowledge_anchor_llama2.yaml` 作为基础配置，主要参数：

```yaml
federate:
  client_num: 3
  method: "KnowledgeAnchor"
  aggregator_type: "knowledge_anchor"

data:
  splits: [0.4, 0.4, 0.2]  # 服务器、客户端、测试
  splitter: 'knowledge_anchor'
  splitter_args:
    - non_iid_alpha: 0.5  # 控制non-IID程度

trainer:
  type: knowledge_anchor_client

llm:
  knowledge_anchor:
    num_instructions: 10
    num_prototypes: 50
    task_relevance_threshold: 0.5
    prototype_alignment_weight: 0.3
```

### 2. 运行微调

```bash
python federatedscope/main.py --cfg fedbiot_script/knowledge_anchor_llama2.yaml
```

## 关键组件说明

### KnowledgeAnchorSplitter
- 实现40/40/20数据分配
- 支持Dirichlet分布进行non-IID分配
- 支持任务不相关数据注入

### KnowledgeAnchorGenerator
- 生成指令模板（基于LLM采样）
- 提取中间层隐藏状态
- K-means聚类生成原型库

### KnowledgeAnchorClientTrainer
- 输入增强（使用指令）
- 原型对齐损失
- 任务相关性判断
- 增量bank管理

### LoRAMapper
- 小模型到大模型的LoRA参数映射
- 支持多种映射策略

### KnowledgeAnchorAggregator
- 聚合客户端LoRA参数
- 整合增量bank
- 更新知识锚点

## 注意事项

1. **客户端模型**：需要在代码中指定客户端使用的小模型（如DistilLlama），当前配置中需要手动设置
2. **数据格式**：确保数据集格式符合LLM训练要求
3. **内存管理**：原型库大小可通过`max_prototypes`控制，超过限制会自动重新聚类
4. **任务相关性阈值**：根据实际任务调整`task_relevance_threshold`

## 扩展建议

1. **指令生成优化**：可以改进指令生成策略，使用更好的prompt工程
2. **原型更新策略**：可以实现更智能的原型更新机制
3. **多任务支持**：扩展支持多任务场景
4. **隐私保护**：可以添加差分隐私等隐私保护机制

## 与FedBiOT的区别

| 特性 | FedBiOT | Knowledge Anchor |
|------|---------|------------------|
| 服务器数据 | Alpaca（通用） | 任务相关（40%） |
| 客户端数据 | 任务相关 | 任务相关（40%，支持异构） |
| 客户端模型 | Emulator | 小型LLM（DistilLlama） |
| 知识传递 | 软标签 | 指令+原型库 |
| 任务对齐 | 蒸馏 | 原型对齐+相关性判断 |
| 微调方式 | Offsite-tuning | LoRA微调 |

## 技术说明

### 微调 vs 训练

本算法是**微调（Fine-tuning）**而非训练（Training）：
- **预训练模型**：服务器端使用预训练的Llama-7B，客户端使用预训练的DistilLlama
- **参数高效微调**：仅更新LoRA参数，不更新全部模型参数
- **任务适应**：在预训练模型基础上适应特定任务，而非从头学习
- **数据量**：使用相对较少的任务相关数据进行微调

## 后续工作

1. 实现服务器端worker，集成知识锚点生成
2. 实现客户端worker，集成知识锚点trainer
3. 添加评估指标和可视化
4. 完善文档和示例

