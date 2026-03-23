# FedBiOT 异构数据设置指南

## 概述

FedBiOT框架支持两种数据设置方式：
1. **同构数据**：所有客户端使用相同的数据集（如 `gsm8k@llm`）
2. **异构数据**：不同客户端使用不同的数据集（如 `hetero_nlp_tasks`）

---

## 方式1：同构数据（当前知识锚点使用）

### 配置示例
```yaml
data:
  type: 'gsm8k@llm'  # 所有客户端使用同一个数据集
  splitter: 'knowledge_anchor'  # 使用自定义splitter分配数据
  splitter_args:
    - server_ratio: 0.4
      client_ratio: 0.4
      test_ratio: 0.2
      non_iid_alpha: 0.5  # 通过Dirichlet分布实现non-IID
      task_irrelevant_ratio: 0.1  # 注入任务不相关数据
```

### 特点
- 所有客户端使用**同一个数据集**（如gsm8k）
- 通过splitter实现**数据分布异构**（non-IID）
- 可以注入**任务不相关数据**模拟异构场景

### 相关代码
- **Splitter**: `federatedscope/core/splitters/generic/knowledge_anchor_splitter.py`
- **数据加载**: `federatedscope/llm/dataloader/dataloader.py`

---

## 方式2：异构数据（不同数据集）

### 配置示例
```yaml
data:
  type: 'hetero_nlp_tasks'  # 异构NLP任务
  root: datasets/
  hetero_data_name: ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']  # 多个数据集
  num_of_client_for_data: [1, 3, 3, 2, 5, 4]  # 每个数据集对应的客户端数量
  # 总客户端数 = 1+3+3+2+5+4 = 18

# 客户端分组配置（可选）
client_group_1:
  model:
    task: imdb  # 客户端组1使用imdb数据集
  train:
    local_update_steps: 200
client_group_2:
  model:
    task: agnews  # 客户端组2使用agnews数据集
  train:
    local_update_steps: 200
# ... 其他组
```

### 关键配置项说明

#### 1. `hetero_data_name`
- **类型**: `List[str]`
- **说明**: 指定要使用的多个数据集名称
- **支持的数据集**: `['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']`
- **示例**: `['imdb', 'agnews', 'squad']`

#### 2. `num_of_client_for_data`
- **类型**: `List[int]`
- **说明**: 每个数据集对应的客户端数量
- **长度**: 必须与 `hetero_data_name` 长度相同
- **总客户端数**: 所有数字之和
- **示例**: `[1, 3, 3, 2, 5, 4]` 表示：
  - imdb: 1个客户端
  - agnews: 3个客户端
  - squad: 3个客户端
  - newsqa: 2个客户端
  - cnndm: 5个客户端
  - msqg: 4个客户端
  - **总计**: 18个客户端

#### 3. 客户端分组配置（可选）
- 可以为每个数据集组配置不同的参数
- 配置文件名通常为 `config_client_*.yaml`
- 每个 `client_group_N` 对应一个数据集

### 完整配置示例

#### 主配置文件 (`config.yaml`)
```yaml
use_gpu: True
device: 0
seed: 12345
outdir: exp/

federate:
  mode: standalone
  method: fedavg
  total_round_num: 100
  client_num: 18  # 必须等于 num_of_client_for_data 的总和

data:
  type: hetero_nlp_tasks
  root: datasets/
  hetero_data_name: ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']
  num_of_client_for_data: [1, 3, 3, 2, 5, 4]  # 总和=18
  num_workers: 0
  cache_dir: cache/

model:
  type: atc_model
  model_type: google/bert_uncased_L-2_H-128_A-2

trainer:
  type: atc_trainer
```

#### 客户端分组配置文件 (`config_client.yaml`)
```yaml
# 客户端组1: imdb (1个客户端)
client_group_1:
  data:
    batch_size: 32
    max_seq_len: 128
  model:
    task: imdb
    num_labels: 2
  train:
    local_update_steps: 200
  eval:
    metrics: ['acc']

# 客户端组2: agnews (3个客户端)
client_group_2:
  data:
    batch_size: 32
    max_seq_len: 128
  model:
    task: agnews
    num_labels: 4
  train:
    local_update_steps: 200
  eval:
    metrics: ['acc']

# 客户端组3: squad (3个客户端)
client_group_3:
  data:
    batch_size: 32
    max_seq_len: 384
    max_query_len: 128
    trunc_stride: 128
  model:
    task: squad
    num_labels: 2
    n_best_size: 20
    max_answer_len: 30
  train:
    local_update_steps: 200
  eval:
    metrics: ['squad']

# ... 其他组
```

### 相关代码文件

#### 1. 数据加载器
**文件**: `federatedscope/nlp/hetero_tasks/dataloader/dataloader.py`

**关键类**:
- `HeteroNLPDataset`: 异构NLP数据集类
- `load_heteroNLP_data()`: 加载异构数据的主函数

**关键逻辑**:
```python
# 1. 加载多个数据集
data = HeteroNLPDataLoader(
    data_dir=self.root,
    data_name=self.data_name,  # ['imdb', 'agnews', ...]
    num_of_clients=self.num_of_clients  # [1, 3, 3, ...]
).get_data()

# 2. 为每个客户端分配对应的数据集
for client_id in range(1, config.federate.client_num + 1):
    cur_task = applied_cfg.model.downstream_tasks[client_id - 1]
    # 处理该客户端对应的数据集
```

#### 2. 数据集加载器
**文件**: `federatedscope/nlp/hetero_tasks/dataset/get_data.py`

**关键类**:
- `HeteroNLPDataLoader`: 加载多个异构数据集

**关键方法**:
```python
def get_data(self):
    # 为每个数据集加载数据并分配给对应数量的客户端
    for each_data, each_client_num in zip(self.data_name, self.num_of_clients):
        train_and_val_data = self._load(each_data, 'train', each_client_num)
        # 将数据分配给 each_client_num 个客户端
        self.train_data.extend(each_train_data)
```

#### 3. 配置处理
**文件**: `federatedscope/nlp/hetero_tasks/dataloader/dataloader.py`

**关键函数**:
```python
def modified_cfg(cfg, cfg_client):
    # 根据 num_of_client_for_data 为每个客户端分配配置
    for group_id, num_clients in enumerate(cfg.data.num_of_client_for_data):
        group_cfg = cfg_client['client_group_{}'.format(group_id + 1)]
        # 为这 num_clients 个客户端应用 group_cfg
        for client_id in range(client_start_id, client_start_id + num_clients):
            cfg_client['client_{}'.format(client_id)] = group_cfg
```

---

## 两种方式的对比

| 特性 | 同构数据（方式1） | 异构数据（方式2） |
|------|------------------|------------------|
| **数据集数量** | 1个（如gsm8k） | 多个（如6个） |
| **客户端数据来源** | 同一数据集的不同部分 | 完全不同的数据集 |
| **异构程度** | 数据分布异构（non-IID） | 任务异构（不同任务） |
| **配置复杂度** | 简单 | 较复杂（需要分组配置） |
| **适用场景** | 同一任务的不同数据分布 | 多个不同任务 |
| **当前知识锚点** | ✅ 使用 | ❌ 未使用 |

---

## 如何在知识锚点中实现异构数据

### 方案1：扩展现有Splitter（推荐）
修改 `knowledge_anchor_splitter.py`，支持多个数据集：

```python
class KnowledgeAnchorSplitter(BaseSplitter):
    def __init__(self, client_num, 
                 server_ratio=0.4, client_ratio=0.4, test_ratio=0.2,
                 hetero_data_name=None,  # 新增：多个数据集
                 num_of_client_for_data=None,  # 新增：每个数据集的客户端数
                 **kwargs):
        # 如果指定了异构数据集，使用异构模式
        if hetero_data_name:
            self.hetero_mode = True
            self.hetero_data_name = hetero_data_name
            self.num_of_client_for_data = num_of_client_for_data
        else:
            self.hetero_mode = False
            # 原有的同构模式逻辑
```

### 方案2：使用hetero_nlp_tasks
直接使用 `hetero_nlp_tasks` 作为数据类型，但需要：
1. 修改数据加载逻辑以支持LLM格式
2. 修改知识锚点生成器以处理多个数据集
3. 修改客户端trainer以适配不同任务

---

## 实际使用示例

### 示例1：同构数据（当前）
```yaml
data:
  type: 'gsm8k@llm'
  splitter: 'knowledge_anchor'
  splitter_args:
    - server_ratio: 0.4
      client_ratio: 0.4
      test_ratio: 0.2
      non_iid_alpha: 0.5
```

### 示例2：异构数据（如果实现）
```yaml
data:
  type: 'hetero_nlp_tasks'
  root: datasets/
  hetero_data_name: ['alpaca', 'gsm8k', 'dolly-15k']
  num_of_client_for_data: [1, 1, 1]  # 3个客户端，每个使用不同数据集
```

---

## 关键代码位置总结

### 同构数据相关
1. **Splitter**: `federatedscope/core/splitters/generic/knowledge_anchor_splitter.py`
2. **数据加载**: `federatedscope/llm/dataloader/dataloader.py`
3. **配置**: `fedbiot_script/knowledge_anchor_llama2.yaml`

### 异构数据相关
1. **数据加载器**: `federatedscope/nlp/hetero_tasks/dataloader/dataloader.py`
2. **数据集加载**: `federatedscope/nlp/hetero_tasks/dataset/get_data.py`
3. **配置示例**: `federatedscope/nlp/hetero_tasks/baseline/config_*.yaml`

---

## 注意事项

1. **客户端数量匹配**: `num_of_client_for_data` 的总和必须等于 `federate.client_num`
2. **数据集格式**: 异构数据需要每个数据集都有对应的处理函数
3. **任务类型**: 不同数据集可能对应不同的任务类型（分类、QA、生成等）
4. **评估指标**: 不同任务需要不同的评估指标

---

## 参考文件

- 异构数据配置示例: `federatedscope/nlp/hetero_tasks/baseline/config_fedavg.yaml`
- 客户端分组配置: `federatedscope/nlp/hetero_tasks/baseline/config_client_fedavg.yaml`
- 数据加载实现: `federatedscope/nlp/hetero_tasks/dataloader/dataloader.py`

