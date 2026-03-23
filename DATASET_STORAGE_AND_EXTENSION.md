# 数据集存储位置和异构数据扩展方案

## 数据集存储位置

### 1. 异构NLP数据集（hetero_nlp_tasks）

这些数据集（`['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']`）**不会预先存储**，而是：

#### 自动下载机制
- **存储位置**: 配置文件中 `data.root` 指定的目录（如 `datasets/`）
- **下载地址**: `https://federatedscope.oss-cn-beijing.aliyuncs.com/{dataset}.zip`
- **下载时机**: 首次运行时，如果数据集目录不存在，会自动下载

#### 代码逻辑
```python
# 在 federatedscope/nlp/hetero_tasks/dataset/get_data.py 中
def _load(self, dataset, split, num_of_client):
    data_dir = os.path.join(self.data_dir, dataset)  # data.root/dataset
    if not os.path.exists(data_dir):
        logger.info(f'Start tp download the dataset {dataset} ...')
        self._download_and_extract(dataset)  # 自动下载
```

#### 下载后的目录结构
```
datasets/  # 或配置的 data.root
├── imdb/
│   ├── train/
│   │   ├── pos/
│   │   └── neg/
│   └── test/
├── agnews/
│   ├── train.csv
│   └── test.csv
├── squad/
│   ├── train.json
│   └── test.json
└── ...
```

### 2. LLM数据集（如gsm8k, alpaca）

这些数据集存储在 `data/` 目录下：
```
data/
├── gsm8k/
├── gsm8k_train.jsonl
├── alpaca/
└── alpaca_data.json
```

---

## 理解您的需求

您提到：**这些不同数据集是异构的，但是所训练的任务是一致的，所以只要解决好异构性问题，就能在这个任务上有所提升**

### 关键理解

1. **任务一致性**: 虽然数据集不同（imdb, agnews等），但都是**文本分类任务**（或都是QA任务、生成任务等）
2. **异构性来源**: 
   - 数据分布不同（不同领域、不同来源）
   - 数据格式可能略有不同
   - 但任务类型相同
3. **提升方向**: 通过解决异构性问题（数据分布、特征对齐等）来提升模型性能

### 对于知识锚点算法的意义

知识锚点算法正好适合这种场景：
- **指令增强**: 将不同数据集的样本对齐到公共语义空间
- **原型对齐**: 通过原型库统一不同数据集的特征表示
- **任务相关性判断**: 过滤掉不相关的数据，保留任务相关的数据

---

## 扩展知识锚点以支持异构数据集

### 方案：支持多个同任务的数据集

可以扩展 `knowledge_anchor_splitter` 以支持多个数据集，但保持任务一致性：

#### 配置示例
```yaml
data:
  type: 'knowledge_anchor_hetero'  # 新的数据类型
  root: data/
  # 多个同任务的数据集
  hetero_datasets:
    - name: 'gsm8k@llm'
      client_ratio: 0.15  # 15%的客户端使用这个数据集
    - name: 'alpaca@llm'
      client_ratio: 0.15  # 15%的客户端使用这个数据集
    - name: 'dolly-15k@llm'
      client_ratio: 0.10  # 10%的客户端使用这个数据集
  # 或者更简单的方式：指定数据集列表和分配方式
  hetero_data_list: ['gsm8k@llm', 'alpaca@llm', 'dolly-15k@llm']
  hetero_distribution: 'uniform'  # 或 'dirichlet' 或 'custom'
  
  splitter: 'knowledge_anchor_hetero'
  splitter_args:
    - server_ratio: 0.4
      client_ratio: 0.4
      test_ratio: 0.2
      # 异构数据集相关参数
      hetero_datasets: ['gsm8k@llm', 'alpaca@llm', 'dolly-15k@llm']
      hetero_distribution: 'uniform'  # 均匀分配或按比例分配
```

#### 实现思路

1. **扩展Splitter** (`knowledge_anchor_splitter.py`)
```python
class KnowledgeAnchorSplitter(BaseSplitter):
    def __init__(self, client_num, 
                 server_ratio=0.4, client_ratio=0.4, test_ratio=0.2,
                 hetero_datasets=None,  # 新增：多个数据集
                 hetero_distribution='uniform',  # 分配方式
                 **kwargs):
        # 如果指定了异构数据集，使用异构模式
        if hetero_datasets:
            self.hetero_mode = True
            self.hetero_datasets = hetero_datasets
            self.hetero_distribution = hetero_distribution
        else:
            self.hetero_mode = False
            # 原有的同构模式逻辑
```

2. **数据加载逻辑**
```python
def __call__(self, dataset, prior=None, **kwargs):
    if self.hetero_mode:
        # 异构模式：加载多个数据集
        return self._split_hetero_datasets()
    else:
        # 同构模式：原有逻辑
        return self._split_single_dataset(dataset)
```

3. **分配策略**
   - **uniform**: 均匀分配，每个数据集分配给相同数量的客户端
   - **proportional**: 按数据集大小比例分配
   - **dirichlet**: 使用Dirichlet分布分配（模拟更真实的异构场景）

---

## 实际应用场景

### 场景1：多个数学推理数据集
```yaml
hetero_datasets:
  - 'gsm8k@llm'      # 小学数学题
  - 'math@llm'        # 数学问题
  - 'svamp@llm'       # 数学应用题
```
**任务**: 都是数学推理，但数据来源和格式不同

### 场景2：多个指令跟随数据集
```yaml
hetero_datasets:
  - 'alpaca@llm'      # Alpaca指令数据
  - 'dolly-15k@llm'   # Dolly指令数据
  - 'self-instruct@llm'  # Self-Instruct数据
```
**任务**: 都是指令跟随，但指令风格和领域不同

### 场景3：多个代码生成数据集
```yaml
hetero_datasets:
  - 'code_search_net@llm'
  - 'rosetta_alpaca@llm'
  - 'codealpaca@llm'
```
**任务**: 都是代码生成，但代码类型和复杂度不同

---

## 知识锚点的优势

在这种"任务一致、数据异构"的场景下，知识锚点算法特别有效：

### 1. 指令增强
- 将不同数据集的样本通过指令模板对齐
- 统一不同数据源的格式和语义

### 2. 原型对齐
- 提取不同数据集的共同特征
- 通过原型库实现跨数据集的特征对齐

### 3. 任务相关性判断
- 过滤掉不相关的数据
- 确保所有客户端都在学习相同的任务

### 4. 增量bank机制
- 捕获不同数据集中的新原型
- 逐步丰富知识锚点

---

## 实现建议

### 阶段1：基础扩展（推荐先实现）
1. 修改 `knowledge_anchor_splitter.py` 支持多个数据集
2. 保持任务类型一致（如都是数学推理）
3. 使用均匀分配或按比例分配

### 阶段2：高级功能
1. 支持Dirichlet分布分配（更真实的异构）
2. 支持数据集权重（某些数据集更重要）
3. 支持动态数据集切换

### 阶段3：优化
1. 知识锚点生成时考虑多个数据集
2. 原型库从多个数据集提取
3. 指令生成融合多个数据集的特点

---

## 快速开始

### 当前状态
- ✅ 支持单个数据集 + non-IID分布
- ❌ 不支持多个数据集

### 扩展后
- ✅ 支持单个数据集 + non-IID分布
- ✅ 支持多个同任务数据集 + 异构分配
- ✅ 知识锚点自动适配多个数据集

---

## 总结

1. **数据集存储**: 异构NLP数据集会自动下载到 `data.root` 目录
2. **您的需求**: 多个数据集但任务一致，通过解决异构性提升性能
3. **知识锚点优势**: 特别适合这种场景，通过指令和原型对齐不同数据集
4. **扩展方案**: 可以扩展splitter支持多个同任务数据集

如果需要在知识锚点中实现多数据集支持，我可以帮您实现！

