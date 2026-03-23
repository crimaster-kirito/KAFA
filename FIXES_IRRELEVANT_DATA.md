# 无关数据注入和训练过程日志修复

## 修复的问题

### 1. ❌ 原问题：无关数据从测试数据中添加
**问题描述**: 原代码从测试数据中抽取一部分作为无关数据，这是不正确的，因为：
- 测试数据应该保持纯净，只在最终评估时使用
- 测试数据不应该参与训练过程

**修复方案**: 
- ✅ 改为从**客户端训练数据**中抽取无关数据
- ✅ 测试数据保持完整，不受影响

**修改文件**: `federatedscope/core/splitters/generic/knowledge_anchor_splitter.py`

### 2. ❌ 原问题：训练过程看不到无关样本被排除
**问题描述**: 训练过程中没有足够的日志输出，无法看到：
- 哪些样本被判断为无关
- 哪些样本被添加到增量bank
- 统计信息

**修复方案**:
- ✅ 添加batch级别的详细日志
- ✅ 添加训练结束时的统计信息
- ✅ 显示相关/无关样本数量和比例
- ✅ 显示增量bank大小

**修改文件**: `federatedscope/llm/trainer/knowledge_anchor_trainer.py`

### 3. ❌ 原问题：无关样本被错误添加到增量bank
**问题描述**: 原代码将无关样本也添加到增量bank，这是不正确的

**修复方案**:
- ✅ 只有**任务相关但无匹配原型**的样本才添加到增量bank
- ✅ 任务无关样本直接排除，不添加到bank

**修改文件**: `federatedscope/llm/trainer/knowledge_anchor_trainer.py`

### 4. ❌ 原问题：comm_manager未初始化错误
**问题描述**: `AttributeError: 'KnowledgeAnchorClient' object has no attribute 'comm_manager'`

**修复方案**:
- ✅ 使用 `register_handlers` 方法而不是直接访问 `comm_manager.register_handler`
- ✅ `register_handlers` 在 `BaseClient` 中定义，不依赖 `comm_manager`

**修改文件**: `federatedscope/llm/knowledge_anchor/client.py`

---

## 修改详情

### 1. Splitter修改（无关数据注入）

**文件**: `federatedscope/core/splitters/generic/knowledge_anchor_splitter.py`

**关键修改**:
```python
# 修改前：从测试数据中抽取
irrelevant_indices = test_indices[:irrelevant_size]
test_indices = test_indices[irrelevant_size:]

# 修改后：从客户端训练数据中抽取
all_client_indices_flat = []
for idx_list in client_indices:
    all_client_indices_flat.extend(idx_list)
irrelevant_indices = random.sample(all_client_indices_flat, irrelevant_size)
# 从客户端数据中移除，然后重新分配给某个客户端
```

**逻辑说明**:
1. 从客户端训练数据中随机抽取一部分作为"无关"数据
2. 这些数据会被注入到某个客户端
3. 在训练时，这些数据会被原型库通过相似度阈值判断为无关并排除
4. 测试数据保持完整，不受影响

### 2. Trainer修改（训练过程日志）

**文件**: `federatedscope/llm/trainer/knowledge_anchor_trainer.py`

#### 2.1 添加batch级别统计
```python
batch_relevant_count = 0
batch_irrelevant_count = 0
batch_new_prototype_count = 0

# 在每个样本判断后更新计数
if is_relevant:
    batch_relevant_count += 1
    # ...
else:
    batch_irrelevant_count += 1
    # 无关样本不添加到bank

# 记录batch统计
logger.info(f"Batch统计: 相关={batch_relevant_count}, 无关={batch_irrelevant_count}, "
           f"新增原型={batch_new_prototype_count}, 增量bank大小={len(self.incremental_bank)}")
```

#### 2.2 修复增量bank逻辑
```python
# 修改前：无关样本也添加到bank
if not is_relevant:
    self.add_to_incremental_bank(hidden_state_i)  # ❌ 错误

# 修改后：只有相关但无原型的样本才添加
if is_relevant:
    if matched_prototype_idx is not None:
        # 有匹配原型，计算对齐损失
    else:
        # 相关但无匹配原型，添加到bank ✅
        self.add_to_incremental_bank(hidden_state_i)
else:
    # 无关样本，直接排除，不添加到bank ✅
    logger.debug(f"样本 {i}: 任务不相关，将被排除")
```

#### 2.3 添加训练结束统计
```python
def _hook_on_fit_end(self, ctx):
    total_samples = self.stats['relevant_samples'] + self.stats['irrelevant_samples']
    if total_samples > 0:
        relevant_ratio = self.stats['relevant_samples'] / total_samples * 100
        irrelevant_ratio = self.stats['irrelevant_samples'] / total_samples * 100
        logger.info("=" * 60)
        logger.info("微调统计信息")
        logger.info("=" * 60)
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"  ✓ 任务相关样本: {self.stats['relevant_samples']} ({relevant_ratio:.2f}%) - 继续训练")
        logger.info(f"  ✗ 任务无关样本: {self.stats['irrelevant_samples']} ({irrelevant_ratio:.2f}%) - 已被排除")
        logger.info(f"  📦 新增原型数: {self.stats['new_prototypes_added']} (增量bank大小: {len(self.incremental_bank)})")
        logger.info("=" * 60)
```

### 3. Client修改（comm_manager修复）

**文件**: `federatedscope/llm/knowledge_anchor/client.py`

**关键修改**:
```python
# 修改前：直接访问comm_manager（此时还未初始化）
self.comm_manager.register_handler('knowledge_anchor', ...)  # ❌ 错误

# 修改后：使用register_handlers方法
self.register_handlers('knowledge_anchor',
                      self.callback_funcs_for_knowledge_anchor,
                      [None])  # ✅ 正确
```

---

## 训练过程日志输出示例

### Batch级别日志
```
Batch统计: 相关=2, 无关=1, 新增原型=0, 增量bank大小=5
✓ Batch处理: 2个相关样本继续训练，1个无关样本被排除
```

### 样本级别日志（调试模式）
```
样本 0: 任务相关，匹配原型 3，计算对齐损失
样本 1: 任务相关但无匹配原型，添加到增量bank（当前bank大小: 6/100）
样本 2: 任务不相关，将被排除（相似度低于阈值 0.5）
```

### 训练结束统计
```
============================================================
微调统计信息
============================================================
总样本数: 1000
  ✓ 任务相关样本: 850 (85.00%) - 继续训练
  ✗ 任务无关样本: 150 (15.00%) - 已被排除
  📦 新增原型数: 25 (增量bank大小: 25/100)
============================================================
```

---

## 工作流程

### 数据分配阶段
1. 数据按40/40/20分配（服务器/客户端/测试）
2. 从客户端训练数据中抽取 `task_irrelevant_ratio` 比例的样本作为"无关"数据
3. 这些无关数据被注入到某个客户端
4. **测试数据保持完整，不受影响**

### 训练阶段
1. 每个batch的每个样本经过模型前向传播
2. 提取中间层隐藏状态
3. 计算与原型库的余弦相似度
4. 判断任务相关性：
   - **相关且有匹配原型** → 计算对齐损失，继续训练
   - **相关但无匹配原型** → 添加到增量bank
   - **无关** → 排除，不参与训练，不添加到bank
5. 记录batch级别统计
6. 如果所有样本都无关，跳过整个batch

### 训练结束
1. 输出详细统计信息
2. 上传增量bank到服务器

---

## 验证要点

运行训练后，您应该能看到：

1. ✅ **数据分配日志**:
   ```
   ✓ 为客户端X注入N个任务不相关样本
   这些样本将在训练时通过原型库相似度判断被识别为无关并排除
   测试数据保持纯净（M个样本），未受影响
   ```

2. ✅ **Batch级别日志**:
   ```
   Batch统计: 相关=X, 无关=Y, 新增原型=Z, 增量bank大小=N
   ```

3. ✅ **无关样本排除日志**:
   ```
   ⚠ 跳过batch：所有N个样本任务不相关（将被排除）
   或
   ✓ Batch处理: X个相关样本继续训练，Y个无关样本被排除
   ```

4. ✅ **新增原型日志**:
   ```
   样本 i: 任务相关但无匹配原型，添加到增量bank（当前bank大小: N/100）
   ✓ 新增原型已添加到增量bank（总数: N/100）
   ```

5. ✅ **训练结束统计**:
   ```
   ============================================================
   微调统计信息
   ============================================================
   总样本数: X
     ✓ 任务相关样本: Y (Z%) - 继续训练
     ✗ 任务无关样本: W (V%) - 已被排除
     📦 新增原型数: N (增量bank大小: N/100)
   ============================================================
   ```

---

## 配置说明

当前配置（`knowledge_anchor_llama2.yaml`）:
```yaml
data:
  splitter_args:
    - task_irrelevant_ratio: 0.1  # 10%的客户端数据作为无关数据

llm:
  knowledge_anchor:
    task_relevance_threshold: 0.5  # 余弦相似度阈值
    max_prototypes: 100  # 增量bank最大大小
```

---

## 总结

✅ **已修复**:
1. 无关数据从训练数据中抽取（不是测试数据）
2. 训练过程有详细的日志输出
3. 无关样本被正确排除，不添加到增量bank
4. 只有相关但无原型的样本才添加到增量bank
5. comm_manager初始化问题已修复

✅ **现在可以看到**:
- 无关样本被排除的过程
- 新原型被添加到增量bank的过程
- 详细的统计信息

现在可以运行训练，您将看到完整的训练过程日志！

