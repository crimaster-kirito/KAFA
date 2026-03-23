# 内存优化指南

## CUDA内存不足问题

如果遇到 `CUDA out of memory` 错误，可以尝试以下优化：

### 1. 减少Batch Size
```yaml
dataloader:
  batch_size: 1  # 从4减少到1
```

### 2. 增加梯度累积步数
```yaml
llm:
  grad_accum_step: 8  # 从4增加到8，保持有效batch size
```

### 3. 减少序列长度
```yaml
llm:
  tok_len: 512  # 从800减少到512
```

### 4. 启用Adapter CPU移动
```yaml
llm:
  adapter:
    mv_to_cpu: True  # 训练后将adapter移到CPU
```

### 5. 启用混合精度训练
```yaml
train:
  is_enable_half: True  # 使用FP16/BF16
```

### 6. 共享本地模型
```yaml
federate:
  share_local_model: True  # 客户端共享基础模型
```

### 7. 清理GPU缓存
代码中已添加自动清理机制，在训练结束后清理GPU缓存。

### 8. 检查其他进程
```bash
# 查看GPU使用情况
nvidia-smi

# 如果有其他进程占用GPU，可以kill掉
# 或者使用其他GPU
device: 1  # 使用GPU 1
```

### 9. 使用DeepSpeed（如果有多GPU）
```yaml
llm:
  deepspeed:
    use: True
    ds_config: "path/to/ds_config.json"
```

### 10. 减少原型数量
```yaml
llm:
  knowledge_anchor:
    num_prototypes: 30  # 从50减少到30
    max_prototypes: 50  # 从100减少到50
```

## 当前配置优化

已优化的配置：
- ✅ batch_size: 1
- ✅ grad_accum_step: 8
- ✅ tok_len: 512
- ✅ mv_to_cpu: True
- ✅ is_enable_half: True
- ✅ share_local_model: True

## 如果仍然OOM

1. **进一步减少batch size到1**
2. **减少序列长度到256或更小**
3. **检查是否有其他进程占用GPU**
4. **考虑使用CPU训练（虽然会很慢）**
5. **使用更小的模型（如Llama-7B -> Llama-3B）**

