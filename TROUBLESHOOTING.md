# 知识锚点算法故障排除指南

## 常见问题

### 1. 层索引越界错误

**错误信息**：
```
IndexError: list index out of range
```

**原因**：
- `intermediate_layer` 配置值超出模型实际层数
- Llama-7B 有32层（索引0-31），如果设置 `intermediate_layer=16` 是安全的
- 但如果使用更小的模型，可能层数不足

**解决方案**：
- 使用负数索引（如 `-2` 表示倒数第二层），会自动适配不同模型
- 代码已添加自动边界检查和回退机制
- 检查模型实际层数：`len(outputs.hidden_states)`

### 2. hidden_states 不存在

**错误信息**：
```
AttributeError: 'ModelOutput' object has no attribute 'hidden_states'
```

**原因**：
- 模型未正确设置 `output_hidden_states=True`
- 某些模型变体可能不支持此功能

**解决方案**：
- 确保模型调用时包含 `output_hidden_states=True`
- 检查模型类型是否支持此功能

### 3. 数据分配错误

**错误信息**：
```
ValueError: 数据比例总和必须为1.0
```

**原因**：
- `server_ratio + client_ratio + test_ratio != 1.0`

**解决方案**：
- 确保三个比例之和为1.0
- 例如：`[0.4, 0.4, 0.2]`

### 4. LoRA映射错误

**错误信息**：
```
KeyError: 'lora_A' or similar
```

**原因**：
- LoRA参数名称不匹配
- 小模型和大模型的LoRA结构不同

**解决方案**：
- 检查LoRA参数名称格式
- 调整映射策略（`direct`, `many_to_one`, `interpolate`）

### 5. 原型库维度不匹配

**错误信息**：
```
RuntimeError: The size of tensor a must match the size of tensor b
```

**原因**：
- 服务器和客户端的隐藏状态维度不一致
- 不同模型的hidden_dim不同

**解决方案**：
- 确保服务器和客户端使用兼容的模型
- 或添加维度投影层

## 调试技巧

### 1. 检查模型层数

```python
# 在代码中添加
outputs = model(input_ids, output_hidden_states=True)
print(f"模型层数: {len(outputs.hidden_states)}")
print(f"隐藏状态维度: {outputs.hidden_states[0].shape}")
```

### 2. 检查数据分配

```python
# 在splitter中添加日志
logger.info(f"数据分配: 服务器={len(server_data)}, "
           f"客户端={[len(c) for c in client_data]}, "
           f"测试={len(test_data)}")
```

### 3. 检查知识锚点

```python
# 检查知识锚点结构
print(f"指令数量: {len(knowledge_anchor['instructions'])}")
print(f"原型数量: {knowledge_anchor['prototype_bank']['num_prototypes']}")
print(f"原型维度: {knowledge_anchor['prototype_bank']['prototypes'].shape}")
```

### 4. 启用详细日志

在配置文件中设置：
```yaml
verbose: 2  # 或更高
```

或在代码中：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化建议

1. **减少原型数量**：如果内存不足，减少 `num_prototypes`
2. **批量处理**：调整 `batch_size` 以适应GPU内存
3. **梯度累积**：使用 `grad_accum_step` 模拟更大的batch size
4. **混合精度**：启用 `is_enable_half: True`

## 联系支持

如果遇到其他问题，请提供：
1. 完整的错误堆栈信息
2. 配置文件内容
3. 模型类型和版本
4. 数据集信息

