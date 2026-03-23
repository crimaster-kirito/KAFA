#!/usr/bin/env python3
"""
快速检查GPU使用情况的脚本
"""
import torch
import sys

print("=" * 60)
print("GPU使用情况检查")
print("=" * 60)

# 检查CUDA是否可用
if torch.cuda.is_available():
    print(f"✓ CUDA可用")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    print(f"  当前GPU: {torch.cuda.current_device()}")
    print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 检查每个GPU的内存
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3  # GB
        print(f"\n  GPU {i}:")
        print(f"    总内存: {total_memory:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
    
    # 检查当前内存使用
    print(f"\n当前GPU内存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 测试GPU计算
    print(f"\n测试GPU计算:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"  ✓ GPU计算测试成功")
        print(f"    结果tensor设备: {z.device}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ GPU计算测试失败: {e}")
        sys.exit(1)
else:
    print("✗ CUDA不可用")
    print("  将使用CPU训练（会很慢）")
    sys.exit(1)

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)

