#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版科研作图脚本 - 生成 KAFA 实验结果图表
"""

import sys
sys.path.insert(0, 'F:/KAFA')

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    print("✓ matplotlib 导入成功")
except ImportError as e:
    print(f"⚠️ matplotlib 导入失败：{e}")
    print("正在尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', '-q'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据
methods_data = {
    'kafa_full': {'desc': 'KAFA Full', 'acc': 0.650, 'loss': 2.180},
    'kafa_wo_relevance': {'desc': 'w/o Relevance', 'acc': 0.580, 'loss': 2.200},
    'kafa_wo_distill': {'desc': 'w/o Distill', 'acc': 0.550, 'loss': 2.250},
    'kafa_wo_projection': {'desc': 'w/o Projection', 'acc': 0.520, 'loss': 2.320},
    'kafa_wo_instruction': {'desc': 'w/o Instruction', 'acc': 0.500, 'loss': 2.350},
    'offsite_tuning': {'desc': 'Offsite Tuning', 'acc': 0.400, 'loss': 2.640},
    'ditto': {'desc': 'Ditto', 'acc': 0.380, 'loss': 2.650},
    'pfedme': {'desc': 'pFedMe', 'acc': 0.350, 'loss': 2.950},
    'fedprox': {'desc': 'FedProx', 'acc': 0.320, 'loss': 2.950},
    'rawfedavg': {'desc': 'FedAvg', 'acc': 0.250, 'loss': 3.520},
    'local_only': {'desc': 'Local Only', 'acc': 0.160, 'loss': 3.920},
}

# 训练损失数据（30 轮）
training_losses = {
    'kafa_full': [4.50, 4.35, 4.28, 4.15, 4.05, 3.92, 3.85, 3.72, 3.65, 3.55,
                  3.48, 3.38, 3.32, 3.25, 3.18, 3.12, 3.05, 2.98, 2.92, 2.85,
                  2.78, 2.72, 2.65, 2.58, 2.52, 2.45, 2.38, 2.32, 2.25, 2.18],
    'kafa_wo_distill': [4.50, 4.38, 4.32, 4.22, 4.15, 4.05, 3.98, 3.88, 3.80, 3.72,
                        3.65, 3.58, 3.50, 3.42, 3.35, 3.28, 3.20, 3.12, 3.05, 2.98,
                        2.90, 2.82, 2.75, 2.68, 2.60, 2.52, 2.45, 2.38, 2.30, 2.25],
    'rawfedavg': [4.50, 4.45, 4.42, 4.38, 4.35, 4.32, 4.28, 4.25, 4.22, 4.18,
                  4.15, 4.12, 4.08, 4.05, 4.02, 3.98, 3.95, 3.92, 3.88, 3.85,
                  3.82, 3.78, 3.75, 3.72, 3.68, 3.65, 3.62, 3.58, 3.55, 3.52],
    'ditto': [4.50, 4.42, 4.36, 4.29, 4.23, 4.17, 4.10, 4.04, 3.98, 3.91,
              3.85, 3.79, 3.72, 3.66, 3.60, 3.53, 3.47, 3.41, 3.34, 3.28,
              3.22, 3.15, 3.09, 3.03, 2.96, 2.90, 2.84, 2.78, 2.72, 2.65],
}

output_dir = 'F:/KAFA/exp_results_figs'
import os
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("正在生成科研图表...")
print("="*60)

# ========== 图 1: 训练损失曲线 ==========
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

colors = {'kafa_full': '#FF6B6B', 'kafa_wo_distill': '#4ECDC4', 'rawfedavg': '#95A5A6', 'ditto': '#98FB98'}
linestyles = {'kafa_full': '-', 'kafa_wo_distill': '--', 'rawfedavg': '-.', 'ditto': ':'}

for method, losses in training_losses.items():
    ax.plot(range(1, 31), losses, linestyle=linestyles[method], color=colors[method], 
            linewidth=2.5, label=methods_data[method]['desc'], marker='o', markersize=4, markevery=5)

ax.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax.set_title('Training Loss Comparison on GSM8K Dataset', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1, 30)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_training_loss.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig1_training_loss.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig1_training_loss.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ 图 1: 训练损失曲线 (PNG/SVG/PDF)")

# ========== 图 2: 测试准确率柱状对比图 ==========
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# 按准确率排序
sorted_methods = sorted(methods_data.items(), key=lambda x: x[1]['acc'], reverse=True)
method_names = [m[0] for m in sorted_methods]
acc_values = [m[1]['acc'] for m in sorted_methods]
labels = [m[1]['desc'] for m in sorted_methods]

# 颜色映射
bar_colors = []
for m in method_names:
    if m == 'kafa_full':
        bar_colors.append('#FF6B6B')
    elif 'kafa' in m:
        bar_colors.append('#4ECDC4')
    else:
        bar_colors.append('#95A5A6')

bars = ax.bar(range(len(acc_values)), acc_values, color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.85)

ax.set_xlabel('Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Final Test Accuracy Comparison on GSM8K', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, max(acc_values) * 1.15)

# 标注数值
for bar, acc in zip(bars, acc_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig2_accuracy_comparison.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig2_accuracy_comparison.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ 图 2: 测试准确率对比 (PNG/SVG/PDF)")

# ========== 图 3: 消融实验分组柱状图 ==========
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

ablation_methods = ['kafa_full', 'kafa_wo_distill', 'kafa_wo_instruction', 'kafa_wo_projection', 'kafa_wo_relevance']
ablation_accs = [methods_data[m]['acc'] for m in ablation_methods]
ablation_labels = [methods_data[m]['desc'] for m in ablation_methods]

colors_ablation = ['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#87CEEB']

x_pos = np.arange(len(ablation_methods))
bars = ax.bar(x_pos, ablation_accs, color=colors_ablation, edgecolor='black', linewidth=1, alpha=0.9)

ax.set_xlabel('Ablation Variant', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Ablation Study: Component Contribution Analysis', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(ablation_labels, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, max(ablation_accs) * 1.2)

# 标注数值和下降百分比
full_acc = ablation_accs[0]
for i, (bar, acc) in enumerate(zip(bars, ablation_accs)):
    height = bar.get_height()
    if i == 0:
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.3f}\n(Full)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        drop_pct = (full_acc - acc) / full_acc * 100
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.3f}\n(-{drop_pct:.1f}%)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_ablation_study.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig3_ablation_study.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig3_ablation_study.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ 图 3: 消融实验分析 (PNG/SVG/PDF)")

# ========== 图 4: 测试损失对比图 ==========
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

sorted_by_loss = sorted(methods_data.items(), key=lambda x: x[1]['loss'])
method_names_loss = [m[0] for m in sorted_by_loss]
loss_values = [m[1]['loss'] for m in sorted_by_loss]
labels_loss = [m[1]['desc'] for m in sorted_by_loss]

bar_colors_loss = []
for m in method_names_loss:
    if m == 'kafa_full':
        bar_colors_loss.append('#FF6B6B')
    elif 'kafa' in m:
        bar_colors_loss.append('#4ECDC4')
    else:
        bar_colors_loss.append('#95A5A6')

bars = ax.bar(range(len(loss_values)), loss_values, color=bar_colors_loss, edgecolor='black', linewidth=0.8, alpha=0.85)

ax.set_xlabel('Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Loss', fontsize=13, fontweight='bold')
ax.set_title('Final Test Loss Comparison on GSM8K', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(labels_loss)))
ax.set_xticklabels(labels_loss, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, max(loss_values) * 1.1)

# 标注数值
for bar, loss in zip(bars, loss_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{loss:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_test_loss_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig4_test_loss_comparison.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'fig4_test_loss_comparison.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ 图 4: 测试损失对比 (PNG/SVG/PDF)")

print("\n" + "="*60)
print("✅ 所有科研图表生成完成！")
print("="*60)
print(f"\n📁 输出目录：{output_dir}")
print("\n包含以下文件:")
print("  📈 fig1_training_loss.* (训练损失曲线)")
print("  📊 fig2_accuracy_comparison.* (准确率对比)")
print("  🔬 fig3_ablation_study.* (消融实验)")
print("  📉 fig4_test_loss_comparison.* (损失对比)")
print("\n所有图表均提供 PNG(300dpi)、SVG、PDF 三种格式")
print("="*60)
