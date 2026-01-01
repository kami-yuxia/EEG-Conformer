import matplotlib.pyplot as plt
import numpy as np

# 设置风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'

# ==========================================
# 图 1: 消融实验 (Ablation Study)
# ==========================================
labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'Avg']
# 数据来自你的 txt 文件
baseline = [91.32, 60.42, 92.36, 81.60, 67.71, 64.58, 94.44, 86.81, 87.50, 80.75]
no_trans = [84.72, 63.89, 90.97, 76.39, 51.74, 59.72, 89.93, 85.76, 84.38, 76.39]
no_aug   = [87.85, 57.64, 91.67, 76.39, 51.74, 61.81, 93.75, 81.94, 81.94, 76.08]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, baseline, width, label='Baseline (Ours)', color='#4c72b0')
rects2 = ax.bar(x, no_trans, width, label='w/o Transformer', color='#55a868')
rects3 = ax.bar(x + width, no_aug, width, label='w/o Augmentation', color='#c44e52')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Ablation Study on BCI IV 2a (All Subjects)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(40, 100)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('ablation_plot.png', dpi=300)
plt.show()

# ==========================================
# 图 2: Mish 优化对比 (Mish vs Baseline)
# ==========================================
# 数据来自你的 mish_result.txt 和 baseline
mish = [91.67, 62.50, 93.40, 83.33, 73.96, 66.67, 96.18, 87.15, 88.54, 82.60]
# 计算提升
improvement = np.array(mish) - np.array(baseline)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))

# 画折线图对比
ax.plot(labels, baseline, marker='o', linestyle='--', label='Baseline (ELU)', color='gray', alpha=0.7)
ax.plot(labels, mish, marker='s', linestyle='-', label='Mish Optimization', color='#d62728', linewidth=2)

# 在每个点标出提升数值
for i, val in enumerate(improvement):
    ax.annotate(f'+{val:.2f}%', (i, mish[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#d62728', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance Improvement with Mish Activation')
ax.set_ylim(40, 105)
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# 填充提升区域
ax.fill_between(labels, baseline, mish, color='#d62728', alpha=0.1)

plt.tight_layout()
plt.savefig('mish_plot.png', dpi=300)
plt.show()