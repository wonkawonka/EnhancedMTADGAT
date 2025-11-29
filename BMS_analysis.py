# BMS 数据集可视化分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import pickle

# 设置中文字体以避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '')
from utils import get_data, get_data_dim
from plotting import Plotter

## 1. 加载 BMS 数据集
# 加载 BMS 数据集
try:
    (train_data, _), (test_data, test_labels) = get_data("BMS", normalize=False)
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # BMS 特征名称
    feature_names = ['SYS_Vol', 'SYS_I', 'SYS_DSOC', 'SYS_SOH', 'SYS_Vmax']
    print(f"特征名称: {feature_names}")
except FileNotFoundError as e:
    print(f"数据文件未找到: {e}")
    print("请确保已运行预处理脚本生成BMS数据集")
    sys.exit(1)

## 2. 训练集数据趋势可视化
# 可视化训练集数据趋势
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('BMS 训练集数据趋势', fontsize=16)

for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
    ax.plot(train_data[:, i], linewidth=0.8)
    ax.set_title(f'{feature_name} 趋势')
    ax.set_xlabel('时间点')
    ax.set_ylabel(feature_name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 3. 测试集数据趋势可视化
# 可视化测试集数据趋势
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('BMS 测试集数据趋势', fontsize=16)

for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
    ax.plot(test_data[:, i], linewidth=0.8, label='正常数据')

    # 标记异常点
    anomaly_points = np.where(test_labels == 1)[0]
    if len(anomaly_points) > 0:
        ax.scatter(anomaly_points, test_data[anomaly_points, i],
                   c='red', s=5, label='异常点', zorder=5)

    ax.set_title(f'{feature_name} 趋势')
    ax.set_xlabel('时间点')
    ax.set_ylabel(feature_name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 4. 训练集与测试集数据分布对比
# 对比训练集和测试集的数据分布
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('BMS 训练集与测试集数据分布对比', fontsize=16)

# 训练集分布
for i, (ax, feature_name) in enumerate(zip(axes[0], feature_names)):
    ax.hist(train_data[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax.set_title(f'{feature_name} (训练集)')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('频率')
    ax.grid(True, alpha=0.3)

# 测试集分布
for i, (ax, feature_name) in enumerate(zip(axes[1], feature_names)):
    ax.hist(test_data[:, i], bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    ax.set_title(f'{feature_name} (测试集)')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('频率')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 5. 异常点详细分析
# 分析测试集中的异常点
anomaly_indices = np.where(test_labels == 1)[0]
normal_indices = np.where(test_labels == 0)[0]

print(f"测试集中异常点数量: {len(anomaly_indices)}")
print(f"测试集中正常点数量: {len(normal_indices)}")
print(f"异常点占比: {len(anomaly_indices) / len(test_labels) * 100:.2f}%")

# 可视化异常点在各特征上的分布
if len(anomaly_indices) > 0:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('异常点在各特征上的值分布', fontsize=16)

    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        # 正常点
        ax.scatter(normal_indices, test_data[normal_indices, i],
                   c='blue', s=1, alpha=0.5, label='正常点')
        # 异常点
        ax.scatter(anomaly_indices, test_data[anomaly_indices, i],
                   c='red', s=5, alpha=0.8, label='异常点')

        ax.set_title(f'{feature_name}')
        ax.set_xlabel('时间点')
        ax.set_ylabel(feature_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

## 6. 特征间相关性分析
# 分析训练集特征间的相关性
train_df = pd.DataFrame(train_data, columns=feature_names)
correlation_matrix = train_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('BMS 训练集特征相关性热力图')
plt.show()

# 分析测试集特征间的相关性
test_df = pd.DataFrame(test_data, columns=feature_names)
correlation_matrix_test = test_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_test, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('BMS 测试集特征相关性热力图')
plt.show()

## 7. 结果分析（如果已有训练结果）
# 检查是否存在训练结果
output_path = 'output/BMS'
if os.path.exists(output_path):
    print("检测到 BMS 训练结果，正在加载...")
    try:
        plotter = Plotter(output_path, model_id='-1')
        plotter.result_summary()

        # 显示全局预测结果
        plotter.plot_global_predictions(type="test")

        # 显示各特征的预测结果
        for i in range(get_data_dim("BMS")):
            plotter.plot_feature(
                feature=i,
                plot_train=True,
                plot_errors=True,
                plot_feature_anom=True,
                start=0,
                end=min(2000, len(plotter.test_output))
            )
    except Exception as e:
        print(f"加载结果时出错: {e}")
else:
    print("未找到 BMS 训练结果，请先运行训练脚本生成结果。")

## 总结