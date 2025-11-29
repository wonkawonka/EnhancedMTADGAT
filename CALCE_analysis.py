# CALCE 数据集可视化分析

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
import glob

# 设置中文字体以避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, '')
from utils import get_data, get_data_dim, get_all_calce_entities, load_calce_entity_data, get_calce_train_test_splits
from plotting import Plotter

## 1. 加载 CALCE 数据集
# 加载 CALCE 数据集
try:
    (train_data, _), (test_data, test_labels) = get_data("CALCE", normalize=False)

    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")

    # CALCE 特征名称
    feature_names = ['Capacity']
    print(f"特征名称: {feature_names}")
    
    # 获取训练/测试划分
    train_entities, test_entities = get_calce_train_test_splits()
except FileNotFoundError as e:
    print(f"数据文件未找到: {e}")
    print("请确保已运行预处理脚本生成CALCE数据集")
    sys.exit(1)

## 2. 训练集数据趋势可视化
# 可视化训练集数据趋势，将所有实体绘制在同一张图上，用不同颜色区分
plt.figure(figsize=(15, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(train_entities[:6])))  # 为前6个实体生成不同颜色

for i, (entity_name, color) in enumerate(zip(train_entities[:6], colors)):
    try:
        (x_train, _), (_, _) = load_calce_entity_data(entity_name)
        if x_train is not None:
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(x_train) // 1000)  # 最多显示1000个点
            x_sampled = x_train[::sample_rate, 0]
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=entity_name)
    except Exception as e:
        print(f"处理 {entity_name} 时出错: {e}")

plt.title('CALCE 训练集数据趋势对比')
plt.xlabel('时间点')
plt.ylabel('容量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## 3. 测试集数据趋势可视化
# 可视化测试集数据趋势，将所有实体绘制在同一张图上，用不同颜色区分
plt.figure(figsize=(15, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(test_entities[:6])))  # 为前6个实体生成不同颜色

for i, (entity_name, color) in enumerate(zip(test_entities[:6], colors)):
    try:
        (_, _), (x_test, y_test) = load_calce_entity_data(entity_name)
        if x_test is not None:
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(x_test) // 1000)  # 最多显示1000个点
            x_sampled = x_test[::sample_rate, 0]
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=entity_name)
    except Exception as e:
        print(f"处理 {entity_name} 时出错: {e}")

plt.title('CALCE 测试集数据趋势对比')
plt.xlabel('时间点')
plt.ylabel('容量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## 4. 训练集与测试集数据分布对比
# 对比训练集和测试集的数据分布（总体）
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('CALCE 训练集与测试集数据分布对比', fontsize=16)

# 训练集分布
axes[0].hist(train_data[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
axes[0].set_title('容量分布 (训练集)')
axes[0].set_xlabel('容量')
axes[0].set_ylabel('频率')
axes[0].grid(True, alpha=0.3)

# 测试集分布
axes[1].hist(test_data[:, 0], bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
axes[1].set_title('容量分布 (测试集)')
axes[1].set_xlabel('容量')
axes[1].set_ylabel('频率')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 5. 异常点详细分析
# 异常点分析已在第6部分按实体分别显示，这里不再重复显示总体异常点分布

## 6. 各个电池单元数据概览
# 获取所有CALCE实体
try:
    calce_entities = get_all_calce_entities()
    print(f"所有CALCE实体: {calce_entities}")
    
    # 获取训练/测试划分（已提前获取）
    print(f"训练实体: {train_entities}")
    print(f"测试实体: {test_entities}")
    
    # 显示前几个训练实体的数据，每个实体一个子图
    plt.figure(figsize=(15, 10))
    for i, entity_name in enumerate(train_entities[:6]):  # 只显示前6个
        try:
            (x_train, _), (_, _) = load_calce_entity_data(entity_name)
            if x_train is not None:
                plt.subplot(3, 2, i+1)
                plt.plot(x_train[:, 0], linewidth=0.8)
                plt.title(f'{entity_name} 容量曲线')
                plt.xlabel('时间点')
                plt.ylabel('容量')
                plt.grid(True, alpha=0.3)
                print(f"成功加载 {entity_name} 训练数据，形状: {x_train.shape}")
        except Exception as e:
            print(f"处理 {entity_name} 时出错: {e}")
    
    plt.suptitle('CALCE 训练实体容量曲线示例')
    plt.tight_layout()
    plt.show()
    
    # 显示前几个测试实体的数据，每个实体一个子图
    plt.figure(figsize=(15, 10))
    for i, entity_name in enumerate(test_entities[:6]):  # 只显示前6个
        try:
            (_, _), (x_test, y_test) = load_calce_entity_data(entity_name)
            if x_test is not None:
                plt.subplot(3, 2, i+1)
                plt.plot(x_test[:, 0], linewidth=0.8)
                plt.title(f'{entity_name} 容量曲线')
                plt.xlabel('时间点')
                plt.ylabel('容量')
                plt.grid(True, alpha=0.3)
                print(f"成功加载 {entity_name} 测试数据，形状: {x_test.shape}")
        except Exception as e:
            print(f"处理 {entity_name} 时出错: {e}")
    
    plt.suptitle('CALCE 测试实体容量曲线示例')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"获取CALCE实体信息时出错: {e}")

## 7. 结果分析（如果已有训练结果）
# 检查是否存在训练结果
output_path = 'output/CALCE'
if os.path.exists(output_path):
    print("检测到 CALCE 训练结果，正在加载...")
    try:
        plotter = Plotter(output_path, model_id='-1')
        plotter.result_summary()

        # 显示全局预测结果
        plotter.plot_global_predictions(type="test")

        # 显示特征的预测结果
        plotter.plot_feature(
            feature=0,
            plot_train=True,
            plot_errors=True,
            plot_feature_anom=True,
            start=0,
            end=min(2000, len(plotter.test_output))
        )
    except Exception as e:
        print(f"加载结果时出错: {e}")
else:
    print("未找到 CALCE 训练结果，请先运行训练脚本生成结果。")

## 总结