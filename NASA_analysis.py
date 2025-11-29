# NASA 数据集可视化分析

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
from utils import get_data, get_data_dim
from plotting import Plotter

## 1. 加载 NASA 数据集
# 加载 NASA 数据集 (以 B0049 为例)
try:
    (train_data, _), (test_data, test_labels) = get_data("NASA", normalize=False)
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # NASA 特征名称 (根据实际数据结构调整)
    # 数据有7列，所以我们需要7个特征名
    feature_names = ['cycle', 'capacity', 'voltage_measured', 'current_measured', 
                     'temperature_measured', 'current_charge', 'voltage_charge']
    print(f"特征名称: {feature_names}")
except FileNotFoundError as e:
    print(f"数据文件未找到: {e}")
    print("请确保已运行预处理脚本生成NASA数据集")
    sys.exit(1)

# 查看所有NASA电池单元
prefix = "datasets/NASA/processed"
if not os.path.exists(prefix):
    print(f"目录不存在: {prefix}")
    sys.exit(1)

pkl_files = glob.glob(os.path.join(prefix, "NASA_*_train.pkl"))
battery_names = [os.path.basename(f).replace("NASA_", "").replace("_train.pkl", "") for f in pkl_files]
print(f"所有NASA电池单元: {battery_names}")

## 2. 训练集数据趋势可视化
# 可视化训练集数据趋势，将所有实体绘制在同一张图上，用不同颜色区分
if battery_names:  # 只有当存在电池单元时才绘图
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(battery_names)))  # 为所有电池单元生成不同颜色

    for i, (battery_name, color) in enumerate(zip(battery_names, colors)):
        try:
            with open(os.path.join(prefix, f"NASA_{battery_name}_train.pkl"), "rb") as f:
                battery_train_data = pickle.load(f)
            
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(battery_train_data) // 1000)  # 最多显示1000个点
            x_sampled = battery_train_data[::sample_rate, 1]  # 只显示容量数据 (索引1)
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=battery_name)
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 训练集数据趋势对比')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

## 3. 测试集数据趋势可视化
# 可视化测试集数据趋势，将所有实体绘制在同一张图上，用不同颜色区分
if battery_names:  # 只有当存在电池单元时才绘图
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(battery_names)))  # 为所有电池单元生成不同颜色

    for i, (battery_name, color) in enumerate(zip(battery_names, colors)):
        try:
            with open(os.path.join(prefix, f"NASA_{battery_name}_test.pkl"), "rb") as f:
                battery_test_data = pickle.load(f)
            
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(battery_test_data) // 1000)  # 最多显示1000个点
            x_sampled = battery_test_data[::sample_rate, 1]  # 只显示容量数据 (索引1)
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=battery_name)
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 测试集数据趋势对比')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

## 4. 训练集与测试集数据分布对比
# 对比训练集和测试集的数据分布（总体）
fig, axes = plt.subplots(2, 7, figsize=(28, 8))
fig.suptitle('NASA 训练集与测试集数据分布对比', fontsize=16)

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
# 分析测试集中的异常点，将所有实体绘制在同一张图上，用不同颜色区分
if battery_names:  # 只有当存在电池单元时才绘图
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(battery_names)))  # 为所有电池单元生成不同颜色

    for i, (battery_name, color) in enumerate(zip(battery_names, colors)):
        try:
            with open(os.path.join(prefix, f"NASA_{battery_name}_test.pkl"), "rb") as f:
                battery_test_data = pickle.load(f)
                
            with open(os.path.join(prefix, f"NASA_{battery_name}_test_label.pkl"), "rb") as f:
                battery_test_labels = pickle.load(f)
            
            # 标记异常点
            anomaly_points = np.where(battery_test_labels == 1)[0]
            normal_indices = np.where(battery_test_labels == 0)[0]
            
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(battery_test_data) // 1000)  # 最多显示1000个点
            x_sampled = battery_test_data[::sample_rate, 1]  # 只显示容量数据 (索引1)
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            # 绘制正常点
            plt.scatter(time_points[normal_indices[::sample_rate]], 
                       x_sampled[normal_indices[::sample_rate]], 
                       c=[color], s=1, alpha=0.5, label=f'{battery_name} 正常点')
            
            # 绘制异常点
            if len(anomaly_points) > 0:
                plt.scatter(time_points[anomaly_points[::sample_rate]], 
                           x_sampled[anomaly_points[::sample_rate]], 
                           c=[color], s=5, alpha=0.8, marker='x', label=f'{battery_name} 异常点')
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 测试集异常点分布')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

## 6. 特征间相关性分析
# 分析训练集特征间的相关性
train_df = pd.DataFrame(train_data, columns=feature_names)
correlation_matrix = train_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('NASA 训练集特征相关性热力图')
plt.show()

# 分析测试集特征间的相关性
test_df = pd.DataFrame(test_data, columns=feature_names)
correlation_matrix_test = test_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix_test, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('NASA 测试集特征相关性热力图')
plt.show()

## 7. 各个电池单元数据概览
# 逐个显示各个电池单元的数据
for battery_name in battery_names:
    try:
        with open(os.path.join(prefix, f"NASA_{battery_name}_train.pkl"), "rb") as f:
            battery_train_data = pickle.load(f)
        
        with open(os.path.join(prefix, f"NASA_{battery_name}_test.pkl"), "rb") as f:
            battery_test_data = pickle.load(f)
            
        print(f"\n{battery_name}:")
        print(f"  训练集形状: {battery_train_data.shape}")
        print(f"  测试集形状: {battery_test_data.shape}")
        
        # 绘制该电池单元的训练集容量曲线
        plt.figure(figsize=(10, 4))
        plt.plot(battery_train_data[:, 1], linewidth=1)  # 容量数据在索引1
        plt.title(f'{battery_name} 训练集容量变化趋势')
        plt.xlabel('时间点')
        plt.ylabel('容量')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 绘制该电池单元的测试集容量曲线
        plt.figure(figsize=(10, 4))
        plt.plot(battery_test_data[:, 1], linewidth=1)  # 容量数据在索引1
        plt.title(f'{battery_name} 测试集容量变化趋势')
        plt.xlabel('时间点')
        plt.ylabel('容量')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"处理 {battery_name} 时出错: {e}")

## 8. 结果分析（如果已有训练结果）
# 检查是否存在训练结果
output_path = 'output/NASA'
if os.path.exists(output_path):
    print("检测到 NASA 训练结果，正在加载...")
    try:
        plotter = Plotter(output_path, model_id='-1')
        plotter.result_summary()

        # 显示全局预测结果
        plotter.plot_global_predictions(type="test")

        # 显示各特征的预测结果
        for i in range(get_data_dim("NASA")):
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
    print("未找到 NASA 训练结果，请先运行训练脚本生成结果。")

## 总结