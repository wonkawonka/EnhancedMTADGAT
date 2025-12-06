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

# 创建保存路径
save_path = "output/NASA/analysis"
os.makedirs(save_path, exist_ok=True)

## 1. 加载 NASA 数据集
# 加载 NASA 数据集 (以 B0049 为例)
try:
    (train_data, _), (test_data, test_labels) = get_data("NASA", normalize=False)
    
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # NASA 特征名称 (根据实际数据结构调整)
    # 数据有7列：[周期编号, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压, 容量]
    feature_names = ['cycle_number', 'voltage_measured', 'current_measured', 
                     'temperature_measured', 'current_load', 'voltage_load', 'capacity']
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
            
            # 不再进行下采样，显示全部数据点
            x_data = battery_train_data[:, -1]  # 只显示容量数据 (索引-1或6)
            time_points = np.arange(0, len(x_data))
            
            plt.plot(time_points, x_data, color=color, linewidth=1, label=battery_name)
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 训练集数据趋势对比')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/训练集数据趋势对比.png", bbox_inches="tight", dpi=300)
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
            
            # 不再进行下采样，显示全部数据点
            x_data = battery_test_data[:, -1]  # 只显示容量数据 (索引-1或6)
            time_points = np.arange(0, len(x_data))
            
            plt.plot(time_points, x_data, color=color, linewidth=1, label=battery_name)
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 测试集数据趋势对比')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/测试集数据趋势对比.png", bbox_inches="tight", dpi=300)
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
plt.savefig(f"{save_path}/训练集与测试集数据分布对比.png", bbox_inches="tight", dpi=300)
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
            
            # 不再进行下采样，显示全部数据点
            x_data = battery_test_data[:, -1]  # 只显示容量数据 (索引-1或6)
            time_points = np.arange(0, len(x_data))
            
            # 绘制正常点
            plt.scatter(time_points, x_data, c=[color], s=1, alpha=0.5, label=f'{battery_name} 正常点')
            
            # 绘制异常点
            if len(anomaly_points) > 0:
                plt.scatter(anomaly_points, x_data[anomaly_points], 
                           c=[color], s=5, alpha=0.8, marker='x', label=f'{battery_name} 异常点')
        except Exception as e:
            print(f"处理 {battery_name} 时出错: {e}")

    plt.title('NASA 测试集异常点分布')
    plt.xlabel('时间点')
    plt.ylabel('容量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/测试集异常点分布.png", bbox_inches="tight", dpi=300)
    plt.show()

## 6. 特征间相关性分析
# 分析训练集特征间的相关性
train_df = pd.DataFrame(train_data, columns=feature_names)
correlation_matrix = train_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('NASA 训练集特征相关性热力图')
plt.savefig(f"{save_path}/训练集特征相关性热力图.png", bbox_inches="tight", dpi=300)
plt.show()

# 分析测试集特征间的相关性
test_df = pd.DataFrame(test_data, columns=feature_names)
correlation_matrix_test = test_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix_test, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('NASA 测试集特征相关性热力图')
plt.savefig(f"{save_path}/测试集特征相关性热力图.png", bbox_inches="tight", dpi=300)
plt.show()

## 7. 各个电池单元数据概览
# 逐个显示各个电池单元的数据
for battery_name in battery_names:
    battery_save_path = f"{save_path}/{battery_name}"
    os.makedirs(battery_save_path, exist_ok=True)
    
    try:
        with open(os.path.join(prefix, f"NASA_{battery_name}_train.pkl"), "rb") as f:
            battery_train_data = pickle.load(f)
        
        with open(os.path.join(prefix, f"NASA_{battery_name}_test.pkl"), "rb") as f:
            battery_test_data = pickle.load(f)
            
        print(f"\n{battery_name}:")
        print(f"  训练集形状: {battery_train_data.shape}")
        print(f"  测试集形状: {battery_test_data.shape}")
        
        # 检查容量数据的有效性
        train_capacities = battery_train_data[:, -1]
        test_capacities = battery_test_data[:, -1]
        
        print(f"  训练集容量范围: [{np.min(train_capacities):.4f}, {np.max(train_capacities):.4f}]")
        print(f"  测试集容量范围: [{np.min(test_capacities):.4f}, {np.max(test_capacities):.4f}]")
        
        # 检查是否存在非正值的容量
        train_negative_capacities = np.sum(train_capacities <= 0)
        test_negative_capacities = np.sum(test_capacities <= 0)
        
        if train_negative_capacities > 0:
            print(f"  警告: 训练集中发现 {train_negative_capacities} 个非正值容量")
        if test_negative_capacities > 0:
            print(f"  警告: 测试集中发现 {test_negative_capacities} 个非正值容量")
        
        # 绘制该电池单元的训练集容量曲线
        plt.figure(figsize=(10, 4))
        plt.plot(train_capacities, linewidth=1)  # 容量数据在索引-1（最后一列）
        plt.title(f'{battery_name} 训练集容量变化趋势')
        plt.xlabel('时间点')
        plt.ylabel('容量')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{battery_save_path}/训练集容量变化趋势.png", bbox_inches="tight", dpi=300)
        plt.show()
        
        # 绘制该电池单元的测试集容量曲线
        plt.figure(figsize=(10, 4))
        plt.plot(test_capacities, linewidth=1)  # 容量数据在索引-1（最后一列）
        plt.title(f'{battery_name} 测试集容量变化趋势')
        plt.xlabel('时间点')
        plt.ylabel('容量')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{battery_save_path}/测试集容量变化趋势.png", bbox_inches="tight", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"处理 {battery_name} 时出错: {e}")

## 8. 每个电池单元的所有特征趋势可视化
# 为每个电池单元创建一个图表，显示所有特征的变化趋势，每个特征用不同颜色表示
if battery_names:
    print("\n正在生成每个电池单元的所有特征趋势可视化...")
    for battery_name in battery_names:
        battery_save_path = f"{save_path}/{battery_name}"
        os.makedirs(battery_save_path, exist_ok=True)
        
        try:
            with open(os.path.join(prefix, f"NASA_{battery_name}_train.pkl"), "rb") as f:
                battery_train_data = pickle.load(f)
            
            # 创建图表显示所有特征
            plt.figure(figsize=(15, 8))
            
            # 为每个特征生成不同颜色
            colors = plt.cm.Set1(np.linspace(0, 1, len(feature_names)))
            
            # 绘制每个特征
            for i, (feature_name, color) in enumerate(zip(feature_names, colors)):
                # 不再进行下采样，显示全部数据点
                y_data = battery_train_data[:, i]
                time_points = np.arange(0, len(y_data))
                
                plt.plot(time_points, y_data, color=color, linewidth=1, label=feature_name)
            
            plt.title(f'{battery_name} 所有特征趋势')
            plt.xlabel('时间点')
            plt.ylabel('特征值')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{battery_save_path}/所有特征趋势.png", bbox_inches="tight", dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"处理 {battery_name} 的所有特征可视化时出错: {e}")

## 9. 结果分析（如果已有训练结果）
# 检查是否存在训练结果
output_path = 'output/NASA'
if os.path.exists(output_path):
    print("检测到 NASA 训练结果，正在加载...")
    try:
        plotter = Plotter(output_path, model_id='-1')
        plotter.result_summary()

        # 显示全局预测结果
        plotter.plot_global_predictions(type="test")

        # 显示各特征的预测结果 (修复索引错误)
        data_dim = get_data_dim("NASA")
        if data_dim is not None:
            for i in range(data_dim):
                try:
                    plotter.plot_feature(
                        feature=i,
                        plot_train=True,
                        plot_errors=True,
                        plot_feature_anom=True,
                        start=0,
                        end=min(2000, len(plotter.test_output))
                    )
                except Exception as e:
                    print(f"绘制特征 {i} 时出错: {e}")
                    continue
        else:
            print("无法获取NASA数据维度信息")
    except Exception as e:
        print(f"加载结果时出错: {e}")
else:
    print("未找到 NASA 训练结果，请先运行训练脚本生成结果。")

## 总结