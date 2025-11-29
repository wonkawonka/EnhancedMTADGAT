# CALCE2 数据集可视化分析

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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# 只导入处理CALCE2数据所需的函数
from utils import get_all_calce2_entities, load_calce2_entity_data, get_calce2_train_test_splits
from plotting import Plotter

## 1. CALCE2 数据集信息
print("CALCE2 数据集信息:")
print("根据项目规范，CALCE2 Dataset2.mat文件包含23个单元格数据:")
print("- 前14个单元格（Cell1-14）作为训练数据，每个单元格独立训练")
print("- 后9个单元格（Cell15-23）作为测试数据，每个单元格独立测试")
print("- 数据路径: datasets/CALCE/Dataset2/")
print()

## 2. 获取所有CALCE2实体
try:
    calce2_entities = get_all_calce2_entities()
    print(f"已处理的CALCE2实体: {calce2_entities}")

    # 获取训练/测试划分
    train_entities, test_entities = get_calce2_train_test_splits()
    print(f"CALCE2 训练实体: {train_entities}")
    print(f"CALCE2 测试实体: {test_entities}")

except Exception as e:
    print(f"获取CALCE2实体信息时出错: {e}")
    # 手动定义实体
    train_entities = [f"Cell{i}" for i in range(1, 15)]  # Cell1-Cell14
    test_entities = [f"Cell{i}" for i in range(15, 24)]  # Cell15-Cell23
    print("使用默认实体划分:")
    print(f"CALCE2 训练实体: {train_entities}")
    print(f"CALCE2 测试实体: {test_entities}")

## 3. 训练实体数据可视化
# 显示训练实体的数据在同一张图上，用不同颜色区分
plt.figure(figsize=(15, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(train_entities[:6])))  # 为前6个实体生成不同颜色

print("\n正在加载训练实体数据...")

for i, (entity_name, color) in enumerate(zip(train_entities[:6], colors)):  # 只显示前6个
    try:
        (x_train, _), (_, _) = load_calce2_entity_data(entity_name)
        if x_train is not None:
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(x_train) // 1000)  # 最多显示1000个点
            x_sampled = x_train[::sample_rate, 0]
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=entity_name)
            print(f"成功加载 {entity_name} 训练数据，形状: {x_train.shape}")
        else:
            print(f"无法加载 {entity_name} 训练数据")
    except Exception as e:
        print(f"处理 {entity_name} 时出错: {e}")

plt.title('CALCE2 训练实体容量曲线对比')
plt.xlabel('时间点')
plt.ylabel('容量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## 4. 测试实体数据可视化
# 显示测试实体的数据在同一张图上，用不同颜色区分
plt.figure(figsize=(15, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(test_entities[:6])))  # 为前6个实体生成不同颜色

print("\n正在加载测试实体数据...")

for i, (entity_name, color) in enumerate(zip(test_entities[:6], colors)):  # 只显示前6个
    try:
        (_, _), (x_test, y_test) = load_calce2_entity_data(entity_name)
        if x_test is not None:
            # 为了避免图过于密集，我们可能需要对数据进行下采样
            sample_rate = max(1, len(x_test) // 1000)  # 最多显示1000个点
            x_sampled = x_test[::sample_rate, 0]
            time_points = np.arange(0, len(x_sampled)) * sample_rate
            
            plt.plot(time_points, x_sampled, color=color, linewidth=1, label=entity_name)
            print(f"成功加载 {entity_name} 测试数据，形状: {x_test.shape}")
        else:
            print(f"无法加载 {entity_name} 测试数据")
    except Exception as e:
        print(f"处理 {entity_name} 时出错: {e}")

plt.title('CALCE2 测试实体容量曲线对比')
plt.xlabel('时间点')
plt.ylabel('容量')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

## 5. 所有实体数据统计
print("\n所有实体数据统计:")
print("-" * 50)

# 训练实体统计
print("训练实体统计:")
for entity_name in train_entities:
    try:
        (x_train, _), (_, _) = load_calce2_entity_data(entity_name)
        if x_train is not None:
            print(
                f"  {entity_name}: {x_train.shape[0]} 个数据点, 容量范围 [{np.min(x_train):.4f}, {np.max(x_train):.4f}]")
    except Exception as e:
        print(f"  {entity_name}: 加载失败 - {e}")

print("\n测试实体统计:")
for entity_name in test_entities:
    try:
        (_, _), (x_test, y_test) = load_calce2_entity_data(entity_name)
        if x_test is not None:
            print(f"  {entity_name}: {x_test.shape[0]} 个数据点, 容量范围 [{np.min(x_test):.4f}, {np.max(x_test):.4f}]")
    except Exception as e:
        print(f"  {entity_name}: 加载失败 - {e}")

## 6. 数据分布对比
# 数据分布对比已在前面的实体图中展示，此处不再重复显示合并分布图

## 7. 结果分析（如果已有训练结果）
# 检查是否存在训练结果
print("\n检查训练结果...")
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
print("\n分析完成!")
print("根据项目规范，需要注意以下几点:")
print("1. CALCE2数据集中，前14个单元格（Cell1-14）必须保持独立，每个单元格单独用于训练")
print("2. 后9个单元格（Cell15-23）同样保持独立，分别用于测试和预测")
print("3. 不得合并或拼接数据，以确保数据独立性和模型泛化能力评估的准确性")