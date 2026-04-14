import os
from ast import literal_eval
from csv import reader
from datetime import datetime
from os import listdir, makedirs, path
from pickle import dump

import datetime
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d

from args import get_parser
from spectral_residual import apply_spectral_residual_cleaning


def interpolate_capacity_to_timesteps(cycle_capacities, cycle_lengths, cycle_types):
    """
    将周期级的容量值插值到每个时间步（仅对充电周期进行插值）
    :param cycle_capacities: 每个周期的容量值数组
    :param cycle_lengths: 每个周期的时间步数
    :param cycle_types: 每个周期的类型（charge/discharge）
    :return: 插值后的每个时间步的容量值
    """
    # print("=" * 50)
    # print("容量插值计算过程详解:")
    # print("=" * 50)
    # print("1. 插值原理:")
    # print("   - 充电周期本身没有容量值，需要使用放电周期的容量值进行插值估算")
    # print("   - 放电周期具有真实的周期级容量值")
    # print("   - 使用相邻放电周期的容量值作为参考点，在充电周期内进行插值")
    
    total_steps = sum(cycle_lengths)
    interpolated_capacities = np.zeros(total_steps, dtype=np.float32)
    
    step_idx = 0
    discharge_cycle_points = []  # 放电周期的中心点和容量值
    charge_cycle_indices = []     # 充电周期在数组中的索引
    charge_cycle_lengths = []    # 充电周期的长度
    
    # 遍历所有周期，记录放电周期信息和充电周期信息
    # print("\n2. 识别周期信息:")
    for i, (capacity, length, cycle_type) in enumerate(zip(cycle_capacities, cycle_lengths, cycle_types)):
        if cycle_type == 'discharge':
            # 记录放电周期的中心点和容量值
            center_point = step_idx + length // 2
            discharge_cycle_points.append((center_point, capacity))
            # print(f"   放电周期 {i}: 时间步范围 [{step_idx}, {step_idx + length - 1}], 中心点={center_point}, 容量={capacity:.4f}")
        else:  # charge
            # 记录充电周期信息
            charge_cycle_indices.append(i)
            charge_cycle_lengths.append(length)
            # print(f"   充电周期 {i}: 时间步范围 [{step_idx}, {step_idx + length - 1}], 容量={capacity:.4f} (默认值)")
        
        # 对于所有周期，先填入周期平均容量值
        interpolated_capacities[step_idx:step_idx+length] = capacity
        step_idx += length
    
    # 对充电周期进行插值处理
    if len(discharge_cycle_points) > 1 and len(charge_cycle_indices) > 0:
        # print(f"\n3. 执行插值:")
        # print(f"   发现 {len(discharge_cycle_points)} 个放电周期和 {len(charge_cycle_indices)} 个充电周期，可以进行插值")
        # 提取放电周期的中心点和容量值
        centers = [point[0] for point in discharge_cycle_points]
        capacities = [point[1] for point in discharge_cycle_points]
        
        # print(f"   参考点坐标: {list(zip(centers, capacities))}")
        
        # 创建插值函数
        f = interp1d(centers, capacities, kind='linear', fill_value='extrapolate')
        # print(f"   插值函数已创建: 使用 {centers[0]}-{centers[-1]} 范围内的点进行线性插值")
        
        # 对充电周期覆盖的区域进行插值
        for i, cycle_idx in enumerate(charge_cycle_indices):
            start_step = sum(cycle_lengths[:cycle_idx])
            end_step = start_step + charge_cycle_lengths[i]
            
            # 在充电周期范围内进行插值
            cycle_steps = np.arange(start_step, end_step)
            interpolated_values = f(cycle_steps)
            interpolated_capacities[start_step:end_step] = interpolated_values
            
            # print(f"   充电周期 {cycle_idx} 插值完成: 时间步 {start_step}-{end_step-1}")
            # if end_step - start_step > 5:
            #     print(f"     前5个插值结果: {interpolated_values[:5]}")
            #     print(f"     后5个插值结果: {interpolated_values[-5:]}")
            # else:
            #     print(f"     所有插值结果: {interpolated_values}")
    # else:
    #     print(f"\n3. 插值条件不足:")
    #     print(f"   放电周期数: {len(discharge_cycle_points)}, 充电周期数: {len(charge_cycle_indices)}")
    #     print(f"   无法进行有效插值，所有周期将保持原始容量值")
    
    # print("\n4. 插值完成!")
    return interpolated_capacities


def load_and_save(category, filename, dataset, dataset_folder, output_folder, apply_sr_cleaning=False):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    
    # 应用谱残差清洗（如果启用）
    if apply_sr_cleaning and category == "train":
        print(f"Applying spectral residual cleaning to {dataset} {category} data...")
        temp = apply_spectral_residual_cleaning(temp, threshold=3.0)
        print(f"Cleaning completed. Shape: {temp.shape}")
    
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset, apply_sr_cleaning=False):
    """ Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly) """

    if dataset == "SMD":
        dataset_folder = "datasets/ServerMachineDataset"
        output_folder = "datasets/ServerMachineDataset/processed"
        makedirs(output_folder, exist_ok=True)
        file_list = listdir(path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                    apply_sr_cleaning,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                    False,  # 不对标签应用清洗
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                    False,  # 不对测试数据应用清洗
                )

    elif dataset == "SMAP" or dataset == "MSL":
        dataset_folder = "datasets/data"
        output_folder = "datasets/data/processed"
        makedirs(output_folder, exist_ok=True)
        with open(path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != "P-2"]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0]: anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(dataset, "test_label", labels.shape)

        with open(path.join(output_folder, dataset + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(path.join(dataset_folder, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            
            # 应用谱残差清洗（如果启用且是训练数据）
            if apply_sr_cleaning and category == "train":
                print(f"Applying spectral residual cleaning to {dataset} {category} data...")
                data = apply_spectral_residual_cleaning(data, threshold=3.0)
                print(f"Cleaning completed. Shape: {data.shape}")
            
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)

    # TODO 自己数据集
    elif dataset == "BMS":
        dataset_folder = "datasets/BMS"
        output_folder = "datasets/BMS/processed"
        makedirs(output_folder, exist_ok=True)

        # 查找数据文件
        data_files = [f for f in listdir(dataset_folder) if f.endswith((".xls", ".xlsx"))]
        
        for filename in data_files:
            print(f"Processing {filename}...")
            # 读取Excel文件
            file_path = path.join(dataset_folder, filename)
            df = pd.read_excel(file_path)
            
            # 选择指定列（去除Date列）
            required_columns = ['SYS_Vol', 'SYS_I', 'SYS_DSOC', 'SYS_SOH', 'SYS_Vmax']
            # 检查这些列是否存在
            available_columns = [col for col in required_columns if col in df.columns]
            print(f"Available columns: {available_columns}")
            
            # 提取所需数据
            selected_data = df[available_columns].values.astype(np.float32)
            
            # 应用谱残差清洗（如果启用）
            if apply_sr_cleaning:
                print(f"Applying spectral residual cleaning to BMS {filename} data...")
                selected_data = apply_spectral_residual_cleaning(selected_data, threshold=3.0)
                print(f"Cleaning completed. Shape: {selected_data.shape}")
            
            # 生成标签（这里简单地将所有标签设为0，表示正常数据）
            labels = np.zeros(len(selected_data), dtype=np.int32)
            
            # 按时间顺序划分训练集和测试集（80%训练，20%测试）
            split_ratio = 0.8
            split_index = int(len(selected_data) * split_ratio)
            
            # 划分训练集和测试集数据
            train_data = selected_data[:split_index]
            test_data = selected_data[split_index:]
            
            # 划分对应的标签
            train_labels = labels[:split_index]
            test_labels = labels[split_index:]
            
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Train labels shape: {train_labels.shape}")
            print(f"Test labels shape: {test_labels.shape}")
            
            # 保存数据
            battery_name = filename.split('.')[0]  # 使用文件名作为电池名称
            with open(path.join(output_folder, f"BMS_{battery_name}_train.pkl"), "wb") as file:
                dump(train_data, file)
            
            with open(path.join(output_folder, f"BMS_{battery_name}_test.pkl"), "wb") as file:
                dump(test_data, file)
            
            with open(path.join(output_folder, f"BMS_{battery_name}_test_label.pkl"), "wb") as file:
                dump(test_labels, file)
            
            print(f"Saved {battery_name} data")
    # TODO 主要还是预测容量的异常，但是其他数据可以作为特征
    elif dataset == "NASA":
        dataset_folder = "datasets/NASA/"
        output_folder = "datasets/NASA/processed"
        makedirs(output_folder, exist_ok=True)

        # 获取所有MAT文件
        mat_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mat")]
        
        # 处理每个电池文件
        for filename in mat_files:
            # 提取电池编号
            battery_id = filename.split(".mat")[0]

            print(f"Processing {filename}...")

            # 加载数据
            try:
                # 首先尝试使用loadmat读取
                data = loadmat(os.path.join(dataset_folder, filename))
                battery_data = data[battery_id][0, 0]
            except Exception as e:
                print(f"使用scipy.io.loadmat读取失败: {e}")
                # 如果失败，尝试使用h5py
                try:
                    with h5py.File(os.path.join(dataset_folder, filename), 'r') as f:
                        battery_data = f[battery_id]
                except Exception as e:
                    raise Exception(f"无法读取文件 {filename}: {e}")

            # 解析NASA电池数据结构
            # MATLAB结构中的存储方式。整个循环数据都存储在cycle[0]这个数组中
            num_cycles = len(battery_data['cycle'][0])
            print('Total cycle data in dataset: ', num_cycles)
            
            # 收集所有周期的数据（包括充电和放电）
            cycle_data_list = []
            
            # 遍历所有cycles
            for i in range(num_cycles):
                row = battery_data['cycle'][0, i]
                cycle_type = row['type'][0]
                
                # 处理充电和放电数据
                if cycle_type in ['charge', 'discharge']:
                    ambient_temperature = row['ambient_temperature'][0][0]
                    date_time = datetime.datetime(int(row['time'][0][0]),
                                                  int(row['time'][0][1]),
                                                  int(row['time'][0][2]),
                                                  int(row['time'][0][3]),
                                                  int(row['time'][0][4])) + datetime.timedelta(
                        seconds=int(row['time'][0][5]))
                    data = row['data']
                    
                    # 检查数据结构，确保Capacity字段存在且可访问
                    try:
                        capacity = data[0][0]['Capacity'][0][0]
                        print(f"周期 {i} ({cycle_type}) 容量值: {capacity}")
                        
                        # 检查容量值是否合理（大于0）
                        if capacity <= 0:
                            print(f"警告: 周期 {i} ({cycle_type}) 的容量值为 {capacity}，这可能是无效数据")
                    except (IndexError, KeyError, ValueError):
                        # 如果没有容量数据，设置为默认值
                        capacity = np.nan  # 改为使用 NaN 而不是 0.0
                        print(f"周期 {i} ({cycle_type}) 无法获取容量值，设置为 NaN: {capacity}")
                    
                    # 检查电压测量值字段是否存在
                    try:
                        voltage_data = data[0][0]['Voltage_measured'][0]
                    except (IndexError, KeyError):
                        print(f"Warning: Could not access Voltage_measured data for cycle {i}. Skipping this cycle.")
                        continue
                    
                    # 收集该周期内的所有测量数据
                    cycle_measurements = []
                    for j in range(len(voltage_data)):
                        try:
                            voltage_measured = data[0][0]['Voltage_measured'][0][j]
                            current_measured = data[0][0]['Current_measured'][0][j]
                            temperature_measured = data[0][0]['Temperature_measured'][0][j]
                            # 根据cycle类型选择合适的字段
                            if cycle_type == 'charge':
                                current_load = data[0][0]['Current_charge'][0][j]
                                voltage_load = data[0][0]['Voltage_charge'][0][j]
                            else:  # discharge
                                current_load = data[0][0]['Current_load'][0][j]
                                voltage_load = data[0][0]['Voltage_load'][0][j]
                            time = data[0][0]['Time'][0][j]
                            
                            # 构造该时间点的数据（暂时使用周期级别的capacity）
                            measurement = [capacity, voltage_measured, current_measured,
                                          temperature_measured, current_load, voltage_load]
                            cycle_measurements.append(measurement)
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Error accessing data at index {j} in cycle {i}: {e}")
                            continue
                    
                    # 将该周期的数据作为一个独立片段添加到列表中
                    if cycle_measurements:
                        cycle_array = np.array(cycle_measurements, dtype=np.float32)
                        cycle_data_list.append({
                            'cycle_index': i,
                            'cycle_type': cycle_type,
                            'date_time': date_time,
                            'data': cycle_array,
                            'capacity': capacity  # 保存该周期的容量值
                        })

            print(f"Processed {len(cycle_data_list)} cycles")
            
            # 按时间顺序排列周期
            cycle_data_list.sort(key=lambda x: x['date_time'])
            
            # 提取所有周期的数据和容量值
            all_cycle_data = [cycle_info['data'] for cycle_info in cycle_data_list]
            all_capacities = [cycle_info['capacity'] for cycle_info in cycle_data_list]
            all_cycle_types = [cycle_info['cycle_type'] for cycle_info in cycle_data_list]
            all_cycle_indices = [cycle_info['cycle_index'] for cycle_info in cycle_data_list]
            
            # 转换为numpy数组
            all_capacities = np.array(all_capacities, dtype=np.float32)
            
            # 检查容量数据中是否存在无效值
            nan_capacities = np.isnan(all_capacities)
            zero_or_negative_capacities = ~nan_capacities & (all_capacities <= 0)
            
            if np.any(nan_capacities):
                print(f"警告: 发现 {np.sum(nan_capacities)} 个 NaN 容量值")
            if np.any(zero_or_negative_capacities):
                print(f"警告: 发现 {np.sum(zero_or_negative_capacities)} 个零或负容量值")
            
            # NASA 在本文中作为无监督退化偏离案例数据使用，这里不再人为构造 20% 容量衰减标签
            cycle_labels = None

            # 按周期顺序划分训练集和测试集（80%训练，20%测试）
            total_cycles = len(all_cycle_data)
            if total_cycles == 0:
                print(f"Warning: No valid cycles found for {battery_id}")
                continue
                
            split_ratio = 0.8
            split_index = int(total_cycles * split_ratio)

            # 划分训练集和测试集数据（按周期分割）
            train_cycle_data = all_cycle_data[:split_index]
            test_cycle_data = all_cycle_data[split_index:]
            
            # 划分对应的标签（按周期）
            train_cycle_labels = None
            test_cycle_labels = None
            
            # 创建展开后的完整数据和标签
            # 将周期级的容量和标签插值到每个时间点（仅针对充电周期）
            train_data_full = []
            train_labels_full = []
            test_data_full = []
            test_labels_full = []
            
            # 处理训练数据
            train_cycle_lengths = []  # 记录每个周期的长度
            train_cycle_capacities = []  # 记录每个周期的容量值
            train_cycle_numbers = []    # 记录每个周期的编号
            
            for i, cycle_data in enumerate(train_cycle_data):
                cycle_length = len(cycle_data)
                cycle_type = all_cycle_types[i]
                cycle_index = all_cycle_indices[i]
                
                # 记录周期信息
                train_cycle_lengths.append(cycle_length)
                train_cycle_capacities.append(cycle_data[0, 0])  # 周期容量
                train_cycle_numbers.append(cycle_index)         # 周期编号
                
                train_data_full.append(cycle_data)
            
            # 处理测试数据
            test_cycle_lengths = []  # 记录每个周期的长度
            test_cycle_capacities = []  # 记录每个周期的容量值
            test_cycle_numbers = []     # 记录每个周期的编号
            
            for i, cycle_data in enumerate(test_cycle_data):
                cycle_length = len(cycle_data)
                cycle_type = all_cycle_types[split_index + i]  # 注意索引偏移
                cycle_index = all_cycle_indices[split_index + i]  # 注意索引偏移
                
                # 记录周期信息
                test_cycle_lengths.append(cycle_length)
                test_cycle_capacities.append(cycle_data[0, 0])  # 周期容量
                test_cycle_numbers.append(cycle_index)         # 周期编号
                
                test_data_full.append(cycle_data)
            
            # 合并所有周期的数据
            if train_data_full:
                train_data_combined = np.vstack(train_data_full)
                train_labels_combined = None
                
                # 打印处理前的数据示例（展平前）
                print("=" * 50)
                print("训练数据展平前的示例（前2个周期）:")
                print("=" * 50)
                for i in range(min(2, len(train_cycle_data))):
                    cycle_type = all_cycle_types[i]
                    cycle_number = all_cycle_indices[i]
                    print(f"周期 {cycle_number} ({cycle_type}) - 前3个时间步数据:")
                    print("列名: [容量, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压]")
                    print(train_cycle_data[i][:3])
                    print()
                
                # 对训练数据中的容量进行插值处理（仅充电周期）
                original_capacities = train_data_combined[:, 0].copy()  # 保存原始容量值用于对比
                train_interpolated_capacities = interpolate_capacity_to_timesteps(
                    np.array(train_cycle_capacities), train_cycle_lengths, all_cycle_types[:split_index])
                train_data_combined[:, 0] = train_interpolated_capacities  # 更新容量列
                
                # 检查插值后的容量值
                negative_interp_capacities = np.sum(train_data_combined[:, 0] <= 0)
                if negative_interp_capacities > 0:
                    print(f"警告: 训练数据插值后发现 {negative_interp_capacities} 个非正值容量")
                
                # 打印插值处理前后的对比
                print("=" * 50)
                print("容量插值处理对比（前10个时间步）:")
                print("=" * 50)
                print("说明: 仅对充电周期进行插值，放电周期保持原值")
                # 计算充电周期的范围
                charge_cycle_ranges = []
                step_idx = 0
                for i, (length, cycle_type) in enumerate(zip(train_cycle_lengths, all_cycle_types[:split_index])):
                    if cycle_type == 'charge':
                        charge_cycle_ranges.append((step_idx, step_idx + length - 1, i))
                    step_idx += length
                
                print(f"充电周期时间步范围: {charge_cycle_ranges}")
                print("前10个时间步插值结果:")
                for i in range(min(10, len(original_capacities))):
                    print(f"  时间步 {i}: 原始值={original_capacities[i]:.4f}, 插值后={train_interpolated_capacities[i]:.4f}")
                print()
                
                # 添加周期编号作为新的一列特征（先添加到最后）
                cycle_number_column = np.zeros(len(train_data_combined), dtype=np.float32)
                step_idx = 0
                for cycle_num, cycle_length in zip(train_cycle_numbers, train_cycle_lengths):
                    cycle_number_column[step_idx:step_idx+cycle_length] = cycle_num
                    step_idx += cycle_length
                
                # 将周期编号列添加到数据中（现在在最后）
                train_data_combined = np.column_stack([train_data_combined, cycle_number_column])
                
                # 调整列顺序：将周期编号移到第一列，容量移到最后一列
                # 当前列顺序：[Capacity, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Cycle_Number]
                # 目标顺序：[Cycle_Number, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Capacity]
                cols_order = [6, 1, 2, 3, 4, 5, 0]  # 新的列索引顺序
                train_data_combined = train_data_combined[:, cols_order]
                
                # 打印示例数据用于调试
                print("=" * 50)
                print("训练数据最终结果示例（前5行）:")
                print("=" * 50)
                print("列名: [周期编号, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压, 容量]")
                for i in range(min(5, len(train_data_combined))):
                    cycle_type_for_row = None
                    accumulated_length = 0
                    for j, length in enumerate(train_cycle_lengths):
                        if i < accumulated_length + length:
                            cycle_type_for_row = all_cycle_types[j]
                            break
                        accumulated_length += length
                    print(f"行 {i}: {train_data_combined[i]} (周期类型: {cycle_type_for_row})")
                print()
            else:
                train_data_combined = np.array([])
                train_labels_combined = None
                
            if test_data_full:
                test_data_combined = np.vstack(test_data_full)
                test_labels_combined = None
                
                # 打印处理前的数据示例（展平前）
                print("=" * 50)
                print("测试数据展平前的示例（前2个周期）:")
                print("=" * 50)
                for i in range(min(2, len(test_cycle_data))):
                    cycle_type = all_cycle_types[split_index + i]
                    cycle_number = all_cycle_indices[split_index + i]
                    print(f"周期 {cycle_number} ({cycle_type}) - 前3个时间步数据:")
                    print("列名: [容量, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压]")
                    print(test_cycle_data[i][:3])
                    print()
                
                # 对测试数据中的容量进行插值处理（仅充电周期）
                original_capacities = test_data_combined[:, 0].copy()  # 保存原始容量值用于对比
                test_interpolated_capacities = interpolate_capacity_to_timesteps(
                    np.array(test_cycle_capacities), test_cycle_lengths, all_cycle_types[split_index:])
                test_data_combined[:, 0] = test_interpolated_capacities  # 更新容量列
                
                # 检查插值后的容量值
                negative_interp_capacities = np.sum(test_data_combined[:, 0] <= 0)
                if negative_interp_capacities > 0:
                    print(f"警告: 测试数据插值后发现 {negative_interp_capacities} 个非正值容量")
                
                # 打印插值处理前后的对比
                print("=" * 50)
                print("测试数据容量插值处理对比（前10个时间步）:")
                print("=" * 50)
                print("说明: 仅对充电周期进行插值，放电周期保持原值")
                # 计算充电周期的范围
                charge_cycle_ranges = []
                step_idx = 0
                for i, (length, cycle_type) in enumerate(zip(test_cycle_lengths, all_cycle_types[split_index:])):
                    if cycle_type == 'charge':
                        charge_cycle_ranges.append((step_idx, step_idx + length - 1, i))
                    step_idx += length
                
                print(f"充电周期时间步范围: {charge_cycle_ranges}")
                print("前10个时间步插值结果:")
                for i in range(min(10, len(original_capacities))):
                    print(f"  时间步 {i}: 原始值={original_capacities[i]:.4f}, 插值后={test_interpolated_capacities[i]:.4f}")
                print()
                
                # 添加周期编号作为新的一列特征（先添加到最后）
                cycle_number_column = np.zeros(len(test_data_combined), dtype=np.float32)
                step_idx = 0
                for cycle_num, cycle_length in zip(test_cycle_numbers, test_cycle_lengths):
                    cycle_number_column[step_idx:step_idx+cycle_length] = cycle_num
                    step_idx += cycle_length  # 修复错误：应该使用cycle_length而不是length
                
                # 将周期编号列添加到数据中（现在在最后）
                test_data_combined = np.column_stack([test_data_combined, cycle_number_column])
                
                # 调整列顺序：将周期编号移到第一列，容量移到最后一列
                # 当前列顺序：[Capacity, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Cycle_Number]
                # 目标顺序：[Cycle_Number, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Capacity]
                cols_order = [6, 1, 2, 3, 4, 5, 0]  # 新的列索引顺序
                test_data_combined = test_data_combined[:, cols_order]
                
                # 打印示例数据用于调试
                print("=" * 50)
                print("测试数据最终结果示例（前5行）:")
                print("=" * 50)
                print("列名: [周期编号, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压, 容量]")
                for i in range(min(5, len(test_data_combined))):
                    cycle_type_for_row = None
                    accumulated_length = 0
                    for j, length in enumerate(test_cycle_lengths):
                        if i < accumulated_length + length:
                            cycle_type_for_row = all_cycle_types[split_index + j]
                            break
                        accumulated_length += length
                    print(f"行 {i}: {test_data_combined[i]} (周期类型: {cycle_type_for_row})")
                print()
            else:
                test_data_combined = np.array([])
                test_labels_combined = None

            # 应用谱残差清洗（如果启用）
            if apply_sr_cleaning and len(train_data_combined) > 0:
                print(f"Applying spectral residual cleaning to NASA {battery_id} train data...")
                train_data_combined = apply_spectral_residual_cleaning(train_data_combined, threshold=3.0)
                print(f"Cleaning completed. Shape: {train_data_combined.shape}")

            print(f"{battery_id} Train data shape: {train_data_combined.shape}")
            print(f"{battery_id} Test data shape: {test_data_combined.shape}")
            print(f"{battery_id} Train labels shape: None")
            print(f"{battery_id} Test labels shape: None")

            # 保存处理后的数据（展开的时间点数据，不按周期组织）
            with open(path.join(output_folder, f"NASA_{battery_id}_train.pkl"), "wb") as file:
                dump(train_data_combined, file)

            with open(path.join(output_folder, f"NASA_{battery_id}_test.pkl"), "wb") as file:
                dump(test_data_combined, file)

            with open(path.join(output_folder, f"NASA_{battery_id}_test_label.pkl"), "wb") as file:
                dump(test_labels_combined, file)
                
            # 保存完整的周期级容量信息，供评估阶段使用
            with open(path.join(output_folder, f"NASA_{battery_id}_capacities.pkl"), "wb") as file:
                dump(all_capacities, file)
                
            # 保存周期类型信息
            with open(path.join(output_folder, f"NASA_{battery_id}_cycle_types.pkl"), "wb") as file:
                dump(all_cycle_types, file)
                
            # 保存周期索引信息
            with open(path.join(output_folder, f"NASA_{battery_id}_cycle_indices.pkl"), "wb") as file:
                dump(all_cycle_indices, file)

            print(f"Saved {battery_id} data with shape: {train_data_combined.shape} in unified folder")

    # TODO 异常标签按曲率计算
    elif dataset == "CALCE":
        # 处理CALCE数据集
        dataset_folder = "datasets/CALCE/Dataset1"
        output_folder = "datasets/CALCE/processed"
        makedirs(output_folder, exist_ok=True)
        
        # 定义文件路径
        qualified_path = path.join(dataset_folder, 'Qualified lots.xlsx')
        subsequent_path = path.join(dataset_folder, 'Subsequent lots for ongoing reliability testing.xlsx')
        
        print("检查文件是否存在:")
        print(f"Qualified lots 文件: {os.path.exists(qualified_path)}")
        print(f"Subsequent lots 文件: {os.path.exists(subsequent_path)}")
        
        # 存储所有实体名称
        entity_names = []
        
        # 处理Qualified lots (前6列作为训练集)
        if os.path.exists(qualified_path):
            print("\n读取 Qualified lots 数据:")
            try:
                df_qualified = pd.read_excel(qualified_path)
                print("形状:", df_qualified.shape)
                print("列名:", df_qualified.columns.tolist())
                print("前几行数据:")
                print(df_qualified.head())
                
                # 获取有效的列名
                valid_cols = [col for col in df_qualified.columns 
                             if 'Unnamed' not in str(col) and str(col).strip() != '' and col is not None]
                
                # 只取前6列作为训练集实体 (实体Cell1-Cell6)
                train_cols = valid_cols[:6] if len(valid_cols) >= 6 else valid_cols
                
                # 处理训练实体 (实体Cell1-Cell6)
                for i, col in enumerate(train_cols):
                    # 获取实体数据并去除NaN值
                    entity_data = df_qualified[col].dropna()
                    if len(entity_data) > 0:  # 确保有数据
                        entity_values = entity_data.values.astype(np.float32)
                        entity_name = f"Cell{str(i + 1)}"  # 实体名称为Cell1-Cell6
                        entity_names.append(entity_name)
                        
                        # 将一维数据转换为二维格式（时间步，特征数）
                        # 对于单特征时间序列，我们将其reshape为(-1, 1)
                        train_data = entity_values.reshape(-1, 1)
                        
                        # 应用谱残差异常检测和清洗 (只处理训练集)
                        print(f"对实体 '{entity_name}' 应用谱残差异常检测和清洗...")
                        train_data_cleaned = apply_spectral_residual_cleaning(train_data, threshold=3.0)
                        print(f"清洗完成，清洗前形状: {train_data.shape}, 清洗后形状: {train_data_cleaned.shape}")
                        
                        print(f"实体 '{entity_name}' 训练数据形状: {train_data_cleaned.shape}")
                        
                        # 保存训练数据，命名为Cell1-Cell6
                        with open(path.join(output_folder, f"CALCE_{entity_name}_train.pkl"), "wb") as file:
                            dump(train_data_cleaned, file)
                        print(f"已保存实体 '{entity_name}' 的训练数据")
                        
                        # 为训练数据生成全0标签（表示正常数据）
                        train_labels = np.zeros(len(train_data_cleaned), dtype=np.int32)
                        with open(path.join(output_folder, f"CALCE_{entity_name}_train_label.pkl"), "wb") as file:
                            dump(train_labels, file)
                        print(f"已保存实体 '{entity_name}' 的训练标签（全为0，表示正常数据）")
                    
            except Exception as e:
                print(f"读取Qualified lots失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 处理Subsequent lots (后6列作为测试集)
        if os.path.exists(subsequent_path):
            print("\n读取 Subsequent lots 数据:")
            try:
                df_subsequent = pd.read_excel(subsequent_path)
                print("形状:", df_subsequent.shape)
                print("列名:", df_subsequent.columns.tolist())
                print("前几行数据:")
                print(df_subsequent.head())
                
                # 获取有效的列名
                valid_cols = [col for col in df_subsequent.columns 
                             if 'Unnamed' not in str(col) and str(col).strip() != '' and col is not None]
                
                # 只取前6列作为测试集实体 (实体Cell7-Cell12)
                test_cols = valid_cols[:6] if len(valid_cols) >= 6 else valid_cols
                
                # 处理测试实体 (实体Cell7-Cell12)
                for i, col in enumerate(test_cols):
                    # 获取实体数据并去除NaN值
                    entity_data = df_subsequent[col].dropna()
                    if len(entity_data) > 0:  # 确保有数据
                        entity_values = entity_data.values.astype(np.float32)
                        entity_name = f"Cell{str(i + 7)}"  # 实体名称为Cell7-Cell12
                        entity_names.append(entity_name)
                        
                        # 将一维数据转换为二维格式
                        test_data = entity_values.reshape(-1, 1)
                        
                        # 不对测试集应用异常检测和清洗，直接保存原始数据
                        print(f"实体 '{entity_name}' 测试数据形状: {test_data.shape}")
                        
                        # 保存测试数据，命名为Cell7-Cell12
                        with open(path.join(output_folder, f"CALCE_{entity_name}_test.pkl"), "wb") as file:
                            dump(test_data, file)
                        print(f"已保存实体 '{entity_name}' 的测试数据")
                        
                        # 为测试数据生成全1标签（表示异常数据）
                        labels = np.ones(len(test_data), dtype=np.int32)
                        
                        # 保存测试标签
                        with open(path.join(output_folder, f"CALCE_{entity_name}_test_label.pkl"), "wb") as file:
                            dump(labels, file)
                        print(f"已保存实体 '{entity_name}' 的测试标签（全为1，表示异常数据）")
                    
            except Exception as e:
                print(f"读取Subsequent lots失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n处理完成！共处理了 {len(entity_names)} 个实体:")
        for name in entity_names:
            print(f"  - {name}")

    elif dataset == "CALCE2":
        # 处理CALCE Dataset2数据集
        dataset_folder = "datasets/CALCE/Dataset2"
        output_folder = "datasets/CALCE/processed"
        makedirs(output_folder, exist_ok=True)
        
        # 定义文件路径
        dataset2_path = path.join(dataset_folder, 'Dataset2.mat')
        
        print("检查文件是否存在:")
        print(f"Dataset2.mat 文件: {os.path.exists(dataset2_path)}")
        
        # 处理Dataset2.mat文件
        if os.path.exists(dataset2_path):
            print("\n读取 Dataset2 数据:")
            try:
                # 使用scipy.io.loadmat读取matlab文件
                mat_data = loadmat(dataset2_path)
                print("MAT文件中的键:", list(mat_data.keys()))
                
                # 通常MATLAB文件会有一个主键包含实际数据
                # 我们需要找到正确的键来访问数据
                data_key = None
                for key in mat_data.keys():
                    if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                        data_key = key
                        break
                
                if data_key:
                    raw_data = mat_data[data_key]
                    print(f"原始数据形状: {raw_data.shape}")
                    
                    # 根据Notes.txt描述：
                    # 前14个样本来自合格批次（单元格1-14）作为训练集
                    # 后9个样本来自不同的后续批次（单元格15-23）作为测试集
                    
                    # 假设数据是一个矩阵，每一行或每一列代表一个单元格
                    # 我们需要根据具体的数据结构来处理
                    
                    # 如果数据是二维的，且行数为23（14+9），则每行代表一个单元格
                    if raw_data.shape[0] == 23:
                        # 前14个单元格作为训练数据
                        train_cells = raw_data[:14]
                        # 后9个单元格作为测试数据
                        test_cells = raw_data[14:]
                        
                        print(f"训练数据形状: {train_cells.shape}")
                        print(f"测试数据形状: {test_cells.shape}")
                        
                        # 为每个训练单元格创建单独的训练文件 (实体Cell1-Cell14)
                        for i in range(14):
                            # 检查单元格数据类型并适当处理
                            cell_raw = train_cells[i, 0]
                            if isinstance(cell_raw, np.ndarray):
                                # 确保数据是二维的（时间步，特征数）
                                # 第二个维度是特征，第一个维度是时间步
                                if cell_raw.ndim == 1:
                                    cell_data = cell_raw.reshape(-1, 1).astype(np.float32)
                                else:
                                    # 如果已经是二维的，我们只取第二列作为特征值
                                    cell_data = cell_raw[:, 1:2].astype(np.float32) if cell_raw.shape[1] > 1 else cell_raw.astype(np.float32)
                            else:
                                # 如果是标量，创建包含单个值的数组
                                cell_data = np.array([[float(cell_raw)]], dtype=np.float32)
                            entity_name = f"Cell{str(i + 1)}"  # 实体名称为Cell1-Cell14
                            
                            # 应用谱残差异常检测和清洗（仅训练数据）
                            print(f"对单元格 {entity_name} 应用谱残差异常检测和清洗...")
                            cell_data_cleaned = apply_spectral_residual_cleaning(cell_data, threshold=3.0)
                            print(f"清洗完成，清洗前形状: {cell_data.shape}, 清洗后形状: {cell_data_cleaned.shape}")
                            
                            with open(path.join(output_folder, f"CALCE2_{entity_name}_train.pkl"), "wb") as file:
                                dump(cell_data_cleaned, file)
                            print(f"已保存单元格 {entity_name} 的训练数据，形状: {cell_data_cleaned.shape}")
                            
                            # 为训练数据生成全0标签（表示正常数据）
                            train_labels = np.zeros(len(cell_data_cleaned), dtype=np.int32)
                            with open(path.join(output_folder, f"CALCE2_{entity_name}_train_label.pkl"), "wb") as file:
                                dump(train_labels, file)
                            print(f"已保存单元格 {entity_name} 的训练标签（全为0，表示正常数据）")
                        
                        # 为每个测试单元格创建单独的测试文件 (实体Cell15-Cell23)
                        for i in range(9):
                            # 检查单元格数据类型并适当处理
                            cell_raw = test_cells[i, 0]
                            if isinstance(cell_raw, np.ndarray):
                                # 确保数据是二维的（时间步，特征数）
                                # 第二个维度是特征，第一个维度是时间步
                                if cell_raw.ndim == 1:
                                    cell_data = cell_raw.reshape(-1, 1).astype(np.float32)
                                else:
                                    # 如果已经是二维的，我们只取第二列作为特征值
                                    cell_data = cell_raw[:, 1:2].astype(np.float32) if cell_raw.shape[1] > 1 else cell_raw.astype(np.float32)
                            else:
                                # 如果是标量，创建包含单个值的数组
                                cell_data = np.array([[float(cell_raw)]], dtype=np.float32)
                            entity_name = f"Cell{str(i + 15)}"  # 实体名称为Cell15-Cell23
                            
                            # 不对测试集应用异常检测和清洗，直接保存原始数据
                            with open(path.join(output_folder, f"CALCE2_{entity_name}_test.pkl"), "wb") as file:
                                dump(cell_data, file)
                            
                            # 为测试数据生成全1标签（表示异常数据）
                            labels = np.ones(len(cell_data), dtype=np.int32)
                            with open(path.join(output_folder, f"CALCE2_{entity_name}_test_label.pkl"), "wb") as file:
                                dump(labels, file)
                            print(f"已保存单元格 {entity_name} 的测试数据和标签，形状: {cell_data.shape}")
                            
                    # 如果数据维度不同，我们需要进一步探索其结构
                    else:
                        print("数据结构不符合预期，请手动检查数据内容")
                        print("数据详情:")
                        print(raw_data)
                        
                else:
                    print("未找到有效的数据键")
                    
            except Exception as e:
                print(f"读取Dataset2.mat失败: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    # 获取是否应用谱残差清洗的参数
    apply_sr_cleaning = args.apply_sr_cleaning
    load_data(ds, apply_sr_cleaning)
