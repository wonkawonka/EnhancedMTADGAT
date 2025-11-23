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

from args import get_parser


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(
        path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
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
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test",
                    filename,
                    filename.strip(".txt"),
                    dataset_folder,
                    output_folder,
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
            with open(path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)

    # TODO 自己数据集
    elif dataset == "BMS":
        dataset_folder = "datasets/BMS"
        output_folder = "datasets/BMS/processed"
        makedirs(output_folder, exist_ok=True)

        for filename in listdir(path.join(dataset_folder, "train")):
            if filename.endswith(".csv"):
                load_and_save(
                    "train",
                    filename,
                    filename.strip(".csv"),
                    dataset_folder,
                    output_folder,
                )
                load_and_save(
                    "test_label",
                    filename,
                    filename.strip(".csv"),
                    dataset_folder,
                    output_folder,
                )
    elif dataset == "NASA":
        # TODO 找连续的数据集
        dataset_folder = "datasets/NASA/"
        output_folder = "datasets/NASA/processed"
        makedirs(output_folder, exist_ok=True)

        # 获取所有MAT文件
        mat_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mat")]

        # 收集所有电池的数据
        all_train_data = []
        all_test_data = []
        all_test_labels = []

        for filename in mat_files:
            # 提取电池编号
            battery_id = filename.split(".mat")[0]  # 更准确地提取电池编号

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
            counter = 0
            print('Total cycle data in dataset: ', num_cycles)
            processed_data = []

            # 遍历所有cycles
            for i in range(num_cycles):
                row = battery_data['cycle'][0, i]
                if row['type'][0] == 'discharge':
                    ambient_temperature = row['ambient_temperature'][0][0]
                    date_time = datetime.datetime(int(row['time'][0][0]),
                                                  int(row['time'][0][1]),
                                                  int(row['time'][0][2]),
                                                  int(row['time'][0][3]),
                                                  int(row['time'][0][4])) + datetime.timedelta(
                        seconds=int(row['time'][0][5]))
                    data = row['data']
                    capacity = data[0][0]['Capacity'][0][0]
                    for j in range(len(data[0][0]['Voltage_measured'][0])):
                        voltage_measured = data[0][0]['Voltage_measured'][0][j]
                        current_measured = data[0][0]['Current_measured'][0][j]
                        temperature_measured = data[0][0]['Temperature_measured'][0][j]
                        current_load = data[0][0]['Current_load'][0][j]
                        voltage_load = data[0][0]['Voltage_load'][0][j]
                        time = data[0][0]['Time'][0][j]
                        processed_data.append([counter + 1, ambient_temperature, date_time, capacity,
                                               voltage_measured, current_measured,
                                               temperature_measured, current_load,
                                               voltage_load, time])

                    counter = counter + 1

            print(processed_data[0])
            df = pd.DataFrame(data=processed_data,
                              columns=['cycle', 'ambient_temperature', 'datetime',
                                       'capacity', 'voltage_measured',
                                       'current_measured', 'temperature_measured',
                                       'current_charge', 'voltage_charge', 'time'])
            pd.set_option('display.max_columns', 10)
            print(df.head())
            print(df.describe())

            # 选择关键特征用于时间序列分析
            selected_features = ['capacity', 'voltage_measured', 'current_measured',
                                 'temperature_measured', 'current_charge', 'voltage_charge']
            
            # 提取特征数据（保持原始时间序列格式，不进行滑动窗口处理）
            raw_data = df[selected_features].values.astype(np.float32)

            # 生成测试标签（基于容量衰减作为异常）
            capacities = df['capacity'].values
            # 计算容量衰减率
            initial_capacity = capacities[0]
            capacity_decay_rate = (initial_capacity - capacities) / initial_capacity

            # 定义阈值，当容量衰减超过一定比例时标记为异常
            threshold = 0.2  # 20%容量衰减作为异常开始点
            labels = (capacity_decay_rate > threshold).astype(np.int32)

            # 按时间顺序划分训练集和测试集（80%训练，20%测试）
            split_ratio = 0.8
            split_index = int(len(raw_data) * split_ratio)

            # 划分训练集和测试集数据
            train_data = raw_data[:split_index]
            test_data = raw_data[split_index:]
            
            # 划分对应的标签
            train_labels = labels[:split_index]
            test_labels = labels[split_index:]

            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            print(f"Train labels shape: {train_labels.shape}")
            print(f"Test labels shape: {test_labels.shape}")

            # 保存单个电池的数据
            with open(path.join(output_folder, f"NASA_{battery_id}_train.pkl"), "wb") as file:
                dump(train_data, file)

            with open(path.join(output_folder, f"NASA_{battery_id}_test.pkl"), "wb") as file:
                dump(test_data, file)

            with open(path.join(output_folder, f"NASA_{battery_id}_test_label.pkl"), "wb") as file:
                dump(test_labels, file)

            # 添加到整体数据集中
            all_train_data.append(train_data)
            all_test_data.append(test_data)
            all_test_labels.extend(test_labels)

            print(f"Saved {battery_id} data with shape: {train_data.shape}")

        # 合并所有电池数据作为整体数据集
        if all_train_data:
            combined_train = np.vstack(all_train_data)
            combined_test = np.vstack(all_test_data)
            combined_labels = np.array(all_test_labels)

            # 保存合并后的数据
            with open(path.join(output_folder, "NASA_train.pkl"), "wb") as file:
                dump(combined_train, file)

            with open(path.join(output_folder, "NASA_test.pkl"), "wb") as file:
                dump(combined_test, file)

            with open(path.join(output_folder, "NASA_test_label.pkl"), "wb") as file:
                dump(combined_labels, file)

            print(f"Combined NASA dataset - Train shape: {combined_train.shape}, Test shape: {combined_test.shape}")

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
        
        # 处理Qualified lots (用于训练)
        if os.path.exists(qualified_path):
            print("\n读取 Qualified lots 数据:")
            try:
                df_qualified = pd.read_excel(qualified_path)
                print("形状:", df_qualified.shape)
                print("列名:", df_qualified.columns.tolist())
                print("前几行数据:")
                print(df_qualified.head())
                
                # 处理每个实体（每列代表一个实体）
                for col in df_qualified.columns:
                    if 'Unnamed' not in str(col) and str(col).strip() != '' and col is not None:  
                        # 获取实体数据并去除NaN值
                        entity_data = df_qualified[col].dropna()
                        if len(entity_data) > 0:  # 确保有数据
                            entity_values = entity_data.values.astype(np.float32)
                            entity_names.append(col)
                            
                            # 将一维数据转换为二维格式（时间步，特征数）
                            # 对于单特征时间序列，我们将其reshape为(-1, 1)
                            train_data = entity_values.reshape(-1, 1)
                            
                            print(f"实体 '{col}' 训练数据形状: {train_data.shape}")
                            
                            # 保存训练数据
                            with open(path.join(output_folder, f"CALCE_{col}_train.pkl"), "wb") as file:
                                dump(train_data, file)
                            print(f"已保存实体 '{col}' 的训练数据")
                        
            except Exception as e:
                print(f"读取Qualified lots失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 处理Subsequent lots (用于测试)
        if os.path.exists(subsequent_path):
            print("\n读取 Subsequent lots 数据:")
            try:
                df_subsequent = pd.read_excel(subsequent_path)
                print("形状:", df_subsequent.shape)
                print("列名:", df_subsequent.columns.tolist())
                print("前几行数据:")
                print(df_subsequent.head())
                
                # 处理每个实体（每列代表一个实体）
                for col in df_subsequent.columns:
                    if 'Unnamed' not in str(col) and str(col).strip() != '' and col is not None:
                        # 获取实体数据并去除NaN值
                        entity_data = df_subsequent[col].dropna()
                        if len(entity_data) > 0:  # 确保有数据
                            entity_values = entity_data.values.astype(np.float32)
                            
                            # 将一维数据转换为二维格式
                            test_data = entity_values.reshape(-1, 1)
                            
                            print(f"实体 '{col}' 测试数据形状: {test_data.shape}")
                            
                            # 保存测试数据
                            with open(path.join(output_folder, f"CALCE_{col}_test.pkl"), "wb") as file:
                                dump(test_data, file)
                            print(f"已保存实体 '{col}' 的测试数据")
                            
                            # 对于健康数据，不生成标签或者生成全0标签（表示无异常）
                            labels = np.zeros(len(test_data), dtype=np.int32)
                            
                            # 保存测试标签
                            with open(path.join(output_folder, f"CALCE_{col}_test_label.pkl"), "wb") as file:
                                dump(labels, file)
                            print(f"已保存实体 '{col}' 的测试标签（全为0，表示健康数据）")
                        
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
                    # 前14个样本来自合格批次（单元格1-14）
                    # 后9个样本来自不同的后续批次（单元格15-23）
                    
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
                        
                        # 为每个单元格创建单独的训练和测试文件
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
                            with open(path.join(output_folder, f"CALCE2_Cell{i+1}_train.pkl"), "wb") as file:
                                dump(cell_data, file)
                            print(f"已保存单元格 {i+1} 的训练数据，形状: {cell_data.shape}")
                        
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
                            with open(path.join(output_folder, f"CALCE2_Cell{i+15}_test.pkl"), "wb") as file:
                                dump(cell_data, file)
                            
                            # 为测试数据生成标签（假设是健康数据，标签为0）
                            labels = np.zeros(len(cell_data), dtype=np.int32)
                            with open(path.join(output_folder, f"CALCE2_Cell{i+15}_test_label.pkl"), "wb") as file:
                                dump(labels, file)
                            print(f"已保存单元格 {i+15} 的测试数据和标签，形状: {cell_data.shape}")
                            
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
    load_data(ds)