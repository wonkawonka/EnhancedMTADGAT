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


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)