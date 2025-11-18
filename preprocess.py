import os
from ast import literal_eval
from csv import reader
from os import listdir, makedirs, path
from pickle import dump

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
                label[anomaly[0] : anomaly[1] + 1] = True
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
    elif dataset == "BMS" :
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
        
        for filename in mat_files:
            # 提取电池编号
            battery_id = filename.split("_")[0]
            
            # 加载数据
            try:
                # 尝试使用h5py读取MAT文件
                with h5py.File(os.path.join(dataset_folder, filename), 'r') as f:
                    data = np.array(list(f.values())[0])
            except Exception as e:
                print(f"使用h5py读取失败: {e}")
                # 如果失败，尝试使用scipy.io.loadmat
                try:
                    data = loadmat(os.path.join(dataset_folder, filename))
                    key_name = list(data.keys())[-1]
                    data = data[key_name]   
                except Exception as e:
                    raise Exception(f"无法读取文件: {e}")
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 保存处理后的数据
            output_path = os.path.join(output_folder, f"{battery_id}.pkl")
            df.to_pickle(output_path)
            
            print(f"Saved {battery_id}.pkl")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    ds = args.dataset.upper()
    load_data(ds)
