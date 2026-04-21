import os
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


BMS_FEATURE_NAMES = [
    "BMSnVol_T",
    "BMSnVol_B",
    "BMSnI",
    "BMSnRSOC",
    "BMSnSOH",
    "BMSnICMax",
    "BMSnIDMax",
    "BMSnVmax",
    "BMSnVmin",
    "BMSnVmean",
    "BMSnTmax",
    "BMSnTmin",
    "BMSnTmean",
    "BMSnETmax",
    "BMSnETmean",
    "cell_v_std",
    "cell_v_range",
    "cell_v_max_dev_from_mean",
    "cell_v_min_dev_from_mean",
    "cell_t_std",
    "cell_t_range",
    "SYS_Vol",
    "SYS_I",
    "SYS_SOH",
    "SYS_Vmax",
    "SYS_Vmin",
    "SYS_Tmax",
    "SYS_Tmin",
]


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_bms_feature_names():
    return list(BMS_FEATURE_NAMES)


def adjust_anomaly_scores(scores, dataset, is_train, window_size):
    """
    调整异常分数
    :param scores: 异常分数
    :param dataset: 数据集名称
    :param is_train: 是否为训练数据
    :param window_size: 窗口大小
    :return: 调整后的异常分数
    """
    # 对于大多数情况，我们直接返回原始分数
    # 如果需要特定的调整，可以在这里实现
    return scores


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    elif dataset == "NASA":
        # NASA电池数据集的特征维度 (不包括时间戳)
        return 7  # cycle_number, voltage_measured, current_measured, 
                  # temperature_measured, current_charge, voltage_charge, capacity
    elif dataset in ["CALCE", "CALCE2"]:
        # CALCE数据集是单特征时间序列
        return 1
    elif dataset == "BMS":
        # BMS当前使用28维特征，维度与 BMS_FEATURE_NAMES 保持同步
        return len(BMS_FEATURE_NAMES)
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    elif dataset == "NASA":
        # 对于NASA电池数据集，我们主要关注容量预测（索引6，最后一列）
        return [6]  # capacity是最重要的特征，用于预测电池退化趋势
    elif dataset in ["CALCE", "CALCE2"]:
        # 对于CALCE数据集，我们关注单个特征
        return [0]
    elif dataset == "BMS":
        # 对于BMS数据集，我们关注所有特征
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))


def inject_point_anomalies(data, anomaly_ratio=0.05):
    """
    在数据中注入点异常
    :param data: 输入数据
    :param anomaly_ratio: 异常比例
    :return: 带异常的数据和标签
    """
    data_copy = data.copy()
    labels = np.zeros(len(data), dtype=np.int32)
    
    # 计算要注入的异常点数量
    num_anomalies = max(1, int(len(data) * anomaly_ratio))
    
    # 随机选择异常点位置
    anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
    
    # 注入点异常
    for idx in anomaly_indices:
        # 获取数据的统计信息
        mean_val = np.mean(data[:, 0])  # 假设是单特征
        std_val = np.std(data[:, 0])
        # 注入3-6倍标准差的异常值
        anomaly_value = mean_val + np.random.choice([-1, 1]) * np.random.uniform(3, 6) * std_val
        data_copy[idx, 0] = anomaly_value
        labels[idx] = 1
    
    return data_copy, labels


def parse_entity_list(entity_input):
    if entity_input is None:
        return None
    if isinstance(entity_input, str):
        items = [item.strip() for item in entity_input.split(",")]
        items = [item for item in items if item]
        return items if items else None
    if isinstance(entity_input, (list, tuple)):
        items = [str(item).strip() for item in entity_input if str(item).strip()]
        return items if items else None
    raise ValueError("entity list input must be string, list, tuple, or None")


def get_available_nasa_batteries(prefix):
    battery_names = []
    for file_name in os.listdir(prefix):
        if file_name.startswith("NASA_") and file_name.endswith("_train.pkl"):
            battery_name = file_name.replace("NASA_", "").replace("_train.pkl", "")
            battery_names.append(battery_name)
    return sorted(battery_names)


def load_nasa_processed_data(prefix, battery_name):
    train_path = os.path.join(prefix, f"NASA_{battery_name}_train.pkl")
    test_path = os.path.join(prefix, f"NASA_{battery_name}_test.pkl")
    label_path = os.path.join(prefix, f"NASA_{battery_name}_test_label.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"NASA train file not found for battery {battery_name}: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"NASA test file not found for battery {battery_name}: {test_path}")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    test_label = None
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            test_label = pickle.load(f)

    return train_data, test_data, test_label


def resolve_nasa_batteries(prefix, nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None):
    available_batteries = get_available_nasa_batteries(prefix)
    if not available_batteries:
        raise FileNotFoundError(f"No processed NASA battery data found in {prefix}")

    selected_single = str(nasa_battery_id).strip() if nasa_battery_id is not None else None
    train_batteries = parse_entity_list(nasa_train_batteries)
    test_batteries = parse_entity_list(nasa_test_batteries)

    if selected_single:
        train_batteries = [selected_single]
        test_batteries = [selected_single]
    elif train_batteries is None and test_batteries is None:
        train_batteries = [available_batteries[0]]
        test_batteries = [available_batteries[0]]
    elif train_batteries is None:
        train_batteries = list(test_batteries)
    elif test_batteries is None:
        test_batteries = list(train_batteries)

    missing_batteries = [b for b in set(train_batteries + test_batteries) if b not in available_batteries]
    if missing_batteries:
        raise FileNotFoundError(
            f"NASA batteries not found in processed data: {missing_batteries}. "
            f"Available batteries: {available_batteries}"
        )

    return train_batteries, test_batteries


def get_nasa_battery_data(nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None,
                          normalize=False, prefix="datasets/NASA/processed"):
    train_batteries, test_batteries = resolve_nasa_batteries(
        prefix,
        nasa_battery_id=nasa_battery_id,
        nasa_train_batteries=nasa_train_batteries,
        nasa_test_batteries=nasa_test_batteries,
    )

    print(f"Using NASA train batteries: {train_batteries}")
    print(f"Using NASA test batteries: {test_batteries}")

    train_data_map = {}
    test_data_map = {}
    test_label_map = {}

    for battery_name in train_batteries:
        battery_train_data, _, _ = load_nasa_processed_data(prefix, battery_name)
        train_data_map[battery_name] = np.asarray(battery_train_data, dtype=np.float32)

    for battery_name in test_batteries:
        _, battery_test_data, battery_test_label = load_nasa_processed_data(prefix, battery_name)
        test_data_map[battery_name] = np.asarray(battery_test_data, dtype=np.float32)
        test_label_map[battery_name] = None if battery_test_label is None else np.asarray(battery_test_label)

    if all(label is None for label in test_label_map.values()):
        print("NASA test labels are unavailable for the selected batteries; supervised metrics should be skipped.")

    if normalize:
        concatenated_train = np.concatenate(list(train_data_map.values()), axis=0)
        _, scaler = normalize_data(concatenated_train, scaler=None)

        normalized_train_map = {}
        for battery_name, battery_train_data in train_data_map.items():
            normalized_train_map[battery_name], _ = normalize_data(battery_train_data, scaler=scaler)
        train_data_map = normalized_train_map

        normalized_test_map = {}
        for battery_name, battery_test_data in test_data_map.items():
            normalized_test_map[battery_name], _ = normalize_data(battery_test_data, scaler=scaler)
        test_data_map = normalized_test_map

    return (train_data_map, None), (test_data_map, test_label_map)


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0,
             nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    用于传统的训练流程（一个模型对应一个实体）
    """
    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP"]:
        prefix += "/data/processed"
    elif dataset == "NASA":
        prefix += "/NASA/processed"
    elif dataset in ["CALCE", "CALCE2"]:
        prefix += "/CALCE/processed"
    elif dataset == "BMS":
        prefix += "/BMS/processed"
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    
    if dataset == "NASA":
        (train_data_map, _), (test_data_map, test_label_map) = get_nasa_battery_data(
            nasa_battery_id=nasa_battery_id,
            nasa_train_batteries=nasa_train_batteries,
            nasa_test_batteries=nasa_test_batteries,
            normalize=False,
            prefix=prefix,
        )
        train_data = np.concatenate(list(train_data_map.values()), axis=0)
        test_data = np.concatenate(list(test_data_map.values()), axis=0)

        has_any_label = any(label is not None for label in test_label_map.values())
        if has_any_label:
            aligned_labels = []
            for battery_name, battery_test_data in test_data_map.items():
                battery_test_label = test_label_map[battery_name]
                if battery_test_label is None:
                    aligned_labels.append(np.zeros(len(battery_test_data), dtype=np.int32))
                else:
                    aligned_labels.append(np.asarray(battery_test_label))
            test_label = np.concatenate(aligned_labels, axis=0)
        else:
            test_label = None
    elif dataset in ["CALCE", "CALCE2"]:
        # 统一处理CALCE和CALCE2数据集
        # 使用训练/测试划分方式加载数据
        
        if dataset == "CALCE":
            # 对于CALCE数据集，前6个实体作为训练集，后6个实体作为测试集
            train_entities, test_entities = get_calce_train_test_splits()
        else:  # CALCE2
            # 对于CALCE2数据集，前14个单元作为训练集，后9个单元作为测试集
            train_entities, test_entities = get_calce2_train_test_splits()
        
        # 加载训练数据
        train_data_list = []
        for entity_name in train_entities:
            if dataset == "CALCE":
                (x_train, _), (_, _) = load_calce_entity_data(entity_name)
            else:  # CALCE2
                (x_train, _), (_, _) = load_calce2_entity_data(entity_name)
            
            if x_train is not None:
                # 确保数据是二维数组
                if np.isscalar(x_train):
                    x_train = np.array([[x_train]], dtype=np.float32)
                elif hasattr(x_train, 'ndim') and x_train.ndim == 0:
                    x_train = np.array([[x_train.item()]], dtype=np.float32)
                elif x_train.ndim == 1:
                    x_train = x_train.reshape(-1, 1)
                train_data_list.append(x_train)
        
        if train_data_list:
            train_data = np.concatenate(train_data_list, axis=0)
        else:
            raise ValueError("No training data loaded from any entity")
        
        # 加载测试数据（在预测阶段使用）
        test_data_list = []
        test_label_list = []
        for entity_name in test_entities:
            if dataset == "CALCE":
                (_, _), (x_test, y_test) = load_calce_entity_data(entity_name)
            else:  # CALCE2
                (_, _), (x_test, y_test) = load_calce2_entity_data(entity_name)
            
            if x_test is not None:
                # 确保数据是二维数组
                if np.isscalar(x_test):
                    x_test = np.array([[x_test]], dtype=np.float32)
                elif hasattr(x_test, 'ndim') and x_test.ndim == 0:
                    x_test = np.array([[x_test.item()]], dtype=np.float32)
                elif x_test.ndim == 1:
                    x_test = x_test.reshape(-1, 1)
                test_data_list.append(x_test)
                
                # 处理标签
                if y_test is not None:
                    test_label_list.append(y_test)
                else:
                    # 对于标签，我们创建全0标签（因为测试数据应该是无标签的正常数据）
                    test_label_list.append(np.zeros(len(x_test), dtype=np.int32))
        
        if test_data_list:
            test_data = np.concatenate(test_data_list, axis=0)
            test_label = np.concatenate(test_label_list, axis=0)
        else:
            test_data, test_label = None, None
    elif dataset == "BMS":
        # 对于BMS数据集，加载处理后的pkl文件
        import glob
        # 尝试加载合并后的数据
        try:
            f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
            train_data = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            # 如果没有合并后的数据，则加载单个电池的数据
            pkl_files = glob.glob(os.path.join(prefix, "BMS_*_train.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No processed BMS battery data found in {prefix}")
            
            # 为了简单起见，我们只使用第一个电池的数据
            battery_file = pkl_files[0]
            f = open(battery_file, "rb")
            train_data = pickle.load(f)
            f.close()
            print(f"Using battery data from: {os.path.basename(battery_file)}")
        
        try:
            f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
            test_data = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            pkl_files = glob.glob(os.path.join(prefix, "BMS_*_test.pkl"))
            if not pkl_files:
                test_data = None
            else:
                battery_file = pkl_files[0]
                f = open(battery_file, "rb")
                test_data = pickle.load(f)
                f.close()
        
        try:
            f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            pkl_files = glob.glob(os.path.join(prefix, "BMS_*_test_label.pkl"))
            if not pkl_files:
                test_label = None
            else:
                battery_file = pkl_files[0]
                f = open(battery_file, "rb")
                test_label = pickle.load(f)
                f.close()
    else:
        f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
        f.close()
        try:
            f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        try:
            f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
            f.close()
        except (KeyError, FileNotFoundError):
            test_label = None

    # 4. 基于训练集统计量进行归一化，并同步应用到测试集
    scaler = None
    if normalize:
        if isinstance(train_data, dict):
            concatenated_train = np.concatenate(list(train_data.values()), axis=0)
            _, scaler = normalize_data(concatenated_train, scaler=None)
            normalized_train_map = {}
            for battery_name, battery_train_data in train_data.items():
                normalized_train_map[battery_name], _ = normalize_data(battery_train_data, scaler=scaler)
            train_data = normalized_train_map
        else:
            train_data, scaler = normalize_data(train_data, scaler=None)

        if test_data is not None:
            if isinstance(test_data, dict):
                normalized_test_map = {}
                for battery_name, battery_test_data in test_data.items():
                    normalized_test_map[battery_name], _ = normalize_data(battery_test_data, scaler=scaler)
                test_data = normalized_test_map
            else:
                test_data, _ = normalize_data(test_data, scaler=scaler)

    if isinstance(train_data, dict):
        print("train set batteries: ", list(train_data.keys()))
        print("train set shapes: ", {k: v.shape for k, v in train_data.items()})
    else:
        print("train set shape: ", train_data.shape)
    if isinstance(test_data, dict):
        print("test set batteries: ", list(test_data.keys()))
        print("test set shapes: ", {k: v.shape for k, v in test_data.items()})
        if isinstance(test_label, dict):
            print("test set label shapes: ", {k: None if v is None else v.shape for k, v in test_label.items()})
        else:
            print("test set label shape: ", None)
    else:
        print("test set shape: ", test_data.shape if test_data is not None else "None")
        print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    
    # 优化参数：针对 GPU 开启 pin_memory，并使用多线程加载
    num_workers = 2 if os.name == 'nt' else 4  # Windows 下 worker 数不宜过多
    pin_memory = torch.cuda.is_available()

    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=num_workers, pin_memory=pin_memory
        )

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory
        )

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))

def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1

def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"

def get_all_calce_entities():
    """
    获取所有CALCE实体的名称
    """
    import glob
    prefix = "datasets/CALCE/processed"
    train_pkl_files = glob.glob(os.path.join(prefix, "CALCE_*_train.pkl"))
    entity_names = []
    for file in train_pkl_files:
        # 从文件名中提取实体名称 (例如: CALCE_Cell1_train.pkl -> Cell1)
        entity_name = os.path.basename(file).replace("CALCE_", "").replace("_train.pkl", "")
        entity_names.append(entity_name)
    # 按数字排序
    return sorted(entity_names, key=lambda x: int(x.replace('Cell', '')))


def load_calce_entity_data(entity_name):
    """
    加载特定CALCE实体的数据
    用于通用模型训练（一个模型对应多个实体）
    """
    prefix = "datasets/CALCE/processed"
    # 加载训练数据
    try:
        with open(os.path.join(prefix, f"CALCE_{entity_name}_train.pkl"), "rb") as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        train_data = None
    
    # 加载测试数据
    try:
        with open(os.path.join(prefix, f"CALCE_{entity_name}_test.pkl"), "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        test_data = None
    
    # 加载测试标签
    try:
        with open(os.path.join(prefix, f"CALCE_{entity_name}_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f)
    except FileNotFoundError:
        # 如果没有标签文件，创建全0标签（表示无异常）
        if test_data is not None:
            test_label = np.zeros(len(test_data), dtype=np.int32)
        else:
            test_label = None
    
    return (train_data, None), (test_data, test_label)


def get_all_calce2_entities():
    """
    获取所有CALCE2实体的名称
    """
    import glob
    prefix = "datasets/CALCE/processed"
    train_pkl_files = glob.glob(os.path.join(prefix, "CALCE2_*_train.pkl"))
    entity_names = []
    for file in train_pkl_files:
        # 从文件名中提取实体名称 (例如: CALCE2_Cell1_train.pkl -> Cell1)
        entity_name = os.path.basename(file).replace("CALCE2_", "").replace("_train.pkl", "")
        entity_names.append(entity_name)
    # 按数字排序
    return sorted(entity_names, key=lambda x: int(x.replace('Cell', '')))


def load_calce2_entity_data(entity_name):
    """
    加载特定CALCE2实体的数据
    """
    prefix = "datasets/CALCE/processed"
    # 加载训练数据
    try:
        with open(os.path.join(prefix, f"CALCE2_{entity_name}_train.pkl"), "rb") as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        train_data = None
    
    # 加载测试数据
    try:
        with open(os.path.join(prefix, f"CALCE2_{entity_name}_test.pkl"), "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        test_data = None
    
    # 加载测试标签
    try:
        with open(os.path.join(prefix, f"CALCE2_{entity_name}_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f)
    except FileNotFoundError:
        # 如果没有标签文件，创建全0标签（表示无异常）
        if test_data is not None:
            test_label = np.zeros(len(test_data), dtype=np.int32)
        else:
            test_label = None
    
    return (train_data, None), (test_data, test_label)


def get_calce_train_test_splits():
    """
    获取CALCE数据集的训练/测试实体划分
    前6个实体作为训练集，后6个实体作为测试集
    """
    # CALCE实体编号为Cell1-Cell12，其中Cell1-Cell6为训练集，Cell7-Cell12为测试集
    train_entities = [f"Cell{i}" for i in range(1, 7)]  # 实体Cell1-Cell6
    test_entities = [f"Cell{i}" for i in range(7, 13)]  # 实体Cell7-Cell12
    return train_entities, test_entities


def get_calce2_train_test_splits():
    """
    获取CALCE2数据集的训练/测试实体划分
    前14个单元作为训练集，后9个单元作为测试集
    """
    # CALCE2实体编号为Cell1-Cell23，其中Cell1-Cell14为训练集，Cell15-Cell23为测试集
    train_entities = [f"Cell{i}" for i in range(1, 15)]  # 实体Cell1-Cell14
    test_entities = [f"Cell{i}" for i in range(15, 24)]  # 实体Cell15-Cell23
    return train_entities, test_entities


def evaluate_without_labels(anomaly_scores, threshold_percentile=95):
    """
    在没有真实标签的情况下评估异常检测结果
    :param anomaly_scores: 异常分数
    :param threshold_percentile: 用于确定异常阈值的百分位数
    :return: 检测到的异常索引
    """
    # 计算阈值
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    
    # 检测异常
    anomalies = np.where(anomaly_scores >= threshold)[0]
    
    return anomalies, threshold


def evaluate_with_capacities(anomaly_scores, capacities, threshold=0.2):
    """
    使用容量信息评估异常检测结果
    :param anomaly_scores: 异常分数
    :param capacities: 容量值数组
    :param threshold: 容量下降阈值（默认0.2，即20%）
    :return: ROC曲线数据和AUC值
    """
    if len(anomaly_scores) != len(capacities):
        raise ValueError("异常分数和容量数组长度必须相同")
    
    # 基于容量衰减创建标签
    # 获取第一个非NaN容量值作为初始容量
    valid_capacity_indices = ~np.isnan(capacities)
    if not np.any(valid_capacity_indices):
        raise ValueError("容量数据中没有有效的数值")
    
    initial_capacity = capacities[valid_capacity_indices][0]
    capacity_decay_rate = (initial_capacity - capacities) / initial_capacity
    labels = (capacity_decay_rate > threshold).astype(int)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc, labels


def plot_roc_curve(fpr, tpr, roc_auc, save_path=""):
    """
    绘制ROC曲线
    :param fpr: 假正率
    :param tpr: 真正率
    :param roc_auc: AUC值
    :param save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(f"{save_path}/roc_curve.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


def plot_anomaly_score_vs_capacity(anomaly_scores, capacities, save_path="", file_name="anomaly_score_vs_capacity.png",
                                   title="Anomaly Score vs Capacity"):
    """
    绘制异常分数与容量的关系图
    :param anomaly_scores: 异常分数
    :param capacities: 容量值
    :param save_path: 保存路径
    """
    capacities = np.asarray(capacities, dtype=np.float32)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
    min_len = min(len(capacities), len(anomaly_scores))
    capacities = capacities[:min_len]
    anomaly_scores = anomaly_scores[:min_len]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    line1 = ax1.plot(capacities, label='Capacity', color='blue')
    ax1.set_ylabel('Capacity')
    ax1.set_xlabel('Time')

    ax2 = ax1.twinx()
    line2 = ax2.plot(anomaly_scores, label='Anomaly Score', color='red')
    ax2.set_ylabel('Anomaly Score')
    ax1.set_title(title)
    lines = line1 + line2
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{file_name}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_nasa_trend(series, save_path, file_name, ylabel, title, threshold=None, color="tab:red"):
    series = np.asarray(series, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series, color=color, linewidth=1.2)
    if threshold is not None:
        ax.axhline(float(threshold), color="black", linestyle="--", linewidth=1, label="Threshold")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{file_name}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_nasa_case_overview(capacities, pred_error, recon_error, anomaly_scores, save_path,
                            file_name="nasa_case_overview.png"):
    capacities = np.asarray(capacities, dtype=np.float32)
    pred_error = np.asarray(pred_error, dtype=np.float32)
    recon_error = np.asarray(recon_error, dtype=np.float32)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)
    min_len = min(len(capacities), len(pred_error), len(recon_error), len(anomaly_scores))
    capacities = capacities[:min_len]
    pred_error = pred_error[:min_len]
    recon_error = recon_error[:min_len]
    anomaly_scores = anomaly_scores[:min_len]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(capacities, color="tab:blue", linewidth=1.2)
    axes[0].set_ylabel("Capacity")
    axes[0].set_title("NASA Degradation Case Overview")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pred_error, color="tab:orange", linewidth=1.2)
    axes[1].set_ylabel("Pred Error")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(recon_error, color="tab:green", linewidth=1.2)
    axes[2].set_ylabel("Recon Error")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(anomaly_scores, color="tab:red", linewidth=1.2)
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel("Time")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{file_name}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_nasa_cycle_trend(cycle_numbers, series, save_path, file_name, ylabel, title, threshold=None, color="tab:red"):
    cycle_numbers = np.asarray(cycle_numbers, dtype=np.float32)
    series = np.asarray(series, dtype=np.float32)
    min_len = min(len(cycle_numbers), len(series))
    cycle_numbers = cycle_numbers[:min_len]
    series = series[:min_len]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cycle_numbers, series, color=color, linewidth=1.2, marker="o", markersize=2)
    if threshold is not None:
        ax.axhline(float(threshold), color="black", linestyle="--", linewidth=1, label="阈值")
        ax.legend(loc="upper right")
    ax.set_xlabel("循环编号")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{file_name}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _find_knee_cycle_candidates(cycle_numbers, capacities, top_k=3):
    cycle_numbers = np.asarray(cycle_numbers, dtype=np.float32)
    capacities = np.asarray(capacities, dtype=np.float32)
    min_len = min(len(cycle_numbers), len(capacities))
    if min_len < 5:
        return []

    cycle_numbers = cycle_numbers[:min_len]
    capacities = capacities[:min_len]

    finite_mask = np.isfinite(cycle_numbers) & np.isfinite(capacities)
    if np.count_nonzero(finite_mask) < 5:
        return []

    cycle_numbers = cycle_numbers[finite_mask]
    capacities = capacities[finite_mask]

    if len(cycle_numbers) < 5:
        return []

    smooth_series = pd.Series(capacities).interpolate(limit_direction="both").rolling(
        window=5, center=True, min_periods=1
    ).mean().values
    first_grad = np.gradient(smooth_series, cycle_numbers)
    second_grad = np.gradient(first_grad, cycle_numbers)

    candidate_scores = -second_grad
    valid_mask = np.ones_like(candidate_scores, dtype=bool)
    valid_mask[:2] = False
    valid_mask[-2:] = False
    candidate_scores = np.where(valid_mask & np.isfinite(candidate_scores), candidate_scores, -np.inf)

    top_indices = np.argsort(candidate_scores)[-top_k:][::-1]
    top_indices = [int(idx) for idx in top_indices if np.isfinite(candidate_scores[idx])]

    seen_cycles = set()
    candidates = []
    for idx in top_indices:
        cycle_num = int(cycle_numbers[idx])
        capacity_val = capacities[idx]
        if cycle_num in seen_cycles:
            continue
        if not np.isfinite(capacity_val):
            continue
        seen_cycles.add(cycle_num)
        candidates.append({
            "cycle_number": cycle_num,
            "capacity": float(capacity_val),
            "score": float(candidate_scores[idx]),
        })
    return candidates


def plot_nasa_cycle_capacity_score(cycle_level_df, save_path, file_name, title, threshold=None, top_k=3):
    if cycle_level_df.empty:
        return

    plot_df = cycle_level_df.sort_values("cycle_number").reset_index(drop=True)
    cycle_numbers = plot_df["cycle_number"].values
    capacities = plot_df["capacity_last"].values
    score_mean = plot_df["score_mean"].values

    fig, ax1 = plt.subplots(figsize=(12, 6))
    line1 = ax1.plot(cycle_numbers, capacities, color="tab:blue", linewidth=2.0, marker="o", markersize=3,
                     label="容量")
    ax1.set_xlabel("循环编号")
    ax1.set_ylabel("容量", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(cycle_numbers, score_mean, color="tab:red", linewidth=1.8, marker="s", markersize=3,
                     label="平均异常分数")
    ax2.set_ylabel("异常分数", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    threshold_handle = None
    if threshold is not None:
        threshold_handle = ax2.axhline(float(threshold), color="black", linestyle="--", linewidth=1.0, label="阈值")

    top_cycle_rows = plot_df.sort_values("score_max", ascending=False).head(top_k)
    for _, row in top_cycle_rows.iterrows():
        cycle_num = row["cycle_number"]
        score_val = row["score_mean"]
        ax2.scatter(cycle_num, score_val, color="gold", edgecolors="black", s=50, zorder=5)
        ax2.annotate(
            f"异常:{int(cycle_num)}",
            xy=(cycle_num, score_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="black",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8),
        )

    knee_candidates = _find_knee_cycle_candidates(cycle_numbers, capacities, top_k=3)
    for idx, candidate in enumerate(knee_candidates, start=1):
        cycle_num = candidate["cycle_number"]
        capacity_val = candidate["capacity"]
        ax1.scatter(cycle_num, capacity_val, color="purple", marker="^", s=60, zorder=6)
        ax1.annotate(
            f"K{idx}:{cycle_num}",
            xy=(cycle_num, capacity_val),
            xytext=(8, -14),
            textcoords="offset points",
            ha="left",
            fontsize=8,
            color="purple",
            arrowprops=dict(arrowstyle="->", color="purple", linewidth=0.8),
        )

    ax1.set_title(title)
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    if threshold_handle is not None:
        lines = lines + [threshold_handle]
        labels = labels + [threshold_handle.get_label()]
    ax1.legend(lines, labels, loc="upper right")
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{file_name}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def build_nasa_cycle_level_df(cycle_numbers, capacities, pred_error, recon_error, anomaly_scores, threshold=None):
    cycle_numbers = np.asarray(cycle_numbers, dtype=np.float32)
    capacities = np.asarray(capacities, dtype=np.float32)
    pred_error = np.asarray(pred_error, dtype=np.float32)
    recon_error = np.asarray(recon_error, dtype=np.float32)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float32)

    min_len = min(len(cycle_numbers), len(capacities), len(pred_error), len(recon_error), len(anomaly_scores))
    if min_len == 0:
        return pd.DataFrame()

    cycle_df = pd.DataFrame({
        "cycle_number": cycle_numbers[:min_len].astype(np.int32),
        "capacity": capacities[:min_len],
        "pred_error": pred_error[:min_len],
        "recon_error": recon_error[:min_len],
        "anomaly_score": anomaly_scores[:min_len],
    })

    grouped = cycle_df.groupby("cycle_number", as_index=False).agg(
        capacity_mean=("capacity", "mean"),
        capacity_last=("capacity", "last"),
        pred_error_mean=("pred_error", "mean"),
        pred_error_max=("pred_error", "max"),
        recon_error_mean=("recon_error", "mean"),
        recon_error_max=("recon_error", "max"),
        score_mean=("anomaly_score", "mean"),
        score_max=("anomaly_score", "max"),
        score_std=("anomaly_score", "std"),
        n_points=("anomaly_score", "size"),
    )
    grouped["score_std"] = grouped["score_std"].fillna(0.0)
    if threshold is not None:
        grouped["score_mean_over_threshold"] = (grouped["score_mean"] >= float(threshold)).astype(int)
        grouped["score_max_over_threshold"] = (grouped["score_max"] >= float(threshold)).astype(int)
    return grouped


def save_nasa_cycle_level_outputs(save_path, battery_name, cycle_level_df, threshold=None):
    if cycle_level_df.empty:
        return {}

    cycle_level_df.to_csv(f"{save_path}/cycle_level_scores.csv", index=False)

    plot_nasa_cycle_capacity_score(
        cycle_level_df,
        save_path=save_path,
        file_name="cycle_level_capacity_vs_score.png",
        title=f"{battery_name} 循环级容量-异常分数图",
        threshold=threshold,
        top_k=3,
    )
    plot_nasa_cycle_trend(
        cycle_level_df["cycle_number"].values,
        cycle_level_df["score_mean"].values,
        save_path=save_path,
        file_name="cycle_level_score_trend.png",
        ylabel="循环级平均异常分数",
        title=f"{battery_name} 循环级平均异常分数趋势图",
        threshold=threshold,
        color="tab:red",
    )

    top_cycle_rows = cycle_level_df.sort_values("score_max", ascending=False).head(5)
    summary = {
        "num_cycles": int(len(cycle_level_df)),
        "global_threshold": threshold,
        "max_cycle_score": float(cycle_level_df["score_max"].max()),
        "mean_cycle_score": float(cycle_level_df["score_mean"].mean()),
        "top_cycles_by_score_max": [int(v) for v in top_cycle_rows["cycle_number"].tolist()],
        "top_cycles_by_score_mean": [
            int(v) for v in cycle_level_df.sort_values("score_mean", ascending=False).head(5)["cycle_number"].tolist()
        ],
    }
    with open(f"{save_path}/cycle_level_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def get_nasa_battery_profiles(prefix, battery_names, window_size):
    profiles = []
    offset = 0
    for battery_name in battery_names:
        _, battery_test_data, _ = load_nasa_processed_data(prefix, battery_name)
        battery_test_data = np.asarray(battery_test_data, dtype=np.float32)
        effective_len = max(len(battery_test_data) - window_size, 0)
        capacities = None
        if battery_test_data.ndim == 2 and battery_test_data.shape[1] >= 1 and effective_len > 0:
            capacities = battery_test_data[window_size:, -1]
        profiles.append({
            "battery_id": battery_name,
            "raw_test_len": int(len(battery_test_data)),
            "effective_len": int(effective_len),
            "start": int(offset),
            "end": int(offset + effective_len),
            "capacities": capacities,
        })
        offset += effective_len
    return profiles


def _get_score_rising_stage(scores):
    if len(scores) == 0:
        return "unknown"
    boundaries = np.linspace(0, len(scores), 4, dtype=int)
    stage_names = ["early", "middle", "late"]
    stage_means = []
    for idx in range(3):
        start, end = boundaries[idx], boundaries[idx + 1]
        if end <= start:
            stage_means.append(float("-inf"))
        else:
            stage_means.append(float(np.mean(scores[start:end])))
    return stage_names[int(np.argmax(stage_means))]


def extract_top_score_segments(anomaly_scores, threshold=None, top_k=5):
    scores = np.asarray(anomaly_scores, dtype=np.float32)
    if len(scores) == 0:
        return []

    if threshold is None:
        threshold = float(np.percentile(scores, 95))

    segments = []
    start = None
    for idx, score in enumerate(scores):
        if score >= threshold and start is None:
            start = idx
        elif score < threshold and start is not None:
            end = idx - 1
            segment_scores = scores[start:end + 1]
            peak_offset = int(np.argmax(segment_scores))
            peak_idx = start + peak_offset
            segments.append({
                "start_idx": int(start),
                "end_idx": int(end),
                "length": int(end - start + 1),
                "peak_idx": int(peak_idx),
                "peak_score": float(scores[peak_idx]),
                "mean_score": float(np.mean(segment_scores)),
            })
            start = None

    if start is not None:
        end = len(scores) - 1
        segment_scores = scores[start:end + 1]
        peak_offset = int(np.argmax(segment_scores))
        peak_idx = start + peak_offset
        segments.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "length": int(end - start + 1),
            "peak_idx": int(peak_idx),
            "peak_score": float(scores[peak_idx]),
            "mean_score": float(np.mean(segment_scores)),
        })

    if not segments:
        top_indices = np.argsort(scores)[-top_k:][::-1]
        for idx in top_indices:
            segments.append({
                "start_idx": int(idx),
                "end_idx": int(idx),
                "length": 1,
                "peak_idx": int(idx),
                "peak_score": float(scores[idx]),
                "mean_score": float(scores[idx]),
            })

    segments.sort(key=lambda item: item["peak_score"], reverse=True)
    return segments[:top_k]


def save_nasa_case_outputs(save_path, test_pred_df, capacities, cycle_numbers=None, battery_name="", train_batteries=None, test_batteries=None):
    anomaly_scores = np.asarray(test_pred_df["A_Score_Global"].values, dtype=np.float32)
    pred_error = np.asarray(test_pred_df["Pred_Error_Global"].values, dtype=np.float32)
    recon_error = np.asarray(test_pred_df["Recon_Error_Global"].values, dtype=np.float32)
    capacities = np.asarray(capacities, dtype=np.float32)
    if cycle_numbers is None:
        cycle_numbers = np.arange(len(capacities), dtype=np.float32)
    else:
        cycle_numbers = np.asarray(cycle_numbers, dtype=np.float32)

    if train_batteries is None:
        train_batteries = []
    if test_batteries is None:
        test_batteries = []

    min_len = min(len(anomaly_scores), len(pred_error), len(recon_error), len(capacities), len(cycle_numbers))
    anomaly_scores = anomaly_scores[:min_len]
    pred_error = pred_error[:min_len]
    recon_error = recon_error[:min_len]
    capacities = capacities[:min_len]
    cycle_numbers = cycle_numbers[:min_len]

    threshold = None
    if "Thresh_Global" in test_pred_df.columns and len(test_pred_df) > 0:
        threshold = float(test_pred_df["Thresh_Global"].iloc[0])

    plot_anomaly_score_vs_capacity(
        anomaly_scores,
        capacities,
        save_path=save_path,
        file_name="capacity_vs_score.png",
        title=f"{battery_name} Capacity vs Anomaly Score",
    )
    plot_nasa_trend(
        anomaly_scores,
        save_path=save_path,
        file_name="score_trend.png",
        ylabel="Anomaly Score",
        title=f"{battery_name} Anomaly Score Trend",
        threshold=threshold,
        color="tab:red",
    )
    plot_nasa_trend(
        pred_error,
        save_path=save_path,
        file_name="prediction_error_trend.png",
        ylabel="Prediction Error",
        title=f"{battery_name} Prediction Error Trend",
        color="tab:orange",
    )
    plot_nasa_trend(
        recon_error,
        save_path=save_path,
        file_name="reconstruction_error_trend.png",
        ylabel="Reconstruction Error",
        title=f"{battery_name} Reconstruction Error Trend",
        color="tab:green",
    )
    plot_nasa_case_overview(capacities, pred_error, recon_error, anomaly_scores, save_path)

    cycle_level_df = build_nasa_cycle_level_df(
        cycle_numbers,
        capacities,
        pred_error,
        recon_error,
        anomaly_scores,
        threshold=threshold,
    )
    cycle_level_summary = save_nasa_cycle_level_outputs(save_path, battery_name, cycle_level_df, threshold=threshold)

    top_segments = extract_top_score_segments(anomaly_scores, threshold=threshold, top_k=5)
    top_segments_df = pd.DataFrame(top_segments)
    top_segments_df.to_csv(f"{save_path}/top_anomaly_segments.csv", index=False)

    valid_capacities = capacities[~np.isnan(capacities)]
    capacity_drop_rate = 0.0
    if len(valid_capacities) > 1 and valid_capacities[0] != 0:
        capacity_drop_rate = float((valid_capacities[0] - valid_capacities[-1]) / valid_capacities[0])

    summary = {
        "battery_id": battery_name,
        "train_batteries": train_batteries,
        "test_batteries": test_batteries,
        "num_points": int(min_len),
        "global_threshold": threshold,
        "max_score": float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "mean_score": float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "score_std": float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else None,
        "score_rising_stage": _get_score_rising_stage(anomaly_scores),
        "capacity_drop_rate": capacity_drop_rate,
        "pred_error_mean": float(np.mean(pred_error)) if len(pred_error) > 0 else None,
        "recon_error_mean": float(np.mean(recon_error)) if len(recon_error) > 0 else None,
        "topk_score_indices": [int(idx) for idx in np.argsort(anomaly_scores)[-5:][::-1].tolist()] if len(anomaly_scores) > 0 else [],
        "num_cycles": int(cycle_level_summary["num_cycles"]) if cycle_level_summary else 0,
        "top_cycles_by_score_max": cycle_level_summary.get("top_cycles_by_score_max", []),
        "top_cycles_by_score_mean": cycle_level_summary.get("top_cycles_by_score_mean", []),
    }
    with open(f"{save_path}/nasa_case_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_nasa_battery_comparison(save_path, case_summaries):
    if not case_summaries:
        return []

    comparison_rows = []
    for summary in case_summaries:
        comparison_rows.append({
            "battery_id": summary["battery_id"],
            "num_points": summary["num_points"],
            "mean_score": summary["mean_score"],
            "max_score": summary["max_score"],
            "score_std": summary["score_std"],
            "pred_error_mean": summary["pred_error_mean"],
            "recon_error_mean": summary["recon_error_mean"],
            "capacity_drop_rate": summary["capacity_drop_rate"],
        })

    if not comparison_rows:
        return []

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(f"{save_path}/battery_comparison.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comparison_df))
    ax1.bar(x - 0.15, comparison_df["mean_score"], width=0.3, label="Mean Score", color="tab:red")
    ax1.bar(x + 0.15, comparison_df["max_score"], width=0.3, label="Max Score", color="tab:orange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["battery_id"].tolist())
    ax1.set_ylabel("Anomaly Score")

    ax2 = ax1.twinx()
    if comparison_df["capacity_drop_rate"].notna().any():
        capacity_values = comparison_df["capacity_drop_rate"].fillna(0.0).values
        ax2.plot(x, capacity_values, color="tab:blue", marker="o", linewidth=1.5, label="Capacity Drop Rate")
        ax2.set_ylabel("Capacity Drop Rate")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title("NASA Battery Comparison")
    fig.tight_layout()
    plt.savefig(f"{save_path}/battery_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return comparison_rows


def interpolate_capacity_to_timesteps(cycle_capacities, cycle_lengths, cycle_types):
    """
    将周期级的容量值插值到每个时间步（仅对充电周期进行插值）
    :param cycle_capacities: 每个周期的容量值数组
    :param cycle_lengths: 每个周期的时间步数
    :param cycle_types: 每个周期的类型（charge/discharge）
    :return: 插值后的每个时间步的容量值
    """
    total_steps = sum(cycle_lengths)
    interpolated_capacities = np.zeros(total_steps, dtype=np.float32)
    
    step_idx = 0
    charge_cycle_points = []  # 充电周期的中心点和容量值
    charge_cycle_indices = []  # 充电周期在数组中的索引
    
    # 遍历所有周期，记录充电周期信息
    for i, (capacity, length, cycle_type) in enumerate(zip(cycle_capacities, cycle_lengths, cycle_types)):
        if cycle_type == 'charge':
            # 记录充电周期的中心点和容量值
            center_point = step_idx + length // 2
            # 只有当容量值有效时才添加到插值点中
            if not np.isnan(capacity) and capacity > 0:
                charge_cycle_points.append((center_point, capacity))
                charge_cycle_indices.append(i)
        
        # 对于所有周期，先填入周期平均容量值
        interpolated_capacities[step_idx:step_idx+length] = capacity if not np.isnan(capacity) else 0
        step_idx += length
    
    # 对充电周期进行插值处理
    if len(charge_cycle_points) > 1:
        # 提取充电周期的中心点和容量值
        centers = [point[0] for point in charge_cycle_points]
        capacities = [point[1] for point in charge_cycle_points]
        
        # 创建插值函数
        f = interp1d(centers, capacities, kind='linear', fill_value='extrapolate')
        
        # 对充电周期覆盖的区域进行插值
        for i in range(len(charge_cycle_indices)):
            cycle_idx = charge_cycle_indices[i]
            start_step = sum(cycle_lengths[:cycle_idx])
            end_step = start_step + cycle_lengths[cycle_idx]
            
            # 在充电周期范围内进行插值
            cycle_steps = np.arange(start_step, end_step)
            interpolated_values = f(cycle_steps)
            interpolated_capacities[start_step:end_step] = interpolated_values
    
    return interpolated_capacities
