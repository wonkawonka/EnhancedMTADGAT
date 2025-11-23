import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


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
        # NASA电池数据集的特征维度
        return 6  # capacity, voltage_measured, current_measured, 
                 # temperature_measured, current_charge, voltage_charge
    elif dataset == "CALCE":
        # CALCE数据集是单特征时间序列
        return 1
    elif dataset == "CALCE2":
        # CALCE2数据集是单特征时间序列
        return 1
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
        # 对于NASA电池数据集，我们关注所有特征
        return None
    elif dataset == "CALCE":
        # 对于CALCE数据集，我们关注单个特征
        return [0]
    elif dataset == "CALCE2":
        # 对于CALCE2数据集，我们关注单个特征
        return [0]
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


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP"]:
        prefix += "/data/processed"
    elif dataset == "NASA":
        prefix += "/NASA/processed"
    elif dataset == "CALCE":
        prefix += "/CALCE/processed"
    elif dataset == "CALCE2":
        prefix += "/CALCE/processed"
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
        # 对于NASA数据集，加载处理后的pkl文件
        import glob
        # 尝试加载合并后的数据
        try:
            f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
            train_data = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            # 如果没有合并后的数据，则加载单个电池的数据
            pkl_files = glob.glob(os.path.join(prefix, "NASA_*_train.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No processed NASA battery data found in {prefix}")
            
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
            pkl_files = glob.glob(os.path.join(prefix, "NASA_*_test.pkl"))
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
            pkl_files = glob.glob(os.path.join(prefix, "NASA_*_test_label.pkl"))
            if not pkl_files:
                test_label = None
            else:
                battery_file = pkl_files[0]
                f = open(battery_file, "rb")
                test_label = pickle.load(f)
                f.close()
    elif dataset == "CALCE":
        # 对于CALCE数据集，加载处理后的pkl文件
        import glob
        # 加载所有实体的训练数据
        train_pkl_files = glob.glob(os.path.join(prefix, "CALCE_*_train.pkl"))
        if not train_pkl_files:
            raise FileNotFoundError(f"No processed CALCE training data found in {prefix}")
        
        # 为了简单起见，我们只使用第一个实体的数据
        entity_file = train_pkl_files[0]
        f = open(entity_file, "rb")
        train_data = pickle.load(f)
        f.close()
        print(f"Using entity data from: {os.path.basename(entity_file)}")
        
        # 加载对应的测试数据
        entity_name = os.path.basename(entity_file).replace("_train.pkl", "")
        test_file = os.path.join(prefix, f"{entity_name}_test.pkl")
        test_label_file = os.path.join(prefix, f"{entity_name}_test_label.pkl")
        
        try:
            f = open(test_file, "rb")
            test_data = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        
        try:
            f = open(test_label_file, "rb")
            test_label = pickle.load(f)
            f.close()
        except (KeyError, FileNotFoundError):
            # 如果没有标签文件，创建全0标签（表示无异常）
            if test_data is not None:
                test_label = np.zeros(len(test_data), dtype=np.int32)
            else:
                test_label = None
    elif dataset == "CALCE2":
        # 对于CALCE2数据集，加载处理后的pkl文件
        import glob
        # 根据CALCE2数据集的结构，前14个单元（Cell1-14）是训练数据
        # 我们选择第一个单元作为训练数据
        train_pkl_files = glob.glob(os.path.join(prefix, "CALCE2_Cell[1-9]_train.pkl")) + \
                          glob.glob(os.path.join(prefix, "CALCE2_Cell1[0-4]_train.pkl"))
        
        if not train_pkl_files:
            raise FileNotFoundError(f"No processed CALCE2 training data found in {prefix}")
        
        # 按照单元编号排序，确保顺序一致，并选择第一个
        train_pkl_files.sort(key=lambda x: int(x.split('_')[1].replace('Cell', '')))
        
        # 只加载第一个训练文件
        train_file = train_pkl_files[0]
        f = open(train_file, "rb")
        train_data = pickle.load(f)
        f.close()
        print(f"Loaded training data from {os.path.basename(train_file)}, type: {type(train_data)}")
        
        # 确保训练数据是二维数组
        if np.isscalar(train_data):
            print("Converting scalar to 2D array")
            train_data = np.array([[train_data]], dtype=np.float32)
        elif hasattr(train_data, 'ndim') and train_data.ndim == 0:
            print("Converting 0D array to 2D")
            train_data = np.array([[train_data.item()]], dtype=np.float32)
        elif train_data.ndim == 1:
            print("Reshaping 1D array to 2D")
            train_data = train_data.reshape(-1, 1)
            
        print(f"Final training data shape: {train_data.shape}")
        
        # 对于测试数据，选择第一个测试单元（Cell15）
        test_pkl_files = glob.glob(os.path.join(prefix, "CALCE2_Cell1[5-9]_test.pkl")) + \
                         glob.glob(os.path.join(prefix, "CALCE2_Cell2[0-3]_test.pkl"))
        
        if not test_pkl_files:
            raise FileNotFoundError(f"No processed CALCE2 test data found in {prefix}")
            
        # 按照单元编号排序，选择第一个作为测试数据
        test_pkl_files.sort(key=lambda x: int(x.split('_')[1].replace('Cell', '')))
        
        # 只加载第一个测试文件
        test_file = test_pkl_files[0]
        f = open(test_file, "rb")
        test_data = pickle.load(f)
        f.close()
        print(f"Loaded test data from {os.path.basename(test_file)}, type: {type(test_data)}")
        
        # 确保测试数据是二维数组
        if np.isscalar(test_data):
            print("Converting scalar to 2D array")
            test_data = np.array([[test_data]], dtype=np.float32)
        elif hasattr(test_data, 'ndim') and test_data.ndim == 0:
            print("Converting 0D array to 2D")
            test_data = np.array([[test_data.item()]], dtype=np.float32)
        elif test_data.ndim == 1:
            print("Reshaping 1D array to 2D")
            test_data = test_data.reshape(-1, 1)
            
        print(f"Final test data shape: {test_data.shape}")
        
        # 加载对应的测试标签
        label_file = test_file.replace('_test.pkl', '_test_label.pkl')
        if os.path.exists(label_file):
            f = open(label_file, "rb")
            test_label = pickle.load(f)
            f.close()
        else:
            # 如果没有标签文件，创建全0标签（表示无异常）
            test_label = np.zeros(len(test_data), dtype=np.int32)
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

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
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
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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


def get_all_calce_entities():
    """
    获取所有CALCE实体的名称
    """
    import glob
    prefix = "datasets/CALCE/processed"
    train_pkl_files = glob.glob(os.path.join(prefix, "CALCE_*_train.pkl"))
    entity_names = []
    for file in train_pkl_files:
        entity_name = os.path.basename(file).replace("_train.pkl", "")
        entity_names.append(entity_name)
    return entity_names


def load_calce_entity_data(entity_name):
    """
    加载特定CALCE实体的数据
    """
    prefix = "datasets/CALCE/processed"
    # 加载训练数据
    try:
        with open(os.path.join(prefix, f"{entity_name}_train.pkl"), "rb") as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        train_data = None
    
    # 加载测试数据
    try:
        with open(os.path.join(prefix, f"{entity_name}_test.pkl"), "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        test_data = None
    
    # 加载测试标签
    try:
        with open(os.path.join(prefix, f"{entity_name}_test_label.pkl"), "rb") as f:
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
        entity_name = os.path.basename(file).replace("_train.pkl", "")
        entity_names.append(entity_name)
    return entity_names


def load_calce2_entity_data(entity_name):
    """
    加载特定CALCE2实体的数据
    """
    prefix = "datasets/CALCE/processed"
    # 加载训练数据
    try:
        with open(os.path.join(prefix, f"{entity_name}_train.pkl"), "rb") as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        train_data = None
    
    # 加载测试数据
    try:
        with open(os.path.join(prefix, f"{entity_name}_test.pkl"), "rb") as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        test_data = None
    
    # 加载测试标签
    try:
        with open(os.path.join(prefix, f"{entity_name}_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f)
    except FileNotFoundError:
        # 如果没有标签文件，创建全0标签（表示无异常）
        if test_data is not None:
            test_label = np.zeros(len(test_data), dtype=np.int32)
        else:
            test_label = None
    
    return (train_data, None), (test_data, test_label)


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