"""汇总共享的数据加载、归一化和数据工具函数。"""

import os
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.project_paths import processed_dataset_path


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
    "hier_vmax_sys_gap",
    "hier_vmin_sys_gap",
    "hier_tmax_sys_gap",
    "hier_tmin_sys_gap",
    "hier_soh_sys_gap",
    "hier_cell_v_range_ratio",
    "hier_cell_t_range_ratio",
]

BMS_HIERARCHICAL_FEATURE_NAMES = [
    "hier_vmax_sys_gap",
    "hier_vmin_sys_gap",
    "hier_tmax_sys_gap",
    "hier_tmin_sys_gap",
    "hier_soh_sys_gap",
    "hier_cell_v_range_ratio",
    "hier_cell_t_range_ratio",
]

NASA_RANDOM_DATASET_PREFIX = {
    "NASA_RANDOM_CHARGE": "NASA_RANDOM_CHARGE",
    "NASA_RANDOM_DISCHARGE": "NASA_RANDOM_DISCHARGE",
}

NASA_RANDOM_DATASETS = {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}
NASA_TELEMETRY_DATASETS = {"MSL", "SMAP"}
INDUSTRIAL_CONTROL_DATASETS = {"SWAT", "WADI"}


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


class NasaRandomPhysicalScaler:
    """Robust voltage/temperature scaling with a zero-preserving current scale."""

    def __init__(self):
        self.center_ = np.zeros(3, dtype=np.float32)
        self.scale_ = np.ones(3, dtype=np.float32)

    @staticmethod
    def _iqr(values):
        q25, q75 = np.quantile(values, [0.25, 0.75])
        return max(float(q75 - q25), 1e-6)

    def fit(self, data):
        values = np.asarray(data, dtype=np.float32)
        # Model-facing order after dropping step_type_code: voltage, current, temperature.
        for index in (0, 2):
            self.center_[index] = float(np.median(values[:, index]))
            self.scale_[index] = self._iqr(values[:, index])
        self.center_[1] = 0.0
        self.scale_[1] = max(float(np.quantile(np.abs(values[:, 1]), 0.95)), 1e-6)
        return self

    def transform(self, data):
        values = np.asarray(data, dtype=np.float32)
        return ((values - self.center_) / self.scale_).astype(np.float32, copy=False)


def get_bms_feature_names():
    return list(BMS_FEATURE_NAMES)


def adjust_anomaly_scores(scores, dataset, is_train, window_size):
    """
    调整异常分数。
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
    :param dataset: 数据集名称
    :return: Number of dimensions in data
    """
    if dataset == "MSL":
        return 55
    elif dataset == "SMAP":
        return 25
    elif dataset == "SWAT":
        feature_path = processed_dataset_path("SWAT") / "SWAT_train.pkl"
        if feature_path.exists():
            with feature_path.open("rb") as handle:
                return int(np.asarray(pickle.load(handle)).shape[1])
        return 51
    elif dataset == "WADI":
        # The official A2 files contain release-specific constant/all-NaN signals;
        # preprocessing records the normal-data-derived schema in this .npy file.
        feature_path = processed_dataset_path("WADI") / "WADI_train.npy"
        if feature_path.exists():
            return int(np.load(feature_path, mmap_mode="r").shape[1])
        return 93
    elif str(dataset).startswith("machine"):
        return 38
    elif dataset in ["NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"]:
        # Raw processed files also store step_type_code, but it is removed before modeling.
        return 3  # voltage, current, temperature
    elif dataset in ["CALCE", "CALCE2"]:
        # CALCE数据集是单特征时间序列
        return 1
    elif dataset == "BMS":
        # BMS 当前特征维度与 BMS_FEATURE_NAMES 保持同步
        return len(BMS_FEATURE_NAMES)
    elif dataset == "TSINGHUA_EV":
        return 7
    elif dataset == "CH_BATTERY_LFP_DISCHARGE":
        # CH-BATTERY 训练/预测当前只保留 7 个系统级核心特征。
        return 7
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
    """
    :param dataset: 数据集名称
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset in NASA_TELEMETRY_DATASETS:
        return [0]
    elif dataset in INDUSTRIAL_CONTROL_DATASETS:
        return None
    elif dataset == "SMD":
        return None
    elif dataset in ["NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"]:
        # 对于随机工况NASA数据，预测和重构所有时序特征
        return None
    elif dataset in ["CALCE", "CALCE2"]:
        # 对于CALCE数据集，我们关注单个特征
        return [0]
    elif dataset == "BMS":
        # 对于BMS数据集，我们关注所有特征
        return None
    elif dataset == "TSINGHUA_EV":
        return None
    elif dataset == "CH_BATTERY_LFP_DISCHARGE":
        # 只保留 7 个汇总量：SUM_VOLTAGE, SUM_CURRENT, SOC, MAX_CELL_VOLT, MIN_CELL_VOLT, MAX_TEMP, MIN_TEMP
        # 丢弃 124 个单体电压，显著降低模型复杂度
        return [0, 1, 2, 3, 4, 5, 6]
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_score_dims(dataset, target_dims=None):
    """Return output-relative dimensions used to form the global anomaly score.

    ``target_dims`` controls what the model predicts, while ``score_dims`` controls
    which predicted responses can raise the global alarm.  Keeping the two notions
    separate prevents operating-condition channels from being treated as faults.
    """
    dataset = str(dataset).upper()

    if dataset in NASA_TELEMETRY_DATASETS:
        response_dims = [0]
    elif dataset in NASA_RANDOM_DATASETS:
        # Current describes the imposed experiment; voltage/temperature respond.
        response_dims = [0, 2]
    elif dataset in {"TSINGHUA_EV", "CH_BATTERY_LFP_DISCHARGE"}:
        # voltage, cell-voltage extrema and temperature extrema; current/SOC are context.
        response_dims = [0, 3, 4, 5, 6]
    elif dataset == "BMS":
        # BMSnI remains a response so cluster-level over-current is not absorbed as
        # a frequency-regulation condition.  System variables and SOC provide context.
        response_names = [
            "BMSnVol_T",
            "BMSnVol_B",
            "BMSnI",
            "BMSnVmax",
            "BMSnVmin",
            "BMSnVmean",
            "BMSnTmax",
            "BMSnTmin",
            "BMSnTmean",
            "cell_v_std",
            "cell_v_range",
            "cell_v_max_dev_from_mean",
            "cell_v_min_dev_from_mean",
            "cell_t_std",
            "cell_t_range",
            *BMS_HIERARCHICAL_FEATURE_NAMES,
        ]
        response_dims = [BMS_FEATURE_NAMES.index(name) for name in response_names]
    else:
        return None

    if target_dims is None:
        return response_dims

    modeled_dims = [target_dims] if isinstance(target_dims, int) else list(target_dims)
    return [modeled_dims.index(dim) for dim in response_dims if dim in modeled_dims]


def inject_point_anomalies(data, anomaly_ratio=0.05):
    """
    向数据中注入点异常。
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
        mean_val = np.mean(data[:, 0])  # 第一列均值
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


def is_sequence_container(data):
    return isinstance(data, (list, tuple))


def ensure_sequence_list(data):
    if is_sequence_container(data):
        return [seq for seq in data if len(seq) > 0]
    if data is None or len(data) == 0:
        return []
    return [data]


def flatten_sequence_collection(sequence_collection, dtype=np.float32):
    arrays = []
    for sequence_data in sequence_collection:
        for seq in ensure_sequence_list(sequence_data):
            arrays.append(np.asarray(seq, dtype=dtype))
    if not arrays:
        raise ValueError("No sequence data available for concatenation")
    return np.concatenate(arrays, axis=0)


def normalize_sequence_container(sequence_data, scaler):
    if is_sequence_container(sequence_data):
        normalized_sequences = []
        for seq in ensure_sequence_list(sequence_data):
            normalized_seq, _ = normalize_data(np.asarray(seq, dtype=np.float32), scaler=scaler)
            normalized_sequences.append(normalized_seq.astype(np.float32, copy=False))
        return normalized_sequences

    normalized_data, _ = normalize_data(np.asarray(sequence_data, dtype=np.float32), scaler=scaler)
    return normalized_data.astype(np.float32, copy=False)


def _split_array_by_lengths(values, lengths):
    sequences = []
    offset = 0
    for length in lengths:
        length = int(length)
        sequences.append(np.asarray(values[offset:offset + length], dtype=np.float32))
        offset += length
    if offset != len(values):
        raise ValueError(f"Sequence lengths sum to {offset}, expected {len(values)}")
    return sequences


def get_nasa_telemetry_sequence_data(dataset, val_ratio=0.1, normalize=True):
    """Load MSL/SMAP without creating windows across entity boundaries."""
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    dataset = str(dataset).upper()
    if dataset not in NASA_TELEMETRY_DATASETS:
        raise ValueError(f"Unsupported NASA telemetry dataset: {dataset}")
    prefix = str(processed_dataset_path("data"))
    paths = {
        "train": os.path.join(prefix, f"{dataset}_train.pkl"),
        "test": os.path.join(prefix, f"{dataset}_test.pkl"),
        "label": os.path.join(prefix, f"{dataset}_test_label.pkl"),
        "train_lengths": os.path.join(prefix, f"{dataset}_train_lengths.pkl"),
        "test_lengths": os.path.join(prefix, f"{dataset}_test_lengths.pkl"),
    }
    missing = [name for name, file_path in paths.items() if not os.path.exists(file_path)]
    if missing:
        raise FileNotFoundError(
            f"Missing processed {dataset} files {missing}. Run preprocessing again to create boundary metadata."
        )
    with open(paths["train"], "rb") as handle:
        train_values = pickle.load(handle)
    with open(paths["test"], "rb") as handle:
        test_values = pickle.load(handle)
    with open(paths["label"], "rb") as handle:
        test_labels = pickle.load(handle)
    with open(paths["train_lengths"], "rb") as handle:
        train_lengths = pickle.load(handle)
    with open(paths["test_lengths"], "rb") as handle:
        test_lengths = pickle.load(handle)

    raw_train_sequences = _split_array_by_lengths(train_values, train_lengths)
    test_sequences = _split_array_by_lengths(test_values, test_lengths)
    test_label_sequences = _split_array_by_lengths(np.asarray(test_labels), test_lengths)
    train_sequences = []
    validation_sequences = []
    for sequence in raw_train_sequences:
        split_index = int(np.floor(len(sequence) * (1.0 - val_ratio)))
        train_sequences.append(sequence[:split_index])
        # A zero validation split is used by the external protocol.  Do not
        # create empty arrays here because sklearn scalers reject zero-row
        # inputs during normalization.
        if val_ratio > 0.0 and split_index < len(sequence):
            validation_sequences.append(sequence[split_index:])

    if normalize:
        _, scaler = normalize_data(np.concatenate(train_sequences, axis=0), scaler=None)
        train_sequences = [scaler.transform(sequence).astype(np.float32, copy=False) for sequence in train_sequences]
        validation_sequences = [scaler.transform(sequence).astype(np.float32, copy=False) for sequence in validation_sequences]
        test_sequences = [scaler.transform(sequence).astype(np.float32, copy=False) for sequence in test_sequences]
    # ``None`` tells the training runner that this protocol has no validation
    # split.  An empty list is treated as a segmented container and would make
    # the runner try to construct an empty validation window dataset.
    validation_data = validation_sequences if validation_sequences else None
    return train_sequences, validation_data, test_sequences, test_label_sequences


def get_msl_sequence_data(val_ratio=0.1, normalize=True):
    """Backward-compatible MSL loader using the shared telemetry protocol."""
    return get_nasa_telemetry_sequence_data("MSL", val_ratio=val_ratio, normalize=normalize)


def flatten_label_container(label_data):
    if label_data is None:
        return None

    if is_sequence_container(label_data):
        label_parts = []
        for seq_label in ensure_sequence_list(label_data):
            label_parts.append(np.asarray(seq_label))
        if not label_parts:
            return None
        return np.concatenate(label_parts, axis=0)

    return np.asarray(label_data)


def is_placeholder_zero_label(label_data):
    flattened = flatten_label_container(label_data)
    if flattened is None:
        return True
    flattened = np.asarray(flattened)
    return flattened.size == 0 or np.all(flattened == 0)


def describe_data_shape(data):
    if is_sequence_container(data):
        return [tuple(np.asarray(seq).shape) for seq in ensure_sequence_list(data)]
    return tuple(np.asarray(data).shape)


def get_available_prefixed_entities(prefix, file_prefix):
    entity_names = []
    for file_name in os.listdir(prefix):
        if file_name.startswith(f"{file_prefix}_") and file_name.endswith("_train.pkl"):
            entity_name = file_name.replace(f"{file_prefix}_", "").replace("_train.pkl", "")
            entity_names.append(entity_name)
    return sorted(entity_names)


def load_prefixed_processed_data(prefix, entity_name, file_prefix):
    train_path = os.path.join(prefix, f"{file_prefix}_{entity_name}_train.pkl")
    test_path = os.path.join(prefix, f"{file_prefix}_{entity_name}_test.pkl")
    label_path = os.path.join(prefix, f"{file_prefix}_{entity_name}_test_label.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{file_prefix} train file not found for entity {entity_name}: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{file_prefix} test file not found for entity {entity_name}: {test_path}")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    test_label = None
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            test_label = pickle.load(f)

    return train_data, test_data, test_label


def resolve_prefixed_entities(prefix, file_prefix, single_entity=None, train_entities=None, test_entities=None):
    available_entities = get_available_prefixed_entities(prefix, file_prefix)
    if not available_entities:
        raise FileNotFoundError(f"No processed {file_prefix} entity data found in {prefix}")

    selected_single = str(single_entity).strip() if single_entity is not None else None
    train_entities = parse_entity_list(train_entities)
    test_entities = parse_entity_list(test_entities)

    if selected_single:
        train_entities = [selected_single]
        test_entities = [selected_single]
    elif train_entities is None and test_entities is None:
        train_entities = [available_entities[0]]
        test_entities = [available_entities[0]]
    elif train_entities is None:
        train_entities = list(test_entities)
    elif test_entities is None:
        test_entities = list(train_entities)

    missing_entities = [b for b in set(train_entities + test_entities) if b not in available_entities]
    if missing_entities:
        raise FileNotFoundError(
            f"{file_prefix} entities not found in processed data: {missing_entities}. "
            f"Available entities: {available_entities}"
        )

    return train_entities, test_entities


def _get_nasa_random_battery_data(dataset, nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None,
                                  normalize=False, prefix=None):
    if dataset not in NASA_RANDOM_DATASET_PREFIX:
        raise ValueError(f"Unsupported NASA Random dataset: {dataset}")

    file_prefix = NASA_RANDOM_DATASET_PREFIX[dataset]
    if prefix is None:
        prefix = str(processed_dataset_path(dataset))

    train_batteries, test_batteries = resolve_prefixed_entities(
        prefix,
        file_prefix,
        single_entity=nasa_battery_id,
        train_entities=nasa_train_batteries,
        test_entities=nasa_test_batteries,
    )

    print(f"Using {dataset} train batteries: {train_batteries}")
    print(f"Using {dataset} test batteries: {test_batteries}")

    train_data_map = {}
    test_data_map = {}
    test_label_map = {}

    if dataset in NASA_RANDOM_DATASETS:
        selected_batteries = sorted(set(train_batteries + test_batteries))
        print(f"Using {dataset} processed split batteries: {selected_batteries}")

    for battery_name in train_batteries:
        battery_train_data, _, _ = load_prefixed_processed_data(prefix, battery_name, file_prefix)
        if is_sequence_container(battery_train_data):
            train_data_map[battery_name] = [
                np.asarray(seq, dtype=np.float32)[:, 1:4]
                for seq in ensure_sequence_list(battery_train_data)
            ]
        else:
            train_data_map[battery_name] = np.asarray(battery_train_data, dtype=np.float32)[:, 1:4]

    for battery_name in test_batteries:
        _, battery_test_data, battery_test_label = load_prefixed_processed_data(prefix, battery_name, file_prefix)
        has_processed_test = battery_test_data is not None and len(battery_test_data) > 0
        if not has_processed_test:
            raise ValueError(
                f"{dataset} battery {battery_name} is missing a valid processed test split. "
                "All dataset splits must be generated during preprocessing."
            )

        if is_sequence_container(battery_test_data):
            test_data_map[battery_name] = [
                np.asarray(seq, dtype=np.float32)[:, 1:4]
                for seq in ensure_sequence_list(battery_test_data)
            ]
        else:
            test_data_map[battery_name] = np.asarray(battery_test_data, dtype=np.float32)[:, 1:4]

        if battery_test_label is None or is_placeholder_zero_label(battery_test_label):
            test_label_map[battery_name] = None
        elif is_sequence_container(battery_test_label):
            test_label_map[battery_name] = [
                np.asarray(seq_label) for seq_label in ensure_sequence_list(battery_test_label)
            ]
        else:
            test_label_map[battery_name] = np.asarray(battery_test_label)

    if all(label is None for label in test_label_map.values()):
        print(f"{dataset} test labels are unavailable for the selected batteries; supervised metrics should be skipped.")

    if normalize:
        concatenated_train = flatten_sequence_collection(train_data_map.values(), dtype=np.float32)
        scaler = NasaRandomPhysicalScaler().fit(concatenated_train)

        normalized_train_map = {}
        for battery_name, battery_train_data in train_data_map.items():
            normalized_train_map[battery_name] = normalize_sequence_container(battery_train_data, scaler=scaler)
        train_data_map = normalized_train_map

        normalized_test_map = {}
        for battery_name, battery_test_data in test_data_map.items():
            normalized_test_map[battery_name] = normalize_sequence_container(battery_test_data, scaler=scaler)
        test_data_map = normalized_test_map

    return (train_data_map, None), (test_data_map, test_label_map)


def get_nasa_random_battery_data(dataset, nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None,
                                 normalize=False, prefix=None):
    return _get_nasa_random_battery_data(
        dataset,
        nasa_battery_id=nasa_battery_id,
        nasa_train_batteries=nasa_train_batteries,
        nasa_test_batteries=nasa_test_batteries,
        normalize=normalize,
        prefix=prefix,
    )


def get_available_bms_clusters(prefix):
    cluster_names = []
    for file_name in os.listdir(prefix):
        if file_name.startswith("BMS_") and file_name.endswith("_train.pkl") and "_cluster" in file_name:
            cluster_name = file_name.replace("_train.pkl", "")
            cluster_names.append(cluster_name)
    return sorted(cluster_names)


def load_bms_cluster_processed_data(prefix, cluster_name):
    train_path = os.path.join(prefix, f"{cluster_name}_train.pkl")
    test_path = os.path.join(prefix, f"{cluster_name}_test.pkl")
    label_path = os.path.join(prefix, f"{cluster_name}_test_label.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"BMS train file not found for cluster {cluster_name}: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"BMS test file not found for cluster {cluster_name}: {test_path}")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    test_label = None
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            test_label = pickle.load(f)
        if is_placeholder_zero_label(test_label):
            test_label = None

    return train_data, test_data, test_label


def get_bms_cluster_data(normalize=False, prefix=None):
    if prefix is None:
        prefix = str(processed_dataset_path("BMS"))
    cluster_names = get_available_bms_clusters(prefix)
    if not cluster_names:
        raise FileNotFoundError(f"No processed BMS cluster data found in {prefix}")

    print(f"Using BMS clusters: {cluster_names}")

    train_data_map = {}
    test_data_map = {}
    test_label_map = {}

    for cluster_name in cluster_names:
        cluster_train_data, cluster_test_data, cluster_test_label = load_bms_cluster_processed_data(prefix, cluster_name)
        train_data_map[cluster_name] = np.asarray(cluster_train_data, dtype=np.float32)
        test_data_map[cluster_name] = np.asarray(cluster_test_data, dtype=np.float32)
        test_label_map[cluster_name] = None if cluster_test_label is None else np.asarray(cluster_test_label)

    if normalize:
        concatenated_train = np.concatenate(list(train_data_map.values()), axis=0)
        scaler = MaxAbsScaler().fit(concatenated_train)

        # Preserve zero current and use shared scales for physically related
        # max/min/mean channels so their differences remain meaningful.
        for feature_group in (
            ["BMSnVmax", "BMSnVmin", "BMSnVmean"],
            ["BMSnTmax", "BMSnTmin", "BMSnTmean"],
        ):
            indices = [BMS_FEATURE_NAMES.index(name) for name in feature_group]
            shared_scale = max(float(np.max(np.abs(concatenated_train[:, indices]))), 1e-6)
            scaler.scale_[indices] = shared_scale
            scaler.max_abs_[indices] = shared_scale

        normalized_train_map = {}
        for cluster_name, cluster_train_data in train_data_map.items():
            normalized_train_map[cluster_name] = scaler.transform(cluster_train_data).astype(np.float32, copy=False)
        train_data_map = normalized_train_map

        normalized_test_map = {}
        for cluster_name, cluster_test_data in test_data_map.items():
            normalized_test_map[cluster_name] = scaler.transform(cluster_test_data).astype(np.float32, copy=False)
        test_data_map = normalized_test_map

    return (train_data_map, None), (test_data_map, test_label_map)


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0,
             nasa_battery_id=None, nasa_train_batteries=None, nasa_test_batteries=None):
    """
    从 pkl 文件读取数据。

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    来自 OmniAnomaly 的方法（https://github.com/NetManAIOps/OmniAnomaly）。
    Used in the traditional training pipeline (one model per entity)
    """
    if str(dataset).startswith("machine"):
        prefix = str(processed_dataset_path("ServerMachineDataset"))
    elif dataset in NASA_TELEMETRY_DATASETS:
        prefix = str(processed_dataset_path("data"))
    elif dataset == "NASA_RANDOM_CHARGE":
        prefix = str(processed_dataset_path("NASA_RANDOM_CHARGE"))
    elif dataset == "NASA_RANDOM_DISCHARGE":
        prefix = str(processed_dataset_path("NASA_RANDOM_DISCHARGE"))
    elif dataset in ["CALCE", "CALCE2"]:
        prefix = str(processed_dataset_path("CALCE"))
    elif dataset == "BMS":
        prefix = str(processed_dataset_path("BMS"))
    else:
        prefix = str(processed_dataset_path(str(dataset)))
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

    if dataset in NASA_RANDOM_DATASETS:
        (train_data_map, _), (test_data_map, test_label_map) = get_nasa_random_battery_data(
            dataset,
            nasa_battery_id=nasa_battery_id,
            nasa_train_batteries=nasa_train_batteries,
            nasa_test_batteries=nasa_test_batteries,
            normalize=False,
            prefix=prefix,
        )
        train_data = flatten_sequence_collection(train_data_map.values(), dtype=np.float32)
        test_data = flatten_sequence_collection(test_data_map.values(), dtype=np.float32)

        has_any_label = any(label is not None for label in test_label_map.values())
        if has_any_label:
            aligned_labels = []
            for battery_name in test_data_map.keys():
                battery_test_label = test_label_map[battery_name]
                if battery_test_label is None:
                    battery_sequences = ensure_sequence_list(test_data_map[battery_name])
                    for seq in battery_sequences:
                        aligned_labels.append(np.zeros(len(seq), dtype=np.int32))
                else:
                    flattened_label = flatten_label_container(battery_test_label)
                    aligned_labels.append(flattened_label)
            test_label = np.concatenate(aligned_labels, axis=0)
        else:
            test_label = None
    elif dataset in ["CALCE", "CALCE2"]:
        # 统一处理CALCE和CALCE2数据集
        # 使用训练/测试划分方式加载数据

        if dataset == "CALCE":
            # 对于CALCE数据集，前6个实体作为训练集，后6个实体作为测试集
            train_entities, test_entities = get_calce_train_test_splits()
        else:  # CALCE2 分支
            # 对于CALCE2数据集，前14个单元作为训练集，后9个单元作为测试集
            train_entities, test_entities = get_calce2_train_test_splits()

        # 加载训练数据
        train_data_list = []
        for entity_name in train_entities:
            if dataset == "CALCE":
                (x_train, _), (_, _) = load_calce_entity_data(entity_name)
            else:  # CALCE2 分支
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
            else:  # CALCE2 分支
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
        try:
            with open(os.path.join(prefix, dataset + "_train.pkl"), "rb") as f:
                train_data = pickle.load(f)
            with open(os.path.join(prefix, dataset + "_test.pkl"), "rb") as f:
                test_data = pickle.load(f)
            try:
                with open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb") as f:
                    test_label = pickle.load(f)
            except (KeyError, FileNotFoundError):
                test_label = None
        except (KeyError, FileNotFoundError):
            (train_data_map, _), (test_data_map, test_label_map) = get_bms_cluster_data(
                normalize=False,
                prefix=prefix,
            )
            train_data = np.concatenate(list(train_data_map.values()), axis=0)
            test_data = np.concatenate(list(test_data_map.values()), axis=0)
            has_any_label = any(label is not None for label in test_label_map.values())
            if has_any_label:
                aligned_labels = []
                for cluster_name, cluster_test_data in test_data_map.items():
                    cluster_test_label = test_label_map[cluster_name]
                    if cluster_test_label is None:
                        aligned_labels.append(np.zeros(len(cluster_test_data), dtype=np.int32))
                    else:
                        aligned_labels.append(np.asarray(cluster_test_label))
                test_label = np.concatenate(aligned_labels, axis=0)
            else:
                test_label = None
    elif dataset == "WADI":
        train_data = np.load(os.path.join(prefix, "WADI_train.npy"), mmap_mode="r")[train_start:train_end, :]
        try:
            test_data = np.load(os.path.join(prefix, "WADI_test.npy"), mmap_mode="r")[test_start:test_end, :]
            test_label = np.load(os.path.join(prefix, "WADI_test_label.npy"), mmap_mode="r")[test_start:test_end]
        except FileNotFoundError:
            test_data, test_label = None, None
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
        if dataset in NASA_RANDOM_DATASETS:
            scaler = NasaRandomPhysicalScaler().fit(train_data)
            train_data = scaler.transform(train_data)
        elif isinstance(train_data, dict):
            concatenated_train = flatten_sequence_collection(train_data.values(), dtype=np.float32)
            _, scaler = normalize_data(concatenated_train, scaler=None)
            normalized_train_map = {}
            for battery_name, battery_train_data in train_data.items():
                normalized_train_map[battery_name] = normalize_sequence_container(battery_train_data, scaler=scaler)
            train_data = normalized_train_map
        else:
            train_data, scaler = normalize_data(train_data, scaler=None)

        if test_data is not None:
            if isinstance(test_data, dict):
                normalized_test_map = {}
                for battery_name, battery_test_data in test_data.items():
                    normalized_test_map[battery_name] = normalize_sequence_container(battery_test_data, scaler=scaler)
                test_data = normalized_test_map
            elif dataset in NASA_RANDOM_DATASETS:
                test_data = scaler.transform(test_data)
            else:
                test_data, _ = normalize_data(test_data, scaler=scaler)

    if isinstance(train_data, dict):
        print("train set batteries: ", list(train_data.keys()))
        print("train set shapes: ", {k: describe_data_shape(v) for k, v in train_data.items()})
    else:
        print("train set shape: ", train_data.shape)
    if isinstance(test_data, dict):
        print("test set batteries: ", list(test_data.keys()))
        print("test set shapes: ", {k: describe_data_shape(v) for k, v in test_data.items()})
        if isinstance(test_label, dict):
            print("test set label shapes: ", {k: None if v is None else describe_data_shape(v) for k, v in test_label.items()})
        else:
            print("test set label shape: ", None)
    else:
        print("test set shape: ", test_data.shape if test_data is not None else "None")
        print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1, stride=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.stride = max(int(stride), 1)

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start : start + self.window]
        y = self.data[start + self.window : start + self.window + self.horizon]
        return x, y

    def __len__(self):
        available = len(self.data) - self.window
        if available <= 0:
            return 0
        return 1 + (available - 1) // self.stride


def resolve_dataloader_options(
    num_workers=4,
    pin_memory=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    num_workers = max(int(num_workers), 0)
    options = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        options["persistent_workers"] = bool(persistent_workers)
        options["prefetch_factor"] = max(int(prefetch_factor), 1)
    return options


def create_data_loaders(
    train_dataset,
    batch_size,
    val_split=0.1,
    shuffle=True,
    test_dataset=None,
    val_dataset=None,
    num_workers=4,
    pin_memory=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    train_loader, val_loader, test_loader = None, None, None
    loader_options = resolve_dataloader_options(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    if val_dataset is not None:
        print(f"train_size: {len(train_dataset)}")
        print(f"validation_size: {len(val_dataset)} (explicit split)")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **loader_options,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_options,
        )
    elif val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **loader_options,
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
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            **loader_options,
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            **loader_options,
        )

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_options,
        )
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: 损失字典
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
    从指定路径加载模型参数。
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
    prefix = str(processed_dataset_path("CALCE"))
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
    prefix = str(processed_dataset_path("CALCE"))
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
    prefix = str(processed_dataset_path("CALCE"))
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
    prefix = str(processed_dataset_path("CALCE"))
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
    # CALCE2 分支实体编号为Cell1-Cell23，其中Cell1-Cell14为训练集，Cell15-Cell23为测试集
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
