"""将原始数据预处理为项目所需的训练和测试产物。"""

import os
import time
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

from src.args import get_parser
from src.data.spectral_residual import apply_spectral_residual_cleaning
from src.project_paths import processed_dataset_path, resolve_dataset_root


BMS_CLUSTER_MAIN_FEATURES = [
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
]

BMS_GROUP_CONTEXT_FEATURES = [
    "SYS_Vol",
    "SYS_I",
    "SYS_SOH",
    "SYS_Vmax",
    "SYS_Vmin",
    "SYS_Tmax",
    "SYS_Tmin",
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

BMS_FEATURE_NAMES = BMS_CLUSTER_MAIN_FEATURES + [
    "cell_v_std",
    "cell_v_range",
    "cell_v_max_dev_from_mean",
    "cell_v_min_dev_from_mean",
    "cell_t_std",
    "cell_t_range",
] + BMS_GROUP_CONTEXT_FEATURES + BMS_HIERARCHICAL_FEATURE_NAMES

NASA_RANDOM_STEP_COMMENTS = {
    "discharge (random walk)",
    "rest (random walk)",
    "rest post random walk discharge",
    "charge (after random walk discharge)",
}

NASA_RANDOM_STEP_TYPE_CODES = {
    "C": 1.0,
    "D": -1.0,
    "R": 0.0,
}

NASA_RANDOM_DOWNSAMPLE_STRIDES = {
    "NASA_RANDOM_DISCHARGE": 4,
}


def _sanitize_bms_entity_name(name):
    keep_chars = []
    for ch in str(name):
        if ch.isalnum():
            keep_chars.append(ch)
        else:
            keep_chars.append("_")
    return "".join(keep_chars).strip("_")


def _get_bms_series_name(bundle_prefix_name):
    prefix = str(bundle_prefix_name)
    marker = "_StartDate_"
    if marker in prefix:
        prefix = prefix.split(marker, 1)[0]
    return _sanitize_bms_entity_name(prefix)


def _load_bms_excel(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if "Date" not in df.columns:
        raise ValueError(f"{file_path} 缺少 Date 列")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def _build_bms_detail_summary(detail_df, prefix):
    value_columns = [col for col in detail_df.columns if col.startswith(prefix)]
    if not value_columns:
        return detail_df[["Date"]].copy()

    detail_values = detail_df[value_columns].apply(pd.to_numeric, errors="coerce")
    summary_df = pd.DataFrame({"Date": detail_df["Date"].values})

    row_mean = detail_values.mean(axis=1)
    row_max = detail_values.max(axis=1)
    row_min = detail_values.min(axis=1)

    if prefix == "BMSnV":
        summary_df["cell_v_std"] = detail_values.std(axis=1, ddof=0)
        summary_df["cell_v_range"] = row_max - row_min
        summary_df["cell_v_max_dev_from_mean"] = row_max - row_mean
        summary_df["cell_v_min_dev_from_mean"] = row_min - row_mean
    else:
        summary_df["cell_t_std"] = detail_values.std(axis=1, ddof=0)
        summary_df["cell_t_range"] = row_max - row_min

    return summary_df


def _build_bms_cluster_feature_frame(stat_df, volt_summary_df, temp_summary_df, group_df):
    merged_df = stat_df[["Date"] + [col for col in BMS_CLUSTER_MAIN_FEATURES if col in stat_df.columns]].copy()
    merged_df = merged_df.merge(volt_summary_df, on="Date", how="left")
    merged_df = merged_df.merge(temp_summary_df, on="Date", how="left")
    merged_df = merged_df.merge(
        group_df[["Date"] + [col for col in BMS_GROUP_CONTEXT_FEATURES if col in group_df.columns]].copy(),
        on="Date",
        how="left",
    )

    for feature_name in BMS_FEATURE_NAMES:
        if feature_name not in merged_df.columns:
            merged_df[feature_name] = np.nan

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    feature_df = merged_df[BMS_FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.interpolate(limit_direction="both")
    feature_df = feature_df.ffill().bfill().fillna(0.0)
    eps = 1e-6
    feature_df["hier_vmax_sys_gap"] = feature_df["BMSnVmax"] - feature_df["SYS_Vmax"]
    feature_df["hier_vmin_sys_gap"] = feature_df["BMSnVmin"] - feature_df["SYS_Vmin"]
    feature_df["hier_tmax_sys_gap"] = feature_df["BMSnTmax"] - feature_df["SYS_Tmax"]
    feature_df["hier_tmin_sys_gap"] = feature_df["BMSnTmin"] - feature_df["SYS_Tmin"]
    feature_df["hier_soh_sys_gap"] = feature_df["BMSnSOH"] - feature_df["SYS_SOH"]
    feature_df["hier_cell_v_range_ratio"] = feature_df["cell_v_range"] / (np.abs(feature_df["BMSnVmean"]) + eps)
    feature_df["hier_cell_t_range_ratio"] = feature_df["cell_t_range"] / (np.abs(feature_df["BMSnTmean"]) + eps)
    feature_df = feature_df.astype(np.float32)
    return pd.concat([merged_df[["Date"]], feature_df], axis=1)


def _save_bms_processed_splits(output_folder, entity_name, feature_df, apply_sr_cleaning=False):
    if isinstance(feature_df, pd.DataFrame) and "Date" in feature_df.columns:
        feature_df = feature_df.drop(columns=["Date"])

    feature_array = np.asarray(feature_df, dtype=np.float32)

    if apply_sr_cleaning:
        print(f"Applying spectral residual cleaning to BMS {entity_name} data...")
        feature_array = apply_spectral_residual_cleaning(feature_array, threshold=3.0)
        print(f"Cleaning completed. Shape: {feature_array.shape}")

    split_index = int(len(feature_array) * 0.8)
    train_data = feature_array[:split_index]
    test_data = feature_array[split_index:]
    test_labels = np.zeros(len(test_data), dtype=np.int32)

    with open(path.join(output_folder, f"{entity_name}_train.pkl"), "wb") as file:
        dump(train_data, file)
    with open(path.join(output_folder, f"{entity_name}_test.pkl"), "wb") as file:
        dump(test_data, file)
    with open(path.join(output_folder, f"{entity_name}_test_label.pkl"), "wb") as file:
        dump(test_labels, file)

    return train_data, test_data, test_labels


def _cleanup_existing_bms_processed_files(output_folder):
    removed_count = 0
    for filename in listdir(output_folder):
        if (
            filename == "BMS_train.pkl"
            or filename == "BMS_test.pkl"
            or filename == "BMS_test_label.pkl"
            or (filename.startswith("BMS_") and filename.endswith(".pkl"))
        ):
            os.remove(path.join(output_folder, filename))
            removed_count += 1
    if removed_count > 0:
        print(f"[BMS] Removed {removed_count} legacy processed file(s) from {output_folder}")
    return removed_count


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
        else:  # 充电
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


def _extract_matlab_scalar(value):
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.flat[0]

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _get_matlab_string(value):
    return str(_extract_matlab_scalar(value)).strip()


def _flatten_nasa_random_step(step_struct, step_stride=1):
    comment = _get_matlab_string(step_struct["comment"])
    if comment not in NASA_RANDOM_STEP_COMMENTS:
        return None

    step_type = _get_matlab_string(step_struct["type"]).upper()
    step_type_code = NASA_RANDOM_STEP_TYPE_CODES.get(step_type, 0.0)

    voltage = np.asarray(step_struct["voltage"][0], dtype=np.float32).reshape(-1)
    current = np.asarray(step_struct["current"][0], dtype=np.float32).reshape(-1)
    temperature = np.asarray(step_struct["temperature"][0], dtype=np.float32).reshape(-1)

    valid_length = min(len(voltage), len(current), len(temperature))
    if valid_length <= 0:
        return None

    step_type_column = np.full(valid_length, step_type_code, dtype=np.float32)
    step_array = np.column_stack([
        step_type_column,
        voltage[:valid_length],
        current[:valid_length],
        temperature[:valid_length],
    ])

    if step_stride > 1:
        step_array = step_array[::step_stride]

    return step_array.astype(np.float32, copy=False)


def _build_nasa_random_segments(steps, step_stride=1):
    segments = []
    current_segment_steps = []

    for step_struct in steps:
        flattened_step = _flatten_nasa_random_step(step_struct, step_stride=step_stride)
        if flattened_step is None:
            if current_segment_steps:
                segments.append(np.vstack(current_segment_steps).astype(np.float32, copy=False))
                current_segment_steps = []
            continue

        current_segment_steps.append(flattened_step)

    if current_segment_steps:
        segments.append(np.vstack(current_segment_steps).astype(np.float32, copy=False))

    return [segment for segment in segments if len(segment) > 0]


def _split_nasa_random_segments(segments, train_ratio=0.8):
    if not segments:
        raise ValueError("No NASA random-walk segments available for splitting")

    total_length = sum(len(segment) for segment in segments)
    if total_length <= 1:
        raise ValueError("NASA random-walk data is too short to split")

    train_target = max(1, int(total_length * train_ratio))
    train_segments = []
    test_segments = []
    accumulated = 0

    for segment in segments:
        segment_len = len(segment)
        if accumulated >= train_target:
            test_segments.append(segment.astype(np.float32, copy=False))
            continue

        next_accumulated = accumulated + segment_len
        if next_accumulated <= train_target:
            train_segments.append(segment.astype(np.float32, copy=False))
            accumulated = next_accumulated
            continue

        split_point = train_target - accumulated
        if split_point > 0:
            train_segments.append(segment[:split_point].astype(np.float32, copy=False))
        if split_point < segment_len:
            test_segments.append(segment[split_point:].astype(np.float32, copy=False))
        accumulated = train_target

    train_segments = [segment for segment in train_segments if len(segment) > 0]
    test_segments = [segment for segment in test_segments if len(segment) > 0]
    if not train_segments or not test_segments:
        raise ValueError("NASA random-walk split failed to produce both train and test segments")

    return train_segments, test_segments


def _process_nasa_random_dataset(dataset_folder, output_folder, file_prefix, apply_sr_cleaning=False):
    mat_files = sorted([f for f in os.listdir(dataset_folder) if f.lower().endswith(".mat")])
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {dataset_folder}")

    step_stride = max(1, int(NASA_RANDOM_DOWNSAMPLE_STRIDES.get(file_prefix, 1)))
    if step_stride > 1:
        print(f"[{file_prefix}] Applying temporal downsampling with stride={step_stride}")

    for filename in mat_files:
        battery_id = filename.split(".mat")[0]
        mat_path = os.path.join(dataset_folder, filename)
        print(f"[{file_prefix}] Processing {filename}...")

        mat_data = loadmat(mat_path)
        if "data" not in mat_data:
            raise KeyError(f"{mat_path} does not contain top-level 'data' struct")

        battery_struct = mat_data["data"][0, 0]
        steps = battery_struct["step"][0]

        segments = _build_nasa_random_segments(steps, step_stride=step_stride)
        if not segments:
            print(f"[{file_prefix}][{battery_id}] No supported random-walk steps found, skipping.")
            continue

        # 保持 NASA_RANDOM 与项目的已处理数据约定一致：
        # 按电池实体划分训练/测试片段
        # 后续保存格式与普通 NASA 数据保持一致
        train_segments, test_segments = _split_nasa_random_segments(segments, train_ratio=0.8)
        train_data = [segment.astype(np.float32, copy=False) for segment in train_segments]
        test_data = [segment.astype(np.float32, copy=False) for segment in test_segments]
        test_labels = [np.zeros(len(segment), dtype=np.int32) for segment in test_data]

        if apply_sr_cleaning:
            print(f"[{file_prefix}][{battery_id}] Applying spectral residual cleaning to train split...")
            cleaned_train_data = []
            for segment in train_data:
                cleaned_segment = apply_spectral_residual_cleaning(segment, threshold=3.0).astype(np.float32, copy=False)
                cleaned_train_data.append(cleaned_segment)
            train_data = cleaned_train_data
            print(
                f"[{file_prefix}][{battery_id}] Cleaning completed. "
                f"Train segments: {len(train_data)}, "
                f"steps: {sum(len(segment) for segment in train_data)}"
            )

        with open(path.join(output_folder, f"{file_prefix}_{battery_id}_train.pkl"), "wb") as file:
            dump(train_data, file)
        with open(path.join(output_folder, f"{file_prefix}_{battery_id}_test.pkl"), "wb") as file:
            dump(test_data, file)
        with open(path.join(output_folder, f"{file_prefix}_{battery_id}_test_label.pkl"), "wb") as file:
            dump(test_labels, file)

        print(
            f"[{file_prefix}][{battery_id}] Saved segments -> "
            f"train_segments={len(train_data)}, train_steps={sum(len(segment) for segment in train_data)}, "
            f"test_segments={len(test_data)}, test_steps={sum(len(segment) for segment in test_data)}, "
            f"label_segments={len(test_labels)}"
        )


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
    """ 来自 OmniAnomaly 的方法（https://github.com/NetManAIOps/OmniAnomaly）。 """

    if dataset == "SMD":
        dataset_folder = str(resolve_dataset_root("SMD", "ServerMachineDataset"))
        output_folder = str(processed_dataset_path("ServerMachineDataset"))
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
        dataset_folder = str(resolve_dataset_root("DATA", "data"))
        output_folder = str(processed_dataset_path("data"))
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

    elif dataset == "NASA_RANDOM_CHARGE":
        dataset_folder = str(resolve_dataset_root("NASA_RANDOM_CHARGE", "NASA_RANDOM_CHARGE"))
        output_folder = str(processed_dataset_path("NASA_RANDOM_CHARGE"))
        makedirs(output_folder, exist_ok=True)
        _process_nasa_random_dataset(
            dataset_folder,
            output_folder,
            file_prefix="NASA_RANDOM_CHARGE",
            apply_sr_cleaning=apply_sr_cleaning,
        )

    elif dataset == "NASA_RANDOM_DISCHARGE":
        dataset_folder = str(resolve_dataset_root("NASA_RANDOM_DISCHARGE", "NASA_RANDOM_DISCHARGE"))
        output_folder = str(processed_dataset_path("NASA_RANDOM_DISCHARGE"))
        makedirs(output_folder, exist_ok=True)
        _process_nasa_random_dataset(
            dataset_folder,
            output_folder,
            file_prefix="NASA_RANDOM_DISCHARGE",
            apply_sr_cleaning=apply_sr_cleaning,
        )

    elif dataset == "BMS":
        dataset_folder = str(resolve_dataset_root("BMS", "BMS"))
        output_folder = str(processed_dataset_path("BMS"))
        makedirs(output_folder, exist_ok=True)
        _cleanup_existing_bms_processed_files(output_folder)
        print(f"[BMS] Dataset folder: {dataset_folder}")
        print(f"[BMS] Output folder: {output_folder}")
        print(f"[BMS] Spectral residual cleaning: {apply_sr_cleaning}")
        suffix_map = {
            "_BMS0Data.xls": "group",
            "_BMSnStatData.xls": "stat",
            "_BMSnDetailTempData.xls": "temp",
            "_BMSnDetailVoltData.xls": "volt",
            "_BMS0Data.xlsx": "group",
            "_BMSnStatData.xlsx": "stat",
            "_BMSnDetailTempData.xlsx": "temp",
            "_BMSnDetailVoltData.xlsx": "volt",
        }

        grouped_files = {}
        for filename in listdir(dataset_folder):
            if filename.startswith("~$"):
                continue
            for suffix, key in suffix_map.items():
                if filename.endswith(suffix):
                    prefix_name = filename[: -len(suffix)]
                    grouped_files.setdefault(prefix_name, {})[key] = path.join(dataset_folder, filename)
                    break

        print(f"[BMS] Detected {len(grouped_files)} candidate bundle(s)")

        bms_start_time = time.perf_counter()
        cluster_frame_map = {}
        processed_bundle_cluster_count = 0

        for bundle_idx, (prefix_name, file_map) in enumerate(sorted(grouped_files.items()), start=1):
            required_keys = {"group", "stat", "temp", "volt"}
            if not required_keys.issubset(file_map.keys()):
                missing_keys = sorted(required_keys - set(file_map.keys()))
                print(f"Skipping {prefix_name}, missing files: {missing_keys}")
                continue

            bundle_start_time = time.perf_counter()
            print(f"[BMS][{bundle_idx}/{len(grouped_files)}] Processing bundle: {prefix_name}")
            print(f"[BMS][{bundle_idx}/{len(grouped_files)}] Loading group table...")
            group_df = _load_bms_excel(file_map["group"])
            print(f"[BMS][{bundle_idx}/{len(grouped_files)}] Group rows: {len(group_df)}")
            print(f"[BMS][{bundle_idx}/{len(grouped_files)}] Reading workbook metadata...")
            stat_xls = pd.ExcelFile(file_map["stat"])
            temp_xls = pd.ExcelFile(file_map["temp"])
            volt_xls = pd.ExcelFile(file_map["volt"])

            common_sheets = [sheet for sheet in stat_xls.sheet_names if sheet in temp_xls.sheet_names and sheet in volt_xls.sheet_names]
            if not common_sheets:
                print(f"No shared cluster sheets found for {prefix_name}")
                continue
            print(f"[BMS][{bundle_idx}/{len(grouped_files)}] Shared cluster sheets: {common_sheets}")

            bundle_name = _sanitize_bms_entity_name(prefix_name)
            for sheet_idx, sheet_name in enumerate(common_sheets, start=1):
                sheet_start_time = time.perf_counter()
                print(f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] Loading stat/temp/volt data for cluster {sheet_name}...")
                stat_df = _load_bms_excel(file_map["stat"], sheet_name=sheet_name)
                temp_df = _load_bms_excel(file_map["temp"], sheet_name=sheet_name)
                volt_df = _load_bms_excel(file_map["volt"], sheet_name=sheet_name)
                print(
                    f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] "
                    f"Rows loaded - stat: {len(stat_df)}, temp: {len(temp_df)}, volt: {len(volt_df)}"
                )

                print(f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] Building temperature summary...")
                temp_summary_df = _build_bms_detail_summary(temp_df, prefix="BMSnT")
                print(f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] Building voltage summary...")
                volt_summary_df = _build_bms_detail_summary(volt_df, prefix="BMSnV")
                print(f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] Merging cluster features...")
                feature_df = _build_bms_cluster_feature_frame(stat_df, volt_summary_df, temp_summary_df, group_df)
                print(
                    f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] "
                    f"Feature frame shape: {feature_df.shape}"
                )

                series_name = _get_bms_series_name(prefix_name)
                cluster_entity_name = f"BMS_{series_name}_cluster{sheet_name}"
                cluster_frame_map.setdefault(cluster_entity_name, []).append(feature_df)
                processed_bundle_cluster_count += 1
                print(
                    f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] "
                    f"Queued segment for {cluster_entity_name}, current segments: "
                    f"{len(cluster_frame_map[cluster_entity_name])}"
                )
                sheet_elapsed = time.perf_counter() - sheet_start_time
                print(
                    f"[BMS][{bundle_idx}/{len(grouped_files)}][Sheet {sheet_idx}/{len(common_sheets)}] "
                    f"Elapsed: {sheet_elapsed:.2f}s"
                )

            bundle_elapsed = time.perf_counter() - bundle_start_time
            print(
                f"[BMS][{bundle_idx}/{len(grouped_files)}] Bundle completed in {bundle_elapsed:.2f}s"
            )

        if not cluster_frame_map:
            raise FileNotFoundError("未找到可用于构建 BMS 特征的数据文件组合")

        print(
            f"[BMS] Aggregating {processed_bundle_cluster_count} daily cluster segment(s) into "
            f"{len(cluster_frame_map)} continuous cluster sequence(s)..."
        )
        saved_cluster_count = 0
        for cluster_entity_name, feature_frames in sorted(cluster_frame_map.items()):
            concat_df = pd.concat(feature_frames, axis=0, ignore_index=True)
            concat_df["Date"] = pd.to_datetime(concat_df["Date"], errors="coerce")
            concat_df = concat_df.dropna(subset=["Date"]).sort_values("Date")
            concat_df = concat_df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
            train_data, test_data, test_labels = _save_bms_processed_splits(
                output_folder,
                cluster_entity_name,
                concat_df,
                apply_sr_cleaning=apply_sr_cleaning,
            )
            saved_cluster_count += 1
            print(
                f"[BMS] Saved continuous cluster {cluster_entity_name}: "
                f"segments={len(feature_frames)}, total_steps={len(concat_df)}, "
                f"train={train_data.shape}, test={test_data.shape}, labels={test_labels.shape}"
            )

        print(
            f"[BMS] Saved {saved_cluster_count} continuous cluster split(s); "
            f"merged BMS pkl files are no longer generated."
        )
        print(f"[BMS] Total elapsed: {time.perf_counter() - bms_start_time:.2f}s")
    elif dataset == "NASA":
        dataset_folder = str(resolve_dataset_root("NASA", "NASA"))
        output_folder = str(processed_dataset_path("NASA"))
        makedirs(output_folder, exist_ok=True)

        # 获取所有MAT文件
        mat_files = [f for f in os.listdir(dataset_folder) if f.endswith(".mat")]

        # 处理每个电池文件
        for filename in mat_files:
            # 提取电池编号
            battery_id = filename.split(".mat")[0]

            print(f"[NASA] Processing {filename}...")

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
            print(f"[NASA][{battery_id}] Total cycle entries: {num_cycles}")

            # 收集所有周期的数据（包括充电和放电）
            cycle_data_list = []

            # 遍历所有周期
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
                            else:  # 放电
                                current_load = data[0][0]['Current_load'][0][j]
                                voltage_load = data[0][0]['Voltage_load'][0][j]
                            step_time = data[0][0]['Time'][0][j]

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

            print(f"[NASA][{battery_id}] Parsed usable charge/discharge cycles: {len(cycle_data_list)}")

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
            print(
                f"[NASA][{battery_id}] Split cycles -> train: {split_index}, "
                f"test: {total_cycles - split_index}"
            )

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

                # 对训练数据中的容量进行插值处理（仅充电周期）
                train_interpolated_capacities = interpolate_capacity_to_timesteps(
                    np.array(train_cycle_capacities), train_cycle_lengths, all_cycle_types[:split_index])
                train_data_combined[:, 0] = train_interpolated_capacities  # 更新容量列

                # 检查插值后的容量值
                negative_interp_capacities = np.sum(train_data_combined[:, 0] <= 0)
                if negative_interp_capacities > 0:
                    print(f"警告: 训练数据插值后发现 {negative_interp_capacities} 个非正值容量")

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
            else:
                train_data_combined = np.array([])
                train_labels_combined = None

            if test_data_full:
                test_data_combined = np.vstack(test_data_full)
                test_labels_combined = None

                # 对测试数据中的容量进行插值处理（仅充电周期）
                test_interpolated_capacities = interpolate_capacity_to_timesteps(
                    np.array(test_cycle_capacities), test_cycle_lengths, all_cycle_types[split_index:])
                test_data_combined[:, 0] = test_interpolated_capacities  # 更新容量列

                # 检查插值后的容量值
                negative_interp_capacities = np.sum(test_data_combined[:, 0] <= 0)
                if negative_interp_capacities > 0:
                    print(f"警告: 测试数据插值后发现 {negative_interp_capacities} 个非正值容量")

                # 添加周期编号作为新的一列特征（先添加到最后）
                cycle_number_column = np.zeros(len(test_data_combined), dtype=np.float32)
                step_idx = 0
                for cycle_num, cycle_length in zip(test_cycle_numbers, test_cycle_lengths):
                    cycle_number_column[step_idx:step_idx+cycle_length] = cycle_num
                    step_idx += cycle_length  # 修复错误：应该使用 cycle_length 而不是 length

                # 将周期编号列添加到数据中（现在在最后）
                test_data_combined = np.column_stack([test_data_combined, cycle_number_column])

                # 调整列顺序：将周期编号移到第一列，容量移到最后一列
                # 当前列顺序：[Capacity, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Cycle_Number]
                # 目标顺序：[Cycle_Number, Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Capacity]
                cols_order = [6, 1, 2, 3, 4, 5, 0]  # 新的列索引顺序
                test_data_combined = test_data_combined[:, cols_order]
            else:
                test_data_combined = np.array([])
                test_labels_combined = None

            # 应用谱残差清洗（如果启用）
            if apply_sr_cleaning and len(train_data_combined) > 0:
                print(f"[NASA][{battery_id}] Applying spectral residual cleaning to train data...")
                train_data_combined = apply_spectral_residual_cleaning(train_data_combined, threshold=3.0)
                print(f"[NASA][{battery_id}] Cleaning completed. Shape: {train_data_combined.shape}")

            print(
                f"[NASA][{battery_id}] Final arrays -> train: {train_data_combined.shape}, "
                f"test: {test_data_combined.shape}, labels: None"
            )

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

            print(f"[NASA][{battery_id}] Saved processed files to {output_folder}")

    elif dataset == "CALCE":
        # 处理CALCE数据集
        dataset_folder = str(resolve_dataset_root("CALCE", "CALCE") / "Dataset1")
        output_folder = str(processed_dataset_path("CALCE"))
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
                        # 对于单特征时间序列，将其重塑为 (-1, 1)
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
        dataset_folder = str(resolve_dataset_root("CALCE", "CALCE") / "Dataset2")
        output_folder = str(processed_dataset_path("CALCE"))
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


