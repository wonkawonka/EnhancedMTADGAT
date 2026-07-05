"""提供谱残差异常评分工具。"""


import numpy as np

from scipy import signal

import warnings


def average_filter(values, n):

    """
    平均滤波器。
    :param values: 输入数值列表
    :param n: 窗口大小
    :return: 滤波结果
    """

    if n >= len(values):

        n = len(values) - 1


    # 构造平均滤波核
    filter_kernel = np.ones(n) / n

    # 使用卷积执行平均滤波
    filtered_values = np.convolve(values, filter_kernel, mode='same')


    return filtered_values


def spectral_residual_transform(values):

    """
    将时间序列转换为谱残差序列。
    :param values: 时间序列数值列表
    :return: 谱残差数值列表
    """

    # 进行快速傅里叶变换
    trans = np.fft.fft(values)

    # 计算幅度谱及其对数表示
    mag = np.sqrt(trans.real**2 + trans.imag**2)
    eps = 1e-8
    mag_log = np.log(mag + eps)

    # 平滑对数幅度谱
    mag_log_mean = average_filter(mag_log, n=3)

    # 计算谱残差
    spectral_residual = mag_log - mag_log_mean

    # 根据谱残差重构频域信号
    trans_real = np.exp(spectral_residual) * np.cos(np.angle(trans))
    trans_imag = np.exp(spectral_residual) * np.sin(np.angle(trans))
    trans = np.complex128(trans_real + 1j * trans_imag)  # 兼容 NumPy 2.0 复数类型弃用
    saliency_map = np.fft.ifft(trans).real


    return saliency_map


def detect_anomalies(values, threshold=3.0):

    """
    使用谱残差方法检测时间序列中的异常点。
    :param values: 时间序列数值列表
    :param threshold: 异常检测阈值，默认 3.0
    :return: 异常索引列表和异常分数
    """

    # 计算谱残差

    saliency_map = spectral_residual_transform(values)


    # 标准化显著性图
    saliency_map_normalized = (saliency_map - np.mean(saliency_map)) / (np.std(saliency_map) + 1e-8)

    # 找出超过阈值的异常点
    anomaly_indices = np.where(np.abs(saliency_map_normalized) > threshold)[0]
    anomaly_scores = np.abs(saliency_map_normalized)


    return anomaly_indices, anomaly_scores


def replace_anomalies_with_neighbors(values, anomaly_indices):

    """
    使用相邻正常值替换异常值。
    :param values: 原始时间序列
    :param anomaly_indices: 异常索引列表
    :return: 修复后的时间序列
    """

    repaired_values = np.copy(values)


    for idx in anomaly_indices:
        # 查找异常点左右两侧的正常邻居
        left_idx = idx - 1
        right_idx = idx + 1

        # 向左查找正常点
        while left_idx >= 0 and left_idx in anomaly_indices:
            left_idx -= 1

        # 向右查找正常点
        while right_idx < len(values) and right_idx in anomaly_indices:
            right_idx += 1

        # 左右邻居都存在时取平均值替换
        if left_idx >= 0 and right_idx < len(values):
            repaired_values[idx] = (values[left_idx] + values[right_idx]) / 2
        # 只有左邻居时使用左邻居替换
        elif left_idx >= 0:
            repaired_values[idx] = values[left_idx]
        # 只有右邻居时使用右邻居替换
        elif right_idx < len(values):
            repaired_values[idx] = values[right_idx]
        # 两侧都没有正常邻居时保留原值

    return repaired_values


def apply_spectral_residual_cleaning(data, threshold=3.0):

    """
    对时间序列数据应用谱残差异常检测和清洗。
    :param data: 形状为（时间步，特征）的二维数组
    :param threshold: 异常检测阈值
    :return: 清洗后的时间序列数据
    """

    cleaned_data = np.copy(data)


    # 对每个特征列单独执行异常检测和清洗
    for feature_idx in range(data.shape[1]):
        feature_series = data[:, feature_idx]

        # 检测异常点
        anomaly_indices, _ = detect_anomalies(feature_series, threshold)

        # 存在异常点时执行替换
        if len(anomaly_indices) > 0:
            # 输出检测到的异常样例
            print(f"  Feature {feature_idx}: Detected {len(anomaly_indices)} anomaly points")
            # 限制打印的异常样例数量
            sample_count = min(3, len(anomaly_indices))
            for i in range(sample_count):
                idx = anomaly_indices[i]
                print(f"    Anomaly at {idx}: original={feature_series[idx]:.6f}")

            cleaned_feature = replace_anomalies_with_neighbors(feature_series, anomaly_indices)
            cleaned_data[:, feature_idx] = cleaned_feature

            # 输出替换后的异常样例
            print(f"  Feature {feature_idx}: Anomaly replacement completed")
            for i in range(sample_count):
                idx = anomaly_indices[i]
                print(f"    Anomaly at {idx}: original={feature_series[idx]:.6f} -> replaced={cleaned_feature[idx]:.6f}")
        else:
            print(f"  Feature {feature_idx}: No anomaly points detected")


    return cleaned_data


