import numpy as np
from scipy import signal
import warnings

def average_filter(values, n):
    """
    平均滤波器
    :param values: 输入值列表
    :param n: 窗口大小
    :return: 滤波后的结果
    """
    if n >= len(values):
        n = len(values) - 1
    
    # 创建滤波器
    filter_kernel = np.ones(n) / n
    
    # 应用卷积进行滤波
    filtered_values = np.convolve(values, filter_kernel, mode='same')
    
    return filtered_values

def spectral_residual_transform(values):
    """
    将时间序列转换为谱残差序列
    :param values: 时间序列值列表
    :return: 谱残差值列表
    """
    # 傅里叶变换
    trans = np.fft.fft(values)
    
    # 获取振幅谱和相位谱
    mag = np.sqrt(trans.real**2 + trans.imag**2)
    eps = 1e-8
    mag_log = np.log(mag + eps)
    
    # 计算平均对数谱
    mag_log_mean = average_filter(mag_log, n=3)
    
    # 计算谱残差
    spectral_residual = mag_log - mag_log_mean
    
    # 傅里叶反变换得到显著图
    trans_real = np.exp(spectral_residual) * np.cos(np.angle(trans))
    trans_imag = np.exp(spectral_residual) * np.sin(np.angle(trans))
    trans = np.complex128(trans_real + 1j * trans_imag)  # 修复NumPy 2.0兼容性问题
    saliency_map = np.fft.ifft(trans).real
    
    return saliency_map

def detect_anomalies(values, threshold=3.0):
    """
    使用谱残差方法检测时间序列中的异常点
    :param values: 时间序列值列表
    :param threshold: 异常检测阈值，默认为3
    :return: 异常点索引列表和异常分数
    """
    # 计算谱残差
    saliency_map = spectral_residual_transform(values)
    
    # 计算异常分数
    saliency_map_normalized = (saliency_map - np.mean(saliency_map)) / (np.std(saliency_map) + 1e-8)
    
    # 检测异常点
    anomaly_indices = np.where(np.abs(saliency_map_normalized) > threshold)[0]
    anomaly_scores = np.abs(saliency_map_normalized)
    
    return anomaly_indices, anomaly_scores

def replace_anomalies_with_neighbors(values, anomaly_indices):
    """
    使用相邻正常值替换异常值
    :param values: 原始时间序列
    :param anomaly_indices: 异常点索引
    :return: 修复后的时间序列
    """
    repaired_values = np.copy(values)
    
    for idx in anomaly_indices:
        # 查找最近的正常值进行替换
        left_idx = idx - 1
        right_idx = idx + 1
        
        # 向左查找正常值
        while left_idx >= 0 and left_idx in anomaly_indices:
            left_idx -= 1
            
        # 向右查找正常值
        while right_idx < len(values) and right_idx in anomaly_indices:
            right_idx += 1
            
        # 如果左右都有正常值，取平均值
        if left_idx >= 0 and right_idx < len(values):
            repaired_values[idx] = (values[left_idx] + values[right_idx]) / 2
        # 如果只有左边有正常值
        elif left_idx >= 0:
            repaired_values[idx] = values[left_idx]
        # 如果只有右边有正常值
        elif right_idx < len(values):
            repaired_values[idx] = values[right_idx]
        # 如果没有正常值，保持原值
            
    return repaired_values

def apply_spectral_residual_cleaning(data, threshold=3.0):
    """
    对时间序列数据应用谱残差异常检测和清洗
    :param data: 二维数组，形状为 (时间步, 特征数)
    :param threshold: 异常检测阈值
    :return: 清洗后的时间序列数据
    """
    cleaned_data = np.copy(data)
    
    # 对每个特征分别进行异常检测和清洗
    for feature_idx in range(data.shape[1]):
        feature_series = data[:, feature_idx]
        
        # 检测异常点
        anomaly_indices, _ = detect_anomalies(feature_series, threshold)
        
        # 替换异常点
        if len(anomaly_indices) > 0:
            cleaned_feature = replace_anomalies_with_neighbors(feature_series, anomaly_indices)
            cleaned_data[:, feature_idx] = cleaned_feature
            
            # 打印处理信息
            print(f"  特征 {feature_idx}: 检测到 {len(anomaly_indices)} 个异常点并已完成替换")
        else:
            print(f"  特征 {feature_idx}: 未检测到异常点")
            
    return cleaned_data