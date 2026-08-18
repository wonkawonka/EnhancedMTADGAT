"""在官方车辆级电池数据上训练和评估项目内部模型。"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from src.args import apply_dataset_defaults, get_parser
from src.data.nc_battery import (
    BatterySnippetWindowDataset,
    PaperChannelNormalizer,
    RESPONSE_DIMS,
    StreamingMinMaxScaler,
    aggregate_vehicle_scores,
    build_index,
    load_snippet,
    split_vehicle_folds,
)
from src.models.model_factory import build_model, resolve_model_args
from src.project_paths import MANUAL_RUNS_ROOT
from src.runners.train import build_trainer, maybe_resume_trainer

def _loader(dataset, args, shuffle):
    """构造 DataLoader，并集中处理 worker/persistent-worker 的兼容性开关。"""
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + (1 if shuffle else 2))
    options = {
        "batch_size": args.bs,
        "shuffle": shuffle,
        "num_workers": max(0, args.num_workers),
        "pin_memory": bool(torch.cuda.is_available()),
        "generator": generator,
    }
    if args.num_workers > 0:
        options["persistent_workers"] = bool(args.persistent_workers)
        options["prefetch_factor"] = max(1, args.prefetch_factor)
    return DataLoader(dataset, **options)


def _configure_reproducibility(seed, deterministic):
    """统一随机源；正式实验默认启用同机同软件栈可复现的 CUDA 算法。"""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def _branch_weights(pred_errors: np.ndarray, recon_errors: np.ndarray, mode: str, gamma: float):
    if mode != "quality_aware":
        return np.ones(pred_errors.shape[1]), np.full(pred_errors.shape[1], gamma)

    def stability(values):
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        return mad / (np.median(np.abs(values), axis=0) + 1e-6)

    pred_quality = 1.0 / (stability(pred_errors) + 1e-6)
    recon_quality = float(gamma) / (stability(recon_errors) + 1e-6)
    total = pred_quality + recon_quality + 1e-6
    return pred_quality / total, recon_quality / total

def _physical_window_error(x: np.ndarray, recons: np.ndarray, requested_terms: str) -> np.ndarray:
    """计算每个窗口的无量纲电学/热学响应偏差。"""
    enabled = {term.strip() for term in requested_terms.split(",") if term.strip()}
    terms = []
    if "voltage_rate" in enabled:
        terms.append(np.mean(np.abs(np.diff(x[:, :, 0], axis=1) - np.diff(recons[:, :, 0], axis=1)), axis=1))
    if "temperature_rate" in enabled:
        terms.append(np.mean(np.abs(np.diff(x[:, :, 5], axis=1) - np.diff(recons[:, :, 5], axis=1)), axis=1))
    if "charge_flow" in enabled:
        actual_flow = np.cumsum(x[:, :, 1], axis=1)
        recon_flow = np.cumsum(recons[:, :, 1], axis=1)
        actual_flow /= np.maximum(np.max(np.abs(actual_flow), axis=1, keepdims=True), 1e-6)
        recon_flow /= np.maximum(np.max(np.abs(recon_flow), axis=1, keepdims=True), 1e-6)
        terms.append(np.mean(np.abs(actual_flow - recon_flow), axis=1))
    if "voltage_spread" in enabled:
        terms.append(np.mean(np.abs((x[:, :, 3] - x[:, :, 4]) - (recons[:, :, 3] - recons[:, :, 4])), axis=1))
    if "temperature_spread" in enabled:
        terms.append(np.mean(np.abs((x[:, :, 5] - x[:, :, 6]) - (recons[:, :, 5] - recons[:, :, 6])), axis=1))
    if "soc_current_coupling" in enabled:
        actual_soc = np.diff(x[:, :, 2], axis=1)
        recon_soc = np.diff(recons[:, :, 2], axis=1)
        actual_current = x[:, 1:, 1]
        recon_current = recons[:, 1:, 1]
        actual_soc = actual_soc / np.maximum(np.mean(np.abs(actual_soc), axis=1, keepdims=True), 1e-6)
        recon_soc = recon_soc / np.maximum(np.mean(np.abs(recon_soc), axis=1, keepdims=True), 1e-6)
        actual_current = actual_current / np.maximum(np.mean(np.abs(actual_current), axis=1, keepdims=True), 1e-6)
        recon_current = recon_current / np.maximum(np.mean(np.abs(recon_current), axis=1, keepdims=True), 1e-6)
        terms.append(np.mean(np.abs((actual_soc - actual_current) - (recon_soc - recon_current)), axis=1))
    if not terms:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.mean(np.stack(terms, axis=1), axis=1)

def _condition_descriptors(
    x: np.ndarray, mileages: np.ndarray | None = None, include_slow_state: bool = False
) -> np.ndarray:
    """提取短时控制量描述，并可选加入不依赖标签的慢变化运行状态。"""
    current = x[:, :, 1]
    soc = x[:, :, 2]
    descriptors = [
        np.mean(current, axis=1),
        np.std(current, axis=1),
        current[:, -1],
        np.mean(np.abs(np.diff(current, axis=1)), axis=1),
        soc[:, 0],
        soc[:, -1],
        soc[:, -1] - soc[:, 0],
    ]
    if include_slow_state:
        if mileages is None:
            raise ValueError("C3 slow-state descriptors require snippet mileage metadata")
        mileage = np.asarray(mileages, dtype=np.float32).reshape(-1)
        # 对数里程是不依赖故障标签的老化代理量。缺失值使用正常数据中位数填充，
        # 避免人为特殊值被模型误认为一种独立工况。
        finite = np.isfinite(mileage)
        fill = np.median(mileage[finite]) if np.any(finite) else 0.0
        descriptors.append(np.log1p(np.where(finite, np.maximum(mileage, 0.0), fill)))
    return np.stack(
        descriptors,
        axis=1,
    ).astype(np.float32)

@torch.no_grad()
def _collect_errors(
    model, loader, device, physical_terms, target_dims=None, use_condition_slow_state=False,
    use_relation_change_score=False, use_relation_prototype_suppression=False,
    relation_change_mode="consecutive_js",
):
    """在一个数据划分上执行推理，保留窗口误差及其车辆/片段元数据。

    此函数不计算 AP 或阈值：它只产生预测误差、重构误差和可选物理误差。将“模型
    推理”与“阈值/评分策略”分开，才能用同一已训练模型公平比较不同评分变体。
    """
    model.eval()
    pred_parts, recon_parts, pred_signed_parts, recon_signed_parts = [], [], [], []
    physical_parts, consistency_parts, condition_parts, relation_parts = [], [], [], []
    relation_embedding_parts, relation_current_parts, relation_next_parts = [], [], []
    c3_joint_parts, c3_value_parts, c3_relation_parts = [], [], []
    cars, labels, snippets = [], [], []
    for batch in loader:
        if len(batch) == 6:
            x, y, batch_cars, batch_labels, batch_snippets, batch_mileages = batch
        else:
            x, y, batch_cars, batch_labels, batch_snippets = batch
            batch_mileages = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds, recons = model(x)
        base_model = model.module if hasattr(model, "module") else model
        if getattr(base_model, "use_c3_joint_relation", False):
            c3_components = base_model.c3_joint_components(x, y, preds)
            c3_joint_parts.append(c3_components["joint_score"].cpu().numpy())
            c3_value_parts.append(c3_components["value_residual"].cpu().numpy())
            c3_relation_parts.append(c3_components["relation_residual"].cpu().numpy())
        consistency_prediction = getattr(base_model, "_physical_consistency_prediction", None)
        if consistency_prediction is None:
            captured_consistency = None
        else:
            captured_consistency = consistency_prediction.detach().clone()
        if use_relation_change_score or use_relation_prototype_suppression:
            current_attention = getattr(base_model, "_feature_attention_weights", None)
            if current_attention is None:
                raise RuntimeError("Relation-change scoring requires Feature-GAT attention")
            current_attention = current_attention.detach().float().clamp_min(1e-7)
            shifted_x = torch.cat((x[:, 1:, :], y), dim=1)
            model(shifted_x)
            shifted_attention = getattr(base_model, "_feature_attention_weights", None)
            if shifted_attention is None:
                raise RuntimeError("Shifted relation scoring requires Feature-GAT attention")
            shifted_attention = shifted_attention.detach().float().clamp_min(1e-7)
            if relation_change_mode == "normal_transition_residual":
                relation_current_parts.append(current_attention.cpu().numpy())
                relation_next_parts.append(shifted_attention.cpu().numpy())
            else:
                mixture = 0.5 * (current_attention + shifted_attention)
                divergence = 0.5 * (
                    current_attention * (current_attention.log() - mixture.log())
                    + shifted_attention * (shifted_attention.log() - mixture.log())
                )
                relation_parts.append(divergence.mean(dim=(1, 2)).cpu().numpy())
            if use_relation_prototype_suppression:
                # 同时保留“当前关系状态”和“一步关系变化”。原型只由正常 validation
                # 窗口拟合；测试故障标签不会参与聚类、尺度或距离阈值估计。
                relation_embedding = torch.cat(
                    (current_attention, shifted_attention - current_attention), dim=2
                )
                relation_embedding_parts.append(
                    relation_embedding.flatten(start_dim=1).cpu().numpy()
                )
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        y_last = y.squeeze(1)
        if target_dims is not None:
            y_last = y_last[:, target_dims]
            x_last = x[:, -1, target_dims]
            recons_full = x.clone()
            recons_full[:, :, target_dims] = recons
        else:
            x_last = x[:, -1, :]
            recons_full = recons
        pred_signed = (y_last - preds).cpu().numpy()
        recon_signed = (x_last - recons[:, -1, :]).cpu().numpy()
        pred_error = np.abs(pred_signed)
        recon_error = np.abs(recon_signed)
        pred_parts.append(pred_error)
        recon_parts.append(recon_error)
        pred_signed_parts.append(pred_signed)
        recon_signed_parts.append(recon_signed)
        state_values = x.cpu().numpy()
        condition_parts.append(
            _condition_descriptors(
                state_values,
                None if batch_mileages is None else batch_mileages.cpu().numpy(),
                use_condition_slow_state,
            )
        )
        physical_parts.append(
            _physical_window_error(state_values, recons_full.cpu().numpy(), physical_terms)
        )
        if captured_consistency is None:
            consistency_parts.append(np.zeros(x.size(0), dtype=np.float32))
        else:
            consistency_target = x[:, :, base_model.physical_consistency_target_dims]
            consistency_parts.append(
                torch.mean(torch.abs(consistency_target - captured_consistency), dim=(1, 2))
                .cpu().numpy()
            )
        cars.extend(str(value) for value in batch_cars)
        labels.extend(int(value) for value in batch_labels)
        snippets.extend(str(value) for value in batch_snippets)
    output = {
        "pred": np.concatenate(pred_parts),
        "recon": np.concatenate(recon_parts),
        "pred_signed": np.concatenate(pred_signed_parts),
        "recon_signed": np.concatenate(recon_signed_parts),
        "physical": np.concatenate(physical_parts),
        "physical_consistency": np.concatenate(consistency_parts),
        "condition": np.concatenate(condition_parts),
        "relation_change": (
            np.concatenate(relation_parts)
            if relation_parts
            else np.zeros(len(cars), dtype=np.float32)
        ),
        "relation_attention_current": (
            np.concatenate(relation_current_parts)
            if relation_current_parts
            else np.zeros((len(cars), 0, 0), dtype=np.float32)
        ),
        "relation_attention_next": (
            np.concatenate(relation_next_parts)
            if relation_next_parts
            else np.zeros((len(cars), 0, 0), dtype=np.float32)
        ),
        "relation_embedding": (
            np.concatenate(relation_embedding_parts)
            if relation_embedding_parts
            else np.zeros((len(cars), 0), dtype=np.float32)
        ),
        "cars": cars,
        "labels": labels,
        "snippets": snippets,
    }
    if c3_joint_parts:
        output["c3_joint_score"] = np.concatenate(c3_joint_parts).astype(np.float32)
        output["c3_value_residual"] = np.concatenate(c3_value_parts).astype(np.float32)
        output["c3_relation_residual"] = np.concatenate(c3_relation_parts).astype(np.float32)
    return output


def _fit_relation_transition_model(errors):
    """Fit A[t+1] = mu + alpha * (A[t] - mu) using normal validation windows only."""
    current = np.asarray(errors["relation_attention_current"], dtype=np.float32)
    following = np.asarray(errors["relation_attention_next"], dtype=np.float32)
    if current.ndim != 3 or current.shape != following.shape or current.shape[1] == 0:
        raise ValueError("Normal-transition relation scoring requires paired Feature-GAT attention")
    mean = np.mean(current, axis=0, dtype=np.float64)
    centered = current.astype(np.float64) - mean
    target = following.astype(np.float64) - mean
    denominator = float(np.sum(centered * centered))
    alpha = 0.0 if denominator <= 1e-12 else float(np.sum(centered * target) / denominator)
    return {"mean": mean.astype(np.float32), "alpha": float(np.clip(alpha, 0.0, 1.0))}


def _relation_transition_residual(errors, transition_model):
    current = np.asarray(errors["relation_attention_current"], dtype=np.float32)
    following = np.asarray(errors["relation_attention_next"], dtype=np.float32)
    mean = np.asarray(transition_model["mean"], dtype=np.float32)
    predicted = mean[None, :, :] + float(transition_model["alpha"]) * (current - mean[None, :, :])
    return np.mean((following - predicted) ** 2, axis=(1, 2)).astype(np.float32)

def _condition_soft_assignments(features, centers, bandwidth):
    """计算样本对各个正常工况原型的连续软归属概率。"""
    squared_distance = np.sum(
        (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    logits = -squared_distance / max(2.0 * bandwidth * bandwidth, 1e-8)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)

def _fit_neural_condition_residual_model(errors, seed):
    """仅使用正常窗口拟合“给定电流/SOC 轨迹时的残差条件分布”。

    该步骤只校准已经冻结的异常检测模型。小型多层感知机不能访问响应通道或故障标签，
    只能根据控制量描述估计 MTAD 有符号残差的条件均值和对角尺度。
    """
    features = np.asarray(errors["condition"], dtype=np.float32)
    feature_center = np.median(features, axis=0)
    feature_scale = 1.4826 * np.median(np.abs(features - feature_center), axis=0)
    feature_scale = np.maximum(feature_scale, 1e-4)
    normalized = (features - feature_center) / feature_scale
    targets = np.concatenate(
        [np.asarray(errors["pred_signed"], dtype=np.float32),
         np.asarray(errors["recon_signed"], dtype=np.float32)], axis=1,
    )
    target_center = np.median(targets, axis=0)
    target_scale = 1.4826 * np.median(np.abs(targets - target_center), axis=0)
    target_scale = np.maximum(target_scale, 1e-5)
    standardized_targets = (targets - target_center) / target_scale

    # 使用局部随机数生成器保证该训练后校准模型可复现，同时不扰动主模型训练采样器的
    # 随机数轨迹。
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 173)
    net = torch.nn.Sequential(
        torch.nn.Linear(normalized.shape[1], 32), torch.nn.SiLU(),
        torch.nn.Linear(32, 32), torch.nn.SiLU(),
        torch.nn.Linear(32, standardized_targets.shape[1] * 2),
    )
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.05)
                module.bias.zero_()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(normalized)
    y_tensor = torch.from_numpy(standardized_targets)
    order = torch.randperm(len(x_tensor), generator=generator)
    batch_size = min(512, len(x_tensor))
    net.train()
    for _ in range(50):
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            output = net(x_tensor[index])
            mean, raw_scale = output.chunk(2, dim=1)
            # 限制模型学到的容差，避免样本稀少工况把所有残差都解释为正常。
            scale = torch.nn.functional.softplus(raw_scale) + 0.15
            z = (y_tensor[index] - mean) / scale
            loss = (0.5 * z.square() + torch.log(scale)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    return {
        "net": net,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "target_center": target_center,
        "target_scale": target_scale,
        "method": "neural_heteroscedastic",
        "descriptor_dim": int(normalized.shape[1]),
    }

def _fit_condition_residual_calibration(
    errors, n_clusters, seed, method="hard_kmeans", temperature=1.0,
    descriptor_names=None,
):
    """只用正常校准集拟合连续工况条件下的残差分布，防止测试集参与评分选择。"""
    if method == "neural_heteroscedastic":
        result = _fit_neural_condition_residual_model(errors, seed)
        result["descriptor_names"] = descriptor_names
        return result
    features = np.asarray(errors["condition"], dtype=np.float64)
    feature_center = np.median(features, axis=0)
    feature_scale = 1.4826 * np.median(np.abs(features - feature_center), axis=0)
    feature_scale = np.maximum(feature_scale, 1e-4)
    normalized = (features - feature_center) / feature_scale
    cluster_count = max(1, min(int(n_clusters), len(normalized)))
    estimator = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=int(seed),
        n_init=10,
        batch_size=min(2048, max(256, len(normalized))),
    ).fit(normalized)
    assignments = estimator.labels_
    if method == "soft_expert":
        center_distance = np.sqrt(
            np.sum((normalized - estimator.cluster_centers_[assignments]) ** 2, axis=1)
        )
        # 从数据估计带宽，使不同折中的软分配温度具有可比较含义。
        bandwidth = max(float(np.median(center_distance)), 0.5) * max(float(temperature), 1e-3)
        responsibilities = _condition_soft_assignments(
            normalized, estimator.cluster_centers_, bandwidth
        )
    else:
        bandwidth = None
        responsibilities = None

    profiles = {}
    for key in ("pred_signed", "recon_signed"):
        values = np.asarray(errors[key], dtype=np.float64)
        global_bias = np.median(values, axis=0)
        global_scale = 1.4826 * np.median(np.abs(values - global_bias), axis=0)
        global_scale = np.maximum(global_scale, 1e-5)
        biases, scales, counts = [], [], []
        for cluster in range(cluster_count):
            if responsibilities is not None:
                weights = responsibilities[:, cluster]
                count = float(np.sum(weights))
                # 截断只用于估计正常残差轮廓，避免验证集中少量类似故障的窗口抬高
                # 某个工况允许的正常误差范围。
                clipped = np.clip(
                    values,
                    global_bias - 6.0 * global_scale,
                    global_bias + 6.0 * global_scale,
                )
                local_bias = np.sum(weights[:, None] * clipped, axis=0) / max(count, 1e-8)
                local_scale = np.sqrt(
                    np.sum(weights[:, None] * (clipped - local_bias) ** 2, axis=0)
                    / max(count, 1e-8)
                )
            else:
                selected = values[assignments == cluster]
                count = len(selected)
                if count:
                    local_bias = np.median(selected, axis=0)
                    local_scale = 1.4826 * np.median(
                        np.abs(selected - local_bias), axis=0
                    )
                else:
                    local_bias, local_scale = global_bias, global_scale
            # 将小样本工况的统计量向全局正常分布收缩，降低方差。
            reliability = count / (count + 64.0)
            biases.append(reliability * local_bias + (1.0 - reliability) * global_bias)
            scales.append(
                np.maximum(
                    reliability * local_scale + (1.0 - reliability) * global_scale,
                    0.1 * global_scale,
                )
            )
            counts.append(count)
        profiles[key] = {
            "bias": np.stack(biases),
            "scale": np.stack(scales),
            "global_bias": global_bias,
            "global_scale": global_scale,
            "counts": counts,
        }
    return {
        "estimator": estimator,
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "profiles": profiles,
        "method": method,
        "bandwidth": bandwidth,
        "descriptor_names": descriptor_names,
    }

def _apply_condition_residual_calibration(errors, calibration):
    if calibration["method"] == "neural_heteroscedastic":
        features = (errors["condition"] - calibration["feature_center"]) / calibration[
            "feature_scale"
        ]
        with torch.no_grad():
            output = calibration["net"](torch.from_numpy(features.astype(np.float32)))
            mean, raw_scale = output.chunk(2, dim=1)
            mean = mean.numpy() * calibration["target_scale"] + calibration["target_center"]
            scale = (
                (torch.nn.functional.softplus(raw_scale) + 0.15).numpy()
                * calibration["target_scale"]
            )
        calibrated = dict(errors)
        dims = errors["pred_signed"].shape[1]
        calibrated["pred"] = np.abs(errors["pred_signed"] - mean[:, :dims]) / scale[:, :dims]
        calibrated["recon"] = np.abs(errors["recon_signed"] - mean[:, dims:]) / scale[:, dims:]
        return calibrated
    features = (errors["condition"] - calibration["feature_center"]) / calibration[
        "feature_scale"
    ]
    if calibration["method"] == "soft_expert":
        responsibilities = _condition_soft_assignments(
            features,
            calibration["estimator"].cluster_centers_,
            calibration["bandwidth"],
        )
        assignments = None
    else:
        assignments = calibration["estimator"].predict(features)
        responsibilities = None
    calibrated = dict(errors)
    for signed_key, output_key in (
        ("pred_signed", "pred"),
        ("recon_signed", "recon"),
    ):
        profile = calibration["profiles"][signed_key]
        if responsibilities is None:
            bias = profile["bias"][assignments]
            scale = profile["scale"][assignments]
        else:
            bias = responsibilities @ profile["bias"]
            # 混合分布方差同时包含各专家内部不确定性和不同专家均值之间的分歧。
            second_moment = responsibilities @ (profile["scale"] ** 2 + profile["bias"] ** 2)
            scale = np.sqrt(np.maximum(second_moment - bias ** 2, 1e-10))
        calibrated[output_key] = np.abs(errors[signed_key] - bias) / scale
    return calibrated

def _condition_calibration_metadata(calibration):
    return {
        "method": (
            "normal_only_continuous_heteroscedastic_residual_density"
            if calibration["method"] == "neural_heteroscedastic"
            else "normal_only_soft_condition_expert_residual_density"
            if calibration["method"] == "soft_expert"
            else "normal_only_operating_regime_robust_zscore"
        ),
        "descriptor_names": calibration["descriptor_names"] or [
            "current_mean", "current_std", "current_last", "current_delta_abs_mean",
            "soc_first", "soc_last", "soc_delta",
        ],
        "cluster_count": (
            None if calibration["method"] == "neural_heteroscedastic"
            else int(calibration["estimator"].n_clusters)
        ),
        "cluster_counts": (
            None if calibration["method"] == "neural_heteroscedastic"
            else calibration["profiles"]["pred_signed"]["counts"]
        ),
        "soft_assignment_bandwidth": (
            None if calibration["method"] == "neural_heteroscedastic"
            else calibration["bandwidth"]
        ),
        "fit_labels_used": False,
        "fit_split": "normal_validation_only",
        "fit_window_count": int(calibration["fit_window_count"]),
        "artifact": calibration.get("artifact"),
    }

def _save_condition_residual_calibration(calibration, output_dir):
    """保存不依赖 sklearn/pickle 对象的 C3 校准参数，供独立推理复用。"""
    payload = {
        "method": calibration["method"],
        "descriptor_names": calibration.get("descriptor_names"),
        "feature_center": torch.as_tensor(calibration["feature_center"]),
        "feature_scale": torch.as_tensor(calibration["feature_scale"]),
    }
    if calibration["method"] == "neural_heteroscedastic":
        payload.update({
            "target_center": torch.as_tensor(calibration["target_center"]),
            "target_scale": torch.as_tensor(calibration["target_scale"]),
            "state_dict": calibration["net"].state_dict(),
            "descriptor_dim": int(calibration["descriptor_dim"]),
        })
    else:
        payload.update({
            "cluster_centers": torch.as_tensor(
                calibration["estimator"].cluster_centers_
            ),
            "bandwidth": calibration["bandwidth"],
            "profiles": {
                key: {
                    name: torch.as_tensor(value)
                    for name, value in profile.items()
                    if name != "counts"
                }
                for key, profile in calibration["profiles"].items()
            },
        })
    artifact = output_dir / "condition_calibrator.pt"
    torch.save(payload, artifact)
    calibration["artifact"] = artifact.name

def _window_scores(errors, pred_weights, recon_weights, dims):
    """将每窗口的预测/重构误差按验证集确定的分支权重合成为基础异常分数。"""
    if "c3_joint_score" in errors:
        return np.asarray(errors["c3_joint_score"], dtype=np.float32)
    combined = errors["pred"][:, dims] * pred_weights[dims]
    combined += errors["recon"][:, dims] * recon_weights[dims]
    return np.mean(combined, axis=1)

def _fit_physical_fusion(model_scores, physical_scores, max_weight):
    def robust_stats(values):
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) + 1e-6)
        stability = scale / (float(np.median(np.abs(values))) + 1e-6)
        return center, scale, stability

    model_center, model_scale, model_stability = robust_stats(model_scores)
    physical_center, physical_scale, physical_stability = robust_stats(physical_scores)
    weight = model_stability / (model_stability + physical_stability + 1e-6)
    return {
        "model_center": model_center,
        "model_scale": model_scale,
        "physical_center": physical_center,
        "physical_scale": physical_scale,
        "weight": float(np.clip(weight, 0.0, max_weight)),
    }

def _apply_physical_fusion(model_scores, physical_scores, calibration):
    aligned = calibration["model_center"] + calibration["model_scale"] * np.maximum(
        0.0,
        (physical_scores - calibration["physical_center"]) / calibration["physical_scale"],
    )
    weight = calibration["weight"]
    return (1.0 - weight) * model_scores + weight * aligned

def _aggregate(errors, window_scores, top_ratio):
    """将窗口分数按片段、再按车辆聚合，返回与车辆标签对齐的分数数组。"""
    score_map, car_map, labels = {}, {}, {}
    for snippet, car, label, score in zip(
        errors["snippets"], errors["cars"], errors["labels"], window_scores
    ):
        score_map.setdefault(snippet, []).append(float(score))
        car_map[snippet] = car
        labels[car] = int(label)
    return aggregate_vehicle_scores(score_map, car_map, labels, top_ratio)

def _labelled_f1_threshold(scores, labels):
    """用有标签校准集选择 F1 最优阈值；并列时选择更严格的较高阈值。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), unique]
    best_threshold, best_f1 = float(candidates[0]), -1.0
    for threshold in candidates:
        predictions = (scores > threshold).astype(np.int64)
        value = float(f1_score(labels, predictions, zero_division=0))
        if value > best_f1 or (value == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), value
    return best_threshold

def _paper_labelled_snippet_threshold(scores, labels, granularity=1000):
    """复现公开 notebook 基于有标签片段排序选择阈值的过程。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    best_fraction, best_n = -1.0, 1
    for n in range(1, 100):
        count = max(round(len(ranked_scores) * n / granularity), 1)
        fraction = float(np.mean(ranked_labels[:count]))
        if fraction > best_fraction:
            best_fraction, best_n = fraction, n
    count = max(round(len(ranked_scores) * best_n / granularity), 1)
    return float(ranked_scores[count - 1])

def _metrics(
    scores,
    labels,
    calibration_scores,
    calibration_labels=None,
    threshold_mode="normal_p99",
    calibration_threshold=None,
):
    """在车辆粒度计算 AP、AUROC、阈值 F1 等最终指标。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    if calibration_labels is not None:
        calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    if calibration_threshold is not None:
        threshold = float(calibration_threshold)
    elif threshold_mode == "paper_labelled_f1":
        if calibration_labels is None:
            raise ValueError("paper_labelled_f1 requires calibration labels")
        threshold = _labelled_f1_threshold(calibration_scores, calibration_labels)
    elif threshold_mode == "normal_p99":
        threshold = float(np.quantile(calibration_scores, 0.99))
    else:
        raise ValueError(f"Unsupported threshold mode: {threshold_mode}")
    predictions = (scores > threshold).astype(np.int64)
    normal_mask = labels == 0
    normal_count = int(np.sum(normal_mask))
    false_positive_count = int(np.sum(predictions[normal_mask])) if normal_count else 0
    calibrated_fpr = (
        float(false_positive_count / normal_count) if normal_count else float("nan")
    )
    fpr, tpr, _ = roc_curve(labels, scores)
    fixed_recall_fpr = {}
    for target_recall in (0.5, 0.7, 0.8, 0.9):
        eligible = fpr[tpr >= target_recall]
        fixed_recall_fpr[f"fpr_at_recall_{int(target_recall * 100)}pct"] = (
            float(np.min(eligible)) if len(eligible) else 1.0
        )
    average_precision = float(average_precision_score(labels, scores))
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    trapezoidal_pr_auc = float(auc(recall_curve, precision_curve))
    result = {
        "vehicle_auroc": float(roc_auc_score(labels, scores)),
        # 论文统一把 PR-AUC 定义为平均精确率；梯形积分结果仅作为名称明确的辅助指标保存。
        "vehicle_pr_auc": average_precision,
        "vehicle_average_precision": average_precision,
        "vehicle_auprc": average_precision,
        "vehicle_pr_auc_trapezoid": trapezoidal_pr_auc,
        "vehicle_f1_at_calibrated_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "vehicle_precision_at_calibrated_threshold": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "vehicle_recall_at_calibrated_threshold": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        # 误报指标必须使用校准集预先确定的同一阈值，不能用 ROC 曲线上的
        # TPR@指定FPR 代替。每万正常车辆误报数只是 FPR 的直观等比例表达。
        "vehicle_false_positive_rate_at_calibrated_threshold": calibrated_fpr,
        "vehicle_specificity_at_calibrated_threshold": float(1.0 - calibrated_fpr),
        "vehicle_false_positives_per_10000_at_calibrated_threshold": float(
            calibrated_fpr * 10000.0
        ),
        "vehicle_false_positive_count_at_calibrated_threshold": false_positive_count,
        "vehicle_normal_count": normal_count,
        "calibration_threshold": threshold,
        "threshold_mode": threshold_mode,
        "tpr_at_fpr_1pct": float(np.max(tpr[fpr <= 0.01], initial=0.0)),
        "tpr_at_fpr_5pct": float(np.max(tpr[fpr <= 0.05], initial=0.0)),
        **fixed_recall_fpr,
    }
    if threshold_mode == "normal_p99":
        result["vehicle_f1_at_normal_p99"] = result["vehicle_f1_at_calibrated_threshold"]
        result["normal_validation_p99_threshold"] = threshold
    else:
        result["vehicle_f1_at_labelled_calibration"] = result[
            "vehicle_f1_at_calibrated_threshold"
        ]
    return result

def _evaluate_scoring_variant(
    normal_calibration_errors,
    threshold_calibration_errors,
    test_errors,
    pred_weights,
    recon_weights,
    dims,
    top_ratio,
    *,
    physical_max_weight=None,
    condition_calibration=None,
    consistency_max_weight=None,
    relation_change_weight=None,
    relation_change_mode="consecutive_js",
    relation_prototype_suppression=None,
    threshold_mode="normal_p99",
    top_ratio_mode="fixed",
):
    """评估一种评分策略：校准权重/阈值，再在测试车辆上固定地报告指标。

    变体可仅改变参与评分的通道，也可融合物理误差、工况残差或物理一致性头。
    所有选择均基于 validation/calibration；函数返回的测试 AP/AUROC 不参与调参。
    """
    if condition_calibration is not None:
        normal_scoring_errors = _apply_condition_residual_calibration(
            normal_calibration_errors, condition_calibration
        )
        threshold_scoring_errors = _apply_condition_residual_calibration(
            threshold_calibration_errors, condition_calibration
        )
        test_scoring_errors = _apply_condition_residual_calibration(
            test_errors, condition_calibration
        )
    else:
        normal_scoring_errors = normal_calibration_errors
        threshold_scoring_errors = threshold_calibration_errors
        test_scoring_errors = test_errors
    normal_calibration_window_scores = _window_scores(
        normal_scoring_errors, pred_weights, recon_weights, dims
    )
    threshold_calibration_window_scores = _window_scores(
        threshold_scoring_errors, pred_weights, recon_weights, dims
    )
    test_window_scores = _window_scores(test_scoring_errors, pred_weights, recon_weights, dims)
    physical_calibration = None
    if physical_max_weight is not None:
        physical_calibration = _fit_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical"],
            physical_max_weight,
        )
        normal_calibration_window_scores = _apply_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical"],
            physical_calibration,
        )
        threshold_calibration_window_scores = _apply_physical_fusion(
            threshold_calibration_window_scores,
            threshold_calibration_errors["physical"],
            physical_calibration,
        )
        test_window_scores = _apply_physical_fusion(
            test_window_scores,
            test_errors["physical"],
            physical_calibration,
        )

    consistency_calibration = None
    if consistency_max_weight is not None:
        consistency_calibration = _fit_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical_consistency"],
            consistency_max_weight,
        )
        normal_calibration_window_scores = _apply_physical_fusion(
            normal_calibration_window_scores,
            normal_calibration_errors["physical_consistency"],
            consistency_calibration,
        )
        threshold_calibration_window_scores = _apply_physical_fusion(
            threshold_calibration_window_scores,
            threshold_calibration_errors["physical_consistency"],
            consistency_calibration,
        )
        test_window_scores = _apply_physical_fusion(
            test_window_scores,
            test_errors["physical_consistency"],
            consistency_calibration,
        )

    relation_calibration = None
    if relation_change_weight is not None:
        transition_model = None
        if relation_change_mode == "normal_transition_residual":
            transition_model = _fit_relation_transition_model(normal_calibration_errors)
            for errors in (
                normal_calibration_errors,
                threshold_calibration_errors,
                test_errors,
            ):
                errors["relation_change"] = _relation_transition_residual(errors, transition_model)
        normal_relation = np.asarray(normal_calibration_errors["relation_change"], dtype=np.float32)
        relation_center = float(np.median(normal_relation))
        relation_mad = float(np.median(np.abs(normal_relation - relation_center)))
        model_center = float(np.median(normal_calibration_window_scores))
        model_mad = float(np.median(np.abs(normal_calibration_window_scores - model_center)))
        relation_calibration = {
            "center": relation_center,
            "scale": max(1.4826 * relation_mad, 1e-7),
            "model_center": model_center,
            "model_scale": max(1.4826 * model_mad, 1e-7),
            "model_p95": float(np.percentile(normal_calibration_window_scores, 95)),
            "weight": float(relation_change_weight),
            "z_cap": 3.0,
            "fusion_mode": "residual_gated",
            "distance": relation_change_mode,
            "calibration_source": "normal_validation_windows",
        }
        if transition_model is not None:
            relation_calibration["transition_alpha"] = float(transition_model["alpha"])

        def fuse_relation(value_scores, relation_values):
            excess = np.maximum(
                0.0,
                (np.asarray(relation_values) - relation_calibration["center"])
                / relation_calibration["scale"],
            )
            bounded_excess = np.minimum(excess, relation_calibration["z_cap"])
            gate_logit = np.clip(
                (value_scores - relation_calibration["model_p95"])
                / relation_calibration["model_scale"],
                -20.0,
                20.0,
            )
            value_gate = 1.0 / (1.0 + np.exp(-gate_logit))
            return value_scores + (
                relation_calibration["weight"]
                * relation_calibration["model_scale"]
                * bounded_excess
                * value_gate
            )

        normal_calibration_window_scores = fuse_relation(
            normal_calibration_window_scores,
            normal_calibration_errors["relation_change"],
        )
        threshold_calibration_window_scores = fuse_relation(
            threshold_calibration_window_scores,
            threshold_calibration_errors["relation_change"],
        )
        test_window_scores = fuse_relation(
            test_window_scores,
            test_errors["relation_change"],
        )

    relation_prototype_calibration = None
    if relation_prototype_suppression is not None:
        normal_embeddings = np.asarray(
            normal_calibration_errors["relation_embedding"], dtype=np.float32
        )
        if normal_embeddings.ndim != 2 or normal_embeddings.shape[1] == 0:
            raise ValueError("Relation-prototype suppression requires relation embeddings")
        center = np.median(normal_embeddings, axis=0)
        scale = 1.4826 * np.median(np.abs(normal_embeddings - center), axis=0)
        scale = np.maximum(scale, 1e-5)
        normalized = np.clip((normal_embeddings - center) / scale, -8.0, 8.0)
        clusters = max(1, min(int(relation_prototype_suppression["clusters"]), len(normalized)))
        prototype_model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=int(relation_prototype_suppression["seed"]),
            batch_size=min(256, max(16, len(normalized))),
            n_init=10,
        ).fit(normalized)

        def nearest_distance(embeddings):
            values = np.asarray(embeddings, dtype=np.float32)
            values = np.clip((values - center) / scale, -8.0, 8.0)
            squared = np.sum(
                (values[:, None, :] - prototype_model.cluster_centers_[None, :, :]) ** 2,
                axis=2,
            )
            return np.sqrt(np.min(squared, axis=1) / values.shape[1])

        normal_distance = nearest_distance(normal_embeddings)
        distance_p95 = max(float(np.percentile(normal_distance, 95)), 1e-6)
        score_p95 = float(np.percentile(normal_calibration_window_scores, 95))
        score_mad = float(np.median(np.abs(
            normal_calibration_window_scores
            - np.median(normal_calibration_window_scores)
        )))
        score_scale = max(1.4826 * score_mad, 1e-7)
        max_suppression = float(np.clip(
            relation_prototype_suppression["max_suppression"], 0.0, 0.5
        ))
        relation_prototype_calibration = {
            "clusters": clusters,
            "embedding_dim": int(normal_embeddings.shape[1]),
            "distance_p95": distance_p95,
            "score_p95": score_p95,
            "score_scale": score_scale,
            "max_suppression": max_suppression,
            "calibration_source": "normal_validation_windows_only",
            "embedding": "feature_attention_state_plus_one_step_signed_transition",
        }

        def suppress_normal_prototype(value_scores, embeddings):
            distance = nearest_distance(embeddings)
            # 只有位于正常原型覆盖范围内、且值残差已进入正常尾部时才允许抑制。
            # 距离超出正常 P95 的关系变化不减分，避免把未知关系异常吸收为工况。
            prototype_confidence = np.clip(1.0 - distance / distance_p95, 0.0, 1.0)
            gate_logit = np.clip((value_scores - score_p95) / score_scale, -20.0, 20.0)
            high_value_gate = 1.0 / (1.0 + np.exp(-gate_logit))
            factor = 1.0 - max_suppression * prototype_confidence * high_value_gate
            return value_scores * factor

        normal_calibration_window_scores = suppress_normal_prototype(
            normal_calibration_window_scores,
            normal_calibration_errors["relation_embedding"],
        )
        threshold_calibration_window_scores = suppress_normal_prototype(
            threshold_calibration_window_scores,
            threshold_calibration_errors["relation_embedding"],
        )
        test_window_scores = suppress_normal_prototype(
            test_window_scores,
            test_errors["relation_embedding"],
        )

    paper_threshold = None
    if threshold_mode.startswith("paper_"):
        paper_threshold = _paper_labelled_snippet_threshold(
            threshold_calibration_window_scores,
            threshold_calibration_errors["labels"],
        )

    aggregation_sensitivity = {}
    ratio_arrays = {}
    # 使用公开鲁棒评分 notebook 的完整 5%～95% 网格，并额外保留最高 1% 和全均值
    # 两个端点，便于分析聚合比例敏感性。
    for percent in [1, *range(5, 100, 5), 100]:
        ratio = percent / 100.0
        ratio_calibration_scores, ratio_calibration_labels, _ = _aggregate(
            threshold_calibration_errors, threshold_calibration_window_scores, ratio
        )
        ratio_scores, ratio_labels, _ = _aggregate(test_errors, test_window_scores, ratio)
        key = f"top_{percent}pct" if percent < 100 else "mean"
        aggregation_sensitivity[key] = _metrics(
            ratio_scores,
            ratio_labels,
            ratio_calibration_scores,
            ratio_calibration_labels,
            threshold_mode,
            paper_threshold,
        )
        ratio_arrays[percent] = (
            ratio_calibration_scores,
            ratio_calibration_labels,
            ratio_scores,
            ratio_labels,
        )

    if threshold_mode.startswith("paper_") and top_ratio_mode == "labelled_calibration":
        # 按公开五折协议，只在有标签校准折上选择鲁棒 Top-p，不使用测试标签。
        # 公开实现遇到并列时会更新，因此这里也选择较大的 p。
        candidate_percents = list(range(5, 100, 5))
        selected_percent = max(
            candidate_percents,
            key=lambda percent: (
                roc_auc_score(ratio_arrays[percent][1], ratio_arrays[percent][0]),
                percent,
            ),
        )
        selection_source = "labelled_calibration_vehicle_auroc"
    else:
        selected_percent = int(round(top_ratio * 100))
        if selected_percent not in ratio_arrays:
            raise ValueError(
                "battery_vehicle_top_ratio must be one of the reported sensitivity-grid ratios"
            )
        selection_source = "predefined_fixed_top_ratio"

    selected_ratio = selected_percent / 100.0
    _, _, normal_calibration_cars = _aggregate(
        normal_calibration_errors, normal_calibration_window_scores, selected_ratio
    )
    calibration_scores, calibration_labels, scores, labels = ratio_arrays[selected_percent]
    _, _, calibration_cars = _aggregate(
        threshold_calibration_errors, threshold_calibration_window_scores, selected_ratio
    )
    _, _, cars = _aggregate(test_errors, test_window_scores, selected_ratio)
    # 与论文式带标签 Top-p/阈值结果并列保存一个完全正常校准的部署口径：
    # 固定 Top-5% 聚合，阈值只取正常 validation 车辆分数 P99。这样 FPR 的降低
    # 不会来自校准折标签选 p 或选阈值，并可与 Recall/F1 联合判断是否只是少报。
    fixed_ratio = float(top_ratio)
    normal_p99_calibration_scores, _, _ = _aggregate(
        normal_calibration_errors, normal_calibration_window_scores, fixed_ratio
    )
    normal_p99_scores, normal_p99_labels, _ = _aggregate(
        test_errors, test_window_scores, fixed_ratio
    )
    normal_p99_fixed_top5_metrics = _metrics(
        normal_p99_scores,
        normal_p99_labels,
        normal_p99_calibration_scores,
        threshold_mode="normal_p99",
    )

    return {
        "score_dims": np.asarray(dims).tolist(),
        "physical_fusion": physical_calibration,
        "condition_residual_calibration": (
            None
            if condition_calibration is None
            else _condition_calibration_metadata(condition_calibration)
        ),
        "physical_consistency_fusion": consistency_calibration,
        "relation_change_fusion": relation_calibration,
        "relation_prototype_suppression": relation_prototype_calibration,
        "metrics": _metrics(
            scores,
            labels,
            calibration_scores,
            calibration_labels,
            threshold_mode,
            paper_threshold,
        ),
        "normal_p99_fixed_top5_metrics": normal_p99_fixed_top5_metrics,
        "aggregation_sensitivity": aggregation_sensitivity,
        "selected_top_ratio": selected_ratio,
        "top_ratio_selection": selection_source,
        "calibration_auroc_at_selected_top_ratio": float(
            roc_auc_score(calibration_labels, calibration_scores)
        ),
        "normal_calibration_vehicle_count": len(normal_calibration_cars),
        "threshold_calibration_vehicle_count": len(calibration_cars),
        "test_vehicle_count": len(cars),
    }, scores, labels, cars, test_window_scores

def _natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )

def _rank_correlation(left, right):
    if len(left) < 2:
        return None

    def average_ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])

def _write_voltage_spread_case(
    output_dir,
    test_records,
    test_errors,
    test_window_scores,
    vehicle_scores,
    vehicle_labels,
    vehicle_cars,
):
    """保存一个故障车辆的分数与单体电压离散度案例，用于物理解释图。"""
    faulty = [
        (float(score), str(car))
        for score, label, car in zip(vehicle_scores, vehicle_labels, vehicle_cars)
        if int(label) == 1
    ]
    if not faulty:
        return {"generated": False, "reason": "no_faulty_vehicle_in_test_split"}
    _, selected_car = max(faulty)

    snippet_scores = {}
    for snippet, car, score in zip(
        test_errors["snippets"], test_errors["cars"], test_window_scores
    ):
        if str(car) == selected_car:
            snippet_scores.setdefault(str(snippet), []).append(float(score))

    rows = []
    for record in test_records:
        if record.car != selected_car:
            continue
        path = Path(record.path)
        snippet = f"{path.parent.name}/{path.stem}"
        if snippet not in snippet_scores:
            continue
        values, metadata = load_snippet(record.path)
        timestamp = metadata.get("timestamp", metadata.get("time"))
        charge_segment = metadata.get("charge_segment", record.charge_segment)
        rows.append({
            "snippet": snippet,
            "timestamp": "" if timestamp is None else str(timestamp),
            "charge_segment": "" if charge_segment is None else str(charge_segment),
            "mileage": "" if record.mileage is None else float(record.mileage),
            "anomaly_score": float(np.mean(snippet_scores[snippet])),
            "mean_vmax_minus_vmin": float(np.mean(values[:, 3] - values[:, 4])),
            "max_vmax_minus_vmin": float(np.max(values[:, 3] - values[:, 4])),
        })
    if not rows:
        return {"generated": False, "reason": "selected_vehicle_has_no_scored_snippets"}

    if all(row["timestamp"] for row in rows):
        order_field = "timestamp"
    elif all(row["charge_segment"] for row in rows):
        order_field = "charge_segment"
    elif all(row["mileage"] != "" for row in rows):
        order_field = "mileage"
    else:
        order_field = "snippet"
    rows.sort(key=lambda row: _natural_key(row[order_field]))
    for index, row in enumerate(rows):
        row["charging_order"] = index
        row["order_source"] = order_field

    csv_path = output_dir / "voltage_spread_case.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.arange(len(rows))
    anomaly_scores = np.asarray([row["anomaly_score"] for row in rows])
    voltage_spreads = np.asarray([row["mean_vmax_minus_vmin"] for row in rows])
    fig, left_axis = plt.subplots(figsize=(12, 4.8))
    right_axis = left_axis.twinx()
    left_axis.plot(order, anomaly_scores, color="tab:red", linewidth=1.1, label="Anomaly score")
    right_axis.plot(
        order, voltage_spreads, color="tab:blue", linewidth=1.1,
        label="Mean Vmax-Vmin",
    )
    left_axis.set_xlabel(f"Charging snippet order ({order_field})")
    left_axis.set_ylabel("Anomaly score", color="tab:red")
    right_axis.set_ylabel("Mean Vmax-Vmin (raw released units)", color="tab:blue")
    left_axis.set_title(
        f"Faulty vehicle {selected_car}: score and cell-voltage spread\n"
        "Interpretation motivated by Zhang et al. (2023), Supplementary Fig. 3"
    )
    left_axis.grid(alpha=0.2)
    fig.tight_layout()
    figure_path = output_dir / "voltage_spread_case.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "generated": True,
        "vehicle": selected_car,
        "selection": "highest_scoring_faulty_vehicle_in_test_split",
        "snippet_count": len(rows),
        "order_source": order_field,
        "score_spread_rank_correlation": _rank_correlation(anomaly_scores, voltage_spreads),
        "csv": csv_path.name,
        "figure": figure_path.name,
        "evidence_source": (
            "Zhang et al., Realistic Fault Detection of Li-ion Battery via "
            "Dynamical Deep Learning, Supplementary Figure 3"
        ),
        "interpretation_limit": "descriptive case analysis; not a training target or causal test",
    }

def main():
    """执行一次“品牌 × 折 × 模型 × 随机种子”的完整清华 EV 实验。

    本函数的边界很重要：训练数据、归一化器和早停验证均来自训练侧；带故障标签的
    测试车辆只在训练完成后用于最终车辆级 AP/AUROC。批量实验的多种子展开与子进程
    管理不在这里，而在 ``compare_experiments.py``。
    """
    args = apply_dataset_defaults(get_parser().parse_args())
    args.dataset = "TSINGHUA_EV"
    resolve_model_args(args)
    _configure_reproducibility(args.seed, args.deterministic)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    # 阶段 1：定位原始 pkl，建立“片段 → 车辆标签/元信息”的轻量索引。
    records = build_index(
        args.tsinghua_ev_root,
        args.battery_brand,
        max_snippets=args.battery_max_index_snippets,
    )
    # 阶段 2：严格按车辆切分，避免同一车辆的不同充电片段泄漏到训练和测试两侧。
    splits = split_vehicle_folds(
        records,
        args.battery_fold,
        seed=args.seed if args.battery_fold_seed < 0 else args.battery_fold_seed,
        protocol=args.battery_split_protocol,
    )
    if args.battery_max_snippets_per_vehicle > 0:
        capped = {}
        for split_name, split_records in splits.items():
            counts = {}
            capped[split_name] = []
            for record in split_records:
                if counts.get(record.car, 0) >= args.battery_max_snippets_per_vehicle:
                    continue
                capped[split_name].append(record)
                counts[record.car] = counts.get(record.car, 0) + 1
        splits = capped
    # 阶段 3：归一化器只能在训练折拟合；物理分支另外保存原始单位下的 offset/scale。
    if not args.normalize:
        scaler = None
    elif args.battery_normalization == "paper_channel":
        scaler = PaperChannelNormalizer(splits["train"])
    else:
        scaler = StreamingMinMaxScaler().fit_records(splits["train"])
    if scaler is not None:
        # 物理特征必须在原始工程单位下形成；若直接使用各通道独立归一化后的值，
        # 电压或温度离散度就失去可比较的物理意义。
        args.physical_data_min = scaler.offset_.astype(np.float32).tolist()
        args.physical_data_scale = scaler.scale_.astype(np.float32).tolist()
    # 阶段 4：把不等长 pkl 懒加载为固定长度的“历史窗口 → 下一点”样本。
    datasets = {
        name: BatterySnippetWindowDataset(
            split,
            args.lookback,
            scaler,
            windows_per_snippet=args.battery_windows_per_snippet,
            include_metadata=name in {"validation", "calibration", "test"},
        )
        for name, split in splits.items()
    }
    output_dir = Path(
        os.environ.get(
            "PLAN_OUTPUT_DIR",
            MANUAL_RUNS_ROOT / "TSINGHUA_EV" / (args.run_id or f"brand{args.battery_brand}_fold{args.battery_fold}"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[NC battery] brand={args.battery_brand} fold={args.battery_fold} "
        f"records={len(records)} windows(train/val/test)="
        f"{len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}"
    )

    # 阶段 5：构造训练/验证 loader，确定预测维度，实例化模型并执行早停训练。
    train_loader = _loader(datasets["train"], args, True)
    validation_train_view = BatterySnippetWindowDataset(
        splits["validation"], args.lookback, scaler,
        windows_per_snippet=args.battery_windows_per_snippet,
        include_metadata=False,
    )
    val_loader = _loader(validation_train_view, args, False)
    target_dims = list(RESPONSE_DIMS) if args.battery_response_only_training else None
    out_dim = len(target_dims) if target_dims is not None else 7
    model = build_model(args, 7, args.lookback, out_dim, target_dims=target_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = build_trainer(
        model, optimizer, args, args.lookback, 7, target_dims,
        str(output_dir), str(output_dir / "logs"), json.dumps(vars(args), ensure_ascii=False),
    )
    evaluation_checkpoint = str(args.battery_evaluation_checkpoint).strip()
    if evaluation_checkpoint:
        checkpoint_path = Path(evaluation_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
        print(f"[NC battery] scoring-only evaluation checkpoint={checkpoint_path}")
    else:
        maybe_resume_trainer(trainer, args)
        trainer.fit(train_loader, val_loader)
        checkpoint_path = output_dir / "model.pt"

    # 阶段 6：加载冻结的最佳 checkpoint；评分规则开发不重新训练或改写该权重。
    try:
        state = torch.load(checkpoint_path, map_location=trainer.device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=trainer.device)
    model.load_state_dict(state)
    # 阶段 7：分别收集验证、阈值校准与测试侧的窗口误差。此时仍未计算最终指标。
    validation_errors = _collect_errors(
        model, _loader(datasets["validation"], args, False), trainer.device,
        args.physical_response_terms, target_dims, args.use_condition_slow_state,
        args.use_relation_change_score, args.use_relation_prototype_suppression,
        args.relation_change_mode,
    )
    if splits["calibration"] is splits["validation"]:
        calibration_errors = validation_errors
    else:
        calibration_errors = _collect_errors(
            model,
            _loader(datasets["calibration"], args, False),
            trainer.device,
            args.physical_response_terms, target_dims, args.use_condition_slow_state,
            args.use_relation_change_score, args.use_relation_prototype_suppression,
            args.relation_change_mode,
        )
    test_errors = _collect_errors(
        model, _loader(datasets["test"], args, False), trainer.device,
        args.physical_response_terms, target_dims, args.use_condition_slow_state,
        args.use_relation_change_score, args.use_relation_prototype_suppression,
        args.relation_change_mode,
    )
    # 阶段 8：用验证误差决定预测分支和重构分支的融合权重。
    pred_weights, recon_weights = _branch_weights(
        validation_errors["pred"], validation_errors["recon"], args.score_fusion_mode, args.gamma
    )
    condition_calibration = None
    if args.use_condition_residual_calibration:
        condition_calibration = _fit_condition_residual_calibration(
            validation_errors,
            args.condition_calibration_clusters,
            args.seed,
            args.condition_calibration_method,
            args.condition_calibration_temperature,
            [
                "current_mean", "current_std", "current_last", "current_delta_abs_mean",
                "soc_first", "soc_last", "soc_delta",
                *( ["log1p_mileage"] if args.use_condition_slow_state else [] ),
            ],
        )
        condition_calibration["fit_window_count"] = len(validation_errors["condition"])
        _save_condition_residual_calibration(condition_calibration, output_dir)
    all_dims = np.arange(out_dim)
    response_dims = np.arange(out_dim) if target_dims is not None else np.asarray(RESPONSE_DIMS)
    # 阶段 9：定义可复核的评分变体；primary_variant 是本次实验写入主结果的策略。
    scoring_specs = {}
    if out_dim == 7:
        # 任意七通道模型都可在完全相同的权重和推理误差上，同时计算原始全通道分数
        # 和仅响应通道分数，保证评分策略比较不受重复训练影响。
        scoring_specs["all_channels"] = (all_dims, None, None, None, None, None)
        scoring_specs["response_channels"] = (response_dims, None, None, None, None, None)
        primary_variant = (
            "all_channels" if args.battery_score_channels == "all" else "response_channels"
        )
    else:
        scoring_specs["response_channels"] = (response_dims, None, None, None, None, None)
        primary_variant = "response_channels"
    if args.use_physical_response_score:
        scoring_specs["response_plus_physics"] = (
            response_dims,
            args.physical_response_max_weight,
            None,
            None,
            None,
            None,
        )
        primary_variant = "response_plus_physics"
    if condition_calibration is not None:
        scoring_specs["condition_calibrated_response"] = (
            response_dims,
            None,
            condition_calibration,
            None,
            None,
            None,
        )
        primary_variant = "condition_calibrated_response"
    if args.use_physical_consistency_head:
        consistency_name = (
            "condition_plus_physical_consistency"
            if condition_calibration is not None
            else "response_plus_physical_consistency"
        )
        scoring_specs[consistency_name] = (
            response_dims,
            None,
            condition_calibration,
            args.physical_consistency_score_max_weight,
            None,
            None,
        )
        primary_variant = consistency_name
    if args.use_relation_change_score:
        scoring_specs["response_plus_relation_change"] = (
            response_dims,
            None,
            None,
            None,
            args.relation_change_weight,
            None,
        )
        primary_variant = "response_plus_relation_change"
    if args.use_relation_prototype_suppression:
        configured_suppression = float(args.relation_prototype_max_suppression)
        for suppression in sorted({0.05, 0.10, configured_suppression}):
            suffix = int(round(suppression * 100))
            variant_name = f"response_relation_prototype_suppress_{suffix}pct"
            scoring_specs[variant_name] = (
                response_dims,
                None,
                None,
                None,
                None,
                {
                    "clusters": args.relation_prototype_clusters,
                    "max_suppression": suppression,
                    "seed": args.seed,
                },
            )
        primary_variant = (
            f"response_relation_prototype_suppress_"
            f"{int(round(configured_suppression * 100))}pct"
        )

    # 阶段 10：每个变体先在 calibration 固定阈值/融合，再仅在 test 上报告车辆级结果。
    scoring_variants = {}
    variant_arrays = {}
    for variant_name, (
        variant_dims,
        max_physical_weight,
        variant_condition_calibration,
        consistency_max_weight,
        relation_change_weight,
        relation_prototype_suppression,
    ) in scoring_specs.items():
        (
            variant_result,
            variant_scores,
            variant_labels,
            variant_cars,
            variant_window_scores,
        ) = _evaluate_scoring_variant(
            validation_errors,
            calibration_errors,
            test_errors,
            pred_weights,
            recon_weights,
            variant_dims,
            args.battery_vehicle_top_ratio,
            physical_max_weight=max_physical_weight,
            condition_calibration=variant_condition_calibration,
            consistency_max_weight=consistency_max_weight,
            relation_change_weight=relation_change_weight,
            relation_change_mode=args.relation_change_mode,
            relation_prototype_suppression=relation_prototype_suppression,
            threshold_mode=(
                "paper_labelled_snippet_rank"
                if args.battery_split_protocol == "paper_protocol"
                else "normal_p99"
            ),
            top_ratio_mode=args.battery_vehicle_top_ratio_mode,
        )
        scoring_variants[variant_name] = variant_result
        variant_arrays[variant_name] = (
            variant_scores, variant_labels, variant_cars, variant_window_scores
        )

    primary = scoring_variants[primary_variant]
    scores, labels, cars, primary_window_scores = variant_arrays[primary_variant]
    metrics = primary["metrics"]
    voltage_spread_case = _write_voltage_spread_case(
        output_dir,
        splits["test"],
        test_errors,
        primary_window_scores,
        scores,
        labels,
        cars,
    )
    # 阶段 11：保存足以复核论文表格的配置、样本量、主指标及所有备选评分变体。
    result = {
        "protocol": args.battery_split_protocol,
        "protocol_source": (
            "project strict normal-only calibration"
            if args.battery_split_protocol == "strict_normal_validation"
            else "Zhang et al. 2023 Supplementary Note 2, Five fold evaluation"
        ),
        "brand": args.battery_brand,
        "fold": args.battery_fold,
        "model_name": args.model_name,
        "primary_scoring_variant": primary_variant,
        "score_channels": args.battery_score_channels,
        "score_dims": (
            [target_dims[index] for index in primary["score_dims"]]
            if target_dims is not None
            else primary["score_dims"]
        ),
        "model_score_dims": primary["score_dims"],
        "model_output_dims": list(range(7)) if target_dims is None else target_dims,
        "response_only_training": bool(args.battery_response_only_training),
        "physical_consistency_encoder_input": (
            args.physical_consistency_encoder_input
            if args.use_physical_consistency_head else None
        ),
        "physical_consistency_encoder_bidirectional": (
            bool(args.physical_consistency_encoder_bidirectional)
            if args.use_physical_consistency_head else None
        ),
        "vehicle_top_ratio": primary["selected_top_ratio"],
        "configured_vehicle_top_ratio": args.battery_vehicle_top_ratio,
        "configured_vehicle_top_ratio_mode": args.battery_vehicle_top_ratio_mode,
        "deterministic": bool(args.deterministic),
        "evaluation_checkpoint": evaluation_checkpoint or None,
        "top_ratio_selection": primary["top_ratio_selection"],
        "calibration_auroc_at_selected_top_ratio": primary[
            "calibration_auroc_at_selected_top_ratio"
        ],
        "counts": {
            "indexed_snippets": len(records),
            "train_snippets": len(splits["train"]),
            "validation_snippets": len(splits["validation"]),
            "calibration_snippets": len(splits["calibration"]),
            "test_snippets": len(splits["test"]),
            "normal_calibration_vehicles": primary["normal_calibration_vehicle_count"],
            "threshold_calibration_vehicles": primary["threshold_calibration_vehicle_count"],
            "test_vehicles": primary["test_vehicle_count"],
        },
        "metrics": metrics,
        "aggregation_sensitivity": primary["aggregation_sensitivity"],
        "scoring_variants": scoring_variants,
        "prediction_weights": pred_weights.tolist(),
        "reconstruction_weights": recon_weights.tolist(),
        "physical_fusion": primary["physical_fusion"],
        "condition_residual_calibration": primary["condition_residual_calibration"],
        "physical_consistency_fusion": primary["physical_consistency_fusion"],
        "relation_change_fusion": primary["relation_change_fusion"],
        "relation_prototype_suppression": primary["relation_prototype_suppression"],
        "voltage_spread_case": voltage_spread_case,
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (output_dir / "vehicle_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["car", "label", "score"])
        writer.writerows(zip(cars, labels.tolist(), scores.tolist()))
    # 同一冻结模型的所有评分变体均保存车辆数组，便于事后做固定召回FPR配对审计；
    # 这些数组不参与原型拟合或超参数选择。
    for variant_name, (variant_scores, variant_labels, variant_cars, _) in variant_arrays.items():
        variant_path = output_dir / f"vehicle_scores_{variant_name}.csv"
        with variant_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["car", "label", "score"])
            writer.writerows(zip(variant_cars, variant_labels.tolist(), variant_scores.tolist()))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
