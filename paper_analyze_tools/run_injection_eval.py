"""为 NASA/BMS 无标签场景构造电池经典异常注入评估，并输出定量结果。

用途：
1. 基于现有 `last_checkpoint.pt` 与 `config.txt` 重建模型；
2. 在原始测试序列中注入更贴近电池机理的经典异常场景；
3. 复用主仓 Predictor 生成统一格式结果，为论文提供补充定量证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_factory import build_model
from prediction import Predictor
from utils import (
    BMS_FEATURE_NAMES,
    flatten_sequence_collection,
    get_bms_cluster_data,
    get_nasa_random_battery_data,
    get_target_dims,
    is_sequence_container,
    normalize_data,
    normalize_sequence_container,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis" / "injection_eval"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NASA_ENTITIES = ["RW1", "RW2", "RW7", "RW8"]
BMS_ENTITIES = [f"BMS_B14_3_2_cluster{i}" for i in range(1, 7)]
NASA_FEATURE_LABELS = {
    0: "Current",
    1: "Voltage",
    2: "Temperature",
    3: "CapacityProxy",
}
VARIANT_SPECS = {
    "NASA_RANDOM_DISCHARGE": {
        "MTAD-GAT": ROOT / "kaggle离线output" / "ch3_main_results",
        "C3": ROOT / "kaggle离线output" / "ch3_main_results",
        "C4": ROOT / "kaggle离线output" / "ch4_bms_main",
    },
    "BMS": {
        "MTAD-GAT": ROOT / "kaggle离线output" / "ch3_main_results" / "bms_mtadgat_baseline",
        "C3": ROOT / "kaggle离线output" / "ch3_main_results" / "bms_c3_full",
        "C4": ROOT / "kaggle离线output" / "ch4_bms_main" / "bms_c3_physics_full",
        "C4+Hier": ROOT / "kaggle离线output" / "ch4_bms_main" / "bms_c4_physics_full",
    },
}
LEVEL_Q_DEFAULTS = {
    "NASA_RANDOM_DISCHARGE": (0.99, 0.001),
    "BMS": (0.99, 0.001),
}
FAULT_LIBRARY = {
    "NASA_RANDOM_DISCHARGE": [
        {
            "name": "soft_short_circuit",
            "zh_name": "软短路前兆",
            "description": "模拟局部微短路导致的电压突降、温升和电流脉动。",
            "window_ratio": 0.10,
            "anchor_ratio": 0.22,
            "preview_feature_index": 1,
            "preview_feature_name": "Voltage",
            "components": [
                {"feature_idx": 1, "feature_name": "Voltage", "op": "step", "scale": -3.8},
                {"feature_idx": 2, "feature_name": "Temperature", "op": "ramp", "scale": 2.2},
                {"feature_idx": 0, "feature_name": "Current", "op": "pulse", "scale": 1.4},
            ],
        },
        {
            "name": "capacity_fade",
            "zh_name": "容量衰减响应",
            "description": "模拟容量衰减引起的放电平台缩短和电压平台整体下移。",
            "window_ratio": 0.18,
            "anchor_ratio": 0.46,
            "preview_feature_index": 1,
            "preview_feature_name": "Voltage",
            "components": [
                {"feature_idx": 1, "feature_name": "Voltage", "op": "ramp", "scale": -2.6},
                {"feature_idx": 3, "feature_name": "CapacityProxy", "op": "ramp", "scale": -2.0},
            ],
        },
        {
            "name": "resistance_rise",
            "zh_name": "内阻上升",
            "description": "模拟内阻增加导致负载阶段电压跌落加深并伴随温升。",
            "window_ratio": 0.14,
            "anchor_ratio": 0.63,
            "preview_feature_index": 1,
            "preview_feature_name": "Voltage",
            "components": [
                {"feature_idx": 1, "feature_name": "Voltage", "op": "sag", "scale": -3.0},
                {"feature_idx": 2, "feature_name": "Temperature", "op": "ramp", "scale": 1.6},
            ],
        },
        {
            "name": "thermal_runaway_precursor",
            "zh_name": "热失控前兆",
            "description": "模拟热异常前兆导致温度持续上升、电压抖动加剧。",
            "window_ratio": 0.12,
            "anchor_ratio": 0.78,
            "preview_feature_index": 2,
            "preview_feature_name": "Temperature",
            "components": [
                {"feature_idx": 2, "feature_name": "Temperature", "op": "ramp", "scale": 3.4},
                {"feature_idx": 1, "feature_name": "Voltage", "op": "pulse", "scale": 1.1},
                {"feature_idx": 1, "feature_name": "Voltage", "op": "step", "scale": -1.0},
            ],
        },
    ],
    "BMS": [
        {
            "name": "soft_short_circuit",
            "zh_name": "软短路前兆",
            "description": "模拟软短路触发的电压下跌、局部发热、电流脉动以及单体离散度同步恶化。",
            "window_ratio": 0.035,
            "anchor_ratio": 0.12,
            "preview_feature_index": BMS_FEATURE_NAMES.index("SYS_Vol"),
            "preview_feature_name": "SYS_Vol",
            "components": [
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vol"), "feature_name": "SYS_Vol", "op": "step", "scale": -3.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_I"), "feature_name": "SYS_I", "op": "pulse", "scale": 1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmean"), "feature_name": "BMSnTmean", "op": "ramp", "scale": 2.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmax"), "feature_name": "BMSnTmax", "op": "ramp", "scale": 2.9},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Tmax"), "feature_name": "SYS_Tmax", "op": "ramp", "scale": 2.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmin"), "feature_name": "BMSnVmin", "op": "step", "scale": -2.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vmin"), "feature_name": "SYS_Vmin", "op": "step", "scale": -2.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_range"), "feature_name": "cell_v_range", "op": "step", "scale": 2.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_std"), "feature_name": "cell_v_std", "op": "ramp", "scale": 1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_min_dev_from_mean"), "feature_name": "cell_v_min_dev_from_mean", "op": "step", "scale": -2.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_vmin_sys_gap"), "feature_name": "hier_vmin_sys_gap", "op": "step", "scale": 1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_tmax_sys_gap"), "feature_name": "hier_tmax_sys_gap", "op": "ramp", "scale": 1.6},
            ],
        },
        {
            "name": "capacity_fade",
            "zh_name": "容量衰减响应",
            "description": "模拟容量衰减导致的 SOC/SOH 下降、电压平台下移以及可用容量相关响应变弱。",
            "window_ratio": 0.05,
            "anchor_ratio": 0.33,
            "preview_feature_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "preview_feature_name": "BMSnRSOC",
            "components": [
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnRSOC"), "feature_name": "BMSnRSOC", "op": "ramp", "scale": -2.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vol"), "feature_name": "SYS_Vol", "op": "ramp", "scale": -1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_SOH"), "feature_name": "SYS_SOH", "op": "step", "scale": -1.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnSOH"), "feature_name": "BMSnSOH", "op": "step", "scale": -1.5},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmean"), "feature_name": "BMSnVmean", "op": "ramp", "scale": -1.5},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmax"), "feature_name": "BMSnVmax", "op": "ramp", "scale": -1.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmin"), "feature_name": "BMSnVmin", "op": "ramp", "scale": -1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vmax"), "feature_name": "SYS_Vmax", "op": "ramp", "scale": -1.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vmin"), "feature_name": "SYS_Vmin", "op": "ramp", "scale": -1.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnIDMax"), "feature_name": "BMSnIDMax", "op": "step", "scale": -1.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_soh_sys_gap"), "feature_name": "hier_soh_sys_gap", "op": "step", "scale": 1.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_max_dev_from_mean"), "feature_name": "cell_v_max_dev_from_mean", "op": "ramp", "scale": 0.8},
            ],
        },
        {
            "name": "resistance_rise",
            "zh_name": "内阻上升",
            "description": "模拟内阻增大导致负载下压降更深、发热更强，并伴随最弱单体更早下探。",
            "window_ratio": 0.04,
            "anchor_ratio": 0.56,
            "preview_feature_index": BMS_FEATURE_NAMES.index("SYS_Vol"),
            "preview_feature_name": "SYS_Vol",
            "components": [
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vol"), "feature_name": "SYS_Vol", "op": "sag", "scale": -2.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmean"), "feature_name": "BMSnTmean", "op": "ramp", "scale": 1.5},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_range"), "feature_name": "cell_v_range", "op": "step", "scale": 2.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_I"), "feature_name": "SYS_I", "op": "pulse", "scale": 1.1},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmax"), "feature_name": "BMSnTmax", "op": "ramp", "scale": 1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Tmax"), "feature_name": "SYS_Tmax", "op": "ramp", "scale": 1.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmin"), "feature_name": "BMSnVmin", "op": "sag", "scale": -2.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vmin"), "feature_name": "SYS_Vmin", "op": "sag", "scale": -2.1},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_std"), "feature_name": "cell_v_std", "op": "ramp", "scale": 1.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_min_dev_from_mean"), "feature_name": "cell_v_min_dev_from_mean", "op": "sag", "scale": -2.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_vmin_sys_gap"), "feature_name": "hier_vmin_sys_gap", "op": "step", "scale": 1.7},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_tmax_sys_gap"), "feature_name": "hier_tmax_sys_gap", "op": "ramp", "scale": 1.2},
            ],
        },
        {
            "name": "thermal_runaway_precursor",
            "zh_name": "热失控前兆",
            "description": "模拟热失控前兆的持续升温、温差扩大、电压不稳和放电末端失衡加剧。",
            "window_ratio": 0.03,
            "anchor_ratio": 0.78,
            "preview_feature_index": BMS_FEATURE_NAMES.index("BMSnTmean"),
            "preview_feature_name": "BMSnTmean",
            "components": [
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmean"), "feature_name": "BMSnTmean", "op": "ramp", "scale": 3.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_range"), "feature_name": "cell_v_range", "op": "pulse", "scale": 1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vol"), "feature_name": "SYS_Vol", "op": "pulse", "scale": 1.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnTmax"), "feature_name": "BMSnTmax", "op": "ramp", "scale": 4.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Tmax"), "feature_name": "SYS_Tmax", "op": "ramp", "scale": 3.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Tmin"), "feature_name": "SYS_Tmin", "op": "ramp", "scale": 1.3},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_t_range"), "feature_name": "cell_t_range", "op": "step", "scale": 2.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_t_std"), "feature_name": "cell_t_std", "op": "ramp", "scale": 2.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnETmax"), "feature_name": "BMSnETmax", "op": "ramp", "scale": 2.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnETmean"), "feature_name": "BMSnETmean", "op": "ramp", "scale": 2.0},
                {"feature_idx": BMS_FEATURE_NAMES.index("BMSnVmin"), "feature_name": "BMSnVmin", "op": "pulse", "scale": -1.8},
                {"feature_idx": BMS_FEATURE_NAMES.index("SYS_Vmin"), "feature_name": "SYS_Vmin", "op": "pulse", "scale": -1.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_std"), "feature_name": "cell_v_std", "op": "pulse", "scale": 1.4},
                {"feature_idx": BMS_FEATURE_NAMES.index("cell_v_min_dev_from_mean"), "feature_name": "cell_v_min_dev_from_mean", "op": "pulse", "scale": -1.6},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_tmax_sys_gap"), "feature_name": "hier_tmax_sys_gap", "op": "ramp", "scale": 2.2},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_tmin_sys_gap"), "feature_name": "hier_tmin_sys_gap", "op": "ramp", "scale": 1.1},
                {"feature_idx": BMS_FEATURE_NAMES.index("hier_vmin_sys_gap"), "feature_name": "hier_vmin_sys_gap", "op": "pulse", "scale": 1.3},
            ],
        },
    ],
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_experiment_args(exp_dir):
    payload = load_json(exp_dir / "config.txt")
    return SimpleNamespace(**payload)


def load_checkpoint_model(exp_dir, args, n_features):
    target_dims = get_target_dims(args.dataset)
    out_dim = n_features if target_dims is None else (1 if isinstance(target_dims, int) else len(target_dims))
    model = build_model(args, n_features, int(args.lookback), out_dim, target_dims=target_dims)
    checkpoint = torch.load(exp_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    if getattr(args, "use_cuda", True) and torch.cuda.is_available():
        model = model.to("cuda")
    return model, target_dims


def make_predictor(exp_dir, output_dir, args, n_features, target_dims):
    default_level, default_q = LEVEL_Q_DEFAULTS.get(args.dataset, (0.99, 0.001))
    predictor = Predictor(
        model=None,
        window_size=int(args.lookback),
        n_features=n_features,
        pred_args={
            "dataset": args.dataset,
            "target_dims": target_dims,
            "scale_scores": bool(getattr(args, "scale_scores", False)),
            "q": getattr(args, "q", None) if getattr(args, "q", None) is not None else default_q,
            "level": getattr(args, "level", None) if getattr(args, "level", None) is not None else default_level,
            "dynamic_pot": bool(getattr(args, "dynamic_pot", False)),
            "use_mov_av": bool(getattr(args, "use_mov_av", False)),
            "gamma": float(getattr(args, "gamma", 1.0)),
            "score_fusion_mode": getattr(args, "score_fusion_mode", "fixed"),
            "use_event_consistency": bool(getattr(args, "use_event_consistency", False)),
            "event_low_ratio": float(getattr(args, "event_low_ratio", 0.5)),
            "event_min_length": int(getattr(args, "event_min_length", 3)),
            "use_hier_consistency": bool(getattr(args, "use_hier_consistency", False)),
            "hier_score_weight": float(getattr(args, "hier_score_weight", 0.5)),
            "reg_level": 2,
            "save_path": str(output_dir),
        },
    )
    predictor.model, _ = load_checkpoint_model(exp_dir, args, n_features)
    predictor.use_cuda = bool(getattr(args, "use_cuda", True))
    return predictor


def normalize_with_train_reference(train_raw, test_raw, normalize_enabled):
    if not normalize_enabled:
        return train_raw, test_raw

    if is_sequence_container(train_raw):
        train_concat = flatten_sequence_collection([train_raw], dtype=np.float32)
        _, scaler = normalize_data(train_concat, scaler=None)
        train_norm = normalize_sequence_container(train_raw, scaler)
        test_norm = normalize_sequence_container(test_raw, scaler)
        return train_norm, test_norm

    train_norm, scaler = normalize_data(np.asarray(train_raw, dtype=np.float32), scaler=None)
    test_norm, _ = normalize_data(np.asarray(test_raw, dtype=np.float32), scaler=scaler)
    return train_norm, test_norm


def to_tensor_data(values):
    if is_sequence_container(values):
        return [torch.from_numpy(np.asarray(item, dtype=np.float32)).float() for item in values]
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).float()


def apply_step(seq, feature_idx, start, end, magnitude):
    seq[start:end, feature_idx] += magnitude


def apply_ramp(seq, feature_idx, start, end, magnitude):
    span = max(end - start, 1)
    seq[start:end, feature_idx] += np.linspace(0.0, magnitude, span, dtype=np.float32)


def apply_pulse(seq, feature_idx, start, end, magnitude):
    span = max(end - start, 1)
    pulse = np.sin(np.linspace(0.0, 6.0 * np.pi, span, dtype=np.float32))
    seq[start:end, feature_idx] += pulse * magnitude


def apply_sag(seq, feature_idx, start, end, magnitude):
    span = max(end - start, 1)
    phase = np.linspace(-1.0, 1.0, span, dtype=np.float32)
    valley = 1.0 - np.abs(phase)
    seq[start:end, feature_idx] += valley * magnitude


def component_magnitude(seq, feature_idx, scale):
    std = float(np.std(seq[:, feature_idx]) + 1e-6)
    return np.float32(std * scale)


def apply_component(seq, component, start, end):
    feature_idx = int(component["feature_idx"])
    magnitude = component_magnitude(seq, feature_idx, float(component["scale"]))
    op = component["op"]
    if op == "step":
        apply_step(seq, feature_idx, start, end, magnitude)
    elif op == "ramp":
        apply_ramp(seq, feature_idx, start, end, magnitude)
    elif op == "pulse":
        apply_pulse(seq, feature_idx, start, end, abs(magnitude))
    elif op == "sag":
        apply_sag(seq, feature_idx, start, end, magnitude)
    else:
        raise ValueError(f"Unsupported injection op: {op}")
    return {
        "feature_idx": feature_idx,
        "feature_name": component["feature_name"],
        "op": op,
        "magnitude": float(magnitude),
    }


def make_interval(length, anchor_ratio, window_ratio):
    start = int(length * anchor_ratio)
    window = max(24, int(length * window_ratio))
    start = min(max(start, 0), max(length - window - 1, 0))
    end = min(length, start + window)
    return start, end


def build_nasa_experiments(entity):
    entity_l = entity.lower()
    return [
        ("MTAD-GAT", VARIANT_SPECS["NASA_RANDOM_DISCHARGE"]["MTAD-GAT"] / f"nasa_random_discharge_{entity_l}_mtadgat_baseline"),
        ("C3", VARIANT_SPECS["NASA_RANDOM_DISCHARGE"]["C3"] / f"nasa_random_discharge_{entity_l}_c3_full"),
        ("C4", VARIANT_SPECS["NASA_RANDOM_DISCHARGE"]["C4"] / f"nasa_random_discharge_{entity_l}_c3_physics"),
    ]


def build_bms_experiments():
    return list(VARIANT_SPECS["BMS"].items())


def build_presets():
    presets = {}
    for entity in NASA_ENTITIES:
        presets[f"nasa_{entity.lower()}"] = [
            {
                "scenario": f"nasa_{entity.lower()}",
                "dataset": "NASA_RANDOM_DISCHARGE",
                "entity": entity,
                "experiments": build_nasa_experiments(entity),
                "faults": FAULT_LIBRARY["NASA_RANDOM_DISCHARGE"],
            }
        ]
    presets["nasa_all"] = [
        {
            "scenario": f"nasa_{entity.lower()}",
            "dataset": "NASA_RANDOM_DISCHARGE",
            "entity": entity,
            "experiments": build_nasa_experiments(entity),
            "faults": FAULT_LIBRARY["NASA_RANDOM_DISCHARGE"],
        }
        for entity in NASA_ENTITIES
    ]
    for entity in BMS_ENTITIES:
        suffix = entity.split("_")[-1].lower()
        presets[f"bms_{suffix}"] = [
            {
                "scenario": f"bms_{suffix}",
                "dataset": "BMS",
                "entity": entity,
                "experiments": build_bms_experiments(),
                "faults": FAULT_LIBRARY["BMS"],
            }
        ]
    presets["bms_all"] = [
        {
            "scenario": entity.lower(),
            "dataset": "BMS",
            "entity": entity,
            "experiments": build_bms_experiments(),
            "faults": FAULT_LIBRARY["BMS"],
        }
        for entity in BMS_ENTITIES
    ]
    presets["all"] = presets["nasa_all"] + presets["bms_all"]
    presets["quick"] = [presets["nasa_rw1"][0], presets["bms_cluster1"][0]]
    return presets


def inject_nasa_sequences(sequence_list, fault_spec):
    injected_sequences = []
    injected_labels = []
    records = []

    for seq_id, raw_seq in enumerate(sequence_list):
        seq = np.asarray(raw_seq, dtype=np.float32).copy()
        labels = np.zeros(len(seq), dtype=np.int32)
        start, end = make_interval(len(seq), fault_spec["anchor_ratio"] + 0.02 * (seq_id % 3), fault_spec["window_ratio"])
        component_records = [apply_component(seq, component, start, end) for component in fault_spec["components"]]
        labels[start:end] = 1
        injected_sequences.append(seq)
        injected_labels.append(labels)
        records.append(
            {
                "fault_name": fault_spec["name"],
                "fault_zh_name": fault_spec["zh_name"],
                "segment_id": seq_id,
                "start": start,
                "end": end,
                "length": end - start,
                "components": component_records,
            }
        )

    return injected_sequences, injected_labels, records


def inject_bms_array(raw_array, fault_spec):
    seq = np.asarray(raw_array, dtype=np.float32).copy()
    labels = np.zeros(len(seq), dtype=np.int32)
    start, end = make_interval(len(seq), fault_spec["anchor_ratio"], fault_spec["window_ratio"])
    component_records = [apply_component(seq, component, start, end) for component in fault_spec["components"]]
    labels[start:end] = 1
    records = [
        {
            "fault_name": fault_spec["name"],
            "fault_zh_name": fault_spec["zh_name"],
            "start": start,
            "end": end,
            "length": end - start,
            "components": component_records,
        }
    ]
    return seq, labels, records


def load_raw_scenario_data(dataset, entity):
    if dataset == "NASA_RANDOM_DISCHARGE":
        (train_map, _), (test_map, _) = get_nasa_random_battery_data(dataset, nasa_battery_id=entity, normalize=False)
        return train_map[entity], test_map[entity]
    if dataset == "BMS":
        (train_map, _), (test_map, _) = get_bms_cluster_data(normalize=False)
        return train_map[entity], test_map[entity]
    raise ValueError(f"Unsupported dataset: {dataset}")


def flatten_labels(labels):
    if is_sequence_container(labels):
        return np.concatenate([np.asarray(item) for item in labels], axis=0)
    return np.asarray(labels)


def flatten_values(values):
    if is_sequence_container(values):
        return np.concatenate([np.asarray(item) for item in values], axis=0)
    return np.asarray(values)


def save_injection_preview(output_dir, values, labels, feature_idx, feature_name, title):
    x = np.arange(len(flatten_values(values)))
    y = flatten_values(values)[:, feature_idx]
    label_arr = flatten_labels(labels)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, color="#444444", linewidth=0.8, label=feature_name)
    if label_arr.any():
        ax.fill_between(x, y.min(), y.max(), where=label_arr > 0, color="#ffcccc", alpha=0.45, label="注入区间")
    ax.set_title(title)
    ax.set_xlabel("位置")
    ax.set_ylabel(feature_name)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_dir / "injection_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def variant_output_name(variant_name):
    return variant_name.replace("+", "_plus_")


def run_single_experiment(scenario, fault_spec, variant_name, exp_dir):
    if not exp_dir.exists():
        return {
            "Scenario": scenario["scenario"],
            "Dataset": scenario["dataset"],
            "Entity": scenario["entity"],
            "Fault": fault_spec["name"],
            "Fault_ZH": fault_spec["zh_name"],
            "Variant": variant_name,
            "Experiment": exp_dir.name,
            "Status": "missing_experiment_dir",
            "Output_Dir": "",
        }

    args = load_experiment_args(exp_dir)
    train_raw, test_raw = load_raw_scenario_data(scenario["dataset"], scenario["entity"])

    if scenario["dataset"] == "NASA_RANDOM_DISCHARGE":
        injected_test_raw, injected_labels, injection_records = inject_nasa_sequences(test_raw, fault_spec)
    else:
        injected_test_raw, injected_labels, injection_records = inject_bms_array(test_raw, fault_spec)

    train_norm, test_norm = normalize_with_train_reference(
        train_raw,
        injected_test_raw,
        normalize_enabled=bool(getattr(args, "normalize", True)),
    )
    train_tensor = to_tensor_data(train_norm)
    test_tensor = to_tensor_data(test_norm)

    output_dir = OUTPUT_ROOT / scenario["scenario"] / fault_spec["name"] / variant_output_name(variant_name)
    ensure_dir(output_dir)
    predictor = make_predictor(exp_dir, output_dir, args, n_features=flatten_values(train_norm).shape[1], target_dims=get_target_dims(args.dataset))
    predictor.predict_anomalies(train_tensor, test_tensor, true_anomalies=injected_labels, load_scores=False, save_output=True)

    save_injection_preview(
        output_dir,
        injected_test_raw,
        injected_labels,
        feature_idx=fault_spec["preview_feature_index"],
        feature_name=fault_spec["preview_feature_name"],
        title=f"{scenario['scenario']} - {fault_spec['zh_name']} - {variant_name}",
    )

    records_path = output_dir / "injection_records.json"
    with records_path.open("w", encoding="utf-8") as f:
        json.dump(injection_records, f, ensure_ascii=False, indent=2)

    summary = load_json(output_dir / "summary_metrics.json")
    return {
        "Scenario": scenario["scenario"],
        "Dataset": scenario["dataset"],
        "Entity": scenario["entity"],
        "Fault": fault_spec["name"],
        "Fault_ZH": fault_spec["zh_name"],
        "Fault_Desc": fault_spec["description"],
        "Variant": variant_name,
        "Experiment": exp_dir.name,
        "Status": "ok",
        "Epsilon_F1": summary.get("epsilon_result", {}).get("f1"),
        "POT_F1": summary.get("pot_result", {}).get("f1"),
        "BF_F1": summary.get("bf_result", {}).get("f1"),
        "Event_F1": summary.get("event_consistency_result", {}).get("event_result", {}).get("f1"),
        "Epsilon_Threshold": summary.get("epsilon_result", {}).get("threshold"),
        "Final_Positive": summary.get("event_consistency_result", {}).get("event_result", {}).get("positive_count"),
        "Output_Dir": str(output_dir),
    }


def fmt(value):
    if value is None:
        return ""
    return round(float(value), 4)


def dataframe_to_markdown(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(rows):
    report_path = OUTPUT_ROOT / "injection_eval_report.md"
    df = pd.DataFrame(rows)
    if df.empty:
        report_path.write_text("# 注入评估\n\n当前没有可用结果。\n", encoding="utf-8")
        return

    lines = [
        "# 注入异常评估",
        "",
        "说明：本报告基于现有 checkpoint 对无标签场景构造电池经典异常注入，并使用统一 Predictor 重新计算分数与 F1。",
        "当前异常库包括：软短路前兆、容量衰减响应、内阻上升、热失控前兆。",
        "",
    ]
    if "Fault" in df.columns:
        df = df[df["Fault"].notna()].copy()
    ok_df = df[df["Status"] == "ok"].copy() if "Status" in df.columns else df.copy()
    if "Status" in df.columns:
        missing_df = df[df["Status"] == "missing_experiment_dir"]
        if not missing_df.empty:
            lines.append("## 缺失实验")
            lines.append("")
            for _, row in missing_df.iterrows():
                lines.append(f"- `{row['Scenario']}/{row['Fault']}/{row['Variant']}` 缺少实验目录 `{row['Experiment']}`。")
            lines.append("")

    for (dataset, entity, fault_name), group in ok_df.groupby(["Dataset", "Entity", "Fault"], sort=False):
        fault_zh = group["Fault_ZH"].iloc[0]
        fault_desc = group["Fault_Desc"].iloc[0]
        lines.extend([f"## {dataset} - {entity} - {fault_zh}", "", f"- 注入场景：`{fault_name}`", f"- 场景含义：{fault_desc}", ""])
        table = group.copy()
        for col in ["Epsilon_F1", "POT_F1", "BF_F1", "Event_F1", "Epsilon_Threshold"]:
            table[col] = table[col].map(fmt)
        lines.append(dataframe_to_markdown(table[["Variant", "Experiment", "Epsilon_F1", "POT_F1", "BF_F1", "Event_F1", "Epsilon_Threshold"]]))
        lines.extend(["", "### 分析"])
        valid = group.copy()
        valid["BF_F1_num"] = pd.to_numeric(valid["BF_F1"], errors="coerce")
        valid = valid[valid["BF_F1_num"].notna()]
        if not valid.empty:
            best = valid.sort_values("BF_F1_num", ascending=False).iloc[0]
            lines.append(f"- 当前注入场景下，`{best['Variant']}` 的 `BF_F1` 最高，说明它对 `{fault_zh}` 的区分能力最强。")
        valid_event = group.copy()
        valid_event["Event_F1_num"] = pd.to_numeric(valid_event["Event_F1"], errors="coerce")
        valid_event = valid_event[valid_event["Event_F1_num"].notna()]
        if len(valid_event) >= 2:
            best_event = valid_event.sort_values("Event_F1_num", ascending=False).iloc[0]
            lines.append(f"- 事件级上，`{best_event['Variant']}` 最优，说明它更容易把离散高分整合为连续告警片段。")
        lines.append("- 这一结果可在论文中写作“经典电池异常场景下的补充定量验证”，与真实无标签案例分析形成互补。")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    presets = build_presets()
    parser = argparse.ArgumentParser(description="Run battery-style injection evaluation for NASA/BMS checkpoints.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="quick",
        choices=sorted(presets.keys()),
        help="Which preset scenario bundle to run.",
    )
    parser.add_argument(
        "--fault",
        type=str,
        default="all",
        help="Optional fault filter, e.g. soft_short_circuit/capacity_fade/resistance_rise/thermal_runaway_precursor.",
    )
    return parser.parse_args(), presets


def main():
    args, presets = parse_args()
    ensure_dir(OUTPUT_ROOT)
    selected_scenarios = presets[args.scenario]

    rows = []
    for scenario in selected_scenarios:
        selected_faults = scenario["faults"]
        if args.fault != "all":
            selected_faults = [fault for fault in selected_faults if fault["name"] == args.fault]
        for fault_spec in selected_faults:
            for variant_name, exp_dir in scenario["experiments"]:
                rows.append(run_single_experiment(scenario, fault_spec, variant_name, exp_dir))

    summary_path = OUTPUT_ROOT / "injection_eval_summary.csv"
    new_df = pd.DataFrame(rows)
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        if "Fault" in existing_df.columns:
            existing_df = existing_df[existing_df["Fault"].notna()].copy()
        else:
            existing_df = existing_df.iloc[0:0].copy()
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["Scenario", "Fault", "Variant"], keep="last")
    else:
        merged_df = new_df
    merged_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_report(merged_df.to_dict("records"))
    print(f"[DONE] wrote results to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
