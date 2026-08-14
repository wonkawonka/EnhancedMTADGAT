"""执行数据探索分析、特征重要性分析和数据集报告导出。"""


from __future__ import annotations


import importlib.util

import json

from dataclasses import dataclass

from pathlib import Path


import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.model_selection import train_test_split


from src.data.ch_battery_utils import (

    CH_BATTERY_DATASET_NAME,

    get_ch_battery_lfp_discharge_data,

)

from src.data.utils import (

    ensure_sequence_list,

    flatten_label_container,

    get_bms_cluster_data,

    get_bms_feature_names,

    get_data,

    get_nasa_random_battery_data,

    is_sequence_container,

)

from src.project_paths import resolve_dataset_root


NASA_RANDOM_FEATURE_NAMES = [

    "voltage",

    "current",

    "temperature",

]


DEFAULT_IMPORTANCE_SAMPLE_SIZE = 20000

DEFAULT_CORRELATION_SAMPLE_SIZE = 10000

DEFAULT_RANDOM_STATE = 3407


@dataclass

class DatasetBundle:

    """承载分析阶段统一使用的数据视图。"""


    dataset: str

    train_array: np.ndarray

    test_array: np.ndarray

    point_labels: np.ndarray | None

    feature_names: list[str]

    train_entity_rows: list[dict]

    test_entity_rows: list[dict]

    extra_metadata: dict


def _ensure_output_dir(output_dir: Path) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def _to_float(value):

    if value is None:

        return None

    return float(value)


def _infer_feature_names(dataset: str, n_features: int, extra_feature_names: list[str] | None = None) -> list[str]:

    if extra_feature_names:

        return [str(item) for item in extra_feature_names]

    if dataset == "BMS":

        return get_bms_feature_names()

    if dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}:

        return NASA_RANDOM_FEATURE_NAMES[:n_features]

    return [f"f{i}" for i in range(n_features)]


def _flatten_label_sequences(label_value, seq_lengths: list[int]) -> list[np.ndarray | None]:

    if label_value is None:

        return [None for _ in seq_lengths]


    if is_sequence_container(label_value):

        label_parts = [np.asarray(item).reshape(-1) for item in ensure_sequence_list(label_value)]

        if len(label_parts) == len(seq_lengths):

            return label_parts

        flattened = flatten_label_container(label_value)

        if flattened is None:

            return [None for _ in seq_lengths]

        label_value = flattened


    label_array = np.asarray(label_value).reshape(-1)

    if len(seq_lengths) == 1:

        return [label_array]


    total_length = int(sum(seq_lengths))

    if label_array.size != total_length:

        return [None for _ in seq_lengths]


    offset = 0

    label_parts = []

    for seq_len in seq_lengths:

        label_parts.append(label_array[offset : offset + seq_len])

        offset += seq_len

    return label_parts


def _flatten_named_collection(

    named_collection: dict,

    label_mapping: dict | None = None,

    scalar_label_mapping: dict | None = None,

) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:

    flat_arrays = []

    flat_labels = []

    entity_rows = []

    any_real_labels = False


    for entity_name, entity_value in named_collection.items():

        seq_arrays = [np.asarray(seq, dtype=np.float32) for seq in ensure_sequence_list(entity_value)]

        if not seq_arrays:

            continue


        seq_lengths = [int(len(seq)) for seq in seq_arrays]

        entity_array = np.concatenate(seq_arrays, axis=0)

        flat_arrays.append(entity_array)


        entity_labels = None

        if scalar_label_mapping is not None and entity_name in scalar_label_mapping:

            scalar_value = int(scalar_label_mapping[entity_name])

            label_parts = [np.full(seq_len, scalar_value, dtype=np.int32) for seq_len in seq_lengths]

            entity_labels = np.concatenate(label_parts, axis=0)

            any_real_labels = True

        elif label_mapping is not None:

            label_parts = _flatten_label_sequences(label_mapping.get(entity_name), seq_lengths)

            if any(part is not None for part in label_parts):

                materialized_parts = []

                for seq_len, part in zip(seq_lengths, label_parts):

                    if part is None:

                        materialized_parts.append(np.zeros(seq_len, dtype=np.int32))

                    else:

                        materialized_parts.append(np.asarray(part, dtype=np.int32).reshape(-1))

                        any_real_labels = True

                entity_labels = np.concatenate(materialized_parts, axis=0)


        if entity_labels is not None:

            flat_labels.append(entity_labels)

            positive_count = int(np.sum(entity_labels == 1))

            positive_ratio = float(np.mean(entity_labels == 1))

        else:

            positive_count = None

            positive_ratio = None


        entity_rows.append(

            {

                "entity": str(entity_name),

                "sequence_count": int(len(seq_arrays)),

                "point_count": int(entity_array.shape[0]),

                "feature_dim": int(entity_array.shape[1]),

                "positive_count": positive_count,

                "positive_ratio": positive_ratio,

            }

        )


    flat_array = np.concatenate(flat_arrays, axis=0) if flat_arrays else np.empty((0, 0), dtype=np.float32)

    flat_label_array = np.concatenate(flat_labels, axis=0) if any_real_labels and flat_labels else None

    return flat_array, flat_label_array, entity_rows


def _build_event_distribution(labels: np.ndarray | None) -> dict:

    if labels is None or labels.size == 0:

        return {

            "label_available": False,

            "positive_count": 0,

            "negative_count": 0,

            "positive_ratio": None,

            "event_count": 0,

            "event_length_mean": None,

            "event_length_max": None,

        }


    labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    event_lengths = []

    current_length = 0

    for label in labels:

        if int(label) == 1:

            current_length += 1

        elif current_length > 0:

            event_lengths.append(current_length)

            current_length = 0

    if current_length > 0:

        event_lengths.append(current_length)


    return {

        "label_available": True,

        "positive_count": int(np.sum(labels == 1)),

        "negative_count": int(np.sum(labels == 0)),

        "positive_ratio": float(np.mean(labels == 1)),

        "event_count": int(len(event_lengths)),

        "event_length_mean": _to_float(np.mean(event_lengths) if event_lengths else None),

        "event_length_max": int(max(event_lengths)) if event_lengths else None,

    }


def _summarize_feature_statistics(

    train_array: np.ndarray,

    test_array: np.ndarray,

    point_labels: np.ndarray | None,

    feature_names: list[str],

) -> pd.DataFrame:

    rows = []

    has_labels = point_labels is not None and len(np.unique(point_labels)) >= 2

    normal_mask = point_labels == 0 if has_labels else None

    anomaly_mask = point_labels == 1 if has_labels else None


    for idx, feature_name in enumerate(feature_names):

        train_col = train_array[:, idx]

        test_col = test_array[:, idx]

        row = {

            "feature": feature_name,

            "train_mean": float(np.mean(train_col)),

            "train_std": float(np.std(train_col)),

            "train_min": float(np.min(train_col)),

            "train_p01": float(np.percentile(train_col, 1)),

            "train_p50": float(np.percentile(train_col, 50)),

            "train_p99": float(np.percentile(train_col, 99)),

            "train_max": float(np.max(train_col)),

            "test_mean": float(np.mean(test_col)),

            "test_std": float(np.std(test_col)),

            "test_min": float(np.min(test_col)),

            "test_p01": float(np.percentile(test_col, 1)),

            "test_p50": float(np.percentile(test_col, 50)),

            "test_p99": float(np.percentile(test_col, 99)),

            "test_max": float(np.max(test_col)),

        }


        if has_labels:

            normal_values = test_col[normal_mask]

            anomaly_values = test_col[anomaly_mask]

            train_std = float(np.std(train_col))

            row["normal_mean"] = float(np.mean(normal_values)) if normal_values.size else None

            row["anomaly_mean"] = float(np.mean(anomaly_values)) if anomaly_values.size else None

            row["mean_shift"] = (

                None

                if row["normal_mean"] is None or row["anomaly_mean"] is None

                else float(row["anomaly_mean"] - row["normal_mean"])

            )

            row["standardized_mean_shift"] = (

                None

                if row["mean_shift"] is None

                else float(abs(row["mean_shift"]) / max(train_std, 1e-8))

            )

            try:

                auc_value = roc_auc_score(point_labels, test_col)

                row["single_feature_auc"] = float(max(auc_value, 1.0 - auc_value))

            except ValueError:

                row["single_feature_auc"] = None

        rows.append(row)


    feature_df = pd.DataFrame(rows)

    if "standardized_mean_shift" in feature_df.columns:

        feature_df = feature_df.sort_values(

            by=["standardized_mean_shift", "single_feature_auc"],

            ascending=[False, False],

            na_position="last",

        )

    return feature_df.reset_index(drop=True)


def _sample_rows(array: np.ndarray, max_rows: int, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:

    if array.shape[0] <= max_rows:

        return array, labels


    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)

    indices = np.arange(array.shape[0])

    if labels is not None and len(np.unique(labels)) >= 2:

        positive_indices = indices[labels == 1]

        negative_indices = indices[labels == 0]

        positive_keep = min(len(positive_indices), max_rows // 2)

        negative_keep = min(len(negative_indices), max_rows - positive_keep)

        sampled_indices = np.concatenate(

            [

                rng.choice(positive_indices, size=positive_keep, replace=False),

                rng.choice(negative_indices, size=negative_keep, replace=False),

            ]

        )

    else:

        sampled_indices = rng.choice(indices, size=max_rows, replace=False)


    sampled_indices = np.sort(sampled_indices)

    sampled_array = array[sampled_indices]

    sampled_labels = None if labels is None else labels[sampled_indices]

    return sampled_array, sampled_labels


def _save_correlation_outputs(test_array: np.ndarray, feature_names: list[str], output_dir: Path) -> dict:

    sampled_array, _ = _sample_rows(test_array, DEFAULT_CORRELATION_SAMPLE_SIZE)

    corr_df = pd.DataFrame(sampled_array, columns=feature_names).corr()

    corr_path = output_dir / "feature_correlation_matrix.csv"

    corr_df.to_csv(corr_path, encoding="utf-8-sig")


    corr_pairs = []

    for i in range(len(feature_names)):

        for j in range(i + 1, len(feature_names)):

            corr_value = float(corr_df.iloc[i, j])

            corr_pairs.append(

                {

                    "feature_a": feature_names[i],

                    "feature_b": feature_names[j],

                    "corr": corr_value,

                    "abs_corr": abs(corr_value),

                }

            )

    top_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    top_pairs_df.to_csv(output_dir / "feature_correlation_top_pairs.csv", index=False, encoding="utf-8-sig")


    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_title("Feature Correlation Heatmap")

    ax.set_xticks(range(len(feature_names)))

    ax.set_yticks(range(len(feature_names)))

    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)

    ax.set_yticklabels(feature_names, fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    fig.savefig(output_dir / "feature_correlation_heatmap.png", dpi=250, bbox_inches="tight")

    plt.close(fig)


    return {

        "correlation_matrix_path": str(corr_path),

        "top_pair_count": int(len(top_pairs_df)),

        "top_pairs_preview": top_pairs_df.head(10).to_dict(orient="records"),

    }


def _save_feature_importance_outputs(

    test_array: np.ndarray,

    point_labels: np.ndarray | None,

    feature_names: list[str],

    output_dir: Path,

) -> dict:

    if point_labels is None or len(np.unique(point_labels)) < 2:

        return {

            "available": False,

            "reason": "当前数据缺少可用于监督分析的二分类标签，已跳过重要性估计。",

        }


    sampled_x, sampled_y = _sample_rows(test_array, DEFAULT_IMPORTANCE_SAMPLE_SIZE, labels=point_labels)

    unique_labels, label_counts = np.unique(sampled_y, return_counts=True)

    if unique_labels.size < 2 or np.min(label_counts) < 2:

        return {

            "available": False,

            "reason": "正负样本不足，无法稳定拟合代理分类器。",

        }


    x_train, x_eval, y_train, y_eval = train_test_split(

        sampled_x,

        sampled_y,

        test_size=0.3,

        random_state=DEFAULT_RANDOM_STATE,

        stratify=sampled_y,

    )


    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=6,

        random_state=DEFAULT_RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced_subsample",

    )

    model.fit(x_train, y_train)

    y_score = model.predict_proba(x_eval)[:, 1]


    importance_df = pd.DataFrame(

        {

            "feature": feature_names,

            "model_importance": model.feature_importances_,

        }

    )

    permutation_result = permutation_importance(

        model,

        x_eval,

        y_eval,

        n_repeats=8,

        random_state=DEFAULT_RANDOM_STATE,

        scoring="roc_auc",

        n_jobs=1,

    )

    importance_df["permutation_importance_mean"] = permutation_result.importances_mean

    importance_df["permutation_importance_std"] = permutation_result.importances_std

    importance_df = importance_df.sort_values(

        by=["permutation_importance_mean", "model_importance"],

        ascending=[False, False],

    ).reset_index(drop=True)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")


    shap_summary = {

        "available": False,

        "reason": "未安装 shap，已保留 permutation importance 结果。",

    }

    if importlib.util.find_spec("shap") is not None:

        try:

            import shap


            shap_sample = x_eval[: min(len(x_eval), 512)]

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(shap_sample)

            if isinstance(shap_values, list):

                shap_matrix = np.asarray(shap_values[-1], dtype=np.float32)

            else:

                shap_array = np.asarray(shap_values, dtype=np.float32)

                shap_matrix = shap_array[..., -1] if shap_array.ndim == 3 else shap_array

            shap_importance = np.mean(np.abs(shap_matrix), axis=0)

            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_importance})

            shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

            shap_df.to_csv(output_dir / "feature_shap_importance.csv", index=False, encoding="utf-8-sig")

            shap_summary = {

                "available": True,

                "output_path": str(output_dir / "feature_shap_importance.csv"),

            }

        except Exception as exc:

            shap_summary = {

                "available": False,

                "reason": f"shap 计算失败：{exc}",

            }


    summary = {

        "available": True,

        "surrogate_auroc": float(roc_auc_score(y_eval, y_score)),

        "surrogate_auprc": float(average_precision_score(y_eval, y_score)),

        "sample_count": int(len(sampled_y)),

        "feature_importance_path": str(output_dir / "feature_importance.csv"),

        "top_features": importance_df.head(10).to_dict(orient="records"),

        "shap": shap_summary,

    }


    fig, ax = plt.subplots(figsize=(10, 6))

    top_df = importance_df.head(15).iloc[::-1]

    ax.barh(top_df["feature"], top_df["permutation_importance_mean"], color="tab:blue")

    ax.set_title("Top Feature Importance (Permutation)")

    ax.set_xlabel("Importance Mean")

    ax.set_ylabel("Feature")

    fig.tight_layout()

    fig.savefig(output_dir / "feature_importance_top15.png", dpi=250, bbox_inches="tight")

    plt.close(fig)


    return summary


def _write_markdown_report(

    bundle: DatasetBundle,

    summary: dict,

    event_summary: dict,

    importance_summary: dict,

    corr_summary: dict,

    output_dir: Path,

) -> Path:

    feature_preview = ", ".join(bundle.feature_names[: min(10, len(bundle.feature_names))])

    lines = [

        f"# {bundle.dataset} 数据分析报告",

        "",

        "## 数据概况",

        "",

        f"- 训练点数：{summary['train_point_count']}",

        f"- 测试点数：{summary['test_point_count']}",

        f"- 特征维度：{summary['feature_count']}",

        f"- 标签可用：{event_summary['label_available']}",

        f"- 特征Ԥ览：{feature_preview}",

        "",

        "## 异常分布",

        "",

        f"- 正样本点数：{event_summary['positive_count']}",

        f"- 正样本比例：{event_summary['positive_ratio']}",

        f"- 异常事件数：{event_summary['event_count']}",

        f"- 平均事件长度：{event_summary['event_length_mean']}",

        f"- 最大事件长度：{event_summary['event_length_max']}",

        "",

        "## 特征分析",

        "",

        f"- 特征统计：`{(output_dir / 'feature_statistics.csv').name}`",

        f"- 相关性矩阵：`{(output_dir / 'feature_correlation_matrix.csv').name}`",

        f"- 相关性热图：`{(output_dir / 'feature_correlation_heatmap.png').name}`",

        f"- 重要性结果可用：{importance_summary.get('available', False)}",

    ]


    if importance_summary.get("available"):

        lines.extend(

            [

                f"- 代理分类器 AUROC：{importance_summary.get('surrogate_auroc')}",

                f"- 代理分类器 AUPRC：{importance_summary.get('surrogate_auprc')}",

            ]

        )

    else:

        lines.append(f"- 重要性说明：{importance_summary.get('reason')}")


    lines.extend(

        [

            "",

            "## 主要产物",

            "",

            f"- 实体分布：`{(output_dir / 'train_entity_distribution.csv').name}` / `{(output_dir / 'test_entity_distribution.csv').name}`",

            f"- 特征有效性：`{(output_dir / 'physical_feature_effectiveness.csv').name}`",

            f"- 相关性强特征对Ԥ览数：{corr_summary.get('top_pair_count')}",

        ]

    )


    if bundle.extra_metadata.get("dataset_notes"):

        lines.extend(["", "## 备注", ""])

        for note in bundle.extra_metadata["dataset_notes"]:

            lines.append(f"- {note}")


    report_path = output_dir / "dataset_report.md"

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report_path


def _build_dataset_summary(bundle: DatasetBundle, output_dir: Path) -> dict:

    summary = {

        "dataset": bundle.dataset,

        "train_point_count": int(bundle.train_array.shape[0]),

        "test_point_count": int(bundle.test_array.shape[0]),

        "feature_count": int(bundle.train_array.shape[1]),

        "feature_names": bundle.feature_names,

        "output_dir": str(output_dir),

        "train_entity_count": int(len(bundle.train_entity_rows)),

        "test_entity_count": int(len(bundle.test_entity_rows)),

    }

    summary.update(bundle.extra_metadata)

    return summary


def _build_ch_fault_distribution(split_meta: dict, output_dir: Path) -> list[str]:

    dataset_notes = []

    test_manifest = split_meta.get("test_manifest")

    if test_manifest is not None and not test_manifest.empty:

        fault_df = (

            test_manifest.groupby(["fault_type", "sample_label"], dropna=False)

            .size()

            .reset_index(name="sample_count")

            .sort_values(["sample_label", "sample_count"], ascending=[False, False])

        )

        fault_df.to_csv(output_dir / "fault_distribution.csv", index=False, encoding="utf-8-sig")

        dataset_notes.append("CH-BATTERY 额外导出了 fault_type 级别的样本分布统计。")

    return dataset_notes


def load_dataset_bundle(dataset: str, **kwargs) -> DatasetBundle:

    dataset = str(dataset).upper()

    dataset_notes = []


    if dataset in {"MSL", "SMAP", "CALCE", "CALCE2"}:

        (train_array, _), (test_array, point_labels) = get_data(dataset, normalize=False)

        feature_names = _infer_feature_names(dataset, train_array.shape[1])

        train_entity_rows = [

            {

                "entity": dataset,

                "sequence_count": 1,

                "point_count": int(train_array.shape[0]),

                "feature_dim": int(train_array.shape[1]),

                "positive_count": None,

                "positive_ratio": None,

            }

        ]

        test_entity_rows = [

            {

                "entity": dataset,

                "sequence_count": 1,

                "point_count": int(test_array.shape[0]),

                "feature_dim": int(test_array.shape[1]),

                "positive_count": int(np.sum(point_labels == 1)) if point_labels is not None else None,

                "positive_ratio": float(np.mean(point_labels == 1)) if point_labels is not None else None,

            }

        ]

        return DatasetBundle(

            dataset=dataset,

            train_array=np.asarray(train_array, dtype=np.float32),

            test_array=np.asarray(test_array, dtype=np.float32),

            point_labels=None if point_labels is None else np.asarray(point_labels, dtype=np.int32),

            feature_names=feature_names,

            train_entity_rows=train_entity_rows,

            test_entity_rows=test_entity_rows,

            extra_metadata={"dataset_notes": dataset_notes},

        )


    if dataset == "BMS":

        (train_map, _), (test_map, label_map) = get_bms_cluster_data(normalize=False)

        train_array, _, train_entity_rows = _flatten_named_collection(train_map)

        test_array, point_labels, test_entity_rows = _flatten_named_collection(test_map, label_mapping=label_map)

        feature_names = _infer_feature_names(dataset, train_array.shape[1])

        dataset_notes.append("BMS 分析基于全部 processed cluster 聚合视图。")

        return DatasetBundle(

            dataset=dataset,

            train_array=train_array,

            test_array=test_array,

            point_labels=point_labels,

            feature_names=feature_names,

            train_entity_rows=train_entity_rows,

            test_entity_rows=test_entity_rows,

            extra_metadata={"dataset_notes": dataset_notes},

        )


    if dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}:

        (train_map, _), (test_map, label_map) = get_nasa_random_battery_data(

            dataset,

            normalize=False,

            nasa_battery_id=kwargs.get("nasa_battery_id"),

            nasa_train_batteries=kwargs.get("nasa_train_batteries"),

            nasa_test_batteries=kwargs.get("nasa_test_batteries"),

        )

        train_array, _, train_entity_rows = _flatten_named_collection(train_map)

        test_array, point_labels, test_entity_rows = _flatten_named_collection(test_map, label_mapping=label_map)

        feature_names = _infer_feature_names(dataset, train_array.shape[1])

        dataset_notes.append("NASA_RANDOM 数据集显式保留实体分布，便于工况分布量化。")

        return DatasetBundle(

            dataset=dataset,

            train_array=train_array,

            test_array=test_array,

            point_labels=point_labels,

            feature_names=feature_names,

            train_entity_rows=train_entity_rows,

            test_entity_rows=test_entity_rows,

            extra_metadata={"dataset_notes": dataset_notes},

        )


    if dataset == CH_BATTERY_DATASET_NAME:

        (train_map, _), (test_map, _), split_meta = get_ch_battery_lfp_discharge_data(

            root=kwargs.get("ch_battery_root", str(resolve_dataset_root("CH-BATTERY", "CH-BATTERY"))),

            normalize=False,

            train_ratio=float(kwargs.get("ch_battery_train_ratio", 0.8)),

            seed=int(kwargs.get("seed", DEFAULT_RANDOM_STATE)),

            preprocessed_dir=kwargs.get("ch_battery_preprocessed_dir"),

        )

        sample_label_map = {

            str(sample_id): int(meta.get("sample_label", 0))

            for sample_id, meta in split_meta["test_metadata"].items()

        }

        train_array, _, train_entity_rows = _flatten_named_collection(train_map)

        test_array, point_labels, test_entity_rows = _flatten_named_collection(

            test_map,

            scalar_label_mapping=sample_label_map,

        )

        feature_names = _infer_feature_names(

            dataset,

            train_array.shape[1],

            extra_feature_names=list(split_meta.get("feature_columns", [])),

        )

        dataset_notes.extend(

            [

                "CH-BATTERY 点级标签由样本级标签扩展得到，用于数据侧特征重要性估计。",

                "CH-BATTERY 同时保留 sample/fault_type 元数据用于补充统计。",

            ]

        )

        return DatasetBundle(

            dataset=dataset,

            train_array=train_array,

            test_array=test_array,

            point_labels=point_labels,

            feature_names=feature_names,

            train_entity_rows=train_entity_rows,

            test_entity_rows=test_entity_rows,

            extra_metadata={

                "dataset_notes": dataset_notes,

                "split_meta": split_meta,

            },

        )


    raise ValueError(f"Unsupported analysis dataset: {dataset}")


def run_dataset_analysis(dataset: str, output_dir: Path, **kwargs) -> dict:

    output_dir = _ensure_output_dir(Path(output_dir).resolve())

    bundle = load_dataset_bundle(dataset, **kwargs)


    pd.DataFrame(bundle.train_entity_rows).to_csv(

        output_dir / "train_entity_distribution.csv",

        index=False,

        encoding="utf-8-sig",

    )

    pd.DataFrame(bundle.test_entity_rows).to_csv(

        output_dir / "test_entity_distribution.csv",

        index=False,

        encoding="utf-8-sig",

    )


    feature_stats_df = _summarize_feature_statistics(

        bundle.train_array,

        bundle.test_array,

        bundle.point_labels,

        bundle.feature_names,

    )

    feature_stats_df.to_csv(output_dir / "feature_statistics.csv", index=False, encoding="utf-8-sig")


    effectiveness_columns = [

        "feature",

        "standardized_mean_shift",

        "single_feature_auc",

        "normal_mean",

        "anomaly_mean",

    ]

    existing_effectiveness_columns = [col for col in effectiveness_columns if col in feature_stats_df.columns]

    feature_stats_df[existing_effectiveness_columns].to_csv(

        output_dir / "physical_feature_effectiveness.csv",

        index=False,

        encoding="utf-8-sig",

    )


    corr_summary = _save_correlation_outputs(bundle.test_array, bundle.feature_names, output_dir)

    importance_summary = _save_feature_importance_outputs(

        bundle.test_array,

        bundle.point_labels,

        bundle.feature_names,

        output_dir,

    )

    event_summary = _build_event_distribution(bundle.point_labels)


    split_meta = bundle.extra_metadata.get("split_meta")

    if split_meta is not None:

        bundle.extra_metadata["dataset_notes"].extend(_build_ch_fault_distribution(split_meta, output_dir))


    summary = _build_dataset_summary(bundle, output_dir)

    summary["event_summary"] = event_summary

    summary["importance_summary"] = importance_summary

    summary["correlation_summary"] = corr_summary


    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)


    report_path = _write_markdown_report(

        bundle=bundle,

        summary=summary,

        event_summary=event_summary,

        importance_summary=importance_summary,

        corr_summary=corr_summary,

        output_dir=output_dir,

    )


    manifest = {

        "dataset": bundle.dataset,

        "output_dir": str(output_dir),

        "summary_path": str(summary_path),

        "report_path": str(report_path),

        "feature_statistics_path": str(output_dir / "feature_statistics.csv"),

        "correlation_matrix_path": str(output_dir / "feature_correlation_matrix.csv"),

        "importance_path": str(output_dir / "feature_importance.csv"),

    }

    manifest_path = output_dir / "analysis_manifest.json"

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


    return manifest
