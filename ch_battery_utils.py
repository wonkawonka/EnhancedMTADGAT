"""Load CH-BATTERY LFP discharge samples and reuse saved preprocess artifacts when available."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from utils import flatten_sequence_collection, normalize_data


CH_BATTERY_DATASET_NAME = "CH_BATTERY_LFP_DISCHARGE"
CH_BATTERY_EXCLUDED_COLUMNS = {"TIME", "CHARGE_STATUS"}
CH_BATTERY_DEFAULT_TOPK_RATIO = 0.05
CH_BATTERY_DEFAULT_PREPROCESSED_DIR = "processed/lfp_discharge"
CH_BATTERY_LEGACY_PREPROCESSED_DIR = "preprocessed/lfp_discharge"
CH_BATTERY_PICKLE_PREFIX = CH_BATTERY_DATASET_NAME
CH_BATTERY_CORE_FEATURE_COLUMNS = [
    "SUM_VOLTAGE",
    "SUM_CURRENT",
    "SOC",
    "MAX_CELL_VOLT",
    "MIN_CELL_VOLT",
    "MAX_TEMP",
    "MIN_TEMP",
]


def _sanitize_token(text):
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text))
    return token.strip("_")


def _parse_cycle_kind(file_stem):
    lowered = str(file_stem).lower()
    if "_discharge_" in lowered:
        return "discharge"
    if "_charge_" in lowered:
        return "charge"
    return "unknown"


def _parse_cycle_index(file_stem):
    match = re.search(r"_(\d+)$", str(file_stem))
    return int(match.group(1)) if match else -1


def _find_fault_detail_file(root, chemistry):
    matches = sorted(root.glob(f"faults.details.-.*{chemistry}*.xlsx"))
    return matches[0] if matches else None


def _read_json(file_path):
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def _read_pickle(file_path):
    with Path(file_path).open("rb") as f:
        return pickle.load(f)


def _extract_dataset_relative_path(saved_path, dataset_root_name):
    normalized = str(saved_path).replace("\\", "/")
    marker = f"/{dataset_root_name}/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if normalized.startswith(f"{dataset_root_name}/"):
        return normalized[len(dataset_root_name) + 1 :]
    return None


def _resolve_manifest_file_path(root, saved_path=None, relative_path=None):
    root = Path(root).resolve()
    if relative_path:
        candidate = (root / str(relative_path)).resolve()
        if candidate.exists():
            return str(candidate)

    if saved_path:
        saved_candidate = Path(saved_path)
        if saved_candidate.exists():
            return str(saved_candidate.resolve())

        dataset_relative = _extract_dataset_relative_path(saved_path, root.name)
        if dataset_relative:
            candidate = (root / dataset_relative).resolve()
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(f"CH-BATTERY sample file not found: saved_path={saved_path}, relative_path={relative_path}")


def _normalize_manifest_file_paths(manifest_df, root):
    manifest_df = manifest_df.copy()
    relative_values = manifest_df["relative_path"] if "relative_path" in manifest_df.columns else [None] * len(manifest_df)
    manifest_df["file_path"] = [
        _resolve_manifest_file_path(root, saved_path=file_path, relative_path=relative_path)
        for file_path, relative_path in zip(manifest_df["file_path"], relative_values)
    ]
    return manifest_df


def _load_saved_minmax_scaler(npz_path):
    scaler_state = np.load(npz_path, allow_pickle=True)
    scaler = MinMaxScaler()
    scaler.scale_ = np.asarray(scaler_state["scale_"], dtype=np.float32)
    scaler.min_ = np.asarray(scaler_state["min_"], dtype=np.float32)
    scaler.data_min_ = np.asarray(scaler_state["data_min_"], dtype=np.float32)
    scaler.data_max_ = np.asarray(scaler_state["data_max_"], dtype=np.float32)
    scaler.data_range_ = np.asarray(scaler_state["data_range_"], dtype=np.float32)
    scaler.n_features_in_ = int(scaler.scale_.shape[0])
    feature_columns = [str(col) for col in scaler_state["feature_columns"].tolist()]
    return scaler, feature_columns


def _coerce_sample_map(sample_map):
    return {
        str(sample_id): np.asarray(sequence, dtype=np.float32)
        for sample_id, sequence in sample_map.items()
    }


def _resolve_ch_battery_core_feature_indices(feature_columns):
    feature_columns = [str(col) for col in feature_columns]
    missing_columns = [col for col in CH_BATTERY_CORE_FEATURE_COLUMNS if col not in feature_columns]
    if missing_columns:
        raise ValueError(f"Missing CH-BATTERY core feature columns: {missing_columns}")
    return [feature_columns.index(col) for col in CH_BATTERY_CORE_FEATURE_COLUMNS]


def _slice_sample_map_features(sample_map, feature_indices):
    return {
        sample_id: np.asarray(sequence[:, feature_indices], dtype=np.float32)
        for sample_id, sequence in sample_map.items()
    }


def _project_ch_battery_to_core_features(feature_columns, train_data_map, test_data_map, scaler=None):
    feature_indices = _resolve_ch_battery_core_feature_indices(feature_columns)
    projected_feature_columns = [feature_columns[idx] for idx in feature_indices]
    projected_train_data_map = _slice_sample_map_features(train_data_map, feature_indices)
    projected_test_data_map = _slice_sample_map_features(test_data_map, feature_indices)

    if scaler is not None and int(getattr(scaler, "n_features_in_", -1)) != len(projected_feature_columns):
        scaler = None

    return projected_feature_columns, projected_train_data_map, projected_test_data_map, scaler


def _load_preprocessed_pkl_bundle(preprocessed_dir, train_ratio, seed):
    preprocessed_dir = Path(preprocessed_dir).resolve()
    summary_path = preprocessed_dir / "summary.json"
    feature_columns_path = preprocessed_dir / "feature_columns.json"
    train_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_train.pkl"
    test_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_test.pkl"
    test_label_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_test_label.pkl"
    train_meta_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_train_meta.pkl"
    test_meta_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_test_meta.pkl"
    scaler_path = preprocessed_dir / f"{CH_BATTERY_PICKLE_PREFIX}_scaler.pkl"
    required_paths = [
        summary_path,
        feature_columns_path,
        train_path,
        test_path,
        test_label_path,
        train_meta_path,
        test_meta_path,
        scaler_path,
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing CH-BATTERY preprocess artifact: {required_path}")

    summary = _read_json(summary_path)
    expected_ratio = float(train_ratio)
    expected_seed = int(seed)
    if abs(float(summary.get("train_ratio", -1.0)) - expected_ratio) > 1e-12 or int(summary.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Preprocess split mismatch: expected train_ratio={expected_ratio}, seed={expected_seed}, "
            f"but got train_ratio={summary.get('train_ratio')}, seed={summary.get('seed')}"
        )

    feature_columns = [str(col) for col in _read_json(feature_columns_path)]
    train_data_map = _coerce_sample_map(_read_pickle(train_path))
    test_data_map = _coerce_sample_map(_read_pickle(test_path))
    test_label = np.asarray(_read_pickle(test_label_path), dtype=np.int32)
    train_meta_map = {str(k): v for k, v in _read_pickle(train_meta_path).items()}
    test_meta_map = {str(k): v for k, v in _read_pickle(test_meta_path).items()}
    scaler = _read_pickle(scaler_path)
    if not isinstance(scaler, MinMaxScaler):
        raise ValueError(f"Invalid CH-BATTERY scaler artifact: {scaler_path}")

    return {
        "feature_columns": feature_columns,
        "train_data_map": train_data_map,
        "test_data_map": test_data_map,
        "test_label": test_label,
        "train_meta_map": train_meta_map,
        "test_meta_map": test_meta_map,
        "scaler": scaler,
        "summary": summary,
    }


def _resolve_preprocessed_dir(root, preprocessed_dir=None):
    if preprocessed_dir and str(preprocessed_dir).strip():
        return Path(preprocessed_dir).resolve()
    root = Path(root).resolve()
    default_dir = (root / CH_BATTERY_DEFAULT_PREPROCESSED_DIR).resolve()
    if default_dir.exists():
        return default_dir
    legacy_dir = (root / CH_BATTERY_LEGACY_PREPROCESSED_DIR).resolve()
    return legacy_dir if legacy_dir.exists() else default_dir


def _load_preprocessed_split(root, preprocessed_dir, train_ratio, seed):
    preprocessed_dir = Path(preprocessed_dir).resolve()
    summary_path = preprocessed_dir / "summary.json"
    train_manifest_path = preprocessed_dir / "train_manifest.csv"
    test_manifest_path = preprocessed_dir / "test_manifest.csv"
    feature_columns_path = preprocessed_dir / "feature_columns.json"
    scaler_path = preprocessed_dir / "train_minmax_scaler.npz"
    required_paths = [summary_path, train_manifest_path, test_manifest_path, feature_columns_path, scaler_path]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing CH-BATTERY preprocess artifact: {required_path}")

    summary = _read_json(summary_path)
    expected_ratio = float(train_ratio)
    expected_seed = int(seed)
    if abs(float(summary.get("train_ratio", -1.0)) - expected_ratio) > 1e-12 or int(summary.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Preprocess split mismatch: expected train_ratio={expected_ratio}, seed={expected_seed}, "
            f"but got train_ratio={summary.get('train_ratio')}, seed={summary.get('seed')}"
        )

    train_manifest = _normalize_manifest_file_paths(pd.read_csv(train_manifest_path), root=root)
    test_manifest = _normalize_manifest_file_paths(pd.read_csv(test_manifest_path), root=root)
    feature_columns = [str(col) for col in _read_json(feature_columns_path)]
    scaler, scaler_feature_columns = _load_saved_minmax_scaler(scaler_path)
    if feature_columns != scaler_feature_columns:
        raise ValueError("CH-BATTERY preprocess artifact mismatch: feature_columns.json differs from scaler metadata")

    return train_manifest, test_manifest, feature_columns, scaler, summary


def load_ch_battery_fault_details(root, chemistry="LFP"):
    detail_path = _find_fault_detail_file(Path(root), chemistry)
    if detail_path is None:
        return {}

    detail_df = pd.read_excel(detail_path)
    detail_df.columns = [str(col).strip() for col in detail_df.columns]
    metadata = {}
    for _, row in detail_df.iterrows():
        raw_vid = row.get("vid")
        if pd.isna(raw_vid):
            continue
        try:
            vin = f"vin_{int(raw_vid)}"
        except Exception:
            vin = str(raw_vid).strip()
            if not vin.startswith("vin_"):
                vin = f"vin_{vin}"
        metadata[vin] = {
            "fault_type_detail": None if pd.isna(row.get("Fault type")) else str(row.get("Fault type")).strip(),
            "severity": None if pd.isna(row.get("Severity")) else str(row.get("Severity")).strip(),
            "fault_cell_id": None if pd.isna(row.get("Fault cell id")) else str(row.get("Fault cell id")).strip(),
            "fault_value": None if pd.isna(row.get("Fault Value")) else float(row.get("Fault Value")),
            "fault_unit": None if pd.isna(row.get("Unit")) else str(row.get("Unit")).strip(),
        }
    return metadata


def build_ch_battery_manifest(root, chemistry="LFP", cycle_kind="discharge"):
    root = Path(root)
    chemistry_root = root / chemistry
    if not chemistry_root.exists():
        raise FileNotFoundError(f"CH-BATTERY chemistry directory not found: {chemistry_root}")

    fault_details = load_ch_battery_fault_details(root, chemistry=chemistry)
    rows = []
    for csv_path in chemistry_root.glob("*/*/*.csv"):
        fault_type = csv_path.parent.parent.name
        vin = csv_path.parent.name
        stem = csv_path.stem
        current_cycle_kind = _parse_cycle_kind(stem)
        if cycle_kind and current_cycle_kind != cycle_kind:
            continue
        cycle_index = _parse_cycle_index(stem)
        sample_id = _sanitize_token(f"{chemistry}_{fault_type}_{vin}_{stem}")
        label = 0 if fault_type == "normal" else 1
        detail_meta = fault_details.get(vin, {})
        rows.append(
            {
                "sample_id": sample_id,
                "file_path": str(csv_path.resolve()),
                "relative_path": str(csv_path.relative_to(root)).replace("\\", "/"),
                "chemistry": chemistry,
                "fault_type": fault_type,
                "vin": vin,
                "cycle_kind": current_cycle_kind,
                "cycle_index": cycle_index,
                "sample_label": label,
                "severity": detail_meta.get("severity"),
                "fault_cell_id": detail_meta.get("fault_cell_id"),
                "fault_value": detail_meta.get("fault_value"),
                "fault_unit": detail_meta.get("fault_unit"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No CH-BATTERY samples found under {chemistry_root} for cycle_kind={cycle_kind}")

    manifest_df = pd.DataFrame(rows).sort_values(
        ["sample_label", "fault_type", "vin", "cycle_index", "sample_id"]
    ).reset_index(drop=True)
    return manifest_df


def _resolve_feature_columns(example_file):
    header_df = pd.read_csv(example_file, nrows=0)
    feature_columns = [col for col in header_df.columns if col not in CH_BATTERY_EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {example_file}")
    return feature_columns


def _load_ch_battery_sequence(file_path, feature_columns):
    df = pd.read_csv(file_path)
    feature_df = df.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.interpolate(limit_direction="both")
    feature_df = feature_df.ffill().bfill().fillna(0.0)
    return feature_df.to_numpy(dtype=np.float32, copy=False)


def _split_normal_vins(normal_vins, train_ratio=0.8, seed=3407):
    normal_vins = sorted(set(normal_vins))
    if len(normal_vins) < 2:
        raise ValueError("CH-BATTERY normal VIN count must be at least 2 to create train/test split")

    rng = np.random.default_rng(int(seed))
    shuffled = normal_vins.copy()
    rng.shuffle(shuffled)

    train_count = int(round(len(shuffled) * float(train_ratio)))
    train_count = min(max(train_count, 1), len(shuffled) - 1)

    train_vins = sorted(shuffled[:train_count])
    test_vins = sorted(shuffled[train_count:])
    return train_vins, test_vins


def _build_sample_map(manifest_df, feature_columns):
    sample_map = {}
    metadata_map = {}
    for row in manifest_df.itertuples(index=False):
        sample_map[row.sample_id] = _load_ch_battery_sequence(row.file_path, feature_columns)
        metadata_map[row.sample_id] = {
            "sample_id": row.sample_id,
            "file_path": row.file_path,
            "relative_path": getattr(row, "relative_path", None),
            "chemistry": row.chemistry,
            "fault_type": row.fault_type,
            "vin": row.vin,
            "cycle_kind": row.cycle_kind,
            "cycle_index": int(row.cycle_index),
            "sample_label": int(row.sample_label),
            "severity": row.severity,
            "fault_cell_id": row.fault_cell_id,
            "fault_value": row.fault_value,
            "fault_unit": row.fault_unit,
        }
    return sample_map, metadata_map


def get_ch_battery_lfp_discharge_data(
    root="datasets/CH-BATTERY",
    normalize=False,
    train_ratio=0.8,
    seed=3407,
    preprocessed_dir=None,
):
    root = Path(root).resolve()
    resolved_preprocessed_dir = _resolve_preprocessed_dir(root, preprocessed_dir=preprocessed_dir)
    scaler = None
    data_source = "raw_csv"
    test_label = None

    if resolved_preprocessed_dir.exists():
        try:
            bundle = _load_preprocessed_pkl_bundle(
                preprocessed_dir=resolved_preprocessed_dir,
                train_ratio=train_ratio,
                seed=seed,
            )
            feature_columns = bundle["feature_columns"]
            train_data_map = bundle["train_data_map"]
            test_data_map = bundle["test_data_map"]
            test_label = bundle["test_label"]
            train_meta_map = bundle["train_meta_map"]
            test_meta_map = bundle["test_meta_map"]
            scaler = bundle["scaler"]
            train_vins = sorted({meta.get("vin") for meta in train_meta_map.values() if meta.get("vin")})
            holdout_normal_vins = sorted(
                {
                    meta.get("vin")
                    for meta in test_meta_map.values()
                    if meta.get("vin") and int(meta.get("sample_label", 0)) == 0
                }
            )
            train_manifest = pd.DataFrame(train_meta_map.values()).sort_values(["sample_label", "fault_type", "vin", "cycle_index", "sample_id"]).reset_index(drop=True)
            test_manifest = pd.DataFrame(test_meta_map.values()).sort_values(["sample_label", "fault_type", "vin", "cycle_index", "sample_id"]).reset_index(drop=True)
            data_source = "preprocessed_pkl"
        except Exception as exc:
            try:
                train_manifest, test_manifest, feature_columns, scaler, _ = _load_preprocessed_split(
                    root=root,
                    preprocessed_dir=resolved_preprocessed_dir,
                    train_ratio=train_ratio,
                    seed=seed,
                )
                train_vins = sorted(train_manifest["vin"].dropna().unique().tolist())
                holdout_normal_vins = sorted(
                    test_manifest.loc[test_manifest["sample_label"] == 0, "vin"].dropna().unique().tolist()
                )
                data_source = "preprocessed_manifest"
            except Exception:
                print(f"[CH-BATTERY] Failed to load preprocess artifacts from {resolved_preprocessed_dir}, fallback to raw CSV: {exc}")
                scaler = None
                data_source = "raw_csv"

    if data_source == "raw_csv":
        manifest_df = build_ch_battery_manifest(root, chemistry="LFP", cycle_kind="discharge")
        normal_df = manifest_df[manifest_df["fault_type"] == "normal"].copy()
        fault_df = manifest_df[manifest_df["fault_type"] != "normal"].copy()
        train_vins, holdout_normal_vins = _split_normal_vins(
            normal_df["vin"].unique(), train_ratio=train_ratio, seed=seed
        )

        train_manifest = normal_df[normal_df["vin"].isin(train_vins)].copy()
        test_manifest = pd.concat(
            [
                normal_df[normal_df["vin"].isin(holdout_normal_vins)].copy(),
                fault_df.copy(),
            ],
            ignore_index=True,
        ).sort_values(["sample_label", "fault_type", "vin", "cycle_index", "sample_id"]).reset_index(drop=True)

        feature_columns = _resolve_feature_columns(train_manifest.iloc[0]["file_path"])
        train_data_map, train_meta_map = _build_sample_map(train_manifest, feature_columns)
        test_data_map, test_meta_map = _build_sample_map(test_manifest, feature_columns)
        test_label = np.asarray(
            [int(meta["sample_label"]) for meta in test_meta_map.values()],
            dtype=np.int32,
        )
    elif data_source == "preprocessed_manifest":
        train_data_map, train_meta_map = _build_sample_map(train_manifest, feature_columns)
        test_data_map, test_meta_map = _build_sample_map(test_manifest, feature_columns)
        test_label = np.asarray(
            [int(meta["sample_label"]) for meta in test_meta_map.values()],
            dtype=np.int32,
        )

    feature_columns, train_data_map, test_data_map, scaler = _project_ch_battery_to_core_features(
        feature_columns,
        train_data_map,
        test_data_map,
        scaler=scaler,
    )

    if normalize:
        if scaler is None:
            concatenated_train = flatten_sequence_collection(train_data_map.values(), dtype=np.float32)
            _, scaler = normalize_data(concatenated_train, scaler=None)
        train_data_map = {
            sample_id: scaler.transform(sequence).astype(np.float32, copy=False)
            for sample_id, sequence in train_data_map.items()
        }
        test_data_map = {
            sample_id: scaler.transform(sequence).astype(np.float32, copy=False)
            for sample_id, sequence in test_data_map.items()
        }

    split_meta = {
        "feature_columns": list(feature_columns),
        "train_vins": list(train_vins),
        "test_normal_vins": list(holdout_normal_vins),
        "train_manifest": train_manifest.reset_index(drop=True),
        "test_manifest": test_manifest.reset_index(drop=True),
        "train_metadata": train_meta_map,
        "test_metadata": test_meta_map,
        "test_label": None if test_label is None else np.asarray(test_label, dtype=np.int32),
        "data_source": data_source,
        "preprocessed_dir": str(resolved_preprocessed_dir),
    }

    print(
        "[CH-BATTERY] LFP discharge split:",
        f"source={data_source}",
        f"train_normal_vins={len(train_vins)}",
        f"test_normal_vins={len(holdout_normal_vins)}",
        f"train_samples={len(train_data_map)}",
        f"test_samples={len(test_data_map)}",
        f"fault_test_samples={int(test_manifest['sample_label'].sum())}",
    )

    return (train_data_map, None), (test_data_map, None if test_label is None else np.asarray(test_label, dtype=np.int32)), split_meta


def aggregate_ch_battery_sample_scores(score_df, topk_ratio=CH_BATTERY_DEFAULT_TOPK_RATIO):
    scores = score_df["A_Score_Global"].to_numpy(dtype=np.float32)
    if scores.size == 0:
        raise ValueError("Score dataframe is empty")

    topk_count = max(1, int(np.ceil(scores.size * float(topk_ratio))))
    topk_values = np.partition(scores, -topk_count)[-topk_count:]
    pred_positive_ratio = None
    if "A_Pred_Global" in score_df.columns:
        pred_positive_ratio = float(score_df["A_Pred_Global"].mean())

    return {
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_topk_mean": float(np.mean(topk_values)),
        "pred_positive_ratio": pred_positive_ratio,
        "window_count": int(scores.size),
        "topk_count": int(topk_count),
    }


def _best_f1_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {"best_f1": None, "best_threshold": None}

    f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-8, None)
    best_index = int(np.nanargmax(f1_values))
    return {
        "best_f1": float(f1_values[best_index]),
        "best_threshold": float(thresholds[best_index]),
        "best_precision": float(precision[best_index]),
        "best_recall": float(recall[best_index]),
    }


def save_ch_battery_sample_level_reports(save_path, sample_rows, score_field="score_topk_mean"):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(sample_rows).sort_values(
        [score_field, "sample_label", "fault_type", "sample_id"],
        ascending=[False, False, True, True],
    )
    report_df.to_csv(save_path / "ch_battery_sample_scores.csv", index=False)

    labels = report_df["sample_label"].to_numpy(dtype=np.int32)
    scores = report_df[score_field].to_numpy(dtype=np.float32)

    summary = {
        "dataset": CH_BATTERY_DATASET_NAME,
        "sample_count": int(len(report_df)),
        "normal_count": int(np.sum(labels == 0)),
        "fault_count": int(np.sum(labels == 1)),
        "score_field": score_field,
        "fault_types": sorted(report_df["fault_type"].dropna().unique().tolist()),
    }

    if len(np.unique(labels)) >= 2:
        summary["sample_auroc"] = float(roc_auc_score(labels, scores))
        summary["sample_auprc"] = float(average_precision_score(labels, scores))
        summary.update(_best_f1_from_scores(labels, scores))

    fault_type_summary = (
        report_df.groupby("fault_type")[["sample_label", "score_max", "score_p95", "score_topk_mean"]]
        .agg(
            sample_count=("sample_label", "size"),
            positive_ratio=("sample_label", "mean"),
            mean_score_max=("score_max", "mean"),
            mean_score_p95=("score_p95", "mean"),
            mean_score_topk=("score_topk_mean", "mean"),
        )
        .reset_index()
    )
    fault_type_summary.to_csv(save_path / "ch_battery_fault_type_summary.csv", index=False)

    with (save_path / "ch_battery_sample_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# CH-BATTERY Sample-Level Summary",
        "",
        f"- sample_count: {summary['sample_count']}",
        f"- normal_count: {summary['normal_count']}",
        f"- fault_count: {summary['fault_count']}",
        f"- score_field: {summary['score_field']}",
    ]
    if "sample_auroc" in summary:
        lines.append(f"- sample_auroc: {summary['sample_auroc']:.4f}")
        lines.append(f"- sample_auprc: {summary['sample_auprc']:.4f}")
        lines.append(f"- best_f1: {summary['best_f1']:.4f}")
        lines.append(f"- best_threshold: {summary['best_threshold']:.6f}")
    (save_path / "ch_battery_sample_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report_df, summary
