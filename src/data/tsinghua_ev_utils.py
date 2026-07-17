"""Loader and snippet-level evaluation for the public Tsinghua EV package.

The released Train.zip stores labeled charging snippets but does not expose a
vehicle identifier. Consequently, this module never claims a vehicle-level
split. Test.zip is unlabeled and is not used to compute thesis metrics.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np

from src.project_paths import resolve_dataset_root


DATASET_NAME = "TSINGHUA_EV"
FEATURE_NAMES = [
    "voltage",
    "current",
    "soc",
    "max_single_voltage",
    "min_single_voltage",
    "max_temperature",
    "min_temperature",
]

_LABEL_KEYS = ("label", "fault_label", "class", "target", "state")


class PhysicalGroupScaler:
    """Robust affine scaler that preserves electrical/thermal differences."""

    def __init__(self):
        self.center_ = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.scale_ = np.ones(len(FEATURE_NAMES), dtype=np.float32)

    @staticmethod
    def _robust_scale(values):
        q25, q75 = np.quantile(values, [0.25, 0.75])
        return max(float(q75 - q25), 1e-6)

    def fit(self, data):
        values = np.asarray(data, dtype=np.float32)
        for indices in ([0, 3, 4], [5, 6], [2]):
            group = values[:, indices].reshape(-1)
            center = float(np.median(group))
            scale = self._robust_scale(group)
            self.center_[indices] = center
            self.scale_[indices] = scale

        current = values[:, 1]
        self.center_[1] = 0.0
        self.scale_[1] = max(float(np.quantile(np.abs(current), 0.95)), 1e-6)
        return self

    def transform(self, data):
        values = np.asarray(data, dtype=np.float32)
        return ((values - self.center_) / self.scale_).astype(np.float32, copy=False)


def _metadata_mapping(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return {str(key).lower(): value for key, value in metadata.items()}
    if hasattr(metadata, "to_dict"):
        value = metadata.to_dict()
        if isinstance(value, dict):
            return {str(key).lower(): item for key, item in value.items()}
    if isinstance(metadata, (tuple, list)):
        result = {}
        if metadata:
            result["label"] = metadata[0]
        if len(metadata) > 1:
            result["mileage"] = metadata[1]
        return result
    return {"label": metadata}


def _normalize_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"00", "0", "normal", "healthy", "false"}:
        return 0
    if text in {"10", "1", "abnormal", "fault", "faulty", "true"}:
        return 1
    return None


def _record_from_payload(payload, sample_id, source_path):
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        return None
    data, raw_metadata = payload[0], payload[1]
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 7:
        return None

    metadata = _metadata_mapping(raw_metadata)
    label_value = next((metadata[key] for key in _LABEL_KEYS if key in metadata), None)
    label = _normalize_label(label_value)
    if label is None:
        return None
    return {
        "sample_id": str(sample_id),
        "label": label,
        "mileage": metadata.get("mileage", metadata.get("mile")),
        "data": np.nan_to_num(array[:, :7], copy=False),
        "source_path": str(source_path),
    }


def _iter_extracted_pickles(root_path: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root_path.rglob("*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        record = _record_from_payload(
            payload,
            sample_id=path.relative_to(root_path).with_suffix(""),
            source_path=path,
        )
        if record is not None:
            yield record


def _iter_zip_pickles(zip_path: Path) -> Iterable[dict[str, Any]]:
    with ZipFile(zip_path) as archive:
        for member in sorted(name for name in archive.namelist() if name.lower().endswith(".pkl")):
            with archive.open(member) as handle:
                payload = pickle.load(handle)
            record = _record_from_payload(
                payload,
                sample_id=f"{zip_path.stem}/{Path(member).with_suffix('')}",
                source_path=f"{zip_path}!{member}",
            )
            if record is not None:
                yield record


def load_tsinghua_ev_snippets(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the labeled public training snippets from extracted files or Train.zip."""
    root_path = Path(root) if root else resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")
    if not root_path.exists():
        raise FileNotFoundError(
            f"Tsinghua EV dataset not found: {root_path}. "
            "Set MTAD_GAT_TSINGHUA_EV_ROOT or --tsinghua_ev_root."
        )

    extracted = list(_iter_extracted_pickles(root_path))
    if extracted:
        records = extracted
    else:
        train_zip = root_path / "Train.zip"
        if not train_zip.exists():
            raise FileNotFoundError(f"Neither labeled .pkl files nor Train.zip were found under {root_path}")
        records = list(_iter_zip_pickles(train_zip))

    if not records:
        raise ValueError(f"No labeled Tsinghua EV snippets were found under {root_path}")
    return records


def _split_normal_records(records, train_ratio, validation_ratio, seed):
    ordered = list(records)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    count = len(ordered)
    train_count = max(1, int(np.floor(count * train_ratio)))
    validation_count = max(1, int(np.floor(count * validation_ratio)))
    if train_count + validation_count >= count:
        validation_count = max(1, count - train_count - 1)
    return (
        ordered[:train_count],
        ordered[train_count:train_count + validation_count],
        ordered[train_count + validation_count:],
    )


def get_tsinghua_ev_data(
    root: str | Path | None = None,
    normalize: bool = True,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    max_train_samples: int = 0,
    max_validation_samples: int = 0,
    max_test_samples_per_class: int = 0,
    seed: int = 3407,
):
    """Return normal train/calibration snippets and a labeled snippet test set."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Tsinghua ratios require train_ratio > 0, validation_ratio > 0 and sum < 1")

    records = load_tsinghua_ev_snippets(root)
    normal_records = [record for record in records if record["label"] == 0]
    abnormal_records = [record for record in records if record["label"] == 1]
    train_records, validation_records, heldout_normal = _split_normal_records(
        normal_records, train_ratio, validation_ratio, seed
    )
    unused_abnormal_train, unused_abnormal_validation, heldout_abnormal = _split_normal_records(
        abnormal_records, train_ratio, validation_ratio, seed + 17
    )
    if max_train_samples > 0:
        train_records = train_records[:max_train_samples]
    if max_validation_samples > 0:
        validation_records = validation_records[:max_validation_samples]
    if max_test_samples_per_class > 0:
        heldout_normal = heldout_normal[:max_test_samples_per_class]
        heldout_abnormal = heldout_abnormal[:max_test_samples_per_class]
    test_records = heldout_normal + heldout_abnormal
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(test_records)

    scaler = None
    if normalize:
        scaler = PhysicalGroupScaler().fit(
            np.concatenate([record["data"] for record in train_records], axis=0)
        )

    def build_map(selected_records):
        data_map = {}
        metadata = {}
        for record in selected_records:
            sample_id = record["sample_id"]
            values = record["data"]
            if scaler is not None:
                values = scaler.transform(values)
            data_map[sample_id] = values
            metadata[sample_id] = {
                "label": record["label"],
                "mileage": record["mileage"],
                "source_path": record["source_path"],
            }
        return data_map, metadata

    train_map, train_metadata = build_map(train_records)
    validation_map, validation_metadata = build_map(validation_records)
    test_map, test_metadata = build_map(test_records)
    split_metadata = {
        "label_level": "charging_snippet",
        "vehicle_identity_available": False,
        "split_warning": "The public package has no vehicle ID; vehicle-independent evaluation is unavailable.",
        "feature_names": list(FEATURE_NAMES),
        "train_ratio_normal": float(train_ratio),
        "validation_ratio_normal": float(validation_ratio),
        "test_ratio_normal": float(1.0 - train_ratio - validation_ratio),
        "train_normal_count": len(train_records),
        "validation_normal_count": len(validation_records),
        "test_normal_count": len(heldout_normal),
        "test_abnormal_count": len(heldout_abnormal),
        "unused_abnormal_train_count": len(unused_abnormal_train),
        "unused_abnormal_validation_count": len(unused_abnormal_validation),
        "train_metadata": train_metadata,
        "validation_data": validation_map,
        "validation_metadata": validation_metadata,
        "test_metadata": test_metadata,
        "scaler": scaler,
    }
    return (train_map, None), (test_map, test_metadata), split_metadata


def aggregate_sample_scores(
    sample_scores: dict[str, np.ndarray | float],
    sample_metadata: dict[str, dict[str, Any]],
    top_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate window scores to one robust score per charging snippet."""
    rows = []
    for sample_id, score_values in sample_scores.items():
        metadata = sample_metadata.get(sample_id)
        if metadata is None:
            continue
        values = np.asarray(score_values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        count = max(1, int(np.ceil(values.size * top_ratio)))
        score = float(np.mean(np.partition(values, -count)[-count:]))
        rows.append((sample_id, int(metadata["label"]), score))

    rows.sort(key=lambda item: item[0])
    return (
        np.asarray([row[2] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        [row[0] for row in rows],
    )
