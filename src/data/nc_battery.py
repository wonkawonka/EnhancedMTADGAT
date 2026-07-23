"""Official Nature Communications EV battery dataset utilities.

The dataset is organised as three manufacturer-specific packages.  Labels are
defined at vehicle level, while each pickle stores one charging snippet.  This
module therefore keeps vehicle identity throughout splitting and evaluation;
it never treats every snippet from a faulty vehicle as a positive example.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset

from src.project_paths import processed_dataset_path, resolve_dataset_root


FEATURE_NAMES = (
    "pack_voltage",
    "current",
    "soc",
    "max_cell_voltage",
    "min_cell_voltage",
    "max_temperature",
    "min_temperature",
)
CONTROL_DIMS = (1, 2)
RESPONSE_DIMS = (0, 3, 4, 5, 6)


@dataclass(frozen=True)
class SnippetRecord:
    path: str
    car: str
    label: int
    length: int
    mileage: float | None = None
    charge_segment: str | None = None


_INDEX_LABELS: dict[str, int] = {}


def _init_index_worker(labels: dict[str, int]) -> None:
    global _INDEX_LABELS
    _INDEX_LABELS = labels


def _parse_index_path(path_value: str) -> SnippetRecord:
    path = Path(path_value)
    values, metadata = load_snippet(path)
    car = str(metadata.get("car"))
    if car not in _INDEX_LABELS:
        raise ValueError(f"Snippet {path.name} references unknown car {car}")
    mileage = metadata.get("mileage")
    return SnippetRecord(
        path=str(path.resolve()),
        car=car,
        label=_INDEX_LABELS[car],
        length=int(values.shape[0]),
        mileage=None if mileage is None else float(mileage),
        charge_segment=None
        if metadata.get("charge_segment") is None
        else str(metadata.get("charge_segment")),
    )


def resolve_brand_root(root: str | Path | None, brand: int) -> Path:
    if brand not in {1, 2, 3}:
        raise ValueError("brand must be 1, 2, or 3")
    base = Path(root) if root else resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")
    layouts = (
        base,
        base / "TSINGHUA_EV",
        base / "datasets" / "TSINGHUA_EV",
    )
    brand_name = f"battery_brand{brand}"
    candidates = []
    for layout in layouts:
        outer = layout / brand_name
        candidates.extend((outer, outer / brand_name))
    # Kaggle preserves every directory selected during Dataset upload.  The
    # official package is therefore commonly mounted as
    # battery_brandN/battery_brandN/{label,train,test}.
    brand_root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "label").is_dir()
            and any((candidate / folder).is_dir() for folder in ("data", "train", "test"))
        ),
        candidates[0],
    )
    if not brand_root.is_dir():
        raise FileNotFoundError(
            f"Official battery brand directory not found: {brand_root}. "
            "Kaggle Input must contain extracted battery_brand1/2/3 directories; "
            "otherwise set MTAD_GAT_TSINGHUA_EV_ROOT to their parent directory."
        )
    return brand_root


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _load_payload(path: Path):
    """Read the official numpy pickle directly, avoiding torch.load per-file overhead."""
    try:
        with ZipFile(path) as archive:
            data_member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
            return pickle.loads(archive.read(data_member))
    except (BadZipFile, StopIteration, pickle.UnpicklingError):
        return _torch_load(path)


def load_snippet(path: str | Path) -> tuple[np.ndarray, dict]:
    payload = _load_payload(Path(path))
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        raise ValueError(f"Unexpected battery snippet payload: {path}")
    values = np.asarray(payload[0], dtype=np.float32)
    metadata = dict(payload[1])
    if values.ndim != 2 or values.shape[1] < len(FEATURE_NAMES):
        raise ValueError(f"Expected at least seven channels in {path}, got {values.shape}")
    values = np.nan_to_num(values[:, : len(FEATURE_NAMES)], copy=False)
    return values, metadata


def _read_vehicle_labels(brand_root: Path) -> dict[str, int]:
    label_paths = sorted((brand_root / "label").glob("*_label.csv"))
    if not label_paths:
        raise FileNotFoundError(f"Vehicle label files not found under: {brand_root / 'label'}")
    labels = {}
    for label_path in label_paths:
        with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                car, label = str(row["car"]), int(row["label"])
                if car in labels and labels[car] != label:
                    raise ValueError(f"Conflicting labels for car {car} in {label_path}")
                labels[car] = label
    return labels


def build_index(
    root: str | Path | None,
    brand: int,
    *,
    force: bool = False,
    max_snippets: int = 0,
    workers: int = 8,
) -> list[SnippetRecord]:
    """Build/load a compact JSONL index without writing to read-only raw data."""
    brand_root = resolve_brand_root(root, brand)
    index_dir = processed_dataset_path("TSINGHUA_EV", for_write=True) / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"battery_brand{brand}_snippet_index.jsonl"
    if index_path.is_file() and not force:
        records = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                indexed_path = Path(payload["path"])
                if not indexed_path.is_file():
                    # Indexes generated before uploading to Kaggle can contain
                    # absolute paths from the source machine. Preserve the
                    # data/train/test subfolder and rebase them to this mount.
                    rebased = brand_root / indexed_path.parent.name / indexed_path.name
                    if not rebased.is_file():
                        raise FileNotFoundError(
                            f"Indexed snippet is missing both at {indexed_path} and {rebased}"
                        )
                    payload["path"] = str(rebased.resolve())
                records.append(SnippetRecord(**payload))
                if max_snippets > 0 and len(records) >= max_snippets:
                    break
        return records

    labels = _read_vehicle_labels(brand_root)
    paths = sorted(
        (
            path
            for folder in ("data", "train", "test")
            for path in (brand_root / folder).glob("*.pkl")
        ),
        key=lambda item: str(item.relative_to(brand_root)),
    )
    if max_snippets > 0:
        paths = paths[:max_snippets]
    if workers <= 1:
        _init_index_worker(labels)
        records = [_parse_index_path(str(path)) for path in paths]
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers), initializer=_init_index_worker, initargs=(labels,)
        ) as pool:
            records = list(pool.map(_parse_index_path, map(str, paths), chunksize=256))

    # A capped smoke index must not replace the complete persistent index.
    if max_snippets <= 0:
        with index_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return records


def prepared_index_path(brand: int) -> Path:
    """Return the writable preprocessing artifact used by all battery runners."""
    return (
        processed_dataset_path("TSINGHUA_EV", for_write=True)
        / "indices"
        / f"battery_brand{brand}_snippet_index.jsonl"
    )


BATTERY_SPLIT_PROTOCOLS = ("strict_normal_validation", "paper_protocol")


def split_vehicle_folds(
    records: Sequence[SnippetRecord],
    fold: int,
    *,
    folds: int = 5,
    seed: int = 3407,
    protocol: str = "strict_normal_validation",
) -> dict[str, list[SnippetRecord]]:
    """Build vehicle-level folds for the strict or Zhang et al. protocol.

    ``strict_normal_validation`` reserves disjoint normal folds for model
    selection and testing; every faulty vehicle is evaluated in every fold.
    ``paper_protocol`` reproduces Supplementary Note 2: four normal folds are
    used for training, the held-out faulty fold is used with those normal cars
    for labelled threshold calibration, and the other faulty folds are tested.

    ``validation`` is always normal-only and is intended for model selection.
    ``calibration`` is the threshold/scoring calibration split and may contain
    labelled faults only under ``paper_protocol``.
    """
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in [0, {folds - 1}]")
    if protocol not in BATTERY_SPLIT_PROTOCOLS:
        raise ValueError(
            f"Unsupported battery split protocol {protocol!r}; "
            f"choose from {BATTERY_SPLIT_PROTOCOLS}"
        )
    labels = {record.car: record.label for record in records}
    normal_cars = sorted(car for car, label in labels.items() if label == 0)
    faulty_cars = sorted(car for car, label in labels.items() if label == 1)
    if protocol == "paper_protocol":
        # The released split notebook sorts vehicle IDs, applies Python's
        # random.shuffle(seed=0), then slices with integer fifth boundaries.
        rng = random.Random(seed)
        rng.shuffle(normal_cars)
        rng.shuffle(faulty_cars)

        def paper_folds(cars):
            return [
                cars[int(index * len(cars) / folds):int((index + 1) * len(cars) / folds)]
                for index in range(folds)
            ]

        normal_folds = paper_folds(normal_cars)
        faulty_folds = paper_folds(faulty_cars)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(normal_cars)
        rng.shuffle(faulty_cars)
        normal_folds = [list(part) for part in np.array_split(normal_cars, folds)]
        faulty_folds = [list(part) for part in np.array_split(faulty_cars, folds)]
    test_normal = set(normal_folds[fold])

    def select(cars: set[str]) -> list[SnippetRecord]:
        return [record for record in records if record.car in cars]

    if protocol == "paper_protocol":
        # Zhang et al., Supplementary Note 2: N_{-i} trains the model,
        # N_{-i} U A_i tunes tau, and N_i U A_{-i} is the test split.
        train_normal = set(normal_cars) - test_normal
        calibration_faulty = set(faulty_folds[fold])
        test_faulty = set(faulty_cars) - calibration_faulty
        train_records = select(train_normal)
        return {
            "train": train_records,
            "validation": train_records,
            "calibration": select(train_normal | calibration_faulty),
            "test": select(test_normal | test_faulty),
        }

    validation_normal = set(normal_folds[(fold + 1) % folds])
    train_normal = set(normal_cars) - test_normal - validation_normal
    validation_records = select(validation_normal)
    return {
        "train": select(train_normal),
        "validation": validation_records,
        "calibration": validation_records,
        "test": select(test_normal | set(faulty_cars)),
    }


class StreamingMinMaxScaler:
    """Per-channel train-only min/max scaler used by all internal models."""

    def __init__(self):
        self.data_min_ = np.full(len(FEATURE_NAMES), np.inf, dtype=np.float64)
        self.data_max_ = np.full(len(FEATURE_NAMES), -np.inf, dtype=np.float64)

    def fit_records(self, records: Iterable[SnippetRecord]) -> "StreamingMinMaxScaler":
        count = 0
        for record in records:
            values, _ = load_snippet(record.path)
            self.data_min_ = np.minimum(self.data_min_, np.min(values, axis=0))
            self.data_max_ = np.maximum(self.data_max_, np.max(values, axis=0))
            count += 1
        if count == 0:
            raise ValueError("Cannot fit scaler on an empty training split")
        return self

    @property
    def scale_(self) -> np.ndarray:
        return np.maximum(self.data_max_ - self.data_min_, 1e-6)

    @property
    def offset_(self) -> np.ndarray:
        return self.data_min_

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.data_min_) / self.scale_).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "kind": "train_fold_minmax",
            "data_min": self.data_min_.tolist(),
            "data_max": self.data_max_.tolist(),
            "feature_names": list(FEATURE_NAMES),
        }


class PaperChannelNormalizer:
    """DyAD public-code channel normalizer fitted on 200 training snippets."""

    def __init__(self, records: Sequence[SnippetRecord], sample_count: int = 200):
        arrays = [load_snippet(record.path)[0] for record in records[:sample_count]]
        if not arrays:
            raise ValueError("No normal training snippets available for normalization")
        stacked = np.stack(arrays)
        self.mean = np.mean(np.mean(stacked, axis=1), axis=0)
        self.std = np.mean(np.std(stacked, axis=1), axis=0)
        self.minimum = np.min(stacked, axis=(0, 1))
        self.maximum = np.max(stacked, axis=(0, 1))
        self.scale = np.maximum(
            np.maximum(1e-4, self.std),
            0.1 * (self.maximum - self.minimum),
        )
        self.sample_count = len(arrays)

    @property
    def offset_(self) -> np.ndarray:
        return self.mean

    @property
    def scale_(self) -> np.ndarray:
        return self.scale

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)

    def state_dict(self) -> dict:
        return {
            "kind": "zhang2023_dyad_first_200_channel_normalizer",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "sample_count": self.sample_count,
            "feature_names": list(FEATURE_NAMES),
        }


class BatterySnippetWindowDataset(Dataset):
    """Lazy fixed-window view; files are loaded only when a batch requests them."""

    def __init__(
        self,
        records: Sequence[SnippetRecord],
        lookback: int,
        scaler: StreamingMinMaxScaler | None,
        *,
        windows_per_snippet: int = 1,
        include_metadata: bool = False,
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.records = [record for record in records if record.length > lookback]
        self.lookback = int(lookback)
        self.scaler = scaler
        self.windows_per_snippet = max(1, int(windows_per_snippet))
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_snippet

    def __getitem__(self, index: int):
        record = self.records[index // self.windows_per_snippet]
        window_index = index % self.windows_per_snippet
        values, _ = load_snippet(record.path)
        max_start = len(values) - self.lookback - 1
        if self.windows_per_snippet == 1:
            start = max_start // 2
        else:
            start = int(round(window_index * max_start / (self.windows_per_snippet - 1)))
        sequence = values[start : start + self.lookback + 1]
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)
        x = torch.from_numpy(sequence[:-1]).float()
        y = torch.from_numpy(sequence[-1:]).float()
        if not self.include_metadata:
            return x, y
        path = Path(record.path)
        mileage = float("nan") if record.mileage is None else float(record.mileage)
        return x, y, record.car, record.label, f"{path.parent.name}/{path.stem}", mileage


def aggregate_vehicle_scores(
    snippet_scores: dict[str, list[float]],
    snippet_cars: dict[str, str],
    vehicle_labels: dict[str, int],
    top_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate window→snippet by mean, then snippet→vehicle by top-p mean."""
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    by_car: dict[str, list[float]] = {}
    for snippet_id, scores in snippet_scores.items():
        if scores:
            by_car.setdefault(snippet_cars[snippet_id], []).append(float(np.mean(scores)))
    cars = sorted(by_car)
    vehicle_scores = []
    for car in cars:
        values = np.asarray(by_car[car], dtype=np.float64)
        count = max(1, int(math.ceil(len(values) * top_ratio)))
        vehicle_scores.append(float(np.mean(np.partition(values, -count)[-count:])))
    return (
        np.asarray(vehicle_scores, dtype=np.float64),
        np.asarray([vehicle_labels[car] for car in cars], dtype=np.int64),
        cars,
    )
