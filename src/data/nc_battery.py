"""清华 EV 电池数据集的读取、索引、车辆划分、归一化和窗口化工具。

数据按三个品牌分别存放，标签定义在车辆级，而每个 pkl 只保存一个充电片段。
因此划分和评估全过程都保留车辆身份，不能把故障车辆的每个片段直接当成独立正样本。
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
# 七个通道中，电流和 SOC 被视为外部工况/控制量；其余五个通道是
# 需要由模型预测并用于故障评分的响应量。该划分贯穿 C3 工况编码与 C4 物理评分。
CONTROL_DIMS = (1, 2)
RESPONSE_DIMS = (0, 3, 4, 5, 6)

@dataclass(frozen=True)
class SnippetRecord:
    """一条充电片段的轻量索引。

    原始 pkl 可能很大，因此调度、车辆划分和 DataLoader 初始化只传递本结构；
    真正的时序数组只在 ``__getitem__`` 取到一个 batch 时才读取。``car`` 与
    ``label`` 是车辆级字段，不能把它们误解为每个片段独立标注。
    """
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
    """读取一个 pkl 的元信息，并将其转换成可持久化的索引行。"""
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
    """定位一个品牌的原始目录，并兼容本地和 Kaggle 的嵌套挂载布局。

    ``root`` 应是三个品牌的共同父目录，而非某个 ``battery_brandN`` 目录。
    Kaggle 上传时常额外保留一层同名目录，故依次尝试
    ``.../battery_brandN`` 和 ``.../battery_brandN/battery_brandN``。
    """
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
    # Kaggle 会保留上传数据集时选中的目录层级，所以官方数据常被挂载成
    # battery_brandN/battery_brandN/{label,train,test} 的双层结构。
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
    except TypeError:  # 兼容不支持 weights_only 参数的旧版 PyTorch
        return torch.load(path, map_location="cpu")

def _load_payload(path: Path):
    """兼容两种官方 pkl：zip 容器中的 ``data.pkl`` 与普通 torch/pickle 文件。"""
    try:
        with ZipFile(path) as archive:
            data_member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
            return pickle.loads(archive.read(data_member))
    except (BadZipFile, StopIteration, pickle.UnpicklingError):
        return _torch_load(path)

def load_snippet(path: str | Path) -> tuple[np.ndarray, dict]:
    """返回单片段的 ``[时间, 7通道]`` float32 数组及其车辆元数据。"""
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
    """合并品牌目录下的车辆标签 CSV，并检查同一车辆是否出现冲突标签。"""
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
    """建立或复用片段索引，而不修改 Kaggle 只读 Input。

    索引文件写到 ``datasets/processed/indices``。复用已有索引时会把旧机器的
    绝对路径重定向到当前挂载目录；这使同一实验能在本地和 Kaggle 间迁移。
    新建索引时，三个可能的数据目录 ``data/train/test`` 会被合并，但训练/测试
    划分随后仍只按车辆标签与协议完成，绝不直接信任目录名作为论文划分。
    """
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
                    # 上传 Kaggle 前生成的索引可能带有原机器绝对路径。这里只保留
                    # data、train 或 test 之后的相对部分，再重定位到当前挂载目录。
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

    # 限量冒烟测试生成的索引不能覆盖完整的持久化索引。
    if max_snippets <= 0:
        with index_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return records

def prepared_index_path(brand: int) -> Path:
    """返回所有电池实验入口共同使用的可写预处理索引路径。"""
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
    """按车辆而非片段构建训练、验证、校准和测试集合。

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
        # 公开划分 notebook 先排序车辆编号，再按固定随机种子打乱，最后用整数边界
        # 切成五折；这里保持同样顺序以复现公开协议。
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
        # 公开论文补充材料中的协议：N_{-i} 用于训练，N_{-i}∪A_i 用于校准阈值，
        # N_i∪A_{-i} 作为最终测试集。
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
    """仅用当前折训练车辆拟合的逐通道 Min-Max 归一化器。

    流式更新避免将全部 pkl 同时放入内存；验证、校准、测试只能调用 ``transform``，
    从而避免使用测试分布信息。
    """

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
    """复现 DyAD 公开实现的通道归一化，仅使用前 200 条训练片段拟合。"""

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
    """把不等长充电片段延迟转换为监督式固定窗口样本。

    一个样本包含 ``lookback`` 个历史点作为输入 ``x``，以及紧随其后的一个点作为
    预测目标 ``y``。同一片段可按均匀位置产生多个窗口；评估时附带车辆、标签和
    片段 ID，以便最终汇总回车辆级指标。
    """

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
    """将窗口异常分数两级汇总为论文报告所用的车辆分数。

    先对同一片段的窗口取均值，再取车辆内最高 ``top_ratio`` 比例片段的均值。
    这能突出故障车辆中最异常的充电过程，同时避免单个异常窗口完全主导结果。
    """
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
