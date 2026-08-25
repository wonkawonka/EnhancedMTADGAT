"""Single data contract for all external time-series baselines.

Every adapter receives normal-only training sequences and entity-preserving test
sequences.  Labelled corpora expose point/snippet labels; BMS deliberately
exposes ``labels=None`` because its released test labels are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.nc_battery import (
    PaperChannelNormalizer,
    StreamingMinMaxScaler,
    build_index,
    load_snippet,
    split_vehicle_folds,
)
from src.data.utils import get_bms_cluster_data, get_nasa_telemetry_sequence_data
from src.project_paths import resolve_dataset_root


@dataclass
class ExternalProtocolData:
    dataset: str
    train_sequences: list[np.ndarray]
    validation_sequences: list[np.ndarray]
    test_sequences: list[np.ndarray]
    validation_labels: Optional[list[np.ndarray]]
    test_labels: Optional[list[np.ndarray]]
    validation_entity_ids: list[str]
    entity_ids: list[str]
    evaluation_kind: str  # point_ranking, vehicle_ranking, or normal_only
    metadata: dict


def load_external_protocol_data(
    dataset: str,
    *,
    brand_fold: int = 0,
    seed: int = 3407,
    val_ratio: float = 0.1,
    brand_split_protocol: str = "paper_protocol",
    brand_fold_seed: int = 0,
    brand_normalization: str = "paper_channel",
) -> ExternalProtocolData:
    """Load MSL/SMAP, Brand3, or BMS under one leakage-safe contract."""
    name = str(dataset).upper()
    if name in {"MSL", "SMAP"}:
        train, validation, test, labels = get_nasa_telemetry_sequence_data(
            name, val_ratio=val_ratio, normalize=True
        )
        return ExternalProtocolData(
            dataset=name,
            train_sequences=[np.asarray(x, dtype=np.float32) for x in train],
            validation_sequences=[np.asarray(x, dtype=np.float32) for x in validation],
            test_sequences=[np.asarray(x, dtype=np.float32) for x in test],
            validation_labels=None,
            test_labels=[np.asarray(x, dtype=np.int32) for x in labels],
            validation_entity_ids=[f"{name}_{index}" for index in range(len(validation))],
            entity_ids=[f"{name}_{index}" for index in range(len(test))],
            evaluation_kind="point_ranking",
            metadata={"normalization": "training_only_minmax", "seed": int(seed)},
        )

    if name == "BMS":
        (train_map, _), (test_map, label_map) = get_bms_cluster_data(normalize=True)
        entities = sorted(train_map)
        # The released BMS labels are known placeholders.  Do not pass all-zero
        # labels as though ranking metrics were meaningful.
        if any(label_map[key] is not None and np.any(np.asarray(label_map[key]) != 0) for key in entities):
            raise ValueError("BMS protocol expects the current normal-only release")
        train_sequences = []
        validation_sequences = []
        for key in entities:
            split = int(len(train_map[key]) * (1.0 - float(val_ratio)))
            train_sequences.append(np.asarray(train_map[key][:split], dtype=np.float32))
            validation_sequences.append(np.asarray(train_map[key][split:], dtype=np.float32))
        return ExternalProtocolData(
            dataset=name,
            train_sequences=train_sequences,
            validation_sequences=validation_sequences,
            test_sequences=[np.asarray(test_map[key], dtype=np.float32) for key in entities],
            validation_labels=None,
            test_labels=None,
            validation_entity_ids=entities,
            entity_ids=entities,
            evaluation_kind="normal_only",
            metadata={"normalization": "training_only_maxabs", "split": "per_cluster_temporal_80_20", "seed": int(seed)},
        )

    if name == "BRAND3":
        protocol = str(brand_split_protocol).strip().lower()
        if protocol not in {"paper_protocol", "strict_normal_validation"}:
            raise ValueError(f"Unsupported Brand3 split protocol: {brand_split_protocol}")
        normalization = str(brand_normalization).strip().lower()
        if normalization not in {"paper_channel", "minmax"}:
            raise ValueError(f"Unsupported Brand3 normalization: {brand_normalization}")
        records = build_index(resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV"), 3)
        splits = split_vehicle_folds(
            records,
            int(brand_fold),
            seed=int(brand_fold_seed),
            protocol=protocol,
        )
        train_records = splits["train"]
        validation_records = (
            splits["calibration"] if protocol == "paper_protocol" else splits["validation"]
        )
        test_records = splits["test"]
        scaler = (
            PaperChannelNormalizer(train_records)
            if normalization == "paper_channel"
            else StreamingMinMaxScaler().fit_records(train_records)
        )
        load_values = lambda record: scaler.transform(load_snippet(record.path)[0])
        train_values = [load_values(record) for record in train_records]
        validation_values = [load_values(record) for record in validation_records]
        test_values = [load_values(record) for record in test_records]
        validation_labels = [np.asarray([int(record.label)], dtype=np.int32) for record in validation_records]
        labels = [np.asarray([int(record.label)], dtype=np.int32) for record in test_records]
        validation_entities = [str(record.car) for record in validation_records]
        entities = [str(record.car) for record in test_records]
        return ExternalProtocolData(
            dataset=name,
            train_sequences=train_values,
            validation_sequences=validation_values,
            test_sequences=test_values,
            validation_labels=validation_labels,
            test_labels=labels,
            validation_entity_ids=validation_entities,
            entity_ids=entities,
            evaluation_kind="vehicle_ranking",
            metadata={
                "brand": 3,
                "fold": int(brand_fold),
                "fold_seed": int(brand_fold_seed),
                "protocol": protocol,
                "normalization": scaler.state_dict(),
                "seed": int(seed),
            },
        )

    raise ValueError(f"Unsupported external protocol dataset: {dataset}")
