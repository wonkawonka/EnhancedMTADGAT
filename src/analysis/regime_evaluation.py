"""Evaluation utilities for condition embeddings and unlabeled BMS operation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from src.data.regime_utils import BMS_REGIME_NAMES, NASA_REGIME_NAMES, derive_bms_regime_labels
from src.data.utils import BMS_FEATURE_NAMES, SlidingWindowDataset, ensure_sequence_list


def _unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def _collect_nasa_embeddings(model, entity_data, window_size, max_windows, device):
    embeddings = []
    labels = []
    base_model = _unwrap_model(model)
    for values in entity_data.values():
        for sequence in ensure_sequence_list(values):
            tensor = sequence if isinstance(sequence, torch.Tensor) else torch.as_tensor(sequence, dtype=torch.float32)
            if len(tensor) <= window_size:
                continue
            dataset = SlidingWindowDataset(tensor, window_size, target_dim=None, stride=1)
            if len(dataset) > max_windows:
                indices = np.linspace(0, len(dataset) - 1, max_windows, dtype=np.int64).tolist()
                dataset = Subset(dataset, indices)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    embedding = base_model.encode_regime(x).detach().cpu().numpy()
                    # The raw discrete step code is deliberately absent from model
                    # inputs. This weak sanity probe groups windows by current direction.
                    current = y[:, 0, 1].detach().cpu().numpy()
                    label = np.zeros(len(current), dtype=np.int64)
                    label[current > 0.05] = 1
                    label[current < -0.05] = 2
                    embeddings.append(embedding)
                    labels.append(label)
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(embeddings), np.concatenate(labels)


def save_nasa_regime_probe(
    output_dir,
    model,
    train_entities,
    test_entities,
    window_size,
    max_windows_per_sequence=3000,
):
    """Fit a train-battery linear probe and evaluate it on held-out batteries."""
    device = next(model.parameters()).device
    train_x, train_y = _collect_nasa_embeddings(
        model, train_entities, window_size, max_windows_per_sequence, device
    )
    test_x, test_y = _collect_nasa_embeddings(
        model, test_entities, window_size, max_windows_per_sequence, device
    )
    train_classes = np.unique(train_y)
    if train_x.size == 0 or test_x.size == 0 or train_classes.size < 2:
        report = {"available": False, "reason": "insufficient windows or fewer than two train regimes"}
    else:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=3407)
        classifier.fit(train_x, train_y)
        prediction = classifier.predict(test_x)
        report = {
            "available": True,
            "label_semantics": NASA_REGIME_NAMES,
            "train_window_count": int(len(train_y)),
            "test_window_count": int(len(test_y)),
            "accuracy": float(accuracy_score(test_y, prediction)),
            "macro_f1": float(f1_score(test_y, prediction, average="macro")),
            "train_classes": train_classes.tolist(),
            "test_classes": np.unique(test_y).tolist(),
            "label_source": "current direction; raw step_type_code excluded from model inputs",
            "evaluation": "weak representation sanity probe across held-out batteries",
        }
    path = Path(output_dir) / "nasa_regime_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def save_bms_operational_report(output_dir, test_entities, window_size):
    """Report score/alarm stability by inferred idle/frequency-regulation regime."""
    output_dir = Path(output_dir)
    rows = []
    transition_scores = []
    transition_alarms = []
    # Define the operating regime from the system-level command/current, not
    # from the cluster current that the detector is expected to monitor.
    current_index = BMS_FEATURE_NAMES.index("SYS_I")
    for cluster_name, tensor in test_entities.items():
        score_path = output_dir / cluster_name / "test_output.pkl"
        if not score_path.exists():
            continue
        frame = pd.read_pickle(score_path)
        values = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        regimes = derive_bms_regime_labels(values, current_index=current_index)[window_size:]
        length = min(len(frame), len(regimes))
        frame = frame.iloc[:length]
        regimes = regimes[:length]
        alarms = frame["A_Pred_Global"].to_numpy(dtype=np.int64)
        scores = frame["A_Score_Global"].to_numpy(dtype=np.float64)
        for regime_id, regime_name in BMS_REGIME_NAMES.items():
            mask = regimes == regime_id
            if not np.any(mask):
                continue
            rows.append({
                "cluster": cluster_name,
                "regime": regime_name,
                "window_count": int(mask.sum()),
                "score_mean": float(scores[mask].mean()),
                "score_std": float(scores[mask].std()),
                "alarm_rate": float(alarms[mask].mean()),
            })
        transitions = np.zeros(length, dtype=bool)
        transitions[1:] = regimes[1:] != regimes[:-1]
        radius = min(10, max(1, window_size // 10))
        transition_neighborhood = np.convolve(transitions.astype(np.int32), np.ones(2 * radius + 1), mode="same") > 0
        if np.any(transition_neighborhood):
            transition_scores.extend(scores[transition_neighborhood].tolist())
            transition_alarms.extend(alarms[transition_neighborhood].tolist())

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "bms_operational_stability.csv", index=False)
    report = {
        "label_source": "regimes inferred from current activity; no anomaly ground truth",
        "regime_current_feature": "SYS_I",
        "reported_supervised_fault_metrics": False,
        "cluster_regime_rows": int(len(frame)),
        "transition_window_count": int(len(transition_scores)),
        "transition_score_mean": float(np.mean(transition_scores)) if transition_scores else None,
        "transition_alarm_rate": float(np.mean(transition_alarms)) if transition_alarms else None,
    }
    (output_dir / "bms_operational_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
