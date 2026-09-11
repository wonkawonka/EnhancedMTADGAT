"""Run every registered external baseline through one dataset/evaluation contract."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.external_protocol import ExternalProtocolData, load_external_protocol_data
from src.data.regime_utils import BMS_REGIME_NAMES, derive_bms_regime_labels
from src.data.utils import BMS_FEATURE_NAMES
from src.models.mtad_gat import Enhanced_MTADGAT
from src.models.unified_external_baselines import (
    DeepSVDDAutoEncoder, build_reconstruction_model, build_reference_model,
)

METHODS = (
    "pca_spe", "usad", "isolation_forest", "auto_encoder", "deep_svdd",
    "lstm_ad", "gdn", "tranad", "mtad_gat", "anomaly_transformer", "dcdetector", "ganf",
)
SOURCE = {
    "pca_spe": "sklearn IncrementalPCA/SPE", "isolation_forest": "sklearn IsolationForest",
    "auto_encoder": "Zhang2023 AE compatibility port", "deep_svdd": "Zhang2023 Deep-SVDD compatibility port",
    "lstm_ad": "PyLink88 recurrent AE compatibility port", "gdn": "Deng2021 learned top-k graph compatibility port",
    "usad": "Audibert2020 two-autoencoder implementation", "tranad": "Tuli2022 self-conditioning compatibility port",
    "mtad_gat": "MTAD-GAT forecasting and reconstruction baseline", "anomaly_transformer": "Anomaly Transformer vendored reference model", "dcdetector": "DCdetector vendored reference model",
    "ganf": "GANF vendored reference model",
}


class WindowDataset(Dataset):
    def __init__(self, sequences, window, stride, limit=0, seed=3407, include_next=False):
        self.sequences = sequences
        self.window = int(window)
        self.span = self.window + int(include_next)
        self.references = [
            (i, start) for i, sequence in enumerate(sequences)
            for start in range(0, max(0, len(sequence) - self.span + 1), int(stride))
        ]
        if limit and len(self.references) > limit:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(self.references), int(limit), replace=False))
            self.references = [self.references[i] for i in selected]

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        sequence_index, start = self.references[index]
        values = self.sequences[sequence_index][start:start + self.span]
        return torch.from_numpy(np.asarray(values, dtype=np.float32)), sequence_index, start + self.span - 1


def _device(args):
    if args.method in {"pca_spe", "isolation_forest"}:
        return torch.device("cpu")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    if args.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Formal deep-baseline plan requested CUDA but CUDA is unavailable")
    return torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


def _parse_feature_dims(value):
    """Parse an optional comma-separated input scope (e.g. ``0`` or ``0,2``)."""
    if value is None or not str(value).strip():
        return None
    try:
        indices = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"feature_dims must be comma-separated integers, got {value!r}") from exc
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError(f"feature_dims must be unique non-negative integers, got {indices}")
    return indices


def _window_and_stride(args):
    native_windows = {"tranad": 10, "gdn": 32}
    if args.dataset.upper() == "BRAND3" and args.method in {
        "isolation_forest", "auto_encoder", "deep_svdd", "lstm_ad"
    }:
        native_windows[args.method] = 128
    window = native_windows.get(args.method, int(args.lookback))
    stride = int(args.window_stride)
    if stride <= 0:
        stride = 10 if args.dataset.upper() in {"BMS", "SWAT", "WADI"} else 1
    return window, stride


def _filter_short_sequences(data, span):
    """Apply one explicit minimum-length rule before any window scoring."""
    def keep(sequences, identifiers, labels=None):
        selected = [index for index, sequence in enumerate(sequences) if len(sequence) >= span]
        result_sequences = [sequences[index] for index in selected]
        result_ids = [identifiers[index] for index in selected]
        result_labels = None if labels is None else [labels[index] for index in selected]
        return result_sequences, result_ids, result_labels, len(sequences) - len(selected)
    data.train_sequences, _, _, train_dropped = keep(data.train_sequences, [str(i) for i in range(len(data.train_sequences))])
    data.validation_sequences, data.validation_entity_ids, data.validation_labels, validation_dropped = keep(data.validation_sequences, data.validation_entity_ids, data.validation_labels)
    data.test_sequences, data.entity_ids, data.test_labels, test_dropped = keep(data.test_sequences, data.entity_ids, data.test_labels)
    if not data.train_sequences or not data.validation_sequences or not data.test_sequences:
        raise ValueError(f"No usable sequences remain after minimum span={span} filtering")
    data.metadata['minimum_sequence_span'] = int(span)
    data.metadata['dropped_short_sequences'] = {'train': train_dropped, 'validation': validation_dropped, 'test': test_dropped}


def _loaders(data, args, window, stride, include_next=False):
    common = dict(window=window, stride=stride, seed=args.seed, include_next=include_next)
    train = WindowDataset(data.train_sequences, limit=args.train_sample_limit, **common)
    validation = WindowDataset(data.validation_sequences, limit=args.evaluation_sample_limit, **common)
    test = WindowDataset(data.test_sequences, limit=args.evaluation_sample_limit, **common)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.use_cuda)
    return (
        DataLoader(train, shuffle=True, drop_last=len(train) > args.batch_size, **kwargs),
        DataLoader(validation, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )


def _kl(left, right):
    return torch.sum(left * (torch.log(left + 1e-4) - torch.log(right + 1e-4)), dim=-1)


def _association_losses(series, prior, window):
    series_loss = 0.0
    prior_loss = 0.0
    for current_series, current_prior in zip(series, prior):
        normalized_prior = current_prior / torch.clamp(current_prior.sum(dim=-1, keepdim=True), min=1e-8)
        series_loss = series_loss + (_kl(current_series, normalized_prior.detach()).mean() + _kl(normalized_prior.detach(), current_series).mean())
        prior_loss = prior_loss + (_kl(normalized_prior, current_series.detach()).mean() + _kl(current_series.detach(), normalized_prior).mean())
    return series_loss / len(prior), prior_loss / len(prior)


def _fit_pca(data, args):
    values = np.concatenate(data.train_sequences)
    if args.train_sample_limit and len(values) > args.train_sample_limit:
        values = values[np.random.default_rng(args.seed).choice(len(values), args.train_sample_limit, replace=False)]
    count = min(args.pca_components, values.shape[1], len(values))
    model = IncrementalPCA(n_components=count, batch_size=max(256, count * 4))
    for start in range(0, len(values), 4096):
        batch = values[start:start + 4096]
        if len(batch) >= count:
            model.partial_fit(batch)
    return model


def _fit_isolation(train_loader, args):
    batches = [values.flatten(1).numpy() for values, _, _ in train_loader]
    values = np.concatenate(batches)
    return IsolationForest(n_estimators=args.isolation_trees, max_samples=min(256, len(values)), random_state=args.seed, n_jobs=-1).fit(values)


def _fit_isolation_points(data, args):
    values = np.concatenate(data.train_sequences)
    if args.train_sample_limit and len(values) > args.train_sample_limit:
        values = values[np.random.default_rng(args.seed).choice(len(values), args.train_sample_limit, replace=False)]
    return IsolationForest(
        n_estimators=args.isolation_trees, max_samples=min(256, len(values)),
        random_state=args.seed, n_jobs=-1,
    ).fit(values)


def _score_isolation_points(model, sequences):
    return [-model.score_samples(sequence) for sequence in sequences]


def _point_scores_to_windows(point_scores, window, stride):
    scores, endpoints = [], []
    for sequence_scores in point_scores:
        starts = range(0, max(0, len(sequence_scores) - window + 1), stride)
        starts = list(starts)
        scores.append(np.asarray([np.mean(sequence_scores[start:start + window]) for start in starts]))
        endpoints.append(np.asarray([start + window - 1 for start in starts], dtype=np.int64))
    return scores, endpoints


def _fit_torch(data, args, train_loader, window, device):
    features = data.train_sequences[0].shape[1]
    if args.method == "deep_svdd":
        pretrain = DeepSVDDAutoEncoder(window * features, args.hidden_dim, args.latent_dim).to(device)
        pretrain_optimizer = torch.optim.Adam(pretrain.parameters(), lr=args.learning_rate)
        for epoch in range(int(args.svdd_pretrain_epochs)):
            pretrain.train(); total = 0.0; batches = 0
            for values, _, _ in train_loader:
                values = values.to(device, non_blocking=True)
                pretrain_optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(pretrain(values), values)
                loss.backward(); pretrain_optimizer.step()
                total += float(loss.detach()); batches += 1
            print(f"[deep_svdd_pretrain] epoch={epoch + 1}/{args.svdd_pretrain_epochs} loss={total / max(1, batches):.6f}")
        model = pretrain.encoder
        with torch.no_grad():
            centers = [model(values.to(device)).cpu() for values, _, _ in train_loader]
        center = torch.cat(centers).mean(0).to(device)
        center[(center.abs() < 0.1) & (center < 0)] = -0.1
        center[(center.abs() < 0.1) & (center >= 0)] = 0.1
        auxiliary = center
    elif args.method == "mtad_gat":
        model = Enhanced_MTADGAT(
            n_features=features, window_size=window, out_dim=features,
            gru_hid_dim=args.hidden_dim, forecast_hid_dim=args.hidden_dim,
            recon_hid_dim=args.hidden_dim, dropout=0.2,
            use_regime_condition=args.mtad_use_regime_condition,
            regime_encoder_type="restricted", regime_emb_dim=args.mtad_regime_emb_dim,
            regime_control_indices=args.mtad_regime_control_indices,
        ).to(device)
        model.score_dims = args.mtad_score_dims or list(range(features))
        auxiliary = None
    elif args.method in {"anomaly_transformer", "dcdetector", "ganf"}:
        model = build_reference_model(args.method, features, window).to(device)
        auxiliary = None
    else:
        model = build_reconstruction_model(args.method, features, window, args.hidden_dim, args.latent_dim).to(device)
        auxiliary = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(int(args.epochs)):
        model.train()
        total = 0.0
        batches = 0
        for values, _, _ in train_loader:
            values = values.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.method == "deep_svdd":
                loss = (model(values) - auxiliary).square().sum(dim=1).mean()
            elif args.method == "gdn":
                loss = nn.functional.mse_loss(model(values[:, :-1]), values[:, -1])
            elif args.method == "usad":
                flat = values.flatten(1); first, second = model
                rec1, rec2 = first(flat), second(flat)
                chained = second(rec1)
                weight = float(epoch + 1) / max(1, int(args.epochs))
                loss = weight * (nn.functional.mse_loss(rec1, flat) + nn.functional.mse_loss(rec2, flat))
                loss = loss + (1.0 - weight) * nn.functional.mse_loss(chained, flat)
            elif args.method == "tranad":
                first, second = model(values)
                target = values[:, -1:]
                loss = 0.5 * nn.functional.mse_loss(first, target) + 0.5 * nn.functional.mse_loss(second, target)
            elif args.method == "mtad_gat":
                forecast, reconstruction = model(values)
                loss = 0.5 * nn.functional.mse_loss(forecast, values[:, -1]) + 0.5 * nn.functional.mse_loss(reconstruction, values)
                if args.mtad_use_regime_condition:
                    loss = loss + args.mtad_regime_aux_lambda * model.regime_auxiliary_loss(values)
            elif args.method == "anomaly_transformer":
                output, series, prior, _ = model(values)
                series_loss, prior_loss = _association_losses(series, prior, window)
                reconstruction = nn.functional.mse_loss(output, values)
                loss = reconstruction - args.association_weight * series_loss + args.association_weight * prior_loss
            elif args.method == "dcdetector":
                series, prior = model(values)
                series_loss, prior_loss = _association_losses(series, prior, window)
                loss = prior_loss - series_loss
            elif args.method == "ganf":
                adjacency = model.adjacency
                acyclicity = torch.trace(torch.matrix_exp(adjacency * adjacency)) - adjacency.shape[0]
                loss = -model(values) + args.graph_lambda * adjacency.abs().mean()
                loss = loss + 0.5 * args.graph_rho * acyclicity.square()
            else:
                loss = nn.functional.mse_loss(model(values), values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(f"[{args.method}] epoch={epoch + 1}/{args.epochs} loss={total / max(1, batches):.6f}")
    return model, auxiliary


def _score_pca(model, sequences):
    result = []
    for sequence in sequences:
        transformed = model.transform(sequence)
        result.append(np.mean((sequence - model.inverse_transform(transformed)) ** 2, axis=1))
    return result


def _score_pca_loader(model, loader, sequence_count):
    scores = [[] for _ in range(sequence_count)]; endpoints = [[] for _ in range(sequence_count)]
    for values, sequence_ids, end_indices in loader:
        arrays = values.numpy()
        batch_scores = []
        for window in arrays:
            transformed = model.transform(window)
            batch_scores.append(float(np.mean((window - model.inverse_transform(transformed)) ** 2)))
        for score, sequence_id, endpoint in zip(batch_scores, sequence_ids.numpy(), end_indices.numpy()):
            scores[int(sequence_id)].append(score); endpoints[int(sequence_id)].append(int(endpoint))
    return [np.asarray(x) for x in scores], [np.asarray(x, dtype=np.int64) for x in endpoints]


def _batch_scores(method, model, auxiliary, values, window):
    if method == "isolation_forest":
        return -model.score_samples(values.flatten(1).numpy())
    if method == "deep_svdd":
        return (model(values) - auxiliary).square().sum(dim=1).detach().cpu().numpy()
    if method == "gdn":
        return (model(values[:, :-1]) - values[:, -1]).square().mean(dim=1).detach().cpu().numpy()
    if method == "usad":
        flat = values.flatten(1); first, second = model
        return (0.5 * ((first(flat) - flat).square().mean(1) + (second(flat) - flat).square().mean(1))).detach().cpu().numpy()
    if method == "tranad":
        _, output = model(values)
        return (output - values[:, -1:]).square().mean((1, 2)).detach().cpu().numpy()
    if method == "mtad_gat":
        forecast, reconstruction = model(values)
        score_dims = getattr(model, "score_dims", None)
        if score_dims is None:
            score_dims = list(range(values.shape[-1]))
        forecast_error = (forecast[:, score_dims] - values[:, -1, score_dims]).square().mean(1)
        reconstruction_error = (reconstruction[:, :, score_dims] - values[:, :, score_dims]).square().mean((1, 2))
        return (0.5 * (forecast_error + reconstruction_error)).detach().cpu().numpy()
    if method == "anomaly_transformer":
        output, series, prior, _ = model(values)
        rec = (output - values).square().mean(-1)
        association = 0.0
        for current_series, current_prior in zip(series, prior):
            normalized = current_prior / torch.clamp(current_prior.sum(-1, keepdim=True), min=1e-8)
            association = association + _kl(current_series, normalized.detach()) + _kl(normalized, current_series.detach())
        metric = torch.softmax(-association.mean(dim=1), dim=-1)
        return (metric * rec)[:, -1].detach().cpu().numpy()
    if method == "dcdetector":
        series, prior = model(values); association = 0.0
        for current_series, current_prior in zip(series, prior):
            normalized = current_prior / torch.clamp(current_prior.sum(-1, keepdim=True), min=1e-8)
            association = association + _kl(current_series, normalized.detach()) + _kl(normalized, current_series.detach())
        return torch.softmax(-association.mean(dim=1), dim=-1)[:, -1].detach().cpu().numpy()
    if method == "ganf":
        return model.score(values).detach().cpu().numpy()
    return (model(values) - values).square().mean((1, 2)).detach().cpu().numpy()


def _score_loader(method, model, auxiliary, loader, device, sequence_count):
    scores = [[] for _ in range(sequence_count)]
    endpoints = [[] for _ in range(sequence_count)]
    if isinstance(model, nn.Module): model.eval()
    with torch.no_grad():
        for values, sequence_ids, end_indices in loader:
            source = values if method == "isolation_forest" else values.to(device, non_blocking=True)
            batch_scores = _batch_scores(method, model, auxiliary, source, values.shape[1] - int(method == "gdn"))
            for score, sequence_id, endpoint in zip(batch_scores, sequence_ids.numpy(), end_indices.numpy()):
                scores[int(sequence_id)].append(float(score)); endpoints[int(sequence_id)].append(int(endpoint))
    return [np.asarray(x) for x in scores], [np.asarray(x, dtype=np.int64) for x in endpoints]


def _aggregate_vehicle(scores, entities, labels, ratio):
    grouped = defaultdict(list); grouped_labels = {}
    for score, entity, label in zip(scores, entities, labels):
        grouped[entity].append(float(np.mean(score)))
        grouped_labels[entity] = int(np.asarray(label).reshape(-1)[0])
    entity_scores, entity_labels = [], []
    for entity in sorted(grouped):
        values = np.sort(np.asarray(grouped[entity]))
        count = max(1, int(np.ceil(len(values) * ratio)))
        entity_scores.append(float(np.mean(values[-count:])))
        entity_labels.append(grouped_labels[entity])
    return np.asarray(entity_scores), np.asarray(entity_labels, dtype=np.int32)


def _paper_labelled_threshold(scores, labels, granularity=1000):
    """Match the paper-compatible ranked calibration threshold used internally."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError("Labelled Brand3 calibration scores and labels must be non-empty and aligned")
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    best_fraction, best_count = -1.0, 1
    for rank in range(1, 100):
        count = max(round(len(ranked_scores) * rank / granularity), 1)
        fraction = float(np.mean(ranked_labels[:count]))
        if fraction > best_fraction:
            best_fraction, best_count = fraction, count
    return float(ranked_scores[best_count - 1])


def _brand3_top_ratio(validation_scores, validation_entities, validation_labels, args, protocol):
    mode = str(args.brand_top_ratio_mode).lower()
    if mode == "fixed":
        return float(args.vehicle_top_ratio), "predefined_fixed_top_ratio"
    if protocol != "paper_protocol":
        raise ValueError(
            "labelled_calibration Top-p selection is only valid for Brand3 paper_protocol; "
            "use --brand_top_ratio_mode fixed for strict_normal_validation"
        )
    candidates = range(5, 100, 5)
    candidate_metrics = []
    for percent in candidates:
        candidate_scores, candidate_labels = _aggregate_vehicle(
            validation_scores,
            validation_entities,
            validation_labels,
            percent / 100.0,
        )
        if np.unique(candidate_labels).size < 2:
            raise ValueError("Brand3 paper calibration must contain both normal and faulty vehicles")
        candidate_metrics.append((roc_auc_score(candidate_labels, candidate_scores), percent))
    best_percent = max(candidate_metrics)[1]
    return best_percent / 100.0, "labelled_calibration_vehicle_auroc"


def _topk_sample_score(window_scores, ratio=0.05):
    values = np.asarray(window_scores, dtype=np.float32)
    if not len(values):
        raise ValueError("CH-BatteryGen sequence has no valid windows; lower lookback or inspect segment length")
    count = max(1, int(np.ceil(len(values) * float(ratio))))
    return float(np.mean(np.partition(values, -count)[-count:])), count


def _ch_aggregate_score(window_scores, mode):
    values = np.asarray(window_scores, dtype=np.float32)
    if not len(values):
        raise ValueError("CH-BatteryGen sequence has no valid windows; lower lookback or inspect segment length")
    if mode == "max":
        return float(np.max(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "top5pct":
        return _topk_sample_score(values, 0.05)[0]
    raise ValueError(f"Unknown CH aggregation {mode}")


def _ch_sensitivity_rows(data, train_scores, validation_scores, test_scores):
    """Compare aggregation and normal-only calibration choices without test labels."""
    labels = np.asarray([int(np.asarray(x).reshape(-1)[0]) for x in data.test_labels], dtype=np.int32)
    rows = []
    for aggregation in ("max", "mean", "top5pct"):
        train = np.asarray([_ch_aggregate_score(x, aggregation) for x in train_scores])
        validation = np.asarray([_ch_aggregate_score(x, aggregation) for x in validation_scores])
        test = np.asarray([_ch_aggregate_score(x, aggregation) for x in test_scores])
        for calibration_set, calibration in (("training_normal", train), ("validation_normal", validation)):
            for quantile in (0.95, 0.99):
                threshold = float(np.quantile(calibration, quantile))
                pred = test >= threshold
                tp = int(np.sum(pred & (labels == 1))); fp = int(np.sum(pred & (labels == 0)))
                fn = int(np.sum(~pred & (labels == 1))); tn = int(np.sum(~pred & (labels == 0)))
                precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
                rows.append({"aggregation": aggregation, "calibration_set": calibration_set,
                             "quantile": quantile, "threshold": threshold,
                             "sample_auroc": float(roc_auc_score(labels, test)),
                             "sample_auprc": float(average_precision_score(labels, test)),
                             "precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / max(1e-8, precision + recall),
                             "fpr": fp / max(1, fp + tn),
                             "threshold_status": "normal_only_calibration"})
    return rows


def _ch_sample_metrics(data, train_scores, test_scores, args):
    ratio = float(data.metadata.get("topk_ratio", 0.05))
    calibration = np.asarray([_topk_sample_score(values, ratio)[0] for values in train_scores], dtype=np.float32)
    threshold = float(np.quantile(calibration, args.threshold_quantile))
    rows = []
    for sample_id, windows, label_array in zip(data.entity_ids, test_scores, data.test_labels):
        score, count = _topk_sample_score(windows, ratio)
        row = dict(data.metadata["sample_metadata"][sample_id])
        row.update(score_topk_mean=score, window_count=int(len(windows)), topk_count=count,
                   sample_label=int(np.asarray(label_array).reshape(-1)[0]), prediction=int(score >= threshold))
        rows.append(row)
    labels = np.asarray([row["sample_label"] for row in rows], dtype=np.int32)
    scores = np.asarray([row["score_topk_mean"] for row in rows], dtype=np.float32)
    prediction = scores >= threshold
    tp = int(np.sum(prediction & (labels == 1))); fp = int(np.sum(prediction & (labels == 0)))
    fn = int(np.sum(~prediction & (labels == 1))); tn = int(np.sum(~prediction & (labels == 0)))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    result = {"dataset": data.dataset, "method": args.method, "seed": args.seed,
              "evaluation_kind": "sample_ranking", "sample_count": int(len(rows)),
              "threshold": threshold, "threshold_source": f"training_normal_sample_top5pct_q{args.threshold_quantile:.4f}",
              "sample_auroc": float(roc_auc_score(labels, scores)), "sample_auprc": float(average_precision_score(labels, scores)),
              "normal_training_p99_precision": precision, "normal_training_p99_recall": recall,
              "normal_training_p99_f1": 2 * precision * recall / max(1e-8, precision + recall),
              "normal_training_p99_fpr": fp / max(1, fp + tn),
              "fault_count": int(np.sum(labels)), "normal_count": int(np.sum(labels == 0)),
              "topk_ratio": ratio, "_sample_rows": rows}
    return result


def _final_metrics(data, validation_scores, validation_endpoints, test_scores, test_endpoints, args, train_scores=None):
    if data.evaluation_kind == "sample_ranking":
        if train_scores is None:
            raise ValueError("sample-level CH evaluation requires training normal scores")
        result = _ch_sample_metrics(data, train_scores, test_scores, args)
        result["_sensitivity_rows"] = _ch_sensitivity_rows(data, train_scores, validation_scores, test_scores)
        return result
    if data.evaluation_kind == "point_ranking":
        validation_flat = np.concatenate(validation_scores)
        scores = np.concatenate(test_scores)
        labels = np.concatenate([
            np.asarray(label)[endpoints] for label, endpoints in zip(data.test_labels, test_endpoints)
        ]).astype(np.int32)
    elif data.evaluation_kind == "vehicle_ranking":
        ratio, top_ratio_source = _brand3_top_ratio(
            validation_scores,
            data.validation_entity_ids,
            data.validation_labels,
            args,
            data.metadata.get("protocol"),
        )
        validation_flat, _ = _aggregate_vehicle(
            validation_scores, data.validation_entity_ids, data.validation_labels, ratio
        )
        scores, labels = _aggregate_vehicle(test_scores, data.entity_ids, data.test_labels, ratio)
    else:
        validation_flat = np.concatenate(validation_scores)
        scores = np.concatenate(test_scores)
        labels = None
    paper_brand3 = (
        data.evaluation_kind == "vehicle_ranking"
        and data.metadata.get("protocol") == "paper_protocol"
    )
    if paper_brand3:
        calibration_window_scores = np.concatenate(validation_scores)
        calibration_window_labels = np.concatenate([
            np.full(len(score), int(np.asarray(label).reshape(-1)[0]), dtype=np.int32)
            for score, label in zip(validation_scores, data.validation_labels)
        ])
        threshold = _paper_labelled_threshold(
            calibration_window_scores, calibration_window_labels
        )
        threshold_source = "paper_labelled_snippet_rank"
    else:
        threshold = float(np.quantile(validation_flat, args.threshold_quantile))
        threshold_source = f"normal_validation_q{args.threshold_quantile:.4f}"
    result = {
        "dataset": data.dataset, "method": args.method, "seed": args.seed,
        "threshold": threshold, "threshold_source": threshold_source,
        "evaluation_kind": data.evaluation_kind, "sample_count": int(len(scores)),
    }
    if data.evaluation_kind == "vehicle_ranking":
        result["vehicle_top_ratio"] = ratio
        result["top_ratio_selection"] = top_ratio_source
    predictions = scores > threshold
    if labels is None:
        rate = float(np.mean(predictions))
        entity_rows = []
        for entity, entity_scores in zip(data.entity_ids, test_scores):
            entity_rate = float(np.mean(entity_scores > threshold))
            entity_rows.append({"entity": entity, "window_count": int(len(entity_scores)), "false_alarm_rate": entity_rate})
        block_rates = []
        for entity_scores in test_scores:
            for start in range(0, len(entity_scores), args.stability_block_size):
                block = entity_scores[start:start + args.stability_block_size]
                if len(block): block_rates.append(float(np.mean(block > threshold)))
        regime_counts = defaultdict(lambda: [0, 0])
        current_index = BMS_FEATURE_NAMES.index("BMSnI")
        for sequence, endpoints, entity_scores in zip(data.test_sequences, test_endpoints, test_scores):
            regimes = derive_bms_regime_labels(sequence, current_index)[endpoints]
            for regime, score in zip(regimes, entity_scores):
                bucket = regime_counts[int(regime)]; bucket[0] += int(score > threshold); bucket[1] += 1
        result.update(
            false_alarm_rate=rate, false_alarms_per_10k_windows=rate * 10000.0,
            false_alarm_by_entity=entity_rows,
            entity_fpr_std=float(np.std([row["false_alarm_rate"] for row in entity_rows])),
            block_fpr_std=float(np.std(block_rates)) if block_rates else None,
            false_alarm_by_regime=[{
                "regime": BMS_REGIME_NAMES.get(regime, str(regime)), "false_alarms": counts[0],
                "window_count": counts[1], "false_alarm_rate": counts[0] / max(1, counts[1]),
            } for regime, counts in sorted(regime_counts.items())],
        )
    else:
        classification = {
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
        }
        prefix = "vehicle_" if data.evaluation_kind == "vehicle_ranking" else ""
        result.update(average_precision=float(average_precision_score(labels, scores)),
                      auroc=float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else None,
                      anomaly_count=int(np.sum(labels > 0)))
        result.update({f"{prefix}{key}_raw": value for key, value in classification.items()})
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Unified formal external baseline runner")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=("MSL", "SMAP", "SWAT", "WADI", "Brand3", "BMS", "CH_LFP_DISCHARGE", "CH_NCM_DISCHARGE", "CH_LFP_CHARGE", "CH_NCM_CHARGE"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--brand_fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--brand_split_protocol",
        choices=("paper_protocol", "strict_normal_validation"),
        default="paper_protocol",
    )
    parser.add_argument("--brand_fold_seed", type=int, default=0)
    parser.add_argument(
        "--brand_normalization",
        choices=("paper_channel", "minmax"),
        default="paper_channel",
    )
    parser.add_argument(
        "--brand_top_ratio_mode",
        choices=("labelled_calibration", "fixed"),
        default="labelled_calibration",
    )
    parser.add_argument(
        "--feature_dims",
        default="",
        help="Optional comma-separated input dimensions. Empty keeps all features; use 0 for dim0-only.",
    )
    parser.add_argument("--ch_train_cycle_kind", choices=("charge", "discharge"), default=None)
    parser.add_argument("--ch_test_cycle_kind", choices=("charge", "discharge"), default=None)
    parser.add_argument("--ch_test_chemistry", choices=("LFP", "NCM"), default=None)
    parser.add_argument("--ch_random_mask_ratio", type=float, default=0.0,
                        help="CH-only MAR value-mask ratio after train-normal-only scaling; masked values are zero-imputed.")
    parser.add_argument("--ch_resample_factor", type=int, default=1,
                        help="CH-only fixed-stride temporal decimation after scaling; 1 keeps the native sampling.")
    parser.add_argument("--ch_append_mask_indicators", action="store_true",
                        help="CH-only: append binary MAR missingness indicators to each input channel.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--vehicle_top_ratio", type=float, default=0.05)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--window_stride", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--mtad_use_regime_condition", action="store_true")
    parser.add_argument("--mtad_regime_emb_dim", type=int, default=8)
    parser.add_argument("--mtad_regime_aux_lambda", type=float, default=0.05)
    parser.add_argument("--mtad_regime_control_indices", type=lambda x: [int(v) for v in x.split(',')], default=None)
    parser.add_argument("--mtad_score_dims", type=lambda x: [int(v) for v in x.split(',')], default=None)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--pca_components", type=int, default=8)
    parser.add_argument("--isolation_trees", type=int, default=200)
    parser.add_argument("--train_sample_limit", type=int, default=100000)
    parser.add_argument("--evaluation_sample_limit", type=int, default=0, help="Smoke-test only; 0 evaluates all windows")
    parser.add_argument("--association_weight", type=float, default=3.0)
    parser.add_argument("--svdd_pretrain_epochs", type=int, default=5)
    parser.add_argument("--graph_lambda", type=float, default=0.0)
    parser.add_argument("--graph_rho", type=float, default=1.0)
    parser.add_argument("--stability_block_size", type=int, default=1000)
    parser.add_argument("--normalize", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--use_cuda", type=lambda value: str(value).lower() == "true", default=True)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_indices = _parse_feature_dims(args.feature_dims)
    if not args.normalize:
        raise ValueError("The formal unified protocol requires training-only normalization")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = _device(args)
    started = time.perf_counter()
    data = load_external_protocol_data(
        args.dataset,
        brand_fold=args.brand_fold,
        seed=args.seed,
        val_ratio=args.val_ratio,
        brand_split_protocol=args.brand_split_protocol,
        brand_fold_seed=args.brand_fold_seed,
        feature_indices=feature_indices,
        brand_normalization=args.brand_normalization,
        ch_train_cycle_kind=args.ch_train_cycle_kind,
        ch_test_cycle_kind=args.ch_test_cycle_kind,
        ch_test_chemistry=args.ch_test_chemistry,
        ch_random_mask_ratio=args.ch_random_mask_ratio,
        ch_append_mask_indicators=args.ch_append_mask_indicators,
        ch_resample_factor=args.ch_resample_factor,
    )
    window, stride = _window_and_stride(args)
    include_next = args.method == "gdn"
    _filter_short_sequences(data, window + int(include_next))
    train_loader, validation_loader, test_loader = _loaders(data, args, window, stride, include_next)
    train_score_loader = DataLoader(WindowDataset(data.train_sequences, window=window, stride=stride, seed=args.seed, include_next=include_next), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.use_cuda)
    load_seconds = time.perf_counter() - started
    training_started = time.perf_counter()
    auxiliary = None
    if args.method == "pca_spe":
        model = _fit_pca(data, args)
    elif args.method == "isolation_forest":
        model = _fit_isolation(train_loader, args) if data.evaluation_kind == "vehicle_ranking" else _fit_isolation_points(data, args)
    else:
        model, auxiliary = _fit_torch(data, args, train_loader, window, device)
    training_seconds = time.perf_counter() - training_started
    inference_started = time.perf_counter()
    if args.method == "isolation_forest" and data.evaluation_kind != "vehicle_ranking":
        validation_point_scores = _score_isolation_points(model, data.validation_sequences)
        test_point_scores = _score_isolation_points(model, data.test_sequences)
        if data.evaluation_kind == "normal_only":
            validation_scores, validation_endpoints = _point_scores_to_windows(validation_point_scores, window, stride)
            test_scores, test_endpoints = _point_scores_to_windows(test_point_scores, window, stride)
        else:
            validation_scores, test_scores = validation_point_scores, test_point_scores
            validation_endpoints = [np.arange(len(x)) for x in validation_scores]
            test_endpoints = [np.arange(len(x)) for x in test_scores]
    elif args.method == "pca_spe" and data.evaluation_kind != "normal_only":
        validation_scores = _score_pca(model, data.validation_sequences)
        test_scores = _score_pca(model, data.test_sequences)
        validation_endpoints = [np.arange(len(x)) for x in validation_scores]
        test_endpoints = [np.arange(len(x)) for x in test_scores]
    elif args.method == "pca_spe":
        validation_scores, validation_endpoints = _score_pca_loader(model, validation_loader, len(data.validation_sequences))
        test_scores, test_endpoints = _score_pca_loader(model, test_loader, len(data.test_sequences))
    else:
        validation_scores, validation_endpoints = _score_loader(args.method, model, auxiliary, validation_loader, device, len(data.validation_sequences))
        test_scores, test_endpoints = _score_loader(args.method, model, auxiliary, test_loader, device, len(data.test_sequences))
    train_scores = None
    if data.evaluation_kind == "sample_ranking":
        if args.method == "isolation_forest":
            train_point_scores = _score_isolation_points(model, data.train_sequences)
            train_scores, _ = _point_scores_to_windows(train_point_scores, window, stride)
        elif args.method == "pca_spe":
            train_scores, _ = _score_pca_loader(model, train_score_loader, len(data.train_sequences))
        else:
            train_scores, _ = _score_loader(args.method, model, auxiliary, train_score_loader, device, len(data.train_sequences))
    inference_seconds = time.perf_counter() - inference_started
    metrics = _final_metrics(data, validation_scores, validation_endpoints, test_scores, test_endpoints, args, train_scores=train_scores)
    sample_rows = metrics.pop("_sample_rows", None)
    sensitivity_rows = metrics.pop("_sensitivity_rows", None)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if sample_rows is not None:
        import pandas as pd
        pd.DataFrame(sample_rows).sort_values(["score_topk_mean", "sample_id"], ascending=[False, True]).to_csv(output / "ch_battery_sample_scores.csv", index=False)
    if sensitivity_rows is not None:
        import pandas as pd
        pd.DataFrame(sensitivity_rows).to_csv(output / "aggregation_threshold_sensitivity.csv", index=False)
        (output / "aggregation_threshold_sensitivity.json").write_text(json.dumps({
            "purpose": "Aggregation and normal-only calibration sensitivity; not a model-ranking table.",
            "validation_optimal_threshold": "not computed: validation contains normal VINs only; label-optimized thresholds would be oracle diagnostics.",
            "rows": sensitivity_rows,
        }, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), encoding="utf-8")
    runtime = {
        "device": str(device), "data_loading_seconds": load_seconds, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "model_parameters": int(sum(p.numel() for p in model.parameters())) if isinstance(model, nn.Module) else 0,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
    }
    config = {**vars(args), "window": window, "resolved_stride": stride, "source": SOURCE[args.method], "data_metadata": data.metadata}
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    test_sequence_ids = np.concatenate([
        np.full(len(scores), index, dtype=np.int32) for index, scores in enumerate(test_scores)
    ])
    test_entities = np.concatenate([
        np.full(len(scores), data.entity_ids[index], dtype=str) for index, scores in enumerate(test_scores)
    ])
    if data.test_labels is None:
        score_labels = np.empty(0, dtype=np.int32)
    elif data.evaluation_kind == "point_ranking":
        score_labels = np.concatenate([
            np.asarray(label)[endpoints] for label, endpoints in zip(data.test_labels, test_endpoints)
        ]).astype(np.int32)
    else:
        score_labels = np.concatenate([
            np.full(len(scores), int(np.asarray(data.test_labels[index]).reshape(-1)[0]), dtype=np.int32)
            for index, scores in enumerate(test_scores)
        ])
    if data.evaluation_kind == "sample_ranking":
        np.savez_compressed(output / "train_scores.npz", scores=np.concatenate(train_scores),
                            sequence_ids=np.concatenate([np.full(len(x), i, dtype=np.int32) for i, x in enumerate(train_scores)]))
        np.savez_compressed(output / "validation_scores.npz", scores=np.concatenate(validation_scores),
                            sequence_ids=np.concatenate([np.full(len(x), i, dtype=np.int32) for i, x in enumerate(validation_scores)]))
    np.savez_compressed(
        output / "scores.npz", scores=np.concatenate(test_scores), endpoints=np.concatenate(test_endpoints),
        sequence_ids=test_sequence_ids, entity_ids=test_entities, labels=score_labels,
    )
    if isinstance(model, nn.Module):
        model_filename = "model.pt"
        torch.save(model.state_dict(), output / model_filename)
    else:
        model_filename = "model.pkl"
        with (output / model_filename).open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "model_artifact.json").write_text(
        json.dumps({"path": model_filename, "source": SOURCE[args.method]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"metrics": metrics, "runtime": runtime}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
