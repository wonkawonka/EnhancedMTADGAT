"""样本级 CH-BatteryGen 轻量评分入口，可复用已完成训练而不重复训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.args import apply_dataset_defaults, get_parser
from src.data.ch_battery_utils import (
    CH_BATTERY_DATASET_NAME,
    aggregate_ch_battery_sample_scores,
    get_ch_battery_lfp_discharge_data,
    save_ch_battery_sample_level_reports,
)
from src.engine.prediction import Predictor
from src.models.model_factory import build_model, resolve_model_args, resolve_physical_state_config


def _topk_mean(values, ratio):
    values = np.asarray(values, dtype=np.float32)
    count = max(1, int(np.ceil(values.size * float(ratio))))
    return float(np.mean(np.partition(values, -count)[-count:]))


def _robust_fusion_calibration(model_scores, consistency_scores, max_weight):
    """Fit a label-free, normal-training-only alignment for the C4 residual."""
    def stats(values):
        values = np.asarray(values, dtype=np.float32)
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) + 1e-6)
        stability = scale / (float(np.median(np.abs(values))) + 1e-6)
        return center, scale, stability

    model_center, model_scale, model_stability = stats(model_scores)
    physical_center, physical_scale, physical_stability = stats(consistency_scores)
    weight = model_stability / (model_stability + physical_stability + 1e-6)
    return {
        "fit_split": "normal_training_samples_only",
        "model_center": model_center, "model_scale": model_scale,
        "physical_center": physical_center, "physical_scale": physical_scale,
        "weight": float(np.clip(weight, 0.0, max_weight)),
    }


def _apply_fusion(model_score, consistency_score, calibration):
    aligned = calibration["model_center"] + calibration["model_scale"] * max(
        0.0, (consistency_score - calibration["physical_center"])
        / calibration["physical_scale"],
    )
    weight = calibration["weight"]
    return float((1.0 - weight) * model_score + weight * aligned)


def run_light_predict(model_dir, batch_size=128, num_workers=0, pin_memory=True, no_cuda=False):
    """Load ``model.pt`` and emit sample-level AUROC/AP without writing per-sample window files."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.txt").read_text())
    parser = get_parser()
    # ``config.txt`` is the fully resolved training configuration.  Do not call
    # ``resolve_model_args`` here: that helper uses ``sys.argv`` to decide whether
    # to apply model-family defaults, while this standalone scorer has no training
    # CLI flags in ``sys.argv`` and would silently turn a saved FiLM checkpoint
    # back into a plain baseline.
    args = apply_dataset_defaults(parser.parse_args([], namespace=argparse.Namespace(**config)))
    if args.dataset != CH_BATTERY_DATASET_NAME:
        raise ValueError(f"Expected {CH_BATTERY_DATASET_NAME}, got {args.dataset}")

    (train_map, _), (test_map, _), split_meta = get_ch_battery_lfp_discharge_data(
        root=args.ch_battery_root,
        normalize=args.normalize,
        train_ratio=args.ch_battery_train_ratio,
        seed=args.seed,
        preprocessed_dir=args.ch_battery_preprocessed_dir,
    )
    train_tensors = {key: torch.from_numpy(value).float() for key, value in train_map.items()}
    test_tensors = {key: torch.from_numpy(value).float() for key, value in test_map.items()}
    first_tensor = next(iter(train_tensors.values()))
    n_features = int(first_tensor.shape[1])
    model = build_model(args, n_features, args.lookback, n_features, target_dims=None)
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device).eval()

    prediction_args = {
        "dataset": args.dataset,
        "target_dims": None,
        "scale_scores": args.scale_scores,
        "level": args.level,
        "q": args.q,
        "dynamic_pot": args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "score_fusion_mode": args.score_fusion_mode,
        "use_physical_response_score": False,
        "physical_response_config": resolve_physical_state_config(args),
        "use_relation_change_score": False,
        "predict_batch_size": int(batch_size),
        "predict_num_workers": int(num_workers),
        "predict_pin_memory": bool(pin_memory),
        "use_cuda": device.type == "cuda",
        "window_stride": args.window_stride,
        "use_event_consistency": False,
        "reg_level": 0,
        "save_path": str(model_dir),
    }
    predictor = Predictor(model, args.lookback, n_features, prediction_args)
    def score_collection(tensors, metadata=None):
        rows = []
        for sample_id, sample_tensor in tensors.items():
        # Sample-level AUROC/AP only needs continuous reconstruction/forecast scores.
        # Do not call predict_anomalies here: it copies a 232k-row training-score dataframe for
        # every sample merely to derive threshold predictions, which is irrelevant and very slow.
            window_scores = predictor.get_score(sample_tensor)
            row = {} if metadata is None else dict(metadata[sample_id])
            row.update(aggregate_ch_battery_sample_scores(window_scores, topk_ratio=args.ch_battery_topk_ratio))
            if "Physical_Consistency_Score" in window_scores:
                row["physical_consistency_topk"] = _topk_mean(
                    window_scores["Physical_Consistency_Score"].to_numpy(), args.ch_battery_topk_ratio
                )
            rows.append(row)
        return rows

    rows = score_collection(test_tensors, split_meta["test_metadata"])
    # Only C4 checkpoints expose the independent physical residual.  Its scale is
    # aligned on normal *training* samples only; test labels are never read here.
    if getattr(model, "use_physical_consistency_head", False):
        train_rows = score_collection(train_tensors)
        calibration = _robust_fusion_calibration(
            [row["score_topk_mean"] for row in train_rows],
            [row["physical_consistency_topk"] for row in train_rows],
            args.physical_consistency_score_max_weight,
        )
        for row in rows:
            row["score_topk_mean_backbone"] = row["score_topk_mean"]
            row["score_topk_mean"] = _apply_fusion(
                row["score_topk_mean_backbone"], row["physical_consistency_topk"], calibration
            )
        (model_dir / "physical_consistency_fusion.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8"
        )
    _, summary = save_ch_battery_sample_level_reports(
        model_dir, rows, score_field=args.ch_battery_sample_score
    )
    print(f"[CH-BATTERY] sample-level summary: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    run_light_predict(args.model_dir, args.batch_size, args.num_workers, no_cuda=args.no_cuda)


if __name__ == "__main__":
    main()
