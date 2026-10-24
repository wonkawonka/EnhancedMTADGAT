"""Controlled CH experiment: MTAD-GAT baseline vs simplified C3 plus RCR.

The simplified C3 keeps marginal current/SOC state conditioning and bounded
fusion FiLM, while removing prototype queries, routing, and the descriptor
auxiliary loss.  RCR linear conditional residual calibration is fitted only on
normal training windows.  Validation normals select checkpoints and thresholds;
test labels are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src.data.ch_battery_utils import load_ch_battery_mixed_cycle_research_split
from src.models.mtad_gat import Enhanced_MTADGAT
from src.models.rcr import EmpiricalCDF, LinearResidualCalibration, condition_features
from src.runners.run_rcr import atomic_checkpoint, atomic_json, state_hash
from src.runners.run_unified_external_baselines import WindowDataset


def make_data(split, window, stride, seed, train_limit=0, eval_limit=0):
    data, ids = {}, {}
    for partition in ("train", "validation", "test"):
        ids[partition] = sorted(k for k, v in split[partition].items() if len(v) >= window + 1)
        data[partition] = WindowDataset(
            [split[partition][k] for k in ids[partition]], window + 1, stride,
            limit=train_limit if partition == "train" else eval_limit, seed=seed
        )
    return data, ids


def build_model(conditioned, window):
    return Enhanced_MTADGAT(
        n_features=7, window_size=window, out_dim=7, dropout=0.3,
        use_regime_condition=conditioned, regime_encoder_type="restricted",
        regime_emb_dim=8, regime_control_indices=[1, 2],
        regime_channel_pooling=False, regime_film_scale=0.1,
    )


def loss_for(model, batch):
    sequence = batch[0].cuda(non_blocking=True)
    x, y = sequence[:, :-1], sequence[:, -1]
    forecast, reconstruction = model(x)
    return F.mse_loss(forecast, y) + F.mse_loss(reconstruction, x)


def train(model, data, output, epochs, seed):
    best_path, last_path = output / "best.pt", output / "last.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(data["train"], 128, shuffle=True, generator=generator,
                            pin_memory=True, num_workers=2, persistent_workers=True),
        "validation": DataLoader(data["validation"], 128, shuffle=False,
                                 pin_memory=True, num_workers=2, persistent_workers=True),
    }
    start, best, stale, best_epoch, history = 1, float("inf"), 0, 0, []
    if (output / "training_complete.json").exists():
        model.load_state_dict(torch.load(best_path, map_location="cuda", weights_only=True))
        return model.eval()
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start, best, stale, best_epoch, history = (
            saved["epoch"] + 1, saved["best"], saved["stale"], saved["best_epoch"], saved["history"]
        )
        torch.set_rng_state(saved["rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
    for epoch in range(start, epochs + 1):
        if stale >= 8:
            break
        clock = time.monotonic(); model.train(); total = count = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CH hybrid training loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(batch[0]); count += len(batch[0])
        model.eval(); val_total = val_count = 0
        with torch.inference_mode():
            for batch in loaders["validation"]:
                value = loss_for(model, batch)
                val_total += float(value) * len(batch[0]); val_count += len(batch[0])
        validation = val_total / val_count
        if validation < best:
            best, best_epoch, stale = validation, epoch, 0
            atomic_checkpoint(best_path, model.state_dict())
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": total / count, "validation_loss": validation,
               "best_epoch": best_epoch, "seconds": time.monotonic() - clock}
        history.append(row); pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        atomic_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": best, "stale": stale, "best_epoch": best_epoch,
            "history": history, "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": generator.get_state(),
        })
        print(json.dumps({"stage": "epoch", **row}), flush=True)
    model.load_state_dict(torch.load(best_path, map_location="cuda", weights_only=True))
    atomic_json(output / "training_complete.json", {
        "best_epoch": best_epoch, "validation_loss": best,
        "epochs_executed": len(history), "model_hash": state_hash(model),
        "selection": "normal_validation_loss", "test_labels_used": False,
    })
    return model.eval()


def residuals(model, dataset):
    pred_parts, recon_parts, q_parts, ref_parts = [], [], [], []
    with torch.inference_mode():
        for sequence, seq, end in DataLoader(dataset, 128, shuffle=False, pin_memory=True):
            sequence = sequence.cuda(non_blocking=True)
            x, y = sequence[:, :-1], sequence[:, -1]
            forecast, reconstruction = model(x)
            pred_parts.append((y - forecast).cpu().numpy())
            recon_parts.append((y - reconstruction[:, -1]).cpu().numpy())
            q_parts.append(condition_features(x, 1, 2).cpu().numpy())
            ref_parts.append(np.column_stack((seq.numpy(), end.numpy())))
    return tuple(map(np.concatenate, (pred_parts, recon_parts, q_parts, ref_parts)))


def fit_scores(train_values, values):
    train_pred, train_recon, train_q, _ = train_values
    pred_cal = LinearResidualCalibration(0.05).fit(train_q, train_pred[:, None])
    recon_cal = LinearResidualCalibration(0.05).fit(train_q, train_recon[:, None])
    raw_train = 0.5 * (np.square(train_pred).mean(1) + np.square(train_recon).mean(1))
    zpred_train = pred_cal.transform(train_q, train_pred[:, None])[:, 0]
    zrecon_train = recon_cal.transform(train_q, train_recon[:, None])[:, 0]
    calibrated_train = 0.5 * (np.square(zpred_train).mean(1) + np.square(zrecon_train).mean(1))
    raw_cdf, calibrated_cdf = EmpiricalCDF().fit(raw_train), EmpiricalCDF().fit(calibrated_train)

    scored = {}
    for partition, (pred, recon, q, refs) in values.items():
        raw = 0.5 * (np.square(pred).mean(1) + np.square(recon).mean(1))
        zpred = pred_cal.transform(q, pred[:, None])[:, 0]
        zrecon = recon_cal.transform(q, recon[:, None])[:, 0]
        calibrated = 0.5 * (np.square(zpred).mean(1) + np.square(zrecon).mean(1))
        scored[partition] = (refs, {
            "raw": raw, "raw_cdf": raw_cdf.transform(raw),
            "rcr_calibrated": calibrated,
            "rcr_calibrated_cdf": calibrated_cdf.transform(calibrated),
        })
    return scored, pred_cal, recon_cal


def aggregate(scored, ids, metadata, partition):
    refs, scores = scored[partition]
    rows = []
    meta = metadata if isinstance(metadata, dict) else {row["sample_id"]: row for row in metadata}
    for sequence, sample_id in enumerate(ids[partition]):
        mask = refs[:, 0] == sequence
        if not np.any(mask):
            continue
        row = {"sample_id": sample_id, **dict(meta[sample_id])}
        for name, values in scores.items():
            selected = values[mask]
            k = max(1, int(np.ceil(len(selected) * 0.05)))
            row[name] = float(np.partition(selected, -k)[-k:].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(validation, test):
    rows = []
    y = test.sample_label.to_numpy(int)
    for score in ("raw", "raw_cdf", "rcr_calibrated", "rcr_calibrated_cdf"):
        threshold = float(validation[score].quantile(0.95))
        pred = test[score].to_numpy() >= threshold
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        rows.append({
            "score": score, "average_precision": float(average_precision_score(y, test[score])),
            "auroc": float(roc_auc_score(y, test[score])), "precision": float(p),
            "recall": float(r), "f1": float(f1), "fpr": float(pred[y == 0].mean()),
            "threshold": threshold, "threshold_source": "normal_validation_p95",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemistry", choices=["LFP", "NCM"], required=True)
    parser.add_argument("--arm", choices=["baseline", "simplified_c3_rcr"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0, help="Smoke only")
    parser.add_argument("--eval-limit", type=int, default=0, help="Smoke only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run outside sandbox")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    split = load_ch_battery_mixed_cycle_research_split(chemistry=args.chemistry, seed=args.seed)
    data, ids = make_data(
        split, args.window, args.stride, args.seed, args.train_limit, args.eval_limit
    )
    protocol = {
        "chemistry": args.chemistry, "arm": args.arm, "seed": args.seed,
        "epochs": args.epochs, "window": args.window, "stride": args.stride,
        "state_encoder": "restricted_current_soc" if args.arm != "baseline" else "none",
        "prototype_query": False, "auxiliary_loss": 0.0,
        "rcr_calibration": args.arm != "baseline", "test_selection": False,
        "train_limit": args.train_limit, "eval_limit": args.eval_limit,
    }
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Changed protocol; use a new output")
    atomic_json(path, protocol)
    model = build_model(args.arm != "baseline", args.window).cuda()
    model = train(model, data, args.output, args.epochs, args.seed)
    values = {name: residuals(model, dataset) for name, dataset in data.items()}
    scored, pred_cal, recon_cal = fit_scores(values["train"], values)
    if args.arm == "baseline":
        for _, score_map in scored.values():
            score_map["rcr_calibrated"] = score_map["raw"]
            score_map["rcr_calibrated_cdf"] = score_map["raw_cdf"]
    else:
        pred_cal.save(args.output / "rcr_prediction_calibration.npz")
        recon_cal.save(args.output / "rcr_reconstruction_calibration.npz")
    validation = aggregate(scored, ids, split["validation_metadata"], "validation")
    test = aggregate(scored, ids, split["test_metadata"], "test")
    validation.to_csv(args.output / "validation_sample_scores.csv", index=False)
    test.to_csv(args.output / "test_sample_scores.csv", index=False)
    result = metrics(validation, test)
    pd.DataFrame(result).to_csv(args.output / "metrics.csv", index=False)
    atomic_json(args.output / "metrics.json", result)
    atomic_json(args.output / "complete.json", {
        "model_hash": state_hash(model), "samples": {k: len(v) for k, v in ids.items()},
        "metrics": result, "test_labels_used_for_selection": False,
    })
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
