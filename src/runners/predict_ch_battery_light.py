"""以低内存方式执行 CH-BATTERY 预测和样本级评分。"""

import argparse
import json
import os
from bisect import bisect_right
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.ch_battery_utils import (
    CH_BATTERY_DATASET_NAME,
    aggregate_ch_battery_sample_scores,
    get_ch_battery_lfp_discharge_data,
    save_ch_battery_sample_level_reports,
)
from src.data.utils import adjust_anomaly_scores, get_target_dims, load
from src.engine.eval_methods import find_epsilon
from src.engine.prediction import Predictor
from src.models.model_factory import build_model, resolve_model_args


def _load_model_args(model_dir):
    config_path = os.path.join(model_dir, "config.txt")
    with open(config_path, "r", encoding="utf-8") as f:
        model_args = SimpleNamespace(**json.load(f))
    resolve_model_args(model_args)
    return model_args


class _MultiSequenceSlidingWindowDataset(torch.utils.data.Dataset):
    """在样本内部遍历滑动窗口，不进行跨样本拼接。"""

    def __init__(self, sample_items, window_size):
        self.window_size = int(window_size)
        self.sample_ids = []
        self.sample_tensors = []
        self.window_counts = []
        self.cumulative_counts = []

        total = 0
        for sample_id, sample_tensor in sample_items:
            if len(sample_tensor) <= self.window_size:
                continue
            self.sample_ids.append(str(sample_id))
            self.sample_tensors.append(sample_tensor)
            count = len(sample_tensor) - self.window_size
            self.window_counts.append(count)
            total += count
            self.cumulative_counts.append(total)

        self.total_windows = total

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        sample_pos = bisect_right(self.cumulative_counts, index)
        prev_count = 0 if sample_pos == 0 else self.cumulative_counts[sample_pos - 1]
        local_index = index - prev_count
        sample_tensor = self.sample_tensors[sample_pos]
        x = sample_tensor[local_index : local_index + self.window_size]
        y = sample_tensor[local_index + self.window_size : local_index + self.window_size + 1]
        return x, y


def _score_sample_collection(predictor, sample_tensors, window_size, batch_size, num_workers, pin_memory):
    """为每个样本生成合法窗口，统一批量推理，并按原始顺序返回完整分数字段。"""

    ordered_items = sorted(sample_tensors.items(), key=lambda item: str(item[0]))
    dataset = _MultiSequenceSlidingWindowDataset(ordered_items, window_size)
    if len(dataset) == 0:
        raise RuntimeError("没有长度大于窗口大小的有效样本")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(int(num_workers), 0),
        pin_memory=bool(pin_memory),
    )

    device = "cuda" if predictor.use_cuda and torch.cuda.is_available() else "cpu"
    predictor.model.eval()
    preds = []
    recons = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat, _ = predictor.model(x)
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = predictor.model(recon_x)
            preds.append(y_hat.detach().cpu().numpy())
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    recons = np.concatenate(recons, axis=0)
    actual = np.concatenate(
        [sample_tensor[window_size:].cpu().numpy() for sample_tensor in dataset.sample_tensors],
        axis=0,
    )

    if predictor.target_dims is not None:
        actual = actual[:, predictor.target_dims]

    anomaly_scores = np.zeros_like(actual)
    pred_errors = np.zeros_like(actual)
    recon_errors = np.zeros_like(actual)
    df_dict = {}
    for i in range(preds.shape[1]):
        df_dict[f"Forecast_{i}"] = preds[:, i]
        df_dict[f"Recon_{i}"] = recons[:, i]
        df_dict[f"True_{i}"] = actual[:, i]
        pred_error = np.sqrt((preds[:, i] - actual[:, i]) ** 2)
        recon_error = np.sqrt((recons[:, i] - actual[:, i]) ** 2)
        pred_errors[:, i] = pred_error
        recon_errors[:, i] = recon_error
        df_dict[f"Pred_Error_{i}"] = pred_error
        df_dict[f"Recon_Error_{i}"] = recon_error

    pred_weights, recon_weights = predictor._compute_fusion_weights(pred_errors, recon_errors)

    for i in range(preds.shape[1]):
        a_score = pred_weights[i] * pred_errors[:, i] + recon_weights[i] * recon_errors[:, i]
        if predictor.scale_scores:
            q75, q25 = np.percentile(a_score, [75, 25])
            iqr = q75 - q25
            median = np.median(a_score)
            a_score = (a_score - median) / (1 + iqr)

        anomaly_scores[:, i] = a_score
        df_dict[f"A_Score_{i}"] = a_score
        df_dict[f"Pred_Weight_{i}"] = np.full_like(a_score, pred_weights[i], dtype=np.float32)
        df_dict[f"Recon_Weight_{i}"] = np.full_like(a_score, recon_weights[i], dtype=np.float32)

    score_df = pd.DataFrame(df_dict)
    score_df["Pred_Error_Global"] = np.mean(pred_errors, axis=1)
    score_df["Recon_Error_Global"] = np.mean(recon_errors, axis=1)
    score_df["A_Score_Global"] = np.mean(anomaly_scores, axis=1)
    score_df["Pred_Weight_Global"] = float(np.mean(pred_weights))
    score_df["Recon_Weight_Global"] = float(np.mean(recon_weights))

    return score_df, dataset.sample_ids, dataset.window_counts


def run_light_predict(model_dir, batch_size=128, num_workers=2, pin_memory=True, no_cuda=False):
    model_args = _load_model_args(model_dir)
    if str(getattr(model_args, "dataset", "")).upper() != CH_BATTERY_DATASET_NAME:
        raise ValueError(f"{model_dir} is not a CH-BATTERY model directory")

    normalize = getattr(model_args, "normalize", True)
    (x_train, _), (x_test, _), split_meta = get_ch_battery_lfp_discharge_data(
        root=getattr(model_args, "ch_battery_root", ""),
        normalize=normalize,
        train_ratio=getattr(model_args, "ch_battery_train_ratio", 0.7),
        seed=getattr(model_args, "seed", 42),
        preprocessed_dir=getattr(model_args, "ch_battery_preprocessed_dir", ""),
    )
    y_test = split_meta["test_metadata"]

    train_tensors = {
        sample_id: torch.from_numpy(sample_data).float()
        for sample_id, sample_data in x_train.items()
    }
    test_tensors = {
        sample_id: torch.from_numpy(sample_data).float()
        for sample_id, sample_data in x_test.items()
    }

    first_train_sample = next(iter(train_tensors.values()))
    window_size = model_args.lookback
    n_features = first_train_sample.shape[1]
    target_dims = get_target_dims(CH_BATTERY_DATASET_NAME)
    out_dim = n_features if target_dims is None else (1 if isinstance(target_dims, int) else len(target_dims))

    model = build_model(model_args, n_features, window_size, out_dim, target_dims=target_dims)
    device = "cuda" if torch.cuda.is_available() and getattr(model_args, "use_cuda", True) and not no_cuda else "cpu"
    load(model, os.path.join(model_dir, "model.pt"), device=device)
    model.to(device)

    level = getattr(model_args, "level", None)
    q = getattr(model_args, "q", None)
    if level is None:
        level = 0.99
    if q is None:
        q = 0.001

    prediction_args = {
        "dataset": CH_BATTERY_DATASET_NAME,
        "target_dims": target_dims,
        "scale_scores": getattr(model_args, "scale_scores", True),
        "level": level,
        "q": q,
        "dynamic_pot": getattr(model_args, "dynamic_pot", False),
        "use_mov_av": getattr(model_args, "use_mov_av", False),
        "gamma": getattr(model_args, "gamma", 1.0),
        "score_fusion_mode": getattr(model_args, "score_fusion_mode", "fixed"),
        "use_event_consistency": getattr(model_args, "use_event_consistency", False),
        "event_low_ratio": getattr(model_args, "event_low_ratio", 0.5),
        "event_min_length": getattr(model_args, "event_min_length", 3),
        "reg_level": 0,
        "save_path": model_dir,
    }

    train_reference = train_tensors
    cache_predictor = Predictor(model, window_size, n_features, dict(prediction_args, save_path=model_dir))
    cache_predictor.batch_size = batch_size
    cache_predictor.use_cuda = device == "cuda"
    cached_train_df = cache_predictor.get_score_for_sequences(train_reference)
    del cache_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictor = Predictor(model, window_size, n_features, dict(prediction_args, save_path=model_dir))
    predictor.batch_size = batch_size
    predictor.use_cuda = device == "cuda"

    # 使用单个 DataLoader 统一遍历所有有效窗口，避免逐样本 DataLoader 的巨大开销。
    train_pred_df, _, _ = _score_sample_collection(
        predictor,
        train_tensors,
        window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_pred_df, test_sample_ids, test_window_counts = _score_sample_collection(
        predictor,
        test_tensors,
        window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_scores = adjust_anomaly_scores(
        train_pred_df["A_Score_Global"].to_numpy(dtype=np.float32),
        CH_BATTERY_DATASET_NAME,
        True,
        window_size,
    )
    test_scores = adjust_anomaly_scores(
        test_pred_df["A_Score_Global"].to_numpy(dtype=np.float32),
        CH_BATTERY_DATASET_NAME,
        False,
        window_size,
    )
    train_pred_df["A_Score_Global"] = train_scores
    test_pred_df["A_Score_Global"] = test_scores

    global_threshold = find_epsilon(train_scores, reg_level=getattr(predictor, "reg_level", 0))
    event_low_threshold = predictor._compute_event_low_threshold(train_scores, global_threshold)

    sample_rows = []
    offset = 0
    for sample_id, window_count in zip(test_sample_ids, test_window_counts):
        sample_df = test_pred_df.iloc[offset : offset + window_count].copy()
        offset += window_count

        raw_preds = (sample_df["A_Score_Global"].to_numpy(dtype=np.float32) >= global_threshold).astype(int)
        if predictor.use_event_consistency:
            sample_event_preds = predictor._apply_event_consistency(
                sample_df,
                global_threshold,
                event_low_threshold,
            )
            sample_df["A_Pred_Global"] = sample_event_preds
        else:
            sample_df["A_Pred_Global"] = raw_preds

        sample_meta = y_test.get(sample_id, {})
        score_row = dict(sample_meta)
        score_row.update(
            aggregate_ch_battery_sample_scores(
                sample_df,
                topk_ratio=getattr(model_args, "ch_battery_topk_ratio", 0.05),
            )
        )
        sample_rows.append(score_row)

    _, sample_summary = save_ch_battery_sample_level_reports(
        model_dir,
        sample_rows,
        score_field=getattr(model_args, "ch_battery_sample_score", "score_topk_mean"),
    )
    print(f"[{CH_BATTERY_DATASET_NAME}] sample-level summary: {sample_summary}")
    return sample_summary


def main():
    parser = argparse.ArgumentParser(description="Run CH-BATTERY light prediction for a saved model directory.")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt and config.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    run_light_predict(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        no_cuda=args.no_cuda,
    )


if __name__ == "__main__":
    main()
