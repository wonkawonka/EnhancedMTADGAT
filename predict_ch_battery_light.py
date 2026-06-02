"""CH-BATTERY 轻量预测脚本：低内存占用版本

用法：
    python predict_ch_battery_light.py --model_dir <模型目录> [--exp_config <计划配置文件> --exp_name <实验名>]
    
    有 config.txt 时自动使用；没有 config.txt 时需要传入 --exp_config 和 --exp_name。

示例：
    python predict_ch_battery_light.py --model_dir "kaggle离线output/ch_battery_main/chbatt_lfp_c3_full" ^
        --exp_config "configs/compare/ch_battery_main.json" --exp_name chbatt_lfp_c3_full
    
    python predict_ch_battery_light.py --model_dir "kaggle离线output/ch_battery_main/chbatt_lfp_mtadgat_baseline" ^
        --exp_config "configs/compare/ch_battery_main.json" --exp_name chbatt_lfp_mtadgat_baseline

说明：
    - batch_size=16 （原256）
    - num_workers=0 （Windows下多进程额外占用内存）
    - 逐样本计算训练分数，不全部加载到内存后再拼接
    - 最终输出格式与原 predict.py 一致
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from args import get_parser
from ch_battery_utils import (
    CH_BATTERY_DATASET_NAME,
    get_ch_battery_lfp_discharge_data,
    save_ch_battery_sample_level_reports,
)
from model_factory import build_model, resolve_model_args
from prediction import Predictor
from utils import load, SlidingWindowDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="训练好的模型目录（含 model.pt）")
    parser.add_argument("--exp_config", type=str, default="", help="计划配置 JSON 文件路径（当目录无 config.txt 时需要）")
    parser.add_argument("--exp_name", type=str, default="", help="计划配置中的实验名（当目录无 config.txt 时需要）")
    parser.add_argument("--batch_size", type=int, default=64, help="预测时的批大小（默认64，小内存可降为32）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 工作进程数（默认0，Windows建议0）")
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == "true", default=True, help="是否启用 pin_memory")
    parser.add_argument("--no_cuda", action="store_true", help="强制使用CPU")
    return parser.parse_args()


def _load_args_from_exp_config(exp_config_path, exp_name):
    """从计划配置 JSON 文件中加载模型参数"""
    with open(exp_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    common_args = config.get("common_args", {})
    experiments = config.get("experiments", [])

    exp_args = None
    for exp in experiments:
        if exp["name"] == exp_name:
            exp_args = exp["args"]
            break

    if exp_args is None:
        raise ValueError(f"在 {exp_config_path} 中未找到实验 '{exp_name}'")

    merged = dict(common_args)
    merged.update(exp_args)
    return merged


def _resolve_target_array(batch_y, target_dims):
    actual = batch_y.squeeze(1)
    if target_dims is None:
        return actual
    if isinstance(target_dims, int):
        return actual[:, [target_dims]]
    return actual[:, target_dims]


def _prepare_sequence_datasets(sequence_map, sample_ids_sorted, window_size, target_dims):
    datasets = []
    valid_ids = []
    window_counts = []

    for sample_id in sample_ids_sorted:
        arr = sequence_map[sample_id]
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        if arr.shape[0] <= window_size:
            continue

        dataset = SlidingWindowDataset(arr, window_size, target_dims)
        datasets.append(dataset)
        valid_ids.append(sample_id)
        window_counts.append(len(dataset))

    if not datasets:
        raise RuntimeError("没有有效样本可用于推理")

    return ConcatDataset(datasets), valid_ids, window_counts


def _score_sequence_map_global(
    predictor,
    sequence_map,
    sample_ids_sorted,
    batch_size,
    num_workers,
    pin_memory,
    device,
):
    dataset, valid_ids, window_counts = _prepare_sequence_datasets(
        sequence_map,
        sample_ids_sorted,
        predictor.window_size,
        predictor.target_dims,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(int(num_workers), 0),
        pin_memory=bool(pin_memory and device == "cuda"),
    )

    preds = []
    recons = []
    actuals = []

    predictor.model.eval()
    with torch.inference_mode():
        for x, y in tqdm(loader):
            x = x.to(device, non_blocking=bool(pin_memory and device == "cuda"))
            y = y.to(device, non_blocking=bool(pin_memory and device == "cuda"))

            y_hat, _ = predictor.model(x)
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = predictor.model(recon_x)

            if y_hat.ndim == 3:
                y_hat = y_hat.squeeze(1)
            preds.append(y_hat.detach().cpu().numpy())
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())
            actuals.append(_resolve_target_array(y, predictor.target_dims).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0).astype(np.float32, copy=False)
    recons = np.concatenate(recons, axis=0).astype(np.float32, copy=False)
    actual = np.concatenate(actuals, axis=0).astype(np.float32, copy=False)

    pred_errors = np.abs(preds - actual)
    recon_errors = np.abs(recons - actual)
    pred_weights, recon_weights = predictor._compute_fusion_weights(pred_errors, recon_errors)
    global_scores = np.mean(
        pred_errors * pred_weights[None, :] + recon_errors * recon_weights[None, :],
        axis=1,
    ).astype(np.float32, copy=False)

    return global_scores, valid_ids, window_counts


def run_light_predict(
    model_dir,
    exp_config="",
    exp_name="",
    batch_size=64,
    num_workers=0,
    pin_memory=True,
    no_cuda=False,
):
    """运行 CH-BATTERY 轻量预测，并将结果写回 model_dir。"""
    cli_args = argparse.Namespace(
        model_dir=str(model_dir),
        exp_config=str(exp_config or ""),
        exp_name=str(exp_name or ""),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        no_cuda=bool(no_cuda),
    )
    return _run(cli_args)


def _run(cli_args):
    model_dir = Path(cli_args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 1. 加载训练配置
    config_path = model_dir / "config.txt"
    if config_path.exists():
        with open(config_path, "r") as f:
            model_args_dict = json.load(f)
        print(f"从 {config_path} 加载配置")
    else:
        if not cli_args.exp_config or not cli_args.exp_name:
            raise FileNotFoundError(
                f"{config_path} 不存在。请通过 --exp_config 和 --exp_name 指定配置来源。\n"
                f"例如：--exp_config configs/compare/ch_battery_main.json --exp_name {model_dir.name}"
            )
        model_args_dict = _load_args_from_exp_config(cli_args.exp_config, cli_args.exp_name)
        print(f"从计划配置 {cli_args.exp_config} 加载实验 [{cli_args.exp_name}] 的配置")

    # 用 CLI 解析器默认值填充所有缺省字段
    _base_parser = get_parser()
    _base_args, _ = _base_parser.parse_known_args([])
    _base_dict = vars(_base_args)
    _base_dict.update(model_args_dict)
    model_args = argparse.Namespace(**_base_dict)
    resolve_model_args(model_args)

    dataset = model_args.dataset
    window_size = model_args.lookback
    normalize = model_args.normalize
    use_cuda = not cli_args.no_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    print(f"[CH-BATTERY Light Predict] batch_size={cli_args.batch_size}, num_workers={cli_args.num_workers}, device={device}")
    print(f"[CH-BATTERY Light Predict] model_dir={model_dir}")

    # 2. 加载数据
    print("加载 CH-BATTERY 数据...")
    (x_train, _), (x_test, _), ch_battery_split_meta = get_ch_battery_lfp_discharge_data(
        root=getattr(model_args, "ch_battery_root", "datasets/CH-BATTERY"),
        normalize=normalize,
        train_ratio=getattr(model_args, "ch_battery_train_ratio", 0.8),
        seed=getattr(model_args, "seed", 3407),
        preprocessed_dir=getattr(model_args, "ch_battery_preprocessed_dir", ""),
    )
    y_test = ch_battery_split_meta["test_metadata"]
    print(f"训练样本数: {len(x_train)}, 测试样本数: {len(x_test)}")

    # 3. 构建模型
    first_train_sample_data = next(iter(x_train.values()))
    n_features = first_train_sample_data.shape[1] if first_train_sample_data.ndim == 2 else 1

    target_dims = getattr(model_args, "target_dims", None)
    if target_dims is None:
        out_dim = n_features
    else:
        out_dim = target_dims if isinstance(target_dims, int) else len(target_dims)

    model = build_model(model_args, n_features, window_size, out_dim, target_dims=target_dims)
    load(model, f"{model_dir}/model.pt", device=device)
    model.to(device)
    model.eval()
    print(f"模型加载完成，特征维度={n_features}，输出维度={out_dim}")

    # 4. 创建轻量 Predictor（覆盖 batch_size 和 num_workers）
    predictor_args = {
        "dataset": dataset,
        "target_dims": target_dims,
        "scale_scores": getattr(model_args, "scale_scores", False),
        "level": getattr(model_args, "level", 0.99),
        "q": getattr(model_args, "q", 0.001),
        "dynamic_pot": getattr(model_args, "dynamic_pot", False),
        "use_mov_av": getattr(model_args, "use_mov_av", False),
        "gamma": getattr(model_args, "gamma", 1.0),
        "score_fusion_mode": getattr(model_args, "score_fusion_mode", "fixed"),
        "use_event_consistency": getattr(model_args, "use_event_consistency", False),
        "event_low_ratio": getattr(model_args, "event_low_ratio", 0.5),
        "event_min_length": getattr(model_args, "event_min_length", 3),
        "use_hier_consistency": getattr(model_args, "use_hier_consistency", False),
        "hier_score_weight": getattr(model_args, "hier_score_weight", 0.5),
        "reg_level": 0,
        "save_path": str(model_dir),
    }

    predictor = Predictor(model, window_size, n_features, predictor_args)
    predictor.batch_size = cli_args.batch_size

    from prediction import adjust_anomaly_scores

    # 5. 在不跨样本建窗的前提下，对所有训练窗口做批量推理
    train_sample_ids_sorted = sorted(x_train.keys(), key=lambda k: str(k))
    print("批量推理训练样本（保持样本边界）...")
    train_anomaly_scores, _, _ = _score_sequence_map_global(
        predictor=predictor,
        sequence_map=x_train,
        sample_ids_sorted=train_sample_ids_sorted,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        device=device,
    )
    train_anomaly_scores = adjust_anomaly_scores(
        train_anomaly_scores, dataset, True, window_size
    )
    gc.collect()
    print(f"训练分数总数: {len(train_anomaly_scores)}")

    from eval_methods import find_epsilon
    global_epsilon = find_epsilon(train_anomaly_scores, reg_level=0)
    print(f"epsilon 阈值: {global_epsilon:.6f}")

    # 6. 对测试样本做同口径批量推理
    test_sample_ids_sorted = sorted(x_test.keys(), key=lambda k: str(k))
    print("批量推理测试样本（保持样本边界）...")
    test_global_scores, test_segment_ids, test_segment_window_counts = _score_sequence_map_global(
        predictor=predictor,
        sequence_map=x_test,
        sample_ids_sorted=test_sample_ids_sorted,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        pin_memory=cli_args.pin_memory,
        device=device,
    )
    test_global_scores = adjust_anomaly_scores(
        test_global_scores, dataset, False, window_size
    )
    test_global_preds = (test_global_scores >= global_epsilon).astype(np.int32)

    # 7. 按样本聚合打分
    print("聚合样本级分数...")
    sample_rows = []
    offset = 0
    for sidx, sample_id in enumerate(test_segment_ids):
        wc = test_segment_window_counts[sidx]
        if wc == 0:
            continue
        scores = test_global_scores[offset: offset + wc]
        preds = test_global_preds[offset: offset + wc]

        sample_meta = y_test.get(str(sample_id), {}) if isinstance(y_test, dict) else {}
        score_row = dict(sample_meta) if sample_meta else {}
        score_row["sample_id"] = str(sample_id)

        topk_ratio = getattr(model_args, "ch_battery_topk_ratio", 0.05)
        topk_count = max(1, int(np.ceil(scores.size * topk_ratio)))
        topk_values = np.partition(scores, -topk_count)[-topk_count:]
        score_row["score_max"] = float(np.max(scores))
        score_row["score_mean"] = float(np.mean(scores))
        score_row["score_p95"] = float(np.percentile(scores, 95))
        score_row["score_topk_mean"] = float(np.mean(topk_values))
        score_row["window_count"] = int(scores.size)
        score_row["pred_positive_ratio"] = float(preds.mean())

        sample_rows.append(score_row)
        offset += wc

    del x_train, x_test
    gc.collect()

    # 7. 输出结果
    save_path = model_dir
    score_field = getattr(model_args, "ch_battery_sample_score", "score_topk_mean")
    report_df, summary = save_ch_battery_sample_level_reports(save_path, sample_rows, score_field=score_field)

    print("\n" + "=" * 60)
    print(f"CH-BATTERY 预测完成")
    print(f"样本总数: {summary['sample_count']}")
    print(f"正常样本: {summary['normal_count']}")
    print(f"故障样本: {summary['fault_count']}")
    if "sample_auroc" in summary:
        print(f"AUROC: {summary['sample_auroc']:.4f}")
        print(f"AUPRC: {summary['sample_auprc']:.4f}")
        print(f"Best-F1: {summary['best_f1']:.4f}")
    summary_path = Path(save_path) / "ch_battery_sample_summary.json"
    report_path = Path(save_path) / "ch_battery_sample_scores.csv"
    print(f"摘要: {summary_path}")
    print(f"详细评分: {report_path}")
    return report_df, summary


def main():
    return _run(parse_args())


if __name__ == "__main__":
    main()
