"""加载已训练模型并执行预测，同时导出结果。"""

import argparse
import datetime
import json

from src.args import apply_dataset_defaults, get_parser, str2bool
from src.data.ch_battery_utils import (
    CH_BATTERY_DATASET_NAME,
    aggregate_ch_battery_sample_scores,
    get_ch_battery_lfp_discharge_data,
    save_ch_battery_sample_level_reports,
)
from src.data.utils import *
from src.data.tsinghua_ev_utils import (
    DATASET_NAME as TSINGHUA_EV_DATASET_NAME,
    aggregate_sample_scores,
    get_tsinghua_ev_data,
)
from src.engine.prediction import Predictor
from src.models.model_factory import build_model, resolve_model_args, resolve_physical_state_config
from src.project_paths import MANUAL_RUNS_ROOT


def _to_tensor_sequence_container(sequence_data):
    if is_sequence_container(sequence_data):
        return [torch.from_numpy(np.asarray(seq, dtype=np.float32)).float() for seq in ensure_sequence_list(sequence_data)]
    return torch.from_numpy(np.asarray(sequence_data, dtype=np.float32)).float()


def _get_first_sequence_tensor(sequence_data):
    if is_sequence_container(sequence_data):
        sequence_list = ensure_sequence_list(sequence_data)
        if not sequence_list:
            raise ValueError("No sequence data available")
        return sequence_list[0]
    return sequence_data


def resolve_manual_model_root(dataset, group=None, universal_model=False):
    model_root = MANUAL_RUNS_ROOT / str(dataset)
    if str(dataset).upper() == "SMD" and group:
        model_root = model_root / str(group)
    if universal_model:
        model_root = model_root / "universal_model"
    return str(model_root)


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default=None,
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    args = apply_dataset_defaults(parser.parse_args())
    resolve_model_args(args)
    print(args)

    dataset = args.dataset
    if args.model_id is None:
        if dataset == 'SMD':
            dir_path = resolve_manual_model_root(dataset, group=args.group)
        elif dataset in ['CALCE', 'CALCE2']:
            dir_path = resolve_manual_model_root(dataset, universal_model=True)
        else:
            dir_path = resolve_manual_model_root(dataset)
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = []
        for subf in subfolders:
            try:
                dt = datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S')
                date_times.append(dt)
            except ValueError:
                # 跳过无法解析为日期时间的目录
                continue
        if not date_times:
            raise Exception(f"No valid datetime directories found in {dir_path}")
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = args.model_id

    if dataset == "SMD":
        model_path = os.path.join(resolve_manual_model_root(dataset, group=args.group), model_id)
    elif dataset in ['MSL', 'SMAP', 'BMS', 'SWAT', 'WADI', CH_BATTERY_DATASET_NAME, TSINGHUA_EV_DATASET_NAME, 'NASA_RANDOM_CHARGE', 'NASA_RANDOM_DISCHARGE']:
        model_path = os.path.join(resolve_manual_model_root(dataset), model_id)
    elif dataset in ['CALCE', 'CALCE2']:
        model_path = os.path.join(resolve_manual_model_root(dataset, universal_model=True), model_id)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # 检查模型文件是否存在
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # 加载模型配置
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()

    # 对于 CALCE 数据集，配置位于主模型目录中
    model_args_path = f"{model_path}/config.txt"
    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)
    # 对已保存配置重新应用冻结结构锁：保留 C3/C4 正式开关，禁用历史分支。
    resolve_model_args(model_args)
    if getattr(model_args, "require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError("This model run requires CUDA, but CUDA is unavailable for prediction.")
    window_size = model_args.lookback

    # 校验预测数据集与训练数据集一致
    if args.dataset.lower() != model_args.dataset.lower():
        raise Exception(f"Model trained on {model_args.dataset}, but asked to predict {args.dataset}.")

    elif args.dataset == "SMD" and args.group != model_args.group:
        print(f"Model trained on SMD group {model_args.group}, but asked to predict SMD group {args.group}.")

    window_size = model_args.lookback
    normalize = model_args.normalize
    n_epochs = model_args.epochs
    batch_size = model_args.bs
    init_lr = model_args.init_lr
    val_split = model_args.val_split
    shuffle_dataset = model_args.shuffle_dataset
    use_cuda = model_args.use_cuda
    print_every = model_args.print_every
    group_index = model_args.group[0]
    index = model_args.group[2:]
    args_summary = str(model_args.__dict__)

    explicit_validation_data = None

    if dataset == "SMD":
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in {"MSL", "SMAP"}:
        nasa_validation_protocol = str(
            getattr(model_args, "nasa_validation_protocol", "temporal_per_entity")
        ).lower()
        nasa_loader_val_ratio = (
            val_split if nasa_validation_protocol == "temporal_per_entity" else 0.0
        )
        x_train, explicit_validation_data, x_test, y_test = get_nasa_telemetry_sequence_data(
            dataset,
            val_ratio=nasa_loader_val_ratio,
            normalize=normalize,
        )
    elif dataset in ["NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"]:
        (x_train, _), (x_test, y_test) = get_nasa_random_battery_data(
            dataset,
            normalize=normalize,
            nasa_battery_id=model_args.nasa_battery_id if hasattr(model_args, "nasa_battery_id") else "",
            nasa_train_batteries=model_args.nasa_train_batteries if hasattr(model_args, "nasa_train_batteries") else "",
            nasa_test_batteries=model_args.nasa_test_batteries if hasattr(model_args, "nasa_test_batteries") else "",
        )
    elif dataset == "BMS":
        (x_train, _), (x_test, y_test) = get_bms_cluster_data(normalize=normalize)
    elif dataset in {"SWAT", "WADI"}:
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset == CH_BATTERY_DATASET_NAME:
        (x_train, _), (x_test, _), ch_battery_split_meta = get_ch_battery_lfp_discharge_data(
            root=getattr(model_args, "ch_battery_root", args.ch_battery_root),
            normalize=normalize,
            train_ratio=getattr(model_args, "ch_battery_train_ratio", args.ch_battery_train_ratio),
            seed=getattr(model_args, "seed", args.seed),
            preprocessed_dir=getattr(model_args, "ch_battery_preprocessed_dir", args.ch_battery_preprocessed_dir),
        )
        y_test = ch_battery_split_meta["test_metadata"]
    elif dataset == TSINGHUA_EV_DATASET_NAME:
        (x_train, _), (x_test, y_test), tsinghua_split_meta = get_tsinghua_ev_data(
            root=getattr(model_args, "tsinghua_ev_root", args.tsinghua_ev_root),
            normalize=normalize,
            train_ratio=getattr(model_args, "tsinghua_ev_train_ratio", args.tsinghua_ev_train_ratio),
            validation_ratio=getattr(model_args, "tsinghua_ev_validation_ratio", args.tsinghua_ev_validation_ratio),
            max_train_samples=getattr(model_args, "tsinghua_ev_max_train_samples", 0),
            max_validation_samples=getattr(model_args, "tsinghua_ev_max_validation_samples", 0),
            max_test_samples_per_class=getattr(model_args, "tsinghua_ev_max_test_samples_per_class", 0),
            seed=getattr(model_args, "seed", args.seed),
        )
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=normalize)

    nasa_train_tensors = None
    bms_train_tensors = None
    ch_battery_train_tensors = None
    tsinghua_train_tensors = None
    tsinghua_validation_tensors = None
    explicit_validation_tensor = None
    segmented_train_tensors = None
    if dataset in ["NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"] and isinstance(x_train, dict):
        nasa_train_tensors = {battery_name: torch.from_numpy(battery_data).float()
                              if not is_sequence_container(battery_data)
                              else _to_tensor_sequence_container(battery_data)
                              for battery_name, battery_data in x_train.items()}
        first_train_battery = next(iter(nasa_train_tensors))
        x_train = _get_first_sequence_tensor(nasa_train_tensors[first_train_battery])
    elif dataset == "BMS" and isinstance(x_train, dict):
        bms_train_tensors = {cluster_name: torch.from_numpy(cluster_data).float()
                             for cluster_name, cluster_data in x_train.items()}
        first_train_cluster = next(iter(bms_train_tensors))
        x_train = bms_train_tensors[first_train_cluster]
    elif dataset == CH_BATTERY_DATASET_NAME and isinstance(x_train, dict):
        ch_battery_train_tensors = {
            sample_id: torch.from_numpy(sample_data).float()
            for sample_id, sample_data in x_train.items()
        }
        first_train_sample = next(iter(ch_battery_train_tensors))
        x_train = ch_battery_train_tensors[first_train_sample]
    elif dataset == TSINGHUA_EV_DATASET_NAME and isinstance(x_train, dict):
        tsinghua_train_tensors = {
            sample_id: torch.from_numpy(sample_data).float()
            for sample_id, sample_data in x_train.items()
        }
        tsinghua_validation_tensors = {
            sample_id: torch.from_numpy(sample_data).float()
            for sample_id, sample_data in tsinghua_split_meta["validation_data"].items()
        }
        x_train = next(iter(tsinghua_train_tensors.values()))
    elif is_sequence_container(x_train):
        segmented_train_tensors = _to_tensor_sequence_container(x_train)
        x_train = _get_first_sequence_tensor(segmented_train_tensors)
        explicit_validation_tensor = _to_tensor_sequence_container(explicit_validation_data)
    else:
        x_train = torch.from_numpy(x_train).float()
        if explicit_validation_data is not None:
            explicit_validation_tensor = torch.from_numpy(explicit_validation_data).float()

    nasa_test_tensors = None
    bms_test_tensors = None
    ch_battery_test_tensors = None
    tsinghua_test_tensors = None
    segmented_test_tensors = None
    if dataset in ["NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"] and isinstance(x_test, dict):
        nasa_test_tensors = {battery_name: torch.from_numpy(battery_data).float()
                             if not is_sequence_container(battery_data)
                             else _to_tensor_sequence_container(battery_data)
                             for battery_name, battery_data in x_test.items()}
        first_test_battery = next(iter(nasa_test_tensors))
        x_test = _get_first_sequence_tensor(nasa_test_tensors[first_test_battery])
    elif dataset == "BMS" and isinstance(x_test, dict):
        bms_test_tensors = {cluster_name: torch.from_numpy(cluster_data).float()
                            for cluster_name, cluster_data in x_test.items()}
        first_test_cluster = next(iter(bms_test_tensors))
        x_test = bms_test_tensors[first_test_cluster]
    elif dataset == CH_BATTERY_DATASET_NAME and isinstance(x_test, dict):
        ch_battery_test_tensors = {
            sample_id: torch.from_numpy(sample_data).float()
            for sample_id, sample_data in x_test.items()
        }
        first_test_sample = next(iter(ch_battery_test_tensors))
        x_test = ch_battery_test_tensors[first_test_sample]
    elif dataset == TSINGHUA_EV_DATASET_NAME and isinstance(x_test, dict):
        tsinghua_test_tensors = {
            sample_id: torch.from_numpy(sample_data).float()
            for sample_id, sample_data in x_test.items()
        }
        x_test = next(iter(tsinghua_test_tensors.values()))
    elif is_sequence_container(x_test):
        segmented_test_tensors = _to_tensor_sequence_container(x_test)
        x_test = _get_first_sequence_tensor(segmented_test_tensors)
    else:
        x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    predict_window_stride = getattr(model_args, "window_stride", args.window_stride)
    if predict_window_stride is None:
        predict_window_stride = args.window_stride

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims, stride=predict_window_stride)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims, stride=predict_window_stride)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims, stride=predict_window_stride)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims, stride=predict_window_stride)

    model = build_model(model_args, n_features, window_size, out_dim, target_dims=target_dims)

    device = "cuda" if model_args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # POT 参数建议
    level_q_dict = {
        "MSL": (0.90, 0.001),
        "SMAP": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "NASA_RANDOM_CHARGE": (0.99, 0.001),
        "NASA_RANDOM_DISCHARGE": (0.99, 0.001),
        "CALCE": (0.99, 0.001),
        "CALCE2": (0.99, 0.001),
        "BMS": (0.99, 0.001),
        "SWAT": (0.99, 0.001),
        "WADI": (0.99, 0.001),
        CH_BATTERY_DATASET_NAME: (0.99, 0.001),
        TSINGHUA_EV_DATASET_NAME: (0.99, 0.001),
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Epsilon 参数建议
    reg_level_dict = {"MSL": 0, "SMAP": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "NASA_RANDOM_CHARGE": 0, "NASA_RANDOM_DISCHARGE": 0, "CALCE": 0, "CALCE2": 0, "BMS": 0, "SWAT": 0, "WADI": 0, CH_BATTERY_DATASET_NAME: 0, TSINGHUA_EV_DATASET_NAME: 0}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    # 对于 CALCE 数据集，需要确定测试实体以创建合适的保存路径
    save_path = model_path
    if dataset in ['CALCE', 'CALCE2']:
        # 查找实体目录
        entity_dirs = [d for d in os.listdir(model_path)
                      if os.path.isdir(os.path.join(model_path, d)) and d.isdigit()]
        if entity_dirs:
            # 默认使用第一个实体目录保存结果
            first_entity = sorted(entity_dirs, key=int)[0]
            save_path = os.path.join(model_path, first_entity)

    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "score_fusion_mode": args.score_fusion_mode,
        "nasa_score_calibration": getattr(model_args, "nasa_score_calibration", "none"),
        "use_physical_response_score": getattr(args, "use_physical_response_score", False),
        "physical_response_config": resolve_physical_state_config(args),
        "physical_response_max_weight": getattr(args, "physical_response_max_weight", 0.35),
        "use_physical_consistency_head": getattr(args, "use_physical_consistency_head", False),
        "physical_consistency_score_max_weight": getattr(
            args, "physical_consistency_score_max_weight", 0.35
        ),
        "normal_threshold_quantile": getattr(args, "normal_threshold_quantile", 0.99),
        "use_relation_change_score": getattr(args, "use_relation_change_score", False),
        "relation_change_weight": getattr(args, "relation_change_weight", 0.2),
        "relation_change_fusion_mode": getattr(args, "relation_change_fusion_mode", "linear_legacy"),
        "relation_change_mode": getattr(args, "relation_change_mode", "consecutive_js"),
        "predict_batch_size": getattr(args, "predict_batch_size", 128),
        "predict_num_workers": getattr(args, "predict_num_workers", 2),
        "predict_pin_memory": getattr(args, "predict_pin_memory", True),
        "use_cuda": getattr(model_args, "use_cuda", True),
        "window_stride": getattr(model_args, "window_stride", 1),
        "use_event_consistency": args.use_event_consistency,
        "event_low_ratio": args.event_low_ratio,
        "event_min_length": args.event_min_length,
        "reg_level": reg_level,
        "save_path": save_path,
    }

    # 每次使用预训练模型产生新预测时创建新的 summary 文件
    count = 0
    for filename in os.listdir(save_path):
        if filename.startswith("summary"):
            count += 1
    if count == 0:
        summary_file_name = "summary.txt"
    else:
        summary_file_name = f"summary_{count}.txt"

    if dataset == "BMS" and bms_test_tensors is not None:
        train_reference = bms_train_tensors if bms_train_tensors is not None else x_train

        # 预计算训练数据分数（所有聚类共享，避免重复模型推理）
        cache_predictor = Predictor(
            model,
            window_size,
            n_features,
            dict(prediction_args, save_path=save_path),
        )
        cached_train_df = cache_predictor.get_score_for_sequences(train_reference)
        del cache_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for cluster_name, cluster_tensor in bms_test_tensors.items():
            cluster_save_path = os.path.join(save_path, cluster_name)
            os.makedirs(cluster_save_path, exist_ok=True)

            cluster_prediction_args = dict(prediction_args)
            cluster_prediction_args["save_path"] = cluster_save_path

            cluster_summary_count = 0
            for filename in os.listdir(cluster_save_path):
                if filename.startswith("summary"):
                    cluster_summary_count += 1
            if cluster_summary_count == 0:
                cluster_summary_name = "summary.txt"
            else:
                cluster_summary_name = f"summary_{cluster_summary_count}.txt"

            predictor = Predictor(
                model,
                window_size,
                n_features,
                cluster_prediction_args,
                summary_file_name=cluster_summary_name,
            )
            cluster_label = None
            if isinstance(y_test, dict):
                raw_label = y_test.get(cluster_name)
                if raw_label is not None:
                    cluster_label = raw_label[window_size:]

            predictor.predict_anomalies(
                train_reference,
                cluster_tensor,
                cluster_label,
                load_scores=args.load_scores,
                save_output=args.save_output,
                cached_train_pred_df=cached_train_df,
            )
    elif dataset == CH_BATTERY_DATASET_NAME and ch_battery_test_tensors is not None:
        train_reference = ch_battery_train_tensors if ch_battery_train_tensors is not None else x_train

        cache_predictor = Predictor(
            model,
            window_size,
            n_features,
            dict(prediction_args, save_path=save_path),
        )
        cached_train_df = cache_predictor.get_score_for_sequences(train_reference)
        del cache_predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        predictor = Predictor(
            model,
            window_size,
            n_features,
            dict(prediction_args, save_path=save_path),
        )

        sample_rows = []
        for sample_id, sample_tensor in ch_battery_test_tensors.items():
            test_pred_df = predictor.predict_anomalies(
                train_reference,
                sample_tensor,
                None,
                load_scores=args.load_scores,
                save_output=False,
                cached_train_pred_df=cached_train_df,
            )
            sample_meta = y_test.get(sample_id, {}) if isinstance(y_test, dict) else {}
            score_row = dict(sample_meta)
            score_row.update(
                aggregate_ch_battery_sample_scores(
                    test_pred_df,
                    topk_ratio=getattr(model_args, "ch_battery_topk_ratio", args.ch_battery_topk_ratio),
                )
            )
            sample_rows.append(score_row)

        _, sample_summary = save_ch_battery_sample_level_reports(
            save_path,
            sample_rows,
            score_field=getattr(model_args, "ch_battery_sample_score", args.ch_battery_sample_score),
        )
        print(f"[{dataset}] sample-level summary: {sample_summary}")
    elif dataset == TSINGHUA_EV_DATASET_NAME and tsinghua_test_tensors is not None:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        predictor = Predictor(model, window_size, n_features, dict(prediction_args, save_path=save_path))
        validation_window_scores = predictor.get_sample_map_scores(
            tsinghua_validation_tensors,
            calibrate_relation=bool(getattr(model_args, "use_relation_change_score", False)),
        )
        validation_value_only_window_scores = dict(predictor.last_sample_value_only_scores)
        calibration_scores, _, _ = aggregate_sample_scores(
            validation_window_scores,
            tsinghua_split_meta["validation_metadata"],
            top_ratio=getattr(model_args, "sample_score_top_ratio", args.sample_score_top_ratio),
        )
        calibration_value_only_scores, _, _ = aggregate_sample_scores(
            validation_value_only_window_scores,
            tsinghua_split_meta["validation_metadata"],
            top_ratio=getattr(model_args, "sample_score_top_ratio", args.sample_score_top_ratio),
        )
        test_window_scores = predictor.get_sample_map_scores(tsinghua_test_tensors)
        test_value_only_window_scores = dict(predictor.last_sample_value_only_scores)
        sample_scores, sample_labels, sample_ids = aggregate_sample_scores(
            test_window_scores,
            y_test,
            top_ratio=getattr(model_args, "sample_score_top_ratio", args.sample_score_top_ratio),
        )
        value_only_sample_scores, value_only_labels, value_only_ids = aggregate_sample_scores(
            test_value_only_window_scores,
            y_test,
            top_ratio=getattr(model_args, "sample_score_top_ratio", args.sample_score_top_ratio),
        )
        if not np.array_equal(sample_labels, value_only_labels) or sample_ids != value_only_ids:
            raise RuntimeError("Paired Tsinghua value-only scores are not sample-aligned")
        threshold = float(np.quantile(calibration_scores, 0.99))
        value_only_threshold = float(np.quantile(calibration_value_only_scores, 0.99))
        sample_predictions = (sample_scores >= threshold).astype(np.int64)
        value_only_predictions = (value_only_sample_scores >= value_only_threshold).astype(np.int64)
        from sklearn.metrics import auc, precision_recall_curve

        average_precision = float(average_precision_score(sample_labels, sample_scores))
        value_only_average_precision = float(
            average_precision_score(sample_labels, value_only_sample_scores)
        )
        value_only_auroc = float(roc_auc_score(sample_labels, value_only_sample_scores))
        precision_curve, recall_curve, _ = precision_recall_curve(sample_labels, sample_scores)
        trapezoidal_pr_auc = float(auc(recall_curve, precision_curve))
        report = {
            "label_level": "charging_snippet",
            "vehicle_identity_available": False,
            "auroc": float(roc_auc_score(sample_labels, sample_scores)),
            "pr_auc": average_precision,
            "pr_auc_trapezoid": trapezoidal_pr_auc,
            "average_precision": average_precision,
            "auprc": average_precision,
            "f1_at_calibration_normal_p99": float(f1_score(sample_labels, sample_predictions)),
            "precision_at_calibration_normal_p99": float(
                precision_score(sample_labels, sample_predictions, zero_division=0)
            ),
            "recall_at_calibration_normal_p99": float(recall_score(sample_labels, sample_predictions)),
            "false_positive_rate_at_calibration_normal_p99": float(np.mean(sample_predictions[sample_labels == 0])),
            "specificity_at_calibration_normal_p99": float(
                1.0 - np.mean(sample_predictions[sample_labels == 0])
            ),
            "balanced_accuracy_at_calibration_normal_p99": float(
                balanced_accuracy_score(sample_labels, sample_predictions)
            ),
            "threshold_validation_normal_p99": threshold,
            "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "inference_efficiency": dict(predictor.last_scoring_stats),
            "physical_response_mae_by_term": dict(predictor.last_physical_response_term_summary),
            "score_calibration": predictor.get_calibration_summary(),
        }
        if getattr(model_args, "use_relation_change_score", False):
            report.update({
                "value_only_average_precision": value_only_average_precision,
                "average_precision_delta_vs_value_only": average_precision - value_only_average_precision,
                "value_only_auroc": value_only_auroc,
                "auroc_delta_vs_value_only": report["auroc"] - value_only_auroc,
                "value_only_f1_at_calibration_normal_p99": float(
                    f1_score(sample_labels, value_only_predictions)
                ),
                "value_only_recall_at_calibration_normal_p99": float(
                    recall_score(sample_labels, value_only_predictions)
                ),
                "value_only_false_positive_rate_at_calibration_normal_p99": float(
                    np.mean(value_only_predictions[sample_labels == 0])
                ),
                "false_positive_rate_delta_vs_value_only": (
                    report["false_positive_rate_at_calibration_normal_p99"]
                    - float(np.mean(value_only_predictions[sample_labels == 0]))
                ),
                "value_only_threshold_validation_normal_p99": value_only_threshold,
            })
        pd.DataFrame({
            "sample_id": sample_ids,
            "label": sample_labels,
            "score": sample_scores,
            "value_only_score": value_only_sample_scores,
            "prediction": sample_predictions,
            "value_only_prediction": value_only_predictions,
        }).to_csv(os.path.join(save_path, "sample_scores.csv"), index=False)
        with open(os.path.join(save_path, "sample_metrics.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"[{dataset}] snippet-level metrics: {report}")
    else:
        # Predictor owns window/stride alignment.  Passing labels already
        # sliced by lookback only works for stride=1 and breaks frozen runs
        # evaluated with a larger window stride.
        label = y_test
        predictor = Predictor(model, window_size, n_features, prediction_args, summary_file_name=summary_file_name)
        calibration_reference = explicit_validation_tensor if explicit_validation_tensor is not None else x_train
        test_reference = segmented_test_tensors if segmented_test_tensors is not None else x_test
        if is_sequence_container(calibration_reference):
            predictor.get_sample_map_scores({
                f"calibration_{index}": sequence
                for index, sequence in enumerate(calibration_reference)
            })
        predictor.predict_anomalies(calibration_reference, test_reference, label,
                                    load_scores=args.load_scores,
                                    save_output=args.save_output)
