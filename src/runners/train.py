"""训练异常检测模型，并为每个数据集执行对应评估流程。"""

import gc
import json
import os
import random
import time
from datetime import datetime

import torch.nn as nn
import numpy as np
import torch

from src.args import apply_dataset_defaults, get_parser
from src.analysis.regime_evaluation import save_bms_operational_report, save_nasa_regime_probe
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
from src.engine.training import Trainer
from src.models.model_factory import build_model, resolve_model_args, resolve_physical_state_config
from src.project_paths import MANUAL_RUNS_ROOT
NASA_SEQUENCE_DATASETS = {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}
NASA_TELEMETRY_SEQUENCE_DATASETS = {"MSL", "SMAP"}


def resolve_manual_output_root(dataset, group=None, universal_model=False):
    output_root = MANUAL_RUNS_ROOT / str(dataset)
    if str(dataset).upper() == "SMD" and group:
        output_root = output_root / str(group)
    if universal_model:
        output_root = output_root / "universal_model"
    return str(output_root)


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


def _build_concat_window_dataset(sequence_data, window_size, target_dims, window_stride=1):
    sub_datasets = []
    for seq in ensure_sequence_list(sequence_data):
        if len(seq) <= window_size:
            continue
        sub_datasets.append(
            SlidingWindowDataset(seq, window_size, target_dims, stride=window_stride)
        )

    if not sub_datasets:
        raise ValueError(f"No valid sequence is longer than lookback={window_size}")

    if len(sub_datasets) == 1:
        return sub_datasets[0]
    return torch.utils.data.ConcatDataset(sub_datasets)


def _get_sequence_lengths(sequence_data):
    if is_sequence_container(sequence_data):
        return [len(seq) for seq in ensure_sequence_list(sequence_data)]
    return [len(sequence_data)]


def _count_sequence_windows(sequence_data, window_size, window_stride=1):
    stride = max(int(window_stride), 1)
    total = 0
    for seq_len in _get_sequence_lengths(sequence_data):
        available = seq_len - window_size
        if available <= 0:
            continue
        total += 1 + (available - 1) // stride
    return total


def _print_nasa_random_window_summary(split_name, data_map, window_size, window_stride=1, val_split=None):
    if not data_map:
        return

    print(f"[NASA_RANDOM] {split_name} summary (lookback={window_size}, stride={window_stride})")

    total_steps = 0
    total_segments = 0
    total_windows = 0
    for battery_name in sorted(data_map):
        sequence_data = data_map[battery_name]
        segment_lengths = _get_sequence_lengths(sequence_data)
        segment_count = len(segment_lengths)
        step_count = sum(segment_lengths)
        window_count = _count_sequence_windows(sequence_data, window_size, window_stride)

        total_steps += step_count
        total_segments += segment_count
        total_windows += window_count

        print(
            f"  [{battery_name}] steps={step_count}, segments={segment_count}, "
            f"windows={window_count}"
        )

    print(
        f"[NASA_RANDOM] {split_name} total: steps={total_steps}, "
        f"segments={total_segments}, windows={total_windows}"
    )

    if split_name == "train" and val_split is not None and val_split > 0:
        validation_windows = int(np.floor(val_split * total_windows))
        effective_train_windows = total_windows - validation_windows
        print(
            f"[NASA_RANDOM] loader split: train_windows={effective_train_windows}, "
            f"validation_windows={validation_windows}"
        )

def set_seed(seed=3407):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_accelerator(args):
    cuda_available = torch.cuda.is_available()
    if getattr(args, "require_cuda", False) and (not args.use_cuda or not cuda_available):
        raise RuntimeError(
            "CUDA is required but unavailable. Enable the Kaggle GPU accelerator and "
            "verify that PyTorch was installed with CUDA support."
        )
    if args.use_cuda and not cuda_available:
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
    elif args.use_cuda:
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA ready: {device_name}, memory={memory_gb:.1f} GiB, torch={torch.__version__}")


def get_run_id(args):
    """
    根据参数生成运行 ID。
    优先级：
    1. 若 args.run_id 为非空字符串，直接返回其 strip 后的值；
    2. 否则自动生成：dataset_multiScaleMode_feattrans-{on|off}_timestamp，
       若 args.comment 为非空字符串，则将其置于最前面，下划线连接。
    """
    run_id = getattr(args, "run_id", None)
    if not isinstance(run_id, (str, type(None))):
        raise ValueError("run_id must be string or None")
    if run_id is not None and run_id.strip():
        return run_id.strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = str(getattr(args, "dataset", "data")).lower()
    multi_scale_mode = str(getattr(args, "multi_scale_mode", "none")).lower()
    feature_att_trans = bool(getattr(args, "feature_att_trans", False))
    feattrans_mode = "on" if feature_att_trans else "off"
    auto_name = f"{dataset}_{multi_scale_mode}_feattrans-{feattrans_mode}_{timestamp}"

    comment = getattr(args, "comment", "")
    if not isinstance(comment, (str, type(None))):
        raise ValueError("comment must be string or None")
    comment = (comment or "").strip()
    if comment:
        prefix = "_".join(comment.split())
        return f"{prefix}_{auto_name}"
    return auto_name


def maybe_resume_trainer(trainer, args):
    if not getattr(args, "resume", False):
        return False
    resumed = trainer.resume_from_checkpoint()
    if not resumed:
        print(f"No checkpoint found under {trainer.dload}, starting from scratch.")
    return resumed


def build_trainer(model, optimizer, args, window_size, n_features, target_dims, save_path, log_dir, args_summary):
    return Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        args.epochs,
        args.bs,
        args.init_lr,
        nn.MSELoss(),
        nn.MSELoss(),
        args.use_cuda,
        save_path,
        log_dir,
        args.print_every,
        args.log_tensorboard,
        args_summary,
        num_workers=getattr(args, "num_workers", 4),
        persistent_workers=getattr(args, "persistent_workers", True),
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        window_stride=getattr(args, "window_stride", 1),
        regime_aux_lambda=getattr(args, "regime_aux_lambda", 0.0),
        regime_prototype_lambda=getattr(args, "regime_prototype_lambda", 0.0),
        early_stopping_patience=getattr(args, "early_stopping_patience", 0),
        early_stopping_min_delta=getattr(args, "early_stopping_min_delta", 1e-4),
    )


def maybe_compile_model(model):
    if os.environ.get("DISABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}:
        print("Skipping torch.compile because DISABLE_TORCH_COMPILE is set.")
        return model
    if os.environ.get("ENABLE_TORCH_COMPILE", "").lower() not in {"1", "true", "yes"}:
        print("Skipping torch.compile by default. Set ENABLE_TORCH_COMPILE=1 to enable.")
        return model
    if hasattr(torch, "compile"):
        try:
            print("Using torch.compile for model optimization...")
            return torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed, falling back to eager mode: {e}")
    return model


def train_universal_model(args):
    """
    训练通用模型（使用训练实体数据训练一个通用模型，然后在测试实体上分别测试）
    实现轮流训练，每个epoch使用不同实体的数据
    """
    dataset = args.dataset
    if dataset not in ['CALCE', 'CALCE2']:
        raise ValueError("Universal model training is only supported for CALCE/CALCE2 datasets")

    # 获取训练/测试实体划分
    if dataset == 'CALCE':
        train_entities, test_entities = get_calce_train_test_splits()
        load_func = load_calce_entity_data
    else:  # CALCE2 分支
        train_entities, test_entities = get_calce2_train_test_splits()
        load_func = load_calce2_entity_data

    print(f"Training entities: {train_entities}")
    print(f"Test entities: {test_entities}")

    # 加载所有训练实体的数据
    train_entity_data = []
    for entity_name in train_entities:
        try:
            (x_train, _), (_, _) = load_func(entity_name)
            if x_train is not None:
                # 确保数据是二维数组
                if np.isscalar(x_train):
                    x_train = np.array([[x_train]], dtype=np.float32)
                elif hasattr(x_train, 'ndim') and x_train.ndim == 0:
                    x_train = np.array([[x_train.item()]], dtype=np.float32)
                elif x_train.ndim == 1:
                    x_train = x_train.reshape(-1, 1)
                train_entity_data.append((entity_name, torch.from_numpy(x_train).float()))
                print(f"Loaded training data from entity {entity_name}, shape: {x_train.shape}")
        except Exception as e:
            print(f"Error loading data from entity {entity_name}: {e}")

    if not train_entity_data:
        raise ValueError("No training data loaded from any entity")

    # 训练通用模型
    id = get_run_id(args)
    window_size = args.lookback
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    args_summary = str(args.__dict__)
    print(args_summary)

    output_path = resolve_manual_output_root(dataset, universal_model=True)
    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    plan_output_dir = os.environ.get("PLAN_OUTPUT_DIR", "")
    if plan_output_dir:
        save_path = plan_output_dir

    n_features = train_entity_data[0][1].shape[1]  # 从第一个实体获取特征数

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    # 创建模型。当前默认仍为 MTAD-GAT，后续基线模型通过 model_factory 扩展。
    model = build_model(args, n_features, window_size, out_dim, target_dims=target_dims)

    model = maybe_compile_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = build_trainer(
        model,
        optimizer,
        args,
        window_size,
        n_features,
        target_dims,
        save_path,
        log_dir,
        args_summary,
    )
    maybe_resume_trainer(trainer, args)

    # 开始轮流训练：每个训练轮次使用不同实体的数据
    trainer.fit_round_robin(train_entity_data, window_size, target_dims, val_split, shuffle_dataset)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 如果训练过程中尚未写出最佳模型，则补存一次当前模型
    if not os.path.isfile(f"{save_path}/model.pt"):
        trainer.save("model.pt")

    # 为每个测试实体进行预测并保存在独立的文件夹中
    for entity_name in test_entities:
        print(f"\nEvaluating model on test entity: {entity_name}")
        try:
            # 加载该实体的测试数据
            (_, _), (entity_test_data, entity_test_labels) = load_func(entity_name)
            if entity_test_data is None:
                print(f"No test data for entity {entity_name}, skipping...")
                continue

            entity_test_tensor = torch.from_numpy(entity_test_data).float()

            # 为该实体创建输出路径（在主路径下创建子文件夹）
            entity_output_path = f'{save_path}/{entity_name}'
            if not os.path.exists(entity_output_path):
                os.makedirs(entity_output_path)

            # 配置预测参数
            level_q_dict = {
                "MSL": (0.90, 0.001),
                "SMD-1": (0.9950, 0.001),
                "SMD-2": (0.9925, 0.001),
                "SMD-3": (0.9999, 0.001),
                "CALCE": (0.95, 0.01),  # 为CALCE调整参数以适应无监督设置
                "CALCE2": (0.90, 0.01)  # 为CALCE2调整参数以适应无监督设置
            }
            key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
            level, q = level_q_dict[key]
            if args.level is not None:
                level = args.level
            if args.q is not None:
                q = args.q

            # Epsilon 参数建议
            reg_level_dict = {"MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "CALCE": 0, "CALCE2": 0}
            key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
            reg_level = reg_level_dict[key]

            # 对于CALCE2数据集，禁用动态POT以避免数值问题
            dynamic_pot = args.dynamic_pot
            if dataset == "CALCE2":
                dynamic_pot = False

            prediction_args = {
                'dataset': dataset,
                "target_dims": target_dims,
                'scale_scores': args.scale_scores,
                "level": level,
                "q": q,
                'dynamic_pot': dynamic_pot,
                "use_mov_av": args.use_mov_av,
                "gamma": args.gamma,
                "score_fusion_mode": args.score_fusion_mode,
                "use_physical_response_score": getattr(args, "use_physical_response_score", False),
                "physical_response_config": resolve_physical_state_config(args),
                "physical_response_max_weight": getattr(args, "physical_response_max_weight", 0.35),
                "use_relation_change_score": getattr(args, "use_relation_change_score", False),
                "relation_change_weight": getattr(args, "relation_change_weight", 0.2),
                "relation_change_fusion_mode": getattr(args, "relation_change_fusion_mode", "linear_legacy"),
                "relation_change_mode": getattr(args, "relation_change_mode", "consecutive_js"),
                "predict_batch_size": getattr(args, "predict_batch_size", 128),
                "predict_num_workers": getattr(args, "predict_num_workers", 2),
                "predict_pin_memory": getattr(args, "predict_pin_memory", True),
                "use_cuda": getattr(args, "use_cuda", True),
                "window_stride": getattr(args, "window_stride", 1),
                "use_event_consistency": args.use_event_consistency,
                "event_low_ratio": args.event_low_ratio,
                "event_min_length": args.event_min_length,
                "reg_level": reg_level,
                "save_path": entity_output_path,  # 使用实体特定的输出路径
            }

            # 创建预测器并进行预测
            predictor = Predictor(
                model,
                window_size,
                n_features,
                prediction_args,
            )

            label = entity_test_labels[window_size:] if entity_test_labels is not None else None
            # 使用第一个训练实体的数据作为参考来建立正常行为基线，这是无监督异常检测的标准做法
            predictor.predict_anomalies(train_entity_data[0][1], entity_test_tensor, label)

        except Exception as e:
            print(f"Error evaluating entity {entity_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存配置
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = get_parser()
    args = apply_dataset_defaults(parser.parse_args())
    resolve_model_args(args)
    runtime_started = time.perf_counter()

    validate_accelerator(args)

    # 设置随机种子以确保实验可重现性
    set_seed(args.seed)

    # 实现CALCE/CALCE2的通用模型训练
    if args.dataset in ['CALCE', 'CALCE2']:
        # 对于CALCE/CALCE2数据集，训练一个通用模型
        train_universal_model(args)
    else:
        # 对于其他数据集，使用原始训练逻辑
        id = get_run_id(args)

        dataset = args.dataset
        window_size = args.lookback
        spec_res = args.spec_res
        normalize = args.normalize
        n_epochs = args.epochs
        batch_size = args.bs
        init_lr = args.init_lr
        val_split = args.val_split
        shuffle_dataset = args.shuffle_dataset
        use_cuda = args.use_cuda
        print_every = args.print_every
        log_tensorboard = args.log_tensorboard
        group_index = args.group[0]
        index = args.group[2:]
        args_summary = str(args.__dict__)
        print(args_summary)

        explicit_validation_data = None

        if dataset == 'SMD':
            output_path = resolve_manual_output_root(dataset, group=args.group)
            (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
        elif dataset in NASA_TELEMETRY_SEQUENCE_DATASETS:
            output_path = resolve_manual_output_root(dataset)
            x_train, explicit_validation_data, x_test, y_test = get_nasa_telemetry_sequence_data(
                dataset,
                val_ratio=val_split,
                normalize=normalize,
            )
        elif dataset in NASA_SEQUENCE_DATASETS:
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, y_test) = get_nasa_random_battery_data(
                dataset,
                normalize=normalize,
                nasa_battery_id=args.nasa_battery_id,
                nasa_train_batteries=args.nasa_train_batteries,
                nasa_test_batteries=args.nasa_test_batteries,
            )
        elif dataset in ['CALCE', 'CALCE2']:
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset == 'BMS':
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, y_test) = get_bms_cluster_data(normalize=normalize)
        elif dataset in {'SWAT', 'WADI'}:
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset == CH_BATTERY_DATASET_NAME:
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, _), ch_battery_split_meta = get_ch_battery_lfp_discharge_data(
                root=args.ch_battery_root,
                normalize=normalize,
                train_ratio=args.ch_battery_train_ratio,
                seed=args.seed,
                preprocessed_dir=args.ch_battery_preprocessed_dir,
            )
            y_test = ch_battery_split_meta["test_metadata"]
        elif dataset == TSINGHUA_EV_DATASET_NAME:
            output_path = resolve_manual_output_root(dataset)
            (x_train, _), (x_test, y_test), tsinghua_split_meta = get_tsinghua_ev_data(
                root=args.tsinghua_ev_root,
                normalize=normalize,
                train_ratio=args.tsinghua_ev_train_ratio,
                validation_ratio=args.tsinghua_ev_validation_ratio,
                max_train_samples=args.tsinghua_ev_max_train_samples,
                max_validation_samples=args.tsinghua_ev_max_validation_samples,
                max_test_samples_per_class=args.tsinghua_ev_max_test_samples_per_class,
                seed=args.seed,
            )
        else:
            raise Exception(f'Dataset "{dataset}" not available.')

        log_dir = f'{output_path}/logs'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_path = f"{output_path}/{id}"
        plan_output_dir = os.environ.get("PLAN_OUTPUT_DIR", "")
        if plan_output_dir:
            save_path = plan_output_dir

        nasa_train_tensors = None
        bms_train_tensors = None
        ch_battery_train_tensors = None
        tsinghua_train_tensors = None
        tsinghua_validation_tensors = None
        explicit_validation_tensor = None
        segmented_train_tensors = None
        if dataset in NASA_SEQUENCE_DATASETS and isinstance(x_train, dict):
            nasa_train_tensors = {battery_name: _to_tensor_sequence_container(battery_data)
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
            if explicit_validation_data is not None:
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
        if dataset in NASA_SEQUENCE_DATASETS and isinstance(x_test, dict):
            nasa_test_tensors = {battery_name: _to_tensor_sequence_container(battery_data)
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

        target_dims = get_target_dims(dataset)
        if target_dims is None:
            out_dim = n_features
            print(f"Will forecast and reconstruct all {n_features} input features")
        elif type(target_dims) == int:
            print(f"Will forecast and reconstruct input feature: {target_dims}")
            out_dim = 1
        else:
            print(f"Will forecast and reconstruct input features: {target_dims}")
            out_dim = len(target_dims)

        validation_dataset = None

        if segmented_train_tensors is not None:
            train_dataset = _build_concat_window_dataset(
                segmented_train_tensors,
                window_size,
                target_dims,
                window_stride=args.window_stride,
            )
            if explicit_validation_tensor is not None:
                validation_dataset = _build_concat_window_dataset(
                    explicit_validation_tensor,
                    window_size,
                    target_dims,
                    window_stride=args.window_stride,
                )
        elif dataset in NASA_SEQUENCE_DATASETS and nasa_train_tensors is not None:
            train_sub_datasets = []
            for battery_tensor in nasa_train_tensors.values():
                train_sub_datasets.append(
                    _build_concat_window_dataset(
                        battery_tensor,
                        window_size,
                        target_dims,
                        window_stride=args.window_stride,
                    )
                )
            if len(train_sub_datasets) == 1:
                train_dataset = train_sub_datasets[0]
            else:
                train_dataset = torch.utils.data.ConcatDataset(train_sub_datasets)
        elif dataset == "BMS" and bms_train_tensors is not None:
            train_sub_datasets = [
                SlidingWindowDataset(cluster_tensor, window_size, target_dims, stride=args.window_stride)
                for cluster_tensor in bms_train_tensors.values()
            ]
            train_dataset = torch.utils.data.ConcatDataset(train_sub_datasets)
        elif dataset == CH_BATTERY_DATASET_NAME and ch_battery_train_tensors is not None:
            train_sub_datasets = [
                SlidingWindowDataset(sample_tensor, window_size, target_dims, stride=args.window_stride)
                for sample_tensor in ch_battery_train_tensors.values()
                if len(sample_tensor) > window_size
            ]
            train_dataset = torch.utils.data.ConcatDataset(train_sub_datasets)
        elif dataset == TSINGHUA_EV_DATASET_NAME and tsinghua_train_tensors is not None:
            train_sub_datasets = [
                SlidingWindowDataset(sample_tensor, window_size, target_dims, stride=args.window_stride)
                for sample_tensor in tsinghua_train_tensors.values()
                if len(sample_tensor) > window_size
            ]
            train_dataset = torch.utils.data.ConcatDataset(train_sub_datasets)
            validation_sub_datasets = [
                SlidingWindowDataset(sample_tensor, window_size, target_dims, stride=args.window_stride)
                for sample_tensor in tsinghua_validation_tensors.values()
                if len(sample_tensor) > window_size
            ]
            validation_dataset = torch.utils.data.ConcatDataset(validation_sub_datasets)
        else:
            train_dataset = SlidingWindowDataset(
                x_train,
                window_size,
                target_dims,
                stride=args.window_stride,
            )
            if explicit_validation_tensor is not None:
                validation_dataset = SlidingWindowDataset(
                    explicit_validation_tensor,
                    window_size,
                    target_dims,
                    stride=args.window_stride,
                )

        if dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"} and nasa_train_tensors is not None:
            _print_nasa_random_window_summary(
                "train",
                nasa_train_tensors,
                window_size,
                window_stride=args.window_stride,
                val_split=val_split,
            )
            _print_nasa_random_window_summary(
                "test",
                nasa_test_tensors,
                window_size,
                window_stride=args.window_stride,
            )
            test_dataset = None
        elif dataset == CH_BATTERY_DATASET_NAME and ch_battery_test_tensors is not None:
            test_sub_datasets = [
                SlidingWindowDataset(sample_tensor, window_size, target_dims, stride=args.window_stride)
                for sample_tensor in ch_battery_test_tensors.values()
                if len(sample_tensor) > window_size
            ]
            test_dataset = torch.utils.data.ConcatDataset(test_sub_datasets)
        elif dataset == TSINGHUA_EV_DATASET_NAME and tsinghua_test_tensors is not None:
            test_sub_datasets = [
                SlidingWindowDataset(sample_tensor, window_size, target_dims, stride=args.window_stride)
                for sample_tensor in tsinghua_test_tensors.values()
                if len(sample_tensor) > window_size
            ]
            test_dataset = torch.utils.data.ConcatDataset(test_sub_datasets)
        else:
            if segmented_test_tensors is not None:
                test_dataset = _build_concat_window_dataset(
                    segmented_test_tensors,
                    window_size,
                    target_dims,
                    window_stride=args.window_stride,
                )
            elif is_sequence_container(x_test):
                test_dataset = _build_concat_window_dataset(
                    x_test,
                    window_size,
                    target_dims,
                    window_stride=args.window_stride,
                )
            else:
                test_dataset = SlidingWindowDataset(
                    x_test,
                    window_size,
                    target_dims,
                    stride=args.window_stride,
                )

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            batch_size,
            val_split,
            shuffle_dataset,
            test_dataset=test_dataset,
            val_dataset=validation_dataset,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        preprocessing_seconds = time.perf_counter() - runtime_started

        model = build_model(args, n_features, window_size, out_dim, target_dims=target_dims)
        model = maybe_compile_model(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        trainer = build_trainer(
            model,
            optimizer,
            args,
            window_size,
            n_features,
            target_dims,
            save_path,
            log_dir,
            args_summary,
        )
        maybe_resume_trainer(trainer, args)

        trainer.fit(train_loader, val_loader)

        plot_losses(trainer.losses, save_path=save_path, plot=False)

        if os.environ.get("SPEED_BENCHMARK_TRAIN_ONLY", "").lower() in {"1", "true", "yes"}:
            print("Skipping test/predict because SPEED_BENCHMARK_TRAIN_ONLY is set.")
            raise SystemExit(0)

        # NASA 实体数据集单独评估测试实体
        if dataset in NASA_SEQUENCE_DATASETS and nasa_test_tensors is not None:
            loader_options = resolve_dataloader_options(
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )
            battery_test_losses = {}
            for battery_name, battery_tensor in nasa_test_tensors.items():
                battery_test_dataset = _build_concat_window_dataset(
                    battery_tensor,
                    window_size,
                    target_dims,
                    window_stride=args.window_stride,
                )
                battery_test_loader = torch.utils.data.DataLoader(
                    battery_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_options,
                )
                battery_test_losses[battery_name] = trainer.evaluate(battery_test_loader)
                print(f"[{battery_name}] Test forecast loss: {battery_test_losses[battery_name][0]:.5f}")
                print(f"[{battery_name}] Test reconstruction loss: {battery_test_losses[battery_name][1]:.5f}")
                print(f"[{battery_name}] Test total loss: {battery_test_losses[battery_name][2]:.5f}")

            mean_test_loss = np.mean(np.array(list(battery_test_losses.values()), dtype=np.float32), axis=0)
            print(f"Mean test forecast loss: {mean_test_loss[0]:.5f}")
            print(f"Mean test reconstruction loss: {mean_test_loss[1]:.5f}")
            print(f"Mean test total loss: {mean_test_loss[2]:.5f}")
        elif dataset == "BMS" and bms_test_tensors is not None:
            loader_options = resolve_dataloader_options(
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )
            cluster_test_losses = {}
            for cluster_name, cluster_tensor in bms_test_tensors.items():
                cluster_test_dataset = SlidingWindowDataset(
                    cluster_tensor,
                    window_size,
                    target_dims,
                    stride=args.window_stride,
                )
                cluster_test_loader = torch.utils.data.DataLoader(
                    cluster_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_options,
                )
                cluster_test_losses[cluster_name] = trainer.evaluate(cluster_test_loader)
                print(f"[{cluster_name}] Test forecast loss: {cluster_test_losses[cluster_name][0]:.5f}")
                print(f"[{cluster_name}] Test reconstruction loss: {cluster_test_losses[cluster_name][1]:.5f}")
                print(f"[{cluster_name}] Test total loss: {cluster_test_losses[cluster_name][2]:.5f}")

            mean_test_loss = np.mean(np.array(list(cluster_test_losses.values()), dtype=np.float32), axis=0)
            print(f"Mean test forecast loss: {mean_test_loss[0]:.5f}")
            print(f"Mean test reconstruction loss: {mean_test_loss[1]:.5f}")
            print(f"Mean test total loss: {mean_test_loss[2]:.5f}")
        elif dataset == CH_BATTERY_DATASET_NAME and ch_battery_test_tensors is not None:
            loader_options = resolve_dataloader_options(
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )
            sample_test_losses = {}
            for sample_id, sample_tensor in ch_battery_test_tensors.items():
                sample_test_dataset = SlidingWindowDataset(
                    sample_tensor,
                    window_size,
                    target_dims,
                    stride=args.window_stride,
                )
                sample_test_loader = torch.utils.data.DataLoader(
                    sample_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_options,
                )
                sample_test_losses[sample_id] = trainer.evaluate(sample_test_loader)

            mean_test_loss = np.mean(np.array(list(sample_test_losses.values()), dtype=np.float32), axis=0)
            print(f"Mean test forecast loss: {mean_test_loss[0]:.5f}")
            print(f"Mean test reconstruction loss: {mean_test_loss[1]:.5f}")
            print(f"Mean test total loss: {mean_test_loss[2]:.5f}")
        elif dataset == TSINGHUA_EV_DATASET_NAME and tsinghua_test_tensors is not None:
            test_loss = trainer.evaluate(test_loader)
            print(f"Tsinghua EV test total loss: {test_loss[2]:.5f}")
        else:
            test_loss = trainer.evaluate(test_loader)
            print(f"Test forecast loss: {test_loss[0]:.5f}")
            print(f"Test reconstruction loss: {test_loss[1]:.5f}")
            print(f"Test total loss: {test_loss[2]:.5f}")

        # POT 参数建议
        level_q_dict = {
            "MSL": (0.90, 0.001),
            "SMAP": (0.90, 0.001),
            "SMD-1": (0.9950, 0.001),
            "SMD-2": (0.9925, 0.001),
            "SMD-3": (0.9999, 0.001),
            "NASA_RANDOM_CHARGE": (0.99, 0.001),
            "NASA_RANDOM_DISCHARGE": (0.99, 0.001),
            "CALCE": (0.95, 0.01),   # 为CALCE调整参数以适应无监督设置
            "CALCE2": (0.90, 0.01),   # 为CALCE2调整参数以适应无监督设置
            "BMS": (0.99, 0.001),      # BMS数据集参数
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
        reg_level_dict = {
            "MSL": 0,
            "SMAP": 0,
            "SMD-1": 1,
            "SMD-2": 1,
            "SMD-3": 1,
            "NASA_RANDOM_CHARGE": 0,
            "NASA_RANDOM_DISCHARGE": 0,
            "CALCE": 0,
            "BMS": 0,
            "SWAT": 0,
            "WADI": 0,
            CH_BATTERY_DATASET_NAME: 0,
            TSINGHUA_EV_DATASET_NAME: 0,
        }
        key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
        reg_level = reg_level_dict[key]

        args_path = f"{save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)

        trainer.load(f"{save_path}/model.pt")
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
            "use_physical_response_score": getattr(args, "use_physical_response_score", False),
            "physical_response_config": resolve_physical_state_config(args),
            "physical_response_max_weight": getattr(args, "physical_response_max_weight", 0.35),
            "use_relation_change_score": getattr(args, "use_relation_change_score", False),
            "relation_change_weight": getattr(args, "relation_change_weight", 0.2),
            "relation_change_fusion_mode": getattr(args, "relation_change_fusion_mode", "linear_legacy"),
            "relation_change_mode": getattr(args, "relation_change_mode", "consecutive_js"),
            "predict_batch_size": getattr(args, "predict_batch_size", 128),
            "predict_num_workers": getattr(args, "predict_num_workers", 2),
            "predict_pin_memory": getattr(args, "predict_pin_memory", True),
            "use_cuda": getattr(args, "use_cuda", True),
            "window_stride": getattr(args, "window_stride", 1),
            "use_event_consistency": args.use_event_consistency,
            "event_low_ratio": args.event_low_ratio,
            "event_min_length": args.event_min_length,
            "reg_level": reg_level,
            "save_path": save_path,
            "preprocessing_seconds": float(preprocessing_seconds),
            "training_runtime": dict(getattr(trainer, "runtime_stats", {})),
            "model_parameters": int(sum(parameter.numel() for parameter in trainer.model.parameters())),
        }
        best_model = trainer.model

        if dataset == CH_BATTERY_DATASET_NAME:
            # 轻量预测器只服务于 CH-BATTERY，其他数据集不应因该可选模块
            # 未安装或被移除而在训练入口导入阶段失败。
            from src.runners.predict_ch_battery_light import run_light_predict

            run_light_predict(
                model_dir=save_path,
                batch_size=getattr(args, "predict_batch_size", 128),
                num_workers=getattr(args, "predict_num_workers", 2),
                pin_memory=getattr(args, "predict_pin_memory", True),
                no_cuda=not bool(getattr(args, "use_cuda", True)),
            )
            raise SystemExit(0)

        if dataset in NASA_SEQUENCE_DATASETS and nasa_test_tensors is not None:
            total_batteries = len(nasa_test_tensors)
            for idx, (battery_name, battery_tensor) in enumerate(nasa_test_tensors.items(), 1):
                print(f"[{dataset}] Predicting battery {idx}/{total_batteries}: {battery_name}")
                battery_save_path = save_path if len(nasa_test_tensors) == 1 else os.path.join(save_path, f"battery_{battery_name}")
                if len(nasa_test_tensors) > 1:
                    os.makedirs(battery_save_path, exist_ok=True)

                battery_prediction_args = dict(prediction_args)
                battery_prediction_args["save_path"] = battery_save_path
                predictor = Predictor(
                    best_model,
                    window_size,
                    n_features,
                    battery_prediction_args,
                )

                battery_label = None
                if isinstance(y_test, dict):
                    raw_label = y_test.get(battery_name)
                    if raw_label is not None:
                        battery_label = raw_label

                train_reference = nasa_train_tensors if nasa_train_tensors is not None else x_train
                predictor.predict_anomalies(train_reference, battery_tensor, battery_label)

                del predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            print(f"{dataset}按电池输出已生成（联合训练、分电池测试）")
            regime_report = save_nasa_regime_probe(
                save_path,
                best_model,
                nasa_train_tensors,
                nasa_test_tensors,
                window_size,
            )
            print(f"[{dataset}] regime probe: {regime_report}")
        elif dataset == "BMS" and bms_test_tensors is not None:
            total_clusters = len(bms_test_tensors)
            train_reference = bms_train_tensors if bms_train_tensors is not None else x_train

            # 预计算训练数据分数（所有聚类共享，避免重复模型推理）
            cache_predictor = Predictor(
                best_model,
                window_size,
                n_features,
                dict(prediction_args, save_path=save_path),
            )
            cached_train_df = cache_predictor.get_score_for_sequences(train_reference)
            del cache_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for idx, (cluster_name, cluster_tensor) in enumerate(bms_test_tensors.items(), 1):
                print(f"[BMS] Predicting cluster {idx}/{total_clusters}: {cluster_name}")
                cluster_save_path = os.path.join(save_path, cluster_name)
                os.makedirs(cluster_save_path, exist_ok=True)

                cluster_prediction_args = dict(prediction_args)
                cluster_prediction_args["save_path"] = cluster_save_path
                predictor = Predictor(
                    best_model,
                    window_size,
                    n_features,
                    cluster_prediction_args,
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
                    cached_train_pred_df=cached_train_df,
                )

                del predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            print("BMS按簇输出已生成（联合训练、分簇测试，训练基线缓存复用）")
            operational_report = save_bms_operational_report(
                save_path,
                bms_test_tensors,
                window_size,
                window_stride=args.window_stride,
            )
            print(f"[BMS] known-normal false alarms and stability: {operational_report}")
        elif dataset == CH_BATTERY_DATASET_NAME and ch_battery_test_tensors is not None:
            total_samples = len(ch_battery_test_tensors)
            train_reference = ch_battery_train_tensors if ch_battery_train_tensors is not None else x_train

            cache_predictor = Predictor(
                best_model,
                window_size,
                n_features,
                dict(prediction_args, save_path=save_path),
            )
            cached_train_df = cache_predictor.get_score_for_sequences(train_reference)
            del cache_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sample_rows = []
            for idx, (sample_id, sample_tensor) in enumerate(ch_battery_test_tensors.items(), 1):
                print(f"[{dataset}] Predicting sample {idx}/{total_samples}: {sample_id}")
                sample_save_path = os.path.join(save_path, sample_id)
                os.makedirs(sample_save_path, exist_ok=True)

                sample_prediction_args = dict(prediction_args)
                sample_prediction_args["save_path"] = sample_save_path
                predictor = Predictor(
                    best_model,
                    window_size,
                    n_features,
                    sample_prediction_args,
                )
                predictor.predict_anomalies(
                    train_reference,
                    sample_tensor,
                    None,
                    cached_train_pred_df=cached_train_df,
                )

                test_pred_df = pd.read_pickle(f"{sample_save_path}/test_output.pkl")
                sample_meta = y_test.get(sample_id, {}) if isinstance(y_test, dict) else {}
                score_row = dict(sample_meta)
                score_row.update(
                    aggregate_ch_battery_sample_scores(
                        test_pred_df,
                        topk_ratio=args.ch_battery_topk_ratio,
                    )
                )
                sample_rows.append(score_row)

                del predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            _, sample_summary = save_ch_battery_sample_level_reports(
                save_path,
                sample_rows,
                score_field=args.ch_battery_sample_score,
            )
            print(f"[{dataset}] sample-level summary: {sample_summary}")
        elif dataset == TSINGHUA_EV_DATASET_NAME and tsinghua_test_tensors is not None:
            from sklearn.metrics import (
                auc,
                average_precision_score,
                balanced_accuracy_score,
                f1_score,
                precision_score,
                precision_recall_curve,
                recall_score,
                roc_auc_score,
            )

            predictor = Predictor(best_model, window_size, n_features, prediction_args)
            validation_window_scores = predictor.get_sample_map_scores(
                tsinghua_validation_tensors,
                calibrate_relation=bool(args.use_relation_change_score),
            )
            validation_value_only_window_scores = dict(predictor.last_sample_value_only_scores)
            calibration_scores, _, _ = aggregate_sample_scores(
                validation_window_scores,
                tsinghua_split_meta["validation_metadata"],
                top_ratio=args.sample_score_top_ratio,
            )
            calibration_value_only_scores, _, _ = aggregate_sample_scores(
                validation_value_only_window_scores,
                tsinghua_split_meta["validation_metadata"],
                top_ratio=args.sample_score_top_ratio,
            )
            test_window_scores = predictor.get_sample_map_scores(tsinghua_test_tensors)
            test_value_only_window_scores = dict(predictor.last_sample_value_only_scores)
            sample_scores, sample_labels, sample_ids = aggregate_sample_scores(
                test_window_scores,
                y_test,
                top_ratio=args.sample_score_top_ratio,
            )
            value_only_sample_scores, value_only_labels, value_only_ids = aggregate_sample_scores(
                test_value_only_window_scores,
                y_test,
                top_ratio=args.sample_score_top_ratio,
            )
            if not np.array_equal(sample_labels, value_only_labels) or sample_ids != value_only_ids:
                raise RuntimeError("Paired Tsinghua value-only scores are not sample-aligned")
            threshold = float(np.quantile(calibration_scores, 0.99))
            value_only_threshold = float(np.quantile(calibration_value_only_scores, 0.99))
            sample_predictions = (sample_scores >= threshold).astype(np.int64)
            value_only_predictions = (value_only_sample_scores >= value_only_threshold).astype(np.int64)
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
                "sample_count": int(len(sample_ids)),
                "normal_sample_count": int(np.sum(sample_labels == 0)),
                "abnormal_sample_count": int(np.sum(sample_labels == 1)),
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
                "model_parameters": int(sum(parameter.numel() for parameter in best_model.parameters())),
                "inference_efficiency": dict(predictor.last_scoring_stats),
                "physical_response_mae_by_term": dict(predictor.last_physical_response_term_summary),
                "score_calibration": predictor.get_calibration_summary(),
            }
            if args.use_relation_change_score:
                report.update({
                    "value_only_average_precision": value_only_average_precision,
                    "average_precision_delta_vs_value_only": (
                        average_precision - value_only_average_precision
                    ),
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
            split_report = {
                key: value for key, value in tsinghua_split_meta.items()
                if key.endswith("_count") or key.endswith("_ratio_normal")
                or key in {"label_level", "vehicle_identity_available", "split_warning"}
            }
            with open(os.path.join(save_path, "dataset_split.json"), "w") as handle:
                json.dump(split_report, handle, indent=2)
            print(f"[{dataset}] snippet-level metrics: {report}")
        else:
            predictor = Predictor(
                best_model,
                window_size,
                n_features,
                prediction_args,
            )

            label = y_test
            calibration_reference = explicit_validation_tensor if explicit_validation_tensor is not None else x_train
            test_reference = segmented_test_tensors if segmented_test_tensors is not None else x_test
            predictor.predict_anomalies(calibration_reference, test_reference, label)

        # 保存训练配置
        args_path = f"{save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)
