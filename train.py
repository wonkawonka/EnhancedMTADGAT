import json
import os
import random
from datetime import datetime

import torch.nn as nn
import numpy as np
import torch

from args import apply_dataset_defaults, get_parser
from model_factory import build_model, resolve_model_args, resolve_physical_state_config
from prediction import Predictor
from training import Trainer
from utils import *

NASA_ENTITY_DATASETS = {"NASA", "NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}


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
        use_physical_regularization=getattr(args, "use_physical_regularization", False),
        physical_reg_config=resolve_physical_state_config(args),
        physical_reg_warmup_ratio=getattr(args, "physical_reg_warmup_ratio", 0.2),
        physical_alg_lambda=getattr(args, "physical_alg_lambda", 0.1),
        physical_smooth_lambda=getattr(args, "physical_smooth_lambda", 0.01),
        physical_transition_threshold=getattr(args, "physical_transition_threshold", 0.05),
        physical_transition_relax=getattr(args, "physical_transition_relax", 0.1),
        num_workers=getattr(args, "num_workers", 4),
        persistent_workers=getattr(args, "persistent_workers", True),
        prefetch_factor=getattr(args, "prefetch_factor", 2),
        window_stride=getattr(args, "window_stride", 1),
    )


def maybe_compile_model(model):
    if os.environ.get("DISABLE_TORCH_COMPILE", "").lower() in {"1", "true", "yes"}:
        print("Skipping torch.compile because DISABLE_TORCH_COMPILE is set.")
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
    else:  # CALCE2
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
    
    output_path = f'output/{dataset}/universal_model'
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
    
    # 创建模型。当前默认仍为 MTAD-GAT，后续 baseline 通过 model_factory 扩展。
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
    
    # 开始轮流训练 - 每个epoch使用不同实体的数据
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
                "SMAP": (0.90, 0.005),
                "MSL": (0.90, 0.001),
                "SMD-1": (0.9950, 0.001),
                "SMD-2": (0.9925, 0.001),
                "SMD-3": (0.9999, 0.001),
                "NASA": (0.99, 0.001),
                "CALCE": (0.95, 0.01),  # 为CALCE调整参数以适应无监督设置
                "CALCE2": (0.90, 0.01)  # 为CALCE2调整参数以适应无监督设置
            }
            key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
            level, q = level_q_dict[key]
            if args.level is not None:
                level = args.level
            if args.q is not None:
                q = args.q

            # Some suggestions for Epsilon args
            reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "NASA": 0, "CALCE": 0, "CALCE2": 0}
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
                "use_event_consistency": args.use_event_consistency,
                "event_low_ratio": args.event_low_ratio,
                "event_min_length": args.event_min_length,
                "use_hier_consistency": args.use_hier_consistency,
                "hier_score_weight": args.hier_score_weight,
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

        if dataset == 'SMD':
            output_path = f'output/SMD/{args.group}'
            (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
        elif dataset in ['MSL', 'SMAP']:
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset in NASA_ENTITY_DATASETS:
            output_path = f'output/{dataset}'
            if dataset == "NASA":
                (x_train, _), (x_test, y_test) = get_nasa_battery_data(
                    normalize=normalize,
                    nasa_battery_id=args.nasa_battery_id,
                    nasa_train_batteries=args.nasa_train_batteries,
                    nasa_test_batteries=args.nasa_test_batteries,
                )
            else:
                (x_train, _), (x_test, y_test) = get_nasa_random_battery_data(
                    dataset,
                    normalize=normalize,
                    nasa_battery_id=args.nasa_battery_id,
                    nasa_train_batteries=args.nasa_train_batteries,
                    nasa_test_batteries=args.nasa_test_batteries,
                )
        elif dataset in ['CALCE', 'CALCE2']:
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset == 'BMS':
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_bms_cluster_data(normalize=normalize)
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
        if dataset in NASA_ENTITY_DATASETS and isinstance(x_train, dict):
            nasa_train_tensors = {battery_name: _to_tensor_sequence_container(battery_data)
                                  for battery_name, battery_data in x_train.items()}
            first_train_battery = next(iter(nasa_train_tensors))
            x_train = _get_first_sequence_tensor(nasa_train_tensors[first_train_battery])
        elif dataset == "BMS" and isinstance(x_train, dict):
            bms_train_tensors = {cluster_name: torch.from_numpy(cluster_data).float()
                                 for cluster_name, cluster_data in x_train.items()}
            first_train_cluster = next(iter(bms_train_tensors))
            x_train = bms_train_tensors[first_train_cluster]
        else:
            x_train = torch.from_numpy(x_train).float()
        nasa_test_tensors = None
        bms_test_tensors = None
        if dataset in NASA_ENTITY_DATASETS and isinstance(x_test, dict):
            nasa_test_tensors = {battery_name: _to_tensor_sequence_container(battery_data)
                                 for battery_name, battery_data in x_test.items()}
            first_test_battery = next(iter(nasa_test_tensors))
            x_test = _get_first_sequence_tensor(nasa_test_tensors[first_test_battery])
        elif dataset == "BMS" and isinstance(x_test, dict):
            bms_test_tensors = {cluster_name: torch.from_numpy(cluster_data).float()
                                for cluster_name, cluster_data in x_test.items()}
            first_test_cluster = next(iter(bms_test_tensors))
            x_test = bms_test_tensors[first_test_cluster]
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

        if dataset in NASA_ENTITY_DATASETS and nasa_train_tensors is not None:
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
        else:
            train_dataset = SlidingWindowDataset(
                x_train,
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
        else:
            if is_sequence_container(x_test):
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
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

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

        # Check test loss
        if dataset in NASA_ENTITY_DATASETS and nasa_test_tensors is not None:
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
        else:
            test_loss = trainer.evaluate(test_loader)
            print(f"Test forecast loss: {test_loss[0]:.5f}")
            print(f"Test reconstruction loss: {test_loss[1]:.5f}")
            print(f"Test total loss: {test_loss[2]:.5f}")

        # Some suggestions for POT args
        level_q_dict = {
            "SMAP": (0.90, 0.005),
            "MSL": (0.90, 0.001),
            "SMD-1": (0.9950, 0.001),
            "SMD-2": (0.9925, 0.001),
            "SMD-3": (0.9999, 0.001),
            "NASA": (0.99, 0.001),
            "NASA_RANDOM_CHARGE": (0.99, 0.001),
            "NASA_RANDOM_DISCHARGE": (0.99, 0.001),
            "CALCE": (0.95, 0.01),   # 为CALCE调整参数以适应无监督设置
            "CALCE2": (0.90, 0.01),   # 为CALCE2调整参数以适应无监督设置
            "BMS": (0.99, 0.001)      # BMS数据集参数
        }
        key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
        level, q = level_q_dict[key]
        if args.level is not None:
            level = args.level
        if args.q is not None:
            q = args.q

        # Some suggestions for Epsilon args
        reg_level_dict = {
            "SMAP": 0,
            "MSL": 0,
            "SMD-1": 1,
            "SMD-2": 1,
            "SMD-3": 1,
            "NASA": 0,
            "NASA_RANDOM_CHARGE": 0,
            "NASA_RANDOM_DISCHARGE": 0,
            "CALCE": 0,
            "BMS": 0,
        }
        key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
        reg_level = reg_level_dict[key]

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
            "use_event_consistency": args.use_event_consistency,
            "event_low_ratio": args.event_low_ratio,
            "event_min_length": args.event_min_length,
            "use_hier_consistency": args.use_hier_consistency,
            "hier_score_weight": args.hier_score_weight,
            "reg_level": reg_level,
            "save_path": save_path,
        }
        best_model = trainer.model

        if dataset == "NASA" and nasa_test_tensors is not None:
            try:
                processed_prefix = "datasets/NASA/processed"
                train_batteries, report_test_batteries = resolve_nasa_batteries(
                    processed_prefix,
                    nasa_battery_id=args.nasa_battery_id,
                    nasa_train_batteries=args.nasa_train_batteries,
                    nasa_test_batteries=args.nasa_test_batteries,
                )

                nasa_case_summaries = []
                for battery_name, battery_tensor in nasa_test_tensors.items():
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
                            battery_label = raw_label[window_size:]

                    train_reference = nasa_train_tensors if nasa_train_tensors is not None else x_train
                    predictor.predict_anomalies(train_reference, battery_tensor, battery_label)

                    _, raw_test_data, _ = load_nasa_processed_data(processed_prefix, battery_name)
                    raw_test_data = np.asarray(raw_test_data, dtype=np.float32)
                    capacities = raw_test_data[window_size:, -1] if raw_test_data.ndim == 2 and raw_test_data.shape[1] >= 7 else None
                    cycle_numbers = raw_test_data[window_size:, 0] if raw_test_data.ndim == 2 and raw_test_data.shape[1] >= 1 else None
                    if capacities is None or cycle_numbers is None:
                        continue

                    test_pred_df = pd.read_pickle(f"{battery_save_path}/test_output.pkl")
                    case_summary = save_nasa_case_outputs(
                        battery_save_path,
                        test_pred_df,
                        capacities,
                        cycle_numbers,
                        battery_name,
                        train_batteries,
                        report_test_batteries,
                    )
                    nasa_case_summaries.append(case_summary)

                save_nasa_battery_comparison(save_path, nasa_case_summaries)

                print("NASA专用输出已生成（曲线图、案例分析、分数趋势、电池间对比）")
            except Exception as e:
                print(f"NASA专用报告生成出错: {e}")
                import traceback
                traceback.print_exc()
        elif dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"} and nasa_test_tensors is not None:
            for battery_name, battery_tensor in nasa_test_tensors.items():
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

            print(f"{dataset}按电池输出已生成（联合训练、分电池测试）")
        elif dataset == "BMS" and bms_test_tensors is not None:
            for cluster_name, cluster_tensor in bms_test_tensors.items():
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

                train_reference = bms_train_tensors if bms_train_tensors is not None else x_train
                predictor.predict_anomalies(train_reference, cluster_tensor, cluster_label)

            print("BMS按簇输出已生成（联合训练、分簇测试）")
        else:
            predictor = Predictor(
                best_model,
                window_size,
                n_features,
                prediction_args,
            )

            label = y_test[window_size:] if y_test is not None else None
            predictor.predict_anomalies(x_train, x_test, label)

        # Save config
        args_path = f"{save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)
