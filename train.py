import json
from datetime import datetime

import torch.nn as nn

from args import get_parser
from mtad_gat import Enhanced_MTADGAT
from prediction import Predictor
from training import Trainer
from utils import *

def train_single_entity(entity_name, args):
    """
    为单个实体训练模型
    """
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

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
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'CALCE':
        output_path = f'output/{dataset}/{entity_name}'
        (x_train, _), (x_test, y_test) = load_calce_entity_data(entity_name)
    else:
        raise ValueError(f"Unsupported dataset for single entity training: {dataset}")

    if x_train is None or x_test is None:
        print(f"Missing data for entity {entity_name}, skipping...")
        return

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
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

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = Enhanced_MTADGAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        correlation_aware=args.correlation_aware,
        use_transformer=args.use_transformer,
        top_k=args.top_k,
        attention_top_k=args.attention_top_k,
        corr_dim=args.corr_dim,
        corr_alpha=args.corr_alpha,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
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
        "CALCE": (0.99, 0.001),
        "CALCE2": (0.99, 0.001)
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
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
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
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
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
    
    # 创建模型
    model = Enhanced_MTADGAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        correlation_aware=args.correlation_aware,
        use_transformer=args.use_transformer,
        top_k=args.top_k,
        attention_top_k=args.attention_top_k,
        corr_dim=args.corr_dim,
        corr_alpha=args.corr_alpha,
    )
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    
    # 创建训练器
    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
    
    # 开始轮流训练 - 每个epoch使用不同实体的数据
    trainer.fit_round_robin(train_entity_data, window_size, target_dims, val_split, shuffle_dataset)
    
    plot_losses(trainer.losses, save_path=save_path, plot=False)
    
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存模型
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
    args = parser.parse_args()
    # 实现CALCE/CALCE2的通用模型训练
    if args.dataset in ['CALCE', 'CALCE2']:
        # 对于CALCE/CALCE2数据集，训练一个通用模型
        train_universal_model(args)
    else:
        # 对于其他数据集，使用原始训练逻辑
        id = datetime.now().strftime("%d%m%Y_%H%M%S")

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
        elif dataset == 'NASA':
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset in ['CALCE', 'CALCE2']:
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        elif dataset == 'BMS':
            output_path = f'output/{dataset}'
            (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        else:
            raise Exception(f'Dataset "{dataset}" not available.')

        log_dir = f'{output_path}/logs'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_path = f"{output_path}/{id}"

        x_train = torch.from_numpy(x_train).float()
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

        train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
        test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
        )

        model = Enhanced_MTADGAT(
            n_features,
            window_size,
            out_dim,
            kernel_size=args.kernel_size,
            use_gatv2=args.use_gatv2,
            feat_gat_embed_dim=args.feat_gat_embed_dim,
            time_gat_embed_dim=args.time_gat_embed_dim,
            gru_n_layers=args.gru_n_layers,
            gru_hid_dim=args.gru_hid_dim,
            forecast_n_layers=args.fc_n_layers,
            forecast_hid_dim=args.fc_hid_dim,
            recon_n_layers=args.recon_n_layers,
            recon_hid_dim=args.recon_hid_dim,
            dropout=args.dropout,
            alpha=args.alpha,
            correlation_aware=args.correlation_aware,
            use_transformer=args.use_transformer,
            top_k=args.top_k,
            attention_top_k=args.attention_top_k,
            corr_dim=args.corr_dim,
            corr_alpha=args.corr_alpha,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
        forecast_criterion = nn.MSELoss()
        recon_criterion = nn.MSELoss()

        trainer = Trainer(
            model,
            optimizer,
            window_size,
            n_features,
            target_dims,
            n_epochs,
            batch_size,
            init_lr,
            forecast_criterion,
            recon_criterion,
            use_cuda,
            save_path,
            log_dir,
            print_every,
            log_tensorboard,
            args_summary
        )

        trainer.fit(train_loader, val_loader)

        plot_losses(trainer.losses, save_path=save_path, plot=False)

        # Check test loss
        test_loss = trainer.evaluate(test_loader)
        print(f"Test forecast loss: {test_loss[0]:.5f}")
        print(f"Test reconstruction loss: {test_loss[1]:.5f}")
        print(f"Test total loss: {test_loss[2]:.5f}")

        # TODO
        # Some suggestions for POT args
        level_q_dict = {
            "SMAP": (0.90, 0.005),
            "MSL": (0.90, 0.001),
            "SMD-1": (0.9950, 0.001),
            "SMD-2": (0.9925, 0.001),
            "SMD-3": (0.9999, 0.001),
            "NASA": (0.99, 0.001),
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
        reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "NASA": 0, "CALCE": 0, "BMS": 0}
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
            "reg_level": reg_level,
            "save_path": save_path,
        }
        best_model = trainer.model
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