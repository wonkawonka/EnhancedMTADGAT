import json
import random
from datetime import datetime

import torch.nn as nn
import numpy as np
import torch

from args import get_parser
from mtad_gat import Enhanced_MTADGAT
from prediction import Predictor
from training import Trainer
from utils import *

def set_seed(seed=3407):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        attention_sparse=args.attention_sparse,  # Add the missing parameter
        corr_dim=args.corr_dim,
        corr_alpha=args.corr_alpha,
        feature_att_trans=args.feature_att_trans,  # Add the new parameter
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
    
    # 设置随机种子以确保实验可重现性
    set_seed(3407)
    
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
            attention_sparse=args.attention_sparse,  # Add the missing parameter
            corr_dim=args.corr_dim,
            corr_alpha=args.corr_alpha,
            feature_att_trans=args.feature_att_trans,  # Add the new parameter
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

        # 对于NASA数据集，添加额外的容量特征评估
        if dataset == "NASA":
            try:
                # 加载容量数据用于评估
                import pickle
                import glob
                import os
                
                # 查找NASA测试数据文件以确定使用的实体
                test_files = glob.glob("datasets/NASA/processed/NASA_*_test.pkl")
                if test_files:
                    # 根据测试数据文件名推断实体名称
                    # 例如: datasets/NASA/processed/NASA_B0049_test.pkl -> B0049
                    test_file = test_files[0]  # 使用找到的第一个实体（与数据加载一致）
                    filename = os.path.basename(test_file)
                    entity_name = filename.replace('NASA_', '').replace('_test.pkl', '')
                    capacity_file = os.path.join("datasets/NASA/processed", f"NASA_{entity_name}_capacities.pkl")
                    
                    print(f"使用实体 {entity_name} 的容量数据进行评估")
                    
                    if os.path.exists(capacity_file):
                        # 读取完整的周期级容量数据
                        with open(capacity_file, 'rb') as f:
                            all_capacities = pickle.load(f)
                        
                        # 读取测试结果
                        test_pred_df = pd.read_pickle(f"{save_path}/test_output.pkl")
                        anomaly_scores = test_pred_df['A_Score_Global'].values
                        
                        # 读取测试数据，其中已经包含了插值后的容量数据（在最后一列）
                        with open(test_file, 'rb') as f:
                            test_data = pickle.load(f)
                        
                        # 从测试数据中提取容量列（最后一列）
                        # 根据preprocess.py中的处理，列顺序是：[周期编号, 测量电压, 测量电流, 测量温度, 负载电流, 负载电压, 容量]
                        if test_data.shape[1] >= 7:
                            capacities_from_test_data = test_data[:, -1]  # 最后一列是容量
                            
                            # 确保长度一致（考虑窗口大小的影响）
                            if len(anomaly_scores) != len(capacities_from_test_data):
                                min_len = min(len(anomaly_scores), len(capacities_from_test_data))
                                anomaly_scores = anomaly_scores[:min_len]
                                capacities_from_test_data = capacities_from_test_data[:min_len]
                                print(f"调整数据长度以匹配: {min_len}")
                            
                            # 检查容量数据的有效性
                            print(f"容量数据统计: 最小值={np.min(capacities_from_test_data):.4f}, "
                                  f"最大值={np.max(capacities_from_test_data):.4f}, "
                                  f"初始值={capacities_from_test_data[0]:.4f}")
                            
                            # 使用完整的容量数据计算初始容量（与预处理阶段保持一致）
                            initial_capacity = all_capacities[0]  # 使用所有数据中的第一个周期作为初始容量
                            capacity_decay_rate = (initial_capacity - capacities_from_test_data) / initial_capacity
                            
                            print(f"基于完整数据集初始容量({initial_capacity:.4f})的容量衰减率统计: "
                                  f"最小值={np.min(capacity_decay_rate):.4f}, "
                                  f"最大值={np.max(capacity_decay_rate):.4f}")
                            
                            # 检查是否有足够的衰减（超过阈值的数据点）
                            threshold = 0.2  # 20%容量衰减阈值
                            positive_samples = np.sum(capacity_decay_rate > threshold)
                            print(f"超过{threshold*100}%容量衰减的数据点数量: {positive_samples}/{len(capacity_decay_rate)} "
                                  f"({positive_samples/len(capacity_decay_rate)*100:.2f}%)")
                            
                            # 只有在有足够的正样本时才进行评估
                            if positive_samples > 0:
                                # 创建基于容量衰减的标签
                                labels = (capacity_decay_rate > threshold).astype(int)
                                
                                # 计算ROC曲线
                                fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
                                roc_auc = auc(fpr, tpr)
                                
                                # 绘制ROC曲线
                                plot_roc_curve(fpr, tpr, roc_auc, save_path)
                                
                                # 绘制异常分数与容量关系图
                                plot_anomaly_score_vs_capacity(anomaly_scores, capacities_from_test_data, save_path)
                                
                                print(f"NASA容量特征评估完成，AUC值: {roc_auc:.4f}")
                            else:
                                print("警告: 容量衰减不足，无法进行有意义的评估")
                                # 即使没有足够正样本，我们也绘制图表以供观察
                                plt.figure(figsize=(12, 6))
                                plt.subplot(2, 1, 1)
                                plt.plot(capacities_from_test_data, label='Capacity', color='blue')
                                plt.ylabel('Capacity')
                                plt.title('Capacity Curve')
                                plt.legend()
                                
                                plt.subplot(2, 1, 2)
                                plt.plot(anomaly_scores, label='Anomaly Score', color='red')
                                plt.ylabel('Anomaly Score')
                                plt.xlabel('Time')
                                plt.title('Anomaly Score Curve')
                                plt.legend()
                                
                                plt.tight_layout()
                                if save_path:
                                    plt.savefig(f"{save_path}/anomaly_score_vs_capacity.png", bbox_inches="tight")
                                plt.show()
                                plt.close()
                                
                        else:
                            print("测试数据列数不足，无法提取容量信息")
                    else:
                        print(f"未找到容量数据文件: {capacity_file}")
                else:
                    print("未找到NASA测试数据文件")
            except Exception as e:
                print(f"NASA容量特征评估出错: {e}")
                import traceback
                traceback.print_exc()

        # Save config
        args_path = f"{save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)