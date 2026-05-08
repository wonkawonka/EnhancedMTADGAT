import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    # -- enhanced params ---
    parser.add_argument("--use_transformer", type=str2bool, default=False,help="目前是在gru前加transformer encoder")
    parser.add_argument("--attention_sparse", type=str2bool, default=False,help="是否使用稀疏注意力矩阵")
    parser.add_argument('--attention_top_k', type=int, default=10, help='特征图注意力top_k连接')
    # Simplified model option - only feature attention + transformer
    parser.add_argument("--feature_att_trans", type=str2bool, default=False,
                        help="仅使用特征注意力和Transformer，跳过时间注意力和GRU")
    # Multi-scale convolution
    parser.add_argument("--multi_scale_mode", type=str, default="none",
                        choices=["none", "basic", "progressive"],
                        help="多尺度卷积模式: none=单尺度, basic=并行拼接, progressive=渐进式融合")
    parser.add_argument("--multi_scale_dilations", type=str, default="1,2,4",
                        help="因果膨胀卷积的膨胀率列表，逗号分隔，如 '1,2,4' (kernel_size 固定为 3)")
    # Spectral Residual cleaning
    parser.add_argument("--apply_sr_cleaning", type=str2bool, default=False,help="是否在预处理阶段应用谱残差异常检测和清洗（CALCE固定清洗）")
    # Transformer layers
    parser.add_argument("--trans_enc_layers", type=int, default=2, help="Transformer encoder的层数（这里是第一次改的gru前的transformer）")
    parser.add_argument("--transformer_ff_mult", type=float, default=2.0, help="轻量 Transformer 的前馈层扩展倍数")
    parser.add_argument("--transformer_norm_first", type=str2bool, default=True, help="轻量 Transformer 是否使用 Pre-LN")
    parser.add_argument("--use_revin", type=str2bool, default=False, help="是否启用 RevIN 以缓解跨工况分布偏移")
    parser.add_argument("--revin_affine", type=str2bool, default=True, help="RevIN 是否启用可学习仿射参数")
    parser.add_argument("--use_regime_condition", type=str2bool, default=False, help="是否启用工况感知关系建模")
    parser.add_argument("--regime_emb_dim", type=int, default=32, help="工况嵌入维度")
    parser.add_argument(
        "--regime_condition_mode",
        type=str,
        default="transformer_residual",
        choices=["transformer_residual", "feature_gat", "temporal_gat", "fusion"],
        help="工况条件化注入位置: transformer_residual=工况感知的Transformer残差增强, 其余为旧版注入方式",
    )
    parser.add_argument(
        "--regime_stat_features",
        type=str,
        default="mean,std,last,delta",
        help="用于构造工况嵌入的窗口统计量，逗号分隔",
    )
    parser.add_argument("--use_hier_consistency", type=str2bool, default=False, help="是否启用 BMS 层级残差双分支异常检测")
    parser.add_argument("--hier_score_weight", type=float, default=0.5, help="残差分支在最终异常分数融合中的权重")
    # -- Data params ---
    parser.add_argument(
        '--dataset',
        type=str.upper,
        default='NASA',
        choices=[
            'SMD',
            'SMAP',
            'MSL',
            'NASA',
            'NASA_RANDOM_CHARGE',
            'NASA_RANDOM_DISCHARGE',
            'CALCE',
            'CALCE2',
            'BMS',
        ],
        help='dataset name'
    )
    parser.add_argument("--group", type=str, default="1-1", help="指定SMD数据集中具体机器编号. <group_index>-<index>")
    parser.add_argument("--nasa_battery_id", type=str, default="", help="指定单个NASA电池ID，如 B0018")
    parser.add_argument("--nasa_train_batteries", type=str, default="", help="NASA训练电池ID列表，逗号分隔，如 B0005,B0006,B0007")
    parser.add_argument("--nasa_test_batteries", type=str, default="", help="NASA测试电池ID列表，逗号分隔，如 B0018")
    parser.add_argument(
        "--nasa_random_split_mode",
        type=str,
        default="fold",
        choices=["fold", "single_battery_split"],
        help="NASA_RANDOM 划分模式: fold=整块电池划分, single_battery_split=单电池内部 80/20 划分",
    )
    parser.add_argument("--lookback", type=int, default=100,help="窗口大小（window size），即模型输入的时间步数")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False,help="特殊分辨率设置，目前未在代码中详细使用")

    # -- Model params ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="mtad_gat",
        help="模型名称。mtad_gat 为 baseline，mtad_gat_c3 为第三章模型，mtad_gat_c4 为第四章层级一致性增强版本",
    )
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None,help="节点特征嵌入维度（GAT 层）")
    parser.add_argument("--time_gat_embed_dim", type=int, default=None,help="时间序列嵌入维度（GAT 层）")

    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1,help="GRU 的层数")
    parser.add_argument("--gru_hid_dim", type=int, default=150,help="GRU 隐藏层的维度")

    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2,help="LeakyReLU 的斜率参数，用于 GAT 的注意力机制")

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1,help="验证集的占比")
    parser.add_argument("--bs", type=int, default=256,help="batch size")
    parser.add_argument("--init_lr", type=float, default=1e-3,help="初始学习率")
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True,help="是否对数据集进行打乱")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1,help="每多少个 epoch 打印一次训练信息")
    parser.add_argument("--log_tensorboard", type=str2bool, default=True,help="是否将训练日志写入 TensorBoard")

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False, help="是否对异常分数进行归一化")
    parser.add_argument("--use_mov_av", type=str2bool, default=False,help="是否使用滑动平均来计算异常分数")
    parser.add_argument("--gamma", type=float, default=1,help="异常评分公式中的权重因子，用于平衡预测误差和重构误差")
    parser.add_argument(
        "--score_fusion_mode",
        type=str,
        default="fixed",
        choices=["fixed", "quality_aware"],
        help="异常分数融合方式: fixed=固定 pred+gamma*recon, quality_aware=质量感知融合",
    )
    parser.add_argument("--use_event_consistency", type=str2bool, default=False, help="是否启用双阈值与持续性约束的事件级异常判别")
    parser.add_argument("--event_low_ratio", type=float, default=0.5, help="低阈值在训练分数中位数与高阈值之间的插值比例，取值越小越严格")
    parser.add_argument("--event_min_length", type=int, default=3, help="保留异常事件所需的最小持续长度")
    parser.add_argument("--level", type=float, default=None,help="POT 方法中的初始阈值")
    parser.add_argument("--q", type=float, default=None,help="POT 方法中的后续误风险参数，表示可接受多大概率误报")
    parser.add_argument("--dynamic_pot", type=str2bool, default=False,help="是否使用动态阈值")

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子，用于实验可复现")
    parser.add_argument("--run_id", type=str, default="", help="实验输出目录ID，留空则自动使用时间戳")
    parser.add_argument("--resume", type=str2bool, default=False, help="是否从输出目录中的 last_checkpoint.pt 继续训练")

    return parser
