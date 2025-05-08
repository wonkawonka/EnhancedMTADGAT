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
    # -- optimize params ---
    parser.add_argument("--dynamic_graph", type=str2bool, default=False)
    parser.add_argument("--correlation_aware", type=str2bool, default=False)
    parser.add_argument("--use_transformer", type=str2bool, default=True)
    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="1-1", help="指定SMD数据集中具体机器编号. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=100,help="窗口大小（window size），即模型输入的时间步数")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False,help="特殊分辨率设置，目前未在代码中详细使用")

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None,help="节点特征嵌入维度（GAT 层）")
    parser.add_argument("--time_gat_embed_dim", type=int, default=None,help="时间序列嵌入维度（GAT 层）")
    # Transformer layers
    parser.add_argument("--trans_enc_layers", type=int, default=2,help="Transformer encoder的层数")
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
    parser.add_argument("--level", type=float, default=None,help="POT 方法中的初始阈值")
    parser.add_argument("--q", type=float, default=None,help="POT 方法中的后续误风险参数，表示可接受多大概率误报")
    parser.add_argument("--dynamic_pot", type=str2bool, default=False,help="是否使用动态阈值")

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    return parser
