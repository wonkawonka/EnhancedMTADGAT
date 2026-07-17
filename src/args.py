"""定义训练和预测共用的命令行参数及数据集默认配置。"""

import argparse

from src.project_paths import resolve_dataset_root


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def apply_dataset_defaults(args):
    dataset = str(getattr(args, "dataset", "")).upper()
    default_window_stride = {
        "BMS": 4,
        "NASA_RANDOM_DISCHARGE": 2,
        "CH_BATTERY_LFP_DISCHARGE": 8,
        "TSINGHUA_EV": 4,
    }
    if getattr(args, "window_stride", None) is None:
        args.window_stride = default_window_stride.get(dataset, 1)
    return args


def get_parser():
    parser = argparse.ArgumentParser()
    # 模型结构选项
    parser.add_argument("--use_transformer", type=str2bool, default=False,help="Add transformer encoder before GRU")
    parser.add_argument("--attention_sparse", type=str2bool, default=False,help="Whether to use sparse attention matrix")
    parser.add_argument('--attention_top_k', type=int, default=10, help='Top-k connections for feature attention')
    # 简化模型选项：仅使用特征注意力和 Transformer
    parser.add_argument("--feature_att_trans", type=str2bool, default=False,
                        help="仅使用特征注意力和Transformer，跳过时间注意力和GRU")
    # 多尺度卷积选项
    parser.add_argument("--multi_scale_mode", type=str, default="none",
                        choices=["none", "basic", "progressive"],
                        help="Multi-scale convolution mode: none=single scale, basic=parallel concat, progressive=progressive fusion")
    parser.add_argument("--multi_scale_dilations", type=str, default="1,2,4",
                        help="List of dilation rates for causal dilated convolution, comma separated, e.g. '1,2,4' (kernel_size fixed to 3)")
    # 谱残差清洗选项
    parser.add_argument("--apply_sr_cleaning", type=str2bool, default=False,help="Whether to apply spectral residual anomaly detection and cleaning in preprocessing (fixed for CALCE)")
    # Transformer 层
    parser.add_argument("--trans_enc_layers", type=int, default=2, help="Number of transformer encoder layers (applied before GRU)")
    parser.add_argument("--transformer_ff_mult", type=float, default=2.0, help="Feed-forward expansion multiplier for lightweight transformer")
    parser.add_argument("--transformer_norm_first", type=str2bool, default=True, help="Whether to use Pre-LN for lightweight transformer")
    parser.add_argument("--use_revin", type=str2bool, default=False, help="Whether to enable RevIN to mitigate cross-regime distribution shift")
    parser.add_argument("--revin_affine", type=str2bool, default=True, help="Whether RevIN uses learnable affine parameters")
    parser.add_argument("--use_regime_condition", type=str2bool, default=False, help="Whether to enable learned dynamic-state conditioning")
    parser.add_argument("--regime_emb_dim", type=int, default=32, help="Dynamic-state embedding dimension")
    parser.add_argument(
        "--regime_encoder_type",
        type=str,
        default="temporal",
        choices=["temporal", "statistics"],
        help="Temporal learned encoder for the proposed model; statistics is retained only for ablation",
    )
    parser.add_argument("--regime_aux_lambda", type=float, default=0.05, help="Self-supervised dynamic descriptor loss weight")
    parser.add_argument(
        "--use_physical_state_encoding",
        type=str2bool,
        default=False,
        help="Whether to enable time-step-level physical state encoding for battery sequences (phase/I/q/V/T)",
    )
    parser.add_argument(
        "--physical_state_hidden_dim",
        type=int,
        default=32,
        help="Intermediate dimension of physical state encoding before projecting to transformer hidden space",
    )
    parser.add_argument(
        "--use_physical_regularization",
        type=str2bool,
        default=False,
        help="Whether to enable physical regularization constraints (derivative consistency + phase-aware smoothness)",
    )
    parser.add_argument(
        "--physical_reg_warmup_ratio",
        type=float,
        default=0.2,
        help="Physical regularization weight warm-up epoch ratio (linearly increasing from 0 to target)",
    )
    parser.add_argument(
        "--physical_alg_lambda",
        type=float,
        default=0.1,
        help="派生一致性约束权重",
    )
    parser.add_argument(
        "--physical_smooth_lambda",
        type=float,
        default=0.01,
        help="Phase-aware smoothness constraint weight",
    )
    parser.add_argument(
        "--physical_transition_threshold",
        type=float,
        default=0.05,
        help="基于电流近似划分阶段时的阈值",
    )
    parser.add_argument(
        "--physical_transition_relax",
        type=float,
        default=0.1,
        help="Smoothness constraint decay coefficient near phase transition points",
    )
    parser.add_argument("--use_physical_response_score", type=str2bool, default=False, help="Fuse electrical/thermal response residuals into anomaly score")
    parser.add_argument("--physical_response_max_weight", type=float, default=0.35, help="Maximum adaptive physical-response score weight")
    parser.add_argument(
        "--physical_response_terms",
        type=str,
        default="voltage_rate,temperature_rate,charge_flow,voltage_spread,temperature_spread,soc_current_coupling",
        help="Comma-separated electrical/thermal response terms used by both training regularization and inference scoring",
    )
    parser.add_argument(
        "--regime_condition_mode",
        type=str,
        default="fusion",
        choices=["transformer_residual", "feature_gat", "temporal_gat", "fusion"],
        help="动态状态条件化位置：fusion为关系融合表示上的FiLM条件化，其他选项用于消融/兼容",
    )
    parser.add_argument(
        "--regime_stat_features",
        type=str,
        default="mean,std,last,delta",
        help="Window statistics used to construct regime embedding, comma separated",
    )
    # 数据集选项
    parser.add_argument(
        '--dataset',
        type=str.upper,
        default='MSL',
        choices=[
            'SMD',
            'MSL',
            'NASA_RANDOM_CHARGE',
            'NASA_RANDOM_DISCHARGE',
            'CALCE',
            'CALCE2',
            'BMS',
            'CH_BATTERY_LFP_DISCHARGE',
            'TSINGHUA_EV',
        ],
        help='dataset name'
    )
    parser.add_argument(
        "--ch_battery_root",
        type=str,
        default=str(resolve_dataset_root("CH-BATTERY", "CH-BATTERY")),
        help="CH-BATTERY root directory",
    )
    parser.add_argument(
        "--tsinghua_ev_root",
        type=str,
        default=str(resolve_dataset_root("TSINGHUA-EV", "TSINGHUA_EV")),
        help="Tsinghua real-world EV battery dataset root directory",
    )
    parser.add_argument("--tsinghua_ev_train_ratio", type=float, default=0.7, help="Normal snippet ratio used for training")
    parser.add_argument("--tsinghua_ev_validation_ratio", type=float, default=0.15, help="Normal snippet ratio used for validation and threshold calibration")
    parser.add_argument("--sample_score_top_ratio", type=float, default=0.05, help="Top-window ratio used to aggregate each charging snippet score")
    parser.add_argument("--tsinghua_ev_max_train_samples", type=int, default=0, help="Optional normal-train cap for smoke tests; 0 uses all")
    parser.add_argument("--tsinghua_ev_max_validation_samples", type=int, default=0, help="Optional normal-validation cap for smoke tests; 0 uses all")
    parser.add_argument("--tsinghua_ev_max_test_samples_per_class", type=int, default=0, help="Optional per-class test cap for smoke tests; 0 uses all")
    parser.add_argument(
        "--ch_battery_preprocessed_dir",
        type=str,
        default="",
        help="Optional CH-BATTERY processed directory; leave empty to auto-resolve from ch_battery_root/processed/lfp_discharge",
    )
    parser.add_argument("--ch_battery_train_ratio", type=float, default=0.8, help="Normal VIN ratio used for training in CH-BATTERY")
    parser.add_argument(
        "--ch_battery_sample_score",
        type=str,
        default="score_topk_mean",
        choices=["score_topk_mean", "score_p95", "score_max", "score_mean"],
        help="Sample-level score aggregation used for CH-BATTERY reporting",
    )
    parser.add_argument(
        "--ch_battery_topk_ratio",
        type=float,
        default=0.05,
        help="Top-k ratio used when aggregating CH-BATTERY anomaly scores into sample-level scores",
    )
    parser.add_argument("--group", type=str, default="1-1", help="Specify machine ID in SMD dataset. <group_index>-<index>")
    parser.add_argument("--nasa_battery_id", type=str, default="", help="Single NASA battery ID, e.g. B0018")
    parser.add_argument("--nasa_train_batteries", type=str, default="", help="NASA training battery IDs, comma separated, e.g. B0005,B0006,B0007")
    parser.add_argument("--nasa_test_batteries", type=str, default="", help="NASA test battery IDs, comma separated, e.g. B0018")
    parser.add_argument("--lookback", type=int, default=100,help="Window size, i.e., number of input time steps")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False,help="Spectral residual setting, not used in detail in the code currently")

    # 模型选择
    parser.add_argument(
        "--model_name",
        type=str,
        default="mtad_gat",
        help="Model name: mtad_gat baseline, mtad_gat_c3_regime chapter 3, mtad_gat_c4_physics chapter 4.",
    )
    # 卷积参数
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT 参数
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None,help="Node feature embedding dimension (GAT layer)")
    parser.add_argument("--time_gat_embed_dim", type=int, default=None,help="Time series embedding dimension (GAT layer)")

    # GRU 参数
    parser.add_argument("--gru_n_layers", type=int, default=1,help="Number of GRU layers")
    parser.add_argument("--gru_hid_dim", type=int, default=150,help="Hidden dimension of GRU")

    # 预测头参数
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # 重构头参数
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # 激活函数参数
    parser.add_argument("--alpha", type=float, default=0.2,help="Slope parameter for LeakyReLU used in GAT attention")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1,help="Validation set ratio")
    parser.add_argument("--bs", type=int, default=256,help="Batch size")
    parser.add_argument("--window_stride", type=int, default=None, help="Sliding window sampling stride; auto-set per dataset if left empty")
    parser.add_argument("--init_lr", type=float, default=1e-3,help="Initial learning rate")
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True,help="Whether to shuffle the dataset")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--require_cuda", type=str2bool, default=False, help="Fail instead of silently falling back to CPU when CUDA is unavailable")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--persistent_workers", type=str2bool, default=True, help="Whether to keep DataLoader workers persistent")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")
    parser.add_argument("--predict_batch_size", type=int, default=128, help="Batch size for scheduled prediction; CH-BATTERY light prediction uses this value")
    parser.add_argument("--predict_num_workers", type=int, default=2, help="Number of workers for scheduled prediction; CH-BATTERY light prediction uses this value")
    parser.add_argument("--predict_pin_memory", type=str2bool, default=True, help="Whether to enable pin_memory for scheduled prediction")
    parser.add_argument("--print_every", type=int, default=1,help="Print training info every N epochs")
    parser.add_argument("--log_tensorboard", type=str2bool, default=True,help="Whether to log training to TensorBoard")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Validation epochs without improvement before stopping; 0 disables")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4, help="Minimum validation-loss improvement for early stopping")

    # 异常分数参数
    parser.add_argument("--scale_scores", type=str2bool, default=False, help="Whether to normalize anomaly scores")
    parser.add_argument("--use_mov_av", type=str2bool, default=False,help="Whether to use moving average for anomaly scores")
    parser.add_argument("--gamma", type=float, default=1,help="Weight factor in anomaly score formula, balancing prediction and reconstruction errors")
    parser.add_argument(
        "--score_fusion_mode",
        type=str,
        default="fixed",
        choices=["fixed", "quality_aware"],
        help="Score fusion mode: fixed=standard pred+gamma*recon, quality_aware=quality-aware fusion",
    )
    parser.add_argument("--use_event_consistency", type=str2bool, default=False, help="Whether to enable dual-threshold event-level anomaly detection with persistence constraints")
    parser.add_argument("--event_low_ratio", type=float, default=0.5, help="Interpolation ratio between training score median and high threshold; smaller values are stricter")
    parser.add_argument("--event_min_length", type=int, default=3, help="Minimum persistence length to retain an anomaly event")
    parser.add_argument("--level", type=float, default=None,help="Initial threshold for POT method")
    parser.add_argument("--q", type=float, default=None,help="Risk parameter for POT method, acceptable false alarm probability")
    parser.add_argument("--dynamic_pot", type=str2bool, default=False,help="Whether to use dynamic threshold")

    # 其他参数
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for experiment reproducibility")
    parser.add_argument("--run_id", type=str, default="", help="Experiment output directory ID; leave empty for auto timestamp")
    parser.add_argument("--resume", type=str2bool, default=False, help="Whether to resume training from last_checkpoint.pt in output directory")

    return parser
