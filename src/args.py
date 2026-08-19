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
    parser.add_argument(
        "--gat_output_activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "elu", "tanh", "identity"],
        help="图注意力聚合后的输出激活；sigmoid 保持原始实现，其余选项仅用于结构诊断",
    )
    parser.add_argument(
        "--fusion_projection_dim",
        type=int,
        default=0,
        help="将卷积、特征GAT和时间GAT的3K维拼接线性压缩到指定维度；0表示不压缩",
    )
    # 简化模型选项：仅使用特征注意力和 Transformer
    parser.add_argument("--feature_att_trans", type=str2bool, default=False,
                        help="仅使用特征注意力和Transformer，跳过时间注意力和GRU")
    # 多尺度卷积选项
    parser.add_argument("--multi_scale_mode", type=str, default="none",
                        choices=["none", "basic", "progressive", "residual"],
                        help="Multi-scale convolution mode: residual keeps the original Conv7 path and adds a zero-gated causal branch")
    parser.add_argument("--multi_scale_dilations", type=str, default="1,2,4",
                        help="List of dilation rates for causal dilated convolution, comma separated, e.g. '1,2,4' (kernel_size fixed to 3)")
    parser.add_argument(
        "--use_learnable_sparse_graph",
        type=str2bool,
        default=False,
        help="Regularize the global Feature-GAT edge logits into a learned soft sparse structure",
    )
    parser.add_argument(
        "--sparse_graph_lambda",
        type=float,
        default=0.0,
        help="Weight of the off-diagonal learned-edge density penalty; 0 disables sparse graph learning",
    )
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
        choices=["restricted", "prototype_query", "temporal", "statistics"],
        help="restricted is the frozen marginal encoder; prototype_query is the proposed C3 upgrade",
    )
    parser.add_argument("--regime_aux_lambda", type=float, default=0.05, help="Self-supervised dynamic descriptor loss weight")
    parser.add_argument("--regime_prototype_lambda", type=float, default=0.0, help="Weight of the prototype-query routing anti-collapse loss")
    parser.add_argument("--regime_query_dim", type=int, default=32, help="Internal token/query width of prototype-query C3")
    parser.add_argument("--regime_num_prototypes", type=int, default=6, help="Number of learnable operating-regime queries")
    parser.add_argument("--regime_query_heads", type=int, default=4, help="Cross-attention head count of prototype-query C3")
    parser.add_argument("--regime_temperature", type=float, default=0.5, help="Soft routing temperature of prototype-query C3")
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
        "--physical_state_injection_mode",
        type=str,
        default="direct",
        choices=["direct", "direct_preserve_rng", "gated_residual"],
        help="Legacy ablation only: how physical state enters the shared backbone; not part of formal C4",
    )
    parser.add_argument(
        "--physical_state_feature_mode",
        type=str,
        default="full",
        choices=["full", "controls_only"],
        help="Physical-state inputs: full includes response channels; controls_only uses phase/current/charge-flow/SOC only",
    )
    parser.add_argument(
        "--use_physical_response_features",
        type=str2bool,
        default=False,
        help="Legacy ablation only: fuse derived response features into the shared backbone",
    )
    parser.add_argument(
        "--physical_feature_fusion_mode",
        type=str,
        default="shared_residual",
        choices=["shared_residual", "shared_film", "feature_gat_residual", "feature_gat_attention_bias"],
        help="Insertion point/form for response-physics features inside the C3 backbone",
    )
    parser.add_argument(
        "--physical_feature_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of the response-physics feature encoder",
    )
    parser.add_argument(
        "--use_control_response_decoder",
        type=str2bool,
        default=False,
        help="Legacy ablation only: condition shared heads on current/SOC histories",
    )
    parser.add_argument(
        "--control_response_hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of the causal control-response decoder conditioner",
    )
    parser.add_argument(
        "--control_response_aux_weight",
        type=float,
        default=0.0,
        help="Weight of the control-only response reconstruction auxiliary objective",
    )
    parser.add_argument(
        "--use_condition_residual_calibration",
        type=str2bool,
        default=False,
        help="C3: calibrate frozen-backbone residuals by normal current/SOC conditions",
    )
    parser.add_argument(
        "--condition_calibration_clusters",
        type=int,
        default=12,
        help="Number of normal operating-regime prototypes used by C3 residual calibration",
    )
    parser.add_argument(
        "--condition_calibration_method",
        type=str,
        default="neural_heteroscedastic",
        choices=("hard_kmeans", "soft_expert", "neural_heteroscedastic"),
        help="C3 residual model: continuous heteroscedastic calibration (formal), or legacy prototype ablations",
    )
    parser.add_argument(
        "--condition_calibration_temperature",
        type=float,
        default=1.0,
        help="Relative bandwidth of C3 soft condition-expert assignments",
    )
    parser.add_argument(
        "--use_condition_slow_state",
        type=str2bool,
        default=False,
        help="Include label-free mileage as a C3 long-horizon operating-state proxy",
    )
    parser.add_argument(
        "--use_control_conditioned_graph",
        type=str2bool,
        default=False,
        help="C3: route feature-graph edge biases from current/SOC trajectories",
    )
    parser.add_argument("--condition_graph_emb_dim", type=int, default=32)
    parser.add_argument("--condition_graph_experts", type=int, default=3)
    parser.add_argument(
        "--condition_graph_router_mode",
        type=str,
        default="learned",
        choices=["learned", "control_quadrant"],
        help="Learn a free condition router or use a fixed soft current/SOC quadrant router",
    )
    parser.add_argument(
        "--condition_graph_router_temperature",
        type=float,
        default=1.0,
        help="Soft-quadrant temperature for the fixed current/SOC condition router",
    )
    parser.add_argument(
        "--use_condition_routed_adapter",
        type=str2bool,
        default=False,
        help="C3: route low-rank hidden-state adapters using current/SOC conditions",
    )
    parser.add_argument("--condition_adapter_rank", type=int, default=16)
    parser.add_argument("--condition_adapter_experts", type=int, default=4)
    parser.add_argument("--condition_adapter_temperature", type=float, default=1.0)
    parser.add_argument(
        "--use_variational_reconstruction",
        type=str2bool,
        default=False,
        help="Replace MTAD-GAT's deterministic reconstruction head with a VAE head",
    )
    parser.add_argument("--variational_reconstruction_latent_dim", type=int, default=32)
    parser.add_argument("--variational_reconstruction_kl_weight", type=float, default=0.0001)
    parser.add_argument(
        "--use_physical_consistency_head",
        type=str2bool,
        default=False,
        help="C4: train an independent bottlenecked control-to-response consistency head",
    )
    parser.add_argument("--physical_consistency_hidden_dim", type=int, default=64)
    parser.add_argument("--physical_consistency_latent_dim", type=int, default=16)
    parser.add_argument(
        "--physical_consistency_encoder_input",
        type=str,
        default="full_window",
        choices=["full_window", "control_only"],
        help=(
            "C4 state input: restored response-aware full window or strict current/SOC-only "
            "ablation"
        ),
    )
    parser.add_argument(
        "--physical_consistency_encoder_bidirectional",
        type=str2bool,
        default=False,
        help="Use a one-way C4 state GRU by default; true enables the historical bidirectional ablation",
    )
    parser.add_argument("--physical_consistency_aux_weight", type=float, default=0.0)
    parser.add_argument("--physical_consistency_kl_weight", type=float, default=0.0001)
    parser.add_argument(
        "--physical_consistency_score_max_weight",
        type=float,
        default=0.35,
        help="Maximum normal-calibrated weight of the independent C4 score",
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
        "--use_relation_change_score",
        type=str2bool,
        default=False,
        help="Fuse label-free Feature-GAT relation-change evidence into the anomaly score",
    )
    parser.add_argument(
        "--relation_change_weight",
        type=float,
        default=0.2,
        help="Fixed bounded weight for the normal-calibrated relation-change score",
    )
    parser.add_argument(
        "--relation_change_fusion_mode",
        type=str,
        default="linear_legacy",
        choices=["linear_legacy", "residual_gated"],
        help="Fusion rule; residual_gated requires both elevated value residual and bounded relation change",
    )
    parser.add_argument(
        "--relation_change_mode",
        type=str,
        default="consecutive_js",
        choices=["consecutive_js", "normal_transition_residual"],
        help="Relation evidence: consecutive JS change, or a normal-fitted first-order attention transition residual",
    )
    parser.add_argument(
        "--use_relation_prototype_suppression",
        type=str2bool,
        default=False,
        help="Suppress high value residuals only when the Feature-GAT relation transition matches a normal-validation prototype",
    )
    parser.add_argument(
        "--relation_prototype_clusters",
        type=int,
        default=4,
        help="Number of normal relation-transition prototypes fitted without fault labels",
    )
    parser.add_argument(
        "--relation_prototype_max_suppression",
        type=float,
        default=0.15,
        help="Maximum multiplicative score suppression for a close normal relation prototype",
    )
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
        choices=[
            "none", "transformer_residual", "feature_gat", "feature_gat_response",
            "temporal_gat", "fusion", "head",
        ],
        help="动态状态条件化位置；feature_gat_response仅调制响应节点，head在GRU隐状态调制",
    )
    parser.add_argument(
        "--regime_film_scale",
        type=float,
        default=0.1,
        help="FiLM最大相对调制幅度；正式候选使用不大于0.1的有界残差调制",
    )
    parser.add_argument(
        "--normal_tail_lambda",
        type=float,
        default=0.0,
        help="Weight of normal-window CVaR excess used to directly reduce the high residual tail",
    )
    parser.add_argument(
        "--normal_tail_fraction",
        type=float,
        default=0.1,
        help="Highest-loss fraction of each normal training batch used by the CVaR excess",
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
            'SMAP',
            'SWAT',
            'WADI',
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
    parser.add_argument("--battery_brand", type=int, default=3, choices=[1, 2, 3], help="Manufacturer package in the official Nature Communications battery dataset")
    parser.add_argument("--battery_fold", type=int, default=0, choices=range(5), help="Vehicle-level five-fold test split")
    parser.add_argument(
        "--battery_split_protocol",
        type=str,
        default="strict_normal_validation",
        choices=["strict_normal_validation", "paper_protocol"],
        help="Strict normal-only calibration or the labelled protocol in Zhang et al. Supplementary Note 2",
    )
    parser.add_argument("--battery_windows_per_snippet", type=int, default=1, help="Evenly spaced model windows sampled from each charging snippet")
    parser.add_argument(
        "--battery_normalization",
        type=str,
        default="minmax",
        choices=["minmax", "paper_channel"],
        help="Internal battery normalization: train-fold MinMax or Zhang et al. DyAD channel normalization",
    )
    parser.add_argument(
        "--battery_fold_seed",
        type=int,
        default=-1,
        help="Vehicle-fold seed; -1 reuses --seed. Zhang et al.'s released split notebook uses 0.",
    )
    parser.add_argument("--battery_vehicle_top_ratio", type=float, default=0.05, help="Top charging-snippet fraction averaged into each vehicle anomaly score")
    parser.add_argument(
        "--battery_evaluation_checkpoint",
        type=str,
        default="",
        help="Optional frozen model.pt used to evaluate a new scoring rule without retraining",
    )
    parser.add_argument(
        "--battery_vehicle_top_ratio_mode",
        type=str,
        default="fixed",
        choices=["fixed", "labelled_calibration"],
        help="Use the configured fixed vehicle Top-p ratio for formal results; labelled calibration is retained only for historical sensitivity analysis",
    )
    parser.add_argument("--battery_score_channels", type=str, default="response", choices=["all", "response"], help="Faithful MTAD-GAT all-channel score or battery response-only score")
    parser.add_argument(
        "--battery_response_only_training",
        type=str2bool,
        default=False,
        help="Use current/SOC as conditions and optimize only voltage/temperature response outputs",
    )
    parser.add_argument(
        "--backbone_feature_indices",
        type=str,
        default="",
        help="Optional comma-separated raw input indices retained by the MTAD-GAT backbone; excluded channels remain available only to condition branches.",
    )
    parser.add_argument(
        "--regime_condition_indices",
        type=str,
        default="",
        help="Optional comma-separated raw input indices read by the FiLM condition encoder; empty uses dataset-specific control channels.",
    )
    parser.add_argument(
        "--regime_condition_shuffle",
        type=str2bool,
        default=False,
        help="Development-only negative control: cyclically mismatch selected FiLM condition channels across samples in each batch while preserving their marginal distribution.",
    )
    parser.add_argument(
        "--regime_group_dro_lambda",
        type=float,
        default=0.0,
        help="Weight of worst-regime excess-risk training; 0 disables the C3 condition-robust objective",
    )
    parser.add_argument(
        "--regime_group_dro_temperature",
        type=float,
        default=0.05,
        help="Smooth-maximum temperature for C3 worst-regime risk",
    )
    parser.add_argument("--battery_max_index_snippets", type=int, default=0, help="Temporary in-memory index cap for smoke tests; 0 builds/uses the complete brand index")
    parser.add_argument("--battery_max_snippets_per_vehicle", type=int, default=0, help="Per-vehicle cap for smoke tests only; 0 uses every charging snippet")
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="Enable deterministic CUDA algorithms and seeded DataLoader generators for reproducible formal experiments",
    )
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
        help="Model name: mtad_gat baseline; mtad_gat_c3 = response-target decoupling plus control-response adapter; mtad_gat_c3_regime = legacy FiLM ablation; mtad_gat_c4_physics = independent physics branch.",
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
    parser.add_argument("--use_c3_joint_relation", type=str2bool, default=False, help="Enable model-internal Feature-GAT transition residual and joint residual score")
    parser.add_argument("--c3_relation_rank", type=int, default=4, help="Compatibility parameter for the Feature-GAT transition head")
    parser.add_argument("--c3_joint_hidden_dim", type=int, default=32, help="Compatibility parameter for the bivariate residual density head")
    parser.add_argument("--c3_relation_loss_weight", type=float, default=0.1, help="Training weight for Feature-GAT transition prediction")
    parser.add_argument("--c3_joint_nll_weight", type=float, default=0.01, help="Training weight for the internal joint residual likelihood")
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
