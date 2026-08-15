"""根据参数构建模型，并配置特定数据集的物理状态设置。"""

import sys

from src.data.utils import BMS_FEATURE_NAMES, get_score_dims

from src.models.mtad_gat import Enhanced_MTADGAT

def normalize_model_name(model_name):
    """将 CLI、JSON 中可能含连字符/大小写差异的名称归一为内部模型名。"""
    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")

def get_available_model_names():
    """返回可由本工厂实例化的模型族；新增模型时必须同时更新这里与 build_model。"""
    return ["mtad_gat", "mtad_gat_c3", "mtad_gat_c3_regime", "mtad_gat_c4_physics"]

def get_model_family_defaults(model_name):
    """
    单一配置来源：每个 ``model_name`` 对应的默认功能开关。
    配置优先级是：模型族默认值 < JSON 计划/CLI 显式参数。这样 ``mtad_gat``
    是干净基线；``mtad_gat_c3`` 当前是尚未冻结的中性占位；
    ``mtad_gat_c3_regime`` 只保留用于复现已否定的 FiLM 工况注入实验；
    ``mtad_gat_c4_physics`` 在原骨干旁路开启独立物理一致性头。C3 与 C4
    是并列研究线，C4 不继承 C3 工况模块。
    """
    model_name = normalize_model_name(model_name)
    defaults = {
        "use_transformer": False,
        "use_regime_condition": False,
        "use_revin": False,
        "score_fusion_mode": "fixed",
        "use_event_consistency": False,
        "regime_aux_lambda": 0.0,
        "use_physical_state_encoding": False,
        "use_physical_regularization": False,
        "use_physical_response_score": False,
        "use_physical_response_features": False,
        "use_control_response_decoder": False,
        "use_physical_consistency_head": False,
        "physical_consistency_aux_weight": 0.0,
        "use_condition_residual_calibration": False,
        "regime_group_dro_lambda": 0.0,
        "use_c3_joint_relation": False,
        "use_learnable_sparse_graph": False,
        "sparse_graph_lambda": 0.0,
    }
    if model_name == "mtad_gat_c3":
        defaults.update({
            # 独立C3尚未通过跨折验证；保持中性默认值。所有历史工况候选
            # 只通过实验计划显式开启，避免模型名暗中启用失败结构。
            "use_transformer": False,
            "use_regime_condition": False,
            "regime_aux_lambda": 0.0,
            "score_fusion_mode": "fixed",
            "battery_response_only_training": False,
            "use_control_response_decoder": False,
            "control_response_aux_weight": 0.0,
            "use_condition_residual_calibration": False,
            "use_condition_slow_state": False,
            "regime_group_dro_lambda": 0.0,
        })
    if model_name == "mtad_gat_c3_regime":
        defaults.update({
            "use_transformer": True,
            "use_regime_condition": True,
            # 保留 RevIN 开关只是为了兼容旧实验，它不属于论文正式的 C3/C4 模型族。
            # 论文主线采用显式连续工况条件化，不设置独立的分布偏移归一化分支。
            "use_revin": False,
            "score_fusion_mode": "quality_aware",
            "use_event_consistency": False,
            "regime_encoder_type": "temporal",
            # 旧四模块只用于历史消融复现，不再作为正式 C3。
            "regime_condition_mode": "feature_gat",
            "regime_aux_lambda": 0.05,
        })
    if model_name == "mtad_gat_c4_physics":
        defaults.update({
            "use_transformer": False,
            "use_regime_condition": False,
            "regime_aux_lambda": 0.0,
            "score_fusion_mode": "quality_aware",
            "use_physical_consistency_head": True,
            "physical_consistency_aux_weight": 1.0,
            "physical_consistency_kl_weight": 0.0001,
            "physical_consistency_score_max_weight": 0.35,
        })
    return defaults

def resolve_model_args(args):
    """
    将模型族默认值填入 argparse 命名空间，但不覆盖命令行已经显式给出的开关。
    ``compare_experiments`` 会把 JSON 参数转成 CLI；因此检查 ``sys.argv`` 能区分
    “用户特意关闭该功能”与“没有写该参数、应采用模型族默认值”两种情况。
    """
    model_name = getattr(args, "model_name", "mtad_gat")
    defaults = get_model_family_defaults(model_name)
    for key, default_val in defaults.items():
        flag_a = f"--{key}"
        flag_b = f"--{key.replace('_', '-')}"
        if flag_a not in sys.argv and flag_b not in sys.argv:
            setattr(args, key, default_val)

def resolve_physical_state_config(args):
    """把不同数据集的通道语义映射为 C4 所需的物理状态配置。
    模型只看到通道序号；本函数在这里统一声明哪一列是电流、SOC、电压、温度，
    以及哪些列是可被评分的响应。非电池数据返回 ``None``，使 C4 物理分支失效。
    """
    dataset = str(getattr(args, "dataset", "")).upper()
    if dataset not in {
        "BMS", "NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE", "TSINGHUA_EV",
        "CH_BATTERY_LFP_DISCHARGE",
    }:
        return None
    config = {
        "dataset_name": dataset,
        "current_index": None,
        "voltage_index": None,
        "temperature_index": None,
        "step_type_index": None,
        "soc_index": None,
        "soh_index": None,
        "smooth_voltage": False,
        "smooth_temperature": True,
        "voltage_max_index": None,
        "voltage_min_index": None,
        "temperature_max_index": None,
        "temperature_min_index": None,
        "response_terms": getattr(args, "physical_response_terms", ""),
        "data_min": getattr(args, "physical_data_min", None),
        "data_scale": getattr(args, "physical_data_scale", None),
    }
    if not config["response_terms"]:
        config.pop("response_terms")
    if dataset == "BMS":
        config.update({
            "current_index": BMS_FEATURE_NAMES.index("BMSnI"),
            "voltage_index": BMS_FEATURE_NAMES.index("BMSnVmean"),
            "temperature_index": BMS_FEATURE_NAMES.index("BMSnTmean"),
            "soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "voltage_max_index": BMS_FEATURE_NAMES.index("BMSnVmax"),
            "voltage_min_index": BMS_FEATURE_NAMES.index("BMSnVmin"),
            "temperature_max_index": BMS_FEATURE_NAMES.index("BMSnTmax"),
            "temperature_min_index": BMS_FEATURE_NAMES.index("BMSnTmin"),
            "smooth_voltage": False,
            # C4 把系统级外加电流和簇 SOC 作为控制量，把簇电流、电压和温度保留为
            # 响应量。若把簇局部电流也作为控制量，局部过流可能被吸收成工况变化，
            # 从而无法形成物理一致性残差。
            "consistency_current_index": BMS_FEATURE_NAMES.index("SYS_I"),
            "consistency_soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "consistency_response_dims": get_score_dims("BMS"),
        })
        return config
    if dataset == "TSINGHUA_EV":
        # 清华 EV 固定七通道顺序：包电压、电流、SOC、最大/最小单体电压、
        # 最大/最小温度。电流与 SOC 描述工况，其余通道用于响应一致性。
        config.update({
            "voltage_index": 0,
            "current_index": 1,
            "soc_index": 2,
            "temperature_index": 5,
            "voltage_max_index": 3,
            "voltage_min_index": 4,
            "temperature_max_index": 5,
            "temperature_min_index": 6,
            "smooth_voltage": False,
            "consistency_current_index": 1,
            "consistency_soc_index": 2,
            "consistency_response_dims": [0, 3, 4, 5, 6],
        })
        return config
    if dataset == "CH_BATTERY_LFP_DISCHARGE":
        # CH-BatteryGen LFP 放电缓存的固定七通道顺序：总压、电流、SOC、
        # 最大/最小单体电压、最大/最小温度。C4 只以电流和 SOC 生成控制序列，
        # 并对其余五个可观测响应建立独立的一致性残差。
        config.update({
            "voltage_index": 0,
            "current_index": 1,
            "soc_index": 2,
            "temperature_index": 5,
            "voltage_max_index": 3,
            "voltage_min_index": 4,
            "temperature_max_index": 5,
            "temperature_min_index": 6,
            "smooth_voltage": False,
            "consistency_current_index": 1,
            "consistency_soc_index": 2,
            "consistency_response_dims": [0, 3, 4, 5, 6],
        })
        return config
    config.update({
        "voltage_index": 0,
        "current_index": 1,
        "temperature_index": 2,
        "smooth_voltage": True,
        # NASA Random 的模型输入顺序为“电压、电流、温度”。外加电流是唯一控制量，
        # 电压和温度是 C4 进行一致性评分的物理响应量。
        "consistency_current_index": 1,
        "consistency_soc_index": None,
        "consistency_response_dims": [0, 2],
    })
    return config

def resolve_regime_config(args, n_features):
    """声明 C3 工况编码器应从哪些通道提取连续运行状态。
    该配置与 C4 的物理通道配置相互独立：C3 关注“当前处于什么工况”，
    C4 关注“在该工况下响应是否自洽”。
    """
    dataset = str(getattr(args, "dataset", "")).upper()
    if dataset == "BMS":
        return {
            "control_indices": [
                BMS_FEATURE_NAMES.index("SYS_I"),
                BMS_FEATURE_NAMES.index("BMSnRSOC"),
            ],
            "current_index": BMS_FEATURE_NAMES.index("SYS_I"),
            "soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
        }
    if dataset == "TSINGHUA_EV":
        return {
            "control_indices": [1, 2], "current_index": 1, "soc_index": 2,
            "pooled_channels": False,
        }
    if dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}:
        # 数据加载器已移除原始 step_type_code；模型看到的顺序是“电压、电流、温度”，
        # 其中只有电流用于描述工况。
        return {"control_indices": [1], "current_index": 1, "soc_index": None}
    if dataset in {"MSL", "SMAP"} and n_features > 1:
        # NASA遥测的通道 0 是预测和检测目标，其余匿名通道只提供潜在上下文，
        # 不能解释成电池物理状态。
        return {
            "control_indices": list(range(1, n_features)),
            "current_index": None,
            "soc_index": None,
            "pooled_channels": True,
        }
    return {
        "control_indices": list(range(n_features)),
        "current_index": None,
        "soc_index": None,
        "pooled_channels": False,
    }

def build_model(args, n_features, window_size, out_dim, target_dims=None):
    """将已解析的实验参数实例化为实际 PyTorch 模型。
    ``target_dims`` 为 ``None`` 时预测全部通道；否则只预测响应通道。这里不训练，
    只负责把数据集通道语义、C3/C4 开关和基础 MTAD-GAT 超参数交给模型构造器。
    """
    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))
    physical_state_config = resolve_physical_state_config(args)
    regime_config = resolve_regime_config(args, n_features)
    raw_condition_indices = str(getattr(args, "regime_condition_indices", "")).strip()
    if raw_condition_indices:
        condition_indices = [
            int(index.strip()) for index in raw_condition_indices.split(",") if index.strip()
        ]
        if not condition_indices or any(index < 0 or index >= n_features for index in condition_indices):
            raise ValueError("regime_condition_indices contains an out-of-range raw input index")
        if len(set(condition_indices)) != len(condition_indices):
            raise ValueError("regime_condition_indices must not contain duplicate indices")
        regime_config["control_indices"] = condition_indices
    raw_backbone_indices = str(getattr(args, "backbone_feature_indices", "")).strip()
    backbone_feature_indices = [
        int(index.strip()) for index in raw_backbone_indices.split(",") if index.strip()
    ]
    if backbone_feature_indices:
        if any(index < 0 or index >= n_features for index in backbone_feature_indices):
            raise ValueError("backbone_feature_indices contains an out-of-range raw input index")
        if len(set(backbone_feature_indices)) != len(backbone_feature_indices):
            raise ValueError("backbone_feature_indices must not contain duplicate indices")
        backbone_n_features = len(backbone_feature_indices)
        backbone_control_indices = [
            position for position, raw_index in enumerate(backbone_feature_indices)
            if raw_index in regime_config["control_indices"]
        ]
    else:
        backbone_n_features = n_features
        backbone_control_indices = list(regime_config["control_indices"])
    if model_name in {"mtad_gat", "mtad_gat_c3", "mtad_gat_c3_regime", "mtad_gat_c4_physics"}:
        # 四个名称共享同一骨干类。实验差异来自传入的可选模块开关，而不是复制四份模型。
        return Enhanced_MTADGAT(
            # 输入、窗口和输出维度由数据加载流程决定。
            backbone_n_features,
            window_size,
            out_dim,
            # 基线 MTAD-GAT 的卷积、图注意力、GRU、预测头和重构头参数。
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
            # C3 使用的长程 Transformer 和注意力结构参数。
            use_transformer=args.use_transformer,
            trans_enc_layers=getattr(args, "trans_enc_layers", 2),
            transformer_ff_mult=getattr(args, "transformer_ff_mult", 2.0),
            transformer_norm_first=getattr(args, "transformer_norm_first", True),
            attention_top_k=args.attention_top_k,
            attention_sparse=getattr(args, "attention_sparse", False),
            gat_output_activation=getattr(args, "gat_output_activation", "sigmoid"),
            fusion_projection_dim=getattr(args, "fusion_projection_dim", 0),
            feature_att_trans=getattr(args, "feature_att_trans", False),
            multi_scale_mode=getattr(args, "multi_scale_mode", "none"),
            multi_scale_dilations=[
                int(d) for d in getattr(args, "multi_scale_dilations", "1,2,4").split(",")
            ],
            use_learnable_sparse_graph=getattr(args, "use_learnable_sparse_graph", False),
            # 归一化和实际需要预测的通道。
            use_revin=getattr(args, "use_revin", False),
            revin_affine=getattr(args, "revin_affine", True),
            target_dims=target_dims,
            # C3 连续工况编码、注入位置和控制通道语义。
            use_regime_condition=getattr(args, "use_regime_condition", False),
            regime_emb_dim=getattr(args, "regime_emb_dim", 32),
            regime_condition_mode=getattr(args, "regime_condition_mode", "feature_gat"),
            regime_stat_features=[
                stat.strip()
                for stat in getattr(args, "regime_stat_features", "mean,std,last,delta").split(",")
                if stat.strip()
            ],
            regime_encoder_type=getattr(args, "regime_encoder_type", "temporal"),
            regime_channel_pooling=regime_config.get("pooled_channels", False),
            regime_control_indices=regime_config["control_indices"],
            regime_current_index=regime_config["current_index"],
            regime_soc_index=regime_config["soc_index"],
            regime_film_scale=getattr(args, "regime_film_scale", 0.1),
            regime_condition_shuffle=getattr(args, "regime_condition_shuffle", False),
            # C4 物理状态编码和物理响应特征分支。
            use_physical_state_encoding=(
                getattr(args, "use_physical_state_encoding", False)
                and physical_state_config is not None
            ),
            physical_state_hidden_dim=getattr(args, "physical_state_hidden_dim", 32),
            physical_state_injection_mode=getattr(args, "physical_state_injection_mode", "direct"),
            physical_state_feature_mode=getattr(args, "physical_state_feature_mode", "full"),
            use_physical_response_features=getattr(args, "use_physical_response_features", False),
            physical_feature_fusion_mode=getattr(args, "physical_feature_fusion_mode", "shared_residual"),
            physical_feature_hidden_dim=getattr(args, "physical_feature_hidden_dim", 32),
            # 可选的控制量到响应量直接修正分支。
            use_control_response_decoder=getattr(args, "use_control_response_decoder", False),
            control_response_hidden_dim=getattr(args, "control_response_hidden_dim", 32),
            control_response_aux_weight=getattr(args, "control_response_aux_weight", 0.0),
            # 正式 C4 使用的独立控制—响应一致性头及其信息瓶颈参数。
            use_physical_consistency_head=getattr(args, "use_physical_consistency_head", False),
            physical_consistency_hidden_dim=getattr(args, "physical_consistency_hidden_dim", 64),
            physical_consistency_latent_dim=getattr(args, "physical_consistency_latent_dim", 16),
            physical_consistency_aux_weight=getattr(args, "physical_consistency_aux_weight", 0.0),
            physical_consistency_kl_weight=getattr(args, "physical_consistency_kl_weight", 0.0001),
            # 以下是用于消融研究的条件图和变分重构分支。
            use_control_conditioned_graph=getattr(args, "use_control_conditioned_graph", False),
            condition_graph_emb_dim=getattr(args, "condition_graph_emb_dim", 32),
            condition_graph_experts=getattr(args, "condition_graph_experts", 3),
            condition_graph_control_indices=regime_config["control_indices"],
            condition_graph_router_mode=getattr(args, "condition_graph_router_mode", "learned"),
            condition_graph_router_temperature=getattr(args, "condition_graph_router_temperature", 1.0),
            use_condition_routed_adapter=getattr(args, "use_condition_routed_adapter", False),
            condition_adapter_rank=getattr(args, "condition_adapter_rank", 16),
            condition_adapter_experts=getattr(args, "condition_adapter_experts", 4),
            condition_adapter_temperature=getattr(args, "condition_adapter_temperature", 1.0),
            use_variational_reconstruction=getattr(args, "use_variational_reconstruction", False),
            variational_reconstruction_latent_dim=getattr(args, "variational_reconstruction_latent_dim", 32),
            variational_reconstruction_kl_weight=getattr(args, "variational_reconstruction_kl_weight", 0.0001),
            physical_state_config=physical_state_config,
            # 仅在条件化实验中将控制量从 MTAD-GAT 主干拿掉；条件编码器仍读取完整原始窗口。
            backbone_feature_indices=backbone_feature_indices,
            condition_source_n_features=n_features,
            backbone_control_indices=backbone_control_indices,
            use_c3_joint_relation=getattr(args, "use_c3_joint_relation", False),
            c3_relation_rank=getattr(args, "c3_relation_rank", 4),
            c3_joint_hidden_dim=getattr(args, "c3_joint_hidden_dim", 32),
            c3_relation_loss_weight=getattr(args, "c3_relation_loss_weight", 0.1),
            c3_joint_nll_weight=getattr(args, "c3_joint_nll_weight", 0.01),
            c3_value_gamma=getattr(args, "gamma", 1.0),
        )
    supported = ", ".join(get_available_model_names())
    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        f"Currently available: {supported}. "
        f"Add new baselines in model_factory.py before using them in train/predict."
    )
