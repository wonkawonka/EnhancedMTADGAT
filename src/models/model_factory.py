"""根据参数构建模型，并配置特定数据集的物理状态设置。"""


import sys


from src.data.utils import BMS_FEATURE_NAMES, get_score_dims

from src.models.mtad_gat import Enhanced_MTADGAT


def normalize_model_name(model_name):

    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")


def get_available_model_names():

    return ["mtad_gat", "mtad_gat_c3", "mtad_gat_c3_regime", "mtad_gat_c4_physics"]


def get_model_family_defaults(model_name):

    """
    单一配置来源：每个 model_name 对应的默认功能开关。
    Plan files or CLI arguments can override these defaults.
    """

    model_name = normalize_model_name(model_name)

    defaults = {

        "use_transformer": False,

        "use_regime_condition": False,

        "use_revin": False,

        "score_fusion_mode": "fixed",

        "use_event_consistency": False,

    }

    if model_name in {"mtad_gat_c3", "mtad_gat_c3_regime", "mtad_gat_c4_physics"}:

        defaults.update({

            "use_transformer": True,

            "use_regime_condition": True,

            # RevIN remains available for compatibility, but it is not part of
            # the formal C3/C4 model families.  The thesis focuses on explicit
            # continuous-state conditioning instead of a separate
            # normalization-based distribution-shift branch.
            "use_revin": False,

            "score_fusion_mode": "quality_aware",

            "use_event_consistency": False,

            "regime_encoder_type": "temporal",

            # C3 conditions the fused relation representation before the shared
            # GRU/Transformer sequence encoders (scheme 3 / FiLM conditioning).
            "regime_condition_mode": "fusion",

        })

    if model_name == "mtad_gat_c4_physics":

        defaults.update({

            "use_physical_state_encoding": True,

            "use_physical_regularization": True,

            "use_physical_response_score": True,

        })

    return defaults


def resolve_model_args(args):

    """
    以 get_model_family_defaults 作为基础配置层，CLI/计划参数作为外层覆盖。
    Only toggles not explicitly present in the command line will be filled with model family defaults.
    """

    model_name = getattr(args, "model_name", "mtad_gat")

    defaults = get_model_family_defaults(model_name)

    for key, default_val in defaults.items():

        flag_a = f"--{key}"

        flag_b = f"--{key.replace('_', '-')}"

        if flag_a not in sys.argv and flag_b not in sys.argv:

            setattr(args, key, default_val)


def resolve_physical_state_config(args):

    dataset = str(getattr(args, "dataset", "")).upper()

    if dataset not in {"BMS", "NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE", "TSINGHUA_EV"}:

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

            # C4 treats the pack-level imposed current and the cluster SOC as
            # controls.  Cluster current/voltage/temperature remain responses:
            # otherwise a cluster-local current excursion could be absorbed as
            # an operating condition and never contribute to the consistency
            # residual.
            "consistency_current_index": BMS_FEATURE_NAMES.index("SYS_I"),
            "consistency_soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "consistency_response_dims": get_score_dims("BMS"),

        })

        return config

    if dataset == "TSINGHUA_EV":

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

        })

        return config


    config.update({

        "voltage_index": 0,

        "current_index": 1,

        "temperature_index": 2,

        "smooth_voltage": True,
        # NASA Random model inputs are [voltage, current, temperature].  The
        # imposed current is the sole control; voltage and temperature are the
        # physical responses scored by C4.
        "consistency_current_index": 1,
        "consistency_soc_index": None,
        "consistency_response_dims": [0, 2],

    })

    return config


def resolve_regime_config(args, n_features):

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

        return {"control_indices": [1, 2], "current_index": 1, "soc_index": 2}

    if dataset in {"NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}:

        # Raw step_type_code is removed by the loader.  Model-facing order is
        # voltage, current, temperature, and only current describes the condition.
        return {"control_indices": [1], "current_index": 1, "soc_index": None}

    if dataset == "MSL" and n_features > 1:

        # Channel 0 is the forecast/detection target in these benchmarks.  The
        # remaining anonymous channels provide latent context, not battery state.
        return {"control_indices": list(range(1, n_features)), "current_index": None, "soc_index": None}

    return {"control_indices": list(range(n_features)), "current_index": None, "soc_index": None}


def build_model(args, n_features, window_size, out_dim, target_dims=None):

    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))

    physical_state_config = resolve_physical_state_config(args)

    regime_config = resolve_regime_config(args, n_features)


    if model_name in {"mtad_gat", "mtad_gat_c3", "mtad_gat_c3_regime", "mtad_gat_c4_physics"}:

        return Enhanced_MTADGAT(

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

            use_transformer=args.use_transformer,

            trans_enc_layers=getattr(args, "trans_enc_layers", 2),

            transformer_ff_mult=getattr(args, "transformer_ff_mult", 2.0),

            transformer_norm_first=getattr(args, "transformer_norm_first", True),

            attention_top_k=args.attention_top_k,

            attention_sparse=getattr(args, "attention_sparse", False),

            feature_att_trans=getattr(args, "feature_att_trans", False),

            multi_scale_mode=getattr(args, "multi_scale_mode", "none"),

            multi_scale_dilations=[

                int(d) for d in getattr(args, "multi_scale_dilations", "1,2,4").split(",")

            ],

            use_revin=getattr(args, "use_revin", False),

            revin_affine=getattr(args, "revin_affine", True),

            target_dims=target_dims,

            use_regime_condition=getattr(args, "use_regime_condition", False),

            regime_emb_dim=getattr(args, "regime_emb_dim", 32),

            regime_condition_mode=getattr(args, "regime_condition_mode", "fusion"),

            regime_stat_features=[

                stat.strip()

                for stat in getattr(args, "regime_stat_features", "mean,std,last,delta").split(",")

                if stat.strip()

            ],

            regime_encoder_type=getattr(args, "regime_encoder_type", "temporal"),

            regime_control_indices=regime_config["control_indices"],

            regime_current_index=regime_config["current_index"],

            regime_soc_index=regime_config["soc_index"],

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

            use_control_response_decoder=getattr(args, "use_control_response_decoder", False),

            control_response_hidden_dim=getattr(args, "control_response_hidden_dim", 32),

            control_response_aux_weight=getattr(args, "control_response_aux_weight", 0.0),

            use_physical_consistency_head=getattr(args, "use_physical_consistency_head", False),

            physical_consistency_hidden_dim=getattr(args, "physical_consistency_hidden_dim", 64),

            physical_consistency_latent_dim=getattr(args, "physical_consistency_latent_dim", 16),

            physical_consistency_aux_weight=getattr(args, "physical_consistency_aux_weight", 0.0),

            physical_consistency_kl_weight=getattr(args, "physical_consistency_kl_weight", 0.0001),

            use_control_conditioned_graph=getattr(args, "use_control_conditioned_graph", False),

            condition_graph_emb_dim=getattr(args, "condition_graph_emb_dim", 32),

            condition_graph_experts=getattr(args, "condition_graph_experts", 3),

            condition_graph_control_indices=regime_config["control_indices"],

            use_variational_reconstruction=getattr(args, "use_variational_reconstruction", False),

            variational_reconstruction_latent_dim=getattr(args, "variational_reconstruction_latent_dim", 32),

            variational_reconstruction_kl_weight=getattr(args, "variational_reconstruction_kl_weight", 0.0001),

            physical_state_config=physical_state_config,

        )


    supported = ", ".join(get_available_model_names())

    raise ValueError(

        f"Unsupported model_name='{model_name}'. "

        f"Currently available: {supported}. "

        f"Add new baselines in model_factory.py before using them in train/predict."

    )
