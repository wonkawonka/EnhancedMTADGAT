"""根据参数构建模型，并配置特定数据集的物理状态设置。"""


import sys


from src.data.utils import BMS_FEATURE_NAMES

from src.models.mtad_gat import Enhanced_MTADGAT


def normalize_model_name(model_name):

    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")


def get_available_model_names():

    return ["mtad_gat", "mtad_gat_c3"]


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

    if model_name == "mtad_gat_c3":

        defaults.update({

            "use_transformer": True,

            "use_regime_condition": True,

            "use_revin": True,

            "score_fusion_mode": "quality_aware",

            "use_event_consistency": True,

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

    if dataset not in {"BMS", "NASA_RANDOM_CHARGE", "NASA_RANDOM_DISCHARGE"}:

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

    }


    if dataset == "BMS":

        config.update({

            "current_index": BMS_FEATURE_NAMES.index("BMSnI"),

            "voltage_index": BMS_FEATURE_NAMES.index("BMSnVmean"),

            "temperature_index": BMS_FEATURE_NAMES.index("BMSnTmean"),

            "smooth_voltage": False,

        })

        return config


    config.update({

        "step_type_index": 0,

        "voltage_index": 1,

        "current_index": 2,

        "temperature_index": 3,

        "smooth_voltage": True,

    })

    return config


def build_model(args, n_features, window_size, out_dim, target_dims=None):

    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))

    physical_state_config = resolve_physical_state_config(args)


    if model_name in {"mtad_gat", "mtad_gat_c3"}:

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

            regime_condition_mode=getattr(args, "regime_condition_mode", "transformer_residual"),

            regime_stat_features=[

                stat.strip()

                for stat in getattr(args, "regime_stat_features", "mean,std,last,delta").split(",")

                if stat.strip()

            ],

            use_physical_state_encoding=(

                getattr(args, "use_physical_state_encoding", False)

                and physical_state_config is not None

            ),

            physical_state_hidden_dim=getattr(args, "physical_state_hidden_dim", 32),

            physical_state_config=physical_state_config,

        )


    supported = ", ".join(get_available_model_names())

    raise ValueError(

        f"Unsupported model_name='{model_name}'. "

        f"Currently available: {supported}. "

        f"Add new baselines in model_factory.py before using them in train/predict."

    )


