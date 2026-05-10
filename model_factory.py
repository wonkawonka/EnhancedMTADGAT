import sys

from mtad_gat import Enhanced_MTADGAT


def normalize_model_name(model_name):
    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")


def get_available_model_names():
    return ["mtad_gat", "mtad_gat_c3", "mtad_gat_c4"]


def get_model_family_defaults(model_name):
    """
    单一真相源：每个 model_name 对应的默认开关。
    plan 文件或 CLI 显式传参可以覆盖这些默认值。
    """
    model_name = normalize_model_name(model_name)
    defaults = {
        "use_transformer": False,
        "use_regime_condition": False,
        "use_revin": False,
        "score_fusion_mode": "fixed",
        "use_event_consistency": False,
        "use_hier_consistency": False,
    }
    if model_name == "mtad_gat_c3":
        defaults.update({
            "use_transformer": True,
            "use_regime_condition": True,
            "use_revin": True,
            "score_fusion_mode": "quality_aware",
            "use_event_consistency": True,
        })
    elif model_name == "mtad_gat_c4":
        defaults.update({
            "use_transformer": True,
            "use_regime_condition": True,
            "use_revin": True,
            "score_fusion_mode": "quality_aware",
            "use_event_consistency": True,
            "use_hier_consistency": True,
        })
    return defaults


def resolve_model_args(args):
    """
    以 get_model_family_defaults 为基层，CLI/计划显式传参为覆盖层。
    只有未显式出现在命令行中的开关才会被模型族默认值填充。
    """
    model_name = getattr(args, "model_name", "mtad_gat")
    defaults = get_model_family_defaults(model_name)
    for key, default_val in defaults.items():
        flag_a = f"--{key}"
        flag_b = f"--{key.replace('_', '-')}"
        if flag_a not in sys.argv and flag_b not in sys.argv:
            setattr(args, key, default_val)


def build_model(args, n_features, window_size, out_dim, target_dims=None):
    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))

    if model_name in {"mtad_gat", "mtad_gat_c3", "mtad_gat_c4"}:
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
        )

    supported = ", ".join(get_available_model_names())
    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        f"Currently available: {supported}. "
        f"Add new baselines in model_factory.py before using them in train/predict."
    )
