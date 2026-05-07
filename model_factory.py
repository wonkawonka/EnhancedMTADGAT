from mtad_gat import Enhanced_MTADGAT


def normalize_model_name(model_name):
    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")


def get_available_model_names():
    return ["mtad_gat", "mtad_gat_c3", "mtad_gat_c4"]


def build_model(args, n_features, window_size, out_dim, target_dims=None):
    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))

    if model_name in {"mtad_gat", "mtad_gat_c3", "mtad_gat_c4"}:
        use_regime_condition = bool(getattr(args, "use_regime_condition", False))
        if model_name in {"mtad_gat_c3", "mtad_gat_c4"}:
            use_regime_condition = True

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
            use_regime_condition=use_regime_condition,
            regime_emb_dim=getattr(args, "regime_emb_dim", 32),
            regime_condition_mode=getattr(args, "regime_condition_mode", "fusion"),
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
