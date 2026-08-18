"""Build the frozen MTAD-GAT baseline, C3 and C4 architectures."""

from src.data.utils import get_score_dims
from src.models.mtad_gat import Enhanced_MTADGAT


def normalize_model_name(model_name):
    return str(model_name or "mtad_gat").strip().lower().replace("-", "_")


def get_available_model_names():
    return ["mtad_gat"]


def get_model_family_defaults(model_name):
    if normalize_model_name(model_name) != "mtad_gat":
        raise ValueError("Only the frozen 'mtad_gat' model family is available")
    return {
        "use_regime_condition": False,
        "regime_aux_lambda": 0.0,
        "use_physical_consistency_head": False,
        "physical_consistency_aux_weight": 0.0,
        "score_fusion_mode": "fixed",
    }


def resolve_model_args(args):
    """Apply the architecture lock shared by training and prediction."""
    model_name = normalize_model_name(getattr(args, "model_name", "mtad_gat"))
    get_model_family_defaults(model_name)
    args.model_name = model_name

    # C3 keeps the fusion-FiLM boundary.  The restricted encoder remains the
    # frozen ablation; prototype_query is the explicit proposed upgrade.
    args.use_transformer = False
    args.feature_att_trans = False
    args.multi_scale_mode = "none"
    args.use_revin = False
    requested_regime_encoder = str(
        getattr(args, "regime_encoder_type", "restricted")
    ).lower()
    args.regime_encoder_type = (
        "prototype_query"
        if requested_regime_encoder == "prototype_query"
        else "restricted"
    )
    args.regime_condition_mode = "fusion"

    # C4 is exactly the independent control-to-response consistency head.
    # These archived branches must never be reachable from an active plan.
    for name in (
        "use_physical_state_encoding",
        "use_physical_regularization",
        "use_physical_response_score",
        "use_physical_response_features",
        "use_control_response_decoder",
        "use_condition_residual_calibration",
        "use_control_conditioned_graph",
        "use_condition_routed_adapter",
        "use_variational_reconstruction",
        "use_c3_joint_relation",
        "use_learnable_sparse_graph",
        "use_relation_change_score",
        "use_relation_prototype_suppression",
    ):
        setattr(args, name, False)
    args.sparse_graph_lambda = 0.0
    args.regime_group_dro_lambda = 0.0
    args.normal_tail_lambda = 0.0
    return args


def resolve_physical_state_config(args):
    """Map Tsinghua EV responses and the configurable C4 state observer."""
    if str(getattr(args, "dataset", "")).upper() != "TSINGHUA_EV":
        return None
    return {
        "dataset_name": "TSINGHUA_EV",
        "current_index": 1,
        "soc_index": 2,
        "consistency_current_index": 1,
        "consistency_soc_index": 2,
        "consistency_response_dims": [0, 3, 4, 5, 6],
        "consistency_encoder_input": str(
            getattr(args, "physical_consistency_encoder_input", "full_window")
        ),
        "consistency_encoder_bidirectional": bool(
            getattr(args, "physical_consistency_encoder_bidirectional", False)
        ),
        "response_score_dims": get_score_dims("TSINGHUA_EV"),
    }


def resolve_regime_config(args, n_features):
    """Select only the control/context channels used by frozen C3."""
    dataset = str(getattr(args, "dataset", "")).upper()
    if str(getattr(args, "regime_encoder_type", "restricted")).lower() == "prototype_query":
        # The proposed C3 is dataset-agnostic: every observed channel becomes
        # one statistical token. Dataset-specific channel filtering is left
        # to explicit future ablations rather than built into the architecture.
        return {"control_indices": list(range(n_features)), "pooled_channels": False}
    if dataset == "TSINGHUA_EV":
        # C3 is a generic data-driven regime encoder.  For Tsinghua EV it sees
        # the full window, just as the anonymous-context route does on MSL/SMAP;
        # only C4 reserves the strict current/SOC-only control-response split.
        return {"control_indices": list(range(n_features)), "pooled_channels": False}
    if dataset in {"MSL", "SMAP"} and n_features > 1:
        return {
            "control_indices": list(range(1, n_features)),
            "pooled_channels": True,
        }
    if dataset in {"SWAT", "WADI"}:
        return {
            "control_indices": list(range(n_features)),
            "pooled_channels": False,
        }
    return {
        "control_indices": list(range(n_features)),
        "pooled_channels": False,
    }


def build_model(args, n_features, window_size, out_dim, target_dims=None):
    """Instantiate the single frozen backbone with optional C3 or C4."""
    resolve_model_args(args)
    regime = resolve_regime_config(args, n_features)
    physical = resolve_physical_state_config(args)
    use_c4 = bool(getattr(args, "use_physical_consistency_head", False))
    if use_c4 and physical is None:
        raise ValueError("Frozen C4 is only defined for TSINGHUA_EV")
    if use_c4 and bool(getattr(args, "use_regime_condition", False)):
        raise ValueError("Frozen C3 and C4 are parallel models and cannot be enabled together")

    return Enhanced_MTADGAT(
        n_features=n_features,
        window_size=window_size,
        out_dim=out_dim,
        kernel_size=args.kernel_size,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        use_gatv2=args.use_gatv2,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        target_dims=target_dims,
        use_regime_condition=bool(getattr(args, "use_regime_condition", False)),
        regime_encoder_type=str(getattr(args, "regime_encoder_type", "restricted")),
        regime_emb_dim=int(getattr(args, "regime_emb_dim", 8)),
        regime_control_indices=regime["control_indices"],
        regime_channel_pooling=regime["pooled_channels"],
        regime_film_scale=float(getattr(args, "regime_film_scale", 0.1)),
        regime_condition_shuffle=bool(getattr(args, "regime_condition_shuffle", False)),
        regime_query_dim=int(getattr(args, "regime_query_dim", 32)),
        regime_num_prototypes=int(getattr(args, "regime_num_prototypes", 6)),
        regime_query_heads=int(getattr(args, "regime_query_heads", 4)),
        regime_top_k=int(getattr(args, "regime_top_k", 2)),
        regime_temperature=float(getattr(args, "regime_temperature", 0.5)),
        use_physical_consistency_head=use_c4,
        physical_consistency_hidden_dim=int(
            getattr(args, "physical_consistency_hidden_dim", 64)
        ),
        physical_consistency_latent_dim=int(
            getattr(args, "physical_consistency_latent_dim", 16)
        ),
        physical_consistency_aux_weight=float(
            getattr(args, "physical_consistency_aux_weight", 0.0)
        ),
        physical_consistency_kl_weight=float(
            getattr(args, "physical_consistency_kl_weight", 0.0001)
        ),
        physical_state_config=physical,
    )
