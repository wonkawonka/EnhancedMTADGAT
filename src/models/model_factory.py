"""Build the frozen MTAD-GAT baseline, C3 and C4 architectures."""

from src.data.utils import BMS_FEATURE_NAMES, get_score_dims
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
        "use_physical_graph_bias": False,
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

    # Retire the archived physical branches.  Formal C4 uses only the explicit
    # physical-graph bias and response-consistency switches defined below.
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
    """Map Brand3/BMS controls and responses for the frozen C4 observer."""
    dataset = str(getattr(args, "dataset", "")).upper()
    shared = {
        "consistency_encoder_input": str(
            getattr(args, "physical_consistency_encoder_input", "full_window")
        ),
        "consistency_encoder_bidirectional": bool(
            getattr(args, "physical_consistency_encoder_bidirectional", False)
        ),
        "physical_data_offset": list(getattr(args, "physical_data_min", []) or []),
        "physical_data_scale": list(getattr(args, "physical_data_scale", []) or []),
    }
    if dataset == "TSINGHUA_EV":
        return {
            **shared,
            "dataset_name": dataset,
            "current_index": 1,
            "soc_index": 2,
            "consistency_current_index": 1,
            "consistency_soc_index": 2,
            "consistency_response_dims": [0, 3, 4, 5, 6],
            "response_score_dims": get_score_dims(dataset),
            "physical_graph_roles": {
                "voltage": [0],
                "current": [1],
                "soc": [2],
                "voltage_max": [3],
                "voltage_min": [4],
                "temperature_max": [5],
                "temperature_min": [6],
            },
        }
    if dataset == "BMS":
        # SYS_I is the imposed system-level control; BMSnI remains a scored
        # cluster response so C4 cannot explain away cluster over-current.
        return {
            **shared,
            "dataset_name": dataset,
            "current_index": BMS_FEATURE_NAMES.index("BMSnI"),
            "soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "consistency_current_index": BMS_FEATURE_NAMES.index("SYS_I"),
            "consistency_soc_index": BMS_FEATURE_NAMES.index("BMSnRSOC"),
            "consistency_response_dims": get_score_dims(dataset),
            "response_score_dims": get_score_dims(dataset),
            "physical_graph_roles": {
                "voltage": [
                    BMS_FEATURE_NAMES.index("BMSnVol_T"),
                    BMS_FEATURE_NAMES.index("BMSnVol_B"),
                    BMS_FEATURE_NAMES.index("BMSnVmean"),
                    BMS_FEATURE_NAMES.index("SYS_Vol"),
                ],
                "current": [
                    BMS_FEATURE_NAMES.index("BMSnI"),
                    BMS_FEATURE_NAMES.index("SYS_I"),
                ],
                "soc": [BMS_FEATURE_NAMES.index("BMSnRSOC")],
                "voltage_max": [
                    BMS_FEATURE_NAMES.index("BMSnVmax"),
                    BMS_FEATURE_NAMES.index("SYS_Vmax"),
                ],
                "voltage_min": [
                    BMS_FEATURE_NAMES.index("BMSnVmin"),
                    BMS_FEATURE_NAMES.index("SYS_Vmin"),
                ],
                "temperature_max": [
                    BMS_FEATURE_NAMES.index("BMSnTmax"),
                    BMS_FEATURE_NAMES.index("SYS_Tmax"),
                ],
                "temperature_min": [
                    BMS_FEATURE_NAMES.index("BMSnTmin"),
                    BMS_FEATURE_NAMES.index("SYS_Tmin"),
                ],
                "voltage_spread": [BMS_FEATURE_NAMES.index("cell_v_range")],
                "temperature_spread": [BMS_FEATURE_NAMES.index("cell_t_range")],
            },
        }
    return None


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
    """Instantiate the shared backbone with optional C3 or dual-path C4."""
    resolve_model_args(args)
    regime = resolve_regime_config(args, n_features)
    physical = resolve_physical_state_config(args)
    use_c4 = bool(getattr(args, "use_physical_consistency_head", False))
    use_physical_graph = bool(getattr(args, "use_physical_graph_bias", False))
    if (use_c4 or use_physical_graph) and physical is None:
        raise ValueError("Physical C4 extensions are only defined for TSINGHUA_EV and BMS")
    if (use_c4 or use_physical_graph) and bool(getattr(args, "use_regime_condition", False)):
        raise ValueError("C3 and the physical C4 extensions cannot be enabled together")

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
        regime_condition_shuffle_mode=str(
            getattr(args, "regime_condition_shuffle_mode", "cyclic")
        ),
        regime_query_dim=int(getattr(args, "regime_query_dim", 32)),
        regime_num_prototypes=int(getattr(args, "regime_num_prototypes", 6)),
        regime_query_heads=int(getattr(args, "regime_query_heads", 4)),
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
        use_physical_graph_bias=use_physical_graph,
        physical_graph_bias_weight=float(
            getattr(args, "physical_graph_bias_weight", 0.5)
        ),
        physical_graph_dynamic_weight=float(
            getattr(args, "physical_graph_dynamic_weight", 1.0)
        ),
        physical_graph_gate_scale=float(
            getattr(args, "physical_graph_gate_scale", 5.0)
        ),
        physical_state_config=physical,
    )
