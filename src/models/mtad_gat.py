"""Frozen MTAD-GAT baseline with the final C3 and C4 modules only."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import (
    ControlConditionedResponseVAE,
    ConvLayer,
    FeatureAttentionLayer,
    FiLMConditioner,
    Forecasting_Model,
    GRULayer,
    PrototypeQueryRegimeEncoder,
    ReconstructionModel,
    RestrictedStateEncoder,
    TemporalAttentionLayer,
)


class Enhanced_MTADGAT(nn.Module):
    """Original MTAD-GAT plus two parallel, frozen research extensions.

    C3 can use the frozen restricted encoder or the prototype-query upgrade;
    both retain fusion-level FiLM. C4 is an independent response-aware conditional
    autoencoder by default; its strict control-only form remains an explicit
    ablation. The two extensions cannot be enabled together.
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        target_dims=None,
        use_regime_condition=False,
        regime_encoder_type="restricted",
        regime_emb_dim=8,
        regime_control_indices=None,
        regime_channel_pooling=False,
        regime_film_scale=0.1,
        regime_condition_shuffle=False,
        regime_query_dim=32,
        regime_num_prototypes=6,
        regime_query_heads=4,
        regime_temperature=0.5,
        use_physical_consistency_head=False,
        physical_consistency_hidden_dim=64,
        physical_consistency_latent_dim=16,
        physical_consistency_aux_weight=0.0,
        physical_consistency_kl_weight=0.0001,
        physical_state_config=None,
    ):
        super().__init__()
        if use_regime_condition and use_physical_consistency_head:
            raise ValueError("Frozen C3 and C4 are parallel and cannot be enabled together")

        self.target_dims = target_dims
        self.use_regime_condition = bool(use_regime_condition)
        self.regime_encoder_type = str(regime_encoder_type).lower()
        self.regime_condition_mode = "fusion"
        self.regime_channel_pooling = bool(regime_channel_pooling)
        self.regime_film_scale = max(0.0, min(float(regime_film_scale), 1.0))
        self.regime_condition_shuffle = bool(regime_condition_shuffle)
        self.use_physical_consistency_head = bool(use_physical_consistency_head)
        self.physical_consistency_aux_weight = max(
            0.0, float(physical_consistency_aux_weight)
        )
        self.physical_consistency_kl_weight = max(
            0.0, float(physical_consistency_kl_weight)
        )
        self.physical_state_config = (
            dict(physical_state_config or {}) if physical_state_config is not None else None
        )

        # The public/industrial frozen checkpoints contain this empty buffer,
        # while the earlier independent C4 checkpoints predate it.  Preserve
        # each frozen route's exact state_dict layout so strict loading keeps
        # working for both model families.
        if not self.use_physical_consistency_head:
            self.register_buffer(
                "backbone_feature_indices", torch.tensor([], dtype=torch.long)
            )

        self._regime_embedding = None
        self._regime_aux_prediction = None
        self._regime_aux_target = None
        self._feature_attention_weights = None
        self._feature_attention_probabilities = None
        self._temporal_attention_weights = None
        self._physical_consistency_prediction = None
        self._physical_consistency_mean = None
        self._physical_consistency_logvar = None

        if self.use_regime_condition:
            # Preserve the original backbone RNG trajectory when C3 is added.
            rng_state = torch.get_rng_state()
            if self.regime_encoder_type == "prototype_query":
                self.regime_encoder = PrototypeQueryRegimeEncoder(
                    n_features,
                    emb_dim=regime_emb_dim,
                    model_dim=regime_query_dim,
                    num_prototypes=regime_num_prototypes,
                    num_heads=regime_query_heads,
                    temperature=regime_temperature,
                    control_indices=regime_control_indices,
                )
            else:
                self.regime_encoder = RestrictedStateEncoder(
                    n_features,
                    emb_dim=regime_emb_dim,
                    control_indices=regime_control_indices,
                    pooled_channels=self.regime_channel_pooling,
                )
            self.fusion_conditioner = FiLMConditioner(
                regime_emb_dim, 3 * n_features
            )
            torch.set_rng_state(rng_state)

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            embed_dim=feat_gat_embed_dim,
            use_gatv2=use_gatv2,
            use_bias=True,
            attention_sparse=False,
            attention_top_k=10,
            output_activation="sigmoid",
            learnable_sparse_graph=False,
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features,
            window_size,
            dropout,
            alpha,
            time_gat_embed_dim,
            use_gatv2,
            output_activation="sigmoid",
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim,
            forecast_hid_dim,
            out_dim,
            forecast_n_layers,
            dropout,
        )
        self.recon_model = ReconstructionModel(
            window_size,
            gru_hid_dim,
            recon_hid_dim,
            out_dim,
            recon_n_layers,
            dropout,
        )

        if self.use_physical_consistency_head:
            response_dims = (self.physical_state_config or {}).get(
                "consistency_response_dims"
            )
            if response_dims is None:
                response_dims = [0, 3, 4, 5, 6] if n_features == 7 else list(range(out_dim))
            response_dims = [
                int(index) for index in response_dims if 0 <= int(index) < out_dim
            ]
            if not response_dims:
                raise ValueError("C4 has no valid response channel")
            self.physical_consistency_target_dims = response_dims
            rng_state = torch.get_rng_state()
            self.physical_consistency_head = ControlConditionedResponseVAE(
                n_features,
                response_dims,
                hidden_dim=physical_consistency_hidden_dim,
                latent_dim=physical_consistency_latent_dim,
                config=self.physical_state_config,
            )
            torch.set_rng_state(rng_state)

    def forward(self, x):
        state_input = x
        regime_embedding = None
        self._regime_aux_prediction = None
        self._regime_aux_target = None

        if self.use_regime_condition:
            condition_input = state_input
            if self.regime_condition_shuffle and state_input.size(0) > 1:
                condition_input = state_input.clone()
                indices = self.regime_encoder.control_indices
                controls = torch.index_select(state_input, dim=2, index=indices)
                condition_input[:, :, indices] = torch.roll(controls, shifts=1, dims=0)
            (
                regime_embedding,
                self._regime_aux_prediction,
                self._regime_aux_target,
            ) = self.regime_encoder(condition_input, return_auxiliary=True)
        self._regime_embedding = regime_embedding

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        self._feature_attention_weights = self.feature_gat.last_attention
        self._feature_attention_probabilities = self.feature_gat.last_attention_raw
        h_temp = self.temporal_gat(x)
        self._temporal_attention_weights = self.temporal_gat.last_attention
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)
        if self.use_regime_condition:
            h_cat = self._apply_condition(
                h_cat, regime_embedding, self.fusion_conditioner
            )
        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)
        predictions = self.forecasting_model(h_end)
        reconstructions = self.recon_model(h_end)

        if self.use_physical_consistency_head:
            (
                self._physical_consistency_prediction,
                self._physical_consistency_mean,
                self._physical_consistency_logvar,
            ) = self.physical_consistency_head(state_input)
        return predictions, reconstructions

    def regime_auxiliary_loss(self, x):
        loss = x.new_tensor(0.0)
        if (
            self._regime_aux_prediction is not None
            and self._regime_aux_target is not None
        ):
            loss = loss + F.smooth_l1_loss(
                self._regime_aux_prediction, self._regime_aux_target
            )
        if (
            self.use_physical_consistency_head
            and self._physical_consistency_prediction is not None
            and self.physical_consistency_aux_weight > 0.0
        ):
            targets = x[:, :, self.physical_consistency_target_dims]
            response_loss = F.smooth_l1_loss(
                self._physical_consistency_prediction, targets.detach()
            )
            mean = self._physical_consistency_mean
            logvar = self._physical_consistency_logvar
            kl_loss = -0.5 * torch.mean(
                1.0 + logvar - mean.square() - logvar.exp()
            )
            loss = loss + self.physical_consistency_aux_weight * (
                response_loss + self.physical_consistency_kl_weight * kl_loss
            )
        return loss

    def control_response_auxiliary_loss(self, x):
        """Compatibility name for the frozen C4 consistency objective."""
        return self.regime_auxiliary_loss(x)

    def regime_prototype_loss(self):
        if not self.use_regime_condition:
            reference = next(self.parameters())
            return reference.new_tensor(0.0)
        collapse_loss = getattr(self.regime_encoder, "routing_collapse_loss", None)
        if collapse_loss is None:
            return self._regime_embedding.new_tensor(0.0)
        return collapse_loss()

    def encode_regime(self, x):
        if not self.use_regime_condition:
            raise RuntimeError("Frozen C3 is disabled")
        return self.regime_encoder(x)

    def _apply_condition(self, hidden_state, embedding, conditioner):
        gamma, beta = conditioner(embedding)
        while gamma.ndim < hidden_state.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        scale = self.regime_film_scale
        return hidden_state * (1.0 + scale * gamma) + scale * beta
