"""实现本项目使用的 MTAD-GAT 模型及其增强变体。"""


import torch

import torch.nn as nn


from src.models.modules import (
    ConvLayer,

    MultiScaleConvLayer,

    FeatureAttentionLayer,

    TemporalAttentionLayer,

    GRULayer,

    Forecasting_Model,

    ReconstructionModel,

    PositionalEncoding,

    PhysicalStateEncoding,

    PhysicalResponseFeatureEncoding,

    PhysicalFeatureAttentionBias,

    ControlResponseDecoderConditioner,

    ControlConditionedResponseVAE,

    ControlConditionedGraphBias,

    VariationalReconstructionModel,

    RevIN,

    WindowRegimeEncoder,

    TemporalRegimeEncoder,

    FiLMConditioner,

    RegimeResidualGate,

)


class Enhanced_MTADGAT(nn.Module):

    """ MTAD-GAT 模型类。


    :param n_features: Number of input features

    :param window_size: Length of the input sequence

    :param out_dim: Number of features to output

    :param kernel_size: size of kernel to use in the 1-D convolution

    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)

           in feat-oriented GAT layer

    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)

           in time-oriented GAT layer

    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT

    :param gru_n_layers: number of layers in the GRU layer

    :param gru_hid_dim: hidden dimension in the GRU layer

    :param forecast_n_layers: number of layers in the FC-based Forecasting Model

    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model

    :param recon_n_layers: number of layers in the GRU-based 重构模型。

    :param recon_hid_dim: hidden dimension in the GRU-based 重构模型。

    :param dropout: dropout rate

    :param alpha: negative slope used in the leaky rely activation function


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

            use_transformer=True,

            trans_enc_layers=2,

            transformer_ff_mult=2.0,

            transformer_norm_first=True,

            attention_top_k=10,

            attention_sparse=False,

            feature_att_trans=False,

            multi_scale_mode='basic',

            multi_scale_dilations=[1, 2, 4],

            use_revin=False,

            revin_affine=True,

            target_dims=None,

            use_regime_condition=False,

            regime_emb_dim=32,

            regime_condition_mode="fusion",

            regime_stat_features=None,

            regime_encoder_type="temporal",

            regime_control_indices=None,

            regime_current_index=None,

            regime_soc_index=None,

            use_physical_state_encoding=False,

            physical_state_hidden_dim=32,

            physical_state_injection_mode="direct",

            physical_state_feature_mode="full",

            use_physical_response_features=False,

            physical_feature_fusion_mode="shared_residual",

            physical_feature_hidden_dim=32,

            use_control_response_decoder=False,

            control_response_hidden_dim=32,

            control_response_aux_weight=0.0,

            use_physical_consistency_head=False,

            physical_consistency_hidden_dim=64,

            physical_consistency_latent_dim=16,

            physical_consistency_aux_weight=0.0,

            physical_consistency_kl_weight=0.0001,

            use_control_conditioned_graph=False,

            condition_graph_emb_dim=32,

            condition_graph_experts=3,

            condition_graph_control_indices=None,

            use_variational_reconstruction=False,

            variational_reconstruction_latent_dim=32,

            variational_reconstruction_kl_weight=0.0001,

            physical_state_config=None,

    ):

        super(Enhanced_MTADGAT, self).__init__()


        self.feature_att_trans = feature_att_trans

        self.use_transformer = use_transformer

        self.use_revin = use_revin

        self.target_dims = target_dims

        self.use_regime_condition = use_regime_condition

        self.regime_condition_mode = regime_condition_mode

        self.regime_encoder_type = regime_encoder_type

        self.regime_current_index = regime_current_index

        self.regime_soc_index = regime_soc_index

        self._regime_aux_prediction = None

        self.physical_state_config = dict(physical_state_config or {}) if physical_state_config is not None else None

        self.use_physical_state_encoding = use_physical_state_encoding and (feature_att_trans or use_transformer)

        self.physical_state_injection_mode = physical_state_injection_mode

        self.physical_state_feature_mode = physical_state_feature_mode

        self.use_physical_response_features = bool(use_physical_response_features)

        self.physical_feature_fusion_mode = physical_feature_fusion_mode

        self.physical_feature_hidden_dim = int(physical_feature_hidden_dim)

        self.use_control_response_decoder = bool(use_control_response_decoder)

        self.control_response_aux_weight = max(0.0, float(control_response_aux_weight))

        self._control_response_probe = None

        self.use_physical_consistency_head = bool(use_physical_consistency_head)

        self.physical_consistency_aux_weight = max(
            0.0, float(physical_consistency_aux_weight)
        )

        self.physical_consistency_kl_weight = max(
            0.0, float(physical_consistency_kl_weight)
        )

        self._physical_consistency_prediction = None

        self._physical_consistency_mean = None

        self._physical_consistency_logvar = None

        self.use_control_conditioned_graph = bool(use_control_conditioned_graph)

        self._condition_graph_routing = None

        self._feature_attention_weights = None

        self.use_variational_reconstruction = bool(use_variational_reconstruction)

        self.variational_reconstruction_kl_weight = max(
            0.0, float(variational_reconstruction_kl_weight)
        )

        self._reconstruction_vae_mean = None

        self._reconstruction_vae_logvar = None

        if self.physical_state_injection_mode not in {"direct", "direct_preserve_rng", "gated_residual"}:

            raise ValueError(

                f"Unsupported physical_state_injection_mode: {self.physical_state_injection_mode}"

            )

        if self.physical_state_feature_mode not in {"full", "controls_only"}:

            raise ValueError(

                f"Unsupported physical_state_feature_mode: {self.physical_state_feature_mode}"

            )

        if self.physical_feature_fusion_mode not in {

            "shared_residual", "shared_film", "feature_gat_residual", "feature_gat_attention_bias",

        }:

            raise ValueError(

                f"Unsupported physical_feature_fusion_mode: {self.physical_feature_fusion_mode}"

            )

        self.use_regime_transformer_residual = (

            self.use_regime_condition

            and self.use_transformer

            and not self.feature_att_trans

            and self.regime_condition_mode == "transformer_residual"

        )

        if self.use_revin:

            self.revin = RevIN(n_features, affine=revin_affine)


        if self.use_regime_condition:

            # Keep construction of the optional C3 branch from changing the
            # initialization and RandomSampler trajectory of the MTAD-GAT
            # backbone.  The branch still receives deterministic parameters;
            # only its RNG consumption is isolated from the shared path.
            regime_rng_state = torch.get_rng_state()

            if regime_encoder_type == "statistics":

                if regime_stat_features is None:

                    regime_stat_features = ["mean", "std", "last", "delta"]

                self.regime_encoder = WindowRegimeEncoder(

                    n_features,

                    emb_dim=regime_emb_dim,

                    stat_features=regime_stat_features,

                    control_indices=regime_control_indices,

                )

            elif regime_encoder_type == "temporal":

                self.regime_encoder = TemporalRegimeEncoder(

                    n_features,

                    emb_dim=regime_emb_dim,

                    control_indices=regime_control_indices,

                )

            else:

                raise ValueError(f"Unsupported regime_encoder_type: {regime_encoder_type}")

            if self.use_regime_transformer_residual:

                self.regime_residual_gate = RegimeResidualGate(regime_emb_dim, gru_hid_dim)

            elif regime_condition_mode == "feature_gat":

                self.feat_conditioner = FiLMConditioner(regime_emb_dim, n_features)

            elif regime_condition_mode == "temporal_gat":

                self.temp_conditioner = FiLMConditioner(regime_emb_dim, n_features)

            elif regime_condition_mode == "fusion":

                fusion_dim = 2 * n_features if feature_att_trans else 3 * n_features

                self.fusion_conditioner = FiLMConditioner(regime_emb_dim, fusion_dim)

            else:

                raise ValueError(f"Unsupported regime_condition_mode: {regime_condition_mode}")

            torch.set_rng_state(regime_rng_state)


        # 根据配置选择卷积模块
        if multi_scale_mode in ['basic', 'progressive']:
            self.conv = MultiScaleConvLayer(n_features, multi_scale_dilations, multi_scale_mode)
        else:
            self.conv = ConvLayer(n_features, kernel_size)
        # 特征注意力层
        self.feature_gat = FeatureAttentionLayer(

            n_features,

            window_size,

            dropout,

            alpha,

            embed_dim=feat_gat_embed_dim,

            use_gatv2=use_gatv2,

            use_bias=True,

            attention_sparse=attention_sparse,

            attention_top_k=attention_top_k,

        )

        if self.use_control_conditioned_graph:
            # The bias experts start at zero, making the initial graph exactly
            # the MTAD-GAT graph; training must earn condition-specific edges.
            graph_rng_state = torch.get_rng_state()
            self.condition_graph = ControlConditionedGraphBias(
                n_features,
                emb_dim=condition_graph_emb_dim,
                expert_count=condition_graph_experts,
                control_indices=condition_graph_control_indices,
            )
            torch.set_rng_state(graph_rng_state)


        # 非简化模式下启用时间注意力层
        if not feature_att_trans:
            self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                       use_gatv2)

        # 简化模式设置：仅特征注意力 + Transformer
        if feature_att_trans:
            d_model = 2 * n_features  # 原始输入 + 特征注意力输出

            self.pos_encoder = PositionalEncoding(d_model, dropout)

            if self.use_physical_state_encoding:

                self.physical_state_encoder = self._build_physical_state_encoder(

                    d_model, physical_state_hidden_dim, physical_state_config, self.physical_state_feature_mode,

                )

                if self.physical_state_injection_mode == "gated_residual":

                    self.physical_state_gate = nn.Parameter(torch.zeros(1))

            nhead = find_largest_valid_nhead(d_model)

            encoder_layer = nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=max(d_model, int(d_model * transformer_ff_mult)),

                dropout=dropout,

                batch_first=True,

                norm_first=transformer_norm_first,

                activation="gelu",

            )

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_enc_layers)
            # 将 Transformer 输出投影到 GRU 隐藏维度
            self.trans_proj = nn.Linear(d_model, gru_hid_dim)
        else:
            d_model = 3 * n_features
            # GRU 仍是主要序列状态模型，Transformer 仅提供残差上下文。

            self.gru = GRULayer(d_model, gru_hid_dim, gru_n_layers, dropout)

            if use_transformer:

                transformer_rng_state = torch.get_rng_state()

                self.pos_encoder = PositionalEncoding(d_model, dropout)

                if self.use_physical_state_encoding:

                    self.physical_state_encoder = self._build_physical_state_encoder(

                        d_model, physical_state_hidden_dim, physical_state_config, self.physical_state_feature_mode,

                    )

                    if self.physical_state_injection_mode == "gated_residual":

                        self.physical_state_gate = nn.Parameter(torch.zeros(1))

                nhead = find_largest_valid_nhead(d_model)

                encoder_layer = nn.TransformerEncoderLayer(

                    d_model=d_model,

                    nhead=nhead,

                    dim_feedforward=max(d_model, int(d_model * transformer_ff_mult)),

                    dropout=dropout,

                    batch_first=True,

                    norm_first=transformer_norm_first,

                    activation="gelu",

                )

                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_enc_layers)

                self.trans_proj = nn.Linear(d_model, gru_hid_dim)

                # C3 starts as the shared GRU backbone and learns whether the
                # long-range Transformer context is useful.  Zeroing only the
                # output projection preserves gradients into the projection on
                # the first step; the encoder begins receiving gradients once
                # that projection departs from zero.
                nn.init.zeros_(self.trans_proj.weight)
                nn.init.zeros_(self.trans_proj.bias)

                torch.set_rng_state(transformer_rng_state)


        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

        if self.use_variational_reconstruction:
            self.recon_model = VariationalReconstructionModel(
                window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,
                dropout, variational_reconstruction_latent_dim,
            )
        else:
            self.recon_model = ReconstructionModel(
                window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout
            )

        if self.use_physical_response_features:

            # Keep the original C3 training trajectory comparable.  The train
            # DataLoader's RandomSampler draws its seed when iteration starts,
            # so constructing an extra module here must not advance the global
            # CPU RNG used by that sampler.
            rng_state = torch.get_rng_state()

            fusion_dim = 2 * n_features if feature_att_trans else 3 * n_features

            try:

                if self.physical_feature_fusion_mode == "shared_film":

                    physical_output_dim = 2 * fusion_dim

                    zero_output = True

                elif self.physical_feature_fusion_mode == "feature_gat_attention_bias":

                    self.physical_feature_attention_bias = PhysicalFeatureAttentionBias(

                        n_features,

                        hidden_dim=self.physical_feature_hidden_dim,

                        config=self.physical_state_config,

                    )

                    physical_output_dim = None

                    zero_output = False

                elif self.physical_feature_fusion_mode == "feature_gat_residual":

                    physical_output_dim = n_features

                    zero_output = False

                    self.physical_feature_gate = nn.Parameter(torch.zeros(1))

                else:

                    physical_output_dim = fusion_dim

                    zero_output = False

                    self.physical_feature_gate = nn.Parameter(torch.zeros(1))

                if physical_output_dim is not None:

                    self.physical_response_feature_encoder = PhysicalResponseFeatureEncoding(

                        physical_output_dim,

                        hidden_dim=self.physical_feature_hidden_dim,

                        config=self.physical_state_config,

                        zero_output=zero_output,

                    )

            finally:

                torch.set_rng_state(rng_state)

        if self.use_control_response_decoder:

            rng_state = torch.get_rng_state()

            try:

                if self.target_dims is None and out_dim == n_features:

                    response_dims = [0, 3, 4, 5, 6] if n_features == 7 else list(range(out_dim))

                    self.control_response_target_dims = response_dims

                else:

                    response_dims = list(range(out_dim))

                    self.control_response_target_dims = (
                        list(self.target_dims) if self.target_dims is not None else response_dims
                    )

                self.control_response_decoder = ControlResponseDecoderConditioner(
                    out_dim,
                    hidden_dim=control_response_hidden_dim,
                    config=self.physical_state_config,
                    response_dims=response_dims,
                )

            finally:

                torch.set_rng_state(rng_state)

        if self.use_physical_consistency_head:

            configured_response_dims = (self.physical_state_config or {}).get(
                "consistency_response_dims"
            )
            if configured_response_dims is not None:
                response_dims = [
                    int(index) for index in configured_response_dims
                    if 0 <= int(index) < out_dim
                ]
                if not response_dims:
                    raise ValueError("C4 consistency_response_dims contains no modeled response channel")
            else:
                response_dims = [0, 3, 4, 5, 6] if n_features == 7 else list(range(out_dim))

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


    def _build_physical_state_encoder(self, d_model, hidden_dim, config, feature_mode):

        if self.physical_state_injection_mode == "direct":

            return PhysicalStateEncoding(
                d_model, hidden_dim=hidden_dim, config=config, feature_mode=feature_mode,
            )

        # A zero gate is only a valid C3-equivalent starting point if adding
        # this dormant branch does not also alter the random initialization of
        # the shared Transformer/GRU path.
        rng_state = torch.get_rng_state()

        encoder = PhysicalStateEncoding(
            d_model, hidden_dim=hidden_dim, config=config, feature_mode=feature_mode,
        )

        torch.set_rng_state(rng_state)

        return encoder


    def _inject_physical_state(self, representation, state_input):

        physical_state = self.physical_state_encoder(state_input)

        if self.physical_state_injection_mode == "gated_residual":

            # Starts as the exact C3 representation.  The scalar can then
            # admit (or suppress) the physical residual from data, without
            # changing the GRU branch or the C3 conditioning location.
            return representation + torch.tanh(self.physical_state_gate) * physical_state

        return representation + physical_state


    def _fuse_physical_feature_gat(self, h_feat, state_input):

        if not self.use_physical_response_features:

            return h_feat

        if self.physical_feature_fusion_mode != "feature_gat_residual":

            return h_feat

        physical = self.physical_response_feature_encoder(state_input)

        return h_feat + torch.tanh(self.physical_feature_gate) * physical


    def _fuse_physical_shared(self, h_cat, state_input):

        if not self.use_physical_response_features:

            return h_cat

        if self.physical_feature_fusion_mode in {"feature_gat_residual", "feature_gat_attention_bias"}:

            return h_cat

        physical = self.physical_response_feature_encoder(state_input)

        if self.physical_feature_fusion_mode == "shared_film":

            gamma, beta = physical.chunk(2, dim=2)

            return h_cat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

        return h_cat + torch.tanh(self.physical_feature_gate) * physical


    def forward(self, x):

        # x 形状为 (b, n, k)：b 为批大小，n 为窗口大小，k 为特征数


        state_input = x

        revin_stats = None

        if self.use_revin:

            x, revin_stats = self.revin(x, mode="norm")


        regime_embedding = None

        if self.use_regime_condition:

            if self.regime_encoder_type == "temporal":

                regime_embedding, self._regime_aux_prediction = self.regime_encoder(

                    state_input,

                    return_auxiliary=True,

                )

            else:

                regime_embedding = self.regime_encoder(x)

                self._regime_aux_prediction = None


        x = self.conv(x)


        physical_attention_bias = None

        if (

            self.use_physical_response_features

            and self.physical_feature_fusion_mode == "feature_gat_attention_bias"

        ):

            physical_attention_bias = self.physical_feature_attention_bias(state_input)

        if self.use_control_conditioned_graph:
            graph_bias, self._condition_graph_routing = self.condition_graph(
                state_input, return_routing=True
            )
            physical_attention_bias = (
                graph_bias if physical_attention_bias is None else physical_attention_bias + graph_bias
            )


        h_feat = self.feature_gat(x, attention_bias=physical_attention_bias)

        self._feature_attention_weights = self.feature_gat.last_attention

        if self.use_regime_condition and self.regime_condition_mode == "feature_gat":

            h_feat = self._apply_condition(h_feat, regime_embedding, self.feat_conditioner)

        h_feat = self._fuse_physical_feature_gat(h_feat, state_input)


        # 特征注意力融合
        if self.feature_att_trans:
            # 简化模式：仅特征注意力 + Transformer
            h_cat = torch.cat([x, h_feat], dim=2)  # 形状：(b, n, 2k)

            if self.use_regime_condition and self.regime_condition_mode == "fusion":

                h_cat = self._apply_condition(h_cat, regime_embedding, self.fusion_conditioner)

            h_cat = self._fuse_physical_shared(h_cat, state_input)


            # 应用 Transformer
            if self.use_physical_state_encoding:

                h_cat = self._inject_physical_state(h_cat, state_input)

            h_cat = self.pos_encoder(h_cat)

            trans_out = self.transformer_encoder(h_cat)

            h_end = trans_out.mean(dim=1)  # 形状：(b, d)

            h_end = self.trans_proj(h_end)  # 形状：(b, gru_hid_dim)

        else:
            # 标准模型：GRU 负责主要状态传播，Transformer 仅提供长程残差上下文。
            h_temp = self.temporal_gat(x)
            if self.use_regime_condition and self.regime_condition_mode == "temporal_gat":
                h_temp = self._apply_condition(h_temp, regime_embedding, self.temp_conditioner)
            h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # 形状：(b, n, 3k)
            if self.use_regime_condition and self.regime_condition_mode == "fusion":
                h_cat = self._apply_condition(h_cat, regime_embedding, self.fusion_conditioner)

            h_cat = self._fuse_physical_shared(h_cat, state_input)

            _, h_gru = self.gru(h_cat)
            h_end = h_gru
            if self.use_transformer:
                if self.use_physical_state_encoding:
                    h_cat = self._inject_physical_state(h_cat, state_input)
                h_cat = self.pos_encoder(h_cat)  # 加入位置编码

                trans_out = self.transformer_encoder(h_cat)

                h_trans = self.trans_proj(trans_out.mean(dim=1))

                if self.use_regime_transformer_residual:

                    residual_gate = self.regime_residual_gate(regime_embedding)

                    h_end = h_gru + residual_gate * h_trans

                else:

                    h_end = h_gru + h_trans


        h_end = h_end.view(x.shape[0], -1)  # 展平为批次二维张量


        predictions = self.forecasting_model(h_end)

        if self.use_variational_reconstruction:
            recons, self._reconstruction_vae_mean, self._reconstruction_vae_logvar = self.recon_model(h_end)
        else:
            recons = self.recon_model(h_end)

        if self.use_control_response_decoder:

            forecast_correction, reconstruction_correction, self._control_response_probe = (
                self.control_response_decoder(state_input)
            )

            predictions = predictions + forecast_correction

            recons = recons + reconstruction_correction

        if self.use_physical_consistency_head:

            (
                self._physical_consistency_prediction,
                self._physical_consistency_mean,
                self._physical_consistency_logvar,
            ) = self.physical_consistency_head(state_input)


        if self.use_revin:

            predictions = self.revin(predictions, mode="denorm", stats=revin_stats, target_dims=self.target_dims)

            recons = self.revin(recons, mode="denorm", stats=revin_stats, target_dims=self.target_dims)


        return predictions, recons


    def regime_auxiliary_loss(self, x):

        """Self-supervise the learned dynamic-state embedding with window descriptors."""

        if self._regime_aux_prediction is None or self.regime_encoder_type != "temporal":

            return self.control_response_auxiliary_loss(x)

        current_index = self.regime_current_index

        if current_index is None or current_index < 0 or current_index >= x.size(2):

            control_indices = getattr(self.regime_encoder, "control_indices", None)

            if control_indices is None:

                current = x.mean(dim=2)

            else:

                current = torch.index_select(x, dim=2, index=control_indices).mean(dim=2)

        else:

            current = x[:, :, current_index]

        current_scale = current.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)

        normalized_current = current / current_scale

        mean_activity = normalized_current.abs().mean(dim=1)

        current_variability = normalized_current.std(dim=1, unbiased=False)

        sign_switch_rate = (

            (normalized_current[:, 1:] * normalized_current[:, :-1]) < 0

        ).float().mean(dim=1)

        soc_index = self.regime_soc_index

        if soc_index is None or soc_index < 0 or soc_index >= x.size(2):

            state_delta = normalized_current[:, -1] - normalized_current[:, 0]

        else:

            soc = x[:, :, soc_index]

            soc_scale = (soc.amax(dim=1) - soc.amin(dim=1)).clamp_min(1e-6)

            state_delta = (soc[:, -1] - soc[:, 0]) / soc_scale

        targets = torch.stack(

            [mean_activity, current_variability, sign_switch_rate, state_delta],

            dim=1,

        ).detach()

        return (
            torch.nn.functional.smooth_l1_loss(self._regime_aux_prediction, targets)
            + self.control_response_auxiliary_loss(x)
        )


    def control_response_auxiliary_loss(self, x):

        loss = x.new_tensor(0.0)

        if (
            self.use_variational_reconstruction
            and self._reconstruction_vae_mean is not None
            and self.variational_reconstruction_kl_weight > 0.0
        ):
            mean = self._reconstruction_vae_mean
            logvar = self._reconstruction_vae_logvar
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
            loss = loss + self.variational_reconstruction_kl_weight * kl_loss

        if self._control_response_probe is not None and self.control_response_aux_weight > 0.0:

            targets = x[:, :, self.control_response_target_dims]

            loss = loss + self.control_response_aux_weight * torch.nn.functional.smooth_l1_loss(
                self._control_response_probe, targets.detach()
            )

        if (
            self._physical_consistency_prediction is not None
            and self.physical_consistency_aux_weight > 0.0
        ):

            targets = x[:, :, self.physical_consistency_target_dims]

            response_loss = torch.nn.functional.smooth_l1_loss(
                self._physical_consistency_prediction, targets.detach()
            )

            mean = self._physical_consistency_mean

            logvar = self._physical_consistency_logvar

            kl_loss = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())

            loss = loss + self.physical_consistency_aux_weight * (
                response_loss + self.physical_consistency_kl_weight * kl_loss
            )

        return loss


    def encode_regime(self, x):

        """Expose dynamic-state embeddings for external probing and visualization."""

        if not self.use_regime_condition:

            raise RuntimeError("Regime conditioning is disabled for this model")

        if self.regime_encoder_type == "temporal":

            return self.regime_encoder(x)

        if self.use_revin:

            x, _ = self.revin(x, mode="norm")

        return self.regime_encoder(x)


    @staticmethod

    def _apply_condition(hidden_state, regime_embedding, conditioner):

        gamma, beta = conditioner(regime_embedding)

        gamma = gamma.unsqueeze(1)

        beta = beta.unsqueeze(1)

        # Condition the backbone conservatively: the operating regime should
        # adapt MTAD-GAT features, not replace their scale or offset.  The
        # bounded 10% modulation also makes the learned C3 increment robust to
        # small development folds.
        return hidden_state * (1.0 + 0.1 * gamma) + 0.1 * beta


def find_largest_valid_nhead(d_model, max_nhead=8):

    for nhead in range(max_nhead, 0, -1):

        if d_model % nhead == 0:

            return nhead

    return 1
