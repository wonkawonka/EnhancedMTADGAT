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

    RevIN,

    WindowRegimeEncoder,

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

            regime_condition_mode="transformer_residual",

            regime_stat_features=None,

            use_physical_state_encoding=False,

            physical_state_hidden_dim=32,

            physical_state_config=None,

    ):

        super(Enhanced_MTADGAT, self).__init__()


        self.feature_att_trans = feature_att_trans

        self.use_transformer = use_transformer

        self.use_revin = use_revin

        self.target_dims = target_dims

        self.use_regime_condition = use_regime_condition

        self.regime_condition_mode = regime_condition_mode

        self.physical_state_config = dict(physical_state_config or {}) if physical_state_config is not None else None

        self.use_physical_state_encoding = use_physical_state_encoding and (feature_att_trans or use_transformer)

        self.use_regime_transformer_residual = (

            self.use_regime_condition

            and self.use_transformer

            and not self.feature_att_trans

            and self.regime_condition_mode == "transformer_residual"

        )

        if self.use_revin:

            self.revin = RevIN(n_features, affine=revin_affine)


        if self.use_regime_condition:

            if regime_stat_features is None:

                regime_stat_features = ["mean", "std", "last", "delta"]

            self.regime_encoder = WindowRegimeEncoder(

                n_features,

                emb_dim=regime_emb_dim,

                stat_features=regime_stat_features,

            )

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


        # 非简化模式下启用时间注意力层
        if not feature_att_trans:
            self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                       use_gatv2)

        # 简化模式设置：仅特征注意力 + Transformer
        if feature_att_trans:
            d_model = 2 * n_features  # 原始输入 + 特征注意力输出

            self.pos_encoder = PositionalEncoding(d_model, dropout)

            if self.use_physical_state_encoding:

                self.physical_state_encoder = PhysicalStateEncoding(

                    d_model,

                    hidden_dim=physical_state_hidden_dim,

                    config=physical_state_config,

                )

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

                self.pos_encoder = PositionalEncoding(d_model, dropout)

                if self.use_physical_state_encoding:

                    self.physical_state_encoder = PhysicalStateEncoding(

                        d_model,

                        hidden_dim=physical_state_hidden_dim,

                        config=physical_state_config,

                    )

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


        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,

                                               dropout)


    def forward(self, x):

        # x 形状为 (b, n, k)：b 为批大小，n 为窗口大小，k 为特征数


        state_input = x

        revin_stats = None

        if self.use_revin:

            x, revin_stats = self.revin(x, mode="norm")


        regime_embedding = None

        if self.use_regime_condition:

            regime_embedding = self.regime_encoder(x)


        x = self.conv(x)


        h_feat = self.feature_gat(x)

        if self.use_regime_condition and self.regime_condition_mode == "feature_gat":

            h_feat = self._apply_condition(h_feat, regime_embedding, self.feat_conditioner)


        # 特征注意力融合
        if self.feature_att_trans:
            # 简化模式：仅特征注意力 + Transformer
            h_cat = torch.cat([x, h_feat], dim=2)  # 形状：(b, n, 2k)

            if self.use_regime_condition and self.regime_condition_mode == "fusion":

                h_cat = self._apply_condition(h_cat, regime_embedding, self.fusion_conditioner)


            # 应用 Transformer
            if self.use_physical_state_encoding:

                h_cat = h_cat + self.physical_state_encoder(state_input)

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

            _, h_gru = self.gru(h_cat)
            h_end = h_gru
            if self.use_transformer:
                if self.use_physical_state_encoding:
                    h_cat = h_cat + self.physical_state_encoder(state_input)
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

        recons = self.recon_model(h_end)


        if self.use_revin:

            predictions = self.revin(predictions, mode="denorm", stats=revin_stats, target_dims=self.target_dims)

            recons = self.revin(recons, mode="denorm", stats=revin_stats, target_dims=self.target_dims)


        return predictions, recons


    @staticmethod

    def _apply_condition(hidden_state, regime_embedding, conditioner):

        gamma, beta = conditioner(regime_embedding)

        gamma = gamma.unsqueeze(1)

        beta = beta.unsqueeze(1)

        return hidden_state * (1.0 + gamma) + beta


def find_largest_valid_nhead(d_model, max_nhead=8):

    for nhead in range(max_nhead, 0, -1):

        if d_model % nhead == 0:

            return nhead

    return 1


