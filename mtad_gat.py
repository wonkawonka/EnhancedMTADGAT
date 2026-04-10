import torch
import torch.nn as nn

from modules import (
    ConvLayer,
    MultiScaleConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel, PositionalEncoding,
)


class Enhanced_MTADGAT(nn.Module):
    """ MTAD-GAT model class.

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
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
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
            attention_top_k=10,
            attention_sparse=False,
            feature_att_trans=False,
            multi_scale_mode='basic',
            multi_scale_dilations=[1, 2, 4]
    ):
        super(Enhanced_MTADGAT, self).__init__()

        self.feature_att_trans = feature_att_trans
        
        # 多尺度卷积（根据mode选择）
        if multi_scale_mode in ['basic', 'progressive']:
            self.conv = MultiScaleConvLayer(n_features, multi_scale_dilations, multi_scale_mode)
        else:
            self.conv = ConvLayer(n_features, kernel_size)
        # 图注意力层
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2, attention_sparse, attention_top_k)
        
        # 仅在不使用简化模型时包含时间注意力层
        if not feature_att_trans:
            self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                       use_gatv2)
        
        # 简化模型的设置（仅特征注意力 + transformer）
        if feature_att_trans:
            d_model = 2 * n_features  # 仅特征注意力输出 + 卷积输出
            self.pos_encoder = PositionalEncoding(d_model, dropout)
            nhead = find_largest_valid_nhead(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_enc_layers)
            # 投影到预测/重构模型所需的维度
            self.trans_proj = nn.Linear(d_model, gru_hid_dim)
        else:
            d_model = 3 * n_features
            # 是否在GRU前加transformer encoder
            self.use_transformer = use_transformer
            if use_transformer:
                self.pos_encoder = PositionalEncoding(d_model, dropout)
                nhead = find_largest_valid_nhead(d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_enc_layers)
                self.trans_proj = nn.Linear(d_model, gru_hid_dim)  # d_model -> 150
            # TODO 注意当特征只有几个的时候需要调整一下
            else:
                self.gru = GRULayer(d_model, gru_hid_dim, gru_n_layers, dropout)
                
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,
                                               dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)

        h_feat = self.feature_gat(x)
        
        # 根据模型配置进行处理
        if self.feature_att_trans:
            # 简化模型：仅特征注意力 + transformer
            h_cat = torch.cat([x, h_feat], dim=2)  # (b, n, 2k)
            
            # 应用transformer
            h_cat = self.pos_encoder(h_cat)
            trans_out = self.transformer_encoder(h_cat)
            h_end = trans_out.mean(dim=1)  # (b, d)
            h_end = self.trans_proj(h_end)  # (b, gru_hid_dim)
        else:
            # 标准模型：特征注意力 + 时间注意力 + 可选GRU/Transformer
            h_temp = self.temporal_gat(x)
            h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

            if self.use_transformer:
                # transformer_out = self.transformer_encoder(h_cat.permute(1, 0, 2))
                # h_end = transformer_out.mean(dim=0)  # [b, d]
                h_cat = self.pos_encoder(h_cat)  # 添加位置信息
                trans_out = self.transformer_encoder(h_cat)
                h_end = trans_out.mean(dim=1)  # (b, d)
                h_end = self.trans_proj(h_end)  # (b, 150)
            else:
                _, h_end = self.gru(h_cat)
                
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons


def find_largest_valid_nhead(d_model, max_nhead=8):
    for nhead in range(max_nhead, 0, -1):
        if d_model % nhead == 0:
            return nhead
    return 1