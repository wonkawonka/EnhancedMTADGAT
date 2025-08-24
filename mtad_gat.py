import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel, PositionalEncoding, CorrelationLayer,
    MultiScaleStackedAttentionLayer
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
            correlation_aware=True,
            use_transformer=True,
            trans_enc_layers=2,
            top_k=20,
            attention_top_k=10,
            attention_sparse=True,
            corr_dim=40,
            corr_alpha=3,
            # 新增参数
            multi_scale_stacked=True,
            window_sizes=None,
            num_attention_stacks=2
    ):
        super(Enhanced_MTADGAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        # 相关性层
        self.correlation_aware = correlation_aware
        if correlation_aware:
            self.corr_adj = CorrelationLayer(n_features, top_k, corr_dim,corr_alpha)

        self.multi_scale_stacked = multi_scale_stacked
        # 根据配置选择不同的注意力机制
        if multi_scale_stacked:
            # 多尺度堆叠注意力
            window_sizes = window_sizes if window_sizes is not None else [window_size // 4, window_size // 2,
                                                                          window_size]
            self.multi_scale_attn = MultiScaleStackedAttentionLayer(
                n_features, window_sizes, dropout, alpha, feat_gat_embed_dim, time_gat_embed_dim,
                use_gatv2, num_attention_stacks, attention_sparse,attention_top_k
            )
        else:
            #图注意力层
            self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2,
                                                     attention_sparse=attention_sparse,attention_top_k=attention_top_k)
            self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                   use_gatv2)
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

        if self.correlation_aware:
            adj_matrix = self.corr_adj(x)
        else:
            adj_matrix = None

        if self.multi_scale_stacked:
            # 多尺度堆叠注意力处理
            # 创建不同尺度的输入
            scales = [max(1, x.size(1) // 4), max(1, x.size(1) // 2), x.size(1)]
            x_list = []

            for scale in scales:
                if scale == x.size(1):
                    x_scaled = x
                else:
                    # 使用自适应平均池化调整时间维度
                    x_scaled = F.adaptive_avg_pool1d(x.transpose(1, 2), scale).transpose(1, 2)
                x_list.append(x_scaled)

            h_combined = self.multi_scale_attn(x_list, adj_matrix)
            h_feat = h_combined
            h_temp = h_combined
        else:
            # 原始注意力机制
            h_feat = self.feature_gat(x,adj_matrix)
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
