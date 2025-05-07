import torch
import torch.nn as nn

from modules import (
    ConvLayer,
    DynamicGraphLearner,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
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
            dynamic_graph=True,
            correlation_aware=True,
            use_transformer=True,
    ):
        super(Enhanced_MTADGAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                   use_gatv2)
        self.dynamic_graph = dynamic_graph
        if dynamic_graph:
            self.graph_learner = DynamicGraphLearner(n_features * window_size, hidden_dim=64)
        #TODO 相关性模块加在这里

        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=3 * n_features, nhead=6)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # TODO GRU选项
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers,
                                               dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        # TODO 动态图加的位置不对，而且没和注意力的图结合，还没想好怎么结合，应该在进入图注意力前
        if self.dynamic_graph:
            adj_matrix = self.graph_learner(x)
            h_cat = torch.bmm(adj_matrix, h_cat)  # 应用图结构

        if self.use_transformer:
            _, h_end = self.transformer_encoder(h_cat.permute(1, 0, 2)).permute(1, 0, 2)  # Transformer 处理
        else:
            _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons
