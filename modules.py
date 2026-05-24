import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization for time-series windows."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    @staticmethod
    def _normalize_target_dims(target_dims):
        if isinstance(target_dims, int):
            return [target_dims]
        return target_dims

    def _slice_stats(self, stats, target_dims):
        target_dims = self._normalize_target_dims(target_dims)
        if target_dims is None:
            return stats["mean"], stats["stdev"]
        return stats["mean"][:, :, target_dims], stats["stdev"][:, :, target_dims]

    def _slice_affine(self, target_dims):
        target_dims = self._normalize_target_dims(target_dims)
        if target_dims is None:
            return self.affine_weight, self.affine_bias
        return self.affine_weight[:, :, target_dims], self.affine_bias[:, :, target_dims]

    def forward(self, x, mode, stats=None, target_dims=None):
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - mean) / stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x, {"mean": mean, "stdev": stdev}

        if mode == "denorm":
            if stats is None:
                raise ValueError("RevIN denorm requires normalization statistics")

            mean, stdev = self._slice_stats(stats, target_dims)
            squeeze_time_dim = x.ndim == 2
            if squeeze_time_dim:
                x = x.unsqueeze(1)

            if self.affine:
                affine_weight, affine_bias = self._slice_affine(target_dims)
                x = (x - affine_bias) / (affine_weight + self.eps)
            x = x * stdev + mean

            if squeeze_time_dim:
                x = x.squeeze(1)
            return x

        raise ValueError(f"Unsupported RevIN mode: {mode}")


class WindowRegimeEncoder(nn.Module):
    """Encode a sliding window into a compact regime embedding."""

    def __init__(self, n_features, emb_dim=32, hidden_dim=None, stat_features=None):
        super(WindowRegimeEncoder, self).__init__()
        if stat_features is None:
            stat_features = ["mean", "std", "last", "delta"]

        self.n_features = n_features
        self.stat_features = list(stat_features)
        input_dim = len(self.stat_features) * n_features
        hidden_dim = hidden_dim or max(emb_dim, min(128, input_dim))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        stats = []
        for stat_name in self.stat_features:
            if stat_name == "mean":
                stats.append(x.mean(dim=1))
            elif stat_name == "std":
                stats.append(torch.std(x, dim=1, unbiased=False))
            elif stat_name == "last":
                stats.append(x[:, -1, :])
            elif stat_name == "delta":
                stats.append(x[:, -1, :] - x[:, 0, :])
            else:
                raise ValueError(f"Unsupported regime stat feature: {stat_name}")

        regime_input = torch.cat(stats, dim=1)
        return self.mlp(regime_input)


class FiLMConditioner(nn.Module):
    """Generate feature-wise affine modulation parameters from regime embedding."""

    def __init__(self, emb_dim, target_dim):
        super(FiLMConditioner, self).__init__()
        self.proj = nn.Linear(emb_dim, target_dim * 2)
        self.target_dim = target_dim

    def forward(self, regime_embedding):
        gamma, beta = torch.chunk(self.proj(regime_embedding), 2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return gamma, beta


class RegimeResidualGate(nn.Module):
    """Generate regime-aware gates for transformer residual enhancement."""

    def __init__(self, emb_dim, target_dim):
        super(RegimeResidualGate, self).__init__()
        self.proj = nn.Linear(emb_dim, target_dim)

    def forward(self, regime_embedding):
        return torch.sigmoid(self.proj(regime_embedding))


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class MultiScaleConvLayer(nn.Module):
    """多尺度因果膨胀卷积层 (Causal Dilated Convolution)
    
    :param n_features: 输入特征数
    :param dilations: 膨胀率列表，如 [1, 2, 4]
    :param mode: 'basic' 或 'progressive'
    """
    
    def __init__(self, n_features, dilations=[1, 2, 4], mode='basic'):
        super(MultiScaleConvLayer, self).__init__()
        
        self.mode = mode
        self.dilations = dilations
        self.n_scales = len(dilations)
        self.kernel_size = 3  # 固定卷积核大小为 3
        
        print(f"初始化多尺度因果膨胀卷积: mode={mode}, dilations={dilations}")
        
        # 多尺度卷积分支：每个分支使用指定的膨胀率
        self.scale_convs = nn.ModuleList()
        for d in dilations:
            # padding=0，我们在 forward 中手动计算并执行因果填充
            self.scale_convs.append(nn.Sequential(
                nn.Conv1d(n_features, n_features, self.kernel_size, dilation=d, padding=0, bias=False),
                nn.BatchNorm1d(n_features),
                nn.ReLU()
            ))
        
        # 基础模式的融合卷积
        self.basic_fusion_conv = nn.Conv1d(
            n_features * self.n_scales, 
            n_features, 
            kernel_size=1,
            bias=False
        )
        
        # 渐进式融合模块（仅在progressive模式下使用）
        if mode == 'progressive':
            self.progressive_fusers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(n_features * 2, n_features, kernel_size=1, bias=False),
                    nn.BatchNorm1d(n_features),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ) for _ in range(self.n_scales - 1)
            ])
            
            # 残差投影
            self.residual_proj = nn.ModuleList([
                nn.Conv1d(n_features, n_features, kernel_size=1, bias=False)
                for _ in range(self.n_scales)
            ])
    
    def forward(self, x):
        """
        x: (batch, window, n_features)
        """
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, window)
        
        # 提取各尺度特征（采用因果膨胀卷积）
        scale_feats = []
        for i, conv in enumerate(self.scale_convs):
            dilation = self.dilations[i]
            
            # 【关键】因果膨胀卷积的填充公式：padding = (k - 1) * dilation
            # 这样可以确保输出长度与输入一致，且只看过去和现在
            padding_size = (self.kernel_size - 1) * dilation
            x_padded = F.pad(x_perm, (padding_size, 0))
            
            feat = conv(x_padded)
            scale_feats.append(feat)
        
        if self.mode == 'basic':
            # 基础模式：拼接后融合
            concatenated = torch.cat(scale_feats, dim=1)
            fused = self.basic_fusion_conv(concatenated)
            return fused.permute(0, 2, 1)
        
        else:  # progressive mode
            # 渐进式融合：从最小尺度开始，逐步融合
            fused = scale_feats[0]
            
            for next_scale, fuser in zip(scale_feats[1:], self.progressive_fusers):
                combined = torch.cat([fused, next_scale], dim=1)
                fused = fuser(combined)
            
            # 添加多尺度残差
            residual_sum = sum(
                proj(scale_feats[i]) 
                for i, proj in enumerate(self.residual_proj)
            )
            
            final_feat = fused + residual_sum * 0.1
            return final_feat.permute(0, 2, 1)



class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(
        self,
        n_features,
        window_size,
        dropout,
        alpha,
        embed_dim=None,
        use_gatv2=True,
        use_bias=True,
        attention_sparse=True,
        attention_top_k=10,
    ):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.attention_sparse = attention_sparse
        self.attention_top_k = attention_top_k

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            # 把每个节点与其他所有节点拼接起来
            a_input = self._make_attention_input(x)  # (b, k, k, 2*window_size)
            # 经过一个线性层 + LeakyReLU 激活函数，得到中间表示
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            # 使用可学习的参数 self.a 做点积，得到原始注意力分数 e
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        if self.attention_sparse:
            # TODO 创新点1-随机稀疏化代替物理拓扑稀疏化，稀疏化注意力权重
            # 只保留每个节点最重要的top-k个连接
            top_k = min(self.attention_top_k, self.n_features)  # 保留每个节点的top-k连接
            # e是注意力分数张量，形状为(batch_size, n_features, n_features)
            topk_values, topk_indices = torch.topk(e, top_k, dim=2)
            sparse_e = torch.full_like(e, float('-inf'))
            sparse_e.scatter_(2, topk_indices, topk_values)
            e = sparse_e

        # Attention weights，softmax over feature dimension
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))  # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # 安全切片防止越界

        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PhysicalStateEncoding(nn.Module):
    """Project battery-oriented state features into the transformer hidden space."""

    def __init__(self, d_model, hidden_dim=32, config=None, eps=1e-6):
        super(PhysicalStateEncoding, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.config = dict(config or {})
        self.eps = eps

        self.current_index = self.config.get("current_index")
        self.voltage_index = self.config.get("voltage_index")
        self.temperature_index = self.config.get("temperature_index")
        self.step_type_index = self.config.get("step_type_index")
        self.soc_index = self.config.get("soc_index")
        self.soh_index = self.config.get("soh_index")

        self.state_dim = 5
        if self.soc_index is not None:
            self.state_dim += 1
        if self.soh_index is not None:
            self.state_dim += 1

        self.proj = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        current = self._get_channel(x, self.current_index)
        phase = self._compute_phase(x, current)
        current_norm = self._signed_normalize(current)
        charge_flow = self._compute_charge_flow(current)
        voltage_rel = self._compute_relative_position(self._get_channel(x, self.voltage_index))
        temperature_rel = self._compute_relative_position(self._get_channel(x, self.temperature_index))

        state_parts = [phase, current_norm, charge_flow, voltage_rel, temperature_rel]

        if self.soc_index is not None:
            state_parts.append(self._compute_absolute_state(self._get_channel(x, self.soc_index)))
        if self.soh_index is not None:
            state_parts.append(self._compute_absolute_state(self._get_channel(x, self.soh_index)))

        state_tensor = torch.cat(state_parts, dim=2)
        return self.proj(state_tensor)

    def _get_channel(self, x, index):
        if index is None or index < 0 or index >= x.size(2):
            return x.new_zeros(x.size(0), x.size(1), 1)
        return x[:, :, index:index + 1]

    def _signed_normalize(self, value):
        scale = value.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.tanh(value / scale)

    def _compute_phase(self, x, current):
        if self.step_type_index is not None:
            step_type = self._get_channel(x, self.step_type_index)
            return torch.clamp(step_type, -1.0, 1.0)

        centered_current = current - current.mean(dim=1, keepdim=True)
        scale = centered_current.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.tanh(centered_current / scale)

    def _compute_charge_flow(self, current):
        cumulative_current = torch.cumsum(current, dim=1)
        scale = cumulative_current.abs().amax(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.clamp(cumulative_current / scale, -1.0, 1.0)

    def _compute_relative_position(self, value):
        value_min = value.amin(dim=1, keepdim=True)
        value_max = value.amax(dim=1, keepdim=True)
        return 2.0 * (value - value_min) / (value_max - value_min + self.eps) - 1.0

    def _compute_absolute_state(self, value):
        value_min = value.amin(dim=1, keepdim=True)
        value_max = value.amax(dim=1, keepdim=True)
        if torch.all((value_min >= -self.eps) & (value_max <= 1.0 + self.eps)):
            return torch.clamp(2.0 * value - 1.0, -1.0, 1.0)

        scale = value.abs().amax(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.clamp(value / scale, -1.0, 1.0)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
