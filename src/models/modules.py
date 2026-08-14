"""定义 MTAD-GAT 变体可复用的神经网络构建模块。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """用于时间序列窗口的可逆实例归一化。"""

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
    """统计量动态状态编码器，仅用于消融。"""

    def __init__(
        self,
        n_features,
        emb_dim=32,
        hidden_dim=None,
        stat_features=None,
        control_indices=None,
    ):
        super(WindowRegimeEncoder, self).__init__()
        if stat_features is None:
            stat_features = ["mean", "std", "last", "delta"]
        valid_indices = [
            int(index) for index in (control_indices or range(n_features))
            if 0 <= int(index) < n_features
        ]
        if not valid_indices:
            valid_indices = list(range(n_features))
        self.register_buffer("control_indices", torch.tensor(valid_indices, dtype=torch.long))
        self.n_features = len(valid_indices)
        self.stat_features = list(stat_features)
        input_dim = len(self.stat_features) * self.n_features
        hidden_dim = hidden_dim or max(emb_dim, min(128, input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        x = torch.index_select(x, dim=2, index=self.control_indices)
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


class TemporalRegimeEncoder(nn.Module):
    """从选定的控制量和状态量通道中学习连续动态工况嵌入。"""

    def __init__(
        self,
        n_features,
        emb_dim=32,
        hidden_dim=48,
        control_indices=None,
        auxiliary_dim=4,
    ):
        super(TemporalRegimeEncoder, self).__init__()
        valid_indices = [
            int(index) for index in (control_indices or range(n_features))
            if 0 <= int(index) < n_features
        ]
        if not valid_indices:
            valid_indices = list(range(n_features))
        self.register_buffer("control_indices", torch.tensor(valid_indices, dtype=torch.long))
        input_dim = len(valid_indices)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.auxiliary_head = nn.Linear(emb_dim, auxiliary_dim)

    def forward(self, x, return_auxiliary=False):
        controls = torch.index_select(x, dim=2, index=self.control_indices)
        hidden = self.temporal_encoder(controls.transpose(1, 2))
        attention = torch.softmax(self.attention(hidden), dim=2)
        pooled = torch.sum(hidden * attention, dim=2)
        embedding = self.embedding_head(torch.cat([pooled, hidden[:, :, -1]], dim=1))
        if return_auxiliary:
            return embedding, self.auxiliary_head(embedding)
        return embedding


class ControlConditionedGraphBias(nn.Module):
    """仅根据控制量轨迹，在少量图结构专家之间进行软路由。
    它与 GATv2 的作用不同：GATv2 根据节点特征改变成对注意力，本模块则把电流、
    SOC 等外部工况显式映射为图结构先验。
    """

    def __init__(
        self, n_features, emb_dim=32, expert_count=3, control_indices=None,
        router_mode="learned", router_temperature=1.0,
    ):
        super().__init__()
        self.router_mode = str(router_mode)
        self.router_temperature = max(float(router_temperature), 1e-4)
        valid_indices = [
            int(index) for index in (control_indices or [])
            if 0 <= int(index) < n_features
        ]
        self.register_buffer("graph_control_indices", torch.tensor(valid_indices, dtype=torch.long))
        self.condition_encoder = TemporalRegimeEncoder(
            n_features, emb_dim=emb_dim, control_indices=control_indices
        )
        if self.router_mode == "control_quadrant":
            if len(valid_indices) < 2:
                raise ValueError("control_quadrant router requires current and SOC control indices")
            expert_count = 4
            self.router = None
        elif self.router_mode == "learned":
            self.router = nn.Linear(emb_dim, max(1, int(expert_count)))
        else:
            raise ValueError(f"Unsupported condition graph router: {self.router_mode}")
        self.expert_bias = nn.Parameter(
            torch.zeros(max(1, int(expert_count)), n_features, n_features)
        )

    def forward(self, x, return_routing=False):
        if self.router_mode == "control_quadrant":
            controls = torch.index_select(x, dim=2, index=self.graph_control_indices[:2])
            current_probability = torch.sigmoid(
                controls[:, :, 0].mean(dim=1) / self.router_temperature
            )
            soc_probability = torch.sigmoid(
                controls[:, :, 1].mean(dim=1) / self.router_temperature
            )
            routing = torch.stack([
                (1.0 - current_probability) * (1.0 - soc_probability),
                current_probability * (1.0 - soc_probability),
                (1.0 - current_probability) * soc_probability,
                current_probability * soc_probability,
            ], dim=1)
        else:
            condition = self.condition_encoder(x)
            routing = torch.softmax(self.router(condition), dim=-1)
        bias = torch.einsum("be,eij->bij", routing, self.expert_bias)
        if return_routing:
            return bias, routing
        return bias


class ControlRoutedLowRankAdapter(nn.Module):
    """用电流/SOC软路由低秩隐状态专家，不直接生成任何物理响应。"""

    def __init__(
        self, hidden_dim, n_features, rank=16, expert_count=4,
        control_indices=None, temperature=1.0,
    ):
        super().__init__()
        valid_indices = [
            int(index) for index in (control_indices or [])
            if 0 <= int(index) < n_features
        ]
        if len(valid_indices) < 2:
            raise ValueError("condition-routed adapter requires current and SOC indices")
        self.register_buffer("control_indices", torch.tensor(valid_indices[:2], dtype=torch.long))
        self.temperature = max(float(temperature), 1e-4)
        self.expert_count = max(4, int(expert_count))
        self.rank = max(1, min(int(rank), int(hidden_dim)))
        self.down = nn.Parameter(torch.empty(self.expert_count, hidden_dim, self.rank))
        self.up = nn.Parameter(torch.zeros(self.expert_count, self.rank, hidden_dim))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def _routing(self, x):
        controls = torch.index_select(x, dim=2, index=self.control_indices)
        current_probability = torch.sigmoid(controls[:, :, 0].mean(dim=1) / self.temperature)
        soc_probability = torch.sigmoid(controls[:, :, 1].mean(dim=1) / self.temperature)
        base = torch.stack([
            (1.0 - current_probability) * (1.0 - soc_probability),
            current_probability * (1.0 - soc_probability),
            (1.0 - current_probability) * soc_probability,
            current_probability * soc_probability,
        ], dim=1)
        if self.expert_count == 4:
            return base
        padding = base.new_zeros(base.size(0), self.expert_count - 4)
        return torch.cat([base, padding], dim=1)

    def forward(self, hidden, x, return_routing=False):
        routing = self._routing(x)
        low_rank = torch.nn.functional.gelu(torch.einsum("bh,ehr->ber", hidden, self.down))
        expert_delta = torch.einsum("ber,erh->beh", low_rank, self.up)
        adapted = hidden + torch.sum(routing.unsqueeze(-1) * expert_delta, dim=1)
        if return_routing:
            return adapted, routing
        return adapted


class FiLMConditioner(nn.Module):
    """根据动态状态嵌入生成按特征调制的仿射参数。"""

    def __init__(self, emb_dim, target_dim):
        super(FiLMConditioner, self).__init__()
        self.proj = nn.Linear(emb_dim, target_dim * 2)
        self.target_dim = target_dim
        # 工况条件化只用于增强共享 MTAD-GAT 表示，因此从严格恒等映射开始；
        # 在条件器尚未学到数据规律前，不向骨干注入任意的工况相关形变。
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, regime_embedding):
        gamma, beta = torch.chunk(self.proj(regime_embedding), 2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return gamma, beta


class RegimeResidualGate(nn.Module):
    """为 Transformer 残差增强生成动态状态条件门控。"""

    def __init__(self, emb_dim, target_dim):
        super(RegimeResidualGate, self).__init__()
        self.proj = nn.Linear(emb_dim, target_dim)
        # 从与工况无关的中性 0.5 门控开始；结合零初始化的 Transformer 输出投影，
        # 初始预测严格保持 MTAD-GAT 原值，随后再由训练形成动态门控。
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, regime_embedding):
        return torch.sigmoid(self.proj(regime_embedding))


class ConvLayer(nn.Module):
    """一维卷积层，用于提取每个时间序列输入的高层特征。
    :param n_features: 输入特征数/nodes
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
        return x.permute(0, 2, 1)  # 还原维度顺序


class ResidualMultiScaleConvLayer(nn.Module):
    """严格保留原始 Conv7 的多尺度时序增强。

    主路完全复用 :class:`ConvLayer`；新增的深度因果膨胀卷积经 1x1
    融合后，通过零初始化门控加回主路。因此初始前向与原 MTAD-GAT
    逐元素一致；构造可选分支后恢复 RNG，避免改变后续 GAT/GRU
    的随机初始化。
    """

    def __init__(self, n_features, kernel_size=7, dilations=(4, 16, 32)):
        super().__init__()
        dilations = [int(value) for value in dilations]
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError("Residual multi-scale dilations must be positive integers")
        self.baseline = ConvLayer(n_features, kernel_size)
        self.dilations = dilations
        branch_rng_state = torch.get_rng_state()
        self.branches = nn.ModuleList([
            nn.Conv1d(
                n_features,
                n_features,
                kernel_size=3,
                dilation=dilation,
                groups=n_features,
                bias=False,
            )
            for dilation in dilations
        ])
        self.fusion = nn.Conv1d(n_features * len(dilations), n_features, kernel_size=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, n_features, 1))
        torch.set_rng_state(branch_rng_state)

    def forward(self, x):
        baseline = self.baseline(x)
        x_channels = x.permute(0, 2, 1)
        scale_features = []
        for dilation, branch in zip(self.dilations, self.branches):
            padding = 2 * dilation
            scale_features.append(F.gelu(branch(F.pad(x_channels, (padding, 0)))))
        residual = self.fusion(torch.cat(scale_features, dim=1))
        residual = torch.tanh(self.gate) * residual
        return baseline + residual.permute(0, 2, 1)


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
            # 填充参数设为 0，在前向传播中手动计算并执行因果填充
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
        # 渐进式融合模块（仅在渐进式模式下使用）
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
        x：形状为（批大小，窗口长度，特征数）
        """
        x_perm = x.permute(0, 2, 1)  # 形状：（批大小，特征数，窗口长度）
        # 提取各尺度特征（采用因果膨胀卷积）
        scale_feats = []
        for i, conv in enumerate(self.scale_convs):
            dilation = self.dilations[i]
            # 【关键】因果膨胀卷积的填充公式：填充大小 = (卷积核大小 - 1) * 膨胀率
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
        else:  # 渐进式模式
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
    """单图特征/空间注意力层。
    :param n_features: 输入特征数/nodes
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
        output_activation="sigmoid",
        learnable_sparse_graph=False,
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
        self.output_activation = str(output_activation).lower()
        self.learnable_sparse_graph = bool(learnable_sparse_graph)
        if self.output_activation not in {"sigmoid", "elu", "tanh", "identity"}:
            raise ValueError(f"Unsupported GAT output activation: {output_activation}")
        # GATv2 会先拼接节点对再做线性变换
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

    def forward(self, x, attention_bias=None):
        # x 形状为 (b, n, k)：b 为批大小，n 为窗口大小，k 为特征数
        # 在特征维度上建图，每个特征视为一个节点
        x = x.permute(0, 2, 1)
        # GATv2 分支
        # 由 Brody 等人在 2021 年提出（https://arxiv.org/pdf/2105.14491.pdf）
        # 先拼接再线性变换，并在带泄露修正线性单元后计算注意力分数
        if self.use_gatv2:
            # 把每个节点与其他所有节点拼接起来
            a_input = self._make_attention_input(x)  # (b, k, k, 2*window_size)
            # 经过一个线性层和带泄露修正线性单元，得到中间表示
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            # 使用可学习参数 self.a 做点积，得到原始注意力分数 e
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)
        # 标准 GAT 分支
        else:
            #对特征做线性变换，获取更高级的表示
            Wx = self.lin(x)  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)
        if self.use_bias:
            e += self.bias
        if attention_bias is not None:
            if attention_bias.shape != e.shape:
                raise ValueError(
                    f"Feature attention bias shape {tuple(attention_bias.shape)} "
                    f"does not match logits {tuple(e.shape)}"
                )
            e = e + attention_bias.to(dtype=e.dtype)
        if self.attention_sparse:
            # 待办：创新点1-随机稀疏化代替物理拓扑稀疏化，稀疏化注意力权重
            # 只保留每个节点最重要的前 k 个连接
            top_k = min(self.attention_top_k, self.n_features)  # 保留每个节点的前 k 个连接
            # e 是注意力分数张量，形状为 (batch_size, n_features, n_features)
            topk_values, topk_indices = torch.topk(e, top_k, dim=2)
            sparse_e = torch.full_like(e, float('-inf'))
            sparse_e.scatter_(2, topk_indices, topk_values)
            e = sparse_e
        # 注意力权重，在特征维度上做归一化指数计算
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        self.last_attention = attention
        # 根据注意力权重聚合特征表示
        h = self._activate_output(torch.matmul(attention, x))
        return h.permute(0, 2, 1)

    def sparse_graph_regularization(self):
        """返回非对角边的软连接密度。

        Feature-GAT 的样本动态注意力保留不变；这个全局边 logit
        只学习哪些连接应长期被抑制。基线原本就有零初始化的全局
        bias，因而开启本项不改变初始前向，只增加稀疏监督。
        """
        if not self.learnable_sparse_graph or not self.use_bias or self.n_features <= 1:
            reference = self.a if hasattr(self, "a") else next(self.parameters())
            return reference.new_tensor(0.0)
        off_diagonal = ~torch.eye(
            self.n_features, dtype=torch.bool, device=self.bias.device
        )
        return torch.sigmoid(self.bias[off_diagonal]).mean()

    def _activate_output(self, value):
        """对聚合结果应用可配置激活；默认值严格保持原实现。"""
        if self.output_activation == "sigmoid":
            return self.sigmoid(value)
        if self.output_activation == "elu":
            return F.elu(value)
        if self.output_activation == "tanh":
            return torch.tanh(value)
        return value

    def _make_attention_input(self, v):
        """构造特征注意力机制的输入。
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
        blocks_repeating = v.repeat_interleave(K, dim=1)  # 重复源节点表示
        blocks_alternating = v.repeat(1, K, 1)  # 交替目标节点表示
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)
        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """单图时间注意力层。
    :param n_features: number of input features/nodes
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
        output_activation="sigmoid",
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.output_activation = str(output_activation).lower()
        if self.output_activation not in {"sigmoid", "elu", "tanh", "identity"}:
            raise ValueError(f"Unsupported GAT output activation: {output_activation}")
        # GATv2 会先拼接节点对再做线性变换
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
        # x 形状为 (b, n, k)：b 为批大小，n 为窗口大小，k 为特征数
        # 在时间维度上建图，每个时间步视为一个节点
        # GATv2 分支
        # 由 Brody 等人在 2021 年提出（https://arxiv.org/pdf/2105.14491.pdf）
        # 先拼接再线性变换，并在带泄露修正线性单元后计算注意力分数
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)
        # 标准 GAT 分支
        else:
            Wx = self.lin(x)  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, n, n, 1)
        if self.use_bias:
            e += self.bias  # (b, n, n, 1)
        # 注意力权重
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self._activate_output(torch.matmul(attention, x))  # 形状：(b, n, k)
        return h

    def _activate_output(self, value):
        """对聚合结果应用可配置激活；默认值严格保持原实现。"""
        if self.output_activation == "sigmoid":
            return self.sigmoid(value)
        if self.output_activation == "elu":
            return F.elu(value)
        if self.output_activation == "tanh":
            return torch.tanh(value)
        return value

    def _make_attention_input(self, v):
        """构造时间注意力机制的输入。
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2
            ...
            ...
            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2
        """
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # 重复源节点表示
        blocks_alternating = v.repeat(1, K, 1)  # 交替目标节点表示
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
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # 奇数位置使用余弦编码
        pe = pe.unsqueeze(0)  # 形状：[1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x：张量，形状为 [batch_size, seq_len, d_model]。
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PhysicalResponseFeatureEncoding(nn.Module):
    """提取无量纲电学/热学响应描述量并投影到模型隐空间。
    输入窗口可能已经做过逐通道归一化。若提供训练折归一化统计量，先恢复原始工程
    单位，保证电压离散度和温度离散度仍具有物理意义。
    """

    def __init__(self, output_dim, hidden_dim=32, config=None, eps=1e-6, zero_output=False):
        super().__init__()
        self.config = dict(config or {})
        self.eps = float(eps)
        self.current_index = self.config.get("current_index")
        self.voltage_index = self.config.get("voltage_index")
        self.soc_index = self.config.get("soc_index")
        self.voltage_max_index = self.config.get("voltage_max_index")
        self.voltage_min_index = self.config.get("voltage_min_index")
        self.temperature_max_index = self.config.get("temperature_max_index")
        self.temperature_min_index = self.config.get("temperature_min_index")
        data_min = self.config.get("data_min")
        data_scale = self.config.get("data_scale")
        if data_min is not None and data_scale is not None:
            self.register_buffer("data_min", torch.as_tensor(data_min).view(1, 1, -1))
            self.register_buffer("data_scale", torch.as_tensor(data_scale).view(1, 1, -1))
        else:
            self.data_min = None
            self.data_scale = None
        self.proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if zero_output:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def _raw(self, x):
        if self.data_min is None or self.data_scale is None:
            return x
        return x * self.data_scale.to(dtype=x.dtype) + self.data_min.to(dtype=x.dtype)

    def _channel(self, x, index):
        if index is None or index < 0 or index >= x.size(2):
            return x.new_zeros(x.size(0), x.size(1), 1)
        return x[:, :, index:index + 1]

    def _signed_scale(self, value):
        centered = value - value.mean(dim=1, keepdim=True)
        scale = centered.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.tanh(centered / scale)

    def _rate(self, value):
        delta = torch.diff(value, dim=1, prepend=value[:, :1, :])
        return self._signed_scale(delta)

    def _relative_positive(self, value):
        scale = value.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.clamp(value / scale, -5.0, 5.0)

    def forward(self, x):
        raw = self._raw(x)
        voltage = self._channel(raw, self.voltage_index)
        current = self._channel(raw, self.current_index)
        soc = self._channel(raw, self.soc_index)
        voltage_spread = self._channel(raw, self.voltage_max_index) - self._channel(
            raw, self.voltage_min_index
        )
        temperature_max = self._channel(raw, self.temperature_max_index)
        temperature_min = self._channel(raw, self.temperature_min_index)
        temperature_mean = 0.5 * (temperature_max + temperature_min)
        temperature_spread = temperature_max - temperature_min
        charge_flow = torch.cumsum(current, dim=1)
        charge_flow = charge_flow / charge_flow.abs().amax(dim=1, keepdim=True).clamp_min(self.eps)
        soc_rate = torch.diff(soc, dim=1, prepend=soc[:, :1, :])
        current_scaled = current / current.abs().mean(dim=1, keepdim=True).clamp_min(self.eps)
        soc_current = self._signed_scale(soc_rate) - torch.tanh(current_scaled)
        descriptors = torch.cat(
            [
                self._rate(voltage),
                self._rate(temperature_mean),
                self._relative_positive(voltage_spread),
                self._relative_positive(temperature_spread),
                charge_flow,
                soc_current,
            ],
            dim=2,
        )
        return self.proj(descriptors)


class PhysicalFeatureAttentionBias(nn.Module):
    """根据每个窗口的物理描述量，为特征图注意力边生成有界偏置。"""

    def __init__(self, n_features, hidden_dim=32, config=None):
        super().__init__()
        self.n_features = int(n_features)
        self.feature_encoder = PhysicalResponseFeatureEncoding(
            hidden_dim, hidden_dim=hidden_dim, config=config,
        )
        self.bias_head = nn.Linear(3 * hidden_dim, self.n_features * self.n_features)
        # 输出层从零开始，使初始化时的计算严格等于 C3。
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, x):
        features = self.feature_encoder(x)
        summary = torch.cat(
            [
                features.mean(dim=1),
                features.std(dim=1, unbiased=False),
                features[:, -1, :],
            ],
            dim=1,
        )
        bias = torch.tanh(self.bias_head(summary))
        return bias.view(x.size(0), self.n_features, self.n_features)


class ControlResponseDecoderConditioner(nn.Module):
    """仅使用因果控制量修正响应预测和响应重构的条件器。"""

    def __init__(self, out_dim, hidden_dim=32, config=None, response_dims=None):
        super().__init__()
        config = dict(config or {})
        self.current_index = config.get("current_index")
        self.soc_index = config.get("soc_index")
        self.out_dim = int(out_dim)
        if response_dims is None:
            response_dims = list(range(self.out_dim))
        self.response_dims = tuple(int(index) for index in response_dims)
        selector = torch.zeros(len(self.response_dims), self.out_dim)
        for source_index, output_index in enumerate(self.response_dims):
            selector[source_index, output_index] = 1.0
        self.register_buffer("response_selector", selector)
        self.encoder = nn.GRU(5, int(hidden_dim), batch_first=True)
        response_dim = len(self.response_dims)
        self.forecast_correction = nn.Linear(int(hidden_dim), response_dim)
        self.reconstruction_correction = nn.Linear(int(hidden_dim), response_dim)
        self.response_probe = nn.Linear(int(hidden_dim), response_dim)
        # 修正层从零开始，使初始化时的解码结果与 C3 完全一致。
        nn.init.zeros_(self.forecast_correction.weight)
        nn.init.zeros_(self.forecast_correction.bias)
        nn.init.zeros_(self.reconstruction_correction.weight)
        nn.init.zeros_(self.reconstruction_correction.bias)

    @staticmethod
    def _channel(x, index):
        if index is None or index < 0 or index >= x.size(2):
            return x.new_zeros(x.size(0), x.size(1), 1)
        return x[:, :, index:index + 1]

    def forward(self, x):
        current = self._channel(x, self.current_index)
        soc = self._channel(x, self.soc_index)
        current_delta = torch.diff(current, dim=1, prepend=current[:, :1, :])
        soc_delta = torch.diff(soc, dim=1, prepend=soc[:, :1, :])
        controls = torch.cat(
            [current, soc, current.square(), current_delta, soc_delta], dim=2
        )
        hidden, _ = self.encoder(controls)
        forecast = self.forecast_correction(hidden[:, -1, :]) @ self.response_selector
        reconstruction = self.reconstruction_correction(hidden) @ self.response_selector
        return forecast, reconstruction, self.response_probe(hidden)


class ControlConditionedResponseVAE(nn.Module):
    """带信息瓶颈的独立正常控制—响应模型。
    编码器把完整观测窗口压缩为低维潜在状态。响应解码器不能直接读取电压或温度
    轨迹；它在每个时间步只接收由电流和 SOC 构造的控制量，潜在状态只通过初始隐
    状态注入一次。因此该分支的残差是独立物理一致性信号，而不是对 MTAD-GAT
    输出的直接修正。
    """

    def __init__(
        self,
        n_features,
        response_dims,
        hidden_dim=64,
        latent_dim=16,
        config=None,
    ):
        super().__init__()
        config = dict(config or {})
        # C4 一致性头使用的控制通道可以与通用物理状态通道不同。对 BMS 而言，
        # SYS_I 是系统级外加控制量，BMSnI 是必须保留用于评分的簇级响应量。
        self.current_index = config.get("consistency_current_index", config.get("current_index"))
        self.soc_index = config.get("consistency_soc_index", config.get("soc_index"))
        self.response_dims = tuple(int(index) for index in response_dims)
        self.state_encoder = nn.GRU(
            int(n_features), int(hidden_dim), batch_first=True, bidirectional=True
        )
        self.mean_head = nn.Linear(2 * int(hidden_dim), int(latent_dim))
        self.logvar_head = nn.Linear(2 * int(hidden_dim), int(latent_dim))
        self.latent_to_hidden = nn.Linear(int(latent_dim), int(hidden_dim))
        self.response_decoder = nn.GRU(5, int(hidden_dim), batch_first=True)
        self.response_head = nn.Linear(int(hidden_dim), len(self.response_dims))

    @staticmethod
    def _channel(x, index):
        if index is None or index < 0 or index >= x.size(2):
            return x.new_zeros(x.size(0), x.size(1), 1)
        return x[:, :, index:index + 1]

    def forward(self, x):
        _, hidden = self.state_encoder(x)
        state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        mean = self.mean_head(state)
        logvar = self.logvar_head(state).clamp(-8.0, 8.0)
        # 训练和推理都使用后验均值。KL 项仍约束信息瓶颈，而确定性解码可避免该独立
        # 分支改变后续批次中 MTAD-GAT 随机失活所使用的随机数轨迹。
        latent = mean
        current = self._channel(x, self.current_index)
        soc = self._channel(x, self.soc_index)
        current_delta = torch.diff(current, dim=1, prepend=current[:, :1, :])
        soc_delta = torch.diff(soc, dim=1, prepend=soc[:, :1, :])
        controls = torch.cat(
            [current, soc, current.square(), current_delta, soc_delta], dim=2
        )
        decoder_hidden = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        decoded, _ = self.response_decoder(controls, decoder_hidden)
        return self.response_head(decoded), mean, logvar


class PhysicalStateEncoding(nn.Module):
    """将面向电池的状态特征投影到 Transformer 隐空间。"""

    def __init__(self, d_model, hidden_dim=32, config=None, eps=1e-6, feature_mode="full"):
        super(PhysicalStateEncoding, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.config = dict(config or {})
        self.eps = eps
        self.feature_mode = feature_mode
        if self.feature_mode not in {"full", "controls_only"}:
            raise ValueError(f"Unsupported physical-state feature mode: {self.feature_mode}")
        self.current_index = self.config.get("current_index")
        self.voltage_index = self.config.get("voltage_index")
        self.temperature_index = self.config.get("temperature_index")
        self.step_type_index = self.config.get("step_type_index")
        self.soc_index = self.config.get("soc_index")
        self.soh_index = self.config.get("soh_index")
        self.state_dim = 5 if self.feature_mode == "full" else 3
        if self.soc_index is not None:
            self.state_dim += 1
        if self.feature_mode == "full" and self.soh_index is not None:
            self.state_dim += 1
        self.proj = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        # x 形状：（批大小，窗口长度，特征数）
        current = self._get_channel(x, self.current_index)
        phase = self._compute_phase(x, current)
        current_norm = self._signed_normalize(current)
        charge_flow = self._compute_charge_flow(current)
        state_parts = [phase, current_norm, charge_flow]
        if self.feature_mode == "full":
            voltage_rel = self._compute_relative_position(self._get_channel(x, self.voltage_index))
            temperature_rel = self._compute_relative_position(self._get_channel(x, self.temperature_index))
            state_parts.extend([voltage_rel, temperature_rel])
        if self.soc_index is not None:
            state_parts.append(self._compute_absolute_state(self._get_channel(x, self.soc_index)))
        if self.feature_mode == "full" and self.soh_index is not None:
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
    """门控循环单元（GRU）层。
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
        out, h = out[-1, :, :], h[-1, :, :]  # 取最后一层隐藏状态
        return out, h


class RNNDecoder(nn.Module):
    """基于 GRU 的解码器网络，用于将潜在向量转换为输出。
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
    """重构模型。
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
        # x 是 GRU 编码后的隐藏表示
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class VariationalReconstructionModel(nn.Module):
    """以共享 MTAD-GAT 状态为条件的窗口变分重构解码器。"""

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout, latent_dim=32):
        super().__init__()
        latent_dim = max(2, int(latent_dim))
        self.window_size = window_size
        self.to_mean = nn.Linear(in_dim, latent_dim)
        self.to_logvar = nn.Linear(in_dim, latent_dim)
        self.decoder = RNNDecoder(latent_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, state):
        mean = self.to_mean(state)
        logvar = self.to_logvar(state).clamp(-10.0, 8.0)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            latent = mean
        repeated = latent.repeat_interleave(self.window_size, dim=1).view(
            state.size(0), self.window_size, -1
        )
        return self.fc(self.decoder(repeated)), mean, logvar


class Forecasting_Model(nn.Module):
    """预测模型（全连接网络）。
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
