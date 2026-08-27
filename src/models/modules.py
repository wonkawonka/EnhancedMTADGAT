"""定义 MTAD-GAT 变体可复用的神经网络构建模块。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RestrictedStateEncoder(nn.Module):
    """用边际窗口描述量构造低维状态，不编码任意通道对关系。

    电池数据保留电流/SOC 的有序语义；匿名遥测数据对每个非目标上下文通道
    使用同一个 MLP，再以 mean/max 做置换不变池化。两种模式都只读取每个通道
    自身的均值、标准差、首尾变化和平均绝对一阶差分。
    """

    def __init__(
        self,
        n_features,
        emb_dim=8,
        hidden_dim=24,
        control_indices=None,
        pooled_channels=False,
    ):
        super().__init__()
        valid_indices = [
            int(index) for index in (control_indices or range(n_features))
            if 0 <= int(index) < n_features
        ]
        if not valid_indices:
            valid_indices = list(range(n_features))
        self.register_buffer(
            "control_indices", torch.tensor(valid_indices, dtype=torch.long)
        )
        self.pooled_channels = bool(pooled_channels)
        self.descriptor_dim = 4
        hidden_dim = max(int(hidden_dim), int(emb_dim))
        if self.pooled_channels:
            self.channel_encoder = nn.Sequential(
                nn.Linear(self.descriptor_dim, hidden_dim),
                nn.GELU(),
            )
            encoder_input_dim = 2 * hidden_dim
            auxiliary_dim = 2 * self.descriptor_dim
        else:
            encoder_input_dim = len(valid_indices) * self.descriptor_dim
            auxiliary_dim = encoder_input_dim
        self.embedding_head = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.auxiliary_head = nn.Linear(emb_dim, auxiliary_dim)

    @staticmethod
    def _marginal_descriptors(channels):
        delta = channels[:, -1, :] - channels[:, 0, :]
        if channels.size(1) > 1:
            mean_abs_diff = (channels[:, 1:, :] - channels[:, :-1, :]).abs().mean(dim=1)
        else:
            mean_abs_diff = torch.zeros_like(delta)
        return torch.stack(
            (
                channels.mean(dim=1),
                channels.std(dim=1, unbiased=False),
                delta,
                mean_abs_diff,
            ),
            dim=-1,
        )

    def descriptor_target(self, x):
        channels = torch.index_select(x, dim=2, index=self.control_indices)
        descriptors = self._marginal_descriptors(channels)
        if self.pooled_channels:
            return torch.cat(
                (descriptors.mean(dim=1), descriptors.amax(dim=1)), dim=1
            )
        return descriptors.flatten(start_dim=1)

    def forward(self, x, return_auxiliary=False):
        channels = torch.index_select(x, dim=2, index=self.control_indices)
        descriptors = self._marginal_descriptors(channels)
        if self.pooled_channels:
            channel_hidden = self.channel_encoder(descriptors)
            encoder_input = torch.cat(
                (channel_hidden.mean(dim=1), channel_hidden.amax(dim=1)), dim=1
            )
            target = torch.cat(
                (descriptors.mean(dim=1), descriptors.amax(dim=1)), dim=1
            )
        else:
            encoder_input = descriptors.flatten(start_dim=1)
            target = encoder_input
        embedding = self.embedding_head(encoder_input)
        if return_auxiliary:
            return embedding, self.auxiliary_head(embedding), target.detach()
        return embedding


class PrototypeQueryRegimeEncoder(nn.Module):
    """用统计 Token、可学习 Query 和软路由构造连续工况状态。"""

    def __init__(
        self,
        n_features,
        emb_dim=8,
        model_dim=32,
        num_prototypes=6,
        num_heads=4,
        temperature=0.5,
        control_indices=None,
    ):
        super().__init__()
        valid_indices = [
            int(index) for index in (control_indices or range(n_features))
            if 0 <= int(index) < n_features
        ]
        if not valid_indices:
            valid_indices = list(range(n_features))
        model_dim = int(model_dim)
        num_prototypes = int(num_prototypes)
        num_heads = int(num_heads)
        if model_dim <= 0 or num_prototypes <= 1:
            raise ValueError("Prototype-query C3 requires model_dim > 0 and K > 1")
        if model_dim % num_heads != 0:
            raise ValueError("regime_query_dim must be divisible by regime_query_heads")
        if num_prototypes > model_dim:
            raise ValueError("Orthogonal prototype initialization requires K <= model_dim")

        self.register_buffer(
            "control_indices", torch.tensor(valid_indices, dtype=torch.long)
        )
        self.descriptor_dim = 4
        self.model_dim = model_dim
        self.num_prototypes = num_prototypes
        self.auxiliary_dim = len(valid_indices) * self.descriptor_dim

        self.channel_encoder = nn.Sequential(
            nn.Linear(self.descriptor_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.channel_embeddings = nn.Parameter(
            torch.empty(len(valid_indices), model_dim)
        )
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, model_dim))
        nn.init.orthogonal_(self.prototypes)
        self.cross_attention = nn.MultiheadAttention(
            model_dim, num_heads, batch_first=True
        )
        self.prototype_norm = nn.LayerNorm(model_dim)
        self.embedding_head = nn.Linear(
            num_prototypes * model_dim,
            emb_dim
        )
        self.auxiliary_head = nn.Linear(int(emb_dim), self.auxiliary_dim)

        self.last_cross_attention = None

    @staticmethod
    def _marginal_descriptors(channels):
        return RestrictedStateEncoder._marginal_descriptors(channels)

    def descriptor_target(self, x):
        channels = torch.index_select(x, dim=2, index=self.control_indices)
        return self._marginal_descriptors(channels).flatten(start_dim=1)

    def forward(self, x, return_auxiliary=False):
        channels = torch.index_select(x, dim=2, index=self.control_indices)
        descriptors = self._marginal_descriptors(channels)
        tokens = self.channel_encoder(descriptors)
        tokens = tokens + self.channel_embeddings.unsqueeze(0)

        queries = self.prototypes.unsqueeze(0).expand(x.size(0), -1, -1)
        prototype_outputs, attention = self.cross_attention(
            queries, tokens, tokens,
            need_weights=True,
            average_attn_weights=False
        )

        prototype_outputs = self.prototype_norm(
            prototype_outputs + queries
        )

        # [B, K, model_dim] -> [B, K * model_dim]
        prototype_state = prototype_outputs.flatten(start_dim=1)

        # [B, K * model_dim] -> [B, emb_dim]
        embedding = self.embedding_head(prototype_state)

        self.last_cross_attention = attention

        if return_auxiliary:
            target = descriptors.flatten(start_dim=1)
            return embedding, self.auxiliary_head(embedding), target.detach()
        return embedding

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


class DynamicPhysicalGraphBias(nn.Module):
    """Zero-centred, state-conditioned relation-type attention modulation.

    Four relation types (load--thermal, voltage-extrema, thermal-extrema and
    SOC--voltage) are gated from the current window.  All gate parameters are
    zero-initialised, therefore this module returns an exact zero matrix before
    learning and the initial forward pass is identical to MTAD-GAT.  The final
    modulation is bounded and may be positive *or* negative.
    """

    _ROLE_NAMES = (
        "voltage",
        "current",
        "soc",
        "voltage_max",
        "voltage_min",
        "temperature_max",
        "temperature_min",
    )

    def __init__(self, n_features, config=None, dynamic_weight=1.0, gate_scale=5.0):
        super().__init__()
        config = dict(config or {})
        self.n_features = int(n_features)
        self.dynamic_weight = max(0.0, float(dynamic_weight))
        self.gate_scale = max(0.0, float(gate_scale))

        roles = {
            name: self._valid_indices(config.get("physical_graph_roles", {}).get(name, []))
            for name in self._ROLE_NAMES
        }
        if not all(roles[name] for name in self._ROLE_NAMES):
            missing = [name for name in self._ROLE_NAMES if not roles[name]]
            raise ValueError(f"Physical graph roles are incomplete: {missing}")
        self.roles = roles
        graph_roles = config.get("physical_graph_roles", {})
        self.voltage_spread_indices = self._valid_indices(
            graph_roles.get("voltage_spread", [])
        )
        self.temperature_spread_indices = self._valid_indices(
            graph_roles.get("temperature_spread", [])
        )

        current_gate = torch.zeros(self.n_features, self.n_features)
        voltage_gate = torch.zeros_like(current_gate)
        temperature_gate = torch.zeros_like(current_gate)
        soc_gate = torch.zeros_like(current_gate)

        # I^2 strengthens load-to-thermal relations.  Voltage/temperature
        # spreads strengthen their corresponding extrema relations, while SOC
        # movement strengthens SOC-to-voltage coupling.
        self._connect(current_gate, roles["current"], roles["temperature_max"])
        self._connect(current_gate, roles["current"], roles["temperature_min"])
        self._connect(voltage_gate, roles["voltage"], roles["voltage_max"])
        self._connect(voltage_gate, roles["voltage"], roles["voltage_min"])
        self._connect(voltage_gate, roles["voltage_max"], roles["voltage_min"])
        self._connect(temperature_gate, roles["temperature_max"], roles["temperature_min"])
        self._connect(soc_gate, roles["soc"], roles["voltage"])
        self._connect(soc_gate, roles["soc"], roles["voltage_max"])
        self._connect(soc_gate, roles["soc"], roles["voltage_min"])

        self.register_buffer("current_gate_mask", current_gate)
        self.register_buffer("voltage_gate_mask", voltage_gate)
        self.register_buffer("temperature_gate_mask", temperature_gate)
        self.register_buffer("soc_gate_mask", soc_gate)
        # Four learnable relation-type gates, conditioned on the four physical
        # state strengths below.  Exact-zero initialization preserves the
        # baseline forward map before the first optimiser update.
        self.relation_gate_weight = nn.Parameter(torch.zeros(4, 4))
        self.relation_gate_bias = nn.Parameter(torch.zeros(4))
        self.max_modulation = 0.15 * self.dynamic_weight
        # Read-only inference caches for the paper visualisation pipeline.
        # They are deliberately not buffers/parameters, so they neither alter
        # the forward values nor the checkpoint/state_dict layout.
        self.last_state_strengths = None
        self.last_relation_gates = None

        offset = torch.as_tensor(config.get("physical_data_offset", []), dtype=torch.float32)
        scale = torch.as_tensor(config.get("physical_data_scale", []), dtype=torch.float32)
        if offset.numel() != self.n_features or scale.numel() != self.n_features:
            offset = torch.empty(0, dtype=torch.float32)
            scale = torch.empty(0, dtype=torch.float32)
        self.register_buffer("data_offset", offset)
        self.register_buffer("data_scale", scale)

    def _valid_indices(self, values):
        return tuple(sorted({int(value) for value in values if 0 <= int(value) < self.n_features}))

    @staticmethod
    def _connect(matrix, left, right):
        for source in left:
            for target in right:
                if source != target:
                    matrix[source, target] = 1.0
                    matrix[target, source] = 1.0

    def _engineering_values(self, x):
        if self.data_offset.numel() == self.n_features:
            return x * self.data_scale.view(1, 1, -1) + self.data_offset.view(1, 1, -1)
        return x

    def _reference_scale(self, indices, x):
        if self.data_scale.numel() == self.n_features:
            return self.data_scale[list(indices)].abs().mean().clamp_min(1e-6)
        return x.new_tensor(1.0)

    def _spread_gate(self, values, high_indices, low_indices, spread_indices=()):
        if spread_indices:
            spread = values[:, :, list(spread_indices)].abs().mean(dim=(1, 2))
            reference = self._reference_scale(spread_indices, values)
            return torch.tanh(self.gate_scale * spread / reference)
        pair_count = min(len(high_indices), len(low_indices))
        high = values[:, :, list(high_indices[:pair_count])]
        low = values[:, :, list(low_indices[:pair_count])]
        reference = 0.5 * (
            self._reference_scale(high_indices[:pair_count], values)
            + self._reference_scale(low_indices[:pair_count], values)
        )
        ratio = (high - low).abs().mean(dim=(1, 2)) / reference
        return torch.tanh(self.gate_scale * ratio)

    def forward(self, x):
        if x.ndim != 3 or x.size(2) != self.n_features:
            raise ValueError(
                f"Physical graph expects [batch, window, {self.n_features}], got {tuple(x.shape)}"
            )
        values = self._engineering_values(x)
        current_indices = self.roles["current"]
        current = values[:, :, list(current_indices)]
        current_reference = self._reference_scale(current_indices, values)
        current_energy = current.square().mean(dim=(1, 2)) / current_reference.square()
        current_strength = torch.tanh(self.gate_scale * current_energy)
        voltage_strength = self._spread_gate(
            values,
            self.roles["voltage_max"],
            self.roles["voltage_min"],
            self.voltage_spread_indices,
        )
        temperature_strength = self._spread_gate(
            values,
            self.roles["temperature_max"],
            self.roles["temperature_min"],
            self.temperature_spread_indices,
        )
        soc_indices = self.roles["soc"]
        soc = values[:, :, list(soc_indices)]
        soc_delta = torch.diff(soc, dim=1).abs().mean(dim=(1, 2))
        soc_reference = self._reference_scale(soc_indices, values)
        soc_strength = torch.tanh(self.gate_scale * soc_delta / soc_reference)

        state = torch.stack(
            (current_strength, voltage_strength, temperature_strength, soc_strength), dim=1
        )
        gates = self.max_modulation * torch.tanh(
            F.linear(state, self.relation_gate_weight, self.relation_gate_bias)
        )
        self.last_state_strengths = state.detach()
        self.last_relation_gates = gates.detach()
        return (
            gates[:, 0, None, None] * self.current_gate_mask
            + gates[:, 1, None, None] * self.voltage_gate_mask
            + gates[:, 2, None, None] * self.temperature_gate_mask
            + gates[:, 3, None, None] * self.soc_gate_mask
        )


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
        # C3 的关系转移目标必须是确定的行概率图。若使用 dropout 后的权重，
        # 同一个正常窗口也会因随机丢边产生伪关系残差。
        self.last_attention_raw = attention
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
        # 仅缓存当前批次，供无标签关系异常评分使用；不参与前向计算，
        # 因而不会改变已有 checkpoint 的参数或数值输出。
        self.last_attention = attention
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


class ControlConditionedResponseVAE(nn.Module):
    """带信息瓶颈的响应感知条件动态自编码器。

    冻结 C4 默认由单向状态编码器读取完整观测窗口，控制条件解码器逐时刻只读取
    电流/SOC 及其派生量。这样潜变量可以估计车辆个体状态，但预测路径并非
    control-only，残差应解释为条件流形重建误差。``control_only`` 和双向状态
    GRU 都只保留为显式消融。
    """

    def __init__(
        self,
        n_features,
        response_dims,
        hidden_dim=64,
        latent_dim=16,
        config=None,
        backbone_state_dim=None,
    ):
        super().__init__()
        config = dict(config or {})
        # C4 一致性头使用的控制通道可以与通用物理状态通道不同。对 BMS 而言，
        # SYS_I 是系统级外加控制量，BMSnI 是必须保留用于评分的簇级响应量。
        self.current_index = config.get("consistency_current_index", config.get("current_index"))
        self.soc_index = config.get("consistency_soc_index", config.get("soc_index"))
        self.response_dims = tuple(int(index) for index in response_dims)
        self.encoder_input = str(
            config.get("consistency_encoder_input", "full_window")
        ).strip().lower()
        if self.encoder_input not in {"full_window", "control_only"}:
            raise ValueError(
                "consistency_encoder_input must be 'full_window' or 'control_only'"
            )
        self.encoder_bidirectional = bool(
            config.get("consistency_encoder_bidirectional", False)
        )
        self.backbone_state_dim = None if backbone_state_dim is None else int(backbone_state_dim)
        if self.backbone_state_dim is None:
            encoder_features = int(n_features) if self.encoder_input == "full_window" else 2
            self.state_encoder = nn.GRU(
                encoder_features, int(hidden_dim), batch_first=True,
                bidirectional=self.encoder_bidirectional,
            )
            state_features = int(hidden_dim) * (2 if self.encoder_bidirectional else 1)
        else:
            self.state_encoder = None
            state_features = self.backbone_state_dim
        self.mean_head = nn.Linear(state_features, int(latent_dim))
        self.logvar_head = nn.Linear(state_features, int(latent_dim))
        self.latent_to_hidden = nn.Linear(int(latent_dim), int(hidden_dim))
        self.response_decoder = nn.GRU(5, int(hidden_dim), batch_first=True)
        self.response_head = nn.Linear(int(hidden_dim), len(self.response_dims))

    @staticmethod
    def _channel(x, index):
        if index is None or index < 0 or index >= x.size(2):
            return x.new_zeros(x.size(0), x.size(1), 1)
        return x[:, :, index:index + 1]

    def forward(self, x, backbone_state=None):
        current = self._channel(x, self.current_index)
        soc = self._channel(x, self.soc_index)
        control_sequence = torch.cat([current, soc], dim=2)
        if self.state_encoder is None:
            if backbone_state is None:
                raise ValueError("C4 v2 response head requires the backbone GRU state")
            state = backbone_state
        else:
            encoder_sequence = x if self.encoder_input == "full_window" else control_sequence
            _, hidden = self.state_encoder(encoder_sequence)
            state = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.encoder_bidirectional else hidden[-1]
        mean = self.mean_head(state)
        logvar = self.logvar_head(state).clamp(-8.0, 8.0)
        # 训练和推理都使用后验均值。KL 项仍约束信息瓶颈，而确定性解码可避免该独立
        # 分支改变后续批次中 MTAD-GAT 随机失活所使用的随机数轨迹。
        latent = mean
        current_delta = torch.diff(current, dim=1, prepend=current[:, :1, :])
        soc_delta = torch.diff(soc, dim=1, prepend=soc[:, :1, :])
        controls = torch.cat(
            [current, soc, current.square(), current_delta, soc_delta], dim=2
        )
        decoder_hidden = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        decoded, _ = self.response_decoder(controls, decoder_hidden)
        return self.response_head(decoded), mean, logvar


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
