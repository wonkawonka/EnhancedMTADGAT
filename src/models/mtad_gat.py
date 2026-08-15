"""实现本项目使用的 MTAD-GAT 模型及其增强变体。"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import (
    ConvLayer,
    MultiScaleConvLayer,
    ResidualMultiScaleConvLayer,
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
    ControlRoutedLowRankAdapter,
    VariationalReconstructionModel,
    RevIN,
    WindowRegimeEncoder,
    TemporalRegimeEncoder,
    RestrictedStateEncoder,
    FiLMConditioner,
    RegimeResidualGate,

)


class FeatureRelationTransitionHead(nn.Module):
    """仅依据当前 Feature-GAT 图预测下一窗口的正常变量关系图。"""

    def __init__(self, n_features, rank=4):
        super().__init__()
        self.n_features = int(n_features)
        self.rank = max(1, int(rank))  # 保留配置兼容性；转移本身不读取状态。
        self.persistence_logit = nn.Parameter(torch.zeros(()))
        self.edge_bias = nn.Parameter(torch.zeros(self.n_features, self.n_features))

    def forward(self, attention):
        persistence = torch.sigmoid(self.persistence_logit)
        logits = persistence * torch.log(attention.clamp_min(1e-7)) + self.edge_bias
        return torch.softmax(logits, dim=-1)


class JointResidualDensityHead(nn.Module):
    """在训练正常窗口上学习数值/关系残差的无条件二维联合高斯。"""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(2))
        self.log_diag = nn.Parameter(torch.zeros(2))
        self.off_diag = nn.Parameter(torch.zeros(()))

    def forward(self, batch_size, dtype, device):
        mean = self.mean.to(dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1)
        log_diag = self.log_diag.clamp(-5.0, 3.0).to(dtype=dtype, device=device)
        off_diag = self.off_diag.clamp(-5.0, 5.0).to(dtype=dtype, device=device)
        scale_tril = torch.zeros(
            batch_size, 2, 2, dtype=dtype, device=device
        )
        scale_tril[:, 0, 0] = torch.exp(log_diag[0]) + 1e-4
        scale_tril[:, 1, 0] = off_diag
        scale_tril[:, 1, 1] = torch.exp(log_diag[1]) + 1e-4
        return mean, scale_tril

    def negative_log_likelihood(self, residuals):
        mean, scale_tril = self(
            residuals.size(0), residuals.dtype, residuals.device
        )
        # CUDA 的 triangular solve 在部分 PyTorch 版本不支持 half；二维评分
        # 始终使用 float32 也可避免协方差求解的数值放大。
        mean = mean.float()
        scale_tril = scale_tril.float()
        delta = residuals.float() - mean
        solved = torch.linalg.solve_triangular(
            scale_tril, delta.unsqueeze(-1), upper=False
        ).squeeze(-1)
        mahalanobis = solved.square().sum(dim=-1)
        log_det = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        return 0.5 * mahalanobis + log_det

class Enhanced_MTADGAT(nn.Module):
    """增强型 MTAD-GAT：以预测误差和重构误差共同检测异常。
    主干按 ``卷积去噪/局部特征 → 特征GAT + 时间GAT → GRU 状态传播 → 可选
    Transformer 长程残差 → 预测头与重构头`` 工作。当前 C3 候选在原
    骨干保持原 Conv + Feature/Temporal-GAT + GRU。旧C3在评分阶段加入正常集
    校准、残差门控的关系变化正增量，但公开外推和误报复核后已撤销冻结；
    当前FiLM/关系抑制候选仍处于开发阶段。
    多尺度、稀疏图、Transformer 和旧工况模块只供否定实验复现。C4 在共享骨干旁路增加独立的
    控制量—响应量物理一致性头。
    其他物理注入和实验分支仅为旧消融复现保留，不属于正式 C4。
    所有可选分支默认关闭，因此 ``mtad_gat`` 保持原始基线含义。
    :param n_features: 输入通道数量
    :param window_size: 输入历史窗口长度
    :param out_dim: 模型输出通道数量
    :param kernel_size: 一维卷积核大小
    :param feat_gat_embed_dim: 特征图注意力内部嵌入维度
    :param time_gat_embed_dim: 时间图注意力内部嵌入维度
    :param use_gatv2: 是否使用 GATv2 注意力计算方式
    :param gru_n_layers: GRU 层数
    :param gru_hid_dim: GRU 隐藏维度
    :param forecast_n_layers: 预测头全连接层数
    :param forecast_hid_dim: 预测头隐藏维度
    :param recon_n_layers: 重构解码器层数
    :param recon_hid_dim: 重构解码器隐藏维度
    :param dropout: 随机失活比例
    :param alpha: 带泄露修正线性单元的负半轴斜率
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
            gat_output_activation="sigmoid",
            fusion_projection_dim=0,
            feature_att_trans=False,
            multi_scale_mode='basic',
            multi_scale_dilations=[1, 2, 4],
            use_learnable_sparse_graph=False,
            use_revin=False,
            revin_affine=True,
            target_dims=None,
            use_regime_condition=False,
            regime_emb_dim=32,
            regime_condition_mode="feature_gat",
            regime_stat_features=None,
            regime_encoder_type="temporal",
            regime_channel_pooling=False,
            regime_control_indices=None,
            regime_current_index=None,
            regime_soc_index=None,
            regime_film_scale=0.1,
            regime_condition_shuffle=False,
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
            condition_graph_router_mode="learned",
            condition_graph_router_temperature=1.0,
            use_condition_routed_adapter=False,
            condition_adapter_rank=16,
            condition_adapter_experts=4,
            condition_adapter_temperature=1.0,
            use_variational_reconstruction=False,
            variational_reconstruction_latent_dim=32,
            variational_reconstruction_kl_weight=0.0001,
            physical_state_config=None,
            backbone_feature_indices=None,
            condition_source_n_features=None,
            backbone_control_indices=None,
            use_c3_joint_relation=False,
            c3_relation_rank=4,
            c3_joint_hidden_dim=32,
            c3_relation_loss_weight=0.1,
            c3_joint_nll_weight=0.01,
            c3_value_gamma=1.0,
    ):
        super(Enhanced_MTADGAT, self).__init__()
        # 记录基础骨干和输出范围开关。
        self.feature_att_trans = feature_att_trans
        self.use_transformer = use_transformer
        self.fusion_projection_dim = max(0, int(fusion_projection_dim))
        self.fusion_projection = None
        self.use_revin = use_revin
        self.target_dims = target_dims
        valid_backbone_indices = [int(index) for index in (backbone_feature_indices or [])]
        self.register_buffer(
            "backbone_feature_indices", torch.tensor(valid_backbone_indices, dtype=torch.long)
        )
        self.condition_source_n_features = int(condition_source_n_features or n_features)
        # 记录 C3 工况编码、注入位置和辅助预测状态。
        self.use_regime_condition = use_regime_condition
        self.regime_condition_mode = regime_condition_mode
        self.regime_encoder_type = regime_encoder_type
        self.regime_channel_pooling = bool(regime_channel_pooling)
        self.regime_current_index = regime_current_index
        self.regime_soc_index = regime_soc_index
        self.regime_film_scale = max(0.0, min(float(regime_film_scale), 1.0))
        # 仅用于工况语义验证的负对照：保持条件边际分布，但破坏其与当前响应窗口的配对。
        self.regime_condition_shuffle = bool(regime_condition_shuffle)
        self._regime_aux_prediction = None
        self._regime_aux_target = None
        # 记录 C4 物理状态和物理响应特征分支配置。
        self.physical_state_config = dict(physical_state_config or {}) if physical_state_config is not None else None
        self.use_physical_state_encoding = use_physical_state_encoding and (feature_att_trans or use_transformer)
        self.physical_state_injection_mode = physical_state_injection_mode
        self.physical_state_feature_mode = physical_state_feature_mode
        self.use_physical_response_features = bool(use_physical_response_features)
        self.physical_feature_fusion_mode = physical_feature_fusion_mode
        self.physical_feature_hidden_dim = int(physical_feature_hidden_dim)
        # 记录可选的控制量到响应量修正分支。
        self.use_control_response_decoder = bool(use_control_response_decoder)
        self.control_response_aux_weight = max(0.0, float(control_response_aux_weight))
        self._control_response_probe = None
        # 记录正式 C4 独立物理一致性头及其训练损失权重。
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
        # 记录工况条件图和变分重构等消融分支；正式主计划不默认启用这些分支。
        self.use_control_conditioned_graph = bool(use_control_conditioned_graph)
        self._condition_graph_routing = None
        self.use_condition_routed_adapter = bool(use_condition_routed_adapter)
        self._condition_adapter_routing = None
        self._feature_attention_weights = None
        self._feature_attention_probabilities = None
        self._temporal_attention_weights = None
        self._regime_embedding = None
        self.use_c3_joint_relation = bool(use_c3_joint_relation)
        self.c3_relation_loss_weight = max(0.0, float(c3_relation_loss_weight))
        self.c3_joint_nll_weight = max(0.0, float(c3_joint_nll_weight))
        self.c3_value_gamma = max(0.0, float(c3_value_gamma))
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
            # C3：从原始窗口提取连续工况嵌入，并用 FiLM 或残差门控调制
            # 特征图注意力、时间图注意力、融合表示或 Transformer 残差中的一个位置。
            # 暂存随机数状态，避免构造可选 C3 分支时改变 MTAD-GAT 共享骨干的初始化，
            # 也避免改变训练数据随机采样器后续使用的随机数轨迹。
            regime_rng_state = torch.get_rng_state()
            if regime_encoder_type == "statistics":
                if regime_stat_features is None:
                    regime_stat_features = ["mean", "std", "last", "delta"]
                self.regime_encoder = WindowRegimeEncoder(
                    self.condition_source_n_features,
                    emb_dim=regime_emb_dim,
                    stat_features=regime_stat_features,
                    control_indices=regime_control_indices,
                )
            elif regime_encoder_type == "temporal":
                self.regime_encoder = TemporalRegimeEncoder(
                    self.condition_source_n_features,
                    emb_dim=regime_emb_dim,
                    control_indices=regime_control_indices,
                )
            elif regime_encoder_type == "restricted":
                self.regime_encoder = RestrictedStateEncoder(
                    self.condition_source_n_features,
                    emb_dim=regime_emb_dim,
                    control_indices=regime_control_indices,
                    pooled_channels=self.regime_channel_pooling,
                )
            else:
                raise ValueError(f"Unsupported regime_encoder_type: {regime_encoder_type}")
            if regime_condition_mode == "none":
                pass
            elif self.use_regime_transformer_residual:
                self.regime_residual_gate = RegimeResidualGate(regime_emb_dim, gru_hid_dim)
            elif regime_condition_mode in {"feature_gat", "feature_gat_response"}:
                self.feat_conditioner = FiLMConditioner(regime_emb_dim, n_features)
                if regime_condition_mode == "feature_gat_response":
                    response_mask = torch.ones(n_features, dtype=torch.float32)
                    for index in backbone_control_indices or []:
                        if 0 <= int(index) < n_features:
                            response_mask[int(index)] = 0.0
                    self.register_buffer("regime_response_mask", response_mask)
            elif regime_condition_mode == "temporal_gat":
                self.temp_conditioner = FiLMConditioner(regime_emb_dim, n_features)
            elif regime_condition_mode == "fusion":
                fusion_dim = 2 * n_features if feature_att_trans else 3 * n_features
                self.fusion_conditioner = FiLMConditioner(regime_emb_dim, fusion_dim)
            elif regime_condition_mode == "head":
                self.head_conditioner = FiLMConditioner(
                    regime_emb_dim, gru_hid_dim * gru_n_layers
                )
            else:
                raise ValueError(f"Unsupported regime_condition_mode: {regime_condition_mode}")
            torch.set_rng_state(regime_rng_state)
        if self.use_c3_joint_relation:
            c3_rng_state = torch.get_rng_state()
            self.c3_relation_head = FeatureRelationTransitionHead(
                n_features, rank=c3_relation_rank
            )
            self.c3_joint_score_head = JointResidualDensityHead(
                hidden_dim=c3_joint_hidden_dim
            )
            torch.set_rng_state(c3_rng_state)
        # 主干第一步：卷积提取短时局部动态；多尺度模式使用多种膨胀率扩大感受野。
        if multi_scale_mode == 'residual':
            self.conv = ResidualMultiScaleConvLayer(
                n_features, kernel_size, multi_scale_dilations
            )
        elif multi_scale_mode in ['basic', 'progressive']:
            self.conv = MultiScaleConvLayer(n_features, multi_scale_dilations, multi_scale_mode)
        else:
            self.conv = ConvLayer(n_features, kernel_size)
        # 主干第二步：feature-GAT 学习“不同传感器通道之间”的依赖关系；
        # temporal-GAT（在后文构造）学习“窗口内不同时刻”的依赖关系。
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
            output_activation=gat_output_activation,
            learnable_sparse_graph=use_learnable_sparse_graph,
        )
        if self.use_control_conditioned_graph:
            # 所有图偏置专家从零开始，使初始图严格等于 MTAD-GAT 原始图；
            # 只有训练数据确实支持时，模型才会学出工况特异的边。
            graph_rng_state = torch.get_rng_state()
            self.condition_graph = ControlConditionedGraphBias(
                n_features,
                emb_dim=condition_graph_emb_dim,
                expert_count=condition_graph_experts,
                control_indices=condition_graph_control_indices,
                router_mode=condition_graph_router_mode,
                router_temperature=condition_graph_router_temperature,
            )
            torch.set_rng_state(graph_rng_state)
        # 非简化模式下启用时间注意力层
        if not feature_att_trans:
            self.temporal_gat = TemporalAttentionLayer(
                n_features,
                window_size,
                dropout,
                alpha,
                time_gat_embed_dim,
                use_gatv2,
                output_activation=gat_output_activation,
            )
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
            raw_fusion_dim = 3 * n_features
            d_model = self.fusion_projection_dim or raw_fusion_dim
            if d_model != raw_fusion_dim:
                self.fusion_projection = nn.Linear(raw_fusion_dim, d_model)
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
                # C3 初始时严格保留共享 GRU 骨干，再学习长程 Transformer 上下文是否有用。
                # 这里只把输出投影初始化为零：第一步投影层仍可获得梯度；投影参数离开
                # 零点后，梯度会继续传入 Transformer 编码器。
                nn.init.zeros_(self.trans_proj.weight)
                nn.init.zeros_(self.trans_proj.bias)
                torch.set_rng_state(transformer_rng_state)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        if self.use_condition_routed_adapter:
            adapter_rng_state = torch.get_rng_state()
            self.condition_routed_adapter = ControlRoutedLowRankAdapter(
                gru_hid_dim * gru_n_layers,
                n_features,
                rank=condition_adapter_rank,
                expert_count=condition_adapter_experts,
                control_indices=condition_graph_control_indices,
                temperature=condition_adapter_temperature,
            )
            torch.set_rng_state(adapter_rng_state)
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
            # 暂存并恢复随机数状态，使增加物理响应模块后仍可与原 C3 训练轨迹公平比较。
            # 训练数据随机采样器在开始迭代时取种子，因此这里构造额外模块不能提前消耗
            # 它所依赖的全局 CPU 随机数。
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
        """为 C4 构造从原始工程量到隐藏物理状态的编码器。"""
        if self.physical_state_injection_mode == "direct":
            return PhysicalStateEncoding(
                d_model, hidden_dim=hidden_dim, config=config, feature_mode=feature_mode,
            )
        # 只有新增分支不改变共享 Transformer/GRU 的随机初始化时，零门控才真正表示
        # “与 C3 完全相同”的起点，所以构造编码器前后需要恢复随机数状态。
        rng_state = torch.get_rng_state()
        encoder = PhysicalStateEncoding(
            d_model, hidden_dim=hidden_dim, config=config, feature_mode=feature_mode,
        )
        torch.set_rng_state(rng_state)
        return encoder
    def _inject_physical_state(self, representation, state_input):
        """把 C4 物理状态注入共享时序表示，供 Transformer/后续头使用。"""
        physical_state = self.physical_state_encoder(state_input)
        if self.physical_state_injection_mode == "gated_residual":
            # 门控初值为零，因此起点严格等于 C3 表示；训练后该标量可按数据放入或抑制
            # 物理残差，而不改变 GRU 分支和 C3 的条件化位置。
            return representation + torch.tanh(self.physical_state_gate) * physical_state
        return representation + physical_state
    def _fuse_physical_feature_gat(self, h_feat, state_input):
        """可选地在 feature-GAT 输出处融合物理响应特征。"""
        if not self.use_physical_response_features:
            return h_feat
        if self.physical_feature_fusion_mode != "feature_gat_residual":
            return h_feat
        physical = self.physical_response_feature_encoder(state_input)
        return h_feat + torch.tanh(self.physical_feature_gate) * physical
    def _fuse_physical_shared(self, h_cat, state_input):
        """可选地在特征/时间关系拼接后的共享表示处融合物理特征。"""
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
        """前向计算一个窗口的下一时刻预测与当前窗口重构。
        参数 ``x`` 为 ``[batch, lookback, features]``。返回值均是
        ``[batch, out_dim]``；训练器把它们与下一点目标/窗口末状态比较，后续评分器
        再把误差变为异常分数。
        """
        # x 形状为 (b, n, k)：b 为批大小，n 为窗口大小，k 为特征数
        # 保留未归一化/未卷积的输入给 C3 工况编码与 C4 物理模块，避免语义丢失。
        state_input = x
        if self.backbone_feature_indices.numel() > 0:
            x = torch.index_select(x, dim=2, index=self.backbone_feature_indices)
        revin_stats = None
        if self.use_revin:
            x, revin_stats = self.revin(x, mode="norm")
        regime_embedding = None
        self._regime_aux_prediction = None
        self._regime_aux_target = None
        if self.use_regime_condition:
            condition_input = state_input
            if self.regime_condition_shuffle and state_input.size(0) > 1:
                condition_input = state_input.clone()
                control_indices = getattr(self.regime_encoder, "control_indices", None)
                if control_indices is not None and control_indices.numel() > 0:
                    controls = torch.index_select(state_input, dim=2, index=control_indices)
                    condition_input[:, :, control_indices] = torch.roll(controls, shifts=1, dims=0)
            if self.regime_encoder_type == "restricted":
                (
                    regime_embedding,
                    self._regime_aux_prediction,
                    self._regime_aux_target,
                ) = self.regime_encoder(condition_input, return_auxiliary=True)
            elif self.regime_encoder_type == "temporal":
                regime_embedding, self._regime_aux_prediction = self.regime_encoder(
                    condition_input,
                    return_auxiliary=True,
                )
            else:
                regime_embedding = self.regime_encoder(condition_input)
                self._regime_aux_prediction = None
        self._regime_embedding = regime_embedding
        # 1) 局部卷积；2) 通道关系 GAT；3)（标准路径中）时间关系 GAT。
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
        # C4 或条件图可以提供注意力偏置，但不会替代数据驱动的图注意力。
        h_feat = self.feature_gat(x, attention_bias=physical_attention_bias)
        self._feature_attention_weights = self.feature_gat.last_attention
        self._feature_attention_probabilities = self.feature_gat.last_attention_raw
        if self.use_regime_condition and self.regime_condition_mode in {
            "feature_gat", "feature_gat_response"
        }:
            mask = (
                self.regime_response_mask
                if self.regime_condition_mode == "feature_gat_response"
                else None
            )
            h_feat = self._apply_condition(
                h_feat, regime_embedding, self.feat_conditioner, mask=mask
            )
        h_feat = self._fuse_physical_feature_gat(h_feat, state_input)
        # 两条编码路径：简化路径仅 feature-GAT + Transformer；标准路径额外保留
        # temporal-GAT 与 GRU，Transformer 只作为长程残差补充。
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
            self._temporal_attention_weights = self.temporal_gat.last_attention
            if self.use_regime_condition and self.regime_condition_mode == "temporal_gat":
                h_temp = self._apply_condition(h_temp, regime_embedding, self.temp_conditioner)
            h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # 形状：(b, n, 3k)
            if self.use_regime_condition and self.regime_condition_mode == "fusion":
                h_cat = self._apply_condition(h_cat, regime_embedding, self.fusion_conditioner)
            h_cat = self._fuse_physical_shared(h_cat, state_input)
            if self.fusion_projection is not None:
                h_cat = self.fusion_projection(h_cat)
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
        if self.use_condition_routed_adapter:
            h_end, self._condition_adapter_routing = self.condition_routed_adapter(
                h_end, state_input, return_routing=True
            )
        if self.use_regime_condition and self.regime_condition_mode == "head":
            h_end = self._apply_condition(
                h_end, regime_embedding, self.head_conditioner
            )
        # 共享隐状态分叉为：下一时刻预测头和窗口重构头。异常分数由两类误差融合。
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

    def _c3_value_residual(self, x, y, predictions, shifted_recons):
        """与正式评分一致地形成下一点预测+重构数值残差。"""
        y_target = y.squeeze(1)
        if self.target_dims is not None:
            y_target = y_target[:, self.target_dims]
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)
        prediction_error = torch.abs(y_target - predictions).mean(dim=1)
        reconstruction_error = torch.abs(
            y_target - shifted_recons[:, -1, :]
        ).mean(dim=1)
        return prediction_error + self.c3_value_gamma * reconstruction_error

    def c3_joint_components(self, x, y, predictions):
        """返回关系预测损失和模型内部二维联合异常分数。

        第二次前向只构造真实下一窗口的关系图与重构值；不会复用历史 checkpoint
        或离线校准器。关系转移和联合密度都不读取状态嵌入。
        """
        if not self.use_c3_joint_relation:
            raise RuntimeError("C3 joint relation head is disabled")
        current_attention = self._feature_attention_probabilities
        if current_attention is None:
            raise RuntimeError("Run the main model forward before C3 joint scoring")
        current_attention = current_attention.detach()
        shifted_x = torch.cat((x[:, 1:, :], y), dim=1)
        with torch.no_grad():
            _, shifted_recons = self(shifted_x)
            next_attention = self._feature_attention_probabilities.detach()
        predicted_attention = self.c3_relation_head(current_attention)
        relation_residual = F.mse_loss(
            predicted_attention, next_attention, reduction="none"
        ).mean(dim=(1, 2))
        value_residual = self._c3_value_residual(
            x, y, predictions, shifted_recons
        )
        residual_vector = torch.log1p(
            torch.stack((value_residual, relation_residual), dim=1).clamp_min(0.0)
        )
        joint_nll = self.c3_joint_score_head.negative_log_likelihood(
            residual_vector.detach()
        )
        return {
            "relation_loss": relation_residual.mean(),
            "joint_nll_loss": joint_nll.mean(),
            "joint_score": joint_nll,
            "value_residual": value_residual,
            "relation_residual": relation_residual,
            "shifted_reconstruction": shifted_recons,
        }
    def regime_auxiliary_loss(self, x):
        """用窗口工况描述量自监督 C3 embedding，防止其退化为无语义的自由向量。"""
        if (
            self.regime_encoder_type == "restricted"
            and self._regime_aux_prediction is not None
            and self._regime_aux_target is not None
        ):
            return (
                torch.nn.functional.smooth_l1_loss(
                    self._regime_aux_prediction, self._regime_aux_target
                )
                + self.control_response_auxiliary_loss(x)
            )
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
        """计算可选控制量→响应量辅助预测损失，供 C4 训练期约束使用。"""
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
        """仅导出 C3 工况 embedding，供分析/可视化而不执行完整预测。"""
        if not self.use_regime_condition:
            raise RuntimeError("Regime conditioning is disabled for this model")
        if self.regime_encoder_type in {"temporal", "restricted"}:
            return self.regime_encoder(x)
        if self.use_revin:
            x, _ = self.revin(x, mode="norm")
        return self.regime_encoder(x)
    def _apply_condition(self, hidden_state, regime_embedding, conditioner, mask=None):
        gamma, beta = conditioner(regime_embedding)
        if mask is not None:
            gamma = gamma * mask
            beta = beta * mask
        while gamma.ndim < hidden_state.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        # 保守地调制骨干：工况只适配 MTAD-GAT 特征，不直接替换其尺度和偏移。
        # 将调制幅度限制为 10%，也能降低小规模开发折上 C3 增量过拟合的风险。
        scale = self.regime_film_scale
        return hidden_state * (1.0 + scale * gamma) + scale * beta

def find_largest_valid_nhead(d_model, max_nhead=8):
    for nhead in range(max_nhead, 0, -1):
        if d_model % nhead == 0:
            return nhead
    return 1
