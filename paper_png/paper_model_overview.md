# 论文版模型总览图

下面这版是按论文正文插图风格重画的源文件：

- 标签尽量缩短，避免大段解释塞进框里
- 只保留主干数据流，不把实现细节全部堆进去
- 用浅灰表示 baseline 主干，用浅蓝表示本文增强模块
- 适合后续导出为 `SVG/PNG` 再放进论文

建议标题：

- `图 X  工况感知 Transformer 残差增强的 MTAD-GAT 总体框架`

```mermaid
flowchart LR
    X["输入窗口 X\nshape=(L,F)"] --> N["RevIN\n归一化"]
    N --> C["局部时序编码\n1D Conv"]
    C --> FG["特征注意力"]
    C --> TG["时间注意力"]
    FG --> Z["融合表示 z"]
    TG --> Z

    Z --> G["GRU 主状态\nh_gru"]
    Z --> T["Transformer 残差上下文\nh_trans"]
    S["工况统计量 s"] --> RG["门控单元 g(s)"]
    T --> FUSE["残差融合\nh = h_gru + g*h_trans"]
    G --> FUSE
    RG --> FUSE

    FUSE --> P["预测头\nForecast"]
    FUSE --> R["重构头\nReconstruction"]
    P --> A["异常分数融合\nA = w_p E_pred + w_r E_recon"]
    R --> A
    A --> E["事件一致性\n双阈值约束"]
    E --> Y["异常告警输出"]

    classDef base fill:#e9eef5,stroke:#8aa1c1,color:#1f2d3d,stroke-width:1px;
    classDef ours fill:#dceeff,stroke:#4f81bd,color:#153a66,stroke-width:1.2px;
    classDef io fill:#f5f7fa,stroke:#9aa5b1,color:#20262e,stroke-width:1px;

    class X,Y io;
    class N,C,FG,TG,Z,G,P,R base;
    class T,RG,FUSE,A,E,S ours;
```

## 图注建议

- `RevIN` 用于削弱不同工况下的统计漂移。
- `1D Conv + 特征/时间注意力` 负责提取局部时序模式与跨变量依赖。
- `GRU` 建模主时序状态，`Transformer` 提供长程上下文残差补偿。
- `工况统计量` 生成门控系数，对 Transformer 残差进行自适应调节。
- 预测误差与重构误差融合为异常分数，再经过事件一致性约束输出最终告警。

## 论文排版建议

- 正文图尽量只保留英文或“中英混排”短标签，不要在框内写公式解释句。
- 如果投稿模板是黑白打印，建议把 `ours` 统一改成浅灰蓝边框，避免高饱和黄色。
- 如果版面较窄，可改成上下两行：
  第一行 `输入 -> 编码 -> 融合 -> GRU/Transformer`
  第二行 `预测/重构 -> 分数融合 -> 事件一致性 -> 输出`

## 更简洁版本

如果你想要更像论文主图的“极简版”，可以用下面这版：

```mermaid
flowchart LR
    X["Input Window"] --> N["RevIN"]
    N --> C["Conv Encoder"]
    C --> A1["Feature Attention"]
    C --> A2["Temporal Attention"]
    A1 --> Z["Fused Representation"]
    A2 --> Z
    Z --> G["GRU State"]
    Z --> T["Transformer Context"]
    S["Regime Statistics"] --> M["Gate"]
    G --> H["Residual Fusion"]
    T --> H
    M --> H
    H --> P["Forecast Head"]
    H --> R["Reconstruction Head"]
    P --> O["Score Fusion"]
    R --> O
    O --> E["Event Consistency"]
    E --> Y["Alarm"]

    classDef base fill:#eef2f7,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef ours fill:#e3f2fd,stroke:#3f6ea8,color:#16324f,stroke-width:1.2px;
    classDef io fill:#ffffff,stroke:#7b8794,color:#1f2933,stroke-width:1px;

    class X,Y io;
    class N,C,A1,A2,Z,G,P,R base;
    class T,S,M,H,O,E ours;
```

## 第四章增强版

如果你希望第四章直接按第三章第一张图的风格扩展，推荐用下面这版。它保持原有中文模块命名和横向主干，只把第四章新增内容插到对应位置：

- 在 `拼接表示` 后增加 `物理状态编码` 与 `位置编码`
- 在 `异常分数计算` 后增加 `层级一致性融合`
- 在重构分支补充 `物理正则化`

建议标题：

- `图 X  第四章物理增强与层级一致性扩展后的统一框架`

```mermaid
flowchart LR
    X["输入窗口 X"] --> N["RevIN"]
    N --> C["1D卷积"]
    C --> A1["特征注意力图"]
    C --> A2["时间注意力图"]
    A1 --> Z["拼接\n[x, h_feat, h_temp]"]
    A2 --> Z

    Z --> G["GRU时序编码\nh_gru"]
    Z --> PE["物理状态编码\n相位/累积电荷\n相对位置/SOC/SOH"]
    Z --> POS["位置编码"]
    PE --> T["Transformer 残差\nEncoder -> Pooling -> h_trans"]
    POS --> T

    S["工况统计量 r"] --> M["工况感知门控\nh = h_gru + g(r) * h_trans"]
    G --> M
    T --> M

    M --> P["预测头"]
    M --> R["重构头"]
    P --> DN["RevIN 反归一化"]
    R --> DN
    DN --> O["异常分数计算\n预测误差 + 重构误差"]
    O --> HC["层级一致性融合\n主分支 + 残差分支"]
    HC --> E["事件一致性判别"]
    E --> Y["异常预测结果"]

    R -.训练期约束.-> PR["物理正则化\nL_alg + L_smooth"]

    classDef base fill:#eef2f7,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef ours fill:#e3f2fd,stroke:#3f6ea8,color:#16324f,stroke-width:1.2px;
    classDef io fill:#ffffff,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef train fill:#f5f0ff,stroke:#8b74c9,color:#372f52,stroke-width:1px;

    class X,Y io;
    class N,C,A1,A2,Z,G,P,R,DN,O,E base;
    class T,S,M,PE,POS,HC ours;
    class PR train;
```

### 第四章图注建议

- 第四章在第三章模型主干上增加物理状态编码，将充放电阶段、相对位置以及 `SOC/SOH` 等状态信息注入 Transformer 分支。
- 在异常分数计算之后增加层级一致性融合，对主分支与残差分支分数进行联合约束，提升复杂工况下的稳定性。
- 重构分支在训练阶段引入物理正则化，用于约束派生一致性与阶段平滑性。

### 使用建议

- 第三章正文继续用上面的第一张主图风格。
- 第四章正文直接用这张“增强版”，整体观感会和第三章保持一致。
- 如果要更接近你原来的图，可以把 `物理状态编码` 和 `层级一致性融合` 框再加宽一点，减少换行。
