# 第四章论文版模型总览图

这版对应第四章增强模型，重点突出 3 个新增部分：

- 物理状态编码 `Physical State Encoding`
- 物理先验正则 `Physical Regularization`
- 层级一致性融合 `Hierarchical Consistency`

建议标题：

- `图 X  融合物理状态编码与层级一致性的工况感知异常检测框架`

## 中文论文版

```mermaid
flowchart TB
    X["输入窗口 X"] --> N["RevIN 归一化"]
    N --> C["局部编码\n1D Conv + Dual Attention"]
    C --> Z["拼接表示 z"]

    Z --> G["GRU 主状态\nh_gru"]
    Z --> PE["物理状态编码\n电流相位/电荷累积\n电压温度相对位置\nSOC / SOH"]
    Z --> POS["位置编码"]
    PE --> ADD["状态融合"]
    POS --> ADD
    ADD --> T["Transformer Encoder"]
    T --> TP["Mean Pooling + Linear\nh_trans"]

    R["工况编码 r"] --> M["工况感知门控"]
    G --> M
    TP --> M
    M --> H["融合状态\nh = h_gru + g(r) * h_trans"]

    H --> P["预测头"]
    H --> Q["重构头"]
    P --> DN["RevIN 反归一化"]
    Q --> DN
    DN --> SF["质量感知分数融合\n预测误差 + 重构误差"]
    SF --> HC["层级一致性融合\n主分支(V/I/T) + 残差分支"]
    HC --> Y["最终异常分数 / 告警"]

    Q -.约束.-> PR["物理先验正则\n派生一致性 L_alg\n阶段平滑 L_smooth"]
    P -.训练损失.-> LT["总损失\nL_pred + L_recon + lambda_alg L_alg + lambda_smooth L_smooth"]
    Q -.训练损失.-> LT
    PR -.并入.-> LT

    classDef base fill:#eef2f7,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef ours fill:#e3f2fd,stroke:#3f6ea8,color:#16324f,stroke-width:1.2px;
    classDef output fill:#ffffff,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef loss fill:#f5f0ff,stroke:#8b74c9,color:#372f52,stroke-width:1px;

    class X,Y output;
    class N,C,Z,G,T,TP,P,Q,DN,SF base;
    class PE,POS,R,M,H,HC ours;
    class PR,LT loss;
```

## 英文精简版

```mermaid
flowchart TB
    X["Input Window"] --> N["RevIN"]
    N --> C["Conv Encoder + Dual Attention"]
    C --> Z["Fused Representation"]

    Z --> G["GRU State"]
    Z --> PE["Physical State Encoding"]
    Z --> POS["Position Encoding"]
    PE --> ADD["State Injection"]
    POS --> ADD
    ADD --> T["Transformer Encoder"]
    T --> TP["Pooling -> h_trans"]

    R["Regime Code"] --> M["Regime-Aware Gate"]
    G --> M
    TP --> M
    M --> H["Residual Fusion"]

    H --> P["Forecast Head"]
    H --> Q["Reconstruction Head"]
    P --> O["Score Fusion"]
    Q --> O
    O --> HC["Hierarchical Consistency"]
    HC --> Y["Alarm"]

    Q -.physics loss.-> PR["Physical Regularization"]
    P -.training loss.-> LT["Overall Loss"]
    Q -.training loss.-> LT
    PR -.add.-> LT

    classDef base fill:#eef2f7,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef ours fill:#e3f2fd,stroke:#3f6ea8,color:#16324f,stroke-width:1.2px;
    classDef output fill:#ffffff,stroke:#7b8794,color:#1f2933,stroke-width:1px;
    classDef loss fill:#f5f0ff,stroke:#8b74c9,color:#372f52,stroke-width:1px;

    class X,Y output;
    class N,C,Z,G,T,TP,P,Q,O base;
    class PE,POS,R,M,H,HC ours;
    class PR,LT loss;
```

## 图注建议

- 第四章在基线模型基础上引入物理状态编码，将充放电阶段、电荷累积、相对位置及 `SOC/SOH` 等状态变量注入 Transformer 分支。
- 通过工况编码生成门控系数，对 `GRU` 主状态与 Transformer 上下文进行自适应融合，以增强工况切换阶段的表征能力。
- 训练阶段加入物理先验正则，包括派生一致性约束与阶段平滑约束，以减小不合理波动。
- 推理阶段先进行质量感知分数融合，再通过主分支与残差分支的层级一致性得到最终异常分数。

## 排版建议

- 正文主图优先使用英文精简版，中文细节放图注。
- 如果版面较窄，可删去损失分支，只保留主干推理路径，把 `Physical Regularization` 画成右侧注释框。
- 若需要和第三章图统一，保留同一套配色：基线浅灰、第四章新增模块浅蓝、训练损失浅紫。
