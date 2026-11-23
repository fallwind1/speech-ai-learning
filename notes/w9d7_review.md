
# W9D7：Transformer / Conformer / CTC 复盘

## 1. Transformer

Transformer是一种基于Attention机制的序列建模网络架构。

核心模块：
- MHSA: 通过Query、Key、Value计算不同位置之间的注意力关系，实现全局上下文建模。
- FFN: 对每个位置的特征进行非线性变换，增强模型的表示能力。
- Residual Connection：保留原始特征，改善梯度传播。
- LayerNorm：规范化特征分布，稳定训练。

对于语音任务，Transformer Encoder可以利用整段音频的上下文信息，对每个时间位置的声学特征进行重新建模。

Encoder输出通常为：
$$
H\in\mathbb R^{B\times T\times C}
$$

其中：
- B：Batch Size
- T：时间帧数
- C：隐藏维度

## 2. Conformer

Conformer 是一种结合 Self-Attention 与卷积的语音网络架构。

其核心思想是同时利用：
- MHSA：建模全局上下文关系。
- ConvModule：提取局部声学特征和短时动态模式。
- FFN：进行特征非线性变换。

语音信号相邻时间帧通常具有较强的相关性，具有明显的局部结构。单纯依靠 Self-Attention 虽然能够建模全局关系，但卷积能够进一步加强局部特征提取。

Conformer Block 的结构：

$$
X\rightarrow FFN_1/2
\rightarrow MHSA
\rightarrow Conv
\rightarrow FFN_2/2
\rightarrow LayerNorm
$$

两个 Half-step FFN 分别位于 Block 前后，通过 0.5 倍残差贡献，使 FFN 的作用分布在整个 Block 中。

自己实现的简化版 Conformer：
- ConformerFFN
- ConformerMHSA
- ConformerConvModule
- ConformerBlock

与原论文的差别：
- 尚未实现相对位置编码机制。
- 只完成了 Block 级别的实现和验证。
- 尚未完成完整 Conformer ASR 的训练。

## 3. CTC

CTC（Connectionist Temporal Classification）是一种用于序列学习的训练准则，主要解决输入序列与输出文本之间缺少明确对齐关系的问题。

例如：
- 音频经过 Encoder 后具有 T 个时间帧。
- 目标文本只有 U 个 Token。
- 通常 T 远大于 U。
- 不需要人工标注每个时间帧对应哪个文本 Token。

CTC 引入 Blank 类别，并允许连续重复 Token。

CTC 映射过程：
1. 合并连续重复 Token。
2. 删除 Blank。

例如：

$$
C,C,\varnothing,A,A,T,T
$$

首先合并连续重复：

$$
C,\varnothing,A,T
$$

然后删除 Blank：

$$
CAT
$$

CTC 的目标概率：

$$
P(Y|X)=
\sum_{\pi\in\mathcal B^{-1}(Y)}
P(\pi|X)
$$

其中：
- X：输入音频。
- Y：目标文本。
- π：帧级预测路径。
- B：CTC 映射函数。

CTC 的条件独立假设：

$$
P(\pi|X)=
\prod_{t=1}^{T}P(\pi_t|X)
$$

CTC Loss：

$$
\mathcal L_{\mathrm{CTC}}
=-\log P(Y|X)
$$

CTC Forward Algorithm 使用动态规划计算所有合法路径的概率总和，避免枚举全部可能路径。

CTC Decoder：
- Greedy Decoder：逐帧 Argmax，合并连续重复 Token，删除 Blank。
- Beam Search：搜索概率较高的路径或前缀，可以结合语言模型。

## 4. 三者之间的关系

Transformer和Conformer都属于网络架构层面。

Conformer可以看作一种结合全局注意力与局部卷积的语音网络架构。

CTC 属于训练目标与序列对齐方法层面，与Encoder架构相互独立。

因此：
- Transformer Encoder 可以搭配 CTC Loss。
- Conformer Encoder 可以搭配 CTC Loss。
- 其他适合的 Encoder 也可以搭配 CTC Loss。

CTC Loss 不是评价指标（Metric），而是用于训练和反向传播的损失函数。

WER、CER 才是常见的 ASR 评价指标。

CTC 不需要传统自回归神经网络 Decoder，但推理时仍然需要 CTC 解码算法。


## 5. 完整 ASR 流程

以 Conformer + CTC 为例。

训练阶段：

音频
→ Feature Encoder
→ Conformer Encoder
→ Linear CTC Head
→ Log Softmax
→ CTC Loss
→ Backward
→ Optimizer

其中：

1. Feature Encoder：提取音频特征，必要时进行下采样。
2. Conformer Encoder：利用 MHSA 和卷积进行全局、局部特征建模。
3. CTC Head：将隐藏特征映射到 CTC 词表类别。
4. Log Softmax：计算每个时间帧的类别对数概率。
5. CTC Loss：根据真实文本计算训练损失。
6. Backward 和 Optimizer：更新模型参数。

CTC Head 的输入：

$$
H\in\mathbb R^{B\times T\times C}
$$

CTC Head 的输出：

$$
Z\in\mathbb R^{B\times T\times V}
$$

其中 V 是包含 Blank 的 CTC 类别数。

PyTorch CTCLoss 的输入形状：

$$
(T,B,V)
$$

还需要：
- targets：真实文本 Token ID。
- input_lengths：Encoder 输出的有效时间长度。
- target_lengths：真实文本 Token 长度。

推理阶段：

音频
→ Feature Encoder
→ Conformer Encoder
→ Linear CTC Head
→ CTC Decoder
→ 文本

CTC Decoder 可以使用 Greedy 或 Beam Search，不需要传统自回归 Decoder。

## 6. 自己的实现情况

已经实现：
- W3 Dot Product Attention。
- ConformerFFN。
- ConformerMHSA。
- ConformerConvModule。
- ConformerBlock。
- W8 CTC Forward Algorithm。
- W8 CTC Greedy Decoder。

已经验证：
- Conformer Block 前向传播。
- Conformer Block 反向传播。
- 输出 Shape 保持。
- 梯度存在且没有 NaN。
- Padding 与 Mask 的部分行为。
- 卷积参数量计算。
- 与 Transformers 开源 ConvModule 的源码对照。
- CTC Forward Algorithm 与暴力枚举结果一致。
- CTC 相关 18 项测试全部通过。

尚未完成：
- Conformer 相对位置机制。
- 完整 Conformer Encoder 的系统集成。
- Conformer + CTC Head。
- 完整 ASR 训练。
- Conformer ASR 的 WER/CER 评估。

因此，目前达到的是简化版 Conformer Block 的实现与验证阶段，而不是完整 Conformer ASR 系统的训练阶段。
