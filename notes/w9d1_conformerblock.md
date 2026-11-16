# W9D1：Conformer Block——局部与全局

## 1. 今日目标

- 读 R08 §2 / Figure 1，理解 Conformer Block 的整体结构。
- 结合 R05 神经 ASR 讲义（若受限则用 R04）理解为什么语音既需要全局建模，也需要局部建模。
- 画出 Conformer Block：
  半步 FFN → MHSA → Conv Module → 半步 FFN → Norm
- 能逐模块解释设计动机。

## 2. Conformer Block 总体结构

结构图：

Conformer Block的主要组成：
1. 第一半步FFN(Feed Forward Module)
2. 多头自注意力模块
3. 卷积模块
4. 第二半步FFN
5. 最终LayerNorm

整体结构：
Input -> 1/2 FFN -> MHSA -> Convolution Module -> 1/2 FFN -> LayerNorm -> Output

输入：
Conformer Block的输入不是原始音频，而是经过前端特征提取和下采样后的语音特征序列。

假设输入为：
$$
X\in\mathbb R^{B\times T\times d_{\text{model}}}
$$

其中：
- (B)：Batch Size。
- (T)：时间帧数量。
- (d_{\text{model}})：特征隐藏维度。

输出：
$$
Y\in\mathbb R^{B\times T\times d_{\text{model}}}
$$

模块顺序：
$$
\boxed{
\frac12 FFN
\rightarrow MHSA
\rightarrow Conv
\rightarrow \frac12 FFN
\rightarrow LayerNorm
}
$$

各模块的核心作用：

- FFN: 对特征进行通道维度上的非线性变换
- MHSA: 建模全局上下文关系
- Convolution Module：提取局部声学特征。
- LayerNorm：规范化输出，帮助稳定训练

为什么叫 Conformer：
Convolution-augmented Transformer的缩写，意思是加入卷积增强的Transformer
它将Transformer的全局建模能力与CNN的局部特征提取能力结合，使模型能够同时建模语音中的局部模式和长距离依赖关系。

## 3. 为什么语音既有局部性又有全局性

### 3.1 语音局部性

含义：
语音信号具有明显的短时局部结构，相邻时间帧之间通常具有较高的相关性，局部时间范围内的声学特征具有一定的连续性。

例子：

1. 相邻语音帧之间通常高度相关。
2. 一个音素的声学特征通常持续若干个时间帧，而不是只出现在一个时间点。
3. 共振峰、爆破音以及音素之间的过渡等现象，都具有明显的局部声学模式。
4. 协同发音（Coarticulation）使当前音素的发音受到相邻音素影响，相邻时间范围内存在连续的声学变化。

为什么卷积适合建模局部性：
卷积具有局部连接和参数共享的特点：

- 局部连接：每次主要处理一个局部时间窗口内的特征。
- 参数共享：同一组卷积核可以在不同时间位置提取相同类型的局部模式。
- 局部归纳偏置：模型天然倾向于利用相邻位置之间的局部关系。

因此，卷积适合提取语音中的短时声学特征和局部动态模式。

### 3.2 语音全局性

含义：
语音识别不仅依赖局部声学信息，还需要利用更长范围的上下文信息。不同时间位置之间可能存在长距离依赖关系。

例子：

1. 某个音素的识别可能受到前后音素影响。
2. 当某个发音不够清晰时，需要结合更长范围的语音上下文进行判断。
3. 单词和句子的识别可能受到远距离语音信息及上下文关系影响。

为什么自注意力适合建模全局性：
Self-Attention允许每个时间位置与序列中其他时间位置进行信息交互。
通过Query、Key和Value的计算，每个位置可以根据注意力权重，对其他位置的信息进行加权融合。
因此：

- 每个时间位置可以直接获取整段序列的上下文信息
- 能够建立长距离依赖关系
- 不需要像普通局部卷积那样通过逐层扩大感受野才能传递远距离信息

## 4. 各模块动机

### 4.1 第一半步 FFN

作用：
FFN（Feed Forward Network）主要负责对每个时间位置的特征进行通道维度上的非线性变换，增强模型的特征表示能力。

典型结构：

$$
FFN(x)=W_2\sigma(W_1x+b_1)+b_2
$$
其中通常先将特征维度扩展，再经过激活函数，最后映射回原来的隐藏维度。

为什么是半步：
Conformer 采用 Macaron-style FFN 结构，将 FFN 分为前后两个独立的模块，并分别使用半步残差更新：
$$
x_1=x+\frac12 FFN_1(x)
$$

$$
x_2=x'+\frac12 FFN_2(x')
$$
这里的 1/2 作用于 FFN 的残差分支输出，而不是将 FFN 的隐藏层维度减半。
两个 FFN 具有各自独立的参数，不是同一个 FFN 重复使用。
这样可以让前馈变换分布在整个 Block 的前后两端，同时使两个 FFN 的残差贡献各自缩放为一半。

为什么放在最前面：
第一个FFN在MHSA和卷积模块之前进行通道维度上的非线性特征变换，为后续的全局上下文建模和局部特征提取提供特征表示。

### 4.2 MHSA

作用：
MHSA（Multi-Head Self-Attention）负责全局上下文建模，使每个时间位置能够与其他时间位置进行信息交互。
基本计算：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$
$$
Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
多头注意力将特征投影到不同子空间，使不同注意力头能够学习不同的关系。

为什么能建模全局关系：
因为每个 Query 都可以与序列中所有位置的 Key 进行计算，进而利用注意力权重对所有位置的 Value 进行加权融合。
因此，每个时间位置都可以直接获取整段序列的信息，建立长距离依赖关系。

相对位置在这里的重要性：
Conformer 采用相对位置编码（Relative Positional Encoding），其设计借鉴了 Transformer-XL 的相对位置编码思想。
传统绝对位置编码主要表示某个 Token 在序列中的绝对位置，而相对位置编码强调不同时间位置之间的相对距离。
对于语音序列，相隔多少时间帧也是重要的时序信息。
相对位置编码使 Attention 在计算时不仅考虑不同位置的内容关系，也能够考虑它们之间的相对位置关系，从而更好地建模语音序列中的时序依赖。

### 4.3 Convolution Module

作用：
Convolution Module 负责局部特征建模，提取短时间范围内的声学模式和动态变化信息。
原论文中的卷积模块由多个操作组成，包括：

- Pointwise Convolution
- GLU（Gated Linear Unit）
- Depthwise Convolution
- BatchNorm
- Swish Activation
- Pointwise Convolution

为什么 Conformer 还要加卷积：
虽然 MHSA 能够关注局部位置，但它主要通过注意力机制进行全局上下文建模，没有卷积那样明确的局部归纳偏置。
语音具有明显的短时局部结构，例如相邻帧之间高度相关、音素持续若干帧以及局部声学特征的连续变化。
因此，Conformer 加入卷积模块，进一步增强模型对局部声学模式的提取能力。

卷积对语音的归纳偏置是什么：
卷积通过局部连接和参数共享，使模型倾向于利用相邻时间位置之间的关系。
这种结构天然适合提取短时局部模式，能够以较少的参数学习不同时间位置上的局部特征。
其中 Depthwise Convolution 对各个通道分别进行时间维度上的卷积，重点提取局部时间模式；Pointwise Convolution 则负责通道维度上的信息混合。

### 4.4 第二半步 FFN

作用：
第二个 FFN 位于 MHSA 和 Convolution Module 之后，对已经融合全局上下文信息和局部声学特征的表示进行进一步的通道维度非线性变换，增强特征表示能力。

为什么又来一个 FFN：
Conformer 采用 Macaron-style 结构，将 FFN 分布在 Block 的前后两端。
第一个 FFN 在全局和局部建模之前进行特征变换，第二个 FFN 则在全局和局部信息融合之后进一步加工特征。

### 4.5 Final LayerNorm

作用：
LayerNorm 用于规范化 Block 的输出，使特征分布更加稳定，帮助模型进行训练以及后续 Conformer Block 的堆叠。

为什么最后做归一化：
经过两个 FFN、MHSA 和卷积模块的特征变换与残差相加后，输出特征可能具有不同的数值分布。

最后使用 LayerNorm 对整个 Block 的输出进行规范化，有助于稳定特征表示。

注意：FFN、MHSA 和 Convolution Module 内部也包含各自的归一化操作，Final LayerNorm 是整个 Block 最后额外进行的输出归一化，并不代表整个 Block 只有这一次 LayerNorm。

## 5. Conformer Block 架构图

                   Input x
                      │
                      ▼
              ┌──────────────┐
              │   1/2 FFN    │
              └──────────────┘
                      │
                 Residual Add
                      │
                      ▼
              ┌──────────────┐
              │     MHSA     │
              │ Relative Pos │
              └──────────────┘
                      │
                 Residual Add
                      │
                      ▼
              ┌──────────────┐
              │ Convolution  │
              │    Module    │
              └──────────────┘
                      │
                 Residual Add
                      │
                      ▼
              ┌──────────────┐
              │   1/2 FFN    │
              └──────────────┘
                      │
                 Residual Add
                      │
                      ▼
              ┌──────────────┐
              │  LayerNorm   │
              └──────────────┘
                      │
                      ▼
                   Output

## 6. 核心结论

- 语音同时具有局部性和全局性，需要模型兼顾短时声学模式和长距离上下文关系。
- MHSA 负责全局上下文建模，但缺乏卷积那样明确的局部归纳偏置。
- Convolution Module 通过局部连接和参数共享，增强模型对短时声学模式的提取能力。
- Conformer 采用 Macaron-style FFN，将 FFN 放在 Block 的前后两端，并分别使用半步残差更新。
- Conformer 通过 MHSA 和卷积的结合，实现局部与全局信息的融合。

## 7. 自测问题

1. 为什么 Transformer 做语音建模还不够？
2. 为什么 Conformer 要同时有 MHSA 和卷积？
3. 为什么 FFN 要拆成两个 half-step FFN？
4. 语音的“局部性”具体体现在哪？