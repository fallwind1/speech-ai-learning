
# W8D5：wav2vec 2.0 架构

## 1. wav2vec 2.0 的基本思想

### 1.1 研究动机

传统监督式 ASR 需要大量带有人工转录文本的语音数据，数据标注成本较高。

但是，现实中存在大量只有音频、没有对应文本的数据。

wav2vec 2.0 的核心思想是：

先利用大量无标注语音进行自监督预训练，学习语音表示；再利用少量带有转录文本的语音数据进行有监督 CTC 微调，使模型具备语音识别能力。

两个阶段：

1. 自监督预训练（Self-supervised Pre-training）
   - 使用无标注语音。
   - 通过 Mask、Quantization 和 Contrastive Learning 学习语音表示。

2. 有监督微调（Supervised Fine-tuning）
   - 使用带有转录文本的语音。
   - 在预训练模型上增加 Linear CTC Head。
   - 使用 CTC Loss 训练语音到文本的映射。

### 1.2 整体架构

wav2vec 2.0 的核心组件：

| 组件 | 作用 |
|---|---|
| CNN Feature Encoder | 从原始语音波形中提取连续潜在特征 |
| Transformer Context Network | 建模不同时间位置之间的上下文关系 |
| Quantization Module | 将连续特征转换为离散量化目标 |
| Linear CTC Head | 将上下文表示映射为文本 Token 类别 |

整体架构：

**阶段一：自监督预训练**

原始语音
    ↓
CNN Feature Encoder
    ↓
连续潜在特征 Z
    ├──────────────────┐
    ↓                  ↓
Mask               Quantization
    ↓                  ↓
Transformer          离散量化
Encoder                 Q
    ↓                  ↓
上下文表示 C          Q
    └────────┬─────────┘
             ↓
      Contrastive Loss
             +
       Diversity Loss

**阶段二：有监督 CTC 微调**

原始语音
    ↓
CNN Feature Encoder
    ↓
Transformer Context Network
    ↓
Linear CTC Head
    ↓
Logits
    ↓
CTC Loss
    ↑
人工转录文本

注意：Quantization Module 仅用于自监督预训练，不是 CTC 微调和最终推理的必经部分。

---

## 2. 自监督预训练：两个分支

### 2.1 CNN 特征提取

输入：

原始音频波形（Raw Audio），而不是必须先转换成 Log-Mel 频谱。

设原始音频为：

X = (x1, x2, ..., xL)

其中 L 表示原始音频的采样点数量。

经过 CNN Feature Encoder：

Z = f(X)

得到：

Z = (z1, z2, ..., zT)

其中：
- X：原始音频波形。
- f：CNN Feature Encoder。
- Z：连续潜在语音特征。
- zt：第 t 个时间位置的特征向量。
- T：经过 CNN 下采样后的时间步数。

CNN 主要负责提取局部语音特征，同时降低时间维度。

注意：原始采样点数量 L 与 CNN 输出的时间步数 T 不相同，通常 T 远小于 L。

### 2.2 Masked Context 分支

数据流：

CNN 输出 Z
    ↓
Mask 部分时间步
    ↓
Transformer Encoder
    ↓
Context Representation C

Mask 位置：

CNN Feature Encoder 之后、Transformer 之前。

Transformer 的作用：

通过 Self-Attention 建模不同时间位置之间的关系，使每个时间位置的输出融合上下文信息。

输出：

C = (c1, c2, ..., cT)

其中 ct 表示第 t 个时间位置的上下文表示。

注意：
- 使用 Transformer Encoder。
- Mask 只作用于这一分支。
- Transformer 可以利用其他时间位置的信息，学习被 Mask 位置的表示。

### 2.3 Quantized Target 分支

数据流：

CNN 输出 Z
    ↓
Quantization Module
    ↓
Quantized Representation Q

输入：

未经过 Mask 的 CNN 连续特征 Z。

Quantization 的作用：

将连续特征映射为离散的量化表示，作为自监督预训练中的学习目标。

输出：

Q = (q1, q2, ..., qT)

其中 qt 表示第 t 个时间位置对应的量化目标。

为什么不进行 Mask？

因为该分支需要提供对应时间位置的真实量化目标。

如果也对该分支进行 Mask，就无法正常构造与 Masked Context 分支对应的量化目标。

**Codebook：**

Codebook 是一组可学习的向量集合，每个向量称为 Codebook Vector。

可以将其理解为一组可学习的向量原型，Quantization 根据输入特征选择相应的向量。

wav2vec 2.0 使用 Product Quantization：

- G：Codebook 的数量。
- V：每个 Codebook 中的向量数量。

原论文使用：

G = 2
V = 320

两个 Codebook 分别选择一个向量，再将选出的向量拼接并经过线性变换，得到最终量化表示。

理论上的组合数量：

V^G = 320^2 = 102400

模型使用 Gumbel-Softmax 实现可微分的离散选择，并通过 Straight-Through Estimator 处理梯度传播。

### 2.4 对比学习目标

首先需要解决两个分支输出维度可能不同的问题。

假设：

Transformer 输出：

C ∈ R^(T × d_model)

Quantization 输出：

Q ∈ R^(T × d_q)

通过两个线性投影层，将它们映射到相同的对比学习维度 f：

C' = CW_h + b_h

Q' = QW_q + b_q

最终：

C' ∈ R^(T × f)

Q' ∈ R^(T × f)

这样就可以逐时间位置计算余弦相似度。

**正样本 Positive Samples：**

同一个时间位置对应的上下文表示和量化目标。

例如：

(c3, q3)

表示第 3 个时间位置的正确匹配。

**负样本 Negative Samples：**

当前上下文表示与其他时间位置的量化表示构成的样本对。

例如：

(c3, q1)
(c3, q4)
(c3, q5)

原论文从同一条语音中其他被 Mask 时间位置对应的量化表示中采样负样本。

正负样本的作用：

- 正样本：提高正确匹配的相似度。
- 负样本：降低错误匹配的相对相似度。

模型需要根据上下文表示，从多个候选量化表示中找到正确的目标。

**Contrastive Loss：**

L_m = -log [
    exp(sim(c_t, q_t) / κ)
    /
    Σ_q exp(sim(c_t, q) / κ)
]

其中：
- c_t：上下文表示。
- q_t：正确的量化目标。
- q：正样本和负样本构成的候选集合。
- sim：余弦相似度。
- κ：温度参数。

该损失可以理解为一个分类任务：

给定 c_t，在多个候选量化表示中识别正确的 q_t。

**Diversity Loss：**

问题：

如果 Codebook 中有大量向量没有被使用，只有少数向量被频繁选择，那么量化表示的多样性就会降低。

解决办法：

引入 Diversity Loss，鼓励不同 Codebook Vector 得到较均衡的利用。

设：

p̄_(g,v)

表示一个 Batch 中，Codebook g 的第 v 个向量被选择的平均概率。

定义信息熵：

H(p̄_g) = -Σ_v p̄_(g,v) log p̄_(g,v)

信息熵越大，说明不同向量的使用越均衡。

Diversity Loss 可以表示为：

L_d = (1 / GV) Σ_g Σ_v p̄_(g,v) log p̄_(g,v)

即最大化平均使用分布的熵。

注意：这里关注的是 Batch 整体的使用分布，而不是要求每个时间位置都均匀选择所有向量。

最终预训练损失：

L = L_m + αL_d

其中：
- L_m：Contrastive Loss，负责识别正确的量化目标。
- L_d：Diversity Loss，负责鼓励充分利用 Codebook。
- α：控制 Diversity Loss 权重的超参数。

---

## 3. 有监督 CTC 微调

### 3.1 微调架构

自监督预训练完成后，已经获得了能够提取语音表示的 CNN Feature Encoder 和 Transformer Context Network。

接下来使用带有人工转录文本的数据进行有监督微调。

完整数据流：

原始语音
    ↓
CNN Feature Encoder
    ↓
Transformer Context Network
    ↓
Linear CTC Head
    ↓
Logits
    ↓
CTC Loss
    ↑
人工转录文本

其中：

**CNN Feature Encoder：**

从原始语音中提取连续潜在特征。

**Transformer Context Network：**

对语音特征进行上下文建模。

**Linear CTC Head：**

将 Transformer 输出映射到 CTC 类别空间。

假设 Transformer 输出：

C ∈ R^(T × d)

线性层参数：

W ∈ R^(d × |V|)

经过线性映射：

O = CW + b

得到：

O ∈ R^(T × |V|)

其中：
- d：Transformer 隐藏层维度。
- |V|：CTC 输出类别数量，包含 Blank 类别。
- O：每个时间步对应的分类 Logits。

### 3.2 CTC Loss

使用的监督信息：

语音对应的人工转录文本（Reference）。

为什么使用 CTC？

模型输出的是每个时间步的类别预测，但训练数据通常只提供完整的文本序列，并没有提供每个 Token 对应的具体时间位置。

CTC 通过考虑所有能够映射到目标文本的合法对齐路径，并对这些路径的概率求和，计算目标文本的概率。

因此不需要人工标注每个 Token 的具体时间位置。

CTC Loss 的作用：

计算模型预测与人工转录文本之间的损失，并通过反向传播更新模型参数。

量化分支是否保留？

不需要。

原因：

- 自监督预训练没有人工文本，需要 Quantization 构造离散学习目标。
- CTC 微调已经有人工文本，可以直接使用 CTC Loss 学习语音到文字的映射。
- 因此不再需要 Quantization Module 和对应的对比学习任务。

---

## 4. 自监督预训练与有监督微调的区别

| 对比维度 | 自监督预训练 | 有监督 CTC 微调 |
|---|---|---|
| 数据 | 无标注语音 | 有标注语音 |
| 监督信号 | 量化目标与负样本 | 人工转录文本 |
| 学习目标 | 学习语音表示 | 学习语音到文本的映射 |
| 损失函数 | Contrastive Loss + Diversity Loss | CTC Loss |
| CNN | 使用 | 使用 |
| Transformer | 使用 | 使用 |
| Mask | 使用 | 不使用预训练 Mask 任务 |
| Quantization | 使用 | 不使用 |
| Linear CTC Head | 不使用 | 使用 |
| 输出 | 上下文语音表示 | 每个时间步的类别 Logits |

核心区别：

自监督预训练阶段：

利用语音自身构造学习目标，使模型学习有用的语音表示。

有监督微调阶段：

利用人工转录文本作为监督信号，让模型学习语音与文字之间的映射关系。

---

## 5. 推理阶段

输入：

待识别的原始语音波形。

模型主体：

CNN Feature Encoder
+
Transformer Context Network
+
Linear CTC Head

完整流程：

原始语音
    ↓
CNN Feature Encoder
    ↓
Transformer Context Network
    ↓
Linear CTC Head
    ↓
Logits
    ↓
CTC Decoder
    ↓
最终识别文本

输出：

模型预测的文本序列。

CTC Decoder：

将模型每个时间步的类别预测转换为最终文本。

例如 Greedy Decoding：

逐帧选择概率最大的类别，再执行 CTC Collapse：

1. 合并连续重复 Token。
2. 删除 Blank。

最终得到识别文本。

Quantization 是否参与？

不参与。

原因：

Quantization Module 只负责自监督预训练阶段的离散目标构造。

推理阶段已经有训练完成的模型，只需要利用 CNN、Transformer 和 Linear CTC Head 生成 Logits，再通过 CTC Decoder 得到文本。

CTC Loss 是否参与？

不参与。

推理阶段不进行参数优化，不需要计算训练损失。

---

## 6. 本章总结

- wav2vec 2.0 最重要的架构特点：

  先利用无标注语音进行自监督预训练，再利用带有文本标注的语音进行 CTC 微调。

- 两个预训练分支的关系：

  CNN 输出连续特征 Z 后分为两个分支：

  1. Masked Context：Mask → Transformer Encoder → 上下文表示 C。
  2. Quantized Target：Quantization → 离散量化目标 Q。

  两个分支共同参与 Contrastive Learning。

- 为什么量化不是推理必经层：

  Quantization 的作用是为自监督预训练构造离散目标。微调阶段有人工文本，可以使用 CTC Loss；推理阶段不需要训练目标，因此都不再需要量化分支。

- 与之前学习的 CTC 的联系：

  wav2vec 2.0 在有监督微调阶段使用 Linear CTC Head 和 CTC Loss。

  模型输出每个时间步的类别 Logits，通过 CTC 建立帧级预测与文本序列之间的关系，再使用 CTC Decoder 得到最终识别文本。

**最重要的一句话：**

wav2vec 2.0 先通过自监督预训练学习语音表示，再通过有监督 CTC 微调学习语音到文本的映射。Quantization Module 是预训练阶段的辅助结构，并不是最终 ASR 推理架构的一部分。
