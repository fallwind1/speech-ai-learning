# W9D2：Conformer Convolution Module

## 1. 今日目标

- 理解 Conformer Convolution Module 的完整结构。
- 使用 PyTorch 实现 Pointwise Conv、GLU、Depthwise Conv、BatchNorm 和激活函数。
- 理解 B/T/C 三个维度及 Conv1d 所需的维度转换。
- 理解 Padding 如何保证时间维度不变。
- 计算 Depthwise Conv 的实际参数量。
- 完成 `src/conformer.py`。

## 2. ConvModule 整体结构

### 2.1 完整结构

Conformer Convolution Module 是 Conformer Block 中负责局部特征建模的模块，通过卷积提取语音中的短时声学模式和局部动态信息。

其完整结构：

```text
Input (B,T,C)
     ↓
LayerNorm
     ↓
Transpose (B,T,C) → (B,C,T)
     ↓
Pointwise Conv1D (C → 2C)
     ↓
GLU (2C → C)
     ↓
Depthwise Conv1D
     ↓
BatchNorm1D
     ↓
SiLU / Swish
     ↓
Pointwise Conv1D (C → C)
     ↓
Dropout
     ↓
Transpose (B,C,T) → (B,T,C)
     ↓
Output (B,T,C)
```

### 2.2 各模块的执行顺序

| 模块 | 主要作用 |
|---|---|
| LayerNorm | 对每个时间位置的特征进行归一化 |
| Transpose | 将输入转换为 Conv1d 所需的格式 |
| Pointwise Conv1 | 扩展通道，为 GLU 门控做准备 |
| GLU | 通过门控机制筛选特征信息 |
| Depthwise Conv | 独立提取各通道的局部时间特征 |
| BatchNorm | 对卷积输出进行批归一化 |
| SiLU | 引入非线性变换 |
| Pointwise Conv2 | 重新融合不同通道的信息 |
| Dropout | 正则化，降低过拟合风险 |
| Transpose | 恢复原来的张量格式 |

其中：

- Pointwise Conv 主要进行通道维度上的信息变换。
- Depthwise Conv 主要进行时间维度上的局部特征提取。
- 两者结合，使模型能够在提取局部时间模式的同时实现通道信息混合。

### 2.3 输入与输出形状

统一定义：

- B：Batch Size，批次大小。
- T：时间帧数量。
- C：特征通道数。

输入：

$$
X\in\mathbb R^{B\times T\times C}
$$

经过完整 ConvModule 后，输出：

$$
Y\in\mathbb R^{B\times T\times C}
$$

各模块的形状变化：

| 操作 | 输出形状 |
|---|---|
| Input | (B,T,C) |
| LayerNorm | (B,T,C) |
| Transpose | (B,C,T) |
| Pointwise Conv1 | (B,2C,T) |
| GLU | (B,C,T) |
| Depthwise Conv | (B,C,T) |
| BatchNorm | (B,C,T) |
| SiLU | (B,C,T) |
| Pointwise Conv2 | (B,C,T) |
| Dropout | (B,C,T) |
| Transpose | (B,T,C) |

ConvModule 不改变输入的时间长度和隐藏维度，因此可以与外层 ConformerBlock 的残差连接配合使用。

## 3. Pointwise Conv + GLU

### 3.1 Pointwise Conv 的作用

Pointwise Convolution 即逐点卷积，是卷积核大小为 1 的一维卷积。

例如：

```python
nn.Conv1d(
    in_channels=C,
    out_channels=2*C,
    kernel_size=1
)
```

它在每个时间位置上独立进行通道维度的线性变换，不直接融合相邻时间位置的信息。

主要作用是改变通道维度，为后续 GLU 门控操作提供特征。

### 3.2 为什么先将通道扩展为 2C？

因为后续 GLU 需要将输入沿通道维度拆分成两个部分：

$$
X=[A,B]
$$

其中：

$$
A,B\in\mathbb R^{B\times C\times T}
$$

因此，Pointwise Conv1 需要先将通道数量从 C 扩展到 2C：

$$
C\rightarrow 2C
$$

这样才能将输出分成两个具有相同形状的部分。

### 3.3 GLU 的计算原理

GLU（Gated Linear Unit）是一种门控机制，通过一个分支控制另一个分支的信息通过程度。

首先将输入拆分为 A 和 B，然后计算：

$$
GLU(A,B)=A\otimes\sigma(B)
$$

其中：

$$
\sigma(B)=\frac{1}{1+e^{-B}}
$$

- A：主要承载特征内容。
- B：通过 Sigmoid 生成门控权重。
- ⊗：逐元素相乘。

当门控值接近 1 时，对应的 A 特征能够较多保留；当门控值接近 0 时，对应特征受到抑制。

因此，GLU 能够通过门控机制动态调节特征信息。

### 3.4 GLU 前后的张量形状

假设 Pointwise Conv1 的输出为：

$$
X\in\mathbb R^{B\times 2C\times T}
$$

沿通道维度拆分：

$$
A,B\in\mathbb R^{B\times C\times T}
$$

经过 GLU：

$$
Y=A\otimes\sigma(B)
$$

最终：

$$
Y\in\mathbb R^{B\times C\times T}
$$

因此，GLU 完成了：

$$
2C\rightarrow C
$$

在 PyTorch 中：

```python
self.glu = nn.GLU(dim=1)
```

这里 `dim=1` 表示沿通道维度进行拆分。

## 4. Depthwise Convolution

### 4.1 Depthwise Conv 的原理

Depthwise Convolution（深度卷积）是一种特殊的分组卷积。

其核心思想是：每个输入通道使用独立的卷积核进行卷积，不与其他通道直接混合。

例如，输入有 64 个通道：

```text
通道1 → 卷积核1 → 输出通道1
通道2 → 卷积核2 → 输出通道2
通道3 → 卷积核3 → 输出通道3
...
通道64 → 卷积核64 → 输出通道64
```

每个通道独立进行时间维度上的卷积，从而提取该通道中的局部时间模式。

### 4.2 groups 参数的作用

PyTorch Conv1d 中的 `groups` 用于控制输入通道和输出通道的分组方式。

普通卷积通常设置：

```python
groups=1
```

表示所有输入通道共同参与卷积。

Depthwise Conv 设置：

```python
groups=C
```

当输入通道数和输出通道数均为 C 时，每组只有一个输入通道和一个输出通道。

因此，每个通道独立进行卷积，不发生跨通道混合。

典型实现：

```python
nn.Conv1d(
    in_channels=C,
    out_channels=C,
    kernel_size=k,
    groups=C
)
```

### 4.3 与普通 Conv1d 的区别

假设：

- 输入通道数 C=64。
- 输出通道数 C=64。
- Kernel Size=31。

普通 Conv1d：

每个输出通道都需要考虑所有输入通道的信息。

权重参数量：

$$
64\times64\times31
$$

Depthwise Conv：

每个通道独立使用一个卷积核。

权重参数量：

$$
64\times31
$$

因此，Depthwise Conv 的参数量远小于相同输入输出通道配置的普通 Conv1d。

不过，Depthwise Conv 不负责充分融合不同通道的信息，需要配合 Pointwise Conv 实现通道混合。

### 4.4 参数量计算

PyTorch Conv1d 的权重张量形状：

$$
(out\_channels,\frac{in\_channels}{groups},k)
$$

对于 Depthwise Conv：

$$
in\_channels=out\_channels=groups=C
$$

所以权重形状为：

$$
(C,1,k)
$$

权重参数量：

$$
\boxed{Parameters=C\times k}
$$

如果启用 Bias，还需要额外增加 C 个偏置参数：

$$
Parameters=C\times k+C
$$

例如：

$$
C=128,\quad k=31
$$

不含 Bias：

$$
128\times31=3968
$$

含 Bias：

$$
128\times31+128=4096
$$

本项目使用：

```python
bias=False
```

因此参数量为 3968。

### 4.5 Padding 与时间维度

卷积操作可能导致输出时间长度发生变化，而 Conformer ConvModule 通常需要保持时间维度不变，以便与输入进行残差相加。

Conv1d 的输出长度公式：

$$
L_{out}=
\left\lfloor
\frac{L_{in}+2p-d(k-1)-1}{s}+1
\right\rfloor
$$

其中：

- L_in：输入长度。
- p：Padding。
- d：Dilation。
- k：Kernel Size。
- s：Stride。

在标准设置下：

$$
s=1,\quad d=1
$$

对于奇数卷积核，设置：

$$
p=\frac{k-1}{2}
$$

即可保持时间长度不变：

$$
L_{out}=L_{in}
$$

例如：

$$
T=100,\quad k=31,\quad p=15
$$

则：

$$
L_{out}=100
$$

奇数卷积核能够进行左右对称 Padding，使卷积窗口以当前时间位置为中心。

因此，本项目限制 Kernel Size 必须为奇数。

## 5. BatchNorm、激活函数与最后的 Pointwise Conv

### 5.1 为什么使用 BatchNorm1d？

Depthwise Conv 完成局部时间特征提取后，使用 BatchNorm1d 对卷积输出进行归一化。

其主要作用：

- 对不同通道的特征进行归一化。
- 稳定特征分布。
- 帮助模型训练。

对于输入形状：

$$
X\in\mathbb R^{B\times C\times T}
$$

BatchNorm1d 针对每个通道，在 Batch 和时间位置上统计均值与方差。

在 PyTorch 中：

```python
nn.BatchNorm1d(C)
```

注意，BatchNorm1d 的输入格式为 `(B,C,T)`，因此这里不需要额外转置。

### 5.2 Swish / SiLU 激活函数

Conformer 使用 Swish 激活函数。

定义：

$$
Swish(x)=x\cdot\sigma(x)
$$

其中：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Swish 和 SiLU 是同一个激活函数。

在 PyTorch 中：

```python
nn.SiLU()
```

其作用是引入非线性变换，增强模型学习复杂特征关系的能力。

与 ReLU 不同，SiLU 不会直接将所有负数截断为零，而是通过 Sigmoid 对输入进行平滑调节。

### 5.3 第二个 Pointwise Conv 的作用

经过 Depthwise Conv 后，各通道主要完成了独立的局部时间特征提取。

此时需要进一步融合不同通道的信息。

因此，Conformer 使用第二个 Pointwise Conv：

```python
nn.Conv1d(
    in_channels=C,
    out_channels=C,
    kernel_size=1
)
```

其主要作用：

- 对不同通道的特征进行线性组合。
- 实现跨通道信息融合。
- 保持通道维度不变。

整个卷积模块中：

- 第一个 Pointwise Conv：扩展通道，为 GLU 做准备。
- Depthwise Conv：提取各通道的局部时间特征。
- 第二个 Pointwise Conv：重新融合不同通道的信息。

### 5.4 Dropout 的作用

Dropout 是一种正则化方法。

训练过程中随机将部分特征元素置零，降低模型对某些特征的过度依赖，从而减轻过拟合。

```python
nn.Dropout(p=dropout)
```

其中 p 表示随机置零的概率。

在推理阶段，Dropout 自动关闭。

## 6. B/T/C 转置与完整数据流

### 6.1 为什么需要 transpose(1, 2)？

Transformer 通常使用：

$$
(B,T,C)
$$

其中 T 表示时间长度，C 表示特征维度。

而 PyTorch Conv1d 要求：

$$
(B,C,T)
$$

其输入格式为：

```text
(batch, channels, length)
```

因此需要交换时间维度和通道维度：

```python
x = x.transpose(1, 2)
```

即：

$$
(B,T,C)\rightarrow(B,C,T)
$$

例如：

$$
(2,100,64)\rightarrow(2,64,100)
$$

### 6.2 ConvModule 完整张量形状变化

假设：

$$
B=2,\quad T=100,\quad C=64,\quad k=31
$$

| 操作 | 输出形状 |
|---|---|
| Input | (2,100,64) |
| LayerNorm | (2,100,64) |
| Transpose | (2,64,100) |
| Pointwise Conv1 | (2,128,100) |
| GLU | (2,64,100) |
| Depthwise Conv | (2,64,100) |
| BatchNorm | (2,64,100) |
| SiLU | (2,64,100) |
| Pointwise Conv2 | (2,64,100) |
| Dropout | (2,64,100) |
| Transpose | (2,100,64) |
| Output | (2,100,64) |

### 6.3 为什么最终需要转置回来？

因为 Conformer Block 中的其他模块，例如 FFN 和 MHSA，通常使用 `(B,T,C)` 格式。

同时，外层残差连接要求输入和输出具有相同的形状。

因此，卷积模块结束后，需要将：

$$
(B,C,T)\rightarrow(B,T,C)
$$

恢复为原来的张量格式。

### 6.4 残差连接在哪里实现？

当前项目将 `ConformerConvModule` 实现为一个独立的特征变换模块，只负责计算：

$$
F_{\text{conv}}(x)
$$

残差连接由外层 ConformerBlock 负责：

$$
\boxed{y=x+F_{\text{conv}}(x)}
$$

这样可以将特征变换和残差更新分开，明确模块职责，也避免重复进行残差相加。

## 7. 工程实现与验收

### 7.1 文件结构

本次实现文件：

```text
speech-ai-learning/
└── src/
    └── conformer.py
```

主要实现：

```python
class ConformerConvModule(nn.Module):
```

初始化参数：

- dim：输入特征维度。
- kernel_size：Depthwise Conv 的卷积核大小。
- dropout：Dropout 概率。

输入和输出均为 `(B,T,C)`。

### 7.2 核心代码实现

```python
import torch
import torch.nn as nn


class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.layernorm = nn.LayerNorm(dim)

        self.pointwise1 = nn.Conv1d(
            dim, 2 * dim, kernel_size=1
        )

        self.glu = nn.GLU(dim=1)

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False
        )

        self.batchnorm = nn.BatchNorm1d(dim)
        self.silu = nn.SiLU()

        self.pointwise2 = nn.Conv1d(
            dim, dim, kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LayerNorm
        x = self.layernorm(x)

        # (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)

        # Pointwise Conv + GLU
        x = self.pointwise1(x)
        x = self.glu(x)

        # Depthwise Conv
        x = self.depthwise(x)

        # BatchNorm + SiLU
        x = self.batchnorm(x)
        x = self.silu(x)

        # Pointwise Conv + Dropout
        x = self.pointwise2(x)
        x = self.dropout(x)

        # (B,C,T) -> (B,T,C)
        x = x.transpose(1, 2)

        return x
```

### 7.3 Shape 测试

测试输入：

```python
x = torch.randn(2, 100, 64)
y = model(x)
```

预期：

```text
Input shape: torch.Size([2, 100, 64])
Output shape: torch.Size([2, 100, 64])
```

实际测试结果与预期一致，说明输入输出维度保持不变。

### 7.4 Gradient 测试

测试代码：

```python
x = torch.randn(
    2, 100, 64,
    requires_grad=True
)

y = model(x)
loss = y.mean()
loss.backward()

assert x.grad is not None
```

实际结果：

```text
Input gradient exists: True
```

说明梯度能够从输出反向传播到输入，当前模块的计算图正常。

### 7.5 参数量测试

Depthwise Conv：

```python
depthwise = model.depthwise

print(depthwise.weight.shape)
print(depthwise.groups)

params = sum(
    p.numel()
    for p in depthwise.parameters()
)

print(params)
```

实际结果：

```text
Weight shape: torch.Size([64, 1, 31])
Groups: 64
Parameters: 1984
```

验证：

$$
64\times1\times31=1984
$$

与理论计算结果一致。

另外，Padding 应在不同奇数卷积核下测试，例如 3、15、31，验证输出时间长度均与输入一致。

### 7.6 今日总结

本次完成了 Conformer Convolution Module 的独立实现。

掌握的核心内容：

1. 理解 Pointwise Conv 通过 1×1 卷积进行通道维度变换。
2. 理解 GLU 通过 Sigmoid 门控调节特征信息。
3. 理解 Depthwise Conv 通过 groups=C 实现各通道独立卷积。
4. 掌握 Depthwise Conv 参数量的计算方法。
5. 理解 Padding 如何保持时间维度不变。
6. 掌握 BatchNorm1d、SiLU 和 Dropout 的作用。
7. 理解 B/T/C 转置的原因。
8. 完成 ConvModule 的 PyTorch 实现，并通过 Shape、Gradient 和参数量测试。

## 8. 自测问题

1. 为什么 Conformer ConvModule 开始时要将输入从 `(B,T,C)` 转置成 `(B,C,T)`？为什么最后还要转置回来？

   因为 Transformer 通常使用 `(B,T,C)`，而 PyTorch Conv1d 要求 `(B,C,T)`。最后转置回来是为了恢复后续模块使用的格式，并满足外层残差连接的形状要求。

2. 第一个 Pointwise Conv 为什么将通道从 C 扩展到 2C？GLU 为什么能够将 2C 再变回 C？

   因为 GLU 需要沿通道维度将输入分成 A、B 两部分，每部分为 C 个通道。通过 A 与 Sigmoid(B) 逐元素相乘，最终输出恢复为 C 个通道。

3. GLU 的计算公式是什么？A 和 B 分别起什么作用？

   $$GLU(A,B)=A\otimes\sigma(B)$$

   A 负责承载特征内容，B 负责生成门控权重。

4. Depthwise Conv 中 `groups=C` 究竟是什么意思？它和普通 Conv1d 有什么区别？

   表示将输入通道划分为 C 组，每组包含一个通道，每个通道独立进行卷积，不发生跨通道混合。普通 Conv1d 则可以融合不同输入通道的信息。

5. 假设 C=128，kernel_size=31，Depthwise Conv 的权重参数量是多少？如果 bias=True，又是多少？

   不含 Bias：

   $$128\times31=3968$$

   含 Bias：

   $$128\times31+128=4096$$

6. 假设输入时间长度 T=100，kernel_size=31，padding=15，stride=1，输出时间长度是多少？为什么？

   输出长度为 100。因为卷积核为奇数，左右各 Padding 15 个位置，在步长为 1、膨胀率为 1 时，输入输出时间长度相同。

7. Depthwise Conv 之后为什么还需要第二个 Pointwise Conv？

   因为 Depthwise Conv 主要负责各通道独立的局部时间特征提取，不能充分融合不同通道的信息。第二个 Pointwise Conv 用于重新进行通道维度的信息混合。

8. BatchNorm1d、SiLU 和 Dropout 分别起什么作用？

   - BatchNorm1d：对每个通道进行批归一化，稳定特征分布。
   - SiLU：引入非线性变换，增强模型表达能力。
   - Dropout：训练时随机将部分元素置零，降低模型对某些特征的过度依赖，减轻过拟合。

9. 为什么建议将残差连接放在 ConformerBlock 外部，而不是在当前独立的 ConvModule 内部和外部重复相加？

   因为 ConvModule 负责特征变换，外层 ConformerBlock 负责残差更新。将两者分离可以明确模块职责，提高独立性和复用性，同时避免重复进行残差相加。