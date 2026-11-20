# W9D3：Conformer Block 集成

## 1. 今日目标

- 复用 W3 已实现的 Scaled Dot-Product Attention，构建简化版 MHSA。
- 复用 W9D2 的 `ConformerConvModule`。
- 实现包含两个 Half-step FFN、MHSA、ConvModule 和 Final LayerNorm 的简化 Conformer Block。
- 验证整个 Block 的前向传播、反向传播及输入输出形状。
- 理解 Padding Mask 在 MHSA 中的作用，以及 Padding 对卷积和 BatchNorm 的影响。
- 明确简化实现与 R08 Conformer 原论文的区别，不进行完整 ASR 训练与收敛对比。

## 2. Conformer Block 整体结构

### 2.1 模块顺序

**完整结构：**

```text
Input (B,T,C)
    ↓
1/2 FFN + Residual
    ↓
MHSA + Residual
    ↓
Convolution Module + Residual
    ↓
1/2 FFN + Residual
    ↓
Final LayerNorm
    ↓
Output (B,T,C)
```

**各模块的作用：**

- 第一半步 FFN：进行通道维度上的非线性特征变换。
- MHSA：通过多头自注意力机制建模全局上下文关系。
- Convolution Module：提取局部声学特征和短时动态模式。
- 第二半步 FFN：对融合后的特征进一步进行非线性变换。
- Final LayerNorm：规范化 Block 输出，帮助稳定训练。

Conformer 通过 MHSA 和卷积分别建模全局依赖与局部声学模式，并使用 Macaron-style FFN 增强特征表示能力。

### 2.2 残差连接与半步 FFN

**第一半步 FFN：**

$$
x_1=x+\frac12 FFN_1(x)
$$

FFN 的输出先乘以 0.5，再与输入进行残差相加。

**MHSA：**

$$
x_2=x_1+MHSA(x_1)
$$

通过 MHSA 建模全局上下文关系，并保留输入特征。

**Convolution Module：**

$$
x_3=x_2+ConvModule(x_2)
$$

卷积模块对 MHSA 处理后的特征进行局部建模，通过残差连接保留原有信息。

**第二半步 FFN：**

$$
x_4=x_3+\frac12 FFN_2(x_3)
$$

第二个 FFN 与第一个 FFN 使用独立参数，同样采用半步残差更新。

**Final LayerNorm：**

$$
\boxed{y=LayerNorm(x_4)}
$$

注意：残差连接发生在外层 ConformerBlock 中，各个子模块本身主要负责计算特征变换。

### 2.3 输入输出形状

**输入：**

$$
X\in\mathbb R^{B\times T\times C}
$$

其中：

- B：Batch Size。
- T：时间帧数量。
- C：隐藏特征维度。

**输出：**

$$
Y\in\mathbb R^{B\times T\times C}
$$

**为什么各模块需要保持相同的形状：**

因为残差连接需要将模块输入与输出逐元素相加：

$$
Y=X+F(X)
$$

所以两者必须具有兼容的形状。本项目中统一保持 `(B,T,C)`，避免意外广播，也方便多个 Conformer Block 堆叠。

FFN 可以在内部进行 `C → 4C → C` 的维度变换，MHSA 可以在内部拆分多个 Head，ConvModule 可以在内部转换为 `(B,C,T)`，但最终都必须恢复到 `(B,T,C)`。

## 3. 复用 W3 Attention

### 3.1 W3 Attention 的接口

**使用的文件：**

`src/attention.py`

**使用的类或函数：**

```python
dot_product_attention(
    Q,
    K,
    V,
    scale=True,
    valid_mask=None
)

make_padding_mask(lengths, max_len)
```

其中 `dot_product_attention` 是 W3 已经实现的 Scaled Dot-Product Attention，而不是完整的 MHSA。

本次在此基础上新增 `ConformerMHSA`，负责 Q/K/V 投影、多头拆分与合并以及输出投影。

**输入：**

设 Head 数量为 H，每个 Head 的维度为 D：

$$
D=\frac{C}{H}
$$

输入特征：

$$
X\in\mathbb R^{B\times T\times C}
$$

经过三个独立的 Linear 投影：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

再进行多头拆分：

$$
(B,T,C)\rightarrow(B,H,T,D)
$$

**输出：**

每个 Head 独立计算 Attention 后，结果重新合并，并通过输出 Linear：

$$
(B,H,T,D)\rightarrow(B,T,C)
$$

最终输出与输入形状一致。

`dot_product_attention` 还会返回 Attention 权重：

$$
Weights\in\mathbb R^{B\times H\times T\times T}
$$

**Mask 参数：**

W3 的 `valid_mask` 采用以下约定：

- True：允许访问。
- False：禁止访问。

在 MHSA 中，需要将 `(B,T)` 的 Mask 扩展为：

```python
attn_mask = valid_mask[:, None, None, :]
```

形状变化：

$$
(B,T)\rightarrow(B,1,1,T)
$$

使其能够广播到 `(B,H,T,T)` 的 Attention Scores。

### 3.2 Attention Mask

**Padding Mask 的作用：**

Padding Mask 用于屏蔽补齐出来的无效时间位置。

例如，Batch 中两条语音的有效长度为 100 和 80：

```python
lengths = torch.tensor([100, 80])

valid_mask = make_padding_mask(
    lengths,
    max_len=100
)
```

第二条语音的前 80 个位置为 True，后 20 个位置为 False。

将 Mask 应用于 Attention Scores 后，有效 Query 不会访问无效的 Padding Key/Value。

**Causal Mask 的作用：**

Causal Mask 用于禁止当前位置访问未来时间位置，通常用于自回归生成模型或需要限制未来信息访问的序列模型。

例如，当前位于第 3 个时间位置时，只允许访问第 1～3 个位置，不能访问之后的位置。

**本次 Conformer Block 是否需要 Causal Mask：**

不需要。

本次实现的是非流式、双向 Conformer Encoder Block，允许每个有效时间位置访问前后所有有效语音帧。

因此：

- 对存在 Padding 的 Batch，需要 Padding Mask。
- 当前非流式结构不需要 Causal Mask。

如果以后实现流式 ASR，则需要重新考虑因果注意力或有限右侧上下文约束。

### 3.3 标准 MHSA 与相对位置注意力

**本次采用的 Attention：**

普通 Multi-Head Self-Attention。

每个 Head 具有自己的 Q/K/V 投影参数，通过 Scaled Dot-Product Attention 计算权重，再合并多个 Head 的输出。

代码中虽然使用一个大的 `q_proj`，但不同 Head 对应其中不同的参数切片，并不是多个 Head 共享相同投影参数。

**原论文采用的 Attention：**

Conformer 原论文采用带有相对位置编码的 Multi-Head Self-Attention，其相对位置机制借鉴 Transformer-XL。

**两者的差异：**

普通 MHSA 主要通过 Q、K 的内容相关性计算 Attention：

$$
Attention(Q,K,V)
=
Softmax\left(
\frac{QK^T}{\sqrt{D}}
\right)V
$$

原论文在注意力计算中进一步引入相对位置信息，使模型能够利用不同时间位置之间的相对距离关系。

本次简化实现没有引入该机制，因此不能认为已经完全复现原论文的 MHSA。

## 4. Macaron FFN 与 Block 实现

### 4.1 FFN 内部结构

ConformerFFN 的结构为：

```text
Input (B,T,C)
    ↓
LayerNorm
    ↓
Linear (C → 4C)
    ↓
SiLU / Swish
    ↓
Dropout
    ↓
Linear (4C → C)
    ↓
Dropout
    ↓
Output (B,T,C)
```

**LayerNorm：**

对每个时间位置的隐藏特征进行归一化，帮助稳定特征分布。

**Linear：**

通过两次线性变换完成特征维度扩展与恢复：

$$
C\rightarrow4C\rightarrow C
$$

第一层增加中间特征维度，第二层映射回原维度，以便残差相加。

**SiLU：**

引入非线性变换：

$$
SiLU(x)=x\cdot\sigma(x)
$$

**Dropout：**

训练过程中随机将部分特征置零，起到正则化作用。

参考实现通常在两个 Linear 之后分别设置 Dropout。

本次初版 FFN 只实现了第一次 Dropout，代码检查时已提出需要补充第二次 Dropout。

### 4.2 Half-step Residual

**为什么乘以 0.5：**

Conformer 借鉴 Macaron-Net，将 FFN 分布在 Block 的前后两端，并采用半步残差权重。

计算形式：

$$
x'=x+\frac12 FFN(x)
$$

注意，这里的 0.5 作用于 FFN 分支输出，不是将隐藏维度减半。

**两个 FFN 是否共享参数：**

不共享。

使用两个独立的模块：

```python
self.ffn1 = ConformerFFN(...)
self.ffn2 = ConformerFFN(...)
```

两者具有独立的可训练参数，可以学习不同的特征变换。

### 4.3 完整 Block 的 forward

**主要计算步骤：**

```python
def forward(self, x, valid_mask=None):
    x = x + 0.5 * self.ffn1(x)
    x = x + self.mhsa(x, valid_mask)
    x = x + self.conv(x)
    x = x + 0.5 * self.ffn2(x)
    x = self.layernorm(x)
    return x
```

这与本次实际完成的 Block 主体一致。`valid_mask=None` 是建议采用的可选参数形式，便于兼容无 Padding 的输入。

**各步骤的形状变化：**

| 操作 | 输出形状 |
|---|---|
| Input | (B,T,C) |
| FFN1 + Residual | (B,T,C) |
| MHSA + Residual | (B,T,C) |
| ConvModule + Residual | (B,T,C) |
| FFN2 + Residual | (B,T,C) |
| Final LayerNorm | (B,T,C) |

各个子模块内部可以改变张量形状，但最终必须恢复到原来的形状。

### 4.4 ConvModule 的复用

**为什么不能重复添加内部残差：**

当前 `ConformerConvModule` 只负责特征变换：

$$
F_{\mathrm{conv}}(x)
$$

外层 ConformerBlock 负责残差更新：

$$
y=x+F_{\mathrm{conv}}(x)
$$

如果在 ConvModule 内部和 ConformerBlock 外部都进行残差相加，会导致原始输入被重复加入，从而改变预期的 Block 计算结构。

**卷积模块的输入输出格式：**

ConvModule 的输入和输出都是：

$$
(B,T,C)
$$

但 PyTorch Conv1d 要求输入为：

$$
(B,C,T)
$$

因此，在 ConvModule 内部需要进行两次转置：

```python
x = x.transpose(1, 2)
# Conv1d operations
x = x.transpose(1, 2)
```

## 5. 前向传播与反向传播测试

### 5.1 Shape 测试

**输入形状：**

$$
(2,100,64)
$$

其中：

- Batch Size = 2。
- 时间长度 = 100。
- Hidden Dim = 64。

**输出形状：**

$$
(2,100,64)
$$

**测试结果：**

实际运行输出：

```text
Input shape: torch.Size([2, 100, 64])
Output shape: torch.Size([2, 100, 64])
```

说明完整 ConformerBlock 的输入输出形状保持一致，Shape 测试通过。

### 5.2 Gradient 测试

**测试方法：**

创建需要梯度的输入：

```python
x = torch.randn(
    2, 100, 64,
    requires_grad=True
)

y = model(x, valid_mask)

loss = y.square().mean()
loss.backward()
```

通过检查输入梯度及模型参数梯度，判断反向传播是否正常。

**输入梯度：**

实际结果：

```text
Input gradient exists: True
```

说明梯度能够从 Block 输出反向传播到输入。

**模型参数梯度：**

可以使用：

```python
missing = [
    name
    for name, p in model.named_parameters()
    if p.requires_grad and p.grad is None
]

print(missing)
```

如果输出为空列表，说明所有可训练参数均获得梯度张量。

**测试结果：**

- Shape 测试：通过。
- 输入梯度存在：通过。
- 输出包含 NaN：False。
- 全部模型参数梯度检查：尚待记录实际结果。

需要注意：输入梯度存在并不能单独证明所有模型参数都获得了梯度。

### 5.3 参数量检查

**Block 总参数量：**

对于：

```python
dim = 64
num_heads = 4
ff_expansion = 4
kernel_size = 31
```

按照当前子模块结构及 Bias 配置，理论总参数量为：

$$
98048
$$

**各模块参数量：**

| 模块 | 理论可训练参数量 |
|---|---:|
| FFN1 | 33,216 |
| MHSA | 16,768 |
| ConvModule | 14,720 |
| FFN2 | 33,216 |
| Final LayerNorm | 128 |
| 总计 | 98,048 |

这些数值是根据代码结构计算的预期值，需要通过实际 `model.parameters()` 统计确认。

其中 Depthwise Conv 的权重参数量为：

$$
64\times31=1984
$$

不能将整个 Block 的参数量误认为 Depthwise Conv 的参数量。

**为什么本次不比较 ASR 收敛：**

本次 W9D3 的目标是验证 Conformer Block 的结构、Shape、Gradient 和 Mask 行为，并没有构建完整 ASR 训练系统。

完整 ASR 收敛比较还涉及数据、训练配置、损失函数、模型规模以及计算资源等因素，不属于本次 MUST 任务。

## 6. Padding Mask 与 padded 帧

### 6.1 Padding 的来源

**为什么需要 Padding：**

批量处理语音时，不同样本的时间长度通常不同。

例如，两条语音分别有 100 帧和 80 帧。为了组成统一形状的 Batch，需要将第二条语音补齐到 100 帧。

补齐出来的 20 帧不是有效语音信息，因此需要区分有效帧和 Padding 帧。

**Valid Mask：**

本项目使用 `valid_mask` 表示时间位置是否有效：

- True：有效时间位置。
- False：Padding 时间位置。

例如：

```python
lengths = torch.tensor([5, 3])
valid_mask = make_padding_mask(lengths, 5)
```

结果：

```text
[
    [True, True, True, True, True],
    [True, True, True, False, False]
]
```

**Padding Mask：**

Padding Mask 用于限制 Attention 对补齐位置的访问，防止有效 Query 读取无效 Key/Value。

需要区分 Padding Mask 和 Causal Mask：

- Padding Mask：屏蔽无效补齐帧。
- Causal Mask：屏蔽未来时间帧。

非流式 Conformer 不需要限制对未来有效帧的访问，但存在 Padding 时仍需要 Padding Mask。

### 6.2 Attention 对 Padding 的处理

**被 Mask 的 Key：**

在当前 W3 Attention 实现中：

```python
attn_mask = valid_mask[:, None, None, :]
```

Mask 作用于 Attention Scores 的 Key 维度。

无效 Key 对应的分数被设置为负无穷，Softmax 后得到零权重。

因此，有效 Query 不会读取 Padding Key/Value。

**被 Padding 的 Query：**

当前 Mask 没有对 Query 维度进行屏蔽。

因此，Padding 位置仍可以作为 Query，访问其他有效位置的 Key/Value，并产生 Attention 输出。

**为什么 Padding 位置的输出不一定为零：**

一方面，Padding Query 仍然能够关注有效位置。

另一方面，输出 Linear、FFN 和残差连接等操作也可能在 Padding 位置产生或保留非零特征。

因此：

**屏蔽 Padding Key/Value，并不等于将 Padding Query 的输出清零。**

### 6.3 ConvModule 对 Padding 的影响

**Depthwise Conv：**

Depthwise Conv 在时间维度上使用局部卷积窗口。

当 `kernel_size=31` 时，以当前帧为中心，左右各覆盖 15 帧。

假设某条语音有 80 个有效帧，Padding 从第 81 帧开始。

在一开始就存在非零 Padding 特征的情况下，第 66～80 个有效帧的卷积窗口可能覆盖 Padding 区域，因此其输出可能受到影响。

采用 Python 的零起始索引时，对应：

```python
x[65:80]
```

需要注意：如果 Padding 位置在卷积前已清零，它们不会通过自身的输入数值引入额外信号；但边界附近的卷积窗口仍包含零填充区域。

**BatchNorm：**

当前 ConvModule 使用：

```python
nn.BatchNorm1d(dim)
```

在训练模式下，对于输入 `(B,C,T)`，BatchNorm1d 按通道在 B、T 两个维度上统计均值和方差。

由于普通 BatchNorm1d 不知道哪些位置是 Padding，Padding 位置也会参与统计。

因此，即使某个有效帧远离 Padding 边界，也可能因为整体统计量变化而产生不同的归一化结果。

这与 Depthwise Conv 的局部影响不同：

- Depthwise Conv：影响局部时间邻域。
- BatchNorm：训练模式下可能通过统计量影响整个 Batch。

在 `eval()` 模式下，默认 BatchNorm1d 使用保存的运行统计量，不再根据当前 Batch 计算新的均值和方差。

**残差连接：**

残差计算：

$$
Y=X+F(X)
$$

残差连接不会自动判断某个时间位置是否为 Padding，也不会自动清零。

因此，Padding 位置经过各个子模块产生的非零特征，可能通过残差连接保留并继续传播。

### 6.4 简化实现的处理方式与局限

**采用的 Mask 约定：**

当前实现：

- `valid_mask=True`：有效位置。
- `valid_mask=False`：Padding 位置。
- MHSA 中只屏蔽无效 Key/Value。
- 不使用 Causal Mask。

**是否在输出端清零 padded 帧：**

当前已经实现的 ConformerBlock 尚未显式清零 Padding 输出。

如果需要，可以使用：

```python
def zero_padding(x, valid_mask):
    if valid_mask is None:
        return x

    return x.masked_fill(
        ~valid_mask.unsqueeze(-1),
        0.0
    )
```

这属于后续可采用的改进方式，而不是当前已经完成的功能。

**这种方法解决了什么：**

如果在合适的子模块边界进行清零：

1. 可以避免 Padding 位置的非零隐藏状态继续传入后续模块。
2. 可以减少非零 Padding 特征通过卷积影响边界附近有效帧的情况。
3. 可以保证清零操作后的 Padding 输出为零。

**仍然存在哪些局限：**

即使卷积输入中的 Padding 位置已经清零，Depthwise Conv 仍可能在靠近 Padding 区域的位置产生非零输出。

这些位置如果参与后续 BatchNorm 的统计，仍可能改变 BatchNorm 的均值和方差，进而影响有效位置。

此外，只在 ConvModule 输出端清零，不能撤销已经发生的 BatchNorm 统计影响。

因此，显式清零可以改善 Padding 隐藏状态的传播问题，但不能等同于完整的 Mask-aware Convolution 或 Masked BatchNorm。

本次 MUST 任务只要求能够解释上述行为，不要求实现完整的 Masked BatchNorm。

## 7. 与原论文的差异

### 7.1 相对位置编码

**原论文：**

R08 Conformer 采用基于 Transformer-XL 的相对位置注意力，将时间位置之间的相对距离信息引入 MHSA。

**本次实现：**

使用普通 Multi-Head Self-Attention，通过 W3 的 `dot_product_attention` 完成注意力计算。

没有实现相对位置编码。

**主要差异：**

原论文能够在 Attention 中显式利用相对位置信息；当前简化实现没有这部分机制。

因此，虽然模块顺序、Macaron-style FFN 和卷积模块的基本结构一致，但不能将当前版本称为原论文的完整复现。

此外，本次采用奇数卷积核 `kernel_size=31` 并使用对称 Padding，而原论文主要模型配置使用 `kernel_size=32`。本次 MHSA 也没有单独实现 Attention 权重 Dropout。

### 7.2 实验范围

**本次验证：**

- FFN、MHSA 和 ConvModule 的集成。
- 输入输出 Shape 是否保持一致。
- 整个 Block 是否能够反向传播。
- 输出是否包含 NaN。
- Padding Mask 的语义。
- Depthwise Conv 与 BatchNorm 对 Padding 的影响。
- 模型参数量的理论计算与待执行统计。

**本次不验证：**

- 相对位置注意力的完整实现。
- 从零训练完整 ASR 系统。
- ASR WER 与收敛速度比较。
- 流式 ASR 的因果或右上下文约束。
- 完整的 Padding-aware 卷积和归一化机制。

### 7.3 可能的后续改进

1. 进一步学习相对位置编码，理解原论文 MHSA 的完整计算。
2. 补充不同输入长度的测试，检查 Mask 行为。
3. 根据需要补充输出 Padding 清零机制。
4. 探索 Masked BatchNorm 或其他适合变长语音的处理方式。
5. 在后续学习中进一步理解 Conformer 的参数效率和模型复杂度。

其中，相对位置版本属于本周 SHOULD 任务，不是 W9D3 的 MUST 要求。

## 8. 工程产出与本日总结

### 8.1 文件

复用已有文件：

```text
src/attention.py
```

其中包含：

- `dot_product_attention`
- `make_padding_mask`

扩展已有文件：

```text
src/conformer.py
```

主要模块：

- `ConformerConvModule`
- `ConformerFFN`
- `ConformerMHSA`
- `ConformerBlock`

### 8.2 测试记录

**已确认的实际结果：**

```text
Input shape: torch.Size([2, 100, 64])
Output shape: torch.Size([2, 100, 64])
Input gradient exists: True
Contains NaN: False
```

**已完成的测试：**

- FFN Shape 测试。
- MHSA Shape 测试。
- ConformerBlock Shape 测试。
- Block 输入梯度存在性测试。
- Block 输出 NaN 检查。

**需要进一步记录的测试：**

- 所有可训练参数的梯度检查。
- MHSA Padding 不变性实验结果。
- Depthwise Conv Padding 边界实验结果。
- BatchNorm Padding 影响实验结果。
- Block 实际参数量统计。

这些尚未提供实际运行结果的项目，不能标记为已经通过。

### 8.3 本日总结

本次完成了简化 ConformerBlock 的集成，实现了：

$$
\boxed{
\frac12 FFN
\rightarrow MHSA
\rightarrow ConvModule
\rightarrow \frac12 FFN
\rightarrow LayerNorm
}
$$

主要收获：

1. 理解完整 MHSA 如何在 W3 Scaled Dot-Product Attention 基础上，通过 Q/K/V 投影、多头拆分和合并构建。
2. 理解两个独立 FFN 及其 Half-step Residual。
3. 能够将 FFN、MHSA 和 ConvModule 集成为一个完整 Block。
4. 理解 Padding Mask 与 Causal Mask 的区别。
5. 理解 Key Padding Mask 不能自动清零 Padding Query。
6. 理解 Depthwise Conv 对 Padding 的局部影响，以及 BatchNorm 对整个 Batch 的统计影响。
7. 明确简化版本与原论文相对位置注意力之间的差异。

## 9. 自测问题

**Q1. 为什么 FFN 内部可以将隐藏维度扩展到 4C，但最终必须恢复到 C？**

因为 FFN 的输出需要与输入进行残差相加，最终必须恢复到相同的隐藏维度。

**Q2. 为什么非流式 Conformer Encoder 需要 Padding Mask，却通常不需要 Causal Mask？**

因为非流式 Encoder 可以利用完整语音的前后上下文，不需要屏蔽未来有效帧；但同一个 Batch 中可能存在补齐的无效帧，需要通过 Padding Mask 防止有效 Query 读取无效 Key/Value。

**Q3. 使用 Key Padding Mask 后，整个 Block 的 Padding 输出是否一定为零？**

不一定。Padding Query 仍可能访问有效 Key/Value，且 Linear、卷积和残差连接等操作也可能使 Padding 位置产生非零特征。

**Q4. Padding Key 与 Padding Query 的区别是什么？**

Padding Key 是不能被有效 Query 读取的无效位置。Padding Query 是本身位于无效时间位置的 Query，在当前 Mask 约定下仍可以访问有效 Key/Value。

**Q5. Kernel Size=31 时，为什么靠近 Padding 边界的 15 个有效帧可能受到影响？**

因为卷积核以当前帧为中心，左右各覆盖 15 帧。边界附近的卷积窗口可能覆盖 Padding 区域，从而受到非零 Padding 特征的影响。

**Q6. BatchNorm 为什么会影响远离 Padding 边界的有效帧？**

因为训练模式下 BatchNorm1d 按通道在 Batch 和时间维度统计均值与方差，Padding 位置也参与统计，因此可能改变其他有效位置的归一化结果。

**Q7. MHSA 后清零 Padding 是否能够彻底解决 Padding 问题？**

不能。Depthwise Conv 仍可能在 Padding 区域产生非零特征，这些特征会参与 BatchNorm 的统计。需要额外处理才能严格排除无效位置对统计量的影响。

**Q8. 不使用 Padding Mask 会有什么问题？**

不使用 Padding Mask 通常不会导致矩阵形状错误，而是使有效 Query 将 Padding Key/Value 也纳入 Attention 计算，造成注意力权重分配和特征表示受到无效位置影响。

**Q9. 当前简化 ConformerBlock 与原论文最重要的区别是什么？**

当前版本采用普通 MHSA，没有实现原论文使用的相对位置注意力。因此它能验证 Conformer 的主要模块结构，但不是原论文的完整复现。