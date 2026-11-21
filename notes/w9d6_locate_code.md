
## 模块2：开源 ConformerConvModule 源码阅读

### 2.1 源码定位

- 开源项目：Hugging Face Transformers
- Transformers 版本：5.21.0
- 源码文件：transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
- 类名：Wav2Vec2ConformerConvolutionModule
- 源码行号：366～423（本地安装版本）

本地源码完整路径：

d:\anaconda3\envs\speech-ai\Lib\site-packages\transformers\models\wav2vec2_conformer\modeling_wav2vec2_conformer.py

定位方法：

```python
import inspect

from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConvolutionModule,
    Wav2Vec2ConformerEncoderLayer
)

print(inspect.getfile(
    Wav2Vec2ConformerConvolutionModule
))

print(inspect.getsource(
    Wav2Vec2ConformerConvolutionModule
))
```

### 2.2 初始化部分（__init__）

**1. 构造函数接收什么参数？**

接收统一的 config 配置对象。

主要使用的配置参数包括：

- config.hidden_size：隐藏维度。
- config.conv_depthwise_kernel_size：Depthwise Conv 的卷积核大小。
- config.hidden_act：激活函数名称。
- config.conformer_conv_dropout：卷积模块的 Dropout 概率。

相比自己的实现，开源实现通过配置对象统一管理模型参数。

**2. 是否进行了 Kernel Size 合法性检查？**

进行了检查：

```python
if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
    raise ValueError(...)
```

要求卷积核大小为奇数。

因为当前实现使用左右对称填充：

$$
P=\frac{K-1}{2}
$$

当 Stride=1、Dilation=1 时，奇数卷积核能够通过对称整数 Padding 保持时间长度不变。

如果设置偶数卷积核，则初始化阶段会抛出 ValueError。

**3. Pointwise Conv1 的配置：**

- in_channels：config.hidden_size，即 C。
- out_channels：2 * config.hidden_size，即 2C。
- kernel_size：1。
- bias：False。

Pointwise Conv1 将通道数从 C 扩展到 2C，为后续 GLU 提供输入。

与自己的实现相比，开源实现不使用 Bias。

**4. Depthwise Conv 的配置：**

- in_channels：C。
- out_channels：C。
- kernel_size：config.conv_depthwise_kernel_size。
- groups：C。
- padding：(kernel_size - 1) // 2。
- bias：False。

groups=C 表示各通道独立进行卷积，负责局部时间特征建模。

Stride=1，Dilation=1。

使用奇数卷积核与对称 Padding，使输入和输出的时间长度保持一致。

由于使用对称 Padding，因此该卷积不是严格因果卷积。

例如 K=31 时，当前位置的感受野为：

$$
[t-15,t+15]
$$

会使用未来 15 帧的信息。

**5. Pointwise Conv2 的配置：**

- in_channels：C。
- out_channels：C。
- bias：False。

Kernel Size=1，用于进行跨通道信息融合。

自己的 Pointwise Conv1 和 Pointwise Conv2 默认使用 Bias，而开源实现均不使用 Bias。

因此两者在 Bias 上的参数量差异为：

$$
\Delta P=2C+C=3C
$$

当 C=64 时：

$$
\Delta P=192
$$

**6. 激活函数和 Dropout 如何设置？**

激活函数：

```python
self.activation = ACT2FN[config.hidden_act]
```

根据配置文件获取对应的激活函数，而不是直接写死 SiLU。

优点是可以通过更换配置使用不同激活函数，提高模型代码的灵活性。

Dropout：

```python
self.dropout = nn.Dropout(
    config.conformer_conv_dropout
)
```

通过配置对象设置 Dropout 概率。

### 2.3 前向传播（forward）

按照源码执行顺序记录：

1. LayerNorm：对最后一个特征维度进行归一化，保持 (B,T,C)。
2. Transpose：将形状转换为 (B,C,T)。
3. Pointwise Conv1：将通道数扩展为 2C，得到 (B,2C,T)。
4. GLU：进行门控操作，将通道数恢复为 C，得到 (B,C,T)。
5. Depthwise Conv：进行局部时间建模，保持 (B,C,T)。
6. BatchNorm：对每个通道进行归一化，保持 (B,C,T)。
7. Activation：进行非线性变换，保持 (B,C,T)。
8. Pointwise Conv2 + Dropout：进行跨通道特征融合和随机失活，保持 (B,C,T)。
9. Transpose：将形状恢复为 (B,T,C)。

**输入和输出的 Shape：**

$$
X\in\mathbb{R}^{B\times T\times C}
$$

$$
Y\in\mathbb{R}^{B\times T\times C}
$$

整个 ConvModule 保持输入和输出的时间长度与隐藏维度不变。

**为什么需要 transpose？**

Conformer Block 使用的输入格式为 (B,T,C)，而 PyTorch Conv1d 要求输入格式为 (B,C,T)。

因此在进入卷积操作前需要交换时间维度和特征维度。

卷积计算完成后，再转置回 (B,T,C)，以便后续模块处理和残差相加。

### 2.4 EncoderLayer 中的调用

**ConvModule 在 Conformer Block 的哪个位置？**

位于 MHSA 和第二个 Half-step FFN 之间。

完整模块顺序：

$$
FFN_1\rightarrow MHSA\rightarrow ConvModule
\rightarrow FFN_2\rightarrow LayerNorm
$$

两个 FFN 均采用 Half-step Residual。

**残差连接在哪里实现？**

残差连接在 Wav2Vec2ConformerEncoderLayer.forward() 中实现。

源码：

```python
residual = hidden_states
hidden_states = self.conv_module(hidden_states)
hidden_states = residual + hidden_states
```

数学表达：

$$
Y=X+ConvModule(X)
$$

与自己的 ConformerBlock 实现一致：

```python
x = x + self.conv(x)
```

**ConvModule 本身是否包含残差连接？**

不包含。

ConvModule 仅负责卷积分支的特征变换。

EncoderLayer 负责子模块之间的连接和残差相加。

因此不应该在 ConvModule 内部重复添加残差连接。



## 模块3：开源实现与自己的实现对照

### 核心结构的一致性

自己的 ConformerConvModule 与 Transformers 5.21.0
中的 Wav2Vec2ConformerConvolutionModule 在核心结构、
执行顺序和张量形状变化上基本一致。

两者均采用：

LayerNorm -> Transpose -> Pointwise Conv1 -> GLU
-> Depthwise Conv -> BatchNorm -> Activation
-> Pointwise Conv2 -> Dropout -> Transpose

残差连接均由外部 Conformer Block 实现。

### 主要差异

1. 开源实现采用 config 对象统一管理模型配置，
   自己的实现通过构造函数直接传入参数。

2. 开源实现显式检查 Kernel Size 是否为奇数，
   避免对称 Padding 无法保持时间长度。

3. 开源实现中的两个 Pointwise Conv 均不使用 Bias，
   自己的实现默认使用 Bias。

4. 开源实现通过 ACT2FN 根据配置选择激活函数，
   自己的实现直接使用 SiLU。

### 参数量差异

在隐藏维度 C=64、Kernel Size=31、
使用无可训练参数的相同激活函数的条件下：

- 自己的 ConvModule：14720。
- 开源 ConvModule：14528。
- 参数量差异：192。

以上为根据结构计算得到的理论参数量，
未进行开源模块的实例化实测。

### 流式能力

两种实现均使用对称 Depthwise Conv，
因此都不满足严格因果约束。

如果需要严格因果卷积，应改为只在左侧填充，
并考虑归一化、Attention Mask 和状态缓存等问题。

### 源码阅读结论

自己的实现已经复现了 ConformerConvModule 的核心结构。

开源实现与自己的实现之间的差异主要集中在
参数配置管理、合法性检查、Bias 和激活函数选择等方面。

结构相似不代表模型性能完全相同。
本次源码阅读不进行 CER 或训练效果比较。
