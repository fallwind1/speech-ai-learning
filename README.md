# Speech AI Learning

This repository records my learning process in Speech AI.

## Goals

- Learn basic speech signal processing
- Learn ASR fundamentals
- Learn PyTorch
- Build Speech AI projects
- Prepare for Speech AI internships

## Learning Log

- W1D1: Set up Python, Git and VS Code.


## Weekly Reviews

### Week 1 — Python / NumPy / PyTorch 基础与第一个训练循环

#### 本周完成

本周完成了从 Python 基础到 PyTorch 第一个完整训练循环的学习，并实现了一个简单线性回归实验。

主要代码与产出：

- `src/word_count.py`：Python 文件读取、字符串处理、正则表达式和 JSON 输出练习。
- `notebooks/numpy_basics.ipynb`：NumPy ndarray、shape、广播、矩阵乘法、均值/方差和稳定 Softmax。
- `notebooks/autograd.ipynb`：PyTorch Tensor、`requires_grad`、计算图、`backward()`、梯度累积、`detach()` 和 `torch.no_grad()`。
- `src/linear_regression.py`：使用 `nn.Linear(1,1)`、MSELoss 和 SGD 拟合线性关系 `y = 2x + 4`。
- `figures/w01_loss.png`：正常训练、遗漏 `zero_grad()` 和修复后的 loss / prediction 对照结果。

#### 本周核心结果

完成了第一个完整 PyTorch 训练闭环：

`zero_grad → forward → loss → backward → step`

其中：

- `optimizer.zero_grad()`：清除上一轮保存在参数 `.grad` 中的梯度。
- `model(x)`：使用当前模型参数进行前向计算，得到预测值。
- `loss_fn(pred, y)`：计算预测值与目标值之间的损失。
- `loss.backward()`：沿计算图反向传播，将梯度累加到模型参数的 `.grad`。
- `optimizer.step()`：根据梯度和 learning rate 更新模型参数。

在线性回归实验中，loss 整体下降，模型学习到的参数逐渐接近真实关系：

- 真实 weight：`2`
- 真实 bias：`4`
- 实测最终 weight：`2.0030`
- 实测最终 bias：`3.9891`
- 初始 loss：`50.337280`
- 最终 loss：`0.000037`

Day 6 进一步通过故意遗漏 `optimizer.zero_grad()` 验证了梯度累积机制：`backward()` 默认将新梯度累加到已有 `.grad`，而 `optimizer.step()` 不会自动清除梯度。

#### 目前还不熟悉 / 需要继续巩固

- `Python 文件读取及相关文件操作还不够熟练。`
- `JSON 数据的读取、组织和输出还需要继续练习。`
- `NumPy / Tensor 的矩阵广播规则还不够熟悉，尤其需要加强对 shape 变化的判断。`

#### 本周实际投入

实际学习时间：`16 小时`

本周没有逐日精确统计实际学习时长，因此该数字为估计值。后续尽量记录每日实际学习时间，以便在周复盘时比较计划时间与实际时间。

#### 下一周三件事

1. `**补齐深度学习训练原理**：学习链式法则、负对数似然与交叉熵，理解 logits、概率、loss 与反向传播之间的关系。`
2. `**建立完整的数据与训练流程**：掌握 Dataset / DataLoader、train/val 划分，并使用两层 MLP 独立完成 MNIST 分类训练。`
3. `**建立可复用训练工程能力**：实现 checkpoint 保存与恢复、随机种子和可复现配置，并开始练习错误分析、命令行运行和 Linux 基础。`


### Week 2：深度学习基础 / 训练工程

#### 1. 本周完成内容

- 复习并理解计算图、链式法则、backward()与梯度累积机制
- 学习Dataset / DataLoader，完成 MNIST 小数据集构建、train/val 划分以及 batch 数据加载。
- 从基础训练循环扩展到两层 MLP 分类模型，使用 CrossEntropyLoss 完成 MNIST train/validation。
- 学习 checkpoint 的保存与恢复，理解 model state、optimizer state、epoch、config、seed 与 RNG state 的作用，以及可复现性的边界。
- 完成无 weight decay / 有 weight decay 的单变量对照实验，并根据 train/val 轨迹分析实验结果。
- 学习基础错误分析方法，能够根据 true label、pred label、confidence 分析错分样例。
- 学习 Linux 基础命令、CWD、绝对/相对路径、进程、日志重定向以及稳定项目路径。
- 学习 argparse，能够通过命令行传入训练超参数。
- 在阶段验收中从空白文件重新写出最小 MNIST 训练循环，并成功运行、生成 train/val loss 曲线。

#### 2. 主要代码与产出

| 文件 / 产出 | 内容 |
|---|---|
| `src/dataset_dataloader.py` |MNIST 小数据集、train/val 划分、Dataset 与 DataLoader|
| `src/train_mnist.py` |两层 MLP、CrossEntropyLoss、train/validation 训练流程|
| `src/train_template.py` |checkpoint 保存/加载、恢复训练、seed/RNG 与可复现实验|
| `src/train_regularization.py` |no_decay 与 weight_decay 对照实验|
| `src/train_cli.py` |稳定项目路径、集中配置以及 argparse 命令行参数|
| `src/error_analysis.py` |加载模型并寻找错分样例，记录 true/pred/confidence|
| `src/w02_stage_check.py` |Week 2 阶段验收：从空白重新构建最小训练循环|
| `results/w02_regularization.csv` |两组 weight decay 实验每个 epoch 的 train/val 指标|
| `results/w02_misclassified.csv` |  |
| `figures/w02_stage_check.png` |阶段验收 train/val loss 曲线|

#### 3. 本周关键结果

##### 最小训练循环阶段验收

- 初始 train loss：2.2709
- 最终 train loss：2.2538
- 初始 val loss：0.9786
- 最终 val loss：1.0453

结论：

> train loss 与 val loss 均稳定下降，说明重新编写的最小训练循环能够正常学习有效信号；当前没有观察到明显、持续的过拟合现象。Week 2 的基础训练循环阶段验收通过。

##### Weight Decay 对照实验

观察：

> 实验中，两组train/val曲线基本重合

结论：

> weight_decay=1e-4没有表现出明显的验证集泛化改善

##### 错误分析

完成：

> 先利用cli传入参数训练模型，然后加载训练后的模型进行一轮验证，取其中错误的5个样本进行错分

---

#### 4. 本周真正理解的内容

我现在能够解释：

- Dataset 与 DataLoader 的职责，以及 train / val / test 的区别。
- MLP 中 Flatten → Linear → ReLU → Linear 的 shape 变化。
- logits 与 probability 的区别，以及为什么 CrossEntropyLoss 直接接收 logits。
- model.train()、model.eval() 和 torch.no_grad() 分别解决什么问题。
- optimizer.zero_grad() → forward → loss → backward → optimizer.step() 的完整训练链路。
- PyTorch 为什么会产生梯度累积，以及无意梯度累积和有意 gradient accumulation 的区别。
- checkpoint 为什么不仅需要保存 model state，还需要 optimizer state、epoch、config 等训练状态。
- seed 与 RNG state 的区别，以及为什么固定 seed 仍不能保证不同硬件和软件环境完全逐位一致。
- 如何通过 train/val 曲线判断模型是否可能出现过拟合，而不是只看最终 accuracy。
- CWD、绝对路径、相对路径与 __file__ 的区别。
- 如何通过 argparse 从命令行传入 lr、batch_size、epochs 等训练参数。
- Linux 中 pwd、ls、cd、ps、grep、> / >> / 2>&1 / & 等基础命令和符号的作用。

---

#### 5. 当前不会 / 薄弱点

1. detach()、计算图生命周期以及部分 autograd 细节仍需要通过后续代码继续巩固。
2. 有意 gradient accumulation 的实际训练代码还没有独立实践，目前主要掌握原理。
3. Linux 与服务器训练目前以基础命令和概念为主，实际远程服务器、后台长时间训练和进程管理经验仍较少。
4. 错误分析目前只掌握基础方法，后续还需要学习如何从少量样例分析扩展到更系统的错误统计。
5. 训练工程代码虽然已经能够编写，但仍需要继续提高代码组织、配置管理和实验复现的熟练度。

---

#### 6. 实际学习时间

> 约 18 小时

---

#### 7. 下一周三件事

1. 在进入新知识的同时保持最小训练循环熟练度，确保能够继续独立写出 zero_grad → forward → loss → backward → step 主链。
2. 按 Week 3 主线完成新的核心学习任务，并优先保证 MUST 内容，不为了扩展知识挤占主线时间。
3. 如果后续任务再次暴露 Dataset/DataLoader、autograd、训练循环或 validation 的基础问题，优先使用 Week 3 的选做时间回补，而不是带着基础漏洞继续向后推进。


### Week 3：Transformer 基础 / Attention 与 Block 实现


#### 1. 本周完成内容

- 学习 Self-Attention 的基本思想，理解 Q / K / V 的作用以及 Attention 中相关性计算、权重分配和信息聚合过程。
- 手写实现 Scaled Dot-Product Attention，完成 QK^T、缩放因子、Softmax、Weighted Sum 的完整流程，并验证输出 shape 与数值稳定性。
- 学习 Multi-Head Self-Attention 的基本思想，理解不同 head 通过不同投影矩阵学习不同子空间中的关系。
- 学习 Causal Mask 与 Padding Mask 的作用，理解二者解决的问题不同，并完成 mask 后 Attention 行为验证。
- 完成 Attention 测试，包括未来 token 修改不影响过去输出、padding token 不影响有效输出以及全 mask 情况下避免 NaN。
- 学习 LayerNorm 的计算过程，手写 LayerNorm 并与 PyTorch nn.LayerNorm 对比验证。
- 理解 Residual Connection 的作用，包括信息保留和梯度传播路径。
- 学习 Pre-LN 与 Post-LN Transformer Block 的区别，理解 Pre-LN 对训练稳定性的帮助以及 warmup 与 Pre-LN 的不同作用。
- 学习 Encoder、Decoder、Encoder-Decoder Transformer 三种结构区别，理解双向编码、因果生成以及 Cross-Attention。
- 结合 Whisper 理解 Encoder-Decoder 架构，明确 Encoder 输出 audio representations，以及 Cross-Attention 中 Q/K/V 的来源。
- 学习 Transformer Block 结构，使用 PyTorch MultiheadAttention、自写 FFN、Residual 和 LayerNorm 组合一个简化 Transformer Block。
- 完成 Transformer Block 的 forward 与 backward 验证，检查 Attention、FFN、LayerNorm 参数是否获得梯度。
- 学习 Attention 复杂度分析，理解 Self-Attention 为什么具有 O(T²d) 的计算复杂度，以及长上下文模型面临的计算和显存问题。
- 学习 KV Cache 的基本思想，理解自回归生成过程中缓存历史 token 的 K/V 可以避免重复计算。


---

#### 2. 主要代码与产出

| 文件 / 产出 | 内容 |
|---|---|
| `src/attention.py` | Scaled Dot-Product Attention 实现，包括 QK^T、Scaling、Softmax、Weighted Sum |
| `tests/test_attention.py` | Attention mask 测试，包括 Causal Mask、Padding Mask、全 Mask 情况 |
| `src/norm.py` | 手写 LayerNorm，并与 PyTorch nn.LayerNorm 对比 |
| `src/transformer_block.py` | 使用 MultiheadAttention、FFN、Residual、LayerNorm 实现简化 Transformer Block |
| Attention 实验结果 | 比较 scaling 前后 Attention entropy，验证 scaling 对 softmax 分布的影响 |
| LayerNorm 对照日志 | 手写 LayerNorm 与 PyTorch LayerNorm 最大误差约 10^-7 |
| Transformer Block 验证结果 | 验证 forward 输出 shape、attention shape 以及 backward 后参数梯度 |


---

#### 3. 本周关键结果


##### **Scaled Dot-Product Attention 实现**

完成：

\[Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V\]


验证：

- Q/K/V shape 推导正确。
- Attention weight 每行和为1。
- 输出与权重不存在 NaN 或 Inf。


Scaling 实验：

结果：

- scaled entropy：1.128677
- no_scaled entropy：0.290982


结论：

> Scaling 可以降低 QK^T 数值范围，避免 Softmax 进入过饱和区域，使注意力分布更加平滑，有助于梯度稳定。


---

##### **Mask 测试**

完成：

- Causal Mask 未来位置屏蔽。
- 修改未来 token 不影响过去位置输出。
- Padding token 不参与有效 token 的 Attention。
- 全 Mask 行不会产生 NaN。


结论：

> Mask 机制能够正确控制 token 之间的信息可见范围。


---

##### **LayerNorm 实现**

完成：

- 手写 LayerNorm。
- 与 PyTorch nn.LayerNorm 进行结果对比。


实验结果：

最大绝对误差：

\[
2.3842\times10^{-7}
\]


结论：

> 手写 LayerNorm 与 PyTorch 实现一致，理解了 LayerNorm 沿 hidden dimension 进行归一化以及 gamma、beta 可学习参数的作用。


---

##### **Transformer Block 验证**

完成：

结构：

LayerNorm

↓

Multi-Head Self-Attention

↓

Residual

↓

LayerNorm

↓

FFN

↓

Residual


实验：

输入：

\[
(2,5,32)
\]


输出：

\[
(2,5,32)
\]


结果：

- Attention 参数获得梯度。
- FFN 参数获得梯度。
- LayerNorm 参数获得梯度。


结论：

> Transformer Block 可以正确建立计算图并完成反向传播验证。


---

#### 4. 本周真正理解的内容

我现在能够解释：

- Self-Attention 为什么需要 Q/K/V，以及 Q/K/V 分别承担匹配关系和信息传递的作用。
- Attention 中 QK^T 计算 token 之间关系，Softmax 将分数转换为权重，最终利用 V 进行信息聚合。
- 为什么需要 Scaling，以及 Scaling 和 Stable Softmax 分别解决的问题不同。
- Multi-Head Attention 为什么需要多个 head，以及不同 head 可以学习不同子空间中的关系。
- Causal Mask 和 Padding Mask 的区别，以及为什么二者通常需要组合使用。
- LayerNorm 的归一化维度、计算方式以及 gamma、beta 参数的作用。
- Residual Connection 如何提供梯度传播路径并保留原始信息。
- Pre-LN 与 Post-LN 的区别，以及为什么 Pre-LN 更容易训练深层 Transformer。
- Encoder 用于理解输入序列，Decoder 用于自回归生成，Encoder-Decoder 结构通过 Cross-Attention 建立输入输出关系。
- Whisper 中 Encoder 处理音频得到 audio representations，Decoder 通过 Cross-Attention 使用 Encoder 信息生成文本。
- Transformer Block 中 Attention 负责 token 间信息交互，FFN 负责 token 内部非线性特征变换。
- Attention 的计算瓶颈来自 QK^T，因此 Self-Attention 具有 O(T²d) 复杂度。
- KV Cache 通过缓存历史 token 的 K/V，减少 GPT 推理过程中的重复计算。


---

#### 5. 当前不会 / 薄弱点

1\. Multi-Head Attention 内部 Q/K/V 投影、多个 head 拆分与拼接过程还没有手写实现，目前主要使用 PyTorch nn.MultiheadAttention。

2\. Transformer Block 当前只完成结构验证，没有进行真实语言任务训练，因此 Attention 权重不能用于语义解释。

3\. 对 Transformer 中位置编码不同方案（Sinusoidal、Learnable、RoPE 等）的具体实现还没有深入学习。

4\. Attention 长序列优化方法（Flash Attention、Sparse Attention 等）目前只了解基本动机，没有进行代码实践。

5\. KV Cache 当前主要理解思想，尚未结合真实 LLM 推理代码实现。

6\. Encoder-Decoder 模型与 Decoder-only 模型在实际大模型中的工程差异仍需要结合 Whisper、GPT 等模型进一步学习。


---

#### 6. 实际学习时间

> 约 16 小时


---

#### 7. 下一周三件事

1\. 继续保持 Transformer 基础代码熟练度，能够独立解释 Attention、Mask、Block 的完整数据流。

2\. 按 Week 4 主线学习后续模型结构与训练方法，优先完成 MUST 内容，不为了扩展知识影响主线推进。

3\. 在后续学习中继续补充 Transformer 工程细节，例如模型配置管理、训练流程和真实模型代码阅读能力。