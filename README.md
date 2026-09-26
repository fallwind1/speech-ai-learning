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