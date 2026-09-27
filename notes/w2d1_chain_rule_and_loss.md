# W2D1 链式法则与损失
| 模块 | 内容 | 时间 |
|---|---|---:|
| 1 | 计算图、局部导数与链式法则基础 | 20 min |
| 2 | 手算 `a*b+c` + PyTorch 验证 | 25 min |
| 3 | 条件概率与分类问题 | 15 min |
| 4 | Likelihood、log 与 NLL | 15 min |
| 5 | Cross Entropy 直觉与 logits | 20 min |
| 6 | 综合串联 + 最终验收 | 10 min |

## 1. 计算图与链式法则

### 1.1 核心概念（模板）

**计算图：**

> 把一个复杂表达式拆成若干简单运算，并记录各个变量之间依赖关系的图结构。

**节点：**

> 可以看成一种运算关系，存在输入和输出

**边 / 运算关系：**

> 

**局部导数：**

> 某一个运算节点的输出，对该节点输入的偏导数

**链式法则：**

> 如果一个变量通过中间变量影响最终输出，则最终输出对该变量的导数，等于沿路径各局部导数的乘积。


### 1.2 示例计算图

表达式：

\[
d = a \times b
\]

\[
L = d + c
\]

计算图：

```text
a ──┐
    × ── d ──┐
b ──┘        + ── L
         c ──┘
```

Forward：

```text
a =2
b =3
c =4

d = a*b =6
L = d+c =10
```

Backward：

```text
∂L/∂L =1

∂L/∂d =1
∂L/∂c =1

∂L/∂a =1*3=3
∂L/∂b =2
```


### 1.3 链式法则

如果：

\[
a \rightarrow d \rightarrow L
\]

则：

\[
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial a}
\]

我的理解：

> 与复合函数求导规则相同，抽丝剥茧，只是越靠近Loss的运算规则越在外层。


### 1.4 当前模块易错点

- 链式法则
- 反向传播的具体内容
- 


### 1.5 当前模块总结

> 当前梯度=上游梯度×局部导数

## 2. 手算计算图与 PyTorch Autograd 验证

### 2.1 实验表达式

\[
d = a \times b
\]

\[
L = d + c
\]

取值：

```text
a = 3
b = 4
c = 2
```

计算图：

```text
a ──┐
    × ── d ──┐
b ──┘        + ── L
         c ──┘
```


### 2.2 Forward 手算

```text
d = a × b = 12

L = d + c = 14
```


### 2.3 Backward 手算

反向传播起点：

\[
\frac{\partial L}{\partial L} = 1
\]

加法节点：

\[
\frac{\partial L}{\partial d} = 1
\]

\[
\frac{\partial L}{\partial c} = 1
\]

乘法节点：

\[
\frac{\partial d}{\partial a} = 4
\]

\[
\frac{\partial d}{\partial b} = 3
\]

使用链式法则：

\[
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial a}
= 4
\]

\[
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial b}
= 3
\]

最终：

```text
a.grad = 4
b.grad = 3
c.grad = 1
```


### 2.4 PyTorch 验证

```python
# 填写实际代码
import torch

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = torch.tensor(2.0, requires_grad=Ture)

d = a*b
L = d+c

L.backward()

print("a.grad=",a.grad)
print("b.grad =", b.grad)
print("c.grad =", c.grad)
```

实际输出：

```text
a.grad = 4
b.grad = 3
c.grad = 1
```

手算与 PyTorch：

> 输出结果相同，PyTorch输出为tensor


### 2.5 backward 的本质

我的理解：

> 利用forward建立的依赖关系和局部导数，把最终loss的梯度往回传播。


### 2.6 当前模块易错点

- 局部梯度≠最终梯度

## 3. 条件概率与分类概率

### 3.1 条件概率

条件概率表示：

> 以P(A|B)为例，即在已经知道B发生的条件下，A发生的概率。

公式：

\[
P(A|B)
=
\frac{P(A\cap B)}{P(B)}
\]

其中：

```text
P(A|B)：

P(A∩B)：

P(B)：
```


### 3.2 分类任务中的条件概率

分类模型希望估计：

\[
P(y=k|x)
\]

我的理解：

> 已知输入x的情况下，它属于类别k的概率


例如图片分类：

```text
输入 x：图片

可能类别：
- 猫：0.10
- 狗：0.80
- 鸟：0.10

模型输出：y=狗
```


### 3.3 正确类别概率

假设真实类别为：

```text
y = class 1
```

模型预测：

```text
P(class 0 | x) = 0.05
P(class 1 | x) = 0.90
P(class 2 | x) = 0.05
```

真实类别对应的预测概率：

\[
P(y|x)= 0.90
\]


### 3.4 当前模块核心关系

```text
输入 x
↓
模型
↓
各类别概率
↓
取真实类别对应概率
↓
衡量模型是否正确
```


### 3.5 当前模块易错点

- 分类概率是给定模型输入x的条件概率


### 3.6 当前模块总结

> 训练分类模型时，除了关心最终预测类别是否正确，还关心模型给真实类别分配了多少概率。
> 真实类别概率越高，通常说明模型当前预测越好。

## 4. Likelihood、log 与 Negative Log-Likelihood

### 4.1 Likelihood

对于一个已知真实标签的样本：

\[
P(y_{\text{true}}|x)
\]

表示：在模型当前参属下，真实答案出现得有多合理。

> 

模型训练希望：

\[
P(y_{\text{true}}|x)
\]

尽可能 大。


### 4.2 多样本 Likelihood

如果多个样本对应的真实类别概率分别为：

\[
p_1,p_2,\dots,p_N
\]

则整体 likelihood 可写为：

\[
\mathcal{L}
=
\prod_{i=1}^{N}p_i
\]

我的理解：

> 整体概率的乘积来判断模型对样本类别判断的能力


### 4.3 为什么使用 log

对数满足：

\[
\log(ab)=\log a+\log b
\]

因此：

\[
\log\mathcal{L}
=
\sum_{i=1}^{N}\log p_i
\]

使用 log 的两个主要作用：

1. 将乘转换为加
2. 放大模型判断错误的误差


### 4.4 Negative Log-Likelihood

单样本：

\[
\text{NLL}
=
-\log P(y_{\text{true}}|x)
\]

规律：

```text
真实类别概率 ↑
→ NLL ↓

真实类别概率 ↓
→ NLL ↑
```

特殊情况：

```text
p = 1    → NLL = 0
p → 0    → NLL → ∞
```


### 4.5 核心因果链

```text
真实类别概率
↓
Likelihood
↓
取 log
↓
加负号
↓
Negative Log-Likelihood
↓
得到“越小越好”的 loss
```


### 4.6 当前模块总结

> NLL，Negative Log-Likelihood是直接通过模型输出的真实类别概率计算得到的损失Loss

## 5. Cross Entropy、Softmax 与 Logits

### 5.1 Logits

**Logits：**

> 模型在分类任务中输出的原始类别分数，还不是概率。

例如三分类任务：

- class 0：cat
- class 1：dog
- class 2：bird

模型可能输出：

```python
logits = [2.0, 1.0, -1.0]
```

Logits 的特点：

- 可以是负数；
- 可以大于 1；
- 不要求所有值之和等于 1；
- 只表示模型对不同类别给出的原始分数。

因此：

```text
logits ≠ probability
```

---

### 5.2 Softmax

Softmax 的作用是：

> 将任意实数形式的 logits 转换成一组可以解释为类别概率的数值。

公式：

\[
p_i =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

其中：

- \(z_i\)：第 \(i\) 个类别的 logit；
- \(p_i\)：第 \(i\) 个类别经过 Softmax 后得到的概率。

例如：

```text
logits:

cat  =  2.0
dog  =  1.0
bird = -1.0
```

经过 Softmax 后可能得到：

```text
probabilities:

cat  ≈ 0.71
dog  ≈ 0.26
bird ≈ 0.03
```

Softmax 后满足：

```text
0 < p_i < 1
```

并且：

\[
\sum_i p_i = 1
\]

因此可以把结果解释为：

\[
P(y=k|x)
\]

即给定输入 \(x\) 时，样本属于类别 \(k\) 的概率。

Softmax 会保持类别分数的相对大小关系：

```text
logits:

cat > dog > bird

Softmax 后：

P(cat|x) > P(dog|x) > P(bird|x)
```

---

### 5.3 Cross Entropy

对于分类问题，Cross Entropy 的一般形式为：

\[
H(y,p)
=
-\sum_k y_k\log p_k
\]

其中：

- \(y_k\)：真实标签；
- \(p_k\)：模型预测第 \(k\) 类的概率。

假设真实类别为 dog，对应 one-hot 标签：

\[
y=[0,1,0]
\]

模型预测概率：

\[
p=[0.71,0.26,0.03]
\]

代入：

\[
CE
=
-
[
0\log0.71
+
1\log0.26
+
0\log0.03
]
\]

因为非真实类别对应的 \(y_k=0\)，这些项都会消失，因此：

\[
CE
=
-\log0.26
\]

也就是说，在单标签多分类任务中：

\[
\boxed{
CE=-\log p_{\text{true}}
}
\]

其中：

\[
p_{\text{true}}
=
P(y_{\text{true}}|x)
\]

表示模型给真实类别分配的概率。

---

### 5.4 Cross Entropy 与 NLL 的关系

上一模块得到：

\[
NLL
=
-\log P(y_{\text{true}}|x)
\]

而单标签多分类中的 Cross Entropy：

\[
CE
=
-\log P(y_{\text{true}}|x)
\]

因此在这种常见分类场景下，可以理解为：

```text
Cross Entropy
↓
关注真实类别对应的预测概率
↓
对该概率取 Negative Log-Likelihood
```

即：

\[
CE
=
NLL
=
-\log P(y_{\text{true}}|x)
\]

直觉：

```text
真实类别概率越高
↓
Cross Entropy 越小

真实类别概率越低
↓
Cross Entropy 越大
```

---

### 5.5 从模型输出到 Cross Entropy

完整过程：

```text
输入 x
↓
神经网络
↓
logits
↓
Softmax
↓
P(y=k|x)
↓
找到真实类别对应的概率
↓
-log
↓
Cross Entropy Loss
```

例如：

```text
模型输出 logits
[2.0, 1.0, -1.0]

↓ Softmax

预测概率
[0.71, 0.26, 0.03]

真实类别：
dog

↓ 选择真实类别对应概率

p_true = 0.26

↓ Negative Log

loss = -log(0.26)
```

---

### 5.6 PyTorch 中的 CrossEntropyLoss

PyTorch 中通常写：

```python
criterion = nn.CrossEntropyLoss()

logits = model(x)

loss = criterion(logits, target)
```

需要注意：

```text
CrossEntropyLoss 的输入应该是 logits，
而不是已经经过 Softmax 的概率。
```

概念上可以理解为：

```text
logits
↓
LogSoftmax
↓
log probabilities
↓
NLLLoss
↓
Cross Entropy Loss
```

因此通常不要写：

```python
probs = torch.softmax(logits, dim=1)

loss = criterion(probs, target)
```

而应该直接写：

```python
loss = criterion(logits, target)
```

这样数值计算更加稳定。

---

### 5.7 为什么使用 LogSoftmax

数学上可以先：

```text
Softmax
↓
log
```

得到 log probability。

实际计算通常使用：

```text
LogSoftmax
```

把两步结合。

原因：

- 数值稳定性更好；
- 避免先计算非常小的概率再取 log；
- 计算效率更高。

因此：

```text
Softmax + log
```

在实现中通常合并为：

```text
LogSoftmax
```

---

### 5.8 Target 的表示

为了理解公式，可以把真实标签写成 one-hot。

例如：

```text
cat  = [1, 0, 0]
dog  = [0, 1, 0]
bird = [0, 0, 1]
```

但 PyTorch 中使用 `nn.CrossEntropyLoss()` 时，单标签多分类任务通常直接使用类别索引。

例如：

```text
cat  → 0
dog  → 1
bird → 2
```

如果真实类别是 dog：

```python
target = torch.tensor([1])
```

而不是必须写：

```python
target = torch.tensor([[0., 1., 0.]])
```

---

### 5.9 Batch 下的 Shape

假设：

```text
batch_size = 32
类别数 = 10
```

模型输出：

```text
logits.shape = (32, 10)
```

含义：

```text
32 个样本
每个样本有 10 个类别 logits
```

真实标签：

```text
target.shape = (32,)
```

其中：

```text
target = [7, 2, 1, 9, ...]
```

表示每个样本对应一个真实类别索引。

对于 MNIST：

```text
类别索引范围：

0 ~ 9
```

---

### 5.10 Cross Entropy 与反向传播

分类模型训练过程：

```text
模型参数
↓
Forward
↓
logits
↓
Cross Entropy
↓
loss
↓
loss.backward()
↓
链式法则
↓
参数梯度
↓
optimizer.step()
↓
更新参数
```

因此今天前半部分学习的：

```text
计算图
+
链式法则
```

和后半部分学习的：

```text
Cross Entropy
```

最终会在：

```python
loss.backward()
```

这里连接起来。

Backward 会从 Cross Entropy Loss 出发：

```text
loss
↑
logits
↑
最后一层
↑
隐藏层
↑
前面的网络层
↑
模型参数
```

利用链式法则逐层计算梯度。

---

### 5.11 核心知识链

```text
分类模型
↓
输出 logits
↓
Softmax
↓
得到 P(y=k|x)
↓
找到真实类别对应概率
↓
-log
↓
Negative Log-Likelihood
↓
Cross Entropy
↓
得到 loss
↓
backward
↓
利用链式法则计算参数梯度
```

---

### 5.12 易错点

#### Logits 和概率不能混淆

```text
logits：
模型原始类别分数

probabilities：
Softmax 后得到的类别概率
```

---

#### CrossEntropyLoss 前不要手动 Softmax

推荐：

```python
logits = model(x)
loss = criterion(logits, target)
```

不推荐：

```python
probs = torch.softmax(logits, dim=1)
loss = criterion(probs, target)
```

---

#### Cross Entropy 与 NLL 的关系有适用场景

对于常见的：

```text
单标签
+
多分类
+
one-hot / 类别索引标签
```

Cross Entropy 可以化简为：

\[
-\log P(y_{\text{true}}|x)
\]

即真实类别对应的 Negative Log-Likelihood。

---

### 5.13 当前模块总结

> 分类模型首先输出 logits，logits 只是各类别的原始分数，还不是概率。Softmax 将 logits 转换为满足和为 1 的类别概率 \(P(y=k|x)\)。对于单标签多分类问题，Cross Entropy 最终关注模型给真实类别分配的概率，并可以化简为 \(-\log P(y_{\text{true}}|x)\)。真实类别概率越高，loss 越小；真实类别概率越低，loss 越大。在 PyTorch 中，`nn.CrossEntropyLoss()` 直接接收 logits 和类别索引 target，因此通常不需要在它之前手动执行 Softmax。Cross Entropy 得到最终 loss 后，`loss.backward()` 再利用计算图和链式法则把梯度逐层传播回模型参数。

## 6. W2D1 综合复盘

### 6.1 今天的两条知识主线

#### 主线一：计算图与反向传播

```text
Forward
↓
建立计算图并计算各节点数值
↓
得到最终 loss
↓
Backward
↓
从 loss.grad = 1 开始
↓
上游梯度 × 局部导数
↓
利用链式法则逐层传播
↓
得到各参数梯度
```

链式法则核心：

\[
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial a}
\]

可以理解为：

```text
当前变量的最终梯度
=
上游梯度
×
当前运算的局部导数
```

---

#### 主线二：分类损失

```text
输入 x
↓
分类模型
↓
logits
↓
Softmax
↓
P(y=k|x)
↓
找到真实类别概率
↓
-log
↓
Negative Log-Likelihood
↓
Cross Entropy Loss
```

对于常见单标签多分类问题：

\[
CE
=
-\log P(y_{\text{true}}|x)
\]

因此：

```text
真实类别概率越高
→ loss 越小

真实类别概率越低
→ loss 越大
```

---

### 6.2 两条主线如何连接

完整分类模型训练流程：

```text
输入 x
↓
模型参数参与 Forward
↓
得到 logits
↓
Cross Entropy
↓
得到 loss
↓
loss.backward()
↓
利用计算图和链式法则
↓
计算每个参数的梯度
↓
optimizer.step()
↓
更新参数
```

因此：

```text
Cross Entropy
负责定义“模型当前错得有多严重”

链式法则 / Backpropagation
负责计算“这个错误应该如何影响每个参数”
```

---

### 6.3 a*b+c 计算图

定义：

\[
d=a\times b
\]

\[
L=d+c
\]

取：

\[
a=3,\quad b=4,\quad c=2
\]

Forward：

\[
d=3\times4=12
\]

\[
L=12+2=14
\]

计算图：

```text
a=3 ──┐
      × ── d=12 ──┐
b=4 ──┘            + ── L=14
               c=2 ┘
```

Backward：

\[
\frac{\partial L}{\partial L}=1
\]

\[
\frac{\partial L}{\partial d}=1
\]

\[
\frac{\partial L}{\partial c}=1
\]

\[
\frac{\partial d}{\partial a}=b=4
\]

\[
\frac{\partial d}{\partial b}=a=3
\]

根据链式法则：

\[
\frac{\partial L}{\partial a}
=
1\times4
=
4
\]

\[
\frac{\partial L}{\partial b}
=
1\times3
=
3
\]

最终：

```text
a.grad = 4
b.grad = 3
c.grad = 1
```

PyTorch Autograd 的结果应与手算一致。

---

### 6.4 分类损失示例

假设三分类：

```text
cat
dog
bird
```

模型输出 logits：

```text
[2.0, 1.0, -1.0]
```

经过 Softmax 后得到概率：

```text
cat  ≈ 0.71
dog  ≈ 0.26
bird ≈ 0.03
```

如果真实类别是 dog，则：

\[
P(y_{\text{true}}|x)=0.26
\]

Cross Entropy：

\[
CE
=
-\log0.26
\]

因此模型训练的目标是：

```text
提高真实类别对应的概率
↓
降低 Cross Entropy Loss
```

---

### 6.5 PyTorch 中的典型分类训练

```python
logits = model(x)

loss = criterion(logits, target)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

其中：

```text
model(x)
→ Forward，得到 logits

CrossEntropyLoss
→ 根据 logits 和真实标签计算 loss

loss.backward()
→ 通过计算图和链式法则计算梯度

optimizer.step()
→ 根据梯度更新模型参数
```

使用：

```python
criterion = nn.CrossEntropyLoss()
```

时，应直接传入：

```text
logits
```

而不是手动 Softmax 后的概率。

---

### 6.6 Logits / Probability / Target

Logits：

```text
模型输出的原始类别分数
不要求位于 0~1
不要求总和为 1
```

Probability：

```text
Softmax 后的类别概率
位于 0~1
总和为 1
```

Target：

```text
真实类别
```

在 PyTorch 单标签多分类中通常使用类别索引。

例如 MNIST：

```text
数字 7
→ target = 7
```

如果：

```text
batch_size = 64
类别数 = 10
```

则通常：

```text
logits.shape = (64, 10)

target.shape = (64,)
```

---

### 6.7 今日核心公式

链式法则：

\[
\frac{\partial L}{\partial a}
=
\frac{\partial L}{\partial d}
\frac{\partial d}{\partial a}
\]

条件概率：

\[
P(A|B)
=
\frac{P(A\cap B)}{P(B)}
\]

单样本 NLL：

\[
NLL
=
-\log P(y_{\text{true}}|x)
\]

Cross Entropy：

\[
H(y,p)
=
-\sum_k y_k\log p_k
\]

对于 one-hot 单标签分类：

\[
CE
=
-\log P(y_{\text{true}}|x)
\]

Softmax：

\[
p_i
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

---

### 6.8 今日最重要的理解

> 神经网络训练可以拆成两个问题：首先需要定义一个 loss，用来衡量模型当前预测得有多差；然后通过计算图和链式法则，把这个 loss 对模型输出的影响逐层反向传播到模型参数。对于单标签多分类任务，Cross Entropy 根据模型给真实类别分配的概率定义 loss；`loss.backward()` 再利用链式法则计算每个参数应该如何变化。

---

### 6.9 易错点

- 局部导数不等于最终 loss 对变量的梯度，还需要乘上上游梯度。
- `backward()` 的核心数学基础是链式法则。
- logits 不是概率。
- Softmax 把 logits 转换成概率。
- `CrossEntropyLoss` 通常直接接收 logits。
- 不要在 `CrossEntropyLoss` 前额外手动执行 Softmax。
- 多分类 target 通常是类别索引，而不是必须使用 one-hot。
- 真实类别概率越高，Cross Entropy 通常越小。