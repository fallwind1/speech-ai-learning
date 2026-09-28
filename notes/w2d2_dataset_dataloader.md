# W2D2 Dataset / DataLoader

## 1. Dataset、Sample、Batch 与 DataLoader

### 1.1 Sample

**Sample 是什么：**

> 

在监督学习中，一个 sample 通常包含：

```text
input
+
target
```

对于 MNIST：

```text
input x：
image

target y：
label
```

因此一个 MNIST sample 可以表示为：

```text
(image, label)
```

对于未来的 ASR 任务，一个 sample 可能包含：

```text
audio
+
transcript
```

---

### 1.2 Dataset

**Dataset 是什么：**

> 

可以把 Dataset 理解成：一个非常特殊的列表，每个位置保存一个sample
例如：`image, label = dataset[0]`

```text
index 0 → sample0
index 1 → sample1
index 2 → sample2
...
```

Dataset 最核心的两个能力：

```text
len(dataset)
→ 返回这个数据集总共有多少个sample

dataset[index]
→ 取数据集dataset中第index个样本
```

因此：

```python
image, label = dataset[0]
```

表示：

> 取出dataset中第0个样本

Dataset 中：

```text
__len__()
作用：

> 有多少个样本


__getitem__(index)
作用：

> 怎么取一个样本，返回的是(image, target)
```

---

### 1.3 Batch

**Batch 是什么：**

> 样本批次，防止单次处理样本过多与full-batch区别，以后mini-batch对应：一小批样本-一次step

如果：

```text
batch_size = 32
```

表示：

> 这一批有32个样本

Sample 与 Batch 的区别：

```text
Sample：
________________________________

Batch：
________________________________
```

训练时通常：

```text
一个 batch
↓
forward
↓
loss
↓
backward
↓
step
```

---

### 1.4 MNIST 的 Shape

单张 MNIST 图片通常表示为：

```text
(C, H, W)
=
(channel, height, width)
```

其中：

```text
C = 通道数
H = 高度
W = 宽度
```

如果：

```text
batch_size = 32
```

那么：

```text
images.shape =
torch.Size([32,1,28,28])

labels.shape =
(32,)
```

一般图像 batch 的 shape 可以写成：

```text
(batch_size, C, H, W)
```

分别表示：

```text
第 1 维：
sample数

第 2 维：
通道数

第 3 维：
每张图片高度

第 4 维：
每张图片宽度
```

为什么 labels 的 shape 通常只有：

```text
(batch_size,)
```

我的理解：

> 因为每个样本只有一个类别index，所以batch_size个样本的lables应该是(batch_size,)的shape

---

### 1.5 DataLoader

**DataLoader 是什么：**

> 负责从Dataset中按规则取样本、组成batch、shuffle，并迭代提供给训练循环

常见写法：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
```

其中：

```text
dataset
→ 从这里取数据

batch_size
→ 每次组织batch_size个sample

shuffle
→ 是否打乱访问顺序
```

在：

```python
for images, labels in loader:
    ...
```

中，每次循环得到的是：

```text
一个batch
```

而不是：

```text
一个sample
```

---

### 1.6 Dataset 与 DataLoader 的区别

Dataset 主要负责：

```text
1. 有什么样本

2. 怎么取一个样本
```

DataLoader 主要负责：

```text
1. 从哪里取样本

2. 一次取多少样本

3. 是否打乱访问顺序
```

因此可以概括为：

```text
Dataset
=
整本书


DataLoader
=
按规则帮你翻书的人
```

---

### 1.7 Sample、Dataset、DataLoader、Batch 的关系

完整关系：

```text
Sample
↓
构成
↓
Dataset
↓
取出样本
↓
DataLoader
↓
组成
↓
Batch
↓
送入
↓
Training Loop
```

我的理解：

> DataLoader从Dataset中取sample组成batch送入训练循环中进行训练

---

### 1.8 与 Week 1 训练循环的关系

Week 1 线性回归中，我们使用：

```text
full-batch
```

一次性参与训练。

而使用 mini-batch 后：

```text
batch 1
↓
forward
↓
loss
↓
backward
↓
step

batch 2
↓
重复上述过程
```

因此：

```text
一个 step 通常表示：
一个batch

一个 epoch 通常表示：
全部batch训练过一遍
```

---

### 1.9 与 W2D1 Cross Entropy 的连接

假设：

```text
batch_size = 64
类别数量 = 10
```

那么通常：

```text
images.shape =
(64,1,28,28)

logits.shape =
(64,10)

labels / target.shape =
(64,)
```

训练过程：

```text
DataLoader
↓
得到 images 和 labels
↓
model(images)
↓
logits
↓
CrossEntropyLoss(logits, target)
↓
loss
↓
backward
↓
optimizer.step()
```

---

### 1.10 当前模块核心总结

**Sample：**

> 一个具体样本；监督学习中通常是 (input, target)。MNIST 中就是 (image, label)。

**Dataset：**

> 一组可按索引访问的 sample。len(dataset) 返回样本数量，dataset[index] 返回一个 sample。

**Batch：**

> 一次送入模型计算的一组 sample。batch_size=32 表示一次处理 32 个样本。

**DataLoader：**

> 负责从 Dataset 中按规则取样本、组成 batch、shuffle，并迭代提供给训练循环。

一句话总结四者关系：

> DataLoader从Dataset中取sample组成batch

---

## 2. 加载 MNIST 与 Transform

### 2.1 MNIST Dataset

PyTorch / Torchvision 提供：

```python
torchvision.datasets.MNIST(...)
```

它的作用：

> 返回一个现成的数据集

创建 Dataset 后，可以通过：

```python
len(dataset)
```

得到：

> sample数

通过：

```python
dataset[index]
```

得到：

> 第index个sample

对于 MNIST，一个 sample 的形式为：

```text
(image, target)
```

分别表示：

```text
第一个元素：
图片

第二个元素：
标签
```

---

### 2.2 `MNIST()` 的主要参数

基本写法：

```python
dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True,
)
```

各参数作用：

#### `root`

> 例如：DATA_DIR = Path("data")，表示MNIST数据应该保存在哪里，控制的是**数据存储位置**

#### `train=True`

> 表示加载官方的training portion

#### `train=False`

> 表示加载官方的test portion 

#### `transform`

> 在Dataset返回图片之前，对图片执行指定转换

#### `download=True`

> 

---

### 2.3 Train / Test 的基本关系

MNIST 原始数据已经提供：

```text
training set
+
test set
```

在 Torchvision 中：

```python
train=True
```

表示：

> 加载官方training portion

```python
train=False
```

表示：

> 加载官方test portion

本周训练时：

```text
原始 training set
↓
再划分
↓
train set + validation set
```

而原始 test set 应：

> 保持独立，主要留到模型和超参数选择完成以后进行最终评估。

原因：

> 如果反复根据test set的表现选择learning rate、模型结构等超参数，就会间接利用test信息做模型决策，使test set不再是独立、公正的最终评估数据。

因此：

```text
train：
用于训练模型、学习模型参数

validation：
用于模型选择、超参数选择以及观察泛化表现

test：
用于模型开发基本结束后的最终独立评估。
```

---

### 2.4 Transform 是什么

Transform 的作用：

> 再Dataset返回图片之前，对图片执行指定转换。

在 Dataset 中：

```python
datasets.MNIST(
    ...,
    transform=transform,
)
```

表示：

> 数据集中的数据已经经过transform转换

因此数据流可以理解为：

```text
原始图片
↓
transform(image)
↓
转换后的图片
↓
Dataset 返回
```

---

### 2.5 `ToTensor()`

本次使用：

```python
transform = transforms.ToTensor()
```

它主要完成：

```text
输入：
图片

↓

ToTensor()

↓

输出：
PyTorch Tensor
```

图片 shape 会从常见的：

```text
(H, W)
或
(H, W, C)
```

转换为适合 PyTorch 图像计算的：

```text
(C, H, W)
```

对于 MNIST：

```text
image.shape =
(batch_size,C,H,W)

其中：

C = 1
H = 28
W = 28
```

数据类型通常变成：

```text
dtype =
torch.float32
```

像素值通常由：

```text
原始范围：
[0,255]

转换为：
[0,1]
```

---

### 2.6 为什么模型需要 Tensor

原始图片不能直接按照普通 PyTorch Tensor 的训练流程参与：

```text
Tensor 数学运算
自动求导相关运算
CPU / GPU上的统一张量计算
```

转换为 Tensor 后，可以：

```text
1. 使用PyTorch Tensor API进行数学计算

2. 以统一shape和dtype输入神经网络

3. 后续方便移动到CPU / GPU并参与模型训练
```

因此：

```text
原始图片
↓
ToTensor
↓
Tensor
↓
后续模型训练
```

---

### 2.7 实际加载 MNIST

代码：

```python
from pathlib import Path

import torch
from torchvision import datasets, transforms


DATA_DIR = Path("data")

transform = transforms.ToTensor()

mnist_train = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True,
)
```

其中：

```text
DATA_DIR：
指定MNIST数据保存和读取的目录

train=True：
加载MNIST官方训练部分

transform=transform：
在读取每张图片时执行ToTensor转换

download=True：
本地没有数据时自动下载MNIST
```

---

### 2.8 查看 Dataset

代码：

```python
print("dataset size:", len(mnist_train))
```

实际输出：

```text
dataset size:
60000
```

它表示：

> 当前mnist_train Dataset中包含60000个samples

---

### 2.9 取一个 Sample

代码：

```python
image, label = mnist_train[0]
```

其中：

```text
image：
经transform转换后的第一张MNIST图片

label：
第一张图片对应的真实数字类别
```

检查：

```python
print(type(image))
print(image.shape)
print(image.dtype)
print(image.min())
print(image.max())

print(type(label))
print(label)
```

实际结果：

```text
image type =
torch.Tensor

image shape =
torch.Size([1,28,28])

image dtype =
torch.float32

image min =
0.0

image max =
1.0

label type =
int

label =
5
```

---

### 2.10 为什么 `image.shape = (1, 28, 28)`

三个维度分别表示：

```text
1：
一个通道

28：
图片高度

28：
图片宽度
```

其中 `1` 不是：

> batch_size

而是：

> 一个灰度channel

如果以后是普通 RGB 图片，channel 通常为：

```text
3
```

---

### 2.11 Image 与 Label 的关系

一个 MNIST sample：

```text
(image, label)
```

可以理解为监督学习中的：

```text
(x, y)
```

其中：

```text
image
→ 模型输入x

label
→ 真是目标y
```

以后训练时：

```text
image
↓
model
↓
logits

label
↓
CrossEntropyLoss 的 target
```

---

### 2.12 当前数据流

完整过程：

```text
MNIST 原始文件
↓
torchvision.datasets.MNIST 负责读取和封装
↓
MNIST Dataset
↓
dataset[index]
↓
找到对应的原始 image 和 label
↓
transform
↓
把 image 转换为 Tensor
↓
(image, label)
↓
后续 DataLoader
```

---

### 2.13 当前模块核心总结

**MNIST Dataset：**

> 将MNIST的图片和标签封装成可以按索引访问的PyTorch Dataset

**Transform：**

> 对 Dataset 读取出的输入数据执行预处理或格式转换。

**ToTensor：**

> 将图片转换为适合 PyTorch 使用的浮点 Tensor，并将常见图像从 HWC / HW 形式转换为 CHW 形式，同时将适用的 0~255 像素缩放到 0~1。

**`dataset[index]`：**

> 获取指定索引的一个sample

**Train / Validation / Test：**

> Train 用于学习模型参数；Validation 用于模型和超参数选择；Test 保持独立，用于最终评估。

一句话总结本模块：

> Torchvision的MNIST Dataset负责封装和访问MNIST样本，transform将原始图片转换为模型可使用的Tensor，而每次 dataset[index] 最终得到一个由输入图片和真实标签组成的 sample。

## 3. 小型 MNIST 子集与 Train / Validation 划分

### 3.1 为什么先使用小型子集

完整 MNIST training set 样本数：

```text
60000
```

小型子集的目标不是：

> 把 MNIST 分类准确率做到很高

而是：

> 把数据管线搭对，并确认数据不会互相污染

---

### 3.2 Subset 是什么

`Subset` 的作用：

> 取出原dataset的一部分索引作为新dataset

可以理解为：

```text
原始 Dataset
↓
选择一部分 index
↓
Subset
```

例如：

```python
indices = [0, 3, 7, 10]

small_dataset = Subset(
    full_dataset,
    indices,
)
```

这里：

```text
full_dataset
→ 原dataset

indices
→ 要取出的索引

small_dataset
→ 新的小型数据集
```

`Subset` 是否会复制一份完整数据：

> 不会

我的理解：

> 从原dataset选中index取出

---

### 3.3 为什么需要固定随机种子

如果每次随机选择 subset / train / val 时都不固定 seed：

> 每次取出的数据集都不同

因此使用固定随机种子的主要目的：

> 固定随即结果，让随机过程可复现

示例：

```python
generator = torch.Generator().manual_seed(________)
```

它的作用：

> 将固定SEED传入generator来得到可复现划分

注意：

固定 seed 的主要目的不是：

> 为了让模型效果更好

而是：

> 让实验条件更稳定，更容易比较

---

### 3.4 构造小型 MNIST 子集

假设：

```text
完整 training set：
60000 个样本

本次选择：
2000 个样本
```

可以先生成随机索引：

```python
generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(mnist_train),
    generator=generator,
)[:SUBSET_SIZE]
```

其中：

```text
torch.randperm(...)
→ 生成0到len(mnist_train)-1的随机排列

[:SUBSET_SIZE]
→ 从随机排列中选择前SUBSET_SIZE个index
```

然后：

```python
small_mnist = Subset(
    mnist_train,
    indices.tolist(),
)
```

此时：

```text
len(small_mnist) =
2000
```

---

### 3.5 Train / Validation 划分

将小型数据集继续划分为：

```text
small_mnist
↓
train_dataset
+
val_dataset
```

假设：

```text
总样本数 =
2000

train 样本数 =
1600

val 样本数 =
400
```

划分代码：

```python
train_dataset, val_dataset = random_split(
    small_mnist,
    [TRAIN_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(SEED),
)
```

其中：

```text
random_split
→ 把一个Dataset随即划分为互不重叠的新Dataset

TRAIN_SIZE
→ 训练集的大小

VAL_SIZE
→ 验证集的大小
```

必须满足：

```text
TRAIN_SIZE + VAL_SIZE =
SUBSET_SIZE
```

---

### 3.6 Train / Validation / Test 的职责

#### Train Set

作用：

> 

训练过程中：

```text
train data
↓
forward / loss / backward
↓
optimizer.step() 更新参数
```

---

#### Validation Set

作用：

> 在模型训练过程中评估未直接用于参数更新的数据表现，并用于模型选择和超参数选择。

通常用于比较：

```text
1. 不同learning rate
2. 不同模型结构
3. 不同训练轮数或其他超参数
```

Validation 是否用于 `optimizer.step()`：

> 通常不用于optimizer.step()，只用于评估

---

#### Test Set

作用：

> 在模型开发和超参数选择基本完成以后，对最终模型进行独立评估

Test set 应该在：

> 模型和超参数选择完成后主要用于最终评估

Test set 不应该反复用于：

> learning rate、模型结构、训练轮数等超参数选择。

---

### 3.7 为什么不能用 Test 选超参数

假设比较：

```text
lr = 0.1
lr = 0.01
lr = 0.001
```

如果每次都根据 test 结果选择：

> test set的表现就会参与模型和超参数决策

这会导致：

> test信息泄漏到模型开发过程中，使最终test结果变得过于乐观，不能再代表真正独立的泛化能力。

正确做法：

```text
train
→ 学习模型参数

validation
→ 选择超参数和比较模型

test
→ 最终独立评估
```

---

### 3.8 什么叫 Train / Val 不重叠

Train / Val 不重叠表示：

> 同一样本不能同时出现在train_dataset 和 val_dataset 中。

从 index 的角度：

```text
train_indices ∩ val_indices =
∅
```

如果交集：

```text
size = 0
```

说明：

> 没有任何样本同时属于train和validation

---

### 3.9 random_split 为什么可以避免重叠

`random_split` 的作用：

> 将一个 Dataset 随机划分为指定大小的互不重叠子 Dataset。

调用：

```python
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)
```

会得到：

```text
train_dataset：
包含其中一部分随机样本。

val_dataset：
包含剩余分配给 validation 的随机样本。
```

两者之间：

> 不共享同一个 split index，因此划分结果互不重叠。

---

### 3.10 Subset 中的 Index

`Subset` 本质上保存：

```text
1. 原始Dataset的引用
2. 被选择的index列表
```

因此：

```python
subset.indices
```

表示：

> 当前 Subset 使用的底层 Dataset 索引。

而：

```python
subset.dataset
```

表示：

> 当前 Subset 所引用的底层 Dataset。

---

### 3.11 random_split 返回值

`random_split` 返回的对象通常也是：

```text
Subset
```

因此可以查看：

```python
train_dataset.indices
val_dataset.indices
```

它们表示：

> train 和 validation 分别使用底层输入 Dataset 中的哪些位置。

注意：

如果 Dataset 本身又是一个 `Subset`，这些 index 可能表示：

> 相对于这个输入 Subset 的位置，而不一定直接是最初原始 Dataset 的 index。

因此判断不重叠时要明确：

> 两个 split 是否基于同一个底层 Dataset / Subset，以及当前比较的 index 属于哪一层。

---

### 3.12 验证 Train / Val 不重叠

一种直接方法：

```python
train_indices = set(train_dataset.indices)
val_indices = set(val_dataset.indices)

overlap = train_indices & val_indices
```

其中：

```text
set(...)
→ 将 index 列表转换成集合。

&
→ 求两个集合的交集。

overlap
→ 同时出现在 train 和 validation 中的 index。
```

如果：

```python
len(overlap) == 0
```

表示：

> train和validation没有样本位置重叠

---

### 3.13 本次实际配置

```text
SEED =
42

SUBSET_SIZE =
2000

TRAIN_SIZE =
1600

VAL_SIZE =
400
```

检查：

```text
SUBSET_SIZE
=
TRAIN_SIZE + VAL_SIZE

结果：
2000 = 1600 + 400
```

实际输出：

```text
small dataset size =
2000

train size =
1600

val size =
400

overlap size =
0
```

---

### 3.14 当前数据结构

```text
MNIST 官方 training set
↓
选择随机索引
↓
small_mnist
↓
random_split
↙             ↘
train_dataset   val_dataset
```

Test set：

```text
MNIST 官方 test set
↓
保持独立，留作后续最终评估
```

---

### 3.15 当前模块核心总结

**Subset：**

> 根据一组index，从已有Dataset中创建只访问部分样本的子Dataset

**random_split：**

> 将一个Dataset随机划分为互不重叠的多个子 Dataset。

**固定 seed：**

> 控制随机过程，提高实验可复现性和不同实验之间的可比较性。

**Train：**

> 用于学习模型参数。

**Validation：**

> 用于评估训练过程、比较模型以及选择超参数。

**Test：**

> 

**不重叠：**

> 

一句话总结：

> 

---

### 3.16 易错点

#### 使用同一个已经训练后的模型继续做另一组实验

问题：

> 

---

#### Train / Val 样本重复

会导致：

> 

---

#### 用 Test 选择 learning rate

会导致：

> 

---

#### 不固定随机种子

会导致：

> 

---

#### 把 Validation 当成 Training Data

问题：

> 