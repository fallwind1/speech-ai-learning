# W8D3：CTC 正确性验证

## 1. 暴力路径枚举（Brute-force Enumeration）

### 1.1 为什么需要暴力枚举？

W8D2 已经实现了 CTC 前向算法 `ctc_forward_log()`，通过动态规划计算目标序列的 Log 概率。

但是，动态规划实现中可能存在错误，例如：

- DP 初始化错误。
- Stay、Advance、Skip 状态转移条件错误。
- Skip 判断条件错误。
- 终止状态选择错误。
- LogSumExp 使用错误。

如果只使用动态规划自身的结果进行验证，可能无法发现算法内部的逻辑错误。

因此，需要一种独立的计算方法进行验证，暴力路径枚举就是其中一种方法。

**暴力枚举的核心思想：**

1. 枚举所有可能的 CTC 路径。
2. 对每条路径执行 CTC Collapse。
3. 判断 Collapse 后的结果是否等于目标序列。
4. 保留所有符合目标序列的路径。
5. 计算每条合法路径的概率。
6. 将所有合法路径的概率相加，得到目标序列的总概率。

**与动态规划的区别：**

| 对比 | 暴力枚举 | 动态规划 |
|---|---|---|
| 核心思想 | 枚举所有完整路径 | 合并相同中间状态 |
| 是否生成完整路径 | 是 | 否 |
| 计算效率 | 低 | 高 |
| 实现难度 | 相对简单 | 相对复杂 |
| 主要用途 | 小规模正确性验证 | 实际模型计算 |

暴力枚举的计算复杂度随时间帧数量和词表大小指数增长，因此只适合小规模测试，不适合真实 ASR 模型推理。

### 1.2 CTC 路径枚举

**（1）CTC 路径的定义**

假设模型输入包含 T 个时间帧，词表包含 C 个 Token。

每个时间帧都可以输出词表中的任意一个 Token，因此一条 CTC 路径就是一个长度为 T 的 Token 序列。

例如：

```python
blank = 0
a = 1
```

当 T=2 时，所有可能路径为：

```text
(0, 0)
(0, 1)
(1, 0)
(1, 1)
```

**（2）所有可能路径的数量**

每个时间帧有 C 种可能输出，T 个时间帧相互组合。

因此：

$$
N_{\text{paths}}=C^T
$$

例如：

| T | C | 路径数量 |
|---:|---:|---:|
| 2 | 2 | 4 |
| 3 | 2 | 8 |
| 4 | 2 | 16 |
| 4 | 3 | 81 |
| 4 | 5 | 625 |

**（3）itertools.product() 的作用**

Python 中的 `itertools.product()` 可以生成多个序列的笛卡尔积。

例如：

```python
from itertools import product

paths = product(
    range(2),
    repeat=2
)

for path in paths:
    print(path)
```

输出：

```text
(0, 0)
(0, 1)
(1, 0)
(1, 1)
```

其中：

- `range(2)`：每个时间帧可以选择 Token 0 或 Token 1。
- `repeat=2`：生成长度为2的路径。

因此：

```python
product(range(C), repeat=T)
```

表示生成所有长度为 T、每个位置取值范围为词表索引的路径。

实际实现时可以直接遍历生成器，不必使用 `list()` 保存所有路径，避免占用过多内存。

### 1.3 CTC Collapse

暴力枚举得到完整路径后，需要将其转换成最终输出，判断是否与目标序列一致。

CTC Collapse 包含两个步骤：

1. 合并相邻且相同的 Token。
2. 删除 Blank。

**示例：**

原始路径：

```text
a a _ a b b
```

第一步：合并相邻重复 Token。

```text
a _ a b
```

第二步：删除 Blank。

```text
a a b
```

最终输出：

```text
aab
```

**为什么不能颠倒顺序？**

因为如果先删除 Blank，那么原本被 Blank 分隔的相同 Token 就会相邻，随后被错误合并。

例如：

```text
a _ a
```

正确处理：

```text
合并相邻重复：a _ a
删除 Blank：a a
```

最终结果：

```text
aa
```

如果先删除 Blank：

```text
a a
```

再合并相邻重复：

```text
a
```

就会产生错误。

因此必须先合并相邻重复 Token，再删除 Blank。

**ctc_collapse() 实现：**

```python
def ctc_collapse(
    path,
    blank=0
):
    """
    将一条 CTC 路径转换为最终 Token 序列。

    步骤：
    1. 合并相邻重复 Token
    2. 删除 Blank
    """

    merged = []

    previous = None

    # 第一步：合并相邻重复 Token
    for token in path:

        if token != previous:
            merged.append(token)

        previous = token

    # 第二步：删除 Blank
    result = [
        token
        for token in merged
        if token != blank
    ]

    return result
```

测试：

```python
print(
    ctc_collapse(
        [1, 1, 0, 1, 2, 2],
        blank=0
    )
)

print(ctc_collapse([1, 0, 1], blank=0))
print(ctc_collapse([1, 1], blank=0))
```

预期输出：

```text
[1, 1, 2]
[1, 1]
[1]
```

### 1.4 单条路径概率计算

假设模型在两个时间帧的输出概率为：

| 时间帧 | Blank | a |
|---|---:|---:|
| t=0 | 0.6 | 0.4 |
| t=1 | 0.3 | 0.7 |

假设某条路径为：

```text
a _
```

那么这条路径的概率等于各时间帧对应 Token 概率的乘积：

$$
P(\pi|x)=
0.4\times0.3
$$

$$
P(\pi|x)=0.12
$$

一般情况下：

$$
P(\pi|x)=
\prod_{t=0}^{T-1}p_t(\pi_t)
$$

其中：

- T：时间帧数量。
- π_t：第 t 个时间帧输出的 Token。
- p_t(π_t)：第 t 个时间帧输出该 Token 的概率。

**为什么使用 Log 概率？**

直接连乘多个较小的概率，容易出现数值下溢。

因此，将概率转换为 Log 概率：

$$
\log P(\pi|x)
=
\log\prod_{t=0}^{T-1}p_t(\pi_t)
$$

利用对数性质：

$$
\boxed{
\log P(\pi|x)=
\sum_{t=0}^{T-1}
\log p_t(\pi_t)
}
$$

代码：

```python
path_score = log_probs.new_zeros(())

for t, token in enumerate(path):

    path_score = (
        path_score
        + log_probs[t, token]
    )
```

这里的 `path_score` 表示当前路径的 Log 概率。

### 1.5 目标序列概率计算

一条 CTC 路径经过 Collapse 后，可能得到目标序列，也可能得到其他序列。

因此，只有 Collapse 结果等于 target 的路径才需要保留。

目标序列的概率不是某一条路径的概率，而是所有能够得到该目标序列的路径概率之和。

数学表达：

$$
P(y|x)=
\sum_{\pi:B(\pi)=y}
P(\pi|x)
$$

其中：

- y：目标序列。
- π：一条 CTC 路径。
- B：CTC Collapse 函数。
- B(π)=y：表示路径经过 Collapse 后等于目标序列。

**示例：**

目标：

```text
a
```

所有可能路径：

| 路径 | Collapse | 是否符合目标 | 概率 |
|---|---|---|---:|
| _ _ | 空序列 | 否 | 0.18 |
| _ a | a | 是 | 0.42 |
| a _ | a | 是 | 0.12 |
| a a | a | 是 | 0.28 |

因此：

$$
P(a|x)=0.42+0.12+0.28
$$

$$
\boxed{P(a|x)=0.82}
$$

**为什么需要 LogSumExp？**

因为我们保存的是每条路径的 Log 概率，而目标序列概率需要将所有合法路径的概率相加。

在 Log 空间中：

$$
\log P(y|x)=
\log\sum_{\pi:B(\pi)=y}
P(\pi|x)
$$

利用 LogSumExp：

$$
\boxed{
\log P(y|x)=
\operatorname{LogSumExp}
\left(
\log P(\pi_1|x),
\ldots,
\log P(\pi_k|x)
\right)
}
$$

不能直接对 Log 概率使用普通 sum，因为那相当于对原始概率进行连乘。

### 1.6 代码实现

**（1）ctc_collapse()**

负责将一条 CTC 路径转换为最终 Token 序列。

```python
def ctc_collapse(
    path,
    blank=0
):
    """
    将一条 CTC 路径转换为最终 Token 序列。
    """

    merged = []
    previous = None

    # 合并相邻重复 Token
    for token in path:

        if token != previous:
            merged.append(token)

        previous = token

    # 删除 Blank
    result = [
        token
        for token in merged
        if token != blank
    ]

    return result
```

**（2）ctc_bruteforce_log()**

使用暴力枚举计算目标序列的 Log 概率。

```python
from itertools import product

import torch


def ctc_bruteforce_log(
    log_probs,
    target,
    blank=0
):
    """
    使用暴力枚举计算 CTC 目标序列的 Log 概率。

    参数：
        log_probs:
            Tensor，形状为 (T, C)
            已经过 Log Softmax

        target:
            原始目标 Token 序列

        blank:
            Blank 对应的 Token ID

    返回：
        scalar Tensor:
            log P(target | input)
    """

    # 1. 获取时间帧数量和词表大小
    if log_probs.ndim != 2:
        raise ValueError(
            "log_probs must have shape (T, C)"
        )

    T, C = log_probs.shape

    target = [
        int(token)
        for token in target
    ]

    if not 0 <= blank < C:
        raise ValueError("Invalid blank ID")

    for token in target:

        if not 0 <= token < C:
            raise ValueError(
                f"Invalid target token: {token}"
            )

        if token == blank:
            raise ValueError(
                "target must not contain blank"
            )

    # 2. 枚举所有可能路径
    path_scores = []

    for path in product(
        range(C),
        repeat=T
    ):

        # 3. 执行 CTC Collapse
        collapsed = ctc_collapse(
            path,
            blank
        )

        # 4. 判断是否与目标序列一致
        if list(collapsed) != target:
            continue

        # 5. 计算当前路径的 Log 概率
        path_score = log_probs.new_zeros(())

        for t, token in enumerate(path):

            path_score = (
                path_score
                + log_probs[t, token]
            )

        path_scores.append(path_score)

    # 6. 没有合法路径
    if len(path_scores) == 0:

        return log_probs.new_full(
            (),
            float("-inf")
        )

    # 7. 汇总所有合法路径
    return torch.logsumexp(
        torch.stack(path_scores),
        dim=0
    )
```

**代码整体流程：**

1. 获取 T 和 C。
2. 使用 product 生成所有可能路径。
3. 使用 ctc_collapse 得到最终 Token 序列。
4. 筛选 Collapse 结果等于 target 的路径。
5. 计算每条合法路径的 Log 概率。
6. 使用 LogSumExp 汇总所有合法路径。
7. 返回目标序列的 Log 概率。

如果没有任何合法路径，则返回负无穷。

### 1.7 与动态规划结果对照

暴力枚举与动态规划是两种不同的计算方法，但计算的是同一个数学量。

因此，两者结果应该一致。

$$
\boxed{
\text{BruteForce}(x,y)
\approx
\text{DP}(x,y)
}
$$

**普通目标测试：**

```python
probs = torch.tensor(
    [
        [0.6, 0.4],
        [0.3, 0.7]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

target = [1]

# 暴力枚举
brute_result = ctc_bruteforce_log(
    log_probs,
    target,
    blank=0
)

# 动态规划
dp_result = ctc_forward_log(
    log_probs,
    target,
    blank=0
)

print("Brute Force:", brute_result.item())
print("Dynamic Programming:", dp_result.item())

assert torch.allclose(
    brute_result,
    dp_result,
    atol=1e-6
)

assert torch.allclose(
    brute_result.exp(),
    torch.tensor(0.82),
    atol=1e-6
)

print("CTC enumeration test passed!")
```

**重复 Token 测试：**

```python
probs = torch.tensor(
    [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

target = [1, 1]

brute_result = ctc_bruteforce_log(
    log_probs,
    target,
    blank=0
)

dp_result = ctc_forward_log(
    log_probs,
    target,
    blank=0
)

assert torch.allclose(
    brute_result,
    dp_result,
    atol=1e-6
)

assert torch.allclose(
    brute_result.exp(),
    torch.tensor(0.125),
    atol=1e-6
)

print("Repeated token test passed!")
```

这里 T=3，目标包含两个相邻相同 Token。

唯一能够产生目标序列的路径是：

```text
a _ a
```

因此：

$$
P(aa|x)=0.5\times0.5\times0.5
$$

$$
P(aa|x)=0.125
$$

**空目标测试：**

```python
probs = torch.tensor(
    [
        [0.6, 0.4],
        [0.3, 0.7]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

brute_result = ctc_bruteforce_log(
    log_probs,
    target=[],
    blank=0
)

dp_result = ctc_forward_log(
    log_probs,
    target=[],
    blank=0
)

assert torch.allclose(
    brute_result,
    dp_result,
    atol=1e-6
)

assert torch.allclose(
    brute_result.exp(),
    torch.tensor(0.18),
    atol=1e-6
)

print("Empty target test passed!")
```

空目标只能通过所有时间帧输出 Blank 的路径产生。

**不可能对齐测试：**

```python
probs = torch.tensor(
    [
        [0.5, 0.5],
        [0.5, 0.5]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

target = [1, 1]

brute_result = ctc_bruteforce_log(
    log_probs,
    target,
    blank=0
)

dp_result = ctc_forward_log(
    log_probs,
    target,
    blank=0
)

assert torch.isneginf(brute_result)
assert torch.isneginf(dp_result)

print("Impossible alignment test passed!")
```

目标为两个相邻相同 Token，但只有两个时间帧，无法在中间插入 Blank，因此不存在合法路径。

**注意：** 暴力枚举与 DP 的比较应使用相同的输入概率、目标序列和 Blank ID。由于浮点计算存在精度误差，使用 `torch.allclose()` 而不是直接使用 `==`。

---

## 2. 本模块总结

### 2.1 暴力枚举整体流程

$$
\text{所有可能路径}
$$

$$
\downarrow
$$

$$
\text{CTC Collapse}
$$

$$
\downarrow
$$

$$
\text{筛选目标序列匹配路径}
$$

$$
\downarrow
$$

$$
\text{计算各路径 Log 概率}
$$

$$
\downarrow
$$

$$
\text{LogSumExp}
$$

$$
\downarrow
$$

$$
\log P(y|x)
$$

### 2.2 核心公式

所有可能路径数量：

$$
N_{\text{paths}}=C^T
$$

单条路径概率：

$$
P(\pi|x)=
\prod_{t=0}^{T-1}p_t(\pi_t)
$$

单条路径 Log 概率：

$$
\log P(\pi|x)=
\sum_{t=0}^{T-1}\log p_t(\pi_t)
$$

目标序列概率：

$$
P(y|x)=
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

目标序列 Log 概率：

$$
\log P(y|x)=
\operatorname{LogSumExp}
\left(
\log P(\pi_1|x),
\ldots,
\log P(\pi_k|x)
\right)
$$

### 2.3 关键函数

| 函数 | 作用 |
|---|---|
| `ctc_collapse()` | 将 CTC 路径转换为最终 Token 序列 |
| `ctc_bruteforce_log()` | 通过暴力枚举计算目标序列 Log 概率 |
| `ctc_forward_log()` | 通过动态规划计算目标序列 Log 概率 |

### 2.4 正确性验证原则

暴力枚举与动态规划的结果应该一致。

如果不一致，优先检查：

1. CTC Collapse 顺序。
2. 路径枚举是否完整。
3. 路径概率是否正确。
4. Stay、Advance、Skip 转移条件。
5. DP 初始化和终止状态。
6. LogSumExp 与无合法路径的处理。

### 2.5 本模块核心认识

暴力枚举不是为了替代动态规划，而是为动态规划提供一个独立的正确性验证方法。

暴力枚举直接计算所有合法完整路径的概率之和；动态规划则通过合并中间状态避免重复计算。

两者最终计算的是同一个目标序列概率。

因此，在小规模条件下，可以使用暴力枚举作为参考答案，验证自己实现的 CTC 前向算法是否正确。


# W8D3 模块2：PyTorch CTCLoss 对照

## 2. PyTorch CTCLoss

### 2.1 CTCLoss 的作用

W8D2 已经实现了 `ctc_forward_log()`，通过动态规划计算目标序列的 Log 概率。

W8D3 模块1又实现了 `ctc_bruteforce_log()`，通过暴力枚举计算相同的 Log 概率。

现在引入 PyTorch 提供的 `torch.nn.CTCLoss()`，用于进一步验证自己实现的 CTC 算法是否正确。

**（1）CTCLoss 计算什么？**

CTCLoss 计算目标序列的负 Log 概率，即 Negative Log-Likelihood。

假设输入音频为 x，目标序列为 y，则：

$$
L_{\mathrm{CTC}}=-\log P(y|x)
$$

**（2）与 ctc_forward_log() 的区别**

`ctc_forward_log()` 返回目标序列的 Log 概率：

$$
\text{DP}=\log P(y|x)
$$

PyTorch CTCLoss 返回目标序列的负 Log 概率：

$$
\text{CTCLoss}=-\log P(y|x)
$$

**（3）为什么两者结果互为相反数？**

神经网络训练通常采用最小化损失的方式。

而最大化正确目标序列的概率等价于最小化其负 Log 概率。

因此：

$$
\boxed{
\text{CTCLoss}=-\text{ctc\_forward\_log}
}
$$

概率越大，负 Log 概率越小；当概率为1时，CTC Loss 为0。

### 2.2 CTCLoss 的输入

PyTorch CTCLoss 要求四个主要输入参数：

- log_probs
- targets
- input_lengths
- target_lengths

**（1）log_probs**

表示模型输出的 Log 概率。

PyTorch 要求形状为：

$$
(T,N,C)
$$

其中：

- T：时间帧数量。
- N：Batch Size。
- C：词表大小。

之前自己实现的 `ctc_forward_log()` 接收形状为 `(T,C)` 的输入，而 PyTorch CTCLoss 需要额外增加 Batch 维度。

例如：

```python
log_probs.shape = (12, 6)

log_probs = log_probs.unsqueeze(1)

log_probs.shape = (12, 1, 6)
```

当前只有一条样本，因此 N=1。

**（2）targets**

表示人工提供的目标 Token 序列，即 Reference。

PyTorch 支持两种形式：

- 一维拼接形式：将 Batch 中各样本的目标序列拼接起来。
- 二维 Padding 形式：将不同长度的目标序列补齐到相同长度。

当前项目只处理单条样本，因此使用一维形式。

例如：

```python
target = [1, 2, 3]

targets = torch.tensor(
    target,
    dtype=torch.long
)
```

形状为 `(3,)`。

注意：targets 中不能包含 Blank。

**（3）input_lengths**

表示每个样本实际参与 CTC 计算的时间帧数量。

例如：

```python
input_lengths = torch.tensor(
    [12],
    dtype=torch.long
)
```

当前只有一条样本，因此形状为 `(1,)`。

这里的长度是模型输出的时间帧数量，不是原始音频采样点数量，也不是音频时长。

**（4）target_lengths**

表示每个样本的真实目标 Token 数量。

例如：

```python
target = [1, 2, 3]

target_lengths = torch.tensor(
    [3],
    dtype=torch.long
)
```

形状为 `(1,)`。

如果目标为空，则目标长度为0。

**（5）输入参数汇总**

| 参数 | 当前单样本形状 | 含义 |
|---|---|---|
| log_probs | (T,1,C) | 模型输出的 Log 概率 |
| targets | (U,) | 目标 Token 序列 |
| input_lengths | (1,) | 输入时间帧长度 |
| target_lengths | (1,) | 目标序列长度 |

长度 Tensor 使用 CPU 上的 `torch.long`，是兼容常见 CUDA/cuDNN 调用要求的稳妥方式。

### 2.3 reduction 参数

`reduction` 决定如何汇总 Batch 中各个样本的损失。

主要有三种取值：

**（1）reduction='none'**

不对 Batch 中的损失进行汇总，保留每个样本独立的损失。

例如 Batch Size 为3：

```text
loss.shape = (3,)
```

每个位置对应一条样本的 CTC Loss。

**（2）reduction='sum'**

将所有样本的损失直接相加，最终返回标量。

**（3）reduction='mean'**

先将每个样本的损失除以其对应的目标长度，再对 Batch 中的样本取平均。

假设两个样本的原始损失分别为6和12，目标长度分别为3和4。

那么：

```text
none: [6, 12]

sum: 18

mean: (6/3 + 12/4)/2 = 2.5
```

**为什么使用 reduction='none'？**

当前任务是验证自己实现的 CTC 前向算法是否正确，需要逐条样本比较：

- 暴力枚举结果。
- 动态规划结果。
- PyTorch CTCLoss 结果。

因此，需要保留每条样本的独立损失，而不是提前对 Batch 进行汇总。

另外，`mean` 会额外按照目标长度进行归一化，不能直接与自己实现的单条样本 Log 概率比较。

### 2.4 CTCLoss 的代码实现

**（1）构造输入参数**

假设：

```python
probs.shape = (12, 6)

target = [1, 2, 3]

T = 12
```

其中 probs 必须是已经归一化的概率。

```python
# 转换为 Log 概率并增加 Batch 维度
log_probs = torch.log(probs).unsqueeze(1)

# 构造目标序列
targets = torch.tensor(
    target,
    dtype=torch.long,
    device=log_probs.device
)

# 输入时间帧长度
input_lengths = torch.tensor(
    [T],
    dtype=torch.long
)

# 目标 Token 长度
target_lengths = torch.tensor(
    [len(target)],
    dtype=torch.long
)
```

如果输入是原始 logits，则应该先进行 Log Softmax：

```python
log_probs = torch.log_softmax(
    logits,
    dim=-1
).unsqueeze(1)
```

**（2）调用 CTCLoss**

```python
criterion = torch.nn.CTCLoss(
    blank=0,
    reduction="none",
    zero_infinity=False
)

loss = criterion(
    log_probs,
    targets,
    input_lengths,
    target_lengths
)
```

当前 Batch Size 为1，因此：

```python
loss.shape == (1,)
```

使用 `loss[0]` 可以得到当前样本的标量损失。

**（3）封装函数**

```python
def torch_ctc_loss_one(
    log_probs,
    target,
    blank=0
):
    """
    使用 PyTorch CTCLoss 计算单条样本的 CTC Loss。

    参数：
        log_probs: (T, C)，Log 概率
        target: 原始目标 Token 序列
        blank: Blank ID

    返回：
        标量 Tensor，负 Log 概率
    """

    T, C = log_probs.shape

    # 增加 Batch 维度
    log_probs_batch = log_probs.unsqueeze(1)

    # 构造目标序列
    targets = torch.tensor(
        target,
        dtype=torch.long,
        device=log_probs.device
    )

    # 输入长度
    input_lengths = torch.tensor(
        [T],
        dtype=torch.long
    )

    # 目标长度
    target_lengths = torch.tensor(
        [len(target)],
        dtype=torch.long
    )

    criterion = torch.nn.CTCLoss(
        blank=blank,
        reduction="none",
        zero_infinity=False
    )

    loss = criterion(
        log_probs_batch,
        targets,
        input_lengths,
        target_lengths
    )

    return loss[0]
```

### 2.5 三种方法的结果对照

现在有三种计算方法：

1. `ctc_bruteforce_log()`：暴力枚举。
2. `ctc_forward_log()`：动态规划。
3. `torch_ctc_loss_one()`：PyTorch CTCLoss。

它们应该满足：

$$
\text{BruteForce}
\approx
\text{DP}
$$

$$
\text{DP}
\approx
-\text{CTCLoss}
$$

即：

$$
\boxed{
\text{BruteForce}
\approx
\text{DP}
\approx
-\text{CTCLoss}
}
$$

**示例：**

假设：

```python
probs = torch.tensor(
    [
        [0.6, 0.4],
        [0.3, 0.7]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

target = [1]
```

计算得到：

$$
P(y|x)=0.82
$$

因此：

$$
\log P(y|x)\approx-0.19845
$$

三种方法的结果为：

| 方法 | 结果 |
|---|---:|
| 暴力枚举 | -0.19845 |
| 动态规划 | -0.19845 |
| PyTorch CTCLoss | 0.19845 |

PyTorch CTCLoss 的结果与前两者符号相反，这是因为它返回负 Log 概率。

**自动对照测试：**

```python
def compare_ctc_implementations(
    log_probs,
    target,
    blank=0
):
    """
    对照暴力枚举、动态规划和 PyTorch CTCLoss。
    """

    brute_result = ctc_bruteforce_log(
        log_probs,
        target,
        blank
    )

    dp_result = ctc_forward_log(
        log_probs,
        target,
        blank
    )

    torch_loss = torch_ctc_loss_one(
        log_probs,
        target,
        blank
    )

    # 比较暴力枚举和 DP
    torch.testing.assert_close(
        brute_result,
        dp_result,
        atol=1e-5,
        rtol=1e-5
    )

    # 比较 DP 和负 CTCLoss
    torch.testing.assert_close(
        dp_result,
        -torch_loss,
        atol=1e-5,
        rtol=1e-5
    )

    return {
        "brute_log_probability": brute_result,
        "dp_log_probability": dp_result,
        "torch_ctc_loss": torch_loss
    }
```

### 2.6 zero_infinity

**（1）zero_infinity 的作用**

当不存在合法 CTC 对齐路径时：

- 目标序列概率为0。
- Log 概率为负无穷。
- CTC Loss 为正无穷。

设置：

```python
zero_infinity=False
```

保留无穷损失。

设置：

```python
zero_infinity=True
```

则将无穷损失及其梯度置零。

**（2）为什么正确性验证必须使用 False？**

因为不可能对齐本身就是需要发现的边界情况。

如果设置为 True，PyTorch 可能将本来应该返回正无穷的损失置为0，掩盖输入长度计算错误或者不可能对齐的问题。

因此，正确性验证统一使用：

```python
zero_infinity=False
```

**（3）最小可行时间帧长度**

设：

- U：目标序列长度。
- R：目标序列中相邻相同 Token 对的数量。

那么最小时间帧长度为：

$$
T_{\min}=U+R
$$

例如：

```python
target = [1, 1, 2, 2]
```

其中：

- U=4。
- R=2。

因此：

$$
T_{\min}=4+2=6
$$

当 T 小于6时，不存在合法对齐路径。

### 2.7 本模块总结

**（1）三种计算方法**

| 方法 | 函数 | 返回值 |
|---|---|---|
| 暴力枚举 | `ctc_bruteforce_log()` | Log 概率 |
| 动态规划 | `ctc_forward_log()` | Log 概率 |
| PyTorch | `torch_ctc_loss_one()` | 负 Log 概率 |

**（2）输入形状**

| 参数 | 形状 | 含义 |
|---|---|---|
| log_probs | (T,N,C) | 模型 Log 概率 |
| targets | (U,) | 拼接后的目标 Token |
| input_lengths | (N,) | 输入时间帧长度 |
| target_lengths | (N,) | 目标序列长度 |

当前单样本情况下：

- log_probs 从 (T,C) 转换为 (T,1,C)。
- targets 为一维 Token 序列。
- input_lengths 为 [T]。
- target_lengths 为 [U]。

**（3）reduction**

- none：保留每个样本独立的损失。
- sum：对所有样本损失求和。
- mean：先按照目标长度归一化，再对 Batch 取平均。

正确性验证使用 `reduction="none"`。

**（4）zero_infinity**

- False：保留无穷损失。
- True：将无穷损失及其梯度置零。

正确性验证使用 `False`，避免掩盖不可能对齐和输入长度错误。

**（5）核心关系**

$$
\text{CTCLoss}=-\log P(y|x)
$$

$$
\boxed{
\text{BruteForce}
\approx
\text{DP}
\approx
-\text{CTCLoss}
}
$$

**（6）本模块核心认识**

PyTorch CTCLoss 与自己实现的 CTC 前向算法在数学上计算的是同一个目标序列概率，只是优化方向不同。

自己的实现返回 Log 概率，而 PyTorch CTCLoss 返回负 Log 概率。

因此，可以通过三种不同的计算方式相互验证，提高 CTC 算法实现的可信度。


# W8D3 模块3：边界情况与长度检查

## 3. CTC 边界情况测试

### 3.1 为什么需要边界测试？

普通目标测试只能验证 CTC 算法在一般情况下能否正确计算目标序列概率，但不能保证算法正确处理所有特殊情况。

CTC 实现中存在一些容易出错的边界情况：

- 相邻重复标签：检验 Blank 插入和 Skip 转移条件。
- 空目标：检验空目标的初始化和终止状态。
- 不可能对齐：检验最小时间帧长度和负无穷处理。

因此，需要通过边界测试进一步验证暴力枚举、动态规划和 PyTorch CTCLoss 三种实现。

### 3.2 最小时间帧长度

**（1）相关变量**

- T：模型输出的时间帧数量。
- U：原始目标序列的 Token 数量。
- R：目标序列中相邻相同 Token 对的数量。

例如：

```python
target = [1, 1, 2, 2]
```

其中：

- U=4。
- R=2，因为存在两组相邻重复 Token。

**（2）最小时间帧长度公式**

$$
\boxed{T_{\min}=U+R}
$$

推导思路：

每个目标 Token 至少需要一个时间帧，因此需要 U 帧。

对于相邻相同 Token，必须在它们之间插入 Blank，否则 CTC Collapse 会将它们合并。因此，每一组相邻重复 Token 还需要额外一个时间帧。

所以：

$$
T_{\min}=U+R
$$

**（3）判断能否完成对齐**

- 当 T≥U+R 时，存在合法对齐路径。
- 当 T小于U+R时，不存在合法对齐路径。

例如：

```python
target = [1, 1, 2, 2]
```

$$
U=4,\quad R=2
$$

$$
T_{\min}=4+2=6
$$

如果 T=6，可以完成对齐。

如果 T=5，则无法完成对齐。

注意：不同 Token 之间不要求必须插入 Blank，只有相邻相同 Token 之间必须插入 Blank。

### 3.3 相邻重复标签测试

**（1）测试目标**

```python
target = [1, 1]
blank = 0
```

目标长度：

$$
U=2
$$

相邻重复对数：

$$
R=1
$$

最小时间帧长度：

$$
T_{\min}=2+1=3
$$

**（2）合法对齐**

当 T=3 时，可以使用路径：

```text
[1, 0, 1]
```

经过 CTC Collapse：

```text
[1, 0, 1]
    ↓
[1, 0, 1]
    ↓
[1, 1]
```

因此能够得到目标序列。

**（3）不可能对齐**

当 T=2 时：

```text
[1, 1]
```

经过 CTC Collapse：

```text
[1]
```

无法得到目标 `[1,1]`。

因此，相邻重复标签测试主要检查：

- 是否正确插入 Blank。
- Skip 两格转移条件是否正确。
- 是否正确识别不可能对齐的情况。

### 3.4 空目标测试

**（1）空目标的定义**

空目标表示最终识别结果不包含任何 Token：

```python
target = []
```

其扩展目标只有一个 Blank：

```text
[blank]
```

**（2）合法路径**

当 T=3 时，只有一条合法路径：

```text
[0, 0, 0]
```

因为只要任意一个时间帧输出非 Blank，最终就会产生非空输出。

**（3）目标序列概率**

假设三个时间帧的 Blank 概率分别为：

```text
0.5, 0.6, 0.8
```

那么：

$$
P(\varnothing|x)
=0.5\times0.6\times0.8
$$

$$
\boxed{P(\varnothing|x)=0.24}
$$

Log 概率：

$$
\log P(\varnothing|x)
=\log0.5+\log0.6+\log0.8
$$

$$
\boxed{\log P(\varnothing|x)=\log0.24}
$$

**（4）CTCLoss**

PyTorch CTCLoss 返回负 Log 概率：

$$
L_{\mathrm{CTC}}
=-\log P(\varnothing|x)
$$

$$
L_{\mathrm{CTC}}=-\log0.24
$$

因此：

- 暴力枚举返回 log(0.24)。
- 动态规划返回 log(0.24)。
- PyTorch CTCLoss 返回 -log(0.24)。

### 3.5 不可能路径测试

**（1）不可能对齐的条件**

当输入时间帧数量小于最小时间帧长度时，不存在合法对齐路径。

$$
T<U+R
$$

例如：

```python
target = [1, 1]
T = 2
```

因为：

$$
T_{\min}=2+1=3
$$

而：

$$
T=2<3
$$

所以不存在合法路径。

**（2）三种方法的结果**

不可能对齐意味着目标序列概率为0。

$$
P(y|x)=0
$$

Log 概率：

$$
\log P(y|x)=-\infty
$$

CTCLoss：

$$
-\log P(y|x)=+\infty
$$

因此：

| 方法 | 预期结果 |
|---|---|
| 暴力枚举 | -inf |
| 动态规划 | -inf |
| PyTorch CTCLoss | +inf |

**（3）zero_infinity**

PyTorch CTCLoss 提供参数：

```python
zero_infinity=False
```

当不存在合法对齐路径时，保留正无穷损失。

如果设置：

```python
zero_infinity=True
```

则将无穷损失及其梯度置零。

这可能掩盖输入长度计算错误或不可能对齐的问题。

因此，正确性验证必须使用：

```python
zero_infinity=False
```

不能通过将无穷损失置零来掩盖算法错误。

### 3.6 测试组织

所有测试统一放在：

```text
tests/test_ctc.py
```

使用三个实现进行交叉验证：

```python
ctc_bruteforce_log()
ctc_forward_log()
torch_ctc_loss_one()
```

建议组织以下测试函数：

```python
def test_repeated_target():
    ...

def test_empty_target():
    ...

def test_impossible_alignment():
    ...

def test_minimum_input_length():
    ...
```

**（1）相邻重复标签测试**

```python
target = [1, 1]
T = 3
```

验证三种实现的结果一致，并检查目标概率为0.125（当每帧 Blank 和 Token 概率均为0.5时）。

**（2）空目标测试**

```python
target = []
T = 3
```

验证：

$$
P(\varnothing|x)=0.24
$$

**（3）不可能对齐测试**

```python
target = [1, 1]
T = 2
```

验证：

- 暴力枚举为负无穷。
- 动态规划为负无穷。
- PyTorch CTCLoss 为正无穷。

**（4）最小时间帧长度测试**

```python
target = [1, 1, 2, 2]
```

验证：

$$
T_{\min}=4+2=6
$$

T=6 时存在合法对齐，T=5 时不存在合法对齐。

**（5）结果比较**

对于有限结果：

```python
torch.testing.assert_close(
    brute_result,
    dp_result,
    atol=1e-5,
    rtol=1e-5
)

torch.testing.assert_close(
    dp_result,
    -torch_loss,
    atol=1e-5,
    rtol=1e-5
)
```

对于不可能对齐：

```python
assert torch.isneginf(brute_result)
assert torch.isneginf(dp_result)
assert torch.isposinf(torch_loss)
```

运行：

```powershell
python -m pytest tests/test_ctc.py -v
```

### 3.7 本模块总结

**（1）最小时间帧长度**

$$
T_{\min}=U+R
$$

其中 U 是目标长度，R 是相邻重复 Token 对数。

**（2）相邻重复标签**

相邻相同 Token 之间必须有 Blank，否则 Collapse 时会被合并。

**（3）空目标**

空目标只有一条合法路径，即所有时间帧均输出 Blank。

$$
P(\varnothing|x)
=\prod_{t=0}^{T-1}p_t(blank)
$$

**（4）不可能对齐**

$$
T<U+R
$$

此时：

$$
P(y|x)=0
$$

$$
\log P(y|x)=-\infty
$$

$$
L_{\mathrm{CTC}}=+\infty
$$

**（5）zero_infinity**

正确性验证必须使用：

```python
zero_infinity=False
```

避免将不可能对齐产生的无穷损失置零，从而掩盖输入长度错误。

**（6）本模块核心认识**

边界测试不是为了增加测试数量，而是为了验证算法在特殊情况下仍然符合 CTC 的数学定义。

相邻重复标签、空目标和不可能对齐分别检验不同的算法逻辑。只有这些情况都通过，才能更有把握地认为自己实现的 CTC 前向算法是正确的。