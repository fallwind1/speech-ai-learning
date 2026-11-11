# W8D4 模块1：Greedy Decoding

## 1. Greedy Decoding

### 1.1 输入与输出

**输入：**

模型输出的 Log 概率矩阵。

$$
\text{log\_probs}\in\mathbb{R}^{T\times C}
$$

其中：

- T：时间帧数量。
- C：Token 类别数量。

**输出：**

最终识别得到的 Token 序列。

**解码目标：**

根据模型每个时间帧输出的概率分布，得到对应的识别结果。

### 1.2 Greedy 解码过程

Greedy Decoding 包含两个核心步骤：

**第一步：逐帧 Argmax**

每个时间帧独立选择概率最大的 Token。

$$
\pi_t=\arg\max_c p_t(c)
$$

由于 Log 函数严格单调递增，因此可以直接对 Log 概率使用 Argmax。

$$
\arg\max_c p_t(c)
=
\arg\max_c \log p_t(c)
$$

不需要先将 Log 概率转换为普通概率。

**第二步：CTC Collapse**

将逐帧得到的最佳路径进行 CTC 折叠：

1. 合并相邻的相同 Token。
2. 删除 Blank。

因此：

$$
\boxed{
\text{Greedy Decoding}
=
\text{Argmax}+\text{CTC Collapse}
}
$$

## 2. CTC Collapse

### 2.1 两个操作

**操作一：合并相邻重复 Token**

只合并连续出现的相同 Token。

例如：

```text
a a b b
  ↓
a b
```

注意：不是全局去重，只处理相邻重复。

**操作二：删除 Blank**

将第一步得到的序列中的 Blank 删除。

例如：

```text
a _ b
  ↓
a b
```

**操作顺序：**

必须先合并相邻重复 Token，再删除 Blank。

不能颠倒顺序，因为 Blank 可以分隔两个相同 Token，使其最终能够连续出现。

### 2.2 典型例子

**例1：**

```text
a a
 ↓
a
```

最终输出：

```text
a
```

**例2：**

```text
a _ a
  ↓
a _ a
  ↓
a a
```

最终输出：

```text
aa
```

**例3：**

```text
a a _ a b
    ↓
a _ a b
    ↓
a a b
```

最终输出：

```text
aab
```

**例4：**

```text
a a b b _ b
      ↓
a b _ b
      ↓
a b b
```

最终输出：

```text
abb
```

### 2.3 常见错误

**错误一：先删除 Blank，再合并重复 Token。**

例如：

```text
a _ a
```

错误处理：

```text
a _ a
  ↓ 删除Blank
a a
  ↓ 合并重复
a
```

正确处理：

```text
a _ a
  ↓ 合并相邻重复
a _ a
  ↓ 删除Blank
a a
```

因此，Blank 不能提前删除。

**错误二：全局去重。**

例如：

```text
a _ a
```

两个 a 虽然相同，但中间有 Blank，不属于相邻重复，最终应该保留两个 a。

**错误三：遇到 Blank 不更新 previous。**

Blank 虽然不会加入最终输出，但它会打断前后两个 Token 的相邻关系。

因此，每次循环都必须更新 previous，包括当前 Token 为 Blank 的情况。

## 3. Greedy 解码算法

### 3.1 核心代码逻辑

**（1）获取每帧最大概率 Token**

```python
best_path = log_probs.argmax(dim=-1)
```

输入形状：

```text
(T, C)
```

输出形状：

```text
(T,)
```

**（2）执行 CTC Collapse**

```python
def ctc_collapse(path, blank=0):

    output = []
    previous = None

    for token in path:

        if token != previous and token != blank:
            output.append(token)

        previous = token

    return output
```

注意：

```python
previous = token
```

必须在每次循环中执行，不能只在非 Blank 情况下执行。

**（3）完整 Greedy 解码**

```python
def ctc_greedy_decode(log_probs, blank=0):

    if log_probs.ndim != 2:
        raise ValueError(
            "log_probs must have shape (T, C)"
        )

    if not 0 <= blank < log_probs.shape[1]:
        raise ValueError("invalid blank ID")

    best_path = (
        log_probs
        .argmax(dim=-1)
        .detach()
        .cpu()
        .tolist()
    )

    return ctc_collapse(
        best_path,
        blank=blank
    )
```

### 3.2 时间复杂度

设：

- T：时间帧数量。
- C：Token 类别数量。

逐帧 Argmax 需要遍历每个时间帧的所有类别，因此时间复杂度为：

$$
O(TC)
$$

Collapse 只需要遍历一次路径：

$$
O(T)
$$

因此整体时间复杂度为：

$$
\boxed{O(TC)}
$$

额外空间复杂度为：

$$
O(T)
$$

主要用于保存最佳路径和最终输出序列。

## 4. 本模块总结

**（1）Greedy Decoding**

每个时间帧独立选择概率最大的 Token，然后执行 CTC Collapse。

**（2）CTC Collapse**

必须先合并相邻重复 Token，再删除 Blank。

**（3）Blank 的作用**

Blank 不会出现在最终识别结果中，但可以分隔相同 Token，使其最终连续出现。

**（4）实现注意事项**

- 可以直接对 Log 概率使用 Argmax。
- 只能合并相邻重复 Token，不能全局去重。
- Blank 也必须更新 previous。
- 不能先删除 Blank 再合并重复 Token。

**（5）核心认识**

Greedy Decoding 是一种简单的 CTC 解码方法，它只考虑每个时间帧概率最大的 Token，不会同时考虑其他可能路径。

因此，它得到的是最佳单条路径经过 Collapse 后的结果，不一定是概率最大的最终识别序列。

这一点将在下一个模块进一步讨论。


# W8D4 模块2：Best Path 与 Best Collapsed Sequence

## 1. CTC 路径概率

### 1.1 Path 的定义

Path 是模型在每个时间帧输出的完整 Token 序列，包含 Blank 和重复 Token。

例如：

```text
[a, _, b]
[a, a, b]
[a, b, b]
[_, a, b]
```

这些都是不同的 Path。

在 CTC 的条件独立假设下，一条完整路径的概率等于各时间帧对应 Token 概率的乘积。

$$
P(\pi|x)=\prod_{t=1}^{T}p_t(\pi_t|x)
$$

其中：

- T：时间帧数量。
- π：一条完整路径。
- π_t：路径在第 t 个时间帧输出的 Token。
- p_t：模型在第 t 个时间帧输出的概率分布。

### 1.2 Sequence 的定义

Sequence 指 Path 经过 CTC Collapse 后得到的最终 Token 序列。

CTC Collapse 包含两个步骤：

1. 合并相邻的相同 Token。
2. 删除 Blank。

不同的 Path 可以得到相同的 Sequence。

例如：

| Path | Collapse 后的 Sequence |
|---|---|
| `[a, _, b]` | `[a,b]` |
| `[a, a, b]` | `[a,b]` |
| `[a, b, b]` | `[a,b]` |
| `[_, a, b]` | `[a,b]` |

因此，一个 Sequence 可能对应多条不同的 Path。

## 2. Best Path

### 2.1 定义

Best Path 指所有可能 Path 中概率最大的那一条路径。

$$
\pi^*=\arg\max_{\pi}P(\pi|x)
$$

由于 CTC 假设给定输入 x 后，各时间帧的输出条件独立，因此一条路径的概率是各时间帧 Token 概率的乘积。

要获得概率最大的完整路径，只需要每个时间帧独立选择概率最大的 Token。

因此：

```python
best_path = log_probs.argmax(dim=-1)
```

Greedy Decoding 得到的就是 Best Path。

### 2.2 Best Path 的局限

Best Path 经过 CTC Collapse 后得到的结果，是 Greedy Decoding 的最终输出。

但是，这个结果不一定是概率最大的最终 Sequence。

原因是：Best Path 只考虑概率最大的单条路径，没有考虑其他能够得到相同 Sequence 的路径。

因此，需要进一步考虑 Sequence 的累计概率。

## 3. Best Collapsed Sequence

### 3.1 定义

一个 Sequence 可能对应多条不同的 Path。

因此，Sequence 的概率不是某一条 Path 的概率，而是所有能够通过 CTC Collapse 得到该 Sequence 的 Path 概率之和。

$$
P(y|x)=
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

其中：

- y：最终 Sequence。
- π：一条可能的 Path。
- B(π)：对 Path 执行 CTC Collapse。
- P(π|x)：该 Path 的概率。

Best Collapsed Sequence 指所有最终 Sequence 中累计概率最大的一个。

$$
y^*=\arg\max_y P(y|x)
$$

展开为：

$$
y^*=
\arg\max_y
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

### 3.2 为什么需要对多条 Path 求和？

不同 Path 可能对应同一个 Sequence。

如果只考虑概率最大的单条 Path，就会遗漏其他路径对该 Sequence 的概率贡献。

因此，必须将所有能够得到该 Sequence 的路径概率累加，才能得到完整的 Sequence 概率。

## 4. 数值例子

假设三个时间帧的输出概率如下：

| 时间帧 | Blank | a | b |
|---|---:|---:|---:|
| T1 | 0.3 | 0.3 | 0.4 |
| T2 | 0.2 | 0.5 | 0.3 |
| T3 | 0.1 | 0.3 | 0.6 |

### 4.1 Greedy 得到 Best Path

逐帧选择概率最大的 Token：

```text
T1：b
T2：a
T3：b
```

因此：

```text
Best Path = [b, a, b]
```

其概率：

$$
P(bab|x)=0.4\times0.5\times0.6
$$

$$
P(bab|x)=0.12
$$

Collapse 后得到：

```text
[b, a, b]
```

即最终 Sequence 为 `bab`。

### 4.2 计算 Sequence [a,b] 的概率

以下五条 Path 都能通过 CTC Collapse 得到 `[a,b]`：

| Path | Path 概率 |
|---|---:|
| `[_,a,b]` | 0.3×0.5×0.6 = 0.090 |
| `[a,_,b]` | 0.3×0.2×0.6 = 0.036 |
| `[a,a,b]` | 0.3×0.5×0.6 = 0.090 |
| `[a,b,_]` | 0.3×0.3×0.1 = 0.009 |
| `[a,b,b]` | 0.3×0.3×0.6 = 0.054 |

所以：

$$
P(ab|x)=0.090+0.036+0.090+0.009+0.054
$$

$$
\boxed{P(ab|x)=0.279}
$$

### 4.3 比较 Best Path 与 Best Collapsed Sequence

Greedy 得到：

```text
bab
```

其 Sequence 概率为：

$$
P(bab|x)=0.12
$$

而：

$$
P(ab|x)=0.279
$$

因此：

$$
P(ab|x)>P(bab|x)
$$

为了确认 ab 确实是所有最终序列中概率最大的，还需要考虑其他 Sequence。

对该三帧示例的全部 27 条路径进行枚举，得到：

| Sequence | 概率 |
|---|---:|
| ab | 0.279 |
| b | 0.191 |
| ba | 0.167 |
| a | 0.144 |
| bab | 0.120 |
| bb | 0.048 |
| aba | 0.027 |
| aa | 0.018 |
| 空序列 | 0.006 |

因此，概率最大的最终 Sequence 确实是 `ab`。

这个例子说明：

Greedy 得到的 Best Path 是 `[b,a,b]`，但其 Collapse 结果 `bab` 并不是概率最大的最终 Sequence。

## 5. Greedy Decoding 的局限

Greedy Decoding 实际优化的是单条路径概率：

$$
\pi^*=\arg\max_\pi P(\pi|x)
$$

然后执行 CTC Collapse：

$$
\hat y=B(\pi^*)
$$

而真正的 Best Collapsed Sequence 是：

$$
y^*=
\arg\max_y
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

因此：

$$
\boxed{
B(\arg\max_\pi P(\pi|x))
\neq
\arg\max_y
\sum_{\pi:B(\pi)=y}P(\pi|x)
}
$$

Greedy 的局限在于：

- 每个时间帧只选择概率最大的 Token。
- 只考虑概率最大的单条路径。
- 没有对不同路径对应的同一 Sequence 进行概率累加。
- 因此不一定找到概率最大的最终识别序列。

## 6. 与 Beam Search 的关系

### 6.1 Beam Search 的基本思想

Beam Search 是一种近似搜索算法。

与 Greedy 每一步只保留一个最佳选择不同，Beam Search 每一步保留多个高概率候选前缀，并继续扩展。

例如：

- Greedy：只保留一个候选。
- Beam Search：保留多个候选。

CTC Prefix Beam Search 会对相同的折叠前缀进行概率合并，并区分以 Blank 和非 Blank 结尾的概率。

这样能够考虑更多可能路径对最终 Sequence 的贡献。

### 6.2 为什么 Beam Search 不保证识别质量一定提升？

**原因一：Beam Search 是近似搜索。**

Beam Search 会限制候选数量。

如果真正最优的候选在某一步被剪枝，后续就无法再找到它。

因此，有限 Beam Width 不保证找到全局概率最优序列。

**原因二：模型概率不等于实际识别质量。**

Beam Search 主要搜索模型认为概率较大的序列。

但是 ASR 实际评价通常使用 CER、WER 等指标。

概率最大的 Sequence 不一定与人工 Reference 的编辑距离最小。

因此，即使 Beam Search 找到全局概率最优序列，也不保证 CER 一定下降。

## 7. 本模块总结

**（1）四个核心概念**

| 概念 | 含义 |
|---|---|
| Path | 每个时间帧输出的完整 Token 序列 |
| Sequence | Path 经过 CTC Collapse 后的最终序列 |
| Best Path | 概率最大的单条路径 |
| Best Collapsed Sequence | 累计概率最大的最终序列 |

**（2）单条路径概率**

$$
P(\pi|x)=\prod_{t=1}^{T}p_t(\pi_t|x)
$$

**（3）最终序列概率**

$$
P(y|x)=
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

**（4）两种优化目标**

$$
\pi^*=\arg\max_\pi P(\pi|x)
$$

$$
y^*=\arg\max_y
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

**（5）核心认识**

Greedy Decoding 寻找概率最大的单条 Path，再执行 Collapse。

Best Collapsed Sequence 则寻找所有最终 Sequence 中累计概率最大的一个。

由于不同 Path 可以对应相同 Sequence，因此两种结果不一定相同。

Beam Search 通过保留多个候选前缀，尝试搜索概率更高的最终 Sequence，但它是近似搜索，也不保证实际 CER 一定下降。


# W8D4 模块3：CTC Prefix Beam Search

## 1. Prefix Beam Search 的基本思想

### 1.1 Prefix 的定义

Prefix 表示当前搜索得到的候选输出前缀，是最终识别序列的一个部分。

例如：

```text
[a]
[a,b]
[a,b,c]
```

它们都可以是搜索过程中的 Prefix。

三个概念的区别：

| 概念 | 含义 |
|---|---|
| Path | 每个时间帧实际输出的完整 Token 序列 |
| Prefix | 当前搜索得到的候选输出前缀 |
| Sequence | 最终完整的识别序列 |

Path 包含 Blank 和重复 Token，而 Prefix 和 Sequence 是经过 CTC Collapse 后得到的序列。

不同 Path 可能对应相同 Prefix，因此需要对这些路径的概率进行合并。

Prefix Beam Search 的基本过程：

1. 每个时间帧扩展当前已有的多个 Prefix。
2. 将得到相同 Prefix 的路径概率合并。
3. 保留概率较高的若干 Prefix。
4. 继续处理下一个时间帧。

### 1.2 Beam Width

Beam Width（束宽）表示每个时间步最多保留多少个候选 Prefix。

例如：

- Beam Width=1：最多保留一个候选。
- Beam Width=3：最多保留三个候选。
- Beam Width=10：最多保留十个候选。

每次扩展候选之后，按照 Prefix 的累计概率排序，保留得分最高的 K 个。

Beam Search 是近似搜索，因为它会主动删除得分较低的候选。

某个 Prefix 当前概率虽然较低，但后续可能具有更有利的扩展。如果它在当前时间步被剪枝，后续就无法再找到对应的最终序列。

因此，有限 Beam Width 不保证找到全局最优序列。

## 2. 两种概率状态

### 2.1 p_blank

对于当前 Prefix l，p_blank(l) 表示：

所有经过 CTC Collapse 后得到 Prefix l，并且最后一个时间帧输出 Blank 的路径概率之和。

例如，以下路径都可能经过 Collapse 得到 `[a]`，并且以 Blank 结尾：

```text
[a, _]
[a, a, _]
[a, _, _]
```

它们的概率都应该累加到 Prefix `[a]` 的 p_blank 中。

### 2.2 p_nonblank

对于当前 Prefix l，p_nonblank(l) 表示：

所有经过 CTC Collapse 后得到 Prefix l，并且最后一个时间帧输出非 Blank Token 的路径概率之和。

例如：

```text
[a]
[a, a]
[_, a]
```

这些路径都可能经过 Collapse 得到 `[a]`，并且以非 Blank 结尾。

它们的概率都应该累加到 Prefix `[a]` 的 p_nonblank 中。

### 2.3 为什么不能直接合并两种状态？

虽然 p_blank 和 p_nonblank 对应的最终 Prefix 相同，但它们的最后输出状态不同。

假设当前 Prefix 为 `[a]`，下一帧输出 `a`：

```text
从非 Blank 状态：

[a] + a → [a,a] → [a]
```

```text
从 Blank 状态：

[a,_] + a → [a,_,a] → [a,a]
```

第一种情况中两个 a 相邻，Collapse 后仍然是 `[a]`。

第二种情况中两个 a 被 Blank 分隔，Collapse 后得到 `[a,a]`。

因此，下一帧的相同 Token 可能产生不同的 Prefix，必须分别维护两种概率状态。

### 2.4 Prefix 总概率

一个 Prefix 对应的所有路径可以分成两类：

- 最后以 Blank 结尾。
- 最后以非 Blank 结尾。

因此，普通概率空间中：

p(l) = p_blank(l) + p_nonblank(l)

Log 空间中：

$$
\log P(l)
=
\operatorname{logaddexp}
\left(
\log p_{\text{blank}}(l),
\log p_{\text{nonblank}}(l)
\right)
$$

Prefix 总概率用于 Beam Pruning。

## 3. Prefix 的状态转移

设：

- l：当前 Prefix。
- c：当前时间帧输出的 Token。
- last(l)：当前 Prefix 最后一个 Token。
- p_t(c)：当前时间帧输出 Token c 的概率。
- p_blank 和 p_nonblank：上一时间帧对应的概率状态。

### 3.1 当前输出 Blank

无论上一时间帧以 Blank 还是非 Blank 结尾，只要当前输出 Blank，Collapse 后的 Prefix 都不会改变。

因此，两种前驱状态都转移到原 Prefix 的 p_blank。

普通概率空间：

$$
p_{\text{blank}}^t(l)
=
\left[
p_{\text{blank}}^{t-1}(l)
+
p_{\text{nonblank}}^{t-1}(l)
\right]p_t(blank)
$$

### 3.2 当前输出不同于最后一个 Token 的字符

当 c 不等于当前 Prefix 的最后一个 Token 时，无论上一时间帧以 Blank 还是非 Blank 结尾，都可以直接扩展 Prefix。

例如：

```text
[a] + b → [a,b]
[a,_] + b → [a,b]
```

因此，两种前驱状态都转移到新 Prefix l+c 的 p_nonblank。

$$
p_{\text{nonblank}}^t(l+c)
\mathrel{+}=
\left[
p_{\text{blank}}^{t-1}(l)
+
p_{\text{nonblank}}^{t-1}(l)
\right]p_t(c)
$$

条件：

$$
c\neq last(l)
$$

### 3.3 当前输出与最后一个 Token 相同

当 c 等于 Prefix 的最后一个 Token 时，必须区分两种前驱状态。

**情况一：从非 Blank 状态转移**

两个相同 Token 相邻，经过 Collapse 后仍然是原 Prefix。

因此：

$$
p_{\text{nonblank}}^t(l)
\mathrel{+}=
p_{\text{nonblank}}^{t-1}(l)p_t(c)
$$

**情况二：从 Blank 状态转移**

由于 Blank 分隔了两个相同 Token，经过 Collapse 后得到扩展 Prefix。

因此：

$$
p_{\text{nonblank}}^t(l+c)
\mathrel{+}=
p_{\text{blank}}^{t-1}(l)p_t(c)
$$

条件：

$$
c=last(l)
$$

### 3.4 三种转移总结

| 当前输出 | 前驱状态 | 目标状态 |
|---|---|---|
| Blank | Blank / Nonblank | 原 Prefix 的 Blank |
| 不同 Token | Blank / Nonblank | 扩展 Prefix 的 Nonblank |
| 相同 Token | Nonblank | 原 Prefix 的 Nonblank |
| 相同 Token | Blank | 扩展 Prefix 的 Nonblank |

核心是：当前输出与 Prefix 最后一个 Token 相同时，必须根据前一时间帧是否为 Blank 决定是否扩展 Prefix。

## 4. 数值与路径示例

假设当前 Prefix 为 `[a]`，存在两种不同的路径：

```text
[a]
[a,_]
```

两条路径经过 Collapse 都得到 `[a]`，但最后输出状态不同。

假设：

- p_blank([a]) = 0.2
- p_nonblank([a]) = 0.3
- 当前时间帧输出 a 的概率为 0.4。

### 情况一：从非 Blank 状态转移

```text
[a] + a → [a,a] → [a]
```

概率增量：

$$
0.3\times0.4=0.12
$$

因此，原 Prefix `[a]` 的 p_nonblank 增加 0.12。

### 情况二：从 Blank 状态转移

```text
[a,_] + a → [a,_,a] → [a,a]
```

概率增量：

$$
0.2\times0.4=0.08
$$

因此，新 Prefix `[a,a]` 的 p_nonblank 增加 0.08。

这个例子说明：同一个 Prefix 的两种结束状态，在下一帧输出相同 Token 时会产生不同结果。

## 5. Log 空间计算

### 5.1 为什么使用 Log 概率？

路径概率需要对多个时间帧的概率进行连乘。

时间帧较多时，直接连乘容易出现数值下溢。

使用 Log 概率可以将乘法转换为加法：

$$
\log(ab)=\log a+\log b
$$

### 5.2 如何合并两个 Log 概率？

普通概率空间中：

$$
p=a+b
$$
Log 空间中：

$$
\log p=\log(a+b)
$$

因此：

$$
\log p
=
\operatorname{logaddexp}
(\log a,\log b)
$$

PyTorch 中可以使用：

```python
torch.logaddexp(a, b)
```

也可以使用：

```python
torch.logsumexp(
    torch.stack([a, b]),
    dim=0
)
```

不能直接将两个 Log 概率相加，因为：

$$
\log a+\log b=\log(ab)
$$

这计算的是概率乘积，而不是概率之和。

### 5.3 Prefix 的总 Log 概率

一个 Prefix 的总概率由 Blank 和 Nonblank 两种状态组成。

因此：

$$
\log P(l)
=
\operatorname{logaddexp}
\left(
\log p_{\text{blank}}(l),
\log p_{\text{nonblank}}(l)
\right)
$$

注意：这里的两个输入必须是 Log 概率，而不是普通概率。

## 6. Prefix Beam Search 完整流程

### 6.1 初始化

初始时只有一个空 Prefix：

```python
prefix = ()
```

普通概率空间：

```python
p_blank = 1.0
p_nonblank = 0.0
```

Log 空间：

```python
log_p_blank = 0.0
log_p_nonblank = -float("inf")
```

原因：

- 空 Prefix 在尚未读取时间帧时的概率为 1。
- 此时不存在以非 Blank 结尾的路径，因此概率为 0。

### 6.2 每个时间帧扩展候选

对于当前保留的每个 Prefix，遍历可能的 Token：

1. 输出 Blank：更新原 Prefix 的 Blank 状态。
2. 输出不同 Token：扩展 Prefix。
3. 输出相同 Token：根据前一状态决定保持原 Prefix，还是扩展 Prefix。

### 6.3 合并相同 Prefix

不同路径可能得到相同 Prefix，因此需要将它们的概率合并。

但必须分别维护：

```python
p_blank
p_nonblank
```

不能在状态转移之前直接合并。

### 6.4 Beam Pruning

对每个 Prefix 计算总 Log 概率：

```python
score = torch.logaddexp(
    log_p_blank,
    log_p_nonblank
)
```

按照总概率从高到低排序，保留前 K 个候选。

### 6.5 最终输出

所有时间帧处理完成后，计算每个 Prefix 的总概率：

$$
\log P(l)
=
\operatorname{logaddexp}
\left(
\log p_{\text{blank}}(l),
\log p_{\text{nonblank}}(l)
\right)
$$

选择得分最高的 Prefix，作为最终识别结果。

## 7. 本模块总结

### 核心概念

- Prefix：当前搜索得到的候选输出前缀。
- Beam Width：每个时间步最多保留的候选数量。
- p_blank：所有得到当前 Prefix 且以 Blank 结尾的路径概率之和。
- p_nonblank：所有得到当前 Prefix 且以非 Blank 结尾的路径概率之和。
- Prefix 总概率：两种状态概率之和。

### 核心转移逻辑

1. 当前输出 Blank：两种前驱状态都转移到原 Prefix 的 Blank 状态。
2. 当前输出不同 Token：两种前驱状态都转移到扩展 Prefix 的 Nonblank 状态。
3. 当前输出相同 Token：
   - 从 Nonblank 状态转移，保持原 Prefix。
   - 从 Blank 状态转移，扩展 Prefix。

### 核心公式

普通概率空间：

$$
p_{\text{blank}}^t(l)
=
\left(
p_{\text{blank}}^{t-1}(l)
+
p_{\text{nonblank}}^{t-1}(l)
\right)p_t(blank)
$$

$$
p_{\text{nonblank}}^t(l+c)
\mathrel{+}=
\left(
p_{\text{blank}}^{t-1}(l)
+
p_{\text{nonblank}}^{t-1}(l)
\right)p_t(c)
$$

当 c 与 Prefix 最后一个 Token 相同时：

$$
p_{\text{nonblank}}^t(l)
\mathrel{+}=
p_{\text{nonblank}}^{t-1}(l)p_t(c)
$$

$$
p_{\text{nonblank}}^t(l+c)
\mathrel{+}=
p_{\text{blank}}^{t-1}(l)p_t(c)
$$

Log 空间中，两个概率相加需要使用 LogSumExp 或 logaddexp。

最终记住：

**Prefix Beam Search 的核心不是简单地保留多条路径，而是对相同 Prefix 的路径进行概率合并，并通过 p_blank 和 p_nonblank 保留足够的状态信息，避免重复 Token 被错误处理。**


# W8D4 模块4：Greedy Decoding 实现与验证

## 1. Greedy Decoding 原理

### 1.1 输入与输出

Greedy Decoding 的输入是模型输出的 Log 概率矩阵。

形状为：

`log_probs.shape = (T, C)`

其中：

- T：时间帧数量。
- C：类别数量，包含 Blank。
- log_probs[t, c]：第 t 个时间帧输出类别 c 的 Log 概率。

输出是经过 CTC Collapse 后得到的 Token ID 序列。

### 1.2 Greedy Decoding 的步骤

**第一步：逐帧选择概率最大的 Token。**

```python
best_path = log_probs.argmax(dim=-1)
```

得到 Best Path。

**第二步：合并相邻的重复 Token。**

例如：

```text
[a, a, Blank, b, b]
```

变为：

```text
[a, Blank, b]
```

**第三步：删除 Blank。**

```text
[a, b]
```

最终得到识别结果。

### 1.3 为什么可以逐帧 argmax？

根据 CTC 的条件独立假设，一条 Path 的概率等于各时间帧对应 Token 概率的乘积。

\[
P(\pi|x)=\prod_{t=1}^{T}p_t(\pi_t|x)
\]

要使整个乘积最大，只需要每个时间帧分别选择概率最大的 Token。

因此：

\[
\pi^*=\arg\max_\pi P(\pi|x)
\]

Log 是严格单调递增函数，因此对 Log 概率执行 argmax 与对普通概率执行 argmax 的结果相同。

注意：逐帧 argmax 得到的是 Best Path，而不一定是概率最大的最终 Sequence。

## 2. CTC Collapse 的正确实现

### 2.1 两个操作

CTC Collapse 包含两个步骤：

1. 合并相邻的相同 Token。
2. 删除 Blank。

顺序不能颠倒。

例如：

```text
[a, Blank, a, b]
```

先合并相邻重复 Token：

```text
[a, Blank, a, b]
```

再删除 Blank：

```text
[a, a, b]
```

如果先删除 Blank，就会错误地将两个 a 合并。

### 2.2 previous_token 的作用

在代码实现中，可以使用 previous_token 记录前一个时间帧的 Token。

核心逻辑：

```python
collapsed = []
previous_token = None

for token in path:
    if token != previous_token and token != blank_id:
        collapsed.append(token)

    previous_token = token
```

其中：

- 当前 Token 与 previous_token 不同，才考虑加入结果。
- 当前 Token 不是 Blank，才真正加入结果。
- 无论当前 Token 是否为 Blank，都必须更新 previous_token。

为什么遇到 Blank 也要更新？

因为 Blank 会打断前后相同 Token 的相邻关系。

例如：

```text
[1, 0, 1]
```

正确结果：

```text
[1, 1]
```

如果遇到 Blank 不更新 previous_token，最后一个 1 就可能被错误地合并掉。

## 3. Greedy Decoder 的代码实现

### 3.1 ctc_collapse()

```python
def ctc_collapse(path, blank_id=0):
    collapsed = []
    previous_token = None

    for token in path:
        token = int(token)

        if token != previous_token and token != blank_id:
            collapsed.append(token)

        previous_token = token

    return collapsed
```

作用：将一条完整 Path 转换为最终 Sequence。

### 3.2 ctc_greedy_decode()

```python
def ctc_greedy_decode(log_probs, blank_id=0):

    if not isinstance(log_probs, torch.Tensor):
        raise TypeError("log_probs must be a torch.Tensor")

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape (T, C)")

    if log_probs.shape[1] == 0:
        raise ValueError("C must be greater than 0")

    if not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("blank_id is out of range")

    if log_probs.shape[0] == 0:
        return []

    best_path = log_probs.argmax(dim=-1).tolist()

    return ctc_collapse(best_path, blank_id)
```

作用：

1. 检查输入类型和形状。
2. 对每个时间帧执行 argmax。
3. 得到 Best Path。
4. 执行 CTC Collapse。
5. 返回最终 Token ID 列表。

输入形状为 `(T,C)`，输出为 Python List。

## 4. 回归测试

回归测试用于验证之前已经修正的问题不会再次出现。

### 4.1 普通重复 Token

```python
path = [1, 1, 0, 2, 2]
```

结果：

```python
[1, 2]
```

### 4.2 Blank 分隔的重复 Token

```python
path = [1, 0, 1]
```

结果：

```python
[1, 1]
```

这是最重要的回归测试。

### 4.3 连续 Blank

```python
path = [1, 0, 0, 2]
```

结果：

```python
[1, 2]
```

### 4.4 全 Blank

```python
path = [0, 0, 0]
```

结果：

```python
[]
```

### 4.5 空输入

```python
path = []
```

结果：

```python
[]
```

### 4.6 测试结果

运行：

```bash
pytest tests/test_ctc_decode.py -v
```

测试结果：

```text
6 passed in 2.74s
```

说明目前六个测试用例全部通过。

## 5. Best Path 与 Best Collapsed Sequence

### 5.1 Best Path

Best Path 是所有可能 Path 中概率最大的路径。

\[
\pi^*=\arg\max_\pi P(\pi|x)
\]

Greedy Decoding 通过逐帧 argmax 得到 Best Path。

### 5.2 Best Collapsed Sequence

一个最终 Sequence 可能由多条不同的 Path 得到。

因此，Sequence 的概率是所有对应 Path 概率之和。

\[
P(y|x)=
\sum_{\pi:B(\pi)=y}P(\pi|x)
\]

概率最大的最终 Sequence：

\[
y^*=\arg\max_y P(y|x)
\]

### 5.3 两者的区别

Best Path 关注概率最大的单条路径。

Best Collapsed Sequence 关注所有能够得到同一个最终序列的路径概率之和。

因此：

\[
B(\arg\max_\pi P(\pi|x))
\neq
\arg\max_y
\sum_{\pi:B(\pi)=y}P(\pi|x)
\]

Greedy Decoding 不一定能够得到概率最大的最终 Sequence。

## 6. Beam Search 的局限

Beam Search 通过保留多个候选 Prefix，尝试搜索概率更高的最终 Sequence。

但它不能保证识别质量一定提升。

**原因一：搜索剪枝。**

有限 Beam Width 会主动删除部分候选。如果真正最优的候选被提前删除，就无法再找到它。

**原因二：模型概率不等于实际识别质量。**

Beam Search 优化的是模型输出序列的概率，而 CER 衡量的是预测文本与人工 Reference 之间的编辑距离。

即使找到了概率最大的 Sequence，也不代表 CER 最低。

因此，Beam Search 可能改善 CER，也可能使 CER 不变甚至变差。

## 7. 本模块总结

Greedy Decoding 的完整流程：

```text
log_probs
    ↓
逐帧 argmax
    ↓
Best Path
    ↓
合并相邻重复 Token
    ↓
删除 Blank
    ↓
最终 Sequence
```

核心认识：

1. Greedy 通过逐帧 argmax 得到概率最大的单条 Path。
2. CTC Collapse 必须先合并相邻重复 Token，再删除 Blank。
3. previous_token 遇到 Blank 也必须更新。
4. Best Path Collapse 不一定等于 Best Collapsed Sequence。
5. Beam Search 是近似搜索，不保证 CER 一定下降。