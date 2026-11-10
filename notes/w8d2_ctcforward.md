# W8D2：CTC 前向递推

## 1. 扩展目标序列（Extended Target）

### 1.1 为什么需要扩展目标？

CTC 模型逐时间帧输出 Token 概率分布，而最终目标文本的长度通常与时间帧数量不同。

为了计算目标文本对应的所有合法对齐路径，CTC 在原始目标序列的开头、结尾以及每两个目标 Token 之间插入 Blank，构造扩展目标序列。

这样可以将目标文本的对齐问题转换为在扩展目标状态之间进行动态规划的问题。

### 1.2 如何构造扩展目标？

假设原始目标序列为：

$$
y=(y_1,y_2,\ldots,y_U)
$$

在开头、结尾以及每两个 Token 之间插入 Blank：

$$
y'=(\varnothing,y_1,\varnothing,y_2,\ldots,\varnothing,y_U,\varnothing)
$$

其中，$\varnothing$ 表示 Blank。

例如：

```text
原始目标：

a a b

扩展目标：

_ a _ a _ b _
```

即使原始目标中存在连续重复 Token，也需要在它们之间插入 Blank。

### 1.3 扩展目标的长度

原始目标包含 U 个 Token，一共插入 U+1 个 Blank。

因此：

$$
\boxed{S=2U+1}
$$

例如：

| 原始目标 | 扩展目标 | S |
|---|---|---:|
| a | _ a _ | 3 |
| ab | _ a _ b _ | 5 |
| aa | _ a _ a _ | 5 |
| aab | _ a _ a _ b _ | 7 |

注意：S 是扩展目标的状态数量，不是模型输出的时间帧数量。

---

## 2. 前向变量（Forward Variable）

### 2.1 alpha[t][s] 的含义

定义：

$$
\boxed{
\alpha_t(s)
=
\text{截至时间 }t\text{，所有合法且到达扩展目标状态 }s
\text{ 的路径概率之和}
}
$$

其中：

- $\alpha_t(s)$：前向变量。
- t：时间帧索引。
- s：扩展目标状态索引。
- $p_t(k)$：模型在时间帧 t 输出 Token k 的概率。

需要注意，alpha 不是某一条路径的概率，而是所有符合条件的路径概率之和。

CTC 的目标概率需要对所有能够折叠为目标文本的合法路径求和：

$$
P(y|x)=
\sum_{\pi:B(\pi)=y}P(\pi|x)
$$

其中：

- x：输入音频。
- y：目标文本。
- $\pi$：一条 CTC Path。
- B：CTC 路径折叠函数。

前向算法通过动态规划计算这些路径的概率总和，避免显式枚举全部路径。

### 2.2 DP 矩阵的两个维度

前向变量构成一个二维 DP 矩阵：

$$
\boxed{\alpha\in\mathbb{R}^{T\times S}}
$$

其中：

- T：模型输出的时间帧数量。
- S：扩展目标序列的长度。

例如，原始目标为 ab：

```text
扩展目标：

_ a _ b _
```

状态编号（从0开始）：

```text
0 1 2 3 4
```

DP 矩阵的形状为：

$$
(T,5)
$$

两个维度不是一一对应关系：

- 每一行对应一个时间帧。
- 每一列对应扩展目标的一个状态。
- 不同时间帧可以停留在同一个状态。
- 某些状态可以在满足条件时被跳过。

因此，S 大于 T 不一定意味着无法完成识别。

**T、U、S 的区别：**

| 符号 | 含义 | 决定因素 |
|---|---|---|
| T | 模型输出的时间帧数量 | 音频及声学模型 |
| U | 原始目标 Token 数量 | Reference |
| S | 扩展目标状态数量 | 扩展目标构造规则 |

三者的关系：

$$
S=2U+1
$$

$$
\alpha\in\mathbb{R}^{T\times S}
$$

**为什么 S 大于 T 不一定无法完成识别？**

因为 CTC 不要求每个扩展目标状态都必须经过。

状态转移允许：

1. 留在当前状态。
2. 向后移动一个状态。
3. 满足条件时向后移动两个状态。

因此，部分状态可以被跳过，但并不是任意状态都可以跳过。

**最少时间帧数：**

假设目标长度为 U，其中相邻重复 Token 对的数量为 R，那么至少需要：

$$
\boxed{T_{\min}=U+R}
$$

原因是每个目标 Token 至少需要一个时间帧，而每一对相邻重复 Token 之间还必须至少有一个 Blank 时间帧。

例如：

| 目标 | U | R | 最少时间帧 |
|---|---:|---:|---:|
| ab | 2 | 0 | 2 |
| aa | 2 | 1 | 3 |
| aab | 3 | 1 | 4 |
| aabb | 4 | 2 | 6 |

### 2.3 初始状态

对于非空目标，假设至少有一个时间帧，处理第一个时间帧时只有两个合法初始状态：

- 状态0：Blank。
- 状态1：原始目标的第一个 Token。

原因是第一个时间帧只能输出 Blank 或目标序列的第一个 Token，不能直接跳到后续目标 Token，否则会跳过目标前缀。

例如：

```text
目标：

ab

扩展目标：

_ a _ b _

状态：

0 1 2 3 4
```

第一个时间帧有两种合法选择：

- 输出 Blank，对应状态0。
- 输出 a，对应状态1。

在普通概率空间中：

$$
\boxed{\alpha_0(0)=p_0(\varnothing)}
$$

$$
\boxed{\alpha_0(1)=p_0(y_1)}
$$

其余状态初始化为：

$$
\boxed{\alpha_0(s)=0,\quad s\geq2}
$$

由于后续采用 Log 空间实现，实际代码中使用：

$$
\boxed{\alpha_0(0)=\log p_0(\varnothing)}
$$

$$
\boxed{\alpha_0(1)=\log p_0(y_1)}
$$

$$
\boxed{\alpha_0(s)=-\infty,\quad s\geq2}
$$

这里的负无穷表示对应状态在第一个时间帧不可能到达。

注意：这里的时间索引从0开始，因此第一个时间帧对应 t=0。在 Python 中对应：

```python
alpha[0][0]
alpha[0][1]
```

另外，以上初始化针对非空目标。空目标需要单独处理，后续模块再讨论。


## 3. 本模块总结

CTC 前向算法通过扩展目标序列，将目标文本的对齐问题转化为动态规划问题。

**扩展目标：**

通过在开头、结尾和相邻目标 Token 之间插入 Blank 构造：

$$
S=2U+1
$$

**前向变量：**

表示截至时间 t，所有合法且到达扩展目标状态 s 的路径概率之和：

$$
\alpha_t(s)
$$

**DP 矩阵：**

$$
\alpha\in\mathbb{R}^{T\times S}
$$

其中 T 表示时间帧数量，S 表示扩展目标状态数量，两者不要求相等。

**初始状态：**

对于非空目标，第一个时间帧只有两个合法初始状态：

1. Blank 状态。
2. 第一个目标 Token 对应的状态。

Log 空间初始化：

$$
\alpha_0(0)=\log p_0(\varnothing)
$$

$$
\alpha_0(1)=\log p_0(y_1)
$$

$$
\alpha_0(s)=-\infty,\quad s\geq2
$$

**时间帧约束：**

如果目标中存在相邻重复 Token，则它们之间必须通过 Blank 分隔。

目标长度为 U，相邻重复 Token 对数量为 R 时：

$$
T_{\min}=U+R
$$

本模块重点是理解扩展目标、前向变量和初始状态，为后续推导 CTC 前向递推公式做准备。


# W8D2：CTC 前向递推——状态转移

## 3. 状态转移

### 3.1 三种转移方式

假设扩展目标为：

```text
状态编号：0  1  2  3  4
扩展目标：_  a  _  b  _
```

在时间 t 到达状态 s 时，可能来自上一时间帧的三个前驱状态。

**1. Stay（留在当前状态）**

状态不发生变化：

$$
s\rightarrow s
$$

对应前向变量：

$$
\alpha_{t-1}(s)\rightarrow\alpha_t(s)
$$

含义：模型可以连续多个时间帧输出同一个 Token。

例如：

```text
b b b
```

经过 CTC 路径折叠后仍然得到：

```text
b
```

**2. Advance（前进一个状态）**

向后移动一个状态：

$$
s-1\rightarrow s
$$

对应前向变量：

$$
\alpha_{t-1}(s-1)\rightarrow\alpha_t(s)
$$

例如：

```text
_ → b
```

Advance 不需要额外条件。

**3. Skip（前进两个状态）**

直接向后跳过一个状态：

$$
s-2\rightarrow s
$$

对应前向变量：

$$
\alpha_{t-1}(s-2)\rightarrow\alpha_t(s)
$$

例如：

```text
a → b
```

这种转移可以跳过两个目标 Token 之间的 Blank，但必须满足特定条件。

### 3.2 前向递推公式

假设当前时间帧 t 到达扩展目标状态 s。

三个前驱状态分别为：

- 状态 s：Stay。
- 状态 s-1：Advance。
- 状态 s-2：Skip。

首先将所有合法前驱状态的概率相加，再乘以当前时间帧输出对应 Token 的概率。

其中：

$$
p_t(y'_s)
$$

表示模型在时间帧 t 输出当前扩展目标 Token 的概率。

普通概率空间下的完整递推公式：

$$
\boxed{
\alpha_t(s)=p_t(y'_s)
\left[
\alpha_{t-1}(s)
+\alpha_{t-1}(s-1)
+I_{\text{skip}}(s)\alpha_{t-1}(s-2)
\right]
}
$$

其中，$I_{\text{skip}}(s)$ 是 Skip 转移的指示函数。

如果 Skip 合法，则取1；否则取0。

三个前驱状态分别代表：

| 前驱状态 | 转移方式 | 含义 |
|---|---|---|
| s | Stay | 停留在当前状态 |
| s-1 | Advance | 前进一个状态 |
| s-2 | Skip | 前进两个状态 |

如果某个前驱状态越界，则其概率贡献为0。

### 3.3 Skip 两格的条件

Skip 必须同时满足以下条件：

**条件一：当前 Token 不能是 Blank。**

$$
y'_s\neq\varnothing
$$

原因：如果当前 Token 是 Blank，就不允许直接跳过前面的目标字符进入 Blank 状态。

**条件二：当前 Token 必须与两格前的 Token 不同。**

$$
y'_s\neq y'_{s-2}
$$

原因：如果两个 Token 相同，直接跳过中间 Blank 会导致两个相同 Token 在原始路径中相邻，经过路径折叠后被合并，无法正确表示重复字符。

此外，必须保证：

$$
s\geq2
$$

否则 $s-2$ 会发生索引越界。

因此，完整 Skip 指示函数为：

$$
\boxed{
I_{\text{skip}}(s)=
\begin{cases}
1,&s\geq2,\ y'_s\neq\varnothing,\ y'_s\neq y'_{s-2}\\
0,&\text{其他情况}
\end{cases}
}
$$

### 3.4 为什么重复目标字符不能直接 Skip？

考虑目标：

```text
aa
```

扩展目标：

```text
状态编号：0  1  2  3  4
扩展目标：_  a  _  a  _
```

如果允许从状态1直接跳到状态3：

```text
a → a
```

模型输出的原始路径为：

```text
a a
```

根据 CTC 路径折叠规则：

```text
a a → a
```

最终只得到一个 a，而不是目标要求的 aa。

因此，两个相同目标 Token 之间必须经过 Blank：

```text
a → _ → a
```

经过路径折叠：

```text
a _ a → aa
```

所以，Skip 必须同时满足当前 Token 非 Blank，以及当前 Token 与两格前 Token 不同这两个条件。

---

## 4. 小规模手算

### Example 1：目标 ab

原始目标：

```text
a b
```

扩展目标：

```text
_ a _ b _
```

状态编号：

```text
0 1 2 3 4
```

当前考虑到达状态3（b）。

三个可能的前驱状态：

| 前驱状态 | Token | 转移方式 |
|---|---|---|
| 状态1 | a | Skip |
| 状态2 | Blank | Advance |
| 状态3 | b | Stay |

Skip 合法，因为：

$$
y'_3=b\neq\varnothing
$$

$$
y'_3=b\neq y'_1=a
$$

因此：

$$
\alpha_t(3)=p_t(b)
\left[
\alpha_{t-1}(3)
+\alpha_{t-1}(2)
+\alpha_{t-1}(1)
\right]
$$

### Example 2：目标 aa

原始目标：

```text
a a
```

扩展目标：

```text
_ a _ a _
```

状态编号：

```text
0 1 2 3 4
```

当前考虑到达状态3（第二个 a）。

三个可能的前驱状态：

| 前驱状态 | Token | 转移方式 | 是否允许 |
|---|---|---|---|
| 状态1 | a | Skip | 不允许 |
| 状态2 | Blank | Advance | 允许 |
| 状态3 | a | Stay | 允许 |

由于：

$$
y'_3=a=y'_1
$$

不满足 Skip 条件。

因此：

$$
\alpha_t(3)=p_t(a)
\left[
\alpha_{t-1}(3)
+\alpha_{t-1}(2)
\right]
$$

不能加入状态1的概率贡献。

---

## 5. 本模块总结

CTC 前向递推中，每个状态在当前时间帧可能来自三个前驱状态：

1. Stay：留在当前状态。
2. Advance：向后移动一个状态。
3. Skip：向后移动两个状态。

普通概率空间下的完整递推公式：

$$
\boxed{
\alpha_t(s)=p_t(y'_s)
\left[
\alpha_{t-1}(s)
+\alpha_{t-1}(s-1)
+I_{\text{skip}}(s)\alpha_{t-1}(s-2)
\right]
}
$$

Skip 必须满足：

$$
\boxed{
s\geq2,\quad
y'_s\neq\varnothing,\quad
y'_s\neq y'_{s-2}
}
$$

其核心目的是保证路径折叠后仍然能够得到正确的目标文本。

特别是对于重复目标字符，不能直接跳过中间 Blank，否则两个相同 Token 会在路径折叠时被合并。

本模块重点是理解三个前驱状态及 Skip 的限制，为后续 Log 空间计算和代码实现做准备。


# W8D2：CTC 前向递推——Log 空间与边界处理

## 4. Log 空间计算

### 4.1 为什么需要 Log 空间？

CTC 需要计算一条路径中所有时间帧输出概率的乘积。

假设一条路径包含 T 个时间帧，其概率为：

$$
P(\pi|x)=\prod_{t=0}^{T-1}p_t(\pi_t)
$$

随着时间帧数量增加，多个较小的概率不断相乘，最终可能变得非常小。

例如：

$$
0.01^{100}=10^{-200}
$$

对于 float32 等浮点数据类型，这种极小的数值可能无法正常表示，导致数值下溢，结果变成0。

因此，CTC 前向算法通常在 Log 空间中进行计算。

定义 Log 前向变量：

$$
\boxed{A_t(s)=\log\alpha_t(s)}
$$

利用对数的基本性质：

$$
\log(ab)=\log a+\log b
$$

原本的概率连乘就转换成 Log 概率的加法，从而避免直接计算大量极小概率的乘积。

Log 转换的作用：

- 将概率连乘转换为加法。
- 避免连续概率相乘导致的数值下溢。
- 便于后续使用 LogSumExp 进行概率求和。

### 4.2 LogSumExp

**定义：**

假设有多个 Log 概率：

$$
a_i=\log P_i
$$

那么：

$$
\boxed{
\operatorname{LogSumExp}(a_1,\ldots,a_n)
=
\log\left(\sum_{i=1}^{n}e^{a_i}\right)
}
$$

它的作用是将 Log 空间中的多个概率转换为普通概率，求和后再取对数。

**为什么不能直接对 Log 概率求和？**

例如，两条路径的概率分别为：

$$
P_1=0.2,\qquad P_2=0.3
$$

我们需要计算两条路径的概率之和：

$$
P=P_1+P_2=0.5
$$

在 Log 空间中：

$$
\log P=\log(0.2+0.3)=\log0.5
$$

如果直接相加两个 Log 概率：

$$
\log0.2+\log0.3
$$

根据对数运算法则：

$$
=\log(0.2\times0.3)=\log0.06
$$

这实际上对应概率相乘，而不是概率相加。

因此，Log 空间中的概率求和必须使用 LogSumExp。

**数值稳定版本：**

直接计算指数可能出现数值上溢。

令：

$$
m=\max(a_1,a_2,\ldots,a_n)
$$

则可以将最大值提取出来：

$$
\boxed{
\operatorname{LogSumExp}(a_1,\ldots,a_n)
=
m+\log\left(
\sum_{i=1}^{n}e^{a_i-m}
\right)
}
$$

因为每个指数中的数值都小于等于0，所以可以避免指数运算过大导致的上溢。

**所有输入均为负无穷时如何处理？**

根据 LogSumExp 的定义：

$$
\operatorname{LogSumExp}(-\infty,\ldots,-\infty)
=
\log(0+\cdots+0)
$$

因此：

$$
\boxed{
\operatorname{LogSumExp}(-\infty,\ldots,-\infty)
=-\infty
}
$$

这里需要区分数学结果与数值实现：

如果直接使用最大值平移公式，此时最大值为负无穷，可能出现：

$$
-\infty-(-\infty)=\mathrm{NaN}
$$

这是不恰当的数值实现导致的问题，不是 LogSumExp 的正确数学结果。

实际实现推荐使用：

```python
torch.logsumexp(values, dim=0)
```

它能够正确处理所有输入均为负无穷的情况。

### 4.3 Log 空间前向递推

定义：

$$
A_t(s)=\log\alpha_t(s)
$$

普通概率空间下的递推公式：

$$
\alpha_t(s)=p_t(y'_s)
\left[
\alpha_{t-1}(s)
+\alpha_{t-1}(s-1)
+I_{\text{skip}}(s)\alpha_{t-1}(s-2)
\right]
$$

对公式取对数，利用乘法变加法、加法变 LogSumExp，得到：

$$
\boxed{
A_t(s)=\log p_t(y'_s)
+
\operatorname{LogSumExp}
\left(
A_{t-1}(s),
A_{t-1}(s-1),
A_{t-1}(s-2)
\right)
}
$$

其中第三项只有在满足 Skip 条件时才能参与计算；否则将其视为负无穷，不产生概率贡献。

Log 空间下的三种转移：

| 转移方式 | 前驱状态 | 处理方式 |
|---|---|---|
| Stay | s | 参与 LogSumExp |
| Advance | s-1 | 合法时参与 LogSumExp |
| Skip | s-2 | 满足 Skip 条件时参与 LogSumExp |

最后再加上当前时间帧的 Log 发射概率：

$$
\log p_t(y'_s)
$$

注意：Log 空间中不能直接将多个前驱状态的 Log 概率相加，必须使用 LogSumExp。

---

## 5. 边界情况

### 5.1 初始状态

对于非空目标，扩展目标：

```text
_ a _ b _
```

状态编号：

```text
0 1 2 3 4
```

第一个时间帧只有两个合法初始状态：

- 状态0：Blank。
- 状态1：第一个目标 Token。

Log 空间初始化：

$$
\boxed{A_0(0)=\log p_0(\varnothing)}
$$

$$
\boxed{A_0(1)=\log p_0(y_1)}
$$

其余状态：

$$
\boxed{A_0(s)=-\infty,\quad s\geq2}
$$

原因：第一个时间帧只能输出 Blank 或第一个目标 Token，不能直接跳过目标前缀。

注意：上述初始化适用于非空目标，且要求时间帧数量至少为1。

### 5.2 递推过程中的越界状态

当前状态 s 的三个潜在前驱：

- s：Stay。
- s-1：Advance。
- s-2：Skip。

需要根据当前状态判断哪些前驱合法。

**情况1：s=0**

当前状态为第一个 Blank。

只有 Stay 合法：

$$
A_t(0)=\log p_t(\varnothing)+A_{t-1}(0)
$$

不能访问状态 -1 和状态 -2。

**情况2：s=1**

当前状态为第一个目标 Token。

合法前驱为：

- 状态0：Advance。
- 状态1：Stay。

因此：

$$
\boxed{
A_t(1)=\log p_t(y'_1)
+
\operatorname{LogSumExp}
\left(
A_{t-1}(0),A_{t-1}(1)
\right)
}
$$

不能访问状态 -1。

**情况3：s>=2**

此时可以检查三个前驱状态：

- s：Stay。
- s-1：Advance。
- s-2：满足 Skip 条件时才允许。

完整递推仍然为：

$$
A_t(s)=\log p_t(y'_s)
+
\operatorname{LogSumExp}
\left(
A_{t-1}(s),
A_{t-1}(s-1),
A_{t-1}(s-2)
\right)
$$

其中不合法的前驱状态视为负无穷，不参与概率计算。

**越界状态的处理原则：**

普通概率空间中，不可能到达的状态概率为0。

Log 空间中：

$$
\boxed{\log0=-\infty}
$$

因此，越界状态在数学上可以视为负无穷。

但在 Python 实现中，必须先检查索引是否合法，再访问数组。

例如：

```python
if s >= 1:
    # Advance

if s >= 2:
    # 才能检查 Skip
```

不能直接访问负索引，因为 Python 会将负索引解释为从数组末尾倒数的位置。

### 5.3 终止状态

对于非空目标，扩展目标长度为：

$$
S=2U+1
$$

扩展目标的最后两个状态分别为：

- S-2：最后一个目标 Token。
- S-1：末尾 Blank。

例如，目标为 ab：

```text
状态编号：0 1 2 3 4
扩展目标：_ a _ b _
```

最后两个状态：

| 状态 | Token | 含义 |
|---|---|---|
| S-2（状态3） | b | 最后一个时间帧输出最后一个目标 Token |
| S-1（状态4） | Blank | 最后一个时间帧输出 Blank |

这两种情况都能通过 CTC Collapse 得到完整目标文本。

例如：

```text
a b b → ab
```

```text
a b _ → ab
```

因此，必须将两个终止状态的概率相加。

普通概率空间：

$$
P(y|x)=
\alpha_{T-1}(S-2)+\alpha_{T-1}(S-1)
$$

Log 空间：

$$
\boxed{
\log P(y|x)=
\operatorname{LogSumExp}
\left(
A_{T-1}(S-2),
A_{T-1}(S-1)
\right)
}
$$

其中 T 为时间帧数量，最后一个时间帧的索引为 T-1。

如果只取 S-1，会遗漏最后一个时间帧直接输出最后一个目标 Token 的合法路径。

如果只取 S-2，则会遗漏最后一个时间帧输出末尾 Blank 的合法路径。

因此，两个终止状态都必须参与求和，而不是只取其中一个，也不是取最大值。

### 5.4 空目标处理

**空目标的扩展目标：**

原始目标：

```text
[]
```

扩展目标：

```text
[_]
```

只有一个 Blank 状态：

$$
S=1
$$

**合法路径：**

如果目标为空，模型输出经过 CTC Collapse 后也必须为空。

因此，所有时间帧只能输出 Blank：

```text
_ _ _ _ _
```

不能出现任何普通 Token。

**普通概率空间下的空目标概率：**

$$
P(\varnothing|x)
=
\prod_{t=0}^{T-1}p_t(\varnothing)
$$

**Log 空间下的空目标概率：**

$$
\boxed{
\log P(\varnothing|x)
=
\sum_{t=0}^{T-1}\log p_t(\varnothing)
}
$$

因为对概率连乘取对数后，转换为所有时间帧 Blank Log 概率之和。

**为什么需要单独处理？**

非空目标的终止公式需要访问：

$$
S-2,\quad S-1
$$

但空目标只有一个状态：

$$
S=1
$$

此时：

$$
S-2=-1
$$

会产生非法索引。

同时，空目标不存在最后一个普通目标 Token 对应的状态，只有 Blank 状态。

因此，不能直接使用非空目标的终止状态公式，需要单独处理。

代码实现：

```python
if target_length == 0:
    return log_probs[:, blank].sum()
```

这里假设 T 大于0。

如果 T=0：

- 空目标：存在一条空路径，概率为1，Log 概率为0。
- 非空目标：不存在合法对齐路径，Log 概率为负无穷。

---

## 6. 本模块总结

### 6.1 Log 空间计算

CTC 路径概率需要不断连乘，容易出现数值下溢。

因此使用 Log 空间：

$$
A_t(s)=\log\alpha_t(s)
$$

将概率乘法转换为 Log 概率加法。

### 6.2 LogSumExp

Log 空间中的概率求和不能直接对 Log 概率相加，而要使用：

$$
\operatorname{LogSumExp}(a_1,\ldots,a_n)
=
\log\left(\sum_i e^{a_i}\right)
$$

数值稳定版本：

$$
\operatorname{LogSumExp}(a_1,\ldots,a_n)
=
m+\log\sum_i e^{a_i-m}
$$

所有输入为负无穷时，正确结果为负无穷。

### 6.3 Log 空间递推

前向递推先通过 LogSumExp 汇总所有合法前驱状态，再加上当前时间帧的 Log 发射概率：

$$
A_t(s)=\log p_t(y'_s)
+
\operatorname{LogSumExp}(\text{合法前驱状态})
$$

### 6.4 边界处理

- 初始状态：只有状态0和状态1有效。
- s=0：只有 Stay。
- s=1：Stay 和 Advance。
- s>=2：检查 Stay、Advance 和 Skip。
- 不合法或越界的状态在 Log 空间中视为负无穷。
- Python 实现时必须先检查索引合法性。

### 6.5 终止状态

对于非空目标，最后两个状态分别为最后一个目标 Token 和末尾 Blank。

两种结束方式都能得到完整目标文本，因此：

$$
\boxed{
\log P(y|x)=
\operatorname{LogSumExp}
\left(
A_{T-1}(S-2),
A_{T-1}(S-1)
\right)
}
$$

### 6.6 空目标

空目标的扩展目标只有一个 Blank 状态。

整个时间序列必须全部输出 Blank：

$$
\boxed{
\log P(\varnothing|x)
=
\sum_{t=0}^{T-1}\log p_t(\varnothing)
}
$$

由于只有一个状态，不能直接使用非空目标的终止状态公式，需要单独处理。

---

## 7. 本模块核心结论

W8D2 的三个理论模块已经建立了完整的 CTC 前向算法框架：

1. 扩展目标：构造 DP 状态序列。
2. 状态转移：通过 Stay、Advance、Skip 计算前向概率。
3. Log 空间及边界处理：保证数值稳定，并正确处理初始状态、终止状态和空目标。

下一步将把这些数学定义转化为 Python 代码，实现 `src/ctc.py`，并通过小规模枚举、PyTorch CTCLoss 和重复目标测试验证正确性。


# W8D2：CTC 代码实现——数据结构与扩展目标

## 1. 输入数据约定

### 1.1 模型输出

模型经过神经网络后，会输出每个时间帧对应的类别分数，即 Logits。

假设词表大小为 C，时间帧数量为 T，则：

$$
\text{logits}\in\mathbb{R}^{T\times C}
$$

其中：

- T：输入时间帧数量。
- C：词表类别数量。
- logits[t,k]：时间帧 t 对类别 k 输出的原始分数。

Logits 不是概率，不能直接用于 CTC 路径概率计算。

因此，需要先进行 Log Softmax：

```python
log_probs = torch.log_softmax(
    logits,
    dim=-1
)
```

得到：

$$
\text{log\_probs}[t,k]=\log p_t(k)
$$

其中 p_t(k) 表示第 t 个时间帧输出类别 k 的概率。

log_probs 的形状：

$$
\boxed{\text{log\_probs.shape}=(T,C)}
$$

后续的 `ctc_forward_log()` 直接接收 log_probs，不再重复计算 Log Softmax。

### 1.2 目标序列

目标序列 target 是人工标注文本经过 Tokenizer 编码后得到的 Token ID 序列。

例如：

```python
target = [1, 2]
```

表示：

```text
a b
```

Blank 是 CTC 专门引入的特殊类别，用于路径对齐。

假设：

```python
blank = 0
```

那么原始 target 必须满足：

1. 只包含普通目标 Token。
2. 不包含 Blank。
3. Blank 由扩展目标构造函数自动插入。

如果 target 中包含 Blank，应当抛出 ValueError，而不是直接删除。

原因：直接删除会擅自修改原始目标，掩盖数据错误。

### 1.3 变量与矩阵形状

统一使用以下变量：

| 符号 | 含义 |
|---|---|
| T | 输入时间帧数量 |
| U | 原始目标 Token 数量 |
| S | 扩展目标长度 |
| C | 词表类别数量 |

扩展目标长度：

$$
\boxed{S=2U+1}
$$

三个主要数据结构：

| 变量 | 形状 | 含义 |
|---|---|---|
| target | (U,) | 原始目标 Token 序列 |
| extended_target | (S,) | 扩展目标序列 |
| log_probs | (T,C) | 每个时间帧的 Log 概率 |
| alpha | (T,S) | 前向 DP 矩阵 |

注意区分：

- C 是词表类别数量。
- S 是扩展目标状态数量。

log_probs 的列对应词表类别，而 alpha 的列对应扩展目标状态，因此二者形状不同。

---

## 2. 扩展目标构造函数

### 2.1 构造规则

CTC 扩展目标需要在原始目标序列的开头、结尾以及每两个目标 Token 之间插入 Blank。

原始目标：

```text
a b
```

扩展目标：

```text
_ a _ b _
```

如果原始目标长度为 U，则扩展目标长度为：

$$
\boxed{S=2U+1}
$$

构造规则：

1. 初始化长度为 S 的序列，所有位置填充 Blank。
2. 将原始目标 Token 放入索引为 1、3、5 等奇数索引的位置。
3. 保留其他位置的 Blank。

### 2.2 Python 实现

函数：

```python
def extend_target(target, blank=0):
    """
    将原始目标序列扩展为 CTC 目标序列。

    参数：
        target: 原始目标 Token ID 序列，不包含 blank
        blank: Blank 对应的 Token ID

    返回：
        extended_target: 扩展目标序列
    """

    target = [int(token) for token in target]

    # 检查 target 是否包含 Blank
    if blank in target:
        raise ValueError(
            "target must not contain blank"
        )

    # 初始化扩展目标
    extended_target = [blank] * (
        2 * len(target) + 1
    )

    # 将原始 Token 放入奇数索引位置
    extended_target[1::2] = target

    return extended_target
```

**代码解释：**

首先：

```python
extended_target = [blank] * (
    2 * len(target) + 1
)
```

创建一个长度为 S 的序列，并将所有位置初始化为 Blank。

然后：

```python
extended_target[1::2] = target
```

使用 Python 切片赋值，将原始目标 Token 放入奇数索引位置。

其中：

- 1：起始索引。
- 2：步长。

因此选择的位置为：

```text
1, 3, 5, ..., 2U-1
```

这些位置一共有 U 个，恰好与原始目标 Token 数量相等。

### 2.3 测试

**测试1：普通目标**

```python
target = [1, 2, 3]

result = extend_target(target)

assert result == [
    0, 1, 0, 2, 0, 3, 0
]
```

**测试2：重复 Token**

```python
target = [1, 1]

result = extend_target(target)

assert result == [
    0, 1, 0, 1, 0
]
```

即使原始目标包含连续重复 Token，也必须在它们之间插入 Blank。

**测试3：空目标**

```python
target = []

result = extend_target(target)

assert result == [0]
```

空目标的扩展目标只有一个 Blank。

**测试4：自定义 Blank**

```python
target = [2, 3]

result = extend_target(
    target,
    blank=4
)

assert result == [4, 2, 4, 3, 4]
```

**测试5：长度检查**

```python
target = [1, 2, 3, 4]

extended = extend_target(target)

assert len(extended) == (
    2 * len(target) + 1
)
```

**测试6：非法 Blank 检查**

```python
try:
    extend_target([1, 0, 2])
except ValueError:
    pass
else:
    raise AssertionError(
        "Blank should not appear in target"
    )
```

所有测试通过后，说明扩展目标构造函数符合预期。

---

## 3. 前向 DP 矩阵的数据结构

### 3.1 矩阵形状

前向 DP 矩阵 alpha 的形状为：

$$
\boxed{\alpha\text{ shape}=(T,S)}
$$

其中：

- 行表示时间帧。
- 列表示扩展目标状态。

例如：

```python
T = 5
target = [1, 2]

extended_target = [0, 1, 0, 2, 0]

S = len(extended_target)

alpha.shape == (5, 5)
```

注意：alpha 的列数是扩展目标状态数量 S，而不是词表类别数量 C。

### 3.2 初始化值

Log 空间中：

$$
\log0=-\infty
$$

尚未到达的状态，其前向概率为0，因此 Log 前向概率为负无穷。

所以，DP 矩阵首先全部初始化为负无穷：

```python
alpha = torch.full(
    (T, S),
    float("-inf"),
    dtype=log_probs.dtype,
    device=log_probs.device
)
```

这样可以保证：

- alpha 与 log_probs 使用相同的数据类型。
- alpha 与 log_probs 位于相同设备。
- 尚未到达的状态不会产生错误的概率贡献。

### 3.3 初始状态赋值

对于非空目标，第一个时间帧只有两个有效状态：

- 状态0：Blank。
- 状态1：第一个目标 Token。

因此：

$$
A_0(0)=\log p_0(\varnothing)
$$

$$
A_0(1)=\log p_0(y_1)
$$

其余状态保持负无穷：

$$
A_0(s)=-\infty,\quad s\geq2
$$

代码：

```python
alpha[0, 0] = log_probs[0, blank]

alpha[0, 1] = log_probs[
    0, extended_target[1]
]
```

注意：上述初始化适用于非空目标且 T 大于0的情况。空目标需要单独处理。

---

## 4. 本模块总结

### 4.1 输入数据约定

- logits：模型输出的原始类别分数。
- log_probs：经过 Log Softmax 后的 Log 概率。
- target：原始目标 Token 序列，不包含 Blank。
- blank：CTC 特殊类别的 Token ID。

### 4.2 扩展目标构造

扩展目标在开头、结尾以及相邻目标 Token 之间插入 Blank。

$$
\boxed{S=2U+1}
$$

使用 `extend_target()` 实现。

非法 target 包含 Blank 时，应当报错，而不是自动删除。

### 4.3 DP 矩阵

前向 DP 矩阵：

$$
\boxed{\alpha\text{ shape}=(T,S)}
$$

初始化为负无穷：

$$
\boxed{\alpha_{t,s}=-\infty}
$$

随后对合法初始状态进行赋值。

本模块完成了 CTC 前向算法的数据结构约定、扩展目标构造和 DP 矩阵初始化准备，为下一步实现 `ctc_forward_log()` 做好准备。


# W8D2：CTC 前向递推代码实现——ctc_forward_log()

## 5. ctc_forward_log() 实现

### 5.1 函数接口

CTC 前向算法的作用是：输入模型每个时间帧的 Log 概率和目标 Token 序列，通过动态规划计算目标序列在 CTC 模型下的总 Log 概率。

数学目标：

$$
\log P(y|x)
$$

这里不是计算某一条路径的概率，而是计算所有经过 CTC Collapse 后能够得到目标序列的合法路径概率之和。

函数接口：

```python
def ctc_forward_log(
    log_probs,
    target,
    blank=0
):
    ...
```

**输入参数：**

| 参数 | 含义 |
|---|---|
| log_probs | 每个时间帧的 Log 概率，形状为 (T,C) |
| target | 原始目标 Token 序列，不包含 Blank |
| blank | Blank 对应的 Token ID |

其中，log_probs 必须经过 Log Softmax：

```python
log_probs = torch.log_softmax(
    logits,
    dim=-1
)
```

不能直接传入 logits，因为 logits 是模型输出的原始分数，不是 Log 概率。

**返回值：**

返回一个标量 Tensor：

```python
log_likelihood
```

其数学含义为：

$$
\boxed{\log P(y|x)}
$$

### 5.2 输入合法性检查

正式进行 DP 计算之前，需要检查输入是否合法。

主要包括：

1. log_probs 必须是二维 Tensor。
2. log_probs 必须使用浮点数据类型。
3. Blank ID 必须位于词表范围内。
4. target 中的 Token ID 必须位于词表范围内。
5. target 不能包含 Blank。

完整实现：

```python
def validate_ctc_inputs(
    log_probs,
    target,
    blank=0
):
    """
    检查 CTC 前向算法输入是否合法。
    """

    if not isinstance(log_probs, torch.Tensor):
        raise TypeError(
            "log_probs must be a torch.Tensor"
        )

    if log_probs.ndim != 2:
        raise ValueError(
            "log_probs must have shape (T, C)"
        )

    if not log_probs.is_floating_point():
        raise TypeError(
            "log_probs must use floating dtype"
        )

    T, C = log_probs.shape

    if C == 0:
        raise ValueError(
            "vocabulary size must be positive"
        )

    blank = int(blank)

    if not 0 <= blank < C:
        raise ValueError(
            "blank ID is out of vocabulary range"
        )

    target = [int(token) for token in target]

    for token in target:

        if not 0 <= token < C:
            raise ValueError(
                f"Invalid target token ID: {token}"
            )

        if token == blank:
            raise ValueError(
                "target must not contain blank"
            )

    return target, blank
```

检查 target 不包含 Blank，是因为 Blank 属于 CTC 的路径对齐机制，不属于原始目标序列。

如果 target 中包含 Blank，应当报错，而不是直接删除，以免掩盖原始数据错误。

---

### 5.3 DP 矩阵初始化

首先构造扩展目标：

```python
extended_target = extend_target(
    target,
    blank
)
```

假设：

```python
target = [1, 2]
blank = 0
```

得到：

```text
extended_target = [0, 1, 0, 2, 0]
```

扩展目标长度：

$$
S=2U+1
$$

DP 矩阵形状：

$$
\alpha\text{ shape}=(T,S)
$$

**为什么初始化为负无穷？**

Log 空间中：

$$
\log 0=-\infty
$$

尚未到达的状态，其前向概率为0，因此 Log 前向概率为负无穷。

```python
neg_inf = log_probs.new_tensor(
    float("-inf")
)
```

**初始状态：**

第一个时间帧只有两个合法状态：

- 状态0：Blank。
- 状态1：第一个目标 Token。

对应公式：

$$
A_0(0)=\log p_0(\varnothing)
$$

$$
A_0(1)=\log p_0(y_1)
$$

其余状态：

$$
A_0(s)=-\infty,\quad s\geq2
$$

代码：

```python
first_row = [neg_inf] * S

first_row[0] = log_probs[0, blank]

first_row[1] = log_probs[
    0, extended_target[1]
]

rows = [torch.stack(first_row)]
```

这里采用逐行构造 DP 矩阵的方式：

- `rows` 保存已经计算完成的时间帧。
- `prev` 表示上一时间帧。
- `current` 保存当前时间帧。

最后使用 `torch.stack()` 将所有行组合成完整 DP 矩阵。

这种写法能够保留 PyTorch Autograd 的计算图，方便后续进行梯度验证。

---

### 5.4 前向递推

前向递推是整个 CTC 算法的核心。

Log 空间递推公式：

$$
A_t(s)=
\log p_t(y'_s)
+
\operatorname{LogSumExp}(\text{合法前驱状态})
$$

每个状态有三个潜在前驱：

| 前驱状态 | 转移方式 |
|---|---|
| s | Stay |
| s-1 | Advance |
| s-2 | Skip |

**（1）时间帧循环**

第0帧已经完成初始化，因此从第1帧开始：

```python
for t in range(1, T):
    prev = rows[-1]
    current = []
```

其中：

- `prev`：上一时间帧的 DP 状态。
- `current`：当前时间帧的 DP 状态。

**（2）状态循环**

遍历当前时间帧的所有扩展目标状态：

```python
for s in range(S):
```

**（3）Stay**

从状态 s 转移到状态 s。

```python
predecessors = [prev[s]]
```

Stay 对所有合法状态都允许，包括连续输出多个 Blank 的情况。

**（4）Advance**

从状态 s-1 转移到状态 s。

条件：

$$
s\geq1
$$

代码：

```python
if s >= 1:
    predecessors.append(
        prev[s - 1]
    )
```

**（5）Skip**

从状态 s-2 直接转移到状态 s。

必须同时满足三个条件：

$$
s\geq2
$$

$$
y'_s\neq blank
$$

$$
y'_s\neq y'_{s-2}
$$

代码：

```python
if (
    s >= 2
    and extended_target[s] != blank
    and extended_target[s] != extended_target[s - 2]
):
    predecessors.append(
        prev[s - 2]
    )
```

Skip 条件的理解：

- 不同 Token 之间可以 Skip。
- 相同 Token 之间不能 Skip。
- 当前 Token 是 Blank 时不能 Skip。

例如：

```text
目标 ab：

_ a _ b _
```

当前到达状态3：

- 前驱状态1：Skip。
- 前驱状态2：Advance。
- 前驱状态3：Stay。

因为当前 Token 为 b，前两个状态对应的 Token 为 a，二者不同，因此 Skip 合法。

而对于：

```text
目标 aa：

_ a _ a _
```

当前到达状态3：

- 前驱状态1：Skip。
- 前驱状态2：Advance。
- 前驱状态3：Stay。

由于当前 Token 与状态1的 Token 相同，因此 Skip 不合法。

**（6）LogSumExp**

收集所有合法前驱状态后，使用：

```python
prev_sum = torch.logsumexp(
    torch.stack(predecessors),
    dim=0
)
```

`torch.stack()` 将多个标量 Log 概率组合成一个 Tensor。

`torch.logsumexp()` 计算：

$$
\log\left(
e^{a_1}+e^{a_2}+\cdots+e^{a_n}
\right)
$$

不能使用普通 `sum()`，因为直接相加 Log 概率对应的是概率连乘，而不是概率求和。

**（7）当前帧发射概率**

最后加上当前时间帧输出目标 Token 的 Log 概率：

```python
current_value = (
    log_probs[t, extended_target[s]]
    + prev_sum
)
```

对应：

$$
A_t(s)=
\log p_t(y'_s)
+
\operatorname{LogSumExp}(\text{合法前驱})
$$

保存当前状态：

```python
current.append(current_value)
```

当前时间帧所有状态计算完成后：

```python
rows.append(torch.stack(current))
```

---

### 5.5 终止状态

对于非空目标，扩展目标最后两个状态分别为：

- S-2：最后一个目标 Token。
- S-1：末尾 Blank。

两种结束方式都可以得到完整目标文本。

例如目标为 a：

```text
_ a _
```

合法结束方式：

```text
a a
```

```text
_ a
```

第一种结束于目标 Token 状态，第二种结束于末尾 Blank 状态。

因此，最终需要对两个终止状态求和。

普通概率空间：

$$
P(y|x)=
\alpha_{T-1}(S-2)
+
\alpha_{T-1}(S-1)
$$

Log 空间：

$$
\boxed{
\log P(y|x)=
\operatorname{LogSumExp}
\left(
A_{T-1}(S-2),
A_{T-1}(S-1)
\right)
}
$$

代码：

```python
log_likelihood = torch.logsumexp(
    alpha[-1, S - 2:S],
    dim=0
)

return log_likelihood
```

这里不能只取 S-1，否则会遗漏最后一个时间帧直接输出目标 Token 的合法路径。

也不能取最大值，因为我们需要的是所有合法路径的概率之和。

**补充：合法中间状态不一定是终止状态。**

例如目标为 a，状态0对应 Blank。

连续输出：

```text
_ _
```

是合法的中间路径，但经过 CTC Collapse 后得到空序列，而不是目标 a。

因此，状态0可以在递推中继续 Stay，但不能作为目标 a 的最终终止状态。

---

### 5.6 特殊情况

**（1）空目标**

如果目标为空，扩展目标只有一个 Blank。

所有时间帧必须输出 Blank。

当 T 大于0：

$$
\log P(\varnothing|x)
=
\sum_{t=0}^{T-1}
\log p_t(\varnothing)
$$

代码：

```python
if U == 0:

    if T == 0:
        return log_probs.new_zeros(())

    return log_probs[:, blank].sum()
```

当 T=0 时，空路径概率为1，因此 Log 概率为0：

$$
\log 1=0
$$

**（2）时间帧数量为0**

如果目标非空：

$$
T=0,\quad U>0
$$

不存在合法路径：

$$
\log P(y|x)=-\infty
$$

代码：

```python
if T == 0:
    return log_probs.new_full(
        (),
        float("-inf")
    )
```

**（3）不可行对齐**

例如：

```python
target = [1, 1]
T = 2
```

扩展目标：

```text
_ a _ a _
```

两个相同 Token 之间必须经过 Blank。

因此至少需要三个时间帧：

```text
a _ a
```

当 T=2 时，无法完成合法对齐。

由于 Skip 不合法，最终两个终止状态的概率均为0：

$$
\log P(y|x)=-\infty
$$

这种情况不需要额外编写特殊分支，前向递推本身就能够得到正确结果。

---

## 6. 正确性测试

代码能够正常运行并不意味着算法一定正确，因此需要通过不同情况验证。

### 6.1 普通目标测试

假设两个时间帧，词表只有 Blank 和 a。

概率矩阵：

| 时间帧 | Blank | a |
|---|---:|---:|
| t=0 | 0.6 | 0.4 |
| t=1 | 0.3 | 0.7 |

目标：

```python
target = [1]
```

所有合法路径：

```text
a a
a _
_ a
```

对应概率：

$$
P(a|x)
=
0.4\times0.7
+
0.4\times0.3
+
0.6\times0.7
$$

$$
=0.28+0.12+0.42
$$

$$
\boxed{P(a|x)=0.82}
$$

因此：

$$
\exp(\text{result})=0.82
$$

**完整 DP 计算：**

第一行：

$$
A_0=
[\log0.6,\log0.4,-\infty]
$$

第二行状态0：

只有 Stay：

$$
A_1(0)=\log0.3+\log0.6
$$

$$
=\log0.18
$$

第二行状态1：

Stay 和 Advance：

$$
A_1(1)=
\log0.7+
\operatorname{LogSumExp}(\log0.4,\log0.6)
$$

$$
=\log0.7
$$

第二行状态2：

Stay 和 Advance：

$$
A_1(2)=
\log0.3+
\operatorname{LogSumExp}(-\infty,\log0.4)
$$

$$
=\log0.12
$$

最终：

$$
P(a|x)=0.7+0.12=0.82
$$

注意：状态0虽然不是终止状态，但可以在后续时间帧继续 Stay。例如路径 `_ _`。它是合法的中间状态，只是不能作为目标 a 的最终结束状态。

**测试代码：**

```python
probs = torch.tensor(
    [
        [0.6, 0.4],
        [0.3, 0.7]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

result = ctc_forward_log(
    log_probs,
    target=[1],
    blank=0
)

assert torch.allclose(
    result.exp(),
    torch.tensor(0.82)
)

print("Test 1 passed")
```

### 6.2 重复 Token 测试

```python
probs = torch.tensor(
    [
        [0.5, 0.5],
        [0.5, 0.5]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

result = ctc_forward_log(
    log_probs,
    target=[1, 1],
    blank=0
)

assert torch.isneginf(result)

print("Test 2 passed")
```

原因：两个相同 Token 之间必须经过 Blank，至少需要三个时间帧。

### 6.3 空目标测试

```python
probs = torch.tensor(
    [
        [0.6, 0.4],
        [0.3, 0.7]
    ],
    dtype=torch.float32
)

log_probs = torch.log(probs)

result = ctc_forward_log(
    log_probs,
    target=[],
    blank=0
)

expected = torch.log(
    torch.tensor(0.6 * 0.3)
)

assert torch.allclose(
    result,
    expected
)

print("Test 3 passed")
```

原因：只有全 Blank 路径能够产生空目标。

### 6.4 不可行对齐测试

```python
log_probs = torch.log(
    torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5]
        ],
        dtype=torch.float32
    )
)

result = ctc_forward_log(
    log_probs,
    target=[1, 1],
    blank=0
)

assert torch.isneginf(result)

print("Test 4 passed")
```

### 6.5 零时间帧测试

```python
log_probs = torch.empty(
    (0, 2),
    dtype=torch.float32
)

# 空目标
result_empty = ctc_forward_log(
    log_probs,
    target=[],
    blank=0
)

assert result_empty.item() == 0.0

# 非空目标
result_nonempty = ctc_forward_log(
    log_probs,
    target=[1],
    blank=0
)

assert torch.isneginf(result_nonempty)

print("Test 5 passed")
```

### 6.6 测试结果

所有测试通过后，说明：

- 普通目标的路径概率求和正确。
- 重复 Token 的 Skip 限制正确。
- 空目标处理正确。
- 不可行对齐处理正确。
- 零时间帧处理正确。

---

## 7. 本模块总结

### 7.1 函数整体流程

`ctc_forward_log()` 的执行流程：

1. 检查输入是否合法。
2. 处理空目标和零时间帧。
3. 构造扩展目标。
4. 初始化 DP 矩阵。
5. 遍历时间帧和目标状态。
6. 根据 Stay、Advance、Skip 收集合法前驱。
7. 使用 LogSumExp 汇总前驱概率。
8. 加上当前时间帧的发射概率。
9. 对最后两个终止状态求和。
10. 返回目标序列的 Log 概率。

### 7.2 三种状态转移

| 转移 | 前驱状态 | 条件 |
|---|---|---|
| Stay | s | 始终允许 |
| Advance | s-1 | s>=1 |
| Skip | s-2 | s>=2，当前 Token 非 Blank，且与前两个状态 Token 不同 |

核心 Skip 条件：

$$
s\geq2
$$

$$
y'_s\neq blank
$$

$$
y'_s\neq y'_{s-2}
$$

### 7.3 核心递推公式

$$
A_t(s)=
\log p_t(y'_s)
+
\operatorname{LogSumExp}(\text{合法前驱})
$$

### 7.4 最终终止公式

$$
\log P(y|x)=
\operatorname{LogSumExp}
\left(
A_{T-1}(S-2),
A_{T-1}(S-1)
\right)
$$

### 7.5 特殊情况

- 空目标：所有时间帧必须输出 Blank。
- 零时间帧、空目标：Log 概率为0。
- 零时间帧、非空目标：Log 概率为负无穷。
- 不可行对齐：最终返回负无穷。

### 7.6 本模块核心认识

CTC 前向算法本质上是一个动态规划问题。

通过扩展目标将目标序列转换为状态序列，再通过 Stay、Advance、Skip 三种转移计算所有合法路径的概率。

使用 Log 空间和 LogSumExp 保证数值稳定，最后将两个合法终止状态的概率相加，得到目标序列的总概率。

至此，完成 `ctc_forward_log()` 的数学推导与代码实现。