# W7D3 模块1：编辑距离与动态规划

## 1. 编辑距离的基本概念

Reference（参考文本）是什么：
真实标注文本，作为模型识别结果的对照标准。

Hypothesis（预测文本）是什么：
ASR模型输出的识别文本。

编辑距离的定义：
编辑距离是指将Reference转换为Hypothesis所需的最少编辑操作次数。

S（Substitution）的含义：将Reference中的一个token替换为另一个token

D（Deletion）的含义：删除Reference中的一个token，即模型漏识别了一个token

I（Insertion）的含义：在Hypothesis中出现了Reference不存在的token，即模型多识别了一个token。

WER 的计算公式：
Word Error Rate（词错误率）表示识别错误的词数占 Reference 总词数的比例。
$$
\boxed{WER=\frac{S+D+I}{N}}
$$

## 2. 动态规划状态定义

参考序列的长度：
$$
m=|Reference|
$$

预测序列的长度：
$$
n=|Hypothesis|
$$

dp[i][j] 表示什么：
dp[i][j] 表示将 Reference 的前 i 个 token 转换为 Hypothesis 的前 j 个 token 所需要的最少编辑操作次数。
即：
$$
dp[i][j]=d(R_{1},H_{1})
$$

为什么 DP 矩阵需要多一行和一列：
为了表示空序列的情况。
DP 矩阵的形状为：
$$
(m+1)\times(n+1)
$$
第一行表示 Reference 为空序列时，逐步转换为 Hypothesis 的不同前缀。
第一列表示 Hypothesis 为空序列时，逐步将 Reference 的不同前缀删除。

dp[0][0] 的含义：
将空序列转换为空序列，不需要任何操作。

## 3. 动态规划初始化

dp[i][0] 的含义和初始值：
将 Reference 的前 i 个 token 转换为空序列，需要删除 i 个 token。
$$
dp[i][0]=i
$$

dp[0][j] 的含义和初始值：
将空序列转换为Hypothesis的前j个token，需要插入j个token
$$
dp[0][j]=j
$$

为什么这样初始化：
动态规划需要从已经求解的子问题出发，逐步计算更大的问题。
空序列是最简单的初始情况：
- 空序列转换为空序列：0次操作。
- 有 i 个 token 转换为空序列：删除 i 次。
- 空序列转换为有 j 个 token 的序列：插入 j 次。

## 4. 状态转移

当前两个 token 相同时：
如果Reference的第i个token与Hypothesis的第j个token相同，那么不需要额外操作，直接继承左上角的值。
$$
dp[i][j]=dp[i-1][j-1]
$$

当前两个 token 不同时：

删除操作对应哪个状态：
$$
dp[i-1][j]+1
$$
表示先将 Reference 前 i-1 个 token 转换为 Hypothesis 前 j 个 token，然后删除 Reference 多出来的第 i 个 token。

插入操作对应哪个状态：
$$
dp[i][j-1]+1
$$
表示先将 Reference 前 i 个 token 转换为 Hypothesis 前 j-1 个 token，然后插入 Hypothesis 多出来的第 j 个 token。

替换操作对应哪个状态：
$$
dp[i-1][j-1]+1
$$
表示先将 Reference 前 i-1 个 token 转换为 Hypothesis 前 j-1 个 token，然后将当前两个不同的 token 替换。

完整状态转移公式：
$$
dp[i][j]=\min
\begin{cases}
dp[i-1][j]+1 & \text{删除}\
dp[i][j-1]+1 & \text{插入}\
dp[i-1][j-1]+c & \text{匹配或替换}
\end{cases}
$$
其中：

$$
c=
\begin{cases}
0 & r_{i}=h_{j}\
1 & r_{i}\neq h_{j}
\end{cases}
$$

通过比较三种操作的代价，选择最小值作为当前状态的最优解。

## 5. 实验结果

Reference：
I like apples

Hypothesis：
I love apples today

Reference token 数量：
3

Hypothesis token 数量：
4

DP 矩阵形状：
$$
(3+1)\times(4+1)=(4,5)
$$

最终编辑距离：
取 DP 矩阵右下角的值：
$$
d=dp[3][4]=2
$$

对应的 WER：
其中有1次替换、0次删除、1次插入。
$$
S=1,\quad D=0,\quad I=1
$$
$$
WER=\frac{1+0+1}{3}
=\frac{2}{3}
\approx66.67%
$$

## 6. 当前理解

为什么编辑距离可以用动态规划：
因为较长序列之间的编辑距离可以通过较短序列之间的编辑距离计算得到。
将大问题拆分为多个具有重叠子问题的小问题，保存已经计算的结果，避免重复计算/

dp[m][n] 为什么表示最小编辑距离：
因为 DP 矩阵按照从短前缀到长前缀的顺序计算。
每一个状态都通过三种可能操作选择最小代价，因此最终右下角的 dp[m][n] 就是将完整 Reference 转换为完整 Hypothesis 所需的最少编辑操作次数。

为什么仅有编辑距离还不能确定 S、D、I 各自的次数：
因为同一个最小编辑距离可能对应不同的编辑路径。


## 7. 模块验收

Q1：在 dp[i][j] 中，i 和 j 分别代表什么？为什么矩阵需要多出第一行和第一列？
i 代表 Reference 的前 i 个 token，j 代表 Hypothesis 的前 j 个 token。
多出的一行和一列用于表示空序列的情况，方便初始化动态规划矩阵。

Q2：为什么 dp[i-1][j]+1 表示删除，而 dp[i][j-1]+1 表示插入？
dp[i-1][j] 表示已经将 Reference 前 i-1 个 token 转换为 Hypothesis 前 j 个 token。
如果现在需要处理 Reference 前 i 个 token，而 Hypothesis 不变，那么 Reference 多出来的第 i 个 token 就需要删除。
因此：
$$
dp[i-1][j]+1
$$
表示删除。
同理，dp[i][j-1] 表示已经将 Reference 前 i 个 token 转换为 Hypothesis 前 j-1 个 token。表示插入。


Q3：对于 Reference: I like apples，Hypothesis: I love apples today，分别有多少次替换、删除和插入？WER 是多少？
$$
S=1,\quad D=0,\quad I=1
$$
$$
WER=\frac{1+0+1}{3}=66.67%
$$

Q4：如果现在已经算出最小编辑距离为3，是否一定能够直接确定 S、D、I 的具体次数？为什么？
不一定。
因为不同的编辑路径可能具有相同的最小编辑距离，但其中替换、删除和插入的次数不同。
需要通过 Backtrace 回溯最优编辑路径，才能统计具体的 S、D、I。


# W7D3 模块2：Backtrace 与 S/D/I 统计

## 1. Backtrace 的作用

为什么仅有编辑距离还不够：
DP矩阵右下角的值只能表示将完整Reference转换为Hypothesis所需要的最少编辑操作次数，无法直接确定其中替换、删除和插入的具体次数。

Backtrace 的起点和终点：

- 起点：DP 矩阵右下角 dp[m][n]。
- 终点：DP 矩阵左上角 dp[0][0]。

Backtrace 的核心思想：
根据 DP 状态转移关系，从右下角反向寻找一条最优路径。
通过判断当前位置与相邻位置的数值关系，确定当前操作属于匹配、替换、删除还是插入。

为什么回溯方向与最终编辑操作顺序相反：
DP 是从左上角向右下角计算的，而 Backtrace 从右下角向左上角寻找路径。
因此，回溯得到的操作顺序与正常文本顺序相反，需要使用 reverse() 将操作列表反转。

## 2. 回溯方向与操作类型

左上角转移表示什么：
从 dp[i][j] 回溯到 dp[i-1][j-1]。
如果当前两个 token 相同，表示正确匹配 C：
$$
dp[i][j]=dp[i-1][j-1]
$$
如果当前两个 token 不同，表示替换 S：
$$
dp[i][j]=dp[i-1][j-1]+1
$$

上方转移表示什么：
从 dp[i][j] 回溯到 dp[i-1][j]，表示删除 D。
$$
dp[i][j]=dp[i-1][j]+1
$$

左侧转移表示什么：
从 dp[i][j] 回溯到 dp[i][j-1]，表示插入 I。
$$
dp[i][j]=dp[i][j-1]+1
$$

为什么要检查 DP 值是否满足转移关系：
因为回溯需要沿着最优编辑路径进行。
只有满足对应的DP状态转移关系，才能保证当前选择的操作仍然能够达到最小编辑距离。

## 3. 多条最优路径

为什么可能存在多条最优编辑路径：
因为不同的编辑操作组合可能具有相同的总编辑距离。

不同最优路径的 S/D/I 是否一定相同：
不一定。
不同路径可能具有不同的替换、删除和插入次数，但只要是最优路径，其总编辑距离就相同。

本次实现的优先级规则：
为了保证程序每次运行得到确定的结果，设置固定的回溯优先级：
1. 正确匹配 C。
2. 替换 S。
3. 删除 D。
4. 插入 I。
当多个操作都满足最优转移条件时，按照上述顺序选择。

为什么与 jiwer 比较时需要注意最优路径的选择：
不同实现可能采用不同的平局处理策略，因此相同的Reference和Hypothesis可能得到不同的S/D/I分解。
但只要使用相同的文本处理和tokenization规则，最小编辑距离和总WER应当一致。

## 4. metrics.py 的实现

build_dp() 的作用：
根据 Reference 和 Hypothesis 构建动态规划矩阵。
返回完整 DP 表，右下角 dp[m][n] 为最小编辑距离。

backtrace() 的作用：
从 DP 矩阵右下角开始，根据状态转移关系寻找一条最优编辑路径。
返回操作列表，每个操作包含：

- 操作类型：C、S、D、I。
- Reference 对应 token。
- Hypothesis 对应 token。

score_tokens() 的作用：
将 DP 和 Backtrace 组合起来，完成 token 级错误统计。
返回：

- S：替换次数。
- D：删除次数。
- I：插入次数。
- C：正确匹配次数。
- N：Reference token 数量。
- distance：最小编辑距离。
- error_rate：错误率。
- operations：具体编辑操作。

为什么需要记录正确匹配 C：
C 表示 Reference 与 Hypothesis 中成功匹配的 token 数量。
它虽然不参与编辑距离计算，但可以帮助检查回溯是否完整。

为什么要检查 S+D+I 与编辑距离是否一致：
DP 计算得到的编辑距离应该等于 Backtrace 统计的替换、删除和插入次数之和。
$$
\boxed{S+D+I=distance}
$$
另外还可以检查：
$$
\boxed{C+S+D=N}
$$

## 5. 实验结果

Reference：
I like apples

Hypothesis：
I love apples today

Backtrace 路径：
C: I -> I
S: like -> love
C: apples -> apples
I: None -> today

替换次数 S：
1

删除次数 D：
0

插入次数 I：
1

正确匹配次数 C：
2

最终编辑距离：
2

WER：
$$
WER=\frac{1+0+1}{3}
\approx66.67%
$$

## 6. 模块验收

Q1：为什么 Backtrace 要从 dp[m][n] 开始，而不是从 dp[0][0] 开始？为什么最后还需要执行 operations.reverse()？
因为 dp[m][n] 表示完整 Reference 转换为完整 Hypothesis 的最小编辑距离。
从右下角开始，可以根据已经计算好的 DP 数值反向寻找最优编辑路径。
由于回溯是从序列末尾向开头进行的，所以记录的操作顺序与正常文本顺序相反，最后需要使用 reverse() 反转操作列表。


Q2：假设当前位置是 dp[i][j]，发现 dp[i][j]=dp[i-1][j]+1，为什么可以沿上方回溯，并记录一次删除？这里被删除的是哪个 token？
因为 dp[i-1][j] 表示将 Reference 前 i-1 个 token 转换为 Hypothesis 前 j 个 token 的最小编辑距离。
现在 Reference 多出了第 i 个 token，而 Hypothesis 不变，因此需要删除 Reference 的第 i 个 token。
对应：

$$
reference_tokens[i-1]
$$

Q3：对于 Reference = "a b"、Hypothesis = "b a"，为什么最小编辑距离虽然固定，但 S、D、I 的统计结果可能不同？这会不会导致 WER 不同？
因为可能存在多条具有相同最小编辑距离的路径，不同路径的替换、删除和插入次数可能不同。
但是不会导致 WER 不同，因为：
$$
WER=\frac{S+D+I}{N}
$$

Q4：为什么 score_tokens() 既要记录 C（正确匹配），又要执行下面的断言？ssert S + D + I == distance
C 用于记录正确匹配的 token 数量，可以检查 Reference 和 Hypothesis 是否被完整对齐。
断言用于检查 Backtrace 统计的替换、删除和插入次数之和是否与 DP 计算的最小编辑距离一致。
两者可以帮助验证回溯结果的正确性。


# W7D3 模块3：jiwer 对照、边界测试与语料级 WER

## 1. jiwer 的作用

**jiwer 是什么：**

jiwer 是一个用于语音识别评估的 Python 库，支持计算 WER（词错误率）、CER（字符错误率）、MER（匹配错误率）等指标。

**为什么需要与 jiwer 对照：**

自己编写的 DP 和 Backtrace 即使能够正常运行，也不代表计算结果一定正确。

因此，需要使用成熟的评估工具进行交叉验证，检查手写算法是否存在逻辑错误。

**jiwer 的 WER 计算方式：**

对于 Reference 非空的情况：

$$
WER=\frac{S+D+I}{N}
$$

其中 N 为 Reference 的词数。

使用方式：

```python
from jiwer import process_words

result = process_words(reference, hypothesis)

print(result.wer)
```

**jiwer 返回的 S/D/I/C 分别是什么：**

- substitutions：S，替换次数。
- deletions：D，删除次数。
- insertions：I，插入次数。
- hits：C，正确匹配次数。

其中 C 不参与 WER 分子的计算。

**为什么不同实现的 S/D/I 可能不同，但 WER 相同：**

因为不同的最优编辑路径可能产生不同的 S/D/I 分解。

例如，两次替换与一次删除加一次插入，编辑距离都可能为2。

只要文本预处理、分词方式和编辑代价相同，不同最优路径的总编辑距离就一致，因此 WER 一致。

## 2. 边界测试

**空 Reference、空 Hypothesis：**

空序列会影响 DP 的初始化和 WER 的分母，因此需要单独测试。

**Reference 和 Hypothesis 都为空：**

```text
Reference:  []
Hypothesis: []
```

$$
S=0,\quad D=0,\quad I=0,\quad C=0
$$

编辑距离为0。

由于 Reference 词数为0，普通 WER 公式出现零分母。

当前手写实现：

```python
error_rate = None
```

本次实验中 jiwer 返回：

```python
WER = 0
```

这是 jiwer 对空输入采用的特殊约定。

**Reference 为空但 Hypothesis 不为空：**

```text
Reference:  []
Hypothesis: ["I", "like", "apples"]
```

由于 Reference 中没有词，而 Hypothesis 有3个词，因此全部属于插入错误。

$$
S=0,\quad D=0,\quad I=3,\quad C=0
$$

$$
distance=3
$$

普通 WER 公式的分母为0，数学上未定义。

当前手写实现返回 `None`。

本次实验中 jiwer 返回 `WER=3`，采用了空 Reference 时返回插入词数的特殊约定。

**Reference 不为空但 Hypothesis 为空：**

```text
Reference:  ["I", "like", "apples"]
Hypothesis: []
```

模型没有输出任何词，因此 Reference 中的全部3个词都需要删除。

$$
S=0,\quad D=3,\quad I=0,\quad C=0
$$

$$
WER=\frac{3}{3}=100\%
$$

手写实现和 jiwer 均得到1.0。

**重复词测试：**

```text
Reference:  a a b
Hypothesis: a b
```

Reference 有3个 token，Hypothesis 有2个 token。

最少需要删除其中一个 `a`。

$$
S=0,\quad D=1,\quad I=0,\quad C=2
$$

$$
WER=\frac{1}{3}\approx33.33\%
$$

不能使用 set 比较序列，因为 set 会丢失重复次数和排列顺序。

**为什么需要测试边界情况：**

正常样本运行成功不代表算法在特殊输入下仍然正确。

边界测试可以检查：

- DP 初始化是否正确。
- Backtrace 能否处理空序列。
- 是否发生除零错误。
- 重复 token 是否被正确处理。
- 错误次数与编辑距离是否一致。

## 3. 语料级 WER

**单条音频 WER：**

对于第 i 条具有非空 Reference 的音频：

$$
WER_i=\frac{S_i+D_i+I_i}{N_i}
$$

**多条音频的语料级 WER：**

语料级 WER 使用所有音频的总编辑次数除以总 Reference 词数。

$$
\boxed{
WER_{corpus}
=
\frac{\sum_{i=1}^{M}(S_i+D_i+I_i)}
{\sum_{i=1}^{M}N_i}
}
$$

其中 M 表示音频样本数。

**为什么不能直接平均每条音频的 WER：**

因为不同音频包含的 Reference 词数不同。

直接平均单句 WER 相当于让每条音频拥有相同权重，而不是让每个参考词拥有相同权重。

例如：

| 样本 | Reference词数 | 编辑错误数 | WER |
|---|---:|---:|---:|
| A | 1 | 1 | 100% |
| B | 9 | 0 | 0% |

直接平均：

$$
WER_{mean}=\frac{100\%+0\%}{2}=50\%
$$

正确的语料级 WER：

$$
WER_{corpus}=\frac{1+0}{1+9}=10\%
$$

50%表示平均每条音频的错误率，而10%表示整个语料的词错误率。

对于 ASR 的标准语料级 WER 评估，应使用后者。

**总编辑次数与总 Reference token 数量：**

先累计全部样本的 S、D、I 和 N：

$$
S_{total}=\sum_i S_i
$$

$$
D_{total}=\sum_i D_i
$$

$$
I_{total}=\sum_i I_i
$$

$$
N_{total}=\sum_i N_i
$$

然后计算：

$$
WER_{corpus}
=
\frac{S_{total}+D_{total}+I_{total}}
{N_{total}}
$$

空 Reference 样本虽然对分母贡献为0，但它产生的插入错误仍然需要进入分子。

## 4. metrics.py 的完善

**单条样本指标：**

`score_tokens(reference_tokens, hypothesis_tokens)` 计算单条样本的：

- S、D、I、C。
- N：Reference token 数量。
- distance：最小编辑距离。
- error_rate：单条样本错误率。
- operations：具体编辑操作。

**语料级指标：**

`score_corpus(token_pairs)` 接收多组 Reference 和 Hypothesis token 列表。

对每条样本调用 `score_tokens()`，然后累加：

```python
totals = {
    "S": 0,
    "D": 0,
    "I": 0,
    "C": 0,
    "N": 0,
    "distance": 0,
}
```

最终计算：

```python
error_rate = (
    totals["distance"] / totals["N"]
    if totals["N"] > 0
    else None
)
```

注意：不能直接对每条样本的 `error_rate` 求算术平均作为语料级 WER。

**如何处理空 Reference：**

单条样本中，如果 Reference 为空：

```python
error_rate = None
```

这样避免零分母，并明确表示普通比例公式无法定义。

语料级统计时，仍然累计该样本的编辑次数和 Reference token 数量。

如果整个语料的总 Reference 词数大于0，就可以计算总体 WER。

如果整个语料的总 Reference 词数也为0，当前实现返回 `None`。

**如何验证自己的实现与 jiwer 一致：**

先确保双方采用一致的文本预处理和分词规则。

然后对照：

```python
from jiwer import process_words

result = process_words(
    references,
    hypotheses,
)

print(result.wer)
```

主要检查：

$$
S+D+I=distance
$$

以及：

$$
WER_{ours}=WER_{jiwer}
$$

对于存在多条最优路径的样本，S/D/I 的具体分解可能不同，因此不能仅因某一类错误次数不同就判断实现错误。

还可以验证：

$$
C+S+D=N
$$

$$
C+S+I=|H|
$$

用于检查 Backtrace 是否完整覆盖两个序列。

## 5. 实验结果

**jiwer 版本：**

本次未记录具体版本号，可使用以下代码查看：

```python
from importlib.metadata import version

print(version("jiwer"))
```

**正常样本对照结果：**

```text
Reference:  I like apples
Hypothesis: I love apples today
```

手写实现：

```text
S=1
D=0
I=1
C=2
N=3
distance=2
WER=0.6666666666666666
```

jiwer：

```text
S=1
D=0
I=1
C=2
WER=0.6666666666666666
```

两者完全一致，对照通过。

**空字符串测试结果：**

| 测试情况 | 手写 WER | jiwer WER |
|---|---:|---:|
| 两者都为空 | None | 0 |
| Reference 空，Hypothesis 有3词 | None | 3 |
| Reference 有3词，Hypothesis 空 | 1.0 | 1.0 |

编辑操作统计均符合预期。两种实现对空 Reference 的单句错误率约定不同。

**重复词测试结果：**

```text
Reference:  a a b
Hypothesis: a b

S=0
D=1
I=0
C=2
distance=1
WER=0.3333333333333333
```

手写边界测试通过。本次没有单独记录 jiwer 对该重复词样本的输出。

**语料级 WER：**

样本A：1个参考词，1次错误。

样本B：9个参考词，0次错误。

实验输出：

```text
总参考词数: 10
总编辑次数: 1
直接平均单句WER: 0.5
正确语料级WER: 0.1
jiwer语料级WER: 0.1

语料级WER验收通过
```

另外测试空 Reference 样本：

```text
总参考词数: 1
总插入次数: 1
语料级错误率: 1.0
```

说明空 Reference 样本中的插入错误也可以正确参与语料级统计。

**最终验收：**

- DP 编辑距离计算通过。
- Backtrace 的 S/D/I/C 统计通过。
- 正常样本与 jiwer 对照通过。
- 空字符串边界测试通过。
- 重复词测试通过。
- 语料级 WER 计算通过。
- 语料级 WER 与 jiwer 对照通过。

## 6. 模块验收

**Q1. 为什么不能简单平均每条音频的 WER？为什么50%不如10%适合表示整个语料的词错误率？**

因为不同音频的 Reference 词数可能不同。

直接平均会让每条音频获得相同权重，而语料级 WER 需要按 Reference 词数进行总体统计。

例如，一条只有1个参考词的音频和一条有9个参考词的音频，如果分别有1次错误和0次错误：

$$
WER_{mean}=50\%
$$

但整个语料只有1次错误、10个参考词：

$$
WER_{corpus}=10\%
$$

因此10%才是该语料的总体词错误率。

**Q2. 一条音频的 Reference 为空、Hypothesis 包含2个词。它应该如何影响语料级 WER 的分子和分母？**

该样本产生2次插入错误：

$$
I=2
$$

因此：

- 语料级分子增加2。
- 语料级分母增加0。

如果语料中其他样本有参考词，整体 WER 仍然可以正常计算。

**Q3. 如果语料库有1000个参考词，S=80、D=30、I=20，语料级 WER 是多少？**

$$
WER=\frac{80+30+20}{1000}
$$

$$
\boxed{WER=13\%}
$$

**Q4. score_corpus() 为什么累计各样本的 distance 和 N，而不是累计 error_rate 再除以样本数？**

因为语料级 WER 的定义是总编辑次数除以总参考词数。

直接平均单句错误率会让不同长度的音频具有相同权重，不能正确反映整个语料的词错误率。

另外，当某条音频的 Reference 为空时，单句 error_rate 可能为 None，也无法直接参与普通的算术平均。

因此需要累计：

$$
distance_{total}=\sum_i distance_i
$$

$$
N_{total}=\sum_i N_i
$$

最后计算：

$$
WER_{corpus}=\frac{distance_{total}}{N_{total}}
$$