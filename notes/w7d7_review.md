# W7D7 模块1：真实 ASR 错误分析

## 1. 为什么需要进行 ASR 错误分析？

CER 和 MER 能够衡量模型预测文本与 Reference 之间的差异，但无法直接说明错误产生的原因。

相同的编辑距离可能来自不同问题：

- 模型没有正确识别音频。
- 人工标注的 Reference 存在错误。
- 文本规范化过程产生了不合理的变化。

因此，ASR 评估不仅需要计算指标，还需要分析具体错误来源。

核心流程：

```text
发现识别差异
    ↓
检查音频与 Reference
    ↓
检查规范化结果
    ↓
确定错误来源
    ↓
采取对应改进措施
```

## 2. 三类错误的定义与区别

### 2.1 模型错误（Model Error）

**定义：**

音频本身能够支持正确的文本，但 ASR 模型生成了不同的文本。

**典型表现：**

- 同音字、近音字识别错误。
- 专有名词识别错误。
- 漏字或多字。
- 噪声、口音等导致识别错误。

例如：

Reference：

```text
另一联赛是阿根廷全国足球甲级联赛
```

Hypothesis：

```text
另一连赛是阿根廷全国足球假籍连赛
```

假设 Reference 正确，则差异属于模型识别错误。

**处理方式：**

- 检查音频质量。
- 统计常见混淆词。
- 比较不同模型。
- 在 Dev 上研究 Prompt 和解码参数。
- 分析模型的领域适应能力。

### 2.2 标注错误（Annotation Error）

**定义：**

人工 Reference 与音频实际内容不一致。

**典型表现：**

音频：

```text
今天学习语音识别
```

Reference：

```text
今天学习语音合成
```

Hypothesis：

```text
今天学习语音识别
```

模型实际上识别正确，但由于 Reference 错误，计算 CER 时仍然产生替换错误。

**处理方式：**

- 重新听取原始音频。
- 核对人工 Reference。
- 根据标注规范修正文本。
- 对无法确定的内容标记 needs_review。

### 2.3 规范化错误（Normalization Error）

**定义：**

文本规范化过程不符合既定评分规则，导致 Reference 和 Hypothesis 被不合理地修改或处理不一致。

**典型表现：**

Reference：

```text
GPT-4
```

Hypothesis：

```text
gpt-4
```

如果双方都经过统一小写转换，则两者应该相同。

繁简体也是一个需要注意的例子：

```text
这 ≠ 這
```

当前 NFKC 不会自动完成繁简体转换。

因此，按照当前评分规范，两者仍然会产生差异。

**处理方式：**

- 检查双方是否使用相同规范化函数。
- 检查数字、英文、特殊符号的处理。
- 明确繁简体是否需要统一。
- 修改规则后更新 NORMALIZER_VERSION。
- 对双方重新规范化并重新计算指标。

## 3. 如何区分三类错误？

```text
发现 Reference 与 Hypothesis 不一致
                  ↓
          检查原始音频
                  ↓
       Reference 是否符合音频？
           /              \
         否                是
         ↓                 ↓
     标注错误          检查规范化
                           ↓
                   规范化是否合理？
                      /       \
                    否         是
                    ↓          ↓
                规范化错误   模型错误
```

注意：一条样本可能同时存在多种错误，需要针对具体片段分析。

## 4. 真实 Dev 样本分析

**UID：**

cv26_20558b6a5c8dbac9

**Reference：**

另一联赛是阿根廷全国足球甲级联赛。

**Hypothesis：**

另一连赛是阿根廷全国足球假籍连赛

**错误位置及类型：**

在确认 Reference 正确的前提下，存在四处字符替换：

- 联 → 连
- 甲 → 假
- 级 → 籍
- 联 → 连

**CER：**

参考字符数为16，编辑次数为4。

$$
CER=\frac{4}{16}=0.25
$$

**最终错误归因：**

模型识别错误。

## 5. 错误分析注意事项

1. 不能只看 CER/MER 判断错误来源。
2. 发现差异后先检查音频和 Reference。
3. Reference 错误时不能归因于模型。
4. Reference 正确后还需要检查规范化。
5. 不能为了降低 CER 随意修改 Hypothesis。
6. 修改评分规则需要更新版本并重新评估。
7. Dev 可以用于模型优化和错误分析。
8. Test 用于独立评估，不能根据 Test 结果反复调参。

## 6. 本模块总结

模型错误：模型没有正确识别音频。

标注错误：人工 Reference 与音频实际内容不一致。

规范化错误：文本处理过程不符合既定评分规范。

错误分析的核心是先定位来源，再采取对应的改进措施，而不是看到 CER 上升就直接修改模型。


# W7D7 模块2：编辑距离递推与 Backtrace

## 1. 编辑距离的定义

编辑距离（Edit Distance）用于衡量两个 Token 序列之间的差异。

通过三种操作将 Reference 转换为 Hypothesis：

| 操作 | 含义 | 代价 |
|---|---|---:|
| S（Substitution） | 替换 | 1 |
| D（Deletion） | 删除 | 1 |
| I（Insertion） | 插入 | 1 |
| C（Correct） | 正确匹配 | 0 |

编辑距离是完成转换所需要的最少编辑次数：

$$
distance=S+D+I
$$

## 2. DP 状态定义

设 Reference 长度为 m，Hypothesis 长度为 n。

定义：

$$
dp[i][j]
$$

表示将 Reference 的前 i 个 Token 转换为 Hypothesis 的前 j 个 Token 所需要的最少编辑次数。

最终：

$$
EditDistance=dp[m][n]
$$

## 3. DP 初始化

矩阵形状：

$$
(m+1)\times(n+1)
$$

第一行：

$$
dp[0][j]=j
$$

含义：将空 Reference 转换为 j 个 Hypothesis Token，需要插入 j 次。

第一列：

$$
dp[i][0]=i
$$

含义：将 i 个 Reference Token 转换为空序列，需要删除 i 次。

初始化代码：

```python
dp = [
    [0] * (n + 1)
    for _ in range(m + 1)
]

for i in range(m + 1):
    dp[i][0] = i

for j in range(n + 1):
    dp[0][j] = j
```

## 4. 状态转移方程

替换 / 匹配：

$$
dp[i][j]=dp[i-1][j-1]+cost
$$

其中：

$$
cost=
\begin{cases}
0 & ref[i-1]=hyp[j-1]\\
1 & ref[i-1]\ne hyp[j-1]
\end{cases}
$$

删除：

$$
dp[i][j]=dp[i-1][j]+1
$$

从 Reference 前 i-1 个 Token 转换到前 i 个 Token，新增的 Reference Token 被删除。

插入：

$$
dp[i][j]=dp[i][j-1]+1
$$

从 Hypothesis 前 j-1 个 Token 转换到前 j 个 Token，新增的 Hypothesis Token 被插入。

完整递推公式：

$$
dp[i][j]=\min
\begin{cases}
dp[i-1][j-1]+cost\\
dp[i-1][j]+1\\
dp[i][j-1]+1
\end{cases}
$$

## 5. Backtrace 回溯

DP 矩阵右下角 dp[m][n] 能够得到最小编辑距离，但不能直接提供完整的编辑操作序列。

Backtrace 从右下角沿最优路径反向回溯，得到具体的 S/D/I/C 操作。

| 回溯方向 | 编辑操作 | 对应 Token |
|---|---|---|
| 对角线，相同 | C | ref[i-1] 与 hyp[j-1] |
| 对角线，不同 | S | ref[i-1] 与 hyp[j-1] |
| 向上 | D | ref[i-1] |
| 向左 | I | hyp[j-1] |

因为 Backtrace 从右下角向左上角进行，所以得到的是反向操作序列，需要 reverse() 才能得到正向操作序列。

当存在多条最优路径时，可能得到不同的 S/D/I 组合，但编辑距离相同。

## 6. 完整计算示例

Reference：

```python
["I", "like", "apples"]
```

Hypothesis：

```python
["I", "love", "apples", "today"]
```

DP 矩阵：

| | 空 | I | love | apples | today |
|---|---:|---:|---:|---:|---:|
| 空 | 0 | 1 | 2 | 3 | 4 |
| I | 1 | 0 | 1 | 2 | 3 |
| like | 2 | 1 | 1 | 2 | 3 |
| apples | 3 | 2 | 2 | 1 | 2 |

回溯过程：

1. dp[3][4] → dp[3][3]：插入 today。
2. dp[3][3] → dp[2][2]：匹配 apples。
3. dp[2][2] → dp[1][1]：替换 like → love。
4. dp[1][1] → dp[0][0]：匹配 I。

最终正向编辑操作：

```text
C: I → I
S: like → love
C: apples → apples
I: None → today
```

统计：

$$
S=1,\quad D=0,\quad I=1,\quad C=2
$$

编辑距离：

$$
distance=S+D+I=2
$$

WER：

$$
WER=\frac{2}{3}=66.67\%
$$

## 7. 与 CER / MER 的关系

编辑距离算法本身不关心 Token 的具体类型，只处理 Token 序列。

CER、MER 和 WER 的区别在于 Tokenization：

| 指标 | Tokenization |
|---|---|
| CER | 按字符划分 |
| MER | 中文字、英文词、连续数字串等混合 Token |
| WER | 按英文单词划分 |

三者都使用相同的编辑距离计算方法：

$$
ER=\frac{S+D+I}{N}
$$

其中 N 为 Reference 的 Token 数量。

因此可以共用 score_tokens() 和 score_corpus()，但因为 Tokenization 不同，最终错误率可能不同。

## 8. 本模块总结

编辑距离的核心是动态规划：

1. 定义 dp[i][j]。
2. 初始化第一行和第一列。
3. 使用替换/匹配、删除、插入三个方向进行状态转移。
4. 取 dp[m][n] 得到最小编辑距离。
5. 通过 Backtrace 获取具体编辑操作。
6. 统计 S/D/I/C，计算 CER、MER 或 WER。

DP 负责计算最小编辑距离，Backtrace 负责恢复一条最优编辑路径。