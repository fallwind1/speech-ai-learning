
# W10D3 模块1：术语词表与术语原子化

## 1. 为什么需要 term-WER

term-WER 的作用：
term-WER 是项目自定义的术语转写错误率，用于衡量ASR模型对领域术语的识别情况。它将完整术语作为一个独立计分单位，专门统计术语的替换、删除和插入。

与 CER / MER 的区别：
- CER：以 Unicode 字符为计分单位，衡量整体字符识别错误。
- MER：以中文字符、英文词、数字串等混合 token 为计分单位，衡量中英混合文本的整体识别错误。
- term-WER：以完整术语原子为计分单位，衡量领域术语的转写错误。

为什么不能直接统计 Hypothesis 中是否包含术语：
因为这种方法无法正确处理术语的重复出现、重叠术语、术语替换、删除和插入，也无法区分完整术语与部分匹配。

因此，需要先将术语原子化，再进行全句对齐，统一统计各类错误。

## 2. 术语词表 terms.csv

术语词表的来源：
从 Train、Dev 和事先确定的领域术语清单建立，不允许根据 Test 错误倒推词表或提示。

canonical_id 的作用：
作为术语的唯一标识，使同一个术语的不同允许书写形式能够映射到同一个术语原子。

允许别名（aliases）的作用：
记录一个术语允许接受的不同书写形式。经过统一规范化后，允许别名映射到同一个 canonical_id。
别名不能随意扩展，不能将语义不同、仅仅相似或近音的词自动合并。

为什么词表需要在 W10 冻结：
确保后续评测使用固定的术语集合和评分规则，使实验结果具有可复现性和可比较性。

为什么不能根据 Test 错误扩充词表：
因为 Test 用于最终评估。如果根据 Test 的识别错误修改术语词表，就相当于利用测试集信息调整评分规则，造成测试信息泄漏。

## 3. 术语原子化（Term Atomicization）

术语原子化的定义：
将术语词表中匹配到的完整术语映射为一个不可拆分的原子单位，普通 token 则继续保留。

TERM:id 表示什么：
表示一个术语的唯一原子标识。例如 TERM_002 可以代表 beam search。

多词术语如何处理：
将完整的多词术语映射为一个原子，而不是将组成术语的多个词分别计分。

重复术语如何处理：
保留每一次出现，不得使用 set 去重。

重叠术语如何处理：
采用从左到右最长匹配（Longest Match）。如果当前位置存在多个重叠候选，优先选择 token 序列最长的术语；如果长度相同，则按照固定词表顺序选择。

英文整词边界的作用：
防止子字符串误匹配。例如 transform 不能匹配 transformer。

## 4. 术语原子化示例

词表：

- TERM_001：beam
- TERM_002：beam search

原始文本：

beam search algorithm

Token 序列：

["beam", "search", "algorithm"]

原子化之后：

["TERM_002", "algorithm"]

原因：

beam search 与 beam 在第一个位置发生重叠，按照最长匹配规则选择 beam search，映射为 TERM_002。

## 5. 模块验收

Q1：为什么 term-WER 需要先进行术语原子化？
为了将完整术语作为一个独立计分单位，避免多词术语被拆分统计，同时正确处理重叠术语和重复术语。


Q2：如何处理重叠术语？
采用从左到右最长匹配，优先选择当前位置能够匹配到的最长术语，等长时按照固定词表顺序决定。

Q3：为什么不能使用 set 存储原子化结果？
因为 set 会自动去重，导致重复出现的术语丢失，进而影响术语分母 N_t 和错误统计。

Q4：为什么不能根据 Test 错误扩充词表？
因为这会利用测试集信息修改评测规则，造成测试信息泄漏，破坏最终评估的客观性。


# W10D3 模块2：全句对齐与术语错误统计

## 1. 为什么需要全句对齐

**全句对齐的作用：**

对术语原子化之后的完整 Reference 和 Hypothesis 序列进行 Levenshtein 对齐，确定每个单位之间的匹配、替换、删除和插入关系，从而准确统计术语错误。

**为什么不能只检查 Hypothesis 是否包含术语：**

仅检查 Hypothesis 是否包含 Reference 中的术语，无法判断术语究竟发生了替换、删除还是插入，也无法发现 Hypothesis 中额外产生的术语。

例如：

Reference：

["TERM_001"]

Hypothesis：

["TERM_001", "TERM_002"]

虽然 Hypothesis 包含正确的 TERM_001，但是还额外出现了 TERM_002，必须统计一次术语插入。

因此，必须对完整句子进行对齐，而不能仅使用字符串包含判断。

## 2. S_t / D_t / I_t / N_t 定义

**N_t：**

Reference 中术语出现的总次数。按照实际出现次数统计，重复术语不能去重。

**S_t：**

Reference 中的术语与不同单位对齐时产生的术语替换次数，包括术语被其他术语替换，以及术语被普通 token 替换。

**D_t：**

Reference 中的术语在 Hypothesis 中没有对应单位，被删除的次数。

**I_t：**

Hypothesis 中额外出现的术语次数，包括：

1. Hypothesis 中额外插入术语。
2. Reference 中的普通单位被 Hypothesis 中的术语替换。

**term-WER 公式：**

$$
term\text{-}WER=\frac{S_t+D_t+I_t}{N_t}
$$

当 $N_t=0$ 时，term-WER 返回 N/A，但仍然保留术语插入次数 $I_t$。

## 3. 三种错误情况

**术语替换：**

Reference 中的术语被其他单位替换，计入 $S_t$。

例如：

TERM_001 → TERM_002

$$
S_t=1
$$

**术语删除：**

Reference 中存在术语，但 Hypothesis 没有对应输出，计入 $D_t$。

例如：

TERM_001 → 空

$$
D_t=1
$$

**术语插入：**

Hypothesis 中出现了 Reference 中不存在的额外术语，计入 $I_t$。

例如：

空 → TERM_002

$$
I_t=1
$$

普通 token 被术语替换时，也计入 $I_t$。

例如：

study → TERM_002

虽然普通 Levenshtein 将其视为 substitution，但按照项目的 term-WER 分类规则，它属于术语插入。

**术语 A 被术语 B 替换时如何统计：**

只统计一次 $S_t$，不再额外统计 $I_t$。

原因是同一次对齐操作已经被识别为术语替换，不能重复计算错误。

## 4. 特殊情况

**Reference 中没有术语：**

$$
N_t=0
$$

如果 Hypothesis 也没有术语插入，则：

$$
I_t=0
$$

term-WER 返回 N/A。

**Hypothesis 中出现额外术语：**

即使 Reference 中没有任何术语，只要 Hypothesis 额外出现了词表术语，就必须统计插入次数。

例如：

$$
N_t=0,\quad I_t=1
$$

此时 term-WER 仍然为 N/A，但保留 $I_t=1$。

**N_t = 0 时如何处理：**

不能将分母人为设置为 1，也不能返回 0。

应该返回 N/A，同时保留 $S_t$、$D_t$、$I_t$、$N_t$ 的原始统计结果。

## 5. 对齐规则

**Levenshtein 对齐：**

对术语原子化之后的完整 Reference 和 Hypothesis 序列进行动态规划对齐。

基本操作：

- Match：相同单位匹配，代价 0。
- Substitution：不同单位替换，代价 1。
- Deletion：删除，代价 1。
- Insertion：插入，代价 1。

**平局优先级：**

当存在多条代价相同的最优对齐路径时，采用固定优先级：

1. Match
2. Substitution
3. Deletion
4. Insertion

**为什么需要保存 alignment：**

因为最终不仅需要编辑距离，还需要知道具体发生了哪些操作，才能正确区分术语替换、删除和插入。

保存对齐结果可以帮助检查错误归因、复现实验结果以及分析具体的术语识别错误。

## 6. 模块验收

**Q1：术语 A 被术语 B 替换时，应该统计什么？**

统计一次 $S_t$，不额外统计 $I_t$，避免同一次错误重复计数。

**Q2：Reference 和 Hypothesis 分别为：**

Reference：

["we", "use", "TERM_001", "today"]

Hypothesis：

["we", "use", "TERM_001", "TERM_002", "today"]

统计：

$$
N_t=1,\quad S_t=0,\quad D_t=0,\quad I_t=1
$$

$$
term\text{-}WER=100\%
$$

**Q3：普通 Reference token 被 Hypothesis 术语替换，计入什么？**

计入 $I_t$。因为 Reference 中原本没有术语，而 Hypothesis 额外输出了一个术语。

**Q4：Reference 没有术语，但是 Hypothesis 出现额外术语，如何处理？**

$$
N_t=0,\quad I_t=1
$$

term-WER 返回 N/A，同时保留插入次数。不能将分母人为设置为 1。


# W10D3 模块3.1：term_eval.py 代码结构

## 1. 文件职责

`term_eval.py` 的主要作用是实现项目自定义的 term-WER 评估。

它负责将普通文本转换为术语原子序列，对完整 Reference 和 Hypothesis 进行 Levenshtein 对齐，并根据对齐结果统计术语替换、删除和插入次数。

**与 normalize.py 的关系：**

`normalize.py` 负责文本规范化与中英混合 Tokenization。

term_eval.py 直接复用 `normalize_score()` 和 `mixed_tokens()`，确保 Reference 和 Hypothesis 使用相同的评分规则。

**与 metrics.py 的关系：**

`metrics.py` 已经实现 Levenshtein 动态规划、Backtrace 和 S/D/I/C 统计。

term_eval.py 不需要重复实现这些功能，而是复用 `score_tokens()` 获取完整的 alignment，再增加术语错误分类逻辑。

## 2. 函数设计

**load_terms()**

读取 terms.csv，获取术语的 canonical_id、canonical 和允许别名。

将术语及别名转换为 token 序列，建立术语匹配结构，供后续原子化使用。

**atomicize()**

将输入的普通 Token 序列中的完整术语替换为 `TERM:id` 原子。

采用从左到右最长匹配；重叠时选择最长术语，等长时按照固定词表顺序处理。

保留所有普通 Token 和重复术语，不能使用 set 去重。

**align_sequences()**

对原子化后的 Reference 和 Hypothesis 进行全句 Levenshtein 对齐。

复用 `metrics.py` 中的 `score_tokens()`，返回编辑距离、S/D/I/C 和具体 alignment。

**compute_term_stats()**

读取 alignment，根据 Reference 和 Hypothesis 对应位置是否为术语，分别统计：

- S_t：术语替换。
- D_t：术语删除。
- I_t：术语插入。
- N_t：Reference 中术语出现总次数。

最终计算 term-WER，并保留完整 alignment。

## 3. 数据流

**输入：**

原始 Reference 和 Hypothesis 文本，以及冻结的术语词表。

**处理过程：**

1. 使用 `mixed_tokens()` 对 Reference 和 Hypothesis 分词。
2. 使用 `atomicize()` 将完整术语转换为 `TERM:id`。
3. 使用 `align_sequences()` 对两个完整序列进行 Levenshtein 对齐。
4. 使用 `compute_term_stats()` 根据 alignment 统计术语错误。

**输出：**

术语错误统计结果，包括 S_t、D_t、I_t、N_t、term-WER 和 alignment。

## 4. 返回结果

建议返回：

```python
{
    "S_t": 1,
    "D_t": 0,
    "I_t": 1,
    "N_t": 3,
    "term_wer": 0.6667,
    "alignment": [...]
}
```

其中：

- S_t：术语替换次数。
- D_t：术语删除次数。
- I_t：术语插入次数。
- N_t：Reference 术语总出现次数。
- term_wer：术语错误率。
- alignment：完整的编辑对齐结果。

当 N_t = 0 时，term_wer 返回 None，表示 N/A，但仍然保留 I_t 等统计结果。

**语料级统计：**

不能直接平均每条音频的 term-WER。

应该先汇总全部音频的 S_t、D_t、I_t、N_t，再计算：

$$
term\text{-}WER_{corpus}
=
\frac{
S_{total}+D_{total}+I_{total}
}{
N_{total}
}
$$

如果 N_total = 0，则返回 N/A。

## 5. 模块验收

**Q1：为什么可以复用 score_tokens()？**

因为术语原子化之后，术语与普通 Token 都是序列中的离散单位，Levenshtein 算法不关心它们的具体语义，可以直接进行编辑距离计算。

**Q2：study → TERM:002 应该计入什么？**

计入 I_t。因为 Reference 中原本是普通 Token，而 Hypothesis 额外出现了一个术语。

**Q3：重复术语如何统计？**

Reference：

```python
["TERM:001", "TERM:001", "today"]
```

Hypothesis：

```python
["TERM:001", "today"]
```

统计：

$$
N_t=2,\quad D_t=1
$$

不能使用 set 去重。

**Q4：为什么不能直接平均单句 term-WER？**

因为每条音频包含的 Reference 术语数量不同，直接平均会让所有音频具有相同权重。

正确方式是分别累计所有样本的 S_t、D_t、I_t、N_t，再计算语料级 term-WER。

另外，即使某条音频 N_t = 0，其术语插入错误仍然需要参与语料级分子统计。


# W10D3 模块3.2：术语词表加载与原子化

## 1. terms.csv 的设计

术语词表用于保存标准术语及其允许别名，作为术语原子化的依据。

主要字段：

- `canonical_id`：术语唯一标识。
- `canonical`：术语标准形式。
- `aliases`：允许的其他书写形式。

建议使用 JSON 数组保存 aliases，方便 Python 解析。

例如：

```csv
canonical_id,canonical,aliases
001,attention,"[""attention"",""attn""]"
002,beam search,"[""beam search"",""beam-search""]"
```

注意：CSV 的具体序列化格式属于项目实现设计，05 §6.1 要求的是 canonical_id 和允许别名，并未强制规定 aliases 的存储格式。

**词表加载时需要检查的问题：**

1. canonical_id 是否唯一。
2. canonical 是否为空。
3. aliases 是否能够正确解析。
4. 别名是否为空。
5. 不同 canonical_id 是否存在相同的匹配形式。
6. canonical 是否也被纳入匹配形式。

不同 canonical_id 不能共享同一个别名，否则会产生术语映射歧义。

词表从 Train、Dev 和预先确定的领域术语清单建立，W10 冻结，不允许根据 Test 错误反向扩充。

## 2. load_terms()

**函数作用：**

读取 terms.csv，将标准术语及其允许别名转换为统一的 Token 序列，建立后续术语匹配所需的数据结构。

**输入：**

terms.csv 文件路径。

**输出：**

包含 canonical_id、canonical、aliases_tokenized 等信息的术语结构。

例如：

```python
{
    "canonical_id": "002",
    "canonical": "beam search",
    "aliases_tokenized": [
        ["beam", "search"],
        ["beam-search"]
    ]
}
```

**为什么需要将 aliases 转换为 Token 序列：**

因为后续 atomicize() 需要比较输入 Token 序列与术语别名序列。

提前使用 mixed_tokens() 转换可以避免每次执行 atomicize() 时重复进行规范化和分词，同时保证所有样本使用统一的词表处理规则。

## 3. atomicize()

**函数作用：**

将普通 Token 序列中的完整术语替换为 `TERM:id` 原子。

**输入：**

普通 Token 序列，以及 load_terms() 返回的词表结构。

**输出：**

术语原子化后的完整 Token 序列。

例如：

输入：

```python
[
    "we",
    "use",
    "beam",
    "search",
    "today"
]
```

输出：

```python
[
    "we",
    "use",
    "TERM:002",
    "today"
]
```

**最长匹配的实现思路：**

从左到右扫描输入 Token，在当前位置寻找所有能够匹配的术语别名，选择匹配长度最长的一个。

匹配成功后输出对应的 `TERM:id`，并跳过已经消耗的 Token；没有匹配时，保留原 Token。

如果出现等长匹配，按照固定词表顺序选择。

**如何处理英文整词边界：**

使用 mixed_tokens() 进行分词后，以完整 Token 序列匹配，而不是使用字符串子串匹配。

例如：

`transform` 不应该匹配 `transformer`。

**如何处理重复术语：**

重复出现的术语分别转换为术语原子，保留原有出现次数。

不能使用 set 去重，因为这会改变 Reference 中的术语总次数 N_t。

**如何处理重叠术语：**

遵循从左到右最长匹配原则。

例如：

- TERM:001 = beam
- TERM:002 = beam search

输入 `beam search algorithm` 时，优先匹配较长的 `beam search`。

## 4. 代码设计

**load_terms()：**

1. 使用 pandas 读取 CSV。
2. 检查必要字段。
3. 检查 canonical_id 唯一性。
4. 使用 json.loads() 解析 aliases。
5. 使用 mixed_tokens() 转换 canonical 和 aliases。
6. 检查不同 canonical_id 之间的匹配形式冲突。
7. 返回预处理后的词表结构。

**atomicize()：**

1. 从左到右遍历输入 Token。
2. 在当前位置寻找所有可能的术语匹配。
3. 选择最长匹配。
4. 匹配成功时输出 TERM:id，并跳过已消耗 Token。
5. 没有匹配时保留原 Token。
6. 返回完整原子化序列。

## 5. 模块验收

**Q1：为什么提前将 aliases 转换为 Token 序列？**

避免每次执行 atomicize() 时重复分词，提高效率，并保证统一的匹配规则。

**Q2：为什么选择 TERM:002？**

因为 beam search 的匹配长度为 2，大于 beam 的长度 1，符合从左到右最长匹配原则。

**Q3：为什么不同 canonical_id 不能拥有相同别名？**

因为相同别名无法唯一确定术语身份，会导致原子化结果不确定，影响后续术语错误统计。

**Q4：为什么不能使用 set？**

因为 set 会自动去重，导致重复术语丢失，影响 Reference 中术语出现总次数 N_t，以及后续的术语错误率计算。


# W10D3 模块3.3：全句对齐与术语错误分类

## 1. Alignment 的作用

术语原子化之后，需要对 Reference 和 Hypothesis 进行完整句子的 Levenshtein 对齐，才能识别术语具体发生了什么错误。

不能仅判断 Hypothesis 是否包含 Reference 中的术语，因为即使正确术语出现了，也可能同时存在术语插入或其他术语错误。

项目复用 `src/metrics.py` 中已有的 `score_tokens()`，不需要重新实现动态规划。

`score_tokens()` 返回的 `operations` 是一个编辑操作列表，每个元素为：

```python
(operation, reference_token, hypothesis_token)
```

四种操作：

- C：Correct，正确匹配。
- S：Substitution，替换。
- D：Deletion，删除。
- I：Insertion，插入。

例如：

```python
[
    ("C", "we", "we"),
    ("S", "TERM:001", "TERM:002"),
    ("C", "today", "today")
]
```

表示两个普通词正确匹配，中间发生了一次术语替换。

## 2. 术语错误分类

term-WER 的 S_t、D_t、I_t 不能直接使用普通 Token 级别的 S、D、I，需要根据 Reference 和 Hypothesis 的 Token 类型重新分类。

定义术语判断函数：

```python
def is_term(token):
    return (
        isinstance(token, str)
        and token.startswith("TERM:")
    )
```

### S_t：术语替换

Reference 中存在术语，但是 Hypothesis 对应位置不是相同的术语。

包括：

- TERM:A → TERM:B
- TERM:A → 普通 Token

两种情况均计入一次 S_t。

### D_t：术语删除

Reference 中存在术语，但是 Hypothesis 没有对应输出。

例如：

```text
TERM:001 → None
```

计入一次 D_t。

### I_t：术语插入

包括两种情况：

1. Reference 中不存在术语，但是 Hypothesis 插入了术语。
2. Reference 中的普通 Token 被 Hypothesis 中的术语替换。

例如：

```text
None → TERM:001
today → TERM:002
```

两种情况均计入一次 I_t。

特别注意：

TERM:A → TERM:B 只计算一次 S_t，不再重复计算 I_t。

### N_t：Reference 中的术语总次数

统计原子化后 Reference 中所有 TERM 的出现次数。

例如：

```python
[
    "we",
    "TERM:001",
    "TERM:001",
    "today"
]
```

则：

\[
N_t=2
\]

重复术语必须保留重复次数，不能使用 set 去重。

### 完整分类表

| operation | Reference | Hypothesis | 术语统计 |
|---|---|---|---|
| C | TERM | 相同 TERM | 不计 |
| C | 普通 | 相同普通 | 不计 |
| S | TERM | TERM | S_t + 1 |
| S | TERM | 普通 | S_t + 1 |
| S | 普通 | TERM | I_t + 1 |
| S | 普通 | 普通 | 不计 |
| D | TERM | None | D_t + 1 |
| D | 普通 | None | 不计 |
| I | None | TERM | I_t + 1 |
| I | None | 普通 | 不计 |

## 3. compute_term_stats()

函数作用：

根据完整句子的 alignment，识别术语替换、删除和插入，计算术语错误统计量。

输入：

- Reference 原子序列。
- Hypothesis 原子序列。
- 完整句子的 operations。

输出：

```python
{
    "S_t": 0,
    "D_t": 0,
    "I_t": 0,
    "N_t": 0,
    "term_wer": None,
    "operations": []
}
```

如何判断一个 Token 是否为术语：

通过 `is_term(token)` 判断是否以 `TERM:` 开头。

如何遍历 operations：

```python
for op, ref_token, hyp_token in operations:
    ...
```

根据 operation 类型及两个 Token 的术语身份，按照分类表更新 S_t、D_t、I_t。

其中 N_t 单独通过遍历 Reference 原子序列计算：

```python
N_t = sum(
    is_term(token)
    for token in reference_atoms
)
```

术语错误率：

\[
term\text{-}WER=
\frac{S_t+D_t+I_t}{N_t}
\]

当 N_t = 0 时：

```python
term_wer = None
```

但仍然保留 I_t，不能丢弃术语插入记录。

## 4. 边界情况

### Reference 没有术语

如果 N_t = 0，则 term-WER 为 N/A。

如果 Hypothesis 中出现额外术语，仍然计入 I_t。

### Hypothesis 为空

Reference 中的所有术语均被删除：

\[
D_t=N_t
\]

因此：

\[
term\text{-}WER=100\%
\]

前提是 N_t > 0。

### 普通 Token 发生替换

普通 Token → 普通 Token 不计入术语错误。

例如：

```text
today → tomorrow
```

只影响整体 MER，不影响 term-WER。

### 重复术语

重复出现的术语分别参与统计。

例如：

```text
Reference:
TERM:001 TERM:001

Hypothesis:
TERM:001
```

结果：

\[
N_t=2,\quad D_t=1
\]

\[
term\text{-}WER=50\%
\]

## 5. 语料级统计

不能直接对每条音频的 term-WER 求算术平均，因为不同音频的 Reference 术语数量不同。

应该先累计：

\[
S_{total}=\sum_i S_{t,i}
\]

\[
D_{total}=\sum_i D_{t,i}
\]

\[
I_{total}=\sum_i I_{t,i}
\]

\[
N_{total}=\sum_i N_{t,i}
\]

然后计算：

\[
term\text{-}WER=
\frac{S_{total}+D_{total}+I_{total}}
{N_{total}}
\]

当 N_total = 0 时，结果为 N/A，但仍然报告 I_total。

即使某条音频没有 Reference 术语，只要它产生了术语插入，也必须计入语料级 I_total。

## 6. 模块验收

Q1：为什么不能直接使用 score_tokens() 的 S、D、I？

因为普通 Token 级别的编辑操作与术语错误分类不同。例如普通 Token → 术语在 Levenshtein 中属于 S，但在 term-WER 中属于 I_t。

Q2：给定 operations：

- TERM:001 → TERM:002
- study → TERM:003
- TERM:004 → None
- None → TERM:005
- None → today

结果：

\[
S_t=1,\quad D_t=1,\quad I_t=2,\quad N_t=2
\]

term-WER = 200%。

Q3：为什么两种 S 的统计方式不同？

TERM → 普通 Token 属于术语替换 S_t；普通 Token → TERM 属于术语插入 I_t。

Q4：两个术语全部删除：

\[
S_t=0,\quad D_t=2,\quad I_t=0,\quad N_t=2
\]

term-WER = 100%。

Q5：为什么语料级先累计再计算？

因为应该按 Reference 术语出现总次数加权，而不是让每条音频具有相同权重。N_t=0 的音频不计算单条错误率，但其术语插入仍计入语料级分子。