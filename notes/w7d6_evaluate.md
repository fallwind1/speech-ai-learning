# W7D6 模块3.1：ASR Evaluation Runner 设计

## 1. Evaluation Runner 的作用

之前已经分别实现了音频处理、特征提取、数据集构建、ASR 推理和指标计算等模块，但这些模块目前相对独立。

Evaluation Runner 的作用是将这些模块连接起来，形成完整的 ASR 批量评估流程。

其主要任务包括：

1. 读取指定数据集中的音频及人工参考文本。
2. 加载 ASR 模型，对音频逐条进行推理。
3. 保存每条音频对应的 UID、Reference 和 Hypothesis。
4. 对预测文本和参考文本进行统一规范化。
5. 计算 CER、MER 等评价指标。
6. 保存逐条预测结果、语料级指标和实验元数据。

因此，Evaluation Runner 本身不负责重新实现 Whisper 推理或编辑距离算法，而是调用之前已有的模块。

## 2. Evaluation Runner 的整体流程

完整流程：

```text
Dev Candidate CSV
       |
       v
读取音频路径、UID、人工参考文本
       |
       v
load_whisper_model()
       |
       v
加载 Whisper 模型（只加载一次）
       |
       v
循环读取音频
       |
       v
transcribe_file()
       |
       v
获得 Hypothesis
       |
       v
组织 UID / Reference / Hypothesis
       |
       v
normalize_score()
       |
       v
CER / MER / English WER
       |
       v
score_corpus()
       |
       v
保存实验结果和元数据
```

这里需要注意：模型只需要加载一次，后续对不同音频反复调用推理函数即可。

如果每处理一条音频就重新加载模型，会造成不必要的时间和显存开销。

## 3. 评估数据的组织方式

Evaluation Runner 需要为每条音频建立统一的评估记录。

例如：

```python
{
    "uid": "cv26_xxxxxxxxxxxxxxxx",
    "reference": "人工核对后的文本",
    "hypothesis": "Whisper生成的文本",
    "score_reference": "规范化后的参考文本",
    "score_hypothesis": "规范化后的预测文本"
}
```

各字段的含义：

| 字段 | 含义 |
|---|---|
| uid | 音频样本的唯一标识 |
| reference | 人工核对后的原始参考文本 |
| hypothesis | 模型生成的原始预测文本 |
| score_reference | 规范化后的参考文本 |
| score_hypothesis | 规范化后的预测文本 |

**Reference 和 Hypothesis 必须区分原始文本与评分文本。**

Reference 应当使用人工核对后的 `raw_text`，不能使用未经人工验证的 Manifest 原始 `text` 或候选队列中的 `source_text`。

评分时，对参考文本和预测文本使用相同的 `normalize_score()` 规则。

例如：

```python
score_reference = normalize_score(reference)
score_hypothesis = normalize_score(hypothesis)
```

这样可以避免由于大小写、Unicode 表示或约定的标点处理差异造成不必要的评分错误。

## 4. CER、MER 与 English WER 的计算

三种指标本质上都使用编辑距离算法，区别主要在于 Tokenization 的方式。

**CER：字符错误率**

按照字符划分 Reference 和 Hypothesis：

```python
ref_tokens = list(score_reference)
hyp_tokens = list(score_hypothesis)
```

计算公式：

$$
CER=\frac{S_c+D_c+I_c}{N_c}
$$

其中：

- S：替换次数。
- D：删除次数。
- I：插入次数。
- N：Reference 字符数。

实际实现还应遵循已经确定的 CER 空白处理规则。

**MER：中英混合错误率**

使用之前实现的 `mixed_tokens()`：

```python
ref_tokens = mixed_tokens(score_reference)
hyp_tokens = mixed_tokens(score_hypothesis)
```

其 Token 包括：

- 中文单字。
- 英文单词。
- 数字串。
- 约定保留的技术术语及特殊符号。

计算公式：

$$
MER=\frac{S_m+D_m+I_m}{N_m}
$$

**English WER：英文词错误率**

在独立的英文评估子集上使用统一的英文分词规则。

$$
WER=\frac{S_w+D_w+I_w}{N_w}
$$

三种指标都可以复用之前的 `score_tokens()` 和 `score_corpus()`，不需要重新实现编辑距离算法。

语料级错误率必须使用：

$$
ER_{\text{corpus}}
=
\frac{\sum_i(S_i+D_i+I_i)}
{\sum_i N_i}
$$

不能直接对每条音频的错误率取算术平均。

## 5. 实验元数据（Metadata）

仅保存 CER、MER 等指标是不够的，因为模型、推理配置和评分规则发生变化，都可能影响实验结果。

因此，每次 Evaluation 都需要记录实验元数据。

| 字段 | 含义 |
|---|---|
| split | 使用的数据集，例如 dev |
| model_name | 模型名称，例如 openai/whisper-small |
| model_revision | 模型权重对应的具体版本 |
| language | 推理语言，例如 zh |
| task | transcribe 或 translate |
| max_new_tokens | 最大生成 Token 数量 |
| normalizer_version | 文本规范化版本 |
| scoring_version | 评分规则版本 |
| num_samples | 实际参与评估的样本数 |
| device | CPU 或 CUDA |

其中：

**model_revision 和 scoring_version 是两个不同的概念。**

- model_revision：记录使用了哪一版模型权重。
- scoring_version：记录使用了哪一版评分规则。

例如，即使 Whisper 模型完全没有变化，只要修改了文本规范化或分词规则，最终 CER、MER 也可能发生变化。

因此，两者都必须记录。

## 6. 实验结果的保存

Evaluation Runner 至少需要保存两类结果。

**逐条预测结果（CSV）**

例如：

| uid | reference | hypothesis |
|---|---|---|
| sample_001 | 今天天气很好 | 今天天气很好 |
| sample_002 | 我喜欢学习 | 我喜欢学 |
| sample_003 | 语音识别模型 | 语音识别模型 |

保存逐条结果的意义：

- 可以定位具体识别错误。
- 可以检查某条音频的 Reference 和 Hypothesis。
- 可以比较不同模型在同一条音频上的表现。
- 可以在不重新推理的情况下，按照新的评分规则重新计算指标。

**实验配置及指标（JSON）**

保存：

- 模型名称和版本。
- 数据集 split。
- 推理参数。
- 评分版本。
- 样本数量。
- CER、MER、English WER 等语料级指标。

这样可以保证实验具有可追溯性和可复现性。

## 7. Dev 与 Test 的使用边界

Train、Dev、Test 的作用不同：

| 数据集 | 主要用途 |
|---|---|
| Train | 模型训练 |
| Dev | 模型选择、Prompt 调整、推理参数优化 |
| Test | 最终模型性能评估 |

当前阶段使用 Dev 运行 Evaluation Runner，验证整个评估流程。

可以根据 Dev 的评估结果调整 Prompt、推理参数和模型方案。

但不能根据 Test 的参考答案调整 Prompt、词表或纠错规则。

最终模型方案确定后，再使用 Test 进行独立评估。

## 8. 本模块需要实现的文件

```text
speech-ai-learning/
│
├── src/
│   ├── asr_inference.py
│   ├── metrics.py
│   ├── normalize.py
│   └── evaluate.py       # 本模块新建
│
└── results/
    └── w07d6/
        ├── predictions.csv
        └── metadata.json
```

`evaluate.py` 负责组织评估流程，调用已有模块，不重复实现已有功能。

本模块的最终验收标准：

1. 能够读取 Dev 候选数据。
2. 模型只加载一次。
3. 能够批量生成 Hypothesis。
4. 每条结果包含 UID、Reference、Hypothesis。
5. 能够正确计算语料级 CER、MER。
6. 能够保存逐条预测结果。
7. Metadata 包含 split、model_revision、normalizer_version 和 scoring_version。
8. 能够在不修改 Test 数据的情况下完成 Dev 评估。


# W7D6 模块3.2：CER / MER Tokenization

## 1. 为什么需要 Tokenization？

ASR 评估需要比较 Reference 和 Hypothesis 两个文本序列。

但不同指标采用的计分单位不同：

- CER：Unicode 字符。
- MER：中文单字、英文单词、数字串及其他有效符号。
- English WER：英文单词。

因此，同一段文本需要根据不同指标采用不同的 Tokenization 方式。

## 2. CER Tokenization

CER 首先使用 normalize_score() 进行统一规范化，然后去除计分空白，最后按 Unicode 字符划分。

$$
CER=\frac{S_c+D_c+I_c}{N_c}
$$

实现：

```python
def cer_tokens(text):
    normalized = normalize_score(text)

    return [
        char
        for char in normalized
        if not char.isspace()
    ]
```

例如：

```text
输入：我喜欢 Python

输出：
['我', '喜', '欢', 'p', 'y',
 't', 'h', 'o', 'n']
```

注意：英文单词在 CER 中仍然按字符计算。

## 3. MER Tokenization

MER 使用 mixed_tokens()，按照中英混合规则划分 Token。

$$
MER=\frac{S_m+D_m+I_m}{N_m}
$$

实现：

```python
def mer_tokens(text):
    return mixed_tokens(text)
```

例如：

```text
输入：我在2026年学习Python

输出：
['我', '在', '2026', '年', '学', '习', 'python']
```

其中：

- 中文逐字划分。
- 英文按词划分。
- 连续数字作为一个 Token。
- 其他有效符号按照 fallback 规则保留。

mixed_tokens() 内部已经调用 normalize_score()，因此不需要重复规范化。

## 4. CER 与 MER 的区别

| 对比 | CER | MER |
|---|---|---|
| 中文 | 单字 | 单字 |
| 英文 | 单字符 | 单词 |
| 数字 | 单字符 | 连续数字串 |
| 空白 | 去除计分空白 | 不作为 Token |
| 特殊符号 | 按字符处理 | 按约定 fallback 处理 |

因此，同一段文本在 CER 和 MER 下的 Token 数量可能不同，最终错误率也可能不同。

## 5. 实现原则

CER 和 MER 共用同一套 normalize_score() 规范化规则，但采用不同的 Tokenization 方法。

这样既保证 Reference 和 Hypothesis 采用一致的文本处理方式，又能体现不同评价指标的计分粒度。


# W7D6 模块3.3：Dev 数据读取与有效样本筛选

## 1. 为什么需要筛选有效样本？

人工标注数据中可能存在不同状态：

- verified：人工校对通过。
- needs_review：需要进一步复核。
- pending：尚未完成校对。

正式评估只能使用 verified 样本。

不能直接使用 Manifest 中未经人工验证的原始文本作为 Reference，也不能将 needs_review 和 pending 样本纳入正式指标计算。

## 2. 评估数据需要哪些字段？

| 字段 | 含义 |
|---|---|
| uid | 音频唯一标识 |
| audio_path | 音频路径 |
| reference | 人工核对后的原始文本 |
| score_reference | 规范化后的参考文本 |
| status | 标注状态 |

其中 Reference 使用人工核对后的 raw_text，而不是 source_text。

## 3. 数据读取流程

```text
读取 Candidate CSV
       ↓
筛选 status == verified
       ↓
检查必要字段
       ↓
检查音频路径
       ↓
检查 Reference 和评分文本
       ↓
统一字段名称
       ↓
得到 Eval DataFrame
```

## 4. 为什么检查 score_text？

score_text 是根据 raw_text 使用 normalize_score() 生成的。

读取时重新执行规范化并与保存结果比较，可以发现数据保存或评分文本处理过程中出现的不一致。

这样能够避免 Reference 在不同实验中被意外修改。

## 5. 当前实验数据

本次 Dev 候选队列：

| 状态 | 数量 |
|---|---:|
| verified | 10 |
| needs_review | 1 |
| pending | 398 |

因此当前正式评估样本数为10。

后续随着 Dev 标注数量增加，Evaluation Runner 可以自动读取全部 verified 样本，不需要手动修改数量。


# W7D6 模块3.4：Whisper 批量推理

## 1. 为什么需要批量推理？

之前的 transcribe_file() 只能对单条音频进行推理。

为了评估整个数据集，需要循环处理多条音频，并将每条音频的识别结果统一保存。

批量推理的核心是复用已经加载的模型，而不是重复加载模型。

## 2. 正确的执行方式

```text
load_whisper_model()
       |
       v
加载 Processor 和 Model
       |
       v
循环遍历评估数据
       |
       v
transcribe_file()
       |
       v
获得 Hypothesis
       |
       v
保存每条音频的预测结果
```

模型只加载一次，循环内部只执行推理。

这样可以避免重复加载模型权重造成的时间和显存开销。

## 3. 批量推理函数的职责

run_batch_inference() 主要完成：

1. 遍历评估数据。
2. 调用 transcribe_file()。
3. 获取模型预测文本 Hypothesis。
4. 使用 normalize_score() 规范化 Hypothesis。
5. 保存 UID、Reference、Hypothesis。
6. 返回统一的 DataFrame。

它不负责计算 CER、MER，也不负责保存最终实验结果。

## 4. 预测结果的数据结构

每条音频保存以下字段：

| 字段 | 含义 |
|---|---|
| uid | 音频唯一标识 |
| audio_path | 音频路径 |
| reference | 人工核对的原始文本 |
| hypothesis | 模型生成的原始文本 |
| score_reference | 规范化后的参考文本 |
| score_hypothesis | 规范化后的预测文本 |
| duration_sec | 音频时长 |

其中 Reference 和 Hypothesis 保存原始文本，score_reference 和 score_hypothesis 用于后续指标计算。

## 5. 为什么推理与评分分离？

推理负责生成预测文本，评分负责比较预测文本与人工参考文本。

二者分离后：

- 可以使用同一批预测结果测试不同评分规则。
- 可以更换模型而不修改指标计算代码。
- 可以保存预测结果，避免每次计算指标都重新运行模型。

因此，Evaluation Runner 应当将数据读取、模型推理、文本处理、指标计算和结果保存设计为相对独立的步骤。

## 6. 本模块验收标准

- 模型只加载一次。
- 能够遍历全部有效评估样本。
- 每条音频都能得到 Hypothesis。
- 每条预测结果都能关联正确 UID。
- Reference 和 Hypothesis 同时保存原始文本及评分文本。
- 推理结果数量与输入样本数量一致。


# W7D6 模块3.5：CER / MER 语料级评估

## 1. 评估的输入

完成 ASR 推理后，每条音频具有：

- Reference：人工核对的参考文本。
- Hypothesis：模型生成的预测文本。
- score_reference：规范化后的参考文本。
- score_hypothesis：规范化后的预测文本。

指标计算使用规范化后的文本。

## 2. 构建 Token Pairs

score_corpus() 接收的输入为：

```python
[
    (reference_tokens, hypothesis_tokens),
    ...
]
```

CER 使用字符级 Tokenization：

```python
cer_pairs = [
    (
        cer_tokens(row.score_reference),
        cer_tokens(row.score_hypothesis)
    )
    for row in predictions.itertuples()
]
```

MER 使用中英混合 Tokenization：

```python
mer_pairs = [
    (
        mer_tokens(row.score_reference),
        mer_tokens(row.score_hypothesis)
    )
    for row in predictions.itertuples()
]
```

两种指标使用相同的编辑距离算法，但 Tokenization 不同。

## 3. 语料级指标计算

调用 score_corpus()：

```python
cer_result = score_corpus(cer_pairs)
mer_result = score_corpus(mer_pairs)
```

语料级错误率：

$$
ER_{\text{corpus}}
=
\frac{\sum_i(S_i+D_i+I_i)}
{\sum_i N_i}
$$

其中：

- S：替换次数。
- D：删除次数。
- I：插入次数。
- N：参考 Token 总数。

不能使用单句错误率的算术平均代替语料级错误率。

## 4. 为什么还需要保存逐条指标？

除了整个语料的 CER、MER，还应保存每条音频的错误统计。

例如：

| uid | cer_error_rate | mer_error_rate |
|---|---:|---:|
| sample_001 | 0.10 | 0.08 |
| sample_002 | 0.25 | 0.20 |

这样可以定位哪些音频识别效果较差，为后续错误分析提供依据。

## 5. 评价指标的解释

CER 和 MER 都是错误率，越低越好。

CER 以字符为计分单位，MER 以中英混合 Token 为计分单位。

由于两者的 Tokenization 不同，同一模型在同一数据集上得到的 CER 和 MER 通常不会相同。

## 6. 实验注意事项

1. Reference 和 Hypothesis 必须使用相同的规范化规则。
2. Reference 必须来自人工核对文本。
3. 语料级指标使用总编辑次数除以总参考 Token 数。
4. 逐条结果应保留 UID，便于错误定位。
5. 修改规范化规则后，应更新 normalizer_version，并重新评估。


# W7D6 模块3.6：实验结果保存与 Metadata

## 1. 为什么需要保存实验结果？

ASR 实验不仅需要得到 CER、MER，还需要保存产生这些指标的具体预测结果和实验配置。

如果只保存最终错误率，后续就无法知道哪些音频识别错误，也无法确认两次实验是否使用了相同的模型和评分规则。

因此需要分别保存逐条预测结果和实验元数据。

## 2. Predictions.csv

逐条保存每条音频的评估结果，包括：

| 字段 | 含义 |
|---|---|
| uid | 音频唯一标识 |
| audio_path | 音频路径 |
| reference | 人工参考文本 |
| hypothesis | 模型预测文本 |
| score_reference | 规范化后的参考文本 |
| score_hypothesis | 规范化后的预测文本 |
| cer_S/D/I/N | CER 编辑统计 |
| mer_S/D/I/N | MER 编辑统计 |

保存逐条结果后，可以进行错误分析，也可以在不重新推理的情况下重新计算指标。

## 3. Metadata.json

Metadata 用于记录实验配置和评估结果。

主要包括：

- 数据集 split。
- 模型名称与模型 revision。
- 推理语言和任务。
- 最大生成 Token 数量。
- 运行设备。
- 规范化版本。
- 评分版本。
- 参与评估的样本数。
- CER、MER 等语料级指标。

## 4. 为什么需要记录模型版本和评分版本？

实验指标不仅受模型影响，也受评分规则影响。

例如：

- 修改模型权重，Hypothesis 可能发生变化。
- 修改 Prompt，Hypothesis 可能发生变化。
- 修改规范化规则，Reference 和 Hypothesis 的评分文本可能发生变化。
- 修改 Tokenization，编辑距离和分母可能发生变化。

因此，模型版本和评分版本需要分别记录。

## 5. 实验可复现性

比较两次 ASR 实验时，需要尽量保持以下条件一致：

1. 使用相同的音频样本。
2. 使用相同的人工 Reference。
3. 使用相同的规范化规则。
4. 使用相同的 Tokenization 规则。
5. 使用相同的语料级聚合方式。

在这些条件一致的情况下，才能更可靠地比较不同模型或推理配置的表现。

## 6. 本模块验收标准

- 成功保存 predictions.csv。
- 成功保存 metadata.json。
- CSV 包含 UID、Reference 和 Hypothesis。
- Metadata 包含 split、model_revision、normalizer_version 和 scoring_version。
- 能够从保存结果中重新读取逐条预测和语料级指标。


# W7D6：ASR Evaluation Pipeline

## 1. 实验目标

本日主要完成 ASR Evaluation Pipeline 的初步实现，使用 Whisper-small 对 Dev 数据进行批量推理，并计算 CER、MER。

主要任务：

1. 完成人工标注，累计60条 verified。
2. 读取 Dev 中已经 verified 的样本。
3. 使用 Whisper-small 批量推理。
4. 实现 CER、MER 语料级计算。
5. 保存逐条预测结果和实验 Metadata。

本次只使用 Dev 数据进行评估，不使用 Test 结果调整模型或 Prompt。

## 2. 评估数据读取

### 2.1 为什么需要筛选 verified？

标注数据包含不同状态：

- verified：人工校对通过。
- needs_review：需要进一步复核。
- pending：尚未完成校对。

正式评估只使用 verified 样本。

不能直接使用 Manifest 中未经人工核对的 source_text 作为 Reference。

### 2.2 评估字段

| 字段 | 含义 |
|---|---|
| uid | 音频唯一标识 |
| audio_path | 音频路径 |
| reference | 人工核对后的原始文本 |
| score_reference | 规范化后的参考文本 |
| status | 标注状态 |

其中 reference 使用 raw_text，score_reference 使用 score_text。

读取时还需要检查：

- 必要字段是否存在。
- 字段是否缺失。
- 音频路径是否有效。
- score_text 是否与 normalize_score(raw_text) 一致。
- UID 是否能够唯一标识样本。

### 2.3 数据筛选流程

```text
读取 Candidate CSV
       ↓
筛选 status == verified
       ↓
检查必要字段
       ↓
检查音频路径
       ↓
验证 score_text
       ↓
统一字段名称
       ↓
得到 Eval DataFrame
```

本次得到10条有效 Dev 样本。

## 3. Whisper 批量推理

### 3.1 单条推理与批量推理

之前实现的 transcribe_file() 负责单条音频推理。

批量推理函数 run_batch_inference() 则负责遍历评估数据，并多次调用 transcribe_file()。

正确的执行方式是：

```text
load_whisper_model()
       ↓
加载 Processor 和 Model
       ↓
遍历评估数据
       ↓
transcribe_file()
       ↓
获得 Hypothesis
       ↓
保存预测结果
```

模型只加载一次，循环中复用相同的 model 和 processor。

这样可以避免重复加载模型权重，减少时间和显存开销。

### 3.2 Whisper 推理过程

单条音频的推理流程：

```text
Audio File
    ↓
load_audio()
    ↓
to_mono()
    ↓
resample_audio()
    ↓
Waveform
    ↓
WhisperProcessor
    ↓
Log-Mel Features
    ↓
Whisper Encoder
    ↓
Whisper Decoder
    ↓
Generated Token IDs
    ↓
Tokenizer Decode
    ↓
Hypothesis
```

本次音频存在不同采样率，包括32kHz和48kHz。

transcribe_file() 根据 Processor 要求的采样率进行重采样，最终输入模型。

本次模型配置：

| 参数 | 值 |
|---|---|
| Checkpoint | openai/whisper-small |
| Language | zh |
| Task | transcribe |
| max_new_tokens | 96 |
| Device | cuda |

### 3.3 预测结果字段

每条音频保存：

| 字段 | 含义 |
|---|---|
| uid | 唯一样本标识 |
| audio_path | 音频路径 |
| reference | 原始参考文本 |
| hypothesis | 原始预测文本 |
| score_reference | 规范化参考文本 |
| score_hypothesis | 规范化预测文本 |
| duration_sec | 音频时长 |

推理和评分分离，可以在不重新运行模型的情况下更换评分规则或重新计算指标。

## 4. CER / MER Tokenization

### 4.1 CER

CER 使用字符作为计分单位。

首先通过 normalize_score() 进行统一规范化，然后去除计分空白，最后按 Unicode 字符划分。

```python
def cer_tokens(text):
    normalized = normalize_score(text)

    return [
        char
        for char in normalized
        if not char.isspace()
    ]
```

CER：

$$
CER=\frac{S+D+I}{N}
$$

其中 N 为参考字符总数。

### 4.2 MER

MER 使用中英混合 Tokenization。

- 中文：逐字划分。
- 英文：按词划分。
- 数字：按连续数字串划分。
- 其他有效符号：按照 fallback 规则保留。

```python
def mer_tokens(text):
    return mixed_tokens(text)
```

MER：

$$
MER=\frac{S+D+I}{N}
$$

其中 N 为参考混合 Token 总数。

CER 和 MER 共用 normalize_score()，但 Tokenization 不同。

## 5. 语料级指标计算

### 5.1 构建 Token Pairs

score_corpus() 接收：

```python
[
    (reference_tokens, hypothesis_tokens),
    ...
]
```

CER 和 MER 分别构建自己的 Token Pairs。

### 5.2 为什么使用语料级聚合？

单句错误率的简单平均不等于语料级错误率。

正确计算方式：

$$
ER_{\text{corpus}}
=
\frac{\sum_i(S_i+D_i+I_i)}
{\sum_iN_i}
$$

必须先累加全部编辑次数和参考 Token 数量，再计算最终错误率。

### 5.3 保存逐条指标

除了语料级指标，还保存每条音频的：

- S、D、I、C。
- N。
- distance。
- error_rate。

这些信息可以用于定位识别错误较多的音频。

## 6. 实验结果保存

### 6.1 Predictions.csv

保存逐条预测结果，包括：

- UID。
- Audio Path。
- Reference。
- Hypothesis。
- Score Reference。
- Score Hypothesis。
- CER 统计。
- MER 统计。

### 6.2 Metadata.json

记录实验配置：

| 字段 | 含义 |
|---|---|
| created_at | 实验时间 |
| split | 数据集划分 |
| model_name | 模型名称 |
| model_revision | 模型实际 commit SHA |
| language | 识别语言 |
| task | 推理任务 |
| max_new_tokens | 最大生成 Token 数 |
| device | 运行设备 |
| normalizer_version | 规范化版本 |
| scoring_version | 评分版本 |
| num_samples | 评估样本数 |
| metrics | 语料级指标 |

### 6.3 为什么需要记录模型版本？

只记录模型名称不足以保证实验可复现。

同一个模型仓库可能随着时间更新，因此需要记录具体 commit SHA。

本次 Whisper-small：

```text
973afd24965f72e36ca33b3055d56a652f456b4d
```

以后比较不同实验时，应同时考虑模型版本、数据划分和评分版本。

## 7. 本次实验结果

实验配置：

| 项目 | 结果 |
|---|---|
| Model | Whisper-small |
| Split | Dev |
| Samples | 10 |
| Device | CUDA |
| Normalizer | w7d4-v1-draft |
| Scoring | w7d6-v1 |

语料级指标：

| 指标 | 数值 |
|---|---:|
| CER | 0.33136 |
| MER | 0.33136 |
| 替换 S | 54 |
| 删除 D | 2 |
| 插入 I | 0 |
| 参考 Token N | 169 |
| 总编辑次数 | 56 |

计算：

$$
CER=MER=\frac{54+2+0}{169}
\approx 0.33136
$$

本次 CER 和 MER 相同，是因为当前10条样本的计分单位主要为中文字符，两种 Tokenization 在这些样本上得到相同的序列。

注意：本次只有10条 Dev 样本，结果主要用于验证评估流程，不应作为完整 Dev 集性能结论。

## 8. 实验结果存储位置

```text
results/
└── w07d6/
    └── whisper_small_baseline/
        ├── predictions.csv
        └── metadata.json
```

predictions.csv 保存逐条预测和评分结果。

metadata.json 保存实验配置和语料级指标。

## 9. 本日核心收获

1. 理解了单条 ASR 推理与批量 ASR 推理的区别。
2. 实现了复用同一模型的批量推理流程。
3. 将推理、Tokenization、指标计算和结果保存分离。
4. 使用 score_corpus() 实现语料级 CER、MER。
5. 理解了为什么实验必须同时保存预测结果和 Metadata。
6. 建立了可用于后续模型对比的 Baseline 评估流程。

## 10. 后续改进方向

- 扩大 Dev 人工验证样本数量。
- 检查繁简体差异对 CER、MER 的影响，但不修改当前冻结的评分规则。
- 处理 Transformers 的 generation_config 参数警告。
- 完善模型 revision 固定机制。
- 后续在统一评估框架下比较不同模型和推理配置。