
# W8D6：英文 CTC 对照实验

## 1. 实验目标与设计

### 1.1 实验目的
- 核心目标：
- 为什么选择英文数据：
- 为什么使用相同的20条音频：
- 本次实验不能得出的因果结论：

### 1.2 对照模型

| 项目 | wav2vec2-base-960h | whisper-small |
|---|---|---|
| Checkpoint | | |
| 模型架构 | | |
| 输入特征 | | |
| 解码方式 | | |
| 训练范式 | | |
| 语言适用范围 | | |

## 2. 英文 Dev 数据准备

### 2.1 数据来源
- 数据集：
- Split：
- 样本数量：
- 采样规则与随机种子：
- 是否固定并保存样本 UID：
- Reference 来源：

### 2.2 数据检查
- audio_path：
- uid：
- reference：
- 采样率：
- 音频时长：
- 音频文件存在性：
- 样本唯一性：

## 3. wav2vec 2.0 CTC 推理

### 3.1 模型结构
- CNN Feature Encoder：
- Transformer Context Network：
- Linear CTC Head：

### 3.2 推理流程
- 输入：
- Processor：
- Logits：
- Greedy CTC Decoder：
- 输出：

### 3.3 实验结果
- 成功推理数量：
- 发现的问题：

## 4. Whisper-small 推理

### 4.1 模型结构
- Encoder：
- Decoder：
- 与 CTC 的区别：

### 4.2 推理流程
- 输入：
- Log-Mel：
- generate()：
- 语言与任务设置：
- 输出：

### 4.3 实验结果
- 成功推理数量：
- 发现的问题：

## 5. 统一 WER 评分

### 5.1 评分规则
- Normalization：
- Tokenization：
- WER 公式：
- 为什么使用语料级 WER：

### 5.2 实验结果

| 模型 | 样本数 | S | D | I | N | WER |
|---|---:|---:|---:|---:|---:|---:|
| wav2vec2-base-960h | | | | | | |
| whisper-small | | | | | | |

- 输出文件：
- 两模型使用相同 Reference 的检查：
- CSV 字段：
- 模型版本与评分版本：

## 6. 结果分析与实验结论

### 6.1 模型预测差异
- 典型错误样本：
- 主要错误类型：
- 可能影响结果的因素：

### 6.2 实验结论
- 本次观察到的 WER：
- 是否能判断 CTC 优于 Encoder-Decoder：
- 英文 Checkpoint 为什么不参加中文混读主排名：
- 20条样本的局限性：

## 7. 本日总结

- 完成的任务：
- 代码与文件产出：
- 理解的知识：
- 尚存问题：
- 最终验收结果：


# W8D6 模块4：wav2vec2 批量推理

## 1. 批量推理的目的

单条推理是对一条音频执行完整的 ASR 流程；批量推理则是自动处理多条音频，并保存所有预测结果。

本次实验使用循环逐条处理 20 条音频，每次输入模型的 Batch Size 为 1。

模型加载与模型推理分离：
- 模型加载：只执行一次，获得 Processor 和 Model。
- 模型推理：循环调用同一个模型对象，不重复加载权重。

这样可以减少重复读取模型权重和 GPU 参数传输带来的时间开销。

## 2. 批量推理流程

1. 读取固定的英文音频清单 metadata。
2. 遍历每条音频的 audio_path。
3. 调用单条推理函数 transcribe_wav2vec2_file()。
4. 执行音频预处理、模型推理和 CTC Greedy Decoding。
5. 获取 Hypothesis 和推理元数据。
6. 将每条音频的 Reference、Hypothesis 等信息保存到列表。
7. 转换为 DataFrame 并保存 CSV。

## 3. 结果保存

输出文件：

results/w08_wav2vec2_english_predictions.csv

主要字段：

| 字段 | 含义 |
|---|---|
| id | 音频 UID |
| speaker_id | 说话人 ID |
| reference | 人工转录文本 |
| hypothesis | 模型预测文本 |
| audio_path | 音频路径 |
| checkpoint | 模型 Checkpoint |
| device | 推理设备 |
| sampling_rate | 模型输入采样率 |
| duration_sec | 音频时长 |
| input_length | 输入采样点数量 |
| logits_time_steps | Logits 时间步数量 |
| vocab_size | CTC 输出类别数量 |

## 4. 结果检查

本次实验结果：

- 样本数量：20。
- UID 全部唯一。
- Reference 空值数量：0。
- Hypothesis 空值数量：0。
- 空预测数量：0。
- Checkpoint：facebook/wav2vec2-base-960h。
- Device：CUDA。

前五条样本展示的 Hypothesis 均与 Reference 完全一致。

## 5. 实验注意事项

- 不重新加载已经加载的模型，直接复用模型对象。
- Reference 保持原始数据集提供的文本，不随意修改。
- Hypothesis 保留模型原始预测结果，不提前进行规范化。
- 两个模型必须使用相同的 20 条音频和 Reference。
- 保存 UID，保证后续预测结果可以正确配对。
- 本次循环逐条推理，不等同于真正的多样本 Batch Inference。

## 6. 本模块总结

本模块将已经验证的单条 wav2vec2 推理函数应用于 20 条英文音频，实现了批量预测和结果保存。

最终获得逐条 Reference 和 Hypothesis，为后续 Whisper 对照及统一 WER 评估提供数据基础。



# W8D6 模块5：Whisper-small 英文推理

## 1. Whisper 模型基本信息

Checkpoint：openai/whisper-small

模型架构：
Transformer Encoder-Decoder。

输入特征：
Log-Mel Spectrogram，而非直接将原始 Waveform 输入 Transformer。

Encoder 的作用：
对音频特征进行编码，通过 Self-Attention 建模语音中的上下文关系，输出音频隐藏表示。

Decoder 的作用：
通过 Causal Self-Attention 利用已生成的文本 Token，通过 Cross-Attention 利用 Encoder 的音频表示，自回归生成新的文本 Token。

训练范式：
大规模弱监督语音到文本序列学习。

语言设置：
language="en"，明确指定英文。

任务设置：
task="transcribe"，执行语音转录而不是翻译。

## 2. 模型加载

load_whisper_model() 的作用：
加载指定 Checkpoint 对应的 Processor 和 Model，设置运行设备，并将模型切换到评估模式。

WhisperProcessor：
包含 Feature Extractor 和 Tokenizer。
前者将 Waveform 转换为 Log-Mel 特征，后者处理文本 Token 的编码和解码。

WhisperForConditionalGeneration：
包含 Whisper Encoder-Decoder，支持使用 model.generate() 进行自回归文本生成。

模型运行设备：
由 load_whisper_model() 自动选择 CUDA 或 CPU，具体设备以运行时 Metadata 为准。

为什么使用 model.eval()：
将模型切换到评估模式，使 Dropout 等模块按推理模式运行。

## 3. Whisper 音频预处理

输入音频：
固定的 LibriSpeech dev-clean 20 条英文 WAV 音频。

原始采样率：
16000 Hz。

单声道转换：
调用已有 to_mono()，将音频统一为单声道。

重采样：
调用已有 resample_audio()。本次输入已为 16000 Hz，无须改变采样率。

Log-Mel 特征：
WhisperProcessor 的 Feature Extractor 将 Waveform 转换为 Log-Mel Spectrogram。

input_features shape：
通常为 (1, 80, 3000)，具体以本次 Metadata 为准。

WhisperProcessor 与 Wav2Vec2Processor 的区别：

- WhisperProcessor：将 Waveform 转换为 Log-Mel 特征。
- Wav2Vec2Processor：对 Waveform 进行归一化和格式整理，不计算 Log-Mel。

## 4. Whisper 推理

model.generate() 的作用：
调用 Whisper Decoder 自回归生成文本 Token ID。

language="en" 的作用：
指定英文识别任务的语言条件，避免沿用之前中文实验的默认设置。

task="transcribe" 的作用：
将语音转录为原语言文本，而非翻译任务。

max_new_tokens：
本次设置为 96，控制新生成 Token 的最大数量。

generated_ids：
Decoder 自回归生成的 Token ID 序列，可能包含语言、任务及结束等特殊 Token。

最终 Transcript：
使用 Tokenizer 将 Generated IDs 解码为英文文本。

## 5. 与 wav2vec2 CTC 的区别

| 对比维度 | wav2vec2 | Whisper |
|---|---|---|
| 架构 | CNN + Transformer Encoder + CTC Head | Transformer Encoder-Decoder |
| 输入 | 原始 Waveform | Log-Mel 特征 |
| 输出方式 | 逐时间步类别预测 | 自回归生成文本 Token |
| 解码方式 | Argmax + CTC Collapse | generate() + Tokenizer Decode |
| 推理接口 | Wav2Vec2ForCTC | WhisperForConditionalGeneration |

核心区别：

wav2vec2 通过帧级分类建立音频特征与输出类别的关系，再使用 CTC Collapse 得到文本。

Whisper 通过 Encoder 提取音频表示，由 Decoder 根据音频表示和此前生成的 Token 依次生成文本。

## 6. 批量推理结果

样本数量：20。

Checkpoint：openai/whisper-small。

语言：en。

任务：transcribe。

成功推理数量：20。

UID 重复数量：0。

Reference 空值数量：0。

Hypothesis 空值数量：0。

空预测数量：0。

Reference：
来自固定的 LibriSpeech dev-clean 英文样本。

Hypothesis：
Whisper-small 生成的英文转录文本。

结果观察：
前五条样本的预测文本与 Reference 在语义和词序上高度一致，主要可见大小写和标点差异。

输出文件：
results/w08_whisper_english_predictions.csv

## 7. 本模块总结

1. 复用了 W7 已实现的 Whisper 推理接口，无须重新编写音频预处理和自回归解码逻辑。
2. 明确指定 language="en" 和 task="transcribe"，完成英文 ASR 任务。
3. 在固定的20条英文音频上完成推理，未出现空预测或重复 UID。
4. 理解了 CTC 帧级预测与 Whisper 自回归 Token 生成的区别。
5. 后续必须统一 Reference 和 Hypothesis 的 Normalization、Tokenization，再计算 WER。
6. 本实验仅用于固定样本的模型推理与评测流程验证，不能据此证明两种模型架构的因果优劣。



# W8D6 模块6：统一 WER 评估与模型对照分析

## 1. 实验目标

本模块在固定的20条LibriSpeech英文dev音频上，对wav2vec2-base-960h和Whisper-small进行统一WER评估。

实验目的：

1. 使用相同的英文音频和Reference进行模型对照。
2. 复用已有的文本规范化和编辑距离算法。
3. 计算两个模型的语料级WER。
4. 分析模型的识别错误及实验局限性。

数据集：LibriSpeech dev-clean。

样本数量：20。

采样率：16000 Hz。

模型：
- facebook/wav2vec2-base-960h
- openai/whisper-small

## 2. 结果配对

通过UID对两个模型的预测结果进行一对一合并。

检查内容：
- UID唯一。
- 样本数量一致。
- Reference完全一致。
- Audio Path完全一致。
- Hypothesis不存在空值。

使用UID而不是直接按照CSV行号进行配对，可以避免不同文件排序不一致造成的样本错配。

## 3. 文本规范化与分词

使用已有的normalize_score()统一处理Reference和Hypothesis。

规范化规则：
- Unicode NFKC规范化。
- 英文统一转换为小写。
- 普通标点替换为空格。
- 非数字之间的句点替换为空格。
- 折叠连续空白。

英文Tokenization：

normalized_text.split()

例如：

Reference:
HOW WOULD THE PAPERS TALK ABOUT IT

Hypothesis:
How would the papers talk about it?

规范化后：

how would the papers talk about it

两个文本得到相同的Token序列。

两个模型必须使用相同的Normalization和Tokenization规则，避免格式差异影响WER。

## 4. WER计算

WER（Word Error Rate）是词错误率。

计算公式：

WER = (S + D + I) / N

其中：
- S：Substitution，替换次数。
- D：Deletion，删除次数。
- I：Insertion，插入次数。
- N：Reference中的总词数。

复用已有接口：
- normalize_score()
- score_tokens()
- score_corpus()

语料级WER：

WER_corpus = (ΣS + ΣD + ΣI) / ΣN

不能直接计算所有单句WER的算术平均值。

## 5. 实验结果

共同条件：
- 数据集：LibriSpeech dev-clean
- 样本数量：20
- Reference总词数：321
- Speaker数量：1
- Chapter数量：1

### wav2vec2-base-960h

S = 6
D = 0
I = 0
N = 321

WER = 6 / 321 ≈ 1.87%

### Whisper-small

S = 9
D = 3
I = 0
N = 321

WER = 12 / 321 ≈ 3.74%

### 结果比较

当前固定样本中：
- wav2vec2 WER为1.87%。
- Whisper WER为3.74%。
- wav2vec2的WER约为Whisper的一半。

当前样本中，Whisper出现了更多替换和删除错误。

## 6. 错误分析

wav2vec2：
- 6次替换。
- 没有删除和插入。

Whisper：
- 9次替换。
- 3次删除。
- 没有插入。

两个模型均取得较低的WER，说明当前样本上的英文识别任务完成情况较好。

## 7. 实验局限性

本次实验存在以下局限：

1. 样本数量只有20条，规模较小。
2. 全部音频来自同一个Speaker（2277）。
3. 全部音频来自同一个Chapter（149896）。
4. 使用顺序读取的前20条音频，不是具有代表性的随机抽样。
5. 两个模型在训练数据、架构、参数规模和解码策略方面存在差异。

因此，本次实验只能说明两个模型在当前固定样本上的表现差异。

不能据此证明CTC架构优于Encoder-Decoder，也不能将当前英文实验结果直接推广到其他语音数据集。

## 8. 实验产出

results/
- w08_wav2vec2_english_predictions.csv
- w08_whisper_english_predictions.csv
- w08_english_ctc.csv
- w08_english_ctc_metadata.json

最终完成：
- 两个模型的英文批量推理。
- UID与Reference一致性检查。
- 统一文本规范化。
- 英文Tokenization。
- 语料级WER计算。
- 两个模型的结果对照。
- 实验局限性分析。

## 9. 模块总结

本模块完成了wav2vec2和Whisper在固定英文dev子集上的统一评估。

理解了WER的计算方式、语料级聚合方法，以及文本规范化对公平评估的重要性。

同时认识到，小规模实验只能支持有限范围内的结论，不能将模型表现差异直接归因于模型架构。



### 英文 Dev 数据集的局限性

本次使用 LibriSpeech dev-clean 中顺序读取的前 20 条音频，
没有进行随机抽样。

经检查：
- 样本数量：20
- 不同说话人数：1（Speaker 2277）
- 不同章节数：1（Chapter 149896）
- 音频采样率：16000 Hz

所有样本均来自同一说话人和同一章节，
因此样本之间可能存在相关性，代表性有限。

本次实验主要用于验证不同 ASR 架构的英文推理、
预测文本保存及统一 WER 评估流程。

两个模型使用完全相同的音频和 Reference，
但实验结果仅反映当前固定样本上的识别表现，
不用于推断模型在一般英文 ASR 任务中的性能，
也不能用于证明不同 ASR 架构范式的因果优劣。