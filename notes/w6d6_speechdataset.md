# W6D6 模块1：SpeechDataset

## 1. Dataset 的职责

Dataset 的作用：Dataset将manifest中的元数据与实际音频文件关联起来，按照索引读取单条音频，并将其转换为后续模型可以使用的Tensor。

`__len__()` 的作用：返回数据集的样本数量；

`__getitem__(index)` 根据索引读取指定样本。

## 2. 单条样本字段

`audio`：重采样后的单声道一维波形 Tensor，形状为 (T,)。

`length`：重采样后波形的实际采样点数，等于 audio.numel()。

`uid`：当前使用 manifest 中的 parent_uid 追踪音频来源。

`text`：manifest 中已有的原始转写文本。本模块不负责文本归一化。

## 3. 音频读取

为什么需要统一采样率：音频采样率决定单位时间内包含多少个采样点。

为什么 length 必须在重采样后计算：对于时长固定的音频，采样率变化会使采样点数相应变化。

为什么 Dataset 暂时不负责 padding：因为 padding 的长度取决于同一批次中最长的音频，应当在 collate 阶段统一处理。

## 4. 实验结果

Dataset 样本总数：2281

第一条样本的 UID：common_voice_zh-CN_27897382。

第一条样本的音频形状：(101376,)

第一条样本的采样点数：101376

第一条样本的文本：大厦的主要租户包括美林集团及美国运通大型金融机构。

## 5. 当前理解

Dataset 和 manifest.csv 分别负责什么：
Dataset负责读取单条样本；manifest负责保存样本元数据及其数据划分信息

为什么 Dataset 返回原始长度：
为了后续正确进行可变长batch、padding mask构造，以及与模型输入相关的处理。


# W6D6 模块2：Collate 与可变长 Batch

## 1. 为什么需要 Collate

为什么不同长度的音频不能直接 torch.stack：不同音频的采样点数量不同，直接使用 torch.stack() 无法组成统一的 batch，因为 torch.stack() 要求参与堆叠的 Tensor 形状一致。

Padding 的作用：通过 Padding 将当前 batch 中较短的音频补零到最长音频的长度。

Padding 后的音频 batch 形状：(B, T_max)

## 2. 三个核心变量

`audio`：保存 Padding 后的音频，形状为 (B, T_max)。

`lengths`：保存每条音频 Padding 前的真实长度，形状为 (B,)，不能将其替换为 Padding 后的长度。

`padding_mask`：形状为 (B, T_max) 的布尔矩阵，True 表示 Padding 位置，False 表示真实音频位置。

## 3. Padding mask 的构造

`positions.unsqueeze(0)` 的形状：(1, T_max)。

`lengths.unsqueeze(1)` 的形状： (B, 1)。

为什么两者可以通过广播进行比较：从右往左比较，维度均有1

## 4. 实验结果

短音频 UID：common_voice_zh-CN_28374955

长音频 UID：common_voice_zh-CN_28333336

短音频长度：35136

长音频长度：105408

Batch shape：(2, 105408)

Lengths：[35136, 105408]

Padding mask shape：(2, 105408)

每条音频的 padding 数量：[70272, 0]

验收结果：所有断言检查通过，Collate 全部验收通过。

## 5. 当前理解

Dataset 与 collate_fn 的职责区别：
Dataset负责读取单条音频并保留真实长度；collate_fn负责将多条不同长度的音频组成batch

为什么只在当前 batch 内进行 padding：
减少不必要的填充

为什么 padding_mask 与原始长度都需要保留：
lengths 保存原始长度，padding_mask 标记具体的无效位置。两者共同保证后续处理能够区分真实音频和人工填充区域。


# W6D6 模块3：Padding 与 Loss 屏蔽

## 1. 两种 Padding Mask

音频 padding_mask 的作用：作用于音频输入，形状为 (B, T_audio)，用于标记音频中人工填充的无效位置。

文本 label_padding_mask 的作用：作用于文本标签，形状为 (B, T_text)，用于标记文本 Token 序列中的填充位置。

两者为什么不能混用：因为它们对应不同的数据对象，时间维度也可能不同。

## 2. ignore_index

ignore_index=-100 的作用：ASR 模型训练时，需要逐个预测文本 Token，并根据模型输出 logits 与真实标签计算交叉熵损失。将 Padding 标签替换为 -100，通过 ignore_index=-100 忽略这些位置。

为什么不能直接将 PAD Token 当作普通标签计算 Loss：因为PAD Token没有实际信息

被 ignore 的位置是否参与 Loss 计算：被忽略的位置不参与有效 Loss 的计算及其归一化。

## 3. 数据流

音频 Dataset 与 collate_fn 负责什么：
Dataset 负责读取单条音频，统一采样率，并返回音频、真实长度、UID 和文本。
collate_fn 负责将不同长度的音频组成 batch，通过 Padding 补齐音频，并生成音频 padding_mask。

Tokenizer 负责什么：
Tokenizer 将原始文本转换为模型可以处理的 Token ID 序列。

文本标签 Padding 后如何处理：
当不同文本的 Token 数量不同时，需要通过 Padding 将它们补齐到相同长度。
随后根据真实文本长度构造 label_padding_mask，并将 Padding 位置的标签替换为 -100。

CrossEntropyLoss 如何忽略无效标签：
将处理后的标签传入 CrossEntropyLoss，并设置：ignore_index=-100
这样，真实标签为 -100 的位置不会参与有效 Loss 计算，模型只根据真实文本 Token 计算交叉熵损失。

## 4. 当前理解

为什么 audio_padding_mask 不能直接用于文本 Loss：
因为 audio_padding_mask 对应音频输入，而 CrossEntropyLoss 比较的是模型预测的文本 Token 与真实文本标签。

为什么保留原始音频长度仍然有意义：
原始音频长度可以区分真实音频和人工 Padding，避免将填充区域误认为有效输入。
同时，原始长度还可以用于后续特征提取、有效帧数计算以及模型输入处理。


## 5. 与 Whisper 的联系

Whisper 的文本标签为什么也需要处理 Padding：
Whisper 使用 Encoder-Decoder 架构进行语音识别。
训练时，不同语音对应的文本 Token 数量可能不同。为了组成统一的 batch，需要对文本标签进行 Padding。如果不屏蔽这些 Padding 位置，模型就可能被要求学习预测无效的 PAD Token。

官方 Processor 与模型训练代码分别负责什么：
WhisperProcessor 由 Feature Extractor 和 Tokenizer 组成。

- Feature Extractor：负责将原始音频转换为模型要求的音频特征，例如 Log-Mel。
- Tokenizer：负责将文本转换为 Token ID。

模型训练的数据处理流程还需要负责组织 batch、处理标签 Padding、构造 -100 忽略标签，并将模型输出与真实标签传入 Loss 函数。
因此，官方 Processor 负责与模型配置匹配的音频和文本预处理，但不能简单地将它等同于完整的训练数据管线。