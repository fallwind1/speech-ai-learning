# W6D3 模块1：Whisper Processor 配置

## 1. Processor 与 Feature Extractor

WhisperProcessor的作用：
将音频特征提取器与文本Tokenizer组织在一起，提供统一的音频与处理和文本编解码接口。

WhisperFeatureExtractor的作用：
将原始音频波形转换为模型需要的Log-Mel特征，并根据调用参数执行padding、truncation和张量转换等操作。

Tokenizer的作用：
将文本转换为Token IDs，也可以将模型生成的Token IDs解码回文本。

为什么今天不需要下载完整模型权重：
本日只需要读取Processor和模型配置，不进行ASR推理。
因此使用AutoProcessor与WhisperConfig读取配置即可，不需要加载WhisperForConditionalGeneration的模型权重。

## 2. whisper-small 的特征参数

选定checkpoint：
openai/whisper-small

sampling_rate：
16000 Hz。输入波形必须与声明的采样率一致。若原始音频是其他采样率，需要先明确重采样流程。

feature_size：
80，表示每个时间帧的Mel特征维数。

n_fft：
400，表示FFT点数。

hop_length：
160个采样点，在16000 Hz下对应10 ms。

chunk_length：
30秒，是标准音频块的时长。

n_samples：
480000

nb_max_frames：
3000

padding_side / padding_value：
- padding_side = "right"
- padding_value = 0.0

return_attention_mask：
默认配置为False。若需要显式处理填充区域或进行特定批处理，应根据模型和接口要求决定是否返回attention mask。

## 3. 参数间的关系

30秒对应多少采样点：
30 × 16000 = 480000。

每个时间帧的帧移是多少毫秒：
160 / 16000 × 1000 = 10 ms。

3000帧是怎样得到的：
480000个采样点除以160点的帧移，得到3000个时间帧。

feature_size与时间帧数量有什么区别：
feature_size描述频率方向的Mel维数，whisper-small为80。

## 4. Padding与Truncation

padding的含义：
当输入波形短于目标长度时，在波形层面添加填充值，使其达到目标长度。

truncation的含义：
当输入波形超过目标长度时，截掉超出部分。

默认padding策略：
该模型的特征提取器配置为从右侧补零。调用时通常采用padding="max_length"，具体行为仍应以实际调用参数和当前库版本为准。

默认truncation设置：
调用时通常使用truncation=True，将过长的波形截断至目标长度。应通过实际调用验证，而不是仅根据配置文件推断。

为什么短音频也可能输出3000帧：
因为5秒波形可以先补零至30秒，再计算Log-Mel特征。

为什么超长音频存在内容丢失风险：
如果35秒音频直接截断至30秒，最后5秒的内容可能被丢弃，导致语音识别遗漏尾部信息。对于长音频，应根据任务需求采用分块、长音频推理或其他明确的处理策略，而不是默认直接截断。

## 5. Checkpoint差异

whisper-small的Mel维数：
80

whisper-large-v3的Mel维数：
128

为什么不能认为所有Whisper模型都使用80维Mel：
不同checkpoint可以采用不同的特征维数和模型配置。
从whisper-small切换到large-v3时，需要重新读取特征提取器的feature_size，并与模型配置中的num_mel_bins核对。
除了Mel维数，还应检查采样率、FFT参数、帧移、chunk_length以及其他相关配置，避免沿用旧checkpoint的假设。

## 6. 当前理解

- Processor、Feature Extractor、Tokenizer之间是什么关系：
Processor统一管理Feature Extractor与Tokenizer；前者处理音频，后者处理文本。

- 采样率与30秒最大输入长度如何对应：
采样率16000 Hz表示每秒16000个采样点，30秒因此对应480000个采样点。

- 为什么feature_size=80不代表只有80个时间帧：
80是Mel频率维度，时间维度由输入长度和帧移等参数决定。whisper-small的标准30秒输入对应3000个时间帧。

- 为什么不能仅凭输出shape判断输入没有被截断：
不同原始时长的音频可以经过padding或truncation后得到相同的标准输出形状。需要同时记录原始时长、有效采样点数和实际处理策略。


# W6D3 模块2：不同音频长度的 Processor 实验

## 1. 实验设置

Checkpoint：
openai/whisper-small

采样率：
16000Hz

标准输入时长：
30s

标准采样点数：
480000

Padding参数：
padding="max_length"

Truncation参数：
truncation=True

测试音频的生成方式：
使用人工生成的正弦波，不依赖真实语音内容。

- 5秒：440 Hz正弦波。
- 30秒：440 Hz正弦波。
- 35秒：前30秒为440 Hz，后5秒为2000 Hz。

## 2. Processor shape 表

| 原始时长 | 原始采样点数 | 补齐或截断后的有效采样点数 | 输出形状 |
|---|---:|---:|---|
| 5秒 |80000|480000|(1,80,3000)|
| 30秒 |480000|480000|(1,80,3000)|
| 35秒 |560000|480000|(1,80,3000)|

## 3. 实验结果分析

为什么5秒和30秒的输出形状相同：
Whisper-small的标准输入对应30s音频，即480000个采样点。
5s音频原本只有80000个采样点，补齐400000个零后达到标准长度；30s音频本来就有480000个采样点。
两者最终都对应80个Mel频带和3000个时间帧

35秒音频丢失了多少秒：
5s

输出shape能否说明全部原始语音都被保留：
不能。
5秒、30秒和35秒音频均输出(1, 80, 3000)，但分别发生了补零、无须补齐或截断。

## 4. 截断验证

35秒音频的前30秒与30秒测试音频是否相同：
相同，35s测试波形的前30s直接复用了30s测试波形

两者经过Processor后的最大绝对误差：
0.0

这项比较验证了什么：
在本次测试配置下，35秒输入经过截断后，得到的特征与30秒输入完全一致。
这证明了当前Processor调用的截断行为符合预期，并且最后5秒的2000Hz测试信号没有进入最终特征。
需要注意，这是对本次Transformers版本、输入信号和参数组合的实际验证，不能仅凭这一结果推断所有接口、参数或版本都采用完全相同的处理策略。

## 5. 当前理解

- Padding发生在特征提取的哪个阶段：
在本次使用的调用策略下，较短输入被处理至目标长度后，再得到标准长度的特征表示。具体实现细节应以当前版本的Feature Extractor为准。

- Truncation可能造成什么风险：
超出目标长度的语音可能被丢弃，导致词语、句子甚至整段内容无法进入后续识别。

- 为什么长音频不能不加检查地直接截断：
因为原始音频的后续部分可能包含重要内容。直接截断会导致信息丢失，而且标准输出shape不会显示这一问题。
对于长音频，应采用分段识别、适当的重叠区域及时间戳对齐等策略，或者使用适合长音频的推理流程。

- 输出形状与有效语音长度有什么区别：
输出形状描述特征矩阵的维度，而有效语音长度描述实际保留的原始语音内容。两段音频即使都输出(1, 80, 3000)，其有效语音时长也可能不同。


# W6D3 模块3：Checkpoint配置与长音频处理

## 1. Checkpoint配置对照

比较的模型：
- openai/whisper-small
- openai/whisper-large-v3
- openai/whisper-large-v3-turbo

small的feature_size / num_mel_bins：
80/80

large-v3的feature_size / num_mel_bins：
128/128

large-v3-turbo的feature_size / num_mel_bins：
128/128

三个模型共同的采样率：
16000Hz

三个模型共同的n_fft / hop_length：
400/160

三个模型共同的标准输入长度：
30s,480000个采样点

配置对照表保存路径：
results/w06d3_checkpoint_comparison.csv

## 2. 配置差异分析

为什么不能把80维Mel推广到所有Whisper checkpoint：
不同checkpoint可以采用不同的Mel频带数量。
Whisper-small使用80维Mel，而large-v3和large-v3-turbo使用128维Mel。
Mel维数不仅影响输入矩阵形状，还与滤波器组的频带划分和模型训练时的输入定义有关。

feature_size与num_mel_bins分别来自哪里：
feature_size来自Feature Extractor配置，表示输出Mel特征的频率维度。
num_mel_bins来自Whisper模型配置，表示模型预期的Mel频带数量。
使用对应checkpoint时，应检查二者是否一致。

为什么配置相同不意味着模型完全相同：
即使采样率、FFT参数和标准输入长度相同，不同checkpoint仍可能采用不同的网络结构、模型权重、Tokenizer或生成设置。

切换checkpoint时需要重新核对哪些参数：

- sampling_rate
- feature_size
- num_mel_bins
- n_fft
- hop_length
- chunk_length
- n_samples
- nb_max_frames
- padding与truncation设置
- 与模型输入相关的其他配置

## 3. 长音频处理

直接截断30秒的风险：
对于超过30s的语音，直接截断可能导致后续内容无法进入模型，造成识别遗漏。
输出形状仍可能完全正常，因此不能仅根据shape判断内容是否完整。

分块推理的基本流程：

1. 读取完整音频并确认采样率
2. 根据长音频推理策略划分片段
3. 分别提取特征并执行ASR
4. 获取并整理各片段的识别文本及时间戳
5. 处理边界重复和遗漏
6. 按时间顺序合并识别结果

为什么片段之间需要重叠上下文：
重叠区域可以为片段边界附近的语音提供额外上下文，减少切断词语、遗漏内容或重复识别的风险。

return_timestamps的作用：
在支持该参数的推理接口中，可以返回文本片段或词语对应的时间信息，帮助确定识别结果在原始音频中的位置。
具体支持的时间戳类型取决于所使用的接口和配置

如何避免片段边界的重复与遗漏：
结合重叠区域、时间戳和文本对齐进行处理，检查相邻片段是否重复识别同一内容，或者遗漏跨边界的词语。

## 4. 实验结果验收

三个checkpoint是否成功加载：
概念上已明确需要分别读取配置；

三个模型的Mel维数是否与模型配置一致：
是

CSV是否成功保存：
是

是否明确区分直接截断与长音频分块推理：
已理解。
直接截断会丢弃超出目标长度的内容；分块推理会处理多个片段，再合并识别结果，但需要额外处理边界问题。

## 5. 当前理解

- 为什么模型输入维数必须与checkpoint匹配：
模型的输入结构和训练时使用的特征定义必须一致。
不能仅通过在较短特征末尾补零来模拟另一套Mel滤波器组产生的特征。

- 为什么长音频不能只检查输出shape：
不同长度的原始音频可能经过补零或截断后得到相同形状。shape不能反映真实语音的有效时长，也不能证明所有原始内容均被保留。

- 直接截断与分块推理的本质区别：
直接截断只保留目标长度内的内容，超出部分被丢弃。
分块推理将长音频拆成多个片段进行处理，再根据时间戳和边界规则合并识别结果，从而利用整段音频。

- 当前项目何时需要真正加载模型权重：
当进入ASR推理、模型微调或模型评估阶段时，需要加载相应的模型权重。
只读取Processor和模型配置时，不需要加载完整模型权重。