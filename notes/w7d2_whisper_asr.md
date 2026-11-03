# W7D2 模块1：音频输入与 Whisper Processor

## 1. 本模块目标

本模块的数据处理流程：
$$
\text{Waveform}
\rightarrow \text{STFT}
\rightarrow \text{Mel Filter Bank}
\rightarrow \text{Log-Mel}
$$

Processor将原始音频转换为符合Whisper模型输入要求的Log-Mel特征

为什么不能直接把原始 Waveform 输入 Whisper Encoder：
Whisper的Encoder接受的是Log-Mel 频谱特征，而不是原始一维时域波形。因此需要先通过 Feature Extractor 将 Waveform 转换为模型要求的特征表示。

为什么使用 WhisperProcessor 而不是直接复用之前的 waveform_to_mel：
因为 WhisperProcessor 内部封装了与预训练模型匹配的特征提取配置和处理逻辑，可以保证推理时的输入特征与模型训练时的特征定义一致。之前自己实现的 waveform_to_mel 主要用于理解特征提取原理，不能保证与 Whisper 的完整预处理完全一致。

## 2. 音频输入

使用的音频路径：
results/w06d7_resampled_16k.wav

原始采样率：
16000

原始 Waveform shape：
$$
(1,101376)
$$
传入 Processor 前使用 squeeze 去掉声道维度：
$$
(101376,)
$$

音频时长：
$$
T=\frac{101376}{16000}=6.336\text{ s}
$$

Whisper 要求的采样率：
16000 Hz。

为什么采样率必须匹配：
采样率决定了每秒采集多少个采样点，影响音频时间尺度和频率解释。如果原始音频实际采样率为32000 Hz，却错误声明为16000 Hz，模型会将音频误认为持续时间增加一倍，频率也会被错误解释。

## 3. WhisperProcessor

WhisperProcessor 由哪两个主要组件组成：
1. WhisperFeatureExtractor。
2. WhisperTokenizer。

Feature Extractor 的作用：
将原始Waveform转换为Whisper所需的Log-Mel特征，包括STFT、Mel滤波器组、对数压缩及相应的特征处理。

Tokenizer 的作用：
负责文本与Token IDs之间的转换。
- 文本编码为Token IDs
- Token IDs解码为文本

processor() 的输入：
输入一维Waveform数组，同时需要提供正确的采样率

processor() 的输出：
本次实验输出字段：input_features 其中包含转换后的 Log-Mel 特征。

input_features 的形状：
$$
(1,80,3000)
$$

为什么短音频也可能得到固定长度的特征：
Whisper 使用30秒的标准输入窗口。
对于不足30秒的音频，Processor 会进行 padding，使其达到模型要求的输入长度。
30秒对应的采样点数：
$$
N=30\times16000=480000
$$
由于 hop_length 为160：
$$
N_{\mathrm{frames}}=\frac{480000}{160}=3000
$$
因此，最终输入特征形状为：
$$
\boxed{(1,80,3000)}
$$

## 4. 与 W6 特征提取实验的联系

W6D7 使用的 STFT 参数：
之前使用自己的features.py进行STFT和Log-Mel特征提取。
其中需要根据实际调用参数记录n_fft、win_length、hop_length、window和center等配置，不能只依据函数默认值判断。

Whisper 的 STFT 参数：
本次 whisper-small 配置：

- sampling_rate = 16000
- n_fft = 400
- hop_length = 160
- feature_size = 80
- chunk_length = 30

两种 Log-Mel 特征是否完全相同：
不一定，即使Mel维数都是80，也可能因为STFT参数、Mel滤波器组、归一化和对数压缩方式不同，导致最终数值不同。

为什么不能直接把 W6D7 的特征送入 Whisper：
因为Whisper的预训练模型依赖特定的输入特征定义。
如果自己计算的特征与Whisper的训练预处理不一致，即使形状相同，也不代表特征分布和数值含义相同。
因此，推理时应优先使用对应checkpoint的WhisperProcessor。

## 5. 实验结果

Processor checkpoint：
openai/whisper-small

Transformers 版本：
5.21.0

输入音频采样率：
16000 Hz。

输入音频样本数：
101376。

Processor 输出字段：
input_features

input_features shape：
$$
(1,80,3000)
$$

input_features dtype：
torch.float32。

实验是否通过：
通过。

## 6. 模块验收

Q1：为什么输入 Waveform 是 (101376,)，而 Processor 输出变成 (1,80,3000)？
因为原始 Waveform 是一维时域信号，经过 STFT、Mel 滤波器组和 Log 压缩后，变成二维 Log-Mel 特征。
由于 Whisper 使用30秒标准输入窗口，不足30秒的音频需要 padding。
最终增加 Batch 维度，得到：
$$
(1,80,3000)
$$
分别表示 Batch Size、Mel 频带数量和时间帧数。

Q2：为什么不能直接使用 W6D7 的 Log-Mel 特征？
因为两者的特征提取参数和处理约定可能不同。
除了 n_fft、hop_length 等 STFT 参数，还需要关注窗函数、Mel 滤波器组、频率范围、Log 压缩和归一化方式。

Q3：Feature Extractor 和 Tokenizer 分别负责什么？
Feature Extractor 负责将原始 Waveform 转换为 Log-Mel 特征。
Tokenizer 负责文本与 Token IDs 之间的转换。

Q4：如果将实际采样率为32000 Hz的音频错误声明为16000 Hz，会发生什么？
如果没有真正重采样，只是错误地声明采样率，那么模型会将音频的持续时间解释为实际的两倍，频率也会被错误解释为原来的一半。
这不等于发生混叠。
真正进行32000 Hz到16000 Hz的降采样时，需要进行适当的抗混叠低通滤波。


# W7D2 模块2：Whisper 模型推理与 Token 解码

## 1. Whisper 模型结构

Whisper 属于哪一种 ASR 范式：
Attention Encoder-Decoder(AED)
Whisper使用Encoder-Decoder Transformer架构，通过Cross-Attention实现音频特征与文本生成之间的联系。

Whisper Encoder 的输入是什么：
Whisper Processor提取的Log-Mel频谱特征。
本次输入形状：
$$
(1,80,3000)
$$

Whisper Encoder 的输出是什么：
高层音频隐藏状态，包含从输入语音提取的上下文信息。

Whisper Decoder 的作用：
根据已经生成的历史Token，以及Encoder输出的音频隐藏状态，自回归预测下一个Token。
Decoder主要利用：

- Masked Self-Attention：建模历史 Token 之间的关系。
- Cross-Attention：利用 Encoder 提供的音频特征。
- Feed Forward Network：进一步进行特征变换。

Whisper 为什么采用自回归生成：
因为模型在预测当前Token时，需要利用已经生成的历史Token。
通过逐步预测下一个Token，最终生成完整文本。
$$
P(Y|X)=\prod_{u=1}^{U}P(y_u|y_{<u},X)
$$
模型不需要预先知道最终需要生成多少个Token，而是通过逐步生成，并根据EOS等停止条件确定最终长度

## 2. 模型加载

使用的 checkpoint：
openai/whisper-small

WhisperForConditionalGeneration 的作用：
封装 Whisper 的 Encoder、Decoder 及生成相关逻辑，用于根据输入音频特征生成文本 Token IDs。

model.eval() 的作用：
将模型切换到评估模式，使Dropout 等模块按照评估模式运行。
对于 BatchNorm 等模块，使用运行统计量而不是当前 Batch 的统计量。
该操作本身不会关闭梯度计算。

torch.inference_mode() 的作用：
在推理过程中关闭梯度跟踪及相关自动求导开销，减少内存占用。

为什么需要将模型与输入放在同一 device：
模型参数和输入 Tensor 必须位于兼容的设备上，否则执行矩阵运算时可能发生设备不一致错误。
本次模型和实际推理输入均位于 CUDA。

## 3. generate() 推理

generate() 接收什么输入：
Whisper Processor 输出的 Log-Mel 特征，即 input_features。

generate() 的作用：
调用 Whisper Encoder-Decoder 进行自回归推理，根据音频特征和历史生成 Token IDs。

language="zh" 的作用：
指定识别语言为中文，影响 Whisper 的语言任务前缀及生成配置。

task="transcribe" 的作用：
指定执行语音转写任务，即将语音转成对应语言的文本，而不是翻译成其他语言。

max_new_tokens 的作用：
限制最多生成的新 Token 数量。
本次设置：
$$
\text{max_new_tokens}=96
$$
它不表示必须生成96个 Token。

generated_ids 的形状及含义：
本次输出：
$$
(1,30)
$$
其中：

- 1：Batch Size。
- 30：本次生成序列的 Token 数量。

## 4. Token 解码

generated_ids 与最终文本的关系：
generated_ids是模型生成的整数Token ID序列

Tokenizer 在解码阶段的作用：
根据词表和解码规则，将整数Token IDs转换为对应的文本。

skip_special_tokens=True 的作用：
解码时跳过特殊Token，例如语言标记、任务标记和结束标记等。
它不会改变已经生成的 Token IDs，也不会改变模型的推理过程。

为什么 Whisper 输出中可能包含特殊 token：
Whisper 是多任务语音模型，使用特殊 Token 表示语言、任务、时间戳及生成结束等信息。
例如：

- <|startoftranscript|>：转写序列开始。
- <|zh|>：中文语言标记。
- <|transcribe|>：转写任务标记。
- <|notimestamps|>：不生成时间戳。
- <|endoftext|>：结束标记。

batch_decode() 与 decode() 的区别：

- decode()：解码单个Token ID序列。
- batch_decode()：批量解码多个Token ID序列

## 5. 实验结果

运行设备：
CUDA。

模型 checkpoint：
openai/whisper-small

input_features shape：
$$
(1,80,3000)
$$

generated_ids shape：
$$
(1,30)
$$

实际 Token IDs：
略

包含特殊 token 的解码结果：
大SHA的主要租户包括美林集团及美国运通大型金融机构

去除特殊 token 的转写文本：
大SHA的主要租户包括美林集团及美国运通大型金融机构

模型推理是否成功：
成功

## 6. 当前理解

为什么 generate() 不直接返回中文文本：
因为模型的输出层首先生成词表中各个 Token 的概率分布，再根据解码策略选择 Token ID。
Tokenizer 负责将生成的 Token IDs 转换成文本。

Whisper 的自回归生成体现在哪里：
Decoder 每次根据 Encoder 音频特征以及已经生成的历史 Token，预测下一个 Token。
新生成的 Token 会成为下一次预测的历史输入。

Processor、Model、Tokenizer 三者如何协作：

- Processor 的 Feature Extractor：Waveform → Log-Mel 特征。
- Model：Log-Mel 特征 → Token IDs。
- Tokenizer：Token IDs → 文本。
完整流程：

$$
\text{Waveform}
\xrightarrow{\text{Feature Extractor}}
\text{Log-Mel}
$$

$$
\xrightarrow{\text{Whisper Model}}
\text{Token IDs}
\xrightarrow{\text{Tokenizer}}
\text{Text}
$$

## 7. 模块验收

Q1：Whisper Encoder 和 Decoder 分别负责什么？
Encoder 将输入 Log-Mel 频谱编码为高层音频特征。
Decoder 根据历史 Token 和 Encoder 输出的音频特征，自回归预测下一个 Token。
input_features 是音频特征，不是文本 Token IDs，因此不能直接通过 Tokenizer 将其转换为中文。

Q2：generate() 的输入和输出是什么？
输入是 Log-Mel 音频特征。
输出是整数 Token ID 序列。

Q3：language、task 和 skip_special_tokens 分别作用于哪个阶段？
language 指定语言，task 指定语音转写或翻译任务，二者影响生成准备和任务配置。
skip_special_tokens 控制 Tokenizer 解码时是否跳过特殊 Token，只影响最终文本显示，不改变模型生成过程。

Q4：为什么同时使用 model.eval() 和 torch.inference_mode()？
model.eval() 将模型切换到评估模式，控制 Dropout 等模块的行为。
torch.inference_mode() 关闭梯度跟踪及相关自动求导开销。
两者作用不同，通常同时使用。


# W7D2 模块3：封装 ASR 推理接口

## 1. 模块设计

为什么需要封装 ASR 推理接口：
之前在 Notebook 中分别实现了音频预处理、Whisper 模型推理和 Token 解码。
但是，如果每次进行 ASR 实验都重新编写这些代码，会产生大量重复工作，也不利于后续批量处理音频。
因此，将这些操作封装为可复用的函数，形成独立的 ASR 推理接口。

本模块包含哪些主要函数：
1. load_whisper_model()
2. transcribe_file()

各函数分别负责什么：
load_whisper_model()：
负责加载 WhisperProcessor 和 WhisperForConditionalGeneration，选择运行设备，并将模型切换为评估模式。

transcribe_file()：
负责读取音频、转换单声道、检查采样率、必要时重采样、提取 Log-Mel 特征、执行模型推理和 Token 解码。

## 2. 模型加载

load_whisper_model() 的作用：
加载指定 checkpoint 对应的 Processor 和 Whisper 模型，返回：
- processor
- model
- device

为什么模型只需要加载一次：
模型权重较大，加载过程需要时间和显存。
如果每识别一条音频都重新加载模型，会产生不必要的计算和内存开销。
因此，应该先加载模型，再重复调用音频推理函数

device 如何选择：
根据 CUDA 是否可用自动选择

为什么需要 model.eval()：
将模型切换到评估模式，使 Dropout 等模块按照评估模式运行。
它本身不会关闭梯度计算。

## 3. 音频预处理

transcribe_file() 如何读取音频：
调用之前实现的 audio_utils.py 中的 load_audio()，返回：
waveform
original_sr
Waveform 的形状为：
$$
(C,T)
$$

为什么需要统一单声道：
Whisper 的输入预处理采用单声道音频。
如果输入有多个声道，需要通过 to_mono() 将多个声道转换为一个声道。

为什么需要检查并重采样：
Whisper-small 要求的采样率为16000 Hz。
如果原始音频采样率不同，需要真正进行重采样，而不是仅仅修改采样率标签。
例如32000 Hz转换为16000 Hz时：
$$
N_{\mathrm{new}}\approx\frac{N_{\mathrm{old}}}{2}
$$
在音频时长不变的情况下，采样点数量约减少一半。

Processor 如何生成 input_features：
先将 Waveform 转换为一维 NumPy 数组，再调用：
inputs = processor(
    audio_np,
    sampling_rate=target_sr,
    return_tensors="pt",
)
获得 Log-Mel 特征。

## 4. 模型推理与解码

generate() 的输入：
Whisper Processor 输出的 input_features。

generate() 的输出：
模型生成的 Token IDs。

Tokenizer 如何得到最终文本：
调用 processor.batch_decode()，将 Token IDs 转换为文本。

为什么返回 generated_ids 和 metadata：
generated_ids 用于分析模型实际生成的 Token 序列。
metadata 用于记录实验信息，例如：

- 音频路径
- 模型 checkpoint
- 原始采样率
- 模型采样率
- 音频时长
- 运行设备
- 输入特征形状
- 生成序列形状

这些信息可以帮助后续进行实验追踪、结果分析和复现。

## 5. 实验结果

模型 checkpoint：
openai/whisper-small

运行设备：
CUDA。

输入音频：
results/w06d7_resampled_16k.wav

原始采样率：
16000 Hz。

模型采样率：
16000 Hz。

音频时长：
6.336 秒。

input_features shape：
$$
(1,80,3000)
$$

generated_ids shape：
$$
(1,30)
$$

最终转写文本：
大SHA的主要租户包括美林集团及美国运通大型金融机构

## 6. 模块验收

Q1：为什么 load_whisper_model() 和 transcribe_file() 要拆成两个函数？
因为模型加载和音频推理是不同的操作。
模型权重较大，如果每处理一条音频都重新加载模型，会浪费时间和显存。
拆分后，可以加载一次模型，再重复调用推理函数处理多条音频

Q2：如果输入音频采样率为32000 Hz，而 Whisper 要求16000 Hz，具体需要进行哪些处理？
首先读取原始音频，记录真实采样率。
然后使用重采样算法将音频从32000Hz转为16000Hz。重采样后采样点数减少约一般，但音频时长保持不变。

Q3：为什么返回 transcript、generated_ids 和 metadata？
transcript 用于获取最终识别文本，后续计算 WER、CER。
generated_ids 用于分析模型实际生成的 Token 序列。
metadata 用于记录实验配置，方便复现和比较不同实验结果。

Q4：如果需要识别100条音频，应该如何调用？
首先调用 load_whisper_model() 加载一次模型。
然后使用循环依次调用 transcribe_file()，处理100条音频。
