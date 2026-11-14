# W8D7：CTC 核心复盘与工程验收

## 1. 本周 CTC 学习目标

本周围绕 CTC（Connectionist Temporal Classification）完成了理论学习、算法实现、测试验证和实际 ASR 应用。

主要目标：
- 理解 CTC 的条件独立性假设。
- 理解 CTC Collapse 和路径概率求和。
- 独立实现 CTC Forward Algorithm。
- 使用暴力枚举和 PyTorch CTCLoss 验证实现。
- 实现 Greedy Decoder。
- 将 CTC 应用于 wav2vec 2.0 的语音识别任务。

## 2. CTC 核心理论

### 2.1 条件独立性

CTC 假设给定输入音频 X 后，各时间步的输出标签条件独立：

P(π|X) = ∏ P(π_t|X)

这里的条件独立不意味着不同时间步的 Encoder 特征相互独立。

以 wav2vec 2.0 为例，Transformer Context Network 通过 Self-Attention 建模整段音频的上下文信息，使每个时间步的特征都能够包含其他时间步的信息。

因此：
- Encoder 负责上下文建模。
- CTC 采用给定输入后的帧级输出条件独立假设。
- 两者并不矛盾。

### 2.2 CTC Collapse

CTC Collapse 包含两个操作：
1. 合并连续重复 Token。
2. 删除 Blank。

例如：

A A Blank A B B Blank

经过 Collapse：

A A B

注意：先合并连续重复 Token，再删除 Blank。

因此：
- A A → A
- A Blank A → A A

重复字符之间需要通过 Blank 或其他不同字符隔开。

### 2.3 CTC 路径概率

一条路径的概率：

P(π|X) = ∏ P(π_t|X)

目标文本的概率：

P(y|X) = Σ P(π|X)，其中 B(π)=y

即将所有能够映射到目标文本的合法路径概率相加。

CTC Loss：

L_CTC = -log P(y|X)

### 2.4 Forward Algorithm

首先对目标序列插入 Blank，构造扩展目标序列：

Blank, y1, Blank, y2, Blank, ...

定义 α_t(s) 为时间步 t 到达扩展目标状态 s 的所有合法路径概率之和。

三种状态转移：
- 停留在当前状态。
- 从前一个状态前进一步。
- 从前两个状态跳跃。

跳跃转移需要满足：
- 当前状态不是 Blank。
- 当前标签与前两个状态的标签不同。

终止时，最终概率需要累加最后一个字符状态和最后一个 Blank 状态的概率。

本项目已经独立实现了 CTC Forward Algorithm，并通过暴力枚举及 PyTorch CTCLoss 对照验证。

## 3. 本周代码实现

主要文件：

src/
- ctc.py
- ctc_decode.py

tests/
- test_ctc.py
- test_ctc_decode.py

主要功能：

1. CTC Forward：计算目标文本的对数概率。
2. CTC Brute Force：通过枚举路径验证 Forward。
3. CTC Greedy Decode：完成帧级预测到文本序列的转换。
4. Beam Search：实现基于候选路径的解码。

以上核心代码已经在之前的 Git Commit 中提交，W8D7 不重复实现。

## 4. 自动化测试结果

项目整体测试：

18 passed in 16.46s

其中 CTC 专项测试共 10 项。

### Forward 测试

| 测试名称 | 验证内容 | 结果 |
|---|---|---|
| test_bruteforce_matches_dp | 暴力枚举与动态规划一致 | PASS |
| test_repeated_tokens | 重复字符 | PASS |
| test_empty_target | 空目标 | PASS |
| test_impossible_alignment | 不可能对齐 | PASS |

### Decode 测试

| 测试名称 | 验证内容 | 结果 |
|---|---|---|
| test_repeated_tokens | 连续重复 Token | PASS |
| test_blank_separates_repeated_tokens | Blank 分隔重复字符 | PASS |
| test_consecutive_blanks | 连续 Blank | PASS |
| test_all_blank | 全 Blank 路径 | PASS |
| test_empty_path | 空路径 | PASS |
| test_greedy_decode | Greedy 解码 | PASS |

CTC 专项测试结果：

- 总数：10
- PASS：10
- FAIL：0

测试报告：
results/w08d7_ctc_test_table.csv

原始报告：
results/w08d7_ctc_junit.xml

## 5. 本周实际遇到的问题

| 问题 | 现象 | 来源 |
|---|---|---|
| Datasets 缺失 | 无法导入 datasets | Python 环境依赖 |
| 数据读取缓慢 | 远程读取等待时间较长 | Hugging Face 数据读取 |
| TorchCodec 缺失 | 音频解码报错 | 音频解码依赖 |
| Speaker 集中 | 20 条音频均来自同一 Speaker | 顺序抽样 |
| Chapter 集中 | 20 条音频均来自同一 Chapter | 顺序抽样 |
| Whisper 语言设置 | 默认中文与英文任务不匹配 | 推理参数 |
| WER 文本差异 | 大小写和标点不同 | 模型输出格式 |

这些问题分别涉及环境配置、数据处理、实验设计和评估规范，不能全部归类为算法错误。

## 6. W8D6 英文 ASR 实验结果

在固定的 20 条 LibriSpeech 英文音频上完成 wav2vec2 和 Whisper 对照。

| 模型 | S | D | I | N | WER |
|---|---:|---:|---:|---:|---:|
| wav2vec2-base-960h | 6 | 0 | 0 | 321 | 1.87% |
| Whisper-small | 9 | 3 | 0 | 321 | 3.74% |

当前固定样本中，wav2vec2 的 WER 更低。

但由于样本量小，且全部来自同一 Speaker 和 Chapter，不能据此推断两种模型架构的普遍优劣。

## 7. 工程管理与 Git

本周主要提交：

- ec703c9：CTC Forward Algorithm
- 0552e98：Brute Force 和边界测试
- 761a0d0：CTC Decode 和 Beam Search
- 8014dcd：W8D5 wav2vec 2.0
- 3f2e0ac：W8 英文 ASR WER 评估

W8D7 补充测试报告和复盘文档。

## 8. 本周总结

本周完成了从 CTC 理论学习、算法独立实现、单元测试，到 wav2vec 2.0 实际推理和 WER 评估的完整学习过程。

最重要的收获是理解了 CTC 如何通过帧级分类和路径概率求和解决输入与输出序列不对齐的问题，并通过独立实现 Forward Algorithm 和 Greedy Decoder，将理论知识转化为可运行、可测试的代码。

同时，通过英文 ASR 对照实验，进一步熟悉了模型推理、数据管理、文本规范化和语料级 WER 评估流程。

后续改进方向：
- 补充 Beam Search 专项测试。
- 扩大英文测试集并增加 Speaker 多样性。
- 继续学习更复杂的 ASR 模型和训练流程。