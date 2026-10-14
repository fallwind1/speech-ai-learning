# W4D5 模块1：为什么先分组再切片


## 1. 数据预处理的基本流程


原始音频：

>通常包含：
- 音频文件；
- 说话人信息；
- 录音场次信息；
- 来源信息

在进入模型训练前，需要经过：
原始音频

↓

数据登记(source_registry)

↓

按照speaker/session/source进行分组

↓

划分train/dev/test

↓

切片生成训练样本

↓

生成manifest.csv

↓

Dataset读取

↓

模型训练



正确流程：

先根据数据来源进行划分：

- speaker；
- session；
- parent_uid；

确定所属split
然后再对每个split内部的音频进行切片


---

## 2. 为什么不能直接随机切片


问题：

>如果先随即切片再随即划分，可能导致同一个说话人或同一个录音场次的片段同时出现在train和test中。


原因：

>同一个session中的音频通常具有相同的信息：

- 说话人声音特征；
- 录音环境；
- 麦克风特征；
- 背景噪声

模型可能学习到这些与语音内容无关的特征，而不是学习真正的语音识别能力


后果：

>测试集结果虚高


---

## 3. Session / Speaker / Segment关系


source_uid：

>表示数据来源的唯一编号


speaker_id：

>表示说话人身份


parent_uid：

>表示当前音频片段来自哪个原始录音文件或录音场次


segment：

>表示最终用于训练的音频片段

通常包含：

- 音频路径；
- 起始时间；
- 结束时间
- 对应文本
- 所属split


它们之间关系：

$$
原始录音(parent)
\rightarrow
多个segment
$$

每个segment继承：

- speaker_id；
- source_uid；
- parent_uid；
- split。

---

## 4. train/dev/test划分原则


train：

>用于训练模型，数据量最大，要求：不能包含dev/test中的speaker或session


dev：

>用于训练过程中的验证和调参

用于：
- 观察训练趋势；
- 调整超参数


test：

>用于最终评估模型泛化能力

要求：尽可能模拟真实未知数据

划分依据：

>不是随机切片，而是按照：

- speaker；
- session；
- parent_uid；

等高层级单位进行划分。


---

## 5. 为什么重叠片段必须属于同一个split


原因：

>如果同一个原始录音中的重叠片段分别进入train和test，那么测试集实际上包含训练数据中的语音信息


否则会导致：

- 数据泄露；
- 测试指标虚高；
- 无法真实评价模型泛化能力。


---

## 6. manifest.csv作用


记录：

- audio_path;
- text
- source_uid
- speaker_id
- parent_uid
- start_time
- end_time
- split

例如：
|audio_path|speaker_id|parent_uid|start|end|split|
|-|-|-|-|-|-|
|001.wav|spk01|record001|0|5|train|

作用：

1. 让Dataset知道读取哪个音频；
2. 保留数据来源和划分信息
3. 保证实验可复现
4. 方便后续错误分析和数据统计

整体流程：

manifest.csv

↓

Dataset

↓

audio_utils

↓

feature extractor

↓

model


---

## 7. 当前理解


>数据预处理的核心不是简单地切割音频，而是保证训练数据和测试数据之间不存在信息泄露

正确的数据处理顺序：

原始音频

↓

记录来源信息

↓

按照speaker/session/source进行分组

↓

划分train/dev/test

↓

切片生成5–30秒segment

↓

生成manifest.csv


其中：

- speaker_id控制说话人泄露；
- parent_uid控制原始录音泄露；
- manifest.csv保证数据可追踪。


最终目标：

让测试集真正反映模型面对未知语音数据时的泛化能力。


# W4D5 模块2：manifest.csv设计与生成流程


## 1. manifest.csv作用


>manifest.csv是模型训练阶段使用的数据索引文件，用于记录每一个训练样本对应的音频位置、文本标签以及数据来源信息。它不是保存音频本身，而是保存音频与metadata之间的关联。


---

## 2. manifest字段设计


audio_path：

>音频路径，表示音频文件在哪里，读取时：waveform, sr = load_audio(audio_path)


text：

>文本标签，表示该音频对应文本
>ASR任务：输入audio，预测text


source_uid：

>数据来源
>作用：后续可以统计不同数据来源占比


speaker_id：

>表示说话人，作用：检查train_speaker ∩ test_speaker是否为空


parent_uid：

>表示原始录音，作用：追踪片段来源


start_time：

>该segment再原始音频中的开始时间


end_time：

>该segment再原始音频中的结束时间


split：

>表示数据属于哪个数据集


---

## 3. 为什么需要parent_uid


>因为一个原始录音通常会被切割成多个segment
如果没有parent_uid：

后续无法判断这些segment是否来自同一个原始录音。


---

## 4. manifest生成流程


输入：

>原始数据：
- Common Voice TSV
- source_registry.csv
- 原始音频文件


处理：

第一步：
读取原始metadata
获取：
- speaker_id
- audio_path
- text
- source信息

第二步：
按照高层级单位划分：
- speaker
- session
- parent_uid

确定train/dev/test

第三步：
对已经确定split的数据进行切片

例如：
原始：record001.wav 60s，切成：0-10、10-20、20-30

第四步：
记录每个segment的信息
包括：
- audio_path
- text
- speaker_id
- parent_uid
- start_time
- end_time
- split

第五步：
保存



---

## 5. segment切片规则


切片长度：

>5-30s


切片方式：

>固定窗口


重叠处理：

如果两个segment来自同一个parent_uid，并且存在时间重叠：

必须属于同一个split。


---

## 6. 数据统计


统计：

- 样本数量
- 总时长
- train speakers
- dev speakers
- test speakers


---

## 7. 当前理解


>manifest.csv是连接原始数据和模型训练之间的重要桥梁。
完整流程：
原始音频

↓

source_registry

记录数据来源和授权

↓

speaker/session划分

↓

train/dev/test划分

↓

音频切片

↓

manifest.csv

↓

Dataset读取

↓

audio_utils处理

↓

feature extraction

↓

ASR模型训练


其中：

- source_uid负责追踪数据来源；
- speaker_id负责控制说话人泄露；
- parent_uid负责控制原始录音泄露；
- start/end_time负责数据溯源；
- manifest负责训练样本索引。


最终目标：

保证训练数据可追踪、实验可复现、测试结果可信。


# W4D5 模块3：生成manifest.csv


## 1. 输入数据


数据来源：

>Common Voice 中文数据集。


已有文件：

- train.tsv
- dev.tsv
- test.tsv
- source_registry.csv


包含字段：

>source_registry.csv用于记录数据来源、授权以及实验使用的数据版本。

train/dev/test.tsv包含：

- client_id
- path
- sentence
- age
- gender
- accents
等字段。


---

## 2. manifest生成流程


Step 1：

>读取Common Voice提供的TSV文件。


Step 2：

>将原始字段映射为manifest字段。


Step 3：

>补充训练需要的metadata。

增加：

- source_uid；
- parent_uid；
- start_time；
- end_time；
- split。


Step 4：

>合并train/dev/test三个split，并保存：

metadata/manifest.csv


---

## 3. manifest字段


最终字段：

- audio_path
- text
- source_uid
- speaker_id
- parent_uid
- start_time
- end_time
- split


每个字段作用：

audio_path：

表示音频文件路径。

Dataset根据该路径加载音频。


text：

表示音频对应的文本标签。

ASR训练中作为模型预测目标。


source_uid：

表示数据来源。

例如：

common_voice_zh-CN_v17


用于：

- 追踪数据来源；
- 管理数据版本；
- 统计不同来源数据比例。


speaker_id：

表示说话人身份。

当前Common Voice中使用：

client_id

作为speaker_id。


用于检查：

train/dev/test之间是否存在说话人交叉。


parent_uid：

表示当前segment来自哪个原始录音。


当前Common Voice中：

每个clip本身就是独立音频，

因此：

parent_uid可以设置为当前音频文件名。


start_time：

表示该segment在parent音频中的开始时间。


当前Common Voice中：

由于clip本身就是完整音频，

通常从：

0秒开始。


end_time：

表示该segment在parent音频中的结束时间。


当前阶段可以为空或设置为音频长度。


未来长录音切片时：

需要记录真实起止时间。


split：

表示当前样本属于：

- train
- dev
- test


用于控制训练、验证和测试数据。


---

## 4. 数据划分验证


speaker交叉检查：

>分别统计train、dev、test的额speaker集合，检查：
$$
S_{train}\cap S_{test}
$$

$$
S_{train}\cap S_{dev}
$$

$$
S_{dev}\cap S_{test}
$$
三个集合之间是否存在交集


parent_uid交叉检查：

检查同一个parent_uid是否被划分到不同split。


对于当前Common Voice：

由于一个clip对应一个parent，

因此不会出现同一parent被拆分的问题。


结果：

>保证：同一来源音频不会同时出现在训练集和测试集。


---

## 5. 统计结果


train：

>样本数量：2281，说话人数量：27


dev：

>样本数量：409，说话人数量：33


test：

>样本数量：170，说话人数量：58



---

## 6. 当前限制


>当前使用的是Common Voice已经切分好的clip
因此：

- parent_uid与segment基本是一一对应；
- 不涉及长音频切片；
- start_time/end_time信息较简单。


---

## 7. 当前理解


>manifest.csv是连接原始数据和模型训练的数据索引文件。
完整流程：

原始数据

↓

source_registry

记录数据来源和授权

↓

确定实验使用的数据版本

↓

生成manifest.csv

↓

Dataset读取

↓

audio_utils处理

↓

特征提取

↓

模型训练


其中：

- source_registry关注数据来源是否合法；
- manifest关注模型实际使用哪些样本。


当前Common Voice流程：

Common Voice TSV

↓

字段转换

↓

添加metadata

↓

生成manifest.csv


未来长音频流程：

长音频

↓

划分split

↓

切片

↓

记录parent_uid和start/end_time

↓

生成manifest.csv


最终目标：

保证训练数据：

- 可追踪；
- 可复现；
- 无数据泄露。


# W4D5：分组与预处理


## 1. 本日完成内容


- 理解语音数据预处理中的数据泄露问题，明确为什么需要先按照 speaker/session/source 等高层级信息划分数据，再进行音频切片。
- 学习 source_registry.csv 与 manifest.csv 的区别：
  - source_registry 负责记录数据来源、授权信息和数据版本；
  - manifest 负责记录模型训练阶段实际使用的样本信息。
- 基于 Common Voice 数据理解 manifest 生成流程，将原始 TSV 字段转换为模型训练需要的数据格式。
- 学习 manifest.csv 中的核心字段设计，包括：
  - audio_path；
  - text；
  - source_uid；
  - speaker_id；
  - parent_uid；
  - start_time；
  - end_time；
  - split。
- 理解当前 Common Voice 数据已经是独立 clip，因此无需重新进行 5–30 秒切片；未来长音频需要增加切片和 parent 映射流程。
- 学习数据质量验证方法，包括：
  - speaker overlap 检查；
  - parent/session 交叉检查；
  - 数据规模统计；
  - 数据可追溯检查。


---

## 2. 主要代码与产出


| 文件 / 产出 | 内容 |
|---|---|
| `src/build_manifest.py` | 将 Common Voice TSV 转换为 manifest.csv |
| `metadata/source_registry.csv` | 数据来源登记、授权信息和实验数据版本记录 |
| `metadata/manifest.csv` | 模型训练使用的数据索引文件 |
| manifest 统计检查脚本 | 检查样本数量、speaker 数量以及数据划分情况 |


---

## 3. 数据处理流程


输入：

Common Voice 原始数据：

- train.tsv；
- dev.tsv；
- test.tsv；

以及：

- source_registry.csv。


处理：

1. 读取 Common Voice TSV 文件。

2. 提取原始字段：

- path；
- sentence；
- client_id。


3. 映射为 manifest 字段：

- path → audio_path；
- sentence → text；
- client_id → speaker_id。


4. 添加额外 metadata：

- source_uid；
- parent_uid；
- start_time；
- end_time；
- split。


5. 合并 train/dev/test 数据。

6. 保存：

```
metadata/manifest.csv
```


输出：

```
metadata/manifest.csv
```


manifest.csv 作为后续：

```
manifest.csv

↓

Dataset

↓

audio_utils

↓

feature extraction

↓

ASR model
```

的数据入口。


---

## 4. 数据统计结果


统计内容：

| split | clips | speakers | duration |
|---|---|---|---|
| train | 根据最终 manifest 统计 | 根据最终 manifest 统计 | 根据最终 manifest 统计 |
| dev | 根据最终 manifest 统计 | 根据最终 manifest 统计 | 根据最终 manifest 统计 |
| test | 根据最终 manifest 统计 | 根据最终 manifest 统计 | 根据最终 manifest 统计 |


总计：

需要注意：

统计结果必须对应最终生成 manifest 的数据版本。


W4D3 中：

```
38.10 min
437 clips
118 speakers
```

对应的是已经筛选后的 source_registry 子集。


如果直接使用 Common Voice 完整 train/dev/test.tsv 生成 manifest，则统计结果会变为完整数据集规模，不能与之前筛选子集的统计混用。


---

## 5. 数据划分验证


speaker overlap：

检查：

$$
S_{train}\cap S_{dev}
$$

$$
S_{train}\cap S_{test}
$$

$$
S_{dev}\cap S_{test}
$$


目标：

三个结果均为空。


说明：

不同数据集之间不存在相同 speaker，可以避免说话人泄露。


parent_uid 检查：

检查同一个 parent_uid 是否同时存在于不同 split。


对于当前 Common Voice：

由于每个 clip 本身就是独立音频，因此：

```
parent_uid ≈ 当前音频文件
```


不会出现长音频切片导致的 parent 交叉问题。


结果：

保证：

- 训练集与测试集来源没有交叉；
- 数据划分符合实验要求。


---

## 6. 本日真正理解的内容


我现在能够解释：

- 为什么语音任务不能直接随机切片后划分 train/test。
- 数据泄露如何导致测试结果虚高。
- speaker_id 用于控制说话人泄露。
- parent_uid 用于控制原始录音来源泄露。
- source_registry.csv 和 manifest.csv 分别解决什么问题。
- manifest.csv 为什么是模型训练阶段的重要数据入口。
- 为什么代码运行成功不代表数据一定正确，需要额外进行数据质量检查。
- 为什么长音频需要记录 start_time 和 end_time。
- 为什么 Common Voice 这种已经切好的 clip 和长录音的 manifest 生成流程不同。
- 为什么实验需要记录数据版本，而不能只记录“使用某某数据集”。


---

## 7. 当前不会 / 薄弱点


1. 当前只完成了基于已有 clip 数据的 manifest 生成流程，对于长音频自动切片、重叠窗口以及 segment 生成逻辑还没有实际代码实践。

2. 当前 manifest 生成主要依靠 pandas 完成，对于大规模 ASR 数据集中的高效数据处理方式仍需要学习。

3. 当前数据质量检查主要关注 speaker 和 parent 交叉，后续还需要学习更系统的数据分布分析方法。

4. 当前 Common Voice 的 speaker_id 直接使用 client_id，实际工业数据中需要进一步理解更复杂的 speaker/session 管理方式。


---

## 8. 实际学习时间


约 2 小时。


---

## 9. 下一步三件事


1. 完成 W4D6 音频数据 Dataset 与 DataLoader 构建相关内容，进一步连接 manifest、Dataset 和模型训练。

2. 保持当前数据工程规范，后续所有实验记录数据来源、版本以及划分方式。

3. 在后续 ASR 任务中实践完整流程：

```
manifest

↓

Dataset

↓

feature extraction

↓

model training
```

确保数据处理和模型训练形成完整闭环。