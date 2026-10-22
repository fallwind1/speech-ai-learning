# W5D6 模块1：SNR 原理与实验设计

## 1. SNR 定义

SNR（Signal-to-Noise Ratio）表示有用信号功率与噪声功率之比。

对于长度为N的离散信号x[n]，平均功率为：

P_x = (1/N) Σ x[n]²

噪声v[n]的平均功率为：

P_v = (1/N) Σ v[n]²

以分贝表示的信噪比为：

SNR_dB = 10 log10(P_x / P_v)

功率比使用10log10；在适用条件下，如果比较的是幅度比，则可以利用功率与幅度平方成正比的关系使用20log10。

## 2. 目标 SNR = 10 dB

当目标SNR为10 dB时：

P_x / P_v = 10

因此：

P_v = P_x / 10

例如，若干净语音功率为0.02，则目标噪声功率为0.002。

这表示噪声功率是语音功率的十分之一，不代表噪声幅度是语音幅度的十分之一。

## 3. 随机噪声缩放

设z[n]为随机生成的高斯白噪声，其实际功率为：

P_z = mean(z²)

为了得到目标噪声功率，需要将其缩放：

v[n] = αz[n]

由于功率随幅度缩放系数的平方变化：

P_v = α²P_z

所以：

α = sqrt(P_v / P_z)

对于目标SNR为10 dB的情况：

α = sqrt(P_x / (10P_z))

最后生成：

y[n] = x[n] + v[n]

其中x[n]为干净语音，v[n]为缩放后的噪声，y[n]为合成的含噪语音。

## 4. 实测 SNR

不能仅因为程序设置了目标SNR为10 dB，就直接把实测结果填写为10 dB。

在浮点数组中，可以通过：

v[n] = y[n] - x[n]

重新估计实际噪声，再计算：

SNR_measured = 10 log10(mean(x²) / mean(v²))

若需要评估写盘后的实际结果，应重新读取保存的音频，并考虑量化、削波等因素。

## 5. 随机种子与来源追踪

seed用于控制随机噪声生成过程，提高实验的可复现性。

parent_uid用于追溯合成样本对应的原始音频。

实验记录还应保存目标SNR、实际SNR、采样率、噪声生成方式和数据条件。

固定seed并不自动保证跨所有软件版本和设备逐位一致，因此还应记录相关实验环境。

## 6. 削波与共同缩放

当信号峰值超过允许范围时，写入有限幅度范围的音频格式可能造成削波。

如果需要降低峰值，应对clean和noisy使用相同的缩放系数，以保持两者的SNR不变。

不能仅对noisy独立归一化，因为这会改变其相对于clean的功率关系。

对于整数PCM格式，还应确保写盘前的数值范围和数据类型正确。

## 7. 合成噪声与自然噪声

clean表示原始干净语音。

synthetic 10 dB表示在干净语音上人工叠加噪声，并将目标SNR设为10 dB。

自然noisy语音则来自真实环境中的含噪录音，两者不能混为一谈。

合成噪声样本需要保留parent_uid、随机seed和实验配置，不应冒充自然录制的测试样本。

## 8. 本模块核心结论

指定SNR的关键不是直接叠加随机噪声，而是根据干净语音功率和随机噪声实际功率计算缩放系数，并在生成后测量实际SNR。

实验应保持原始音频不变，记录来源与随机参数，避免削波，并明确区分合成噪声和自然噪声。


# W5D6 模块2：noise_utils.py 实现

## 1. 模块目标

为什么需要单独创建noise_utils.py：
将可复用的信号处理逻辑与实验流程分开。noise_utils.py负责计算功率、生成指定SNR的含噪波形、测量实际SNR以及进行峰值保护；Notebook负责选择音频、调用函数、保存结果和记录实验配置。

本模块的输入与输出：
输入为一维干净语音波形、目标SNR和随机seed；输出为含噪波形与实际添加的噪声。峰值保护函数可以返回共同缩放后的clean、noisy和缩放系数。

clean和synthetic 10dB分别表示什么：
clean表示不添加噪声的干净语音。synthetic 10dB表示在干净语音上人工叠加噪声，并将目标SNR设为10dB。

## 2. 核心函数

### measure_power

输入：非空、一维、有限数值的波形数组
计算方法：P = mean(x²)
返回值：波形的均方功率，使用浮点数表示。

### measure_snr_db

输入：形状相同的clean和noisy波形。
如何提取实际噪声：
noise = noisy - clean
计算公式：
SNR_dB = 10 × log10(P_clean / P_noise)

为什么需要重新测量：
目标SNR只是生成时的设定值。实际波形、峰值处理、写盘和量化等过程都可能影响最终结果，因此应根据实际信号重新计算。

### add_gaussian_noise_at_snr

输入：干净语音、目标SNR和随机seed
随机seed的作用：
使相同输入、参数和seed在相同软件环境下能够重复生成一致的随机噪声。

目标噪声功率：
P_noise = P_clean / 10^(target_snr_db / 10)

缩放系数：
alpha = sqrt(P_noise / P_random)

输出：
noisy = clean + noise

### protect_pair_peak

为什么需要这个函数：
防止配对波形的峰值超过预先设定的安全上限，减少写盘时削波的风险。

如何计算共同缩放系数：
scale = min(1, ceiling / peak)

为什么不能仅缩放noisy：
因为单独缩放noisy会改变他相对于clean的功率关系，从而改变实际SNR。共同缩放clean和noisy可以保持SNR不变。

## 3. 代码衔接

manifest.csv的作用：
提供原始音频路径、parent_uid和split等元数据。今天选择train音频进行加噪实验，不修改原始manifest。

audio_utils.py与noise_utils.py的分工：
audio_utils.py负责读取音频及必要的通道、采样率处理。
noise_utils.py负责数值信号处理，包括计算功率、生成指定SNR的噪声、测量SNR和共同峰值保护。

snr_experiment.ipynb的作用：
负责选择音频、调用工具函数、播放clean和noisy、保存音频、重新读取并验证结果，以及组织实验记录。

parent_uid与合成音频的关系：
parent_uid从manife.csv读取，标识合成音频的来源。noise_utils.py不生成新的parent_uid，也不负责管理数据来源。

## 4. 实验结果

原始音频采样率：
32000Hz

目标SNR：10 dB。

浮点数组实测SNR：10.0 dB。

共同缩放系数：1.0。

写盘后实测SNR：10.0 dB。

是否存在削波：不存在

相同seed是否可复现：可以复现

## 5. 异常处理与实验限制

为什么不能对零功率信号计算目标SNR：
当clean功率为0时，SNR不是有限值，没有意义

为什么不能直接clip含噪波形：
直接截断超出范围的采样点会改变波形，并可能改变实际SNR。当前实验采用对clean和noisy共同缩放的方式降低峰值。

为什么写盘后仍需检查：
保存格式可能涉及量化、数值范围限制或其他处理。重新读取并测量，才能验证最终文件是否仍符合实验要求。

为什么synthetic不能作为自然noisy样本：
synthetic是人工添加噪声得到的合成数据，而自然noisy样本来自真实环境录制。两者的噪声分布、背景特征和生成过程可能不同，不能混为一谈。

## 6. 当前理解

- add_gaussian_noise_at_snr与measure_snr_db的区别：
前者负责生成目标SNR的含噪语音，后者负责测量已有clean和noisy波形之间的实际SNR。一个负责生成，一个负责验证。

- 为什么功率计算要使用均方值：
因为功率和振幅平方相关，波形的正负振幅不能直接通过均值反映其能量

- 如何保证噪声实验可复现：
固定seed，并记录输入来源、目标SNR、采样率、噪声生成方式和软件环境等参数。

- 为什么今天只修改train派生数据，不修改test_candidates：
今天的目的是验证加噪工具的实现，不是创建新的正式测试样本。将合成噪声混入正式测试集会改变测试数据的性质，也可能使后续评估产生混淆。


# W5D6 模块3：实验记录与最终验收

## 1. 实验记录

结果文件：

`results/w05_snr.csv`

本次实际测试音频数量：1条原始音频。

本次实验记录总数：2条。

实验条件：

- `clean`：原始干净语音，不添加噪声。
- `synthetic`：在干净语音上添加目标SNR为10 dB的高斯白噪声。

本次只有一条原始音频，因此生成两条条件记录，而不是两个独立的原始样本。

## 2. 来源追踪

parent_uid：

`common_voice_zh-CN_27897382`

source_split：

`train`

source_audio_path：

`common_voice_zh-CN_27897382.mp3`

输出音频路径：

- clean：`data/derived/w05_snr/demo_clean.wav`
- synthetic：`data/derived/w05_snr/demo_synthetic_10db.wav`

为什么两条实验记录可以共享parent_uid：

parent_uid表示原始音频的来源标识。clean是原始干净语音，synthetic是在同一条原始语音上添加噪声得到的派生样本，因此两条记录应当共享同一个parent_uid。

parent_uid用于追踪来源，不是每个派生文件独有的唯一标识。output_path用于定位具体文件，condition、seed和target_snr_db等字段用于区分实验条件。

## 3. SNR验证

目标SNR：

10 dB。

写盘后实测SNR：

约10.000000002 dB，CSV中显示为10.0 dB。

目标与实测的误差：

约0.000000002 dB。

是否通过0.5 dB容差：

是。

最大绝对幅值：

约0.416684。

是否通过峰值检查：

是。最大绝对幅值低于0.99上限，未触发共同缩放，peak_scale为1.0。

写盘回读后，clean和synthetic的形状与采样率一致，SNR和峰值检查通过。

## 4. 可复现性

随机seed：

2026。

相同seed重复生成是否一致：

是。相同输入、参数和seed重复生成的数组完全一致。

是否记录噪声生成方式：

是，noise_type为`gaussian_white`。

是否保存输出音频：

是。clean与synthetic音频均已保存，并且可以通过output_path定位。

实验记录同时保存了parent_uid、source_split、source_audio_path、condition、target_snr_db、measured_snr_db、seed、sample_rate、peak_scale和output_path等字段。

需要注意：固定seed有助于复现实验，但跨软件版本或不同计算环境时，不一定保证所有结果逐位一致。

## 5. 最终验收

- 实验表是否成功创建：是，`results/w05_snr.csv`已创建。
- clean和synthetic是否均有记录：是，共2条。
- parent_uid与source_split是否正确：是，来源为同一条train音频。
- 实测SNR是否通过：是，与10 dB目标的误差小于0.5 dB。
- 输出文件是否存在：是，两个输出路径检查通过。
- 原始数据与测试候选是否保持分离：本次实验代码仅写入派生音频和结果CSV，没有向manifest.csv或test_candidates.csv写入数据。

说明：由于没有在实验开始前记录原始文件的哈希基线，不能仅凭本次检查证明这些文件在整个操作期间从未发生变化。

## 6. 本模块总结

### 为什么clean的SNR应留空？

本次clean条件没有添加噪声，不存在本实验定义下的有限目标SNR。将其填写为10 dB会错误地表示clean也经过了10 dB加噪处理，因此目标SNR、实测SNR和seed均留空。

### 为什么必须记录measured_snr_db？

target_snr_db表示希望生成的目标值，measured_snr_db表示实际生成并保存后测量得到的结果。二者可能因为处理流程、量化或其他误差而不同，因此不能直接用目标值代替实测值。

### 为什么合成噪声不能冒充自然noisy测试样本？

合成噪声是在干净语音上通过程序人工添加的噪声；自然noisy语音来自真实环境录制。两者的噪声分布、背景特征和生成过程可能不同，因此必须明确区分，不能将合成样本当作自然噪声测试数据。

### 从manifest到noise_utils.py再到实验CSV的数据流

1. 从`manifest.csv`中选择一条train音频，读取audio_path和parent_uid。
2. 使用音频读取工具加载干净语音。
3. 调用`noise_utils.py`生成目标SNR为10 dB的synthetic语音。
4. 检查峰值，必要时对clean和noisy共同缩放。
5. 保存clean和synthetic WAV文件。
6. 重新读取保存的文件，测量实际SNR并检查峰值。
7. 将来源、实验条件、随机seed、实测结果和输出路径写入`results/w05_snr.csv`。

### 本次实验尚未验证的限制

- 目前只对一条train音频进行了完整实验。
- 尚未验证不同语音时长、不同信号功率和不同seed下的批量处理情况。
- 尚未验证整数PCM格式写盘后的量化误差。
- 本实验使用的是合成高斯白噪声，不能代表真实环境中的全部噪声类型。
- 当前结果验证了本次实验流程，不代表所有音频输入和处理条件都已经通过测试。

## 7. 最终结论

本次实验表包含1条原始音频对应的clean与synthetic两条记录。synthetic目标SNR为10 dB，写盘后实测值约为10.000000002 dB，峰值保护、随机seed复现和写盘回读检查均通过。

W5D6模块3验收通过，本日MUST任务完成。