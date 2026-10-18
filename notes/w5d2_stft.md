# W5D2 模块1：STFT 输出形状与 center/padding

## 1. STFT 输出矩阵

### STFT 的基本处理过程：
STFT（Short-Time Fourier Transform，短时傅里叶变换）通过滑动窗口对信号进行分帧，并对每一帧进行加窗和FFT。
基本流程：

原始音频
↓
分帧（Framing）
↓
加窗（Windowing）
↓
每帧进行FFT
↓
组合各帧频谱
↓
STFT矩阵

### STFT 输出矩阵的含义：
对于单声道一维音频，librosa.stft返回二维复数矩阵：

$$
D\in\mathbb{C}^{F\times T}
$$

其中：

- F：频率点数；
- T：时间帧数。

每一列表示某一个时间帧的复数频谱。

每一行表示某个频率点在不同时间帧上的频谱变化。


频率轴代表：

频率轴对应不同的FFT频点。

第k个频点的实际频率为：

$$
f_k=\frac{k\cdot sr}{n_{\mathrm{fft}}}
$$

其中sr为采样率。


时间轴代表：

时间轴对应不同的分析帧。

相邻时间帧的间隔由hop_length决定：

$$
\Delta t=\frac{hop\_length}{sr}
$$

当center=True时，第m帧以原始信号的m×hop_length采样位置为中心。

当center=False时，第m帧的n_fft点分析块从原始信号的m×hop_length位置开始。

## 2. STFT 的频点数

### n_fft 的作用：
n_fft表示每一帧进行FFT时使用的变换点数。

### 频点数公式：
对于实数音频，librosa.stft使用实数FFT。

频点数为：

$$
F=\left\lfloor\frac{n_{\mathrm{fft}}}{2}\right\rfloor+1
$$

当n_fft为偶数时：

$$
F=\frac{n_{\mathrm{fft}}}{2}+1
$$

例如：

$$
n_{\mathrm{fft}}=512
$$

则：

$$
F=257
$$

### 频率网格间距公式：
$$
\Delta f=\frac{sr}{n_{\mathrm{fft}}}
$$

例如：

$$
sr=16000,\quad n_{\mathrm{fft}}=512
$$

则：

$$
\Delta f=31.25\text{ Hz}
$$

需要注意：

频点间距不等于实际频率分辨能力。

如果只增加n_fft，而不增加真实观测时长，只会细化频率采样网格。

## 3. STFT 的帧数

### center=False 时：
librosa不会为了让分析帧居中而在原始音频两端进行padding。

每帧使用长度为n_fft的分析块。

假设：

- N：原始音频采样点数；
- H：hop_length；
- n_fft：FFT变换点数。

### 第一帧的采样范围：
使用Python切片表示：

[0:n_fft]

例如：

n_fft=512

则第一帧为：

[0:512]

包含x[0]至x[511]。

### 第 m 帧的采样范围：
$$
[mH:mH+n_{\mathrm{fft}}]
$$

其中m从0开始。

例如：

H=160，n_fft=512。

第0帧：

[0:512]

第1帧：

[160:672]

第2帧：

[320:832]

### 帧数计算公式：
为了保证完整的n_fft点分析块位于原始音频范围内，需要满足：

$$
mH+n_{\mathrm{fft}}\le N
$$

因此，当N大于或等于n_fft时：

$$
T=1+\left\lfloor\frac{N-n_{\mathrm{fft}}}{H}\right\rfloor
$$

例如：

$$
N=32000
$$

$$
n_{\mathrm{fft}}=512
$$

$$
H=160
$$

则：

$$
T=1+\left\lfloor\frac{32000-512}{160}\right\rfloor
$$

$$
T=197
$$

频点数为257，因此：

$$
D.shape=(257,197)
$$

## 4. center 与 padding

### center=True 的含义：
center=True表示第m帧的n_fft点分析块以原始音频的m×hop_length位置为中心。

例如：

hop_length=160

则：

第0帧中心位置：0

第1帧中心位置：160

第2帧中心位置：320

需要注意：

这是分析块在原始音频坐标中的位置定义，并不是先生成center=False的帧，再移动已经生成的帧。


### center=False 的含义：
center=False表示每一帧的n_fft点分析块直接从原始音频的m×hop_length位置开始。

例如：

第0帧从0开始；

第1帧从160开始；

第2帧从320开始。

### center=True 时为什么需要 padding：
因为分析块以指定位置为中心，靠近音频起点或终点的帧可能超出原始音频边界。

例如：

n_fft=512

第0帧中心位置为0。

对应512点分析块的原始坐标范围为：

-256～255

其中-256～-1并不存在于原始音频中。

因此，需要根据pad_mode对边界进行填充。

当：

pad_mode="constant"

且填充值为0时，边界使用零填充。

对于偶数n_fft，左右各填充：

$$
P=\frac{n_{\mathrm{fft}}}{2}
$$

例如：

$$
P=256
$$


### center=True 时的帧数公式：
对于偶数n_fft：

补零后总长度：

$$
N_{\mathrm{pad}}=N+n_{\mathrm{fft}}
$$

因此：

$$
T=1+\left\lfloor
\frac{N_{\mathrm{pad}}-n_{\mathrm{fft}}}{H}
\right\rfloor
$$

化简得到：

$$
T=1+\left\lfloor\frac{N}{H}\right\rfloor
$$

例如：

$$
N=32000,\quad H=160
$$

则：

$$
T=1+\frac{32000}{160}=201
$$

因此：

$$
D.shape=(257,201)
$$

注意：

上述简化公式针对本次偶数n_fft的情况。

### pad_mode 的作用：
常见方式：

1. constant：使用常数填充，默认常数为0。
2. reflect：通过反射边界附近的采样值进行填充。

不同pad_mode可能影响边界帧的频谱数值。

但在填充长度相同的情况下，通常不改变STFT输出shape。

## 5. win_length 与 n_fft 的关系

win_length：表示实际使用的窗函数长度。

n_fft：表示每帧FFT的计算点数。

### 为什么计算帧数时不能简单用 win_length 代替 n_fft：
因为librosa.stft以n_fft点作为每帧的分析块长度。

对于center=False，要求整个n_fft点分析块不能越过原始音频边界。

所以帧数公式使用n_fft，而不是win_length。


### 窗口补零与音频边界补零的区别：
第一种：窗函数补零。

发生原因：

$$
win\_length<n_{\mathrm{fft}}
$$

操作对象：

窗函数的权重数组。

例如：

400点Hann窗左右各补56个零权重，形成512点窗函数。

目的：

使窗函数长度与FFT分析块长度匹配。

它不要求center=False，也不要求center=True。


第二种：音频边界补零。

发生原因：

center=True时，需要让靠近边界的分析帧也能够以指定原始音频采样位置为中心。

操作对象：

原始音频的边界。

例如：

n_fft=512时，原始音频左右各补256个采样位置。

目的：

允许边界附近的分析帧使用补充的音频值。


两种补零互不冲突，可以同时存在。



## 6. 参数计算示例

sr：

$$
sr=16000\text{ Hz}
$$


音频采样点数：

$$
N=16000
$$

即1秒音频。


win_length：

$$
win\_length=400
$$


hop_length：

$$
hop\_length=160
$$


n_fft：

$$
n_{\mathrm{fft}}=512
$$


频点数：

$$
F=\frac{512}{2}+1=257
$$


center=False 的帧数：

$$
T_{\mathrm{false}}
=1+\left\lfloor\frac{16000-512}{160}\right\rfloor
$$

$$
T_{\mathrm{false}}=97
$$


center=True 的帧数：

原始音频左右各补256点。

补零后总长度：

$$
N_{\mathrm{pad}}=16000+512=16512
$$

因此：

$$
T_{\mathrm{true}}
=1+\left\lfloor\frac{16512-512}{160}\right\rfloor
$$

$$
T_{\mathrm{true}}=101
$$


最终 shape：

center=False：

$$
\boxed{(257,97)}
$$

center=True：

$$
\boxed{(257,101)}
$$


## 7. 当前理解

- STFT 输出矩阵两个维度分别表示什么：
对于单声道音频：
$$
D.shape=(F,T)
$$
F表示频率点数，T表示时间帧数。
由于STFT对多个时间帧分别执行FFT，因此输出比整段FFT多一个时间维度。每一列表示一个时间帧的频谱。

- 为什么 center 会影响帧数：
因为center=False不对原始音频进行居中边界填充，只有能够完整落在原始音频内部的n_fft点分析块才参与计算。
center=True则通过音频边界padding，使以原始音频不同采样位置为中心的分析块可以覆盖边界附近区域。
因此，center=True通常会产生更多时间帧。

- 为什么 win_length 与 n_fft 不相等时需要特别注意：
因为win_length决定窗函数覆盖的长度，而n_fft决定FFT分析块的长度。
当win_length小于n_fft时，librosa会对窗函数进行居中补零。

因此：
有效加窗长度不等于FFT分析块长度。
在librosa.stft中，帧数计算需要根据n_fft和center设置确定，不能直接将win_length代入帧数公式.

- 如何验证理论 shape 与实际 shape 一致：
首先根据参数计算：

$$
F=\left\lfloor\frac{n_{\mathrm{fft}}}{2}\right\rfloor+1
$$

然后根据center设置计算时间帧数。

center=False：

$$
T=1+\left\lfloor\frac{N-n_{\mathrm{fft}}}{H}\right\rfloor
$$

center=True（偶数n_fft）：

$$
T=1+\left\lfloor\frac{N}{H}\right\rfloor
$$

最后执行：

`D = librosa.stft(x, ...)`

读取：

`D.shape`

并通过：

`assert D.shape == (F, T)`

验证结果一致。


核心总结：

1. win_length：决定窗函数覆盖多少个采样位置。
2. hop_length：决定相邻帧的起始位置或中心位置相隔多少个采样点。
3. n_fft：决定FFT分析块的长度、频率网格间距以及输出频点数。
4. center=False：分析块从mH位置开始，不进行居中边界padding。
5. center=True：分析块以mH位置为中心，需要进行音频边界padding。
6. 窗函数补零与音频边界补零是两个不同操作，可以同时存在。


# W5D2 模块2：真实语音语谱图与 features.py

## 1. 从 STFT 到语谱图

STFT 输出：
对于单声道音频，STFT返回二维复数矩阵：

$$
D\in\mathbb{C}^{F\times T}
$$

其中：
- F：频率点数；
- T：时间帧数。

每个复数值包含对应时间帧和频点的幅度、相位信息。

复数形式可以表示为：

$$
D[k,m]=|D[k,m]|e^{j\phi[k,m]}
$$

幅度谱：
对复数STFT取模：

$$
M[k,m]=|D[k,m]|
$$

得到实数幅度谱。

幅度谱表示每个时间—频率位置上的频谱幅度。

如果计算：

$$
P[k,m]=|D[k,m]|^2
$$

则得到功率谱的一种常用表示。

dB 频谱：
由于频谱中不同分量的幅度可能相差多个数量级，直接绘制线性幅度谱时，较弱分量不容易观察。

因此通常将幅度谱转换为dB：

$$
S_{\mathrm{dB}}[k,m]
=
20\log_{10}
\left(
\frac{M[k,m]}{M_{\mathrm{ref}}}
\right)
$$

其中M_ref是参考幅度。

当使用幅度谱最大值作为参考时，最强位置对应0 dB。

实际编程中可以使用：

`librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80)`

其中top_db=80表示将显示的动态范围限制在峰值以下80 dB。


语谱图三个视觉维度分别表示：
- 横轴：时间；
- 纵轴：频率；
- 颜色：相应时间—频率位置上的幅度、功率或其dB表示。

为什么语谱图比整段 FFT 更适合分析语音：
真实语音整体通常是非平稳信号，频率成分会随时间变化。
整段FFT只能得到整段信号的频域表示，整段幅度谱无法直接展示不同频率出现的时间。
STFT通过分帧和逐帧FFT，为每个频谱建立时间位置。

因此，语谱图能够观察：
- 语音活动出现的时间；
- 不同频率成分随时间的变化；
- 有声语音的部分谐波结构；
- 不同时间段的频谱分布差异。

## 2. librosa.stft 实现

输入数据格式：
本项目的compute_stft函数接受一维NumPy单声道音频：

$$
waveform.shape=(N,)
$$

其中N为音频采样点数。

使用soundfile读取音频时：

`sf.read(..., always_2d=True)`

返回：

$$
(T,C)
$$

这里T是采样点数，C是声道数。

对于已经转换为单声道的音频：

$$
(T,1)
$$

可以使用：

`waveform = audio.squeeze(axis=1)`

删除大小为1的声道维度，得到：

$$
(T,)
$$

需要注意：

- squeeze是删除大小为1的维度;
- unsqueeze是新增大小为1的维度。

如果需要兼容多声道输入，可以使用：
`audio.mean(axis=1)`
进行简单的声道平均。

使用的参数：
- n_fft=512；
- win_length=400；
- hop_length=160；
- window="hann"；
- center=True；
- pad_mode="constant"。

输出数据格式：
对于单声道音频：

$$
D.shape=(F,T)
$$

D为二维复数矩阵。

F是频点数；

T是时间帧数。


每帧的频点数：
$$
F=\left\lfloor\frac{n_{\mathrm{fft}}}{2}\right\rfloor+1
$$

当前：

$$
F=\frac{512}{2}+1=257
$$

时间帧数：
当前使用center=True，且n_fft为偶数。

因此：

$$
T=1+\left\lfloor\frac{N}{hop\_length}\right\rfloor
$$

实际音频：

$$
N=47616
$$

所以：

$$
T=1+\left\lfloor\frac{47616}{160}\right\rfloor
=298
$$

最终：

$$
D.shape=(257,298)
$$

## 3. features.py 的作用

为什么要封装特征提取函数：
将特征计算封装为独立模块，可以避免在不同Notebook和训练脚本中重复编写相同逻辑。

项目中不同模块负责不同任务：

audio_utils.py：

- 音频读取；
- 声道转换；
- 重采样；
- 音频保存。

features.py：

- STFT计算；
- 幅度谱或功率谱计算；
- dB频谱转换；
- 后续扩展Mel频谱等特征。

STFT 函数的输入与输出：
函数：

`compute_stft(waveform, sr, n_fft, win_length, hop_length, center)`

输入：

一维单声道NumPy数组：

$$
waveform.shape=(N,)
$$

输出：

二维复数STFT矩阵：

$$
D.shape=(F,T)
$$


dB 频谱函数的输入与输出：
函数：

`compute_log_magnitude_db(D, top_db=80)`

输入：

二维复数STFT矩阵：

$$
D.shape=(F,T)
$$

处理：

首先计算：

$$
M=|D|
$$

再转换为dB：

$$
S_{\mathrm{dB}}=20\log_{10}(M/M_{\mathrm{ref}})
$$

实际使用librosa.amplitude_to_db完成稳定的数值转换。

输出：

二维实数dB幅度谱：

$$
S_{\mathrm{dB}}.shape=(F,T)
$$

因此，转换前后的shape相同。

为什么将特征计算与绘图分开：
因为模型训练并不一定需要绘图。

不同任务可能需要不同特征：

- 复数STFT；
- 线性幅度谱；
- 功率谱；
- 对数幅度谱；
- Mel频谱。

将STFT计算和dB转换分开，可以保留原始复数谱，方便后续扩展。

同时，将绘图留在Notebook中，可以避免特征提取函数与matplotlib等可视化逻辑直接耦合。



## 4. shape 验证

理论频点数公式：
$$
F=\left\lfloor\frac{n_{\mathrm{fft}}}{2}\right\rfloor+1
$$

center=False 的理论帧数公式：

当原始音频长度满足N≥n_fft时：

$$
T_{\mathrm{false}}
=
1+\left\lfloor
\frac{N-n_{\mathrm{fft}}}{hop\_length}
\right\rfloor
$$

该公式使用n_fft，而不是win_length。

原因：

librosa.stft按长度为n_fft的分析块进行分帧。
当win_length小于n_fft时，窗函数会居中补零到n_fft长度。


center=True 的理论帧数公式：
当n_fft为偶数时：

$$
T_{\mathrm{true}}
=
1+\left\lfloor
\frac{N}{hop\_length}
\right\rfloor
$$

原因：

center=True将第m帧的分析块中心放在原始音频的m×hop_length位置。

为了让边界处也能完成分析，需要在原始音频两端进行padding。


实际运行的 shape：
STFT shape：

(257,298)

dB谱 shape：

(257,298)

验证结果：
与理论一致

## 5. 真实语音实验观察

音频采样率：

$$
sr=16000\text{ Hz}
$$


音频时长：

实际采样点数：

$$
N=47616
$$

因此：

$$
T_{\mathrm{audio}}
=
\frac{47616}{16000}
=
2.976\text{ s}
$$


波形观察：

根据之前对同一音频绘制的时域波形，可以观察到：

- 音频总长约3秒；
- 主要语音活动集中在约0.8–2.0秒；
- 其他时间段的幅度较小；
- 不同语音片段的振幅明显变化。


语谱图观察：

本次使用librosa.stft绘制真实语音语谱图。

图像横轴为时间，纵轴为频率，颜色表示相对dB幅度。

从实际图像观察到：

1. 主要语音活动集中在约0.8–2.0秒。
2. 低频区域存在较强的频谱成分。
3. 在较高频率区域也存在随时间变化的频谱结构。
4. 不同时间位置的频谱明显不同。
5. 静音或低幅度区域在语谱图中通常表现为较暗的颜色。

本次图像显示频率范围：

0–4000 Hz。

需要注意：

由于采样率为16000 Hz，完整STFT的最高非负频率为8000 Hz。

图像只显示到4000 Hz是绘图范围限制，并不是STFT只计算到4000 Hz。

## 6. STFT 参数与语谱图

改变 win_length：
win_length控制窗函数覆盖的采样点数。

当采样率固定时：

窗长增大：

- 有效分析时间范围增大；
- 通常有利于区分相近频率；
- 可能增加对快速时间变化的模糊。

窗长减小：

- 单次分析覆盖的时间范围减小；
- 通常有利于定位快速变化；
- 频率分辨能力可能下降。


改变 hop_length：
hop_length控制相邻时间帧的位置间隔。

hop_length减小：

- 帧与帧之间的时间间隔缩短；
- 时间帧数通常增加；
- 语谱图时间方向采样更密集；
- 计算量通常增加。

hop_length增大：

- 时间帧数通常减少；
- 时间方向的采样更稀疏；
- 计算量通常减少。

改变 n_fft：
n_fft控制每帧FFT变换点数。

频率网格间距：

$$
\Delta f=\frac{sr}{n_{\mathrm{fft}}}
$$

n_fft增大：

- 频点数增加；
- 频率采样网格变密；
- 如果win_length不变，主要体现补零效果。


改变 center：
center=False：

每帧的n_fft点分析块从m×hop_length位置开始。
不进行居中边界padding。

center=True：

每帧的n_fft点分析块以m×hop_length位置为中心。
靠近音频边界时需要padding。

因此：

center会影响时间帧数、帧与原始音频的对应位置以及边界处的频谱值。
在其他参数相同时，center通常不会改变频点数。

## 7. 当前理解

- 为什么 librosa.stft 返回复数矩阵：
因为傅里叶变换使用复指数基函数计算频率分量，结果包含幅度和相位信息。

可以表示为：

$$
D[k,m]=|D[k,m]|e^{j\phi[k,m]}
$$
因此STFT返回复数矩阵。

- 为什么需要将复数 STFT 转换为幅度谱或 dB 频谱：
语谱图通常需要使用一个实数来表示每个时间—频率位置的强弱。

所以可以计算复数谱的模：
$$
|D[k,m]|
$$
或者模平方：
$$
|D[k,m]|^2
$$
得到幅度谱或功率谱。
进一步转换为dB尺度，可以更清晰地观察相差多个数量级的频谱成分。

- 如何根据参数预测 STFT shape：
首先计算频点数：

$$
F=\left\lfloor\frac{n_{\mathrm{fft}}}{2}\right\rfloor+1
$$

然后根据center设置计算时间帧数。

center=False：

$$
T=1+\left\lfloor\frac{N-n_{\mathrm{fft}}}{H}\right\rfloor
$$

center=True（n_fft为偶数）：

$$
T=1+\left\lfloor\frac{N}{H}\right\rfloor
$$

最终：

$$
D.shape=(F,T)
$$

通过实际执行librosa.stft，再使用assert检查理论shape与实际shape是否一致。

- features.py 在后续 ASR 项目中的作用：
features.py用于统一管理音频特征提取操作。

其作用包括：

- 避免重复编写STFT代码；
- 统一特征提取参数；
- 方便在不同实验中复用相同流程；
- 便于继续扩展Mel频谱等特征；
- 为后续Dataset、DataLoader和ASR模型提供特征提取接口。

完整流程：

原始音频
↓
audio_utils
↓
读取 / 单声道 / 重采样
↓
features.py
↓
STFT / 幅度谱 / dB频谱
↓
后续语音特征处理
↓
ASR模型


核心总结：

$$
\boxed{\text{STFT输出shape}=(\text{频点数},\text{时间帧数})}
$$

$$
\boxed{\text{n_fft主要决定频率轴}}
$$

$$
\boxed{\text{hop_length主要影响时间帧数}}
$$

$$
\boxed{\text{center决定帧的定位方式和边界处理}}
$$