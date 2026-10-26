import numpy as np
import librosa

def compute_stft(
        waveform: np.ndarray,
        sr: int,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        center: bool = True
) -> np.ndarray:
    """
    对单声道音频计算STFT。

    Args:
        waveform: 一维音频数组，shape=(N,)
        sr: 采样率
        n_fft: FFT计算点数
        win_length: 窗函数长度
        hop_length: 帧移
        center: 是否使用居中帧

    Returns:
        复数STFT矩阵，shape=(F, T)
    """
    waveform = np.asarray(waveform)

    if waveform.ndim != 1:
        raise ValueError("waveform必须是一维单声道音频")

    if waveform.size == 0:
        raise ValueError("waveform不能为空")

    if sr <= 0:
        raise ValueError("sr必须为正数")

    if not (0 < win_length <= n_fft):
        raise ValueError("需要满足0 < win_length <= n_fft")

    if hop_length <= 0:
        raise ValueError("hop_length必须为正数")

    if not center and len(waveform) < n_fft:
        raise ValueError("center=False时，音频长度不能小于n_fft")

    waveform = waveform.astype(np.float32, copy=False)

    D = librosa.stft(
        waveform,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hann",
        center=center,
        pad_mode="constant"
    )

    return D

def compute_log_magnitude_db(
        D: np.ndarray,
        top_db: float = 80.0
) -> np.ndarray:
    """
    将复数STFT转换为相对dB幅度谱。

    Args:
        D: 复数STFT矩阵，shape=(F,T)
        top_db: 最大值以下保留的dB范围

    Returns:
        dB幅度谱，shape=(F,T)
    """
    if D.ndim != 2:
        raise ValueError("D必须是二维STFT矩阵")

    magnitude = np.abs(D)

    S_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
        top_db=top_db
    )

    return S_db

def hz_to_mel(frequencies, *, scale="htk"):
    """
    Hz 转换为 Mel。

    scale:
        "htk"    : HTK Mel scale
        "slaney" : Slaney Mel scale
    """
    f = np.asarray(frequencies, dtype=np.float64)

    if not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("frequencies必须是非负有限值")

    if scale == "htk":
        mel = 2595.0 * np.log10(1.0 + f / 700.0)

    elif scale == "slaney":
        #1000Hz以下：线性
        f_sp = 200.0 / 3.0
        linear_mel = f / f_sp

        #1000Hz以上：对数
        logstep = np.log(6.4) / 27.0
        log_mel = (
            15.0
            + np.log(np.maximum(f, 1000.0) / 1000.0)
            / logstep
        )

        mel = np.where(f < 1000.0, linear_mel, log_mel)
    
    else:
        raise ValueError("scale 必须是 'htk' 或 'slaney'")

    return mel.item() if mel.ndim == 0 else mel


def waveform_to_mel(
        waveform,
        sr:int,
        *,
        n_fft:int=1024,
        win_length:int=800,
        hop_length:int=160,
        window:str="hann",
        center:bool=True,
        pad_mode:str="constant",
        n_mels:int=80,
        fmin:float=0.0,
        fmax:float | None = None,
        htk:bool=False,
        norm:str="slaney",
        eps:float=1e-10
)->np.ndarray:
    """
    教学版 waveform -> Log-Mel 功率谱（dB）。

    固定处理约定：
    - 输入：一维单声道 waveform，采样率由 sr 指定
    - 功率谱：abs(STFT) ** 2
    - Mel：librosa.filters.mel
    - Log：10 * log10(max(mel_power, eps))
    - dB 功率参考值：1.0
    - 不使用 top_db 截断
    - 不进行自动重采样、补齐固定时长或 Whisper 特殊归一化

    返回形状：(n_mels, n_frames)
    """
    y = np.asarray(waveform, dtype=np.float64)

    if y.ndim != 1 or y.size == 0:
        raise ValueError("waveform 必须为非空一维数组")

    if not np.all(np.isfinite(y)):
        raise ValueError("waveform 不能包含 NaN 或 Inf")

    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    if not (0 < win_length <= n_fft):
        raise ValueError("必须满足 0 < win_length <= n_fft")

    if hop_length <= 0 or n_mels <= 0:
        raise ValueError("hop_length 和 n_mels 必须大于 0")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps必须为正的有限数值")

    if fmax is None:
        fmax = sr / 2

    if not(0 <= fmin < fmax <= sr/2):
        raise ValueError("需要满足 0 <= fmin < fmax <= sr/2")

    #1.计算复数STFT
    D = librosa.stft(
        y=y,
        n_fft=n_fft,
        win_length = win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        pad_mode=pad_mode
    )

    #2.功率谱，形状(F,T)
    power_spec = np.abs(D) ** 2

    #3.Mel滤波器组，形状(M,F)
    mel_filter = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk,
        norm=norm,
        dtype=np.float64
    )

    #4.Mel功率谱，形状(M,T)
    mel_power = mel_filter @ power_spec

    #5.功率dB,参考值为1
    log_mel = 10.0 * np.log10(
        np.maximum(mel_power, eps)
    )

    return log_mel