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