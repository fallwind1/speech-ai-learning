from pathlib import Path
import logging
import torch
import torchaudio
import soundfile as sf

# =====================
# 日志设置
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# =====================
# 读取音频
# =====================
def load_audio(path):
    """
    读取音频
    返回：
        waveform: Tensor
        sample_rate: int
    """

    try:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"audio not found : {path}"
            )

        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=True
        )

        #值转换为Pytorch的(C,T)
        waveform = torch.from_numpy(audio.T.copy())

        #waveform:
        #(channel, time)

        waveform = waveform.float()

        logger.info(
            f"Loaded {path},"
            f"shape={waveform.shape}, sr={sr}"
        )

        return waveform, sr

    except Exception as e:

        logger.error(
            f"Failed loading {path}: {e}"
        )

        raise

# =====================
# 转单声道
# =====================

def to_mono(waveform):
    """
    输入：
        (channel, time)
    输出：
        (1,time)
    """
    if waveform.dim() != 2:
        raise ValueError(
            "Expected waveform shape (C,T)"
        ) 

    if waveform.size(0) == 1:
        return waveform

    mono = waveform.mean(dim=0,keepdim=True)

    return mono

# =====================
# 重采样
# =====================

def resample_audio(
        waveform,
        orig_sr,
        target_sr
):
    """
    修改重采样
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    )

    return resampler(waveform)

# =====================
# 保存音频
# =====================
def save_audio(
        path,
        waveform,
        sample_rate
):
    """
    保存wav
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    #soundfile要求：
    #(time,) 或 (time,channel)

    audio = waveform.squeeze(0)     #squeeze去掉第0维，即将channel去掉变为(time,)

    sf.write(
        path,
        audio.numpy(),
        sample_rate
    )

    logging.info(
        f"Saved {path}"
    )

