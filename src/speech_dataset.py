from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(
            self,
            manifest_path,
            audio_root,
            split="train",
            sample_rate=16000
    ):
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate

        df = pd.read_csv(manifest_path)

        required = {"audio_path", "split", "parent_uid"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"manifest缺少字段:{missing}")

        #只读取指定数据集划分
        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError(
                f"split={split!r}中没有样本"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = self.audio_root / str(
            row["audio_path"]
        )

        if not audio_path.is_file():
            raise FileExistsError(
                f"音频文件不存在:{audio_path}"
            )

        #统一为单声道、指定采样率
        waveform,_ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True
        )

        waveform = np.asarray(
            waveform,
            dtype=np.float32
        )

        audio = torch.from_numpy(waveform.copy())

        #重采样后的实际采样点数
        length = audio.numel()

        if length == 0:
            raise ValueError(f"音频为空：{audio_path}")

        return {
            "audio":audio,
            "length":length,
            "uid": str(row["parent_uid"]),
            "text": (
                str(row["text"])
                if "text" in self.df.columns
                and pd.notna(row["text"])
                else None
            )
        }

def collate_fn(samples):
    """
将多条不同长度的音频组成一个 batch。

Returns:
    audio:        [B, T_max]
    lengths:      [B]
    padding_mask: [B, T_max]
    uid:          list
    text:         list
"""

    if not samples:
        raise ValueError("samples不能为空")

    #1.获取每条音频的真实长度
    lengths = torch.tensor(
        [sample["length"] for sample in samples],
        dtype=torch.long
    )

    #2.找到当前batch中的最大长度
    max_length = int(lengths.max().item())

    batch_size = len(samples)

    #3.创建全零矩阵
    audio_batch = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32
    )

    #4.将每条音频放入矩阵，剩余位置自动补零
    for i, sample in enumerate(samples):
        audio = sample["audio"]

        if audio.ndim != 1:
            raise ValueError("音频必须是一维Tensor")

        if audio.numel() != sample["length"]:
            raise ValueError("音频长度与记录的length不一致")

        audio_batch[i, :sample["length"]] = audio

    #5.构造padding mask
    positions = torch.arange(max_length)

    padding_mask = (
        positions.unsqueeze(0) 
        >= lengths.unsqueeze(1)
    )

    #6.返回batch
    return {
        "audio": audio_batch,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "uid": [sample["uid"] for sample in samples],
        "text": [sample["text"] for sample in samples]
    }