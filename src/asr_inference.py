from pathlib import Path
from huggingface_hub import model_info
import torch

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from src.audio_utils import (
    load_audio,
    to_mono,
    resample_audio,
)

def load_whisper_model(
        checkpoint="openai/whisper-small",
        device=None
):
    """
    加载 Whisper Processor 和模型。

    返回：
        processor
        model
        device
    """

    #自动选择设备
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device)

    #确定模型版本
    info = model_info(checkpoint)
    model_revision = info.sha

    #加载Processor
    processor = WhisperProcessor.from_pretrained(checkpoint,revision=model_revision)

    #加载模型
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint,revision=model_revision)

    model = model.to(device)

    #切换评估模式
    model.eval()

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    return processor, model, device

def transcribe_file(
        audio_path,
        processor,
        model,
        device=None,
        language="zh",
        task="transcribe",
        max_new_tokens=96
):
    """
    对单条音频进行 ASR 推理。

    参数：
        audio_path: 音频路径
        processor: WhisperProcessor
        model: Whisper模型
        device: 运行设备
        language: 识别语言
        task: transcribe 或 translate
        max_new_tokens: 最大生成Token数量

    返回：
        dict:
            transcript
            generated_ids
            metadata
    """

    audio_path = Path(audio_path)

    #1.确定设备
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

        if next(model.parameters()).device != device:
            model = model.to(device)

    model.eval()

    #2.读取音频
    waveform, sr = load_audio(audio_path)

    #3.转换为单声道
    waveform = to_mono(waveform)

    #4.获取模型要求的采样率
    target_sr = processor.feature_extractor.sampling_rate

    #5.必要时进行真正的重采样
    if sr != target_sr:
        waveform = resample_audio(
            waveform,
            sr,
            target_sr
        )

    #6.计算音频时长
    duration = waveform.shape[-1] / target_sr

    #7.转换为Numpy一维数组
    audio_np = waveform.squeeze(0).detach().cpu().numpy()

    #8.Processor提取Log-Mel
    inputs = processor(
        audio_np,
        sampling_rate=target_sr,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(device)

    #9.Whisper推理
    with torch.inference_mode():

        generated_ids = model.generate(
            input_features=input_features,
            language=language,
            task=task,
            return_timestamps=False,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    #10.Token IDs解码为文本
    transcript = processor.batch_decode(
        generated_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    #11.整理Token IDs
    token_ids = generated_ids[0].detach().cpu().tolist()

    #12.整理元数据
    metadata = {
        "audio_path": str(audio_path),
        "checkpoint": getattr(
            model.config,
            "_name_or_path",
            "unknown",
        ),
        "original_sampling_rate": sr,
        "model_sampling_rate": target_sr,
        "duration_sec": duration,
        "device": str(device),
        "input_features_shape": tuple(
            input_features.shape
        ),
        "generated_ids_shape": tuple(
            generated_ids.shape
        )
    }

    return {
        "transcript": transcript,
        "generated_ids": token_ids,
        "metadata": metadata,
    }

if __name__ == "__main__":

    root = Path(__file__).resolve().parent.parent

    audio_path = (
        root
        / "results"
        / "w06d7_resampled_16k.wav"
    )

    # 加载模型
    processor, model, device = load_whisper_model(
        checkpoint="openai/whisper-small"
    )

    # 进行推理
    result = transcribe_file(
        audio_path=audio_path,
        processor=processor,
        model=model,
        device=device,
        language="zh",
        task="transcribe",
        max_new_tokens=96,
    )

    print("\n========== ASR Result ==========")

    print("\n转写文本:")
    print(result["transcript"])

    print("\nToken IDs:")
    print(result["generated_ids"])

    print("\nMetadata:")
    for key, value in result["metadata"].items():
        print(f"{key}: {value}")


    result_32k = transcribe_file(
        audio_path=r"D:\Projects\speech-ai-learning\data\derived\w05_snr\demo_clean.wav",
        processor=processor,
        model=model,
        device=device,
    )

    print(result_32k["metadata"])