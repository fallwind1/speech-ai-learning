import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import model_info

from src.normalize import NORMALIZER_VERSION


SCORING_VERSION = "w7d6-v1"

root = r"D:\Projects\speech-ai-learning"

if root not in sys.path:
    sys.path.insert(0, str(root))

from src.normalize import (
    normalize_score,
    mixed_tokens
)

from src.asr_inference import (
    transcribe_file
)

from src.metrics import score_corpus

def cer_tokens(text: str) -> list[str]:
     """
    CER字符级Tokenization。

    1. 统一规范化
    2. 去除计分空白
    3. 按Unicode字符划分
    """
     normalized = normalize_score(text)

     return [
          char 
          for char in normalized
          if not char.isspace()
     ]

def mer_tokens(text: str) -> list[str]:
    """
    MER混合Tokenization。

    复用mixed_tokens，
    内部已经执行normalize_score。
    """
    return mixed_tokens(text)

def load_eval_data(csv_path):
    """
    读取已经人工验证的评估数据。

    只保留 status == verified 的样本。
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    #只保留人工验证通过的样本
    df = df[
        df["status"] == "verified"
    ].copy()

    required_columns = [
        "candidate_uid",
        "audio_path",
        "raw_text",
        "score_text",
        "status"
    ]

    #检查必要字段
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "No verified samples found."
        )

    #检查空字段
    for col in [
        "candidate_uid",
        "audio_path",
        "raw_text",
        "score_text"
    ]:
        if df[col].isna().any():
            raise ValueError(
                f"{col} contains missing values."
            )

    #检查音频路径
    for audio_path in df["audio_path"]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)

    #检查保存的评分文本是否与规范化结果一致
    for _, row in df.iterrows():

        expected = normalize_score(
            row["raw_text"]
        )

        if expected != row["score_text"]:
            raise ValueError(
                f"Score text mismatch: "
                f"{row['candidate_uid']}"
            )

    #统一评估字段名称
    df = df.rename(
        columns={
            "candidate_uid": "uid",
            "raw_text": "reference",
            "score_text": "score_reference"
        }
    )

    df = df.sort_values(
        "selection_rank"
    ).reset_index(drop=True)

    return df


def run_batch_inference(
    eval_df,
    processor,
    model,
    device,
    language="zh",
    task="transcribe",
    max_new_tokens=96
):
    """
    批量执行ASR推理。

    参数：
        eval_df: 已筛选的评估数据
        processor: WhisperProcessor
        model: Whisper模型
        device: 运行设备

    返回：
        predictions: 每条音频的推理结果
    """
    if eval_df.empty:
        raise ValueError("eval_df is empty")

    results = []

    total = len(eval_df)

    for index, row in enumerate(
        eval_df.itertuples(index=False),
        start=1
    ):
        print(
            f"[{index}/{total}] "
            f"Transcribing {row.uid}"
        ) 

        #单条音频推理
        result = transcribe_file(
            audio_path=row.audio_path,
            processor=processor,
            model=model,
            device=device,
            language=language,
            task=task,
            max_new_tokens=max_new_tokens
        )

       #获取原始预测文本
        hypothesis = result["transcript"]

       # 统一规范化
        score_hypothesis = normalize_score(
            hypothesis
        )

        #保存结果
        results.append({
            "uid": row.uid,
            "audio_path": row.audio_path,

            "reference": row.reference,
            "hypothesis": hypothesis,

            "score_reference": row.score_reference,
            "score_hypothesis": score_hypothesis,

            "duration_sec": (
                result["metadata"]["duration_sec"]
            )
        })

    predictions = pd.DataFrame(results)

    # 基本检查
    assert len(predictions) == total

    assert predictions["uid"].is_unique

    assert (
        predictions["score_hypothesis"].notna().all()
    )

    return predictions

def compute_corpus_metrics(predictions):
    """
    计算语料级 CER 和 MER。

    参数：
        predictions:
            包含 score_reference、
            score_hypothesis 的 DataFrame

    返回：
        predictions_with_metrics
        corpus_metrics
    """

    if predictions.empty:
        raise ValueError("predictions is empty")

    #构建CER token pairs
    cer_pairs = [
        (
            cer_tokens(row.score_reference),
            cer_tokens(row.score_hypothesis)
        )
        for row in predictions.itertuples()
    ]

    #构建MER token pairs
    mer_pairs = [
        (
            mer_tokens(row.score_reference),
            mer_tokens(row.score_hypothesis)
        )
        for row in predictions.itertuples()
    ]

    #语料级计算
    cer_result = score_corpus(cer_pairs)
    mer_result = score_corpus(mer_pairs)

    #保留原始预测结果
    output = predictions.copy()

    # 添加逐条 CER 统计
    for key in ["S", "D", "I", "C", "N",
                "distance", "error_rate"]:

        output[f"cer_{key}"] = [
            result[key]
            for result in cer_result["sample_results"]
        ]

    #添加逐条MER统计
    for key in ["S", "D", "I", "C", "N",
                "distance", "error_rate"]:

        output[f"mer_{key}"] = [
            result[key]
            for result in mer_result["sample_results"]
        ]

    #汇总指标
    corpus_metrics = {
        "num_samples": len(predictions),

        "CER": cer_result["error_rate"],
        "CER_S": cer_result["S"],
        "CER_D": cer_result["D"],
        "CER_I": cer_result["I"],
        "CER_N": cer_result["N"],

        "MER": mer_result["error_rate"],
        "MER_S": mer_result["S"],
        "MER_D": mer_result["D"],
        "MER_I": mer_result["I"],
        "MER_N": mer_result["N"],
    }

    return output, corpus_metrics

def save_evaluation_results(
    predictions,
    corpus_metrics,
    model,
    output_dir,
    checkpoint="openai/whisper-small",
    split="dev",
    language="zh",
    task="transcribe",
    max_new_tokens=96
):
    """
    保存逐条预测结果和实验元数据。
    """
    if predictions.empty:
        raise ValueError("predictions is empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    #检查UID
    if not predictions["uid"].is_unique:
        raise ValueError("Duplicate UID found")

    #添加SPLIT字段
    output = predictions.copy()

    if "split" in output.columns:
        output["split"] = split
    else:
        output.insert(0, "split", split)

    # 保存 CSV
    csv_path = output_dir / "predictions.csv"

    output.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    #获取模型版本
    model_revision = getattr(
        model.config,
        "_commit_hash",
        None
    )

    if model_revision is None:
        model_revision = model_info(
            checkpoint
        ).sha

    # 获取实际设备
    device = str(
        next(model.parameters()).device
    )

    #构建Metadata
    metadata = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "split": split,

        "model_name": checkpoint,
        "model_revision": model_revision,

        "language": language,
        "task": task,
        "max_new_tokens": max_new_tokens,
        "device": device,

        "normalizer_version": NORMALIZER_VERSION,
        "scoring_version": SCORING_VERSION,

        "num_samples": len(output),

        "metrics": corpus_metrics
    }

    #保存JSON
    json_path = output_dir / "metadata.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("实验结果保存成功")
    print("CSV:", csv_path)
    print("JSON:", json_path)

    return csv_path, json_path