
from pathlib import Path
import random

import pandas as pd
from mutagen.mp3 import MP3


# 1. 项目及数据路径
root = Path("data/raw/common_voice_beijing26")
output = Path("metadata/source_registry.csv")

source_url = (
    "https://mozilladatacollective.com/datasets/"
    "cmruwsbag00c8md07lk1n7i1e"
)

# 各集合需要抽取的分钟数
targets = {"train": 25, "dev": 8, "test": 5}


# 2. 找到 MP3 文件夹
clip_dirs = [
    p for p in root.rglob("clips") if p.is_dir()
]
if len(clip_dirs) != 1:
    raise RuntimeError(
        f"预期找到1个clips目录，实际找到{len(clip_dirs)}个"
    )

clips_dir = clip_dirs[0]


# 3. 读取 TSV 并检查 MP3
frames = []

for split in ("train", "dev", "test"):
    matches = list(root.rglob(f"{split}.tsv"))

    if len(matches) != 1:
        raise RuntimeError(
            f"{split}.tsv 找到{len(matches)}个"
        )

    df = pd.read_csv(
        matches[0],
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df["source_split"] = split
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

# 不使用缺少匿名说话人 ID 的记录
all_df = all_df[all_df["client_id"] != ""].copy()


# 4. 检查不同 split 是否有相同说话人
speaker_sets = {
    split: set(
        all_df.loc[
            all_df["source_split"] == split,
            "client_id"
        ]
    )
    for split in targets
}

for a, b in [
    ("train", "dev"),
    ("train", "test"),
    ("dev", "test")
]:
    overlap = speaker_sets[a] & speaker_sets[b]
    print(f"{a}/{b} 说话人重叠数：{len(overlap)}")


# 5. 为每个说话人只保留一个来源 split
# 优先保留 test，其次 dev，最后 train
speaker_owner = {}

for split in ("test", "dev", "train"):
    for speaker in sorted(speaker_sets[split]):
        speaker_owner.setdefault(speaker, split)

all_df = all_df[
    all_df["client_id"].map(speaker_owner)
    == all_df["source_split"]
].copy()

# 防止重复音频
all_df = all_df.drop_duplicates(subset=["path"])


# 6. 从实际 MP3 文件读取时长
records = []
missing = []

for row in all_df.itertuples(index=False):
    audio_path = clips_dir / row.path

    if not audio_path.is_file():
        missing.append(str(audio_path))
        continue

    duration = MP3(str(audio_path)).info.length

    records.append({
        "source_id": "MDC_CV26_BEIJING",
        "clip_uid": Path(row.path).stem,
        "session_uid": "unknown",
        "speaker_id": row.client_id,
        "audio_path": audio_path.as_posix(),
        "duration_sec": round(duration, 3),
        "transcript": row.sentence,
        "source_split": row.source_split,
        "split": row.source_split,
        "license": "CC0-1.0",
        "license_evidence": source_url,
        "train_allowed": "true",
        "cloud_allowed": "pending_review",
        "redistribute_allowed": "pending_review",
        "notes": "original_session_not_provided"
    })

df = pd.DataFrame(records)

print(f"\n缺失的MP3文件数：{len(missing)}")
if missing:
    print("缺失文件示例：", missing[:3])

if df.empty:
    raise RuntimeError("没有找到可用音频")


# 7. 按说话人轮流选取，避免只抽中一个人
rng = random.Random(42)
selected = []

for split, minutes in targets.items():
    part = df[df["split"] == split]

    groups = []
    for _, group in part.groupby("speaker_id"):
        items = group.to_dict("records")
        rng.shuffle(items)
        groups.append(items)

    rng.shuffle(groups)
    total = 0.0
    target = minutes * 60

    while total < target and any(groups):
        for items in groups:
            if total >= target:
                break
            if items:
                item = items.pop()
                selected.append(item)
                total += item["duration_sec"]

    print(
        f"{split}: {total / 60:.2f}分钟"
    )


# 8. 验证筛选结果
result = pd.DataFrame(selected)

duration_min = result["duration_sec"].sum() / 60
speaker_count = result["speaker_id"].nunique()

print(f"\n总音频数：{len(result)}")
print(f"总时长：{duration_min:.2f}分钟")
print(f"独立说话人ID数：{speaker_count}")

print("\n各集合：")
print(
    result.groupby("split").agg(
        clips=("clip_uid", "count"),
        speakers=("speaker_id", "nunique"),
        seconds=("duration_sec", "sum")
    )
)

# 检查各 split 的说话人是否互斥
speaker_splits = result.groupby(
    "speaker_id"
)["split"].nunique()

assert speaker_splits.max() == 1

if not (
    30 <= duration_min <= 60
    and speaker_count >= 6
    and set(result["split"]) == {"train", "dev", "test"}
):
    raise RuntimeError(
        "未达到30-60分钟、至少6个说话人"
        "且三个split均非空的要求，请检查统计结果"
    )


# 9. 保存CSV
output.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n已保存：{output}")
