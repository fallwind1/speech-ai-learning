import pandas as pd
from pathlib import Path

def build_manifest(tsv_path, split):
    df = pd.read_csv(
        tsv_path,
        sep="\t"
    )

    manifest = pd.DataFrame()

    manifest["audio_path"] = df["path"]

    manifest["speaker_id"] = df["client_id"]

    manifest["text"] = df["sentence"]

    manifest["source_id"] = "common_voice_zh-CN_17"

    manifest["parent_uid"] = df["path"].str.replace(".mp3","")

    manifest["start_time"] = 0.0

    manifest["end_time"] = None

    manifest["split"] = split

    return manifest

def check_manifest(manifest):
    print(
        "Total samples:",
        len(manifest)
    )

    print(
        manifest.groupby("split").size()
    )

    for split in ["train","dev","test"]:
        speakers = manifest[manifest["split"] == split]["speaker_id"].nunique()

        print(split,"speakers:",speakers)

    train_speakers = set(
        manifest[manifest["split"] == "train"].speaker_id
    )

    test_speakers = set(
        manifest[
            manifest.split=="test"
        ]
        .speaker_id
    )

    dev_speakers = set(
        manifest[manifest["split"] == "dev"].speaker_id
    )

    print(
        "train/test overlap:",
        train_speakers & test_speakers
    )

    print(
            "train/dev overlap:",
            train_speakers & dev_speakers
        )

    print(
            "dev/test overlap:",
            dev_speakers & test_speakers
        )

root = Path("data/raw/common_voice_beijing26/zh-CN-beijing")

train = build_manifest(root/"train.tsv","train")

test = build_manifest(root/"test.tsv","test")

dev = build_manifest(root/"dev.tsv","dev")

manifest = pd.concat([train,dev,test])

print(manifest.sample(1))

check_manifest(manifest)

manifest.to_csv(
    "metadata/manifest.csv",
    index=False
)