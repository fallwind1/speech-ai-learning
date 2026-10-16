import  pandas as pd
import matplotlib.pyplot as plt

manifest_path = "metadata/manifest.csv"

df = pd.read_csv(manifest_path)

# -------------------------
# 1. 数据集基本统计
# -------------------------

stats = (
    df.groupby("split").agg(
        clips=("audio_path","count"),
        speakers=("speaker_id","nunique")
    ).reset_index()
)

print(stats)

# -------------------------
# 2. 音频数量
# -------------------------
plt.figure(figsize=(7,4))

plt.bar(stats["split"], stats["clips"])

plt.xlabel("Split")
plt.ylabel("Number of clips")
plt.title("Number of Audio Clips")
plt.grid(axis="y")

plt.tight_layout()
plt.show()

# -------------------------
# 3. Speaker数量
# -------------------------
plt.figure(figsize=(7,4))

plt.bar(stats["split"],stats["speakers"])

plt.xlabel("Split")
plt.ylabel("Number of Speakers")
plt.title("Number of Speakers")
plt.grid(axis="y")

plt.tight_layout()
plt.show()