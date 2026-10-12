from pathlib import Path
import pandas as pd

root = Path("data/raw/common_voice_beijing26")

# 查找解压后的 TSV 文件
tsv_files = sorted(root.rglob("*.tsv"))

if not tsv_files:
    raise FileNotFoundError("未找到 TSV 文件，请检查解压目录")

for path in tsv_files:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(f"\n文件：{path.name}")
    print(f"样本数：{len(df)}")
    print(f"字段：{list(df.columns)}")

    if "client_id" in df.columns:
        speakers = df.loc[
            df["client_id"] != "",
            "client_id"
        ].nunique()
        print(f"有 ID 的不同说话人数：{speakers}")

    print(df.head(3).to_string(index=False))