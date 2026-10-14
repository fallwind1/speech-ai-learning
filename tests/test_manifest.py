import pandas as pd

df = pd.read_csv("metadata/manifest.csv")

print(df.head())
print(df.columns)