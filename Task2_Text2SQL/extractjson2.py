import pandas as pd

df = pd.read_csv("dataset.csv")
## df = df.drop(columns=["popularity"])

numeric_df = df.select_dtypes(include="number")
medians = numeric_df.median()
for column, median in medians.items():
    print(f"{column}: {median}")

df = df.drop(columns=["number"])
sampled_df = df.sample(n=1000, random_state=42)
sampled_df.to_json("final_dataset.json", orient="records", indent=2)
