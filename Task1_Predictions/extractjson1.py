import pandas as pd

df = pd.read_csv("dataset.csv")

numeric_df = df.select_dtypes(include="number")
medians = numeric_df.median()
for column, median in medians.items():
    print(f"{column}: {median}")

sampled_df = df.sample(n=100, random_state=42)

# Create final_dataset_withpop.json
minimal_df = sampled_df[['track_id', 'track_name', 'popularity']].copy()
minimal_df.to_json("final_dataset_withpop.json", orient="records", indent=2)

# Create final_dataset.json
sampled_df.drop("popularity", axis=1, inplace=True)
sampled_df.to_json("final_dataset.json", orient="records", indent=2)