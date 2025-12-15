import pandas as pd

gemini = pd.read_json("popularity_buckets_Gemini.json")
claude = pd.read_json("popularity_classifications_Claude.json")
gpt = pd.read_json("popularity_classification_GPT.json")

# TOOD: add code to compare the popularity predictions with the actual popularities, along with all the other results we need for the report