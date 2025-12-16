import pandas as pd
import sklearn.metrics

# gemini = pd.read_json("popularity_buckets_Gemini.json")
# claude = pd.read_json("popularity_classifications_Claude.json")
# gpt = pd.read_json("popularity_classification_GPT.json")

# TOOD: add code to compare the popularity predictions with the actual popularities, along with all the other results we need for the report

POP_RESULTS_FILE = "final_dataset_withpop.json"
 
MODEL_FILES = {
    "Gemini": "classifications_Claude.json",
    "GPT": "classifications_GPT.json",
    "Claude": "classifications_Claude.json"
}

ORIG_DATASET_FILE = "final_dataset.json"

OUTPUT_FILE = "analysis.output"

def bucket(pop_score):
    """Gets bucket for given populartion score.

    Args:
        pop_score (int): The popularity score in the range 0-100 of the song

    Returns:
        int: Either 1, 2, or 3, depending on defined buckets.
    """
    pop_score = int(pop_score)
    if pop_score < 0:
        return "N\\A"
    elif pop_score < 34:
        return "LOW"
    elif pop_score < 68:
        return "MID"
    else:
        return "HIGH"

def hallucination_rate(data, evidence_col):
    """Returns the amount of hallucinations of data

    Args:
        model_data (DataFrame): Output data predictions by AI model
        evidence_col (str): Name of column with evidence
    
    Returns:
        int: Counts how many times the model cited a number that doesn't match the input
    """
    CUT_LIST = ["_bpm", "_db"]
    hall_cnt = 0
    check_cnt = 0
    
    for _, row in data.iterrows():
        evidence = row.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        
        for feature, cited_val in evidence.items():
            check_cnt += 1
            
            feature_cleaned = feature
            for cut_word in CUT_LIST:
                feature_cleaned = feature_cleaned.replace(cut_word, "")
            
            if feature_cleaned in row:
                real_val = row[feature_cleaned]
                #Sometimes the values are slightly rounded, check if they are too far off
                if (abs(cited_val - real_val) > 0.1):
                    hall_cnt += 1
                    # print(feature_cleaned, cited_val, real_val)
                
            
            else:
                # print("not there", feature_cleaned)
                
                hall_cnt += 1

    return hall_cnt / check_cnt
        
        
        
        
            
    
#Setup actual popularity data
actual_pop_data = pd.read_json(POP_RESULTS_FILE)
actual_pop_data['bucket_true'] = actual_pop_data["popularity"].apply(bucket)

#Setup original song dataset
orig_song_data = pd.read_json(ORIG_DATASET_FILE)

#Print table header
print(f"{'Model':<10} | {'ACC':<6} | {'F1':<6} | {'MAE':<6} | {'HALL'}")
#Iterate through each model
for model, path in MODEL_FILES.items():
    prediction_data = pd.read_json(path)
    merged_data = pd.merge(prediction_data, actual_pop_data, on="track_name", suffixes=("_pred", "_true"))
    
    if "track_id" in merged_data.columns:
        merged_data = pd.merge(orig_song_data, merged_data, on=["track_name", "track_id"])
    else:
        merged_data = pd.merge(orig_song_data, merged_data, on="track_name")
    

    
    acc = sklearn.metrics.accuracy_score(merged_data["bucket_true"], merged_data["bucket"])    
    f1 = sklearn.metrics.f1_score(merged_data["bucket_true"], merged_data["bucket"], average="macro")
    mae = sklearn.metrics.mean_absolute_error(merged_data['popularity_true'], merged_data["popularity_pred"])
    
    hall_rate = hallucination_rate(merged_data, "evidence")

    print(f"{model:<10} | {acc:<6} | {str(f1)[:6]:<6} | {mae:<6} | {hall_rate}")