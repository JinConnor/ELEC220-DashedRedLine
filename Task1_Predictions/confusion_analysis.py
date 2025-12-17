import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

#Load files
def load_json(filename):
    with open(filename, 'r') as f:
        return pd.DataFrame(json.load(f))

#Load data
df_gt = load_json('final_dataset_withpop.json')
df_claude = load_json('classifications_Claude.json')
df_gpt = load_json('classifications_GPT.json')
df_gemini = load_json('classifications_Gemini.json')

#Process true values
def get_bucket(popularity):
    if popularity <= 33: return 'LOW'
    elif popularity <= 67: return 'MID'
    else: return 'HIGH'

df_gt['true_bucket'] = df_gt['popularity'].apply(get_bucket)

models = {'Claude': df_claude, 'GPT': df_gpt, 'Gemini': df_gemini}
labels = ['LOW', 'MID', 'HIGH']

plt.rcParams["font.family"] = "serif"

#Create Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (model_name, df_pred) in enumerate(models.items()):
    merged = pd.merge(df_gt, df_pred[['track_id', 'bucket']], on='track_id', how='inner')
    
    y_true = merged['true_bucket']
    y_pred = merged['bucket']
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    #Plot heatmap
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=labels, yticklabels=labels)
    
    axes[i].set_title(f'{model_name}\nAcc: {acc:.2f}, Macro-F1: {f1:.2f}',size=18)
    axes[i].set_xlabel('Predicted Label', size=14)
    axes[i].set_ylabel('True Label', size=14)
    # axes[i].set_xmargin(3000)

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()