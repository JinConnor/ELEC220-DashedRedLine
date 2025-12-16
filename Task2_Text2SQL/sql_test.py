import json
import sqlite3
import pandas as pd
import os
import re

# --- CONFIGURATION ---
DATASET_FILE = "song_dataset.json"
GOLD_STANDARD_FILE = "question_list_withanswer.json"
PREDICTION_FILES = [
    ("Gemini", "results_Gemini.json"),
    ("GPT", "results_GPT.json"),
    ("Claude", "results_Claude.json")
]

def load_gold_standard():
    """Load the gold standard SQL queries from question_list_withanswer.json."""
    if not os.path.exists(GOLD_STANDARD_FILE):
        print(f"Error: {GOLD_STANDARD_FILE} not found.")
        return None
    try:
        with open(GOLD_STANDARD_FILE, 'r') as f:
            questions = json.load(f)
        # Create map from id to sql (assuming 'sql' field exists in the file)
        return {item['id']: item for item in questions if 'sql' in item}
    except Exception as e:
        print(f"Error loading gold standard: {e}")
        print(Exception)
        return None

def load_db():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found.")
        return None
    try:
        df = pd.read_json(DATASET_FILE)
        conn = sqlite3.connect(":memory:")
        df.to_sql("song_dataset", conn, index=False, if_exists="replace")
        return conn
    except Exception as e:
        print(f"DB Create Error: {e}")
        return None

def normalize_sql(sql):
    if not sql: return ""
    sql = sql.strip().lower().rstrip(";")
    sql = re.sub(r'\s+', ' ', sql)
    if sql.startswith("(") and sql.endswith(")"): sql = sql[1:-1].strip()
    return sql

def execute_safe(conn, sql):
    if not sql: return None, "empty"
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        # Sort output to make comparison order-independent
        rows = sorted([tuple(str(x) for x in row) for row in cursor.fetchall()])
        return rows, None
    except sqlite3.OperationalError as e:
        if "no such column" in str(e).lower(): return None, "schema_error"
        return None, "syntax_error"
    except Exception:
        return None, "runtime_error"

def evaluate(name, filename, conn, gold_map):
    if not os.path.exists(filename):
        print(f"Skipping {name}: {filename} not found.")
        return

    print(f"\nEVALUATION: {name}")
    with open(filename, 'r') as f: preds = json.load(f)

    stats = {"total": 0, "exact": 0, "correct_exec": 0, 
             "errors": {"schema": 0, "syntax": 0, "runtime": 0, "semantic": 0}}

    for item in preds:
        q_id = item.get("id")
        pred_sql = item.get("sql", "")
        gold_item = gold_map.get(q_id)
        
        if not gold_item or 'sql' not in gold_item: continue
        stats["total"] += 1

        # 1. Exact Match
        if normalize_sql(pred_sql) == normalize_sql(gold_item["sql"]):
            stats["exact"] += 1

        # 2. Execution
        gold_res, _ = execute_safe(conn, gold_item["sql"])
        pred_res, err = execute_safe(conn, pred_sql)

        if err:
            stats["errors"][err.split('_')[0]] += 1 # schema, syntax, runtime
        elif gold_res == pred_res:
            stats["correct_exec"] += 1
        else:
            stats["errors"]["semantic"] += 1

    print(f"Questions: {stats['total'] + 1}")
    print(f"Exact Match: {stats['exact']}")
    print(f"Exec Match:  {stats['correct_exec']}")
    print(f"Schema Err:  {stats['errors']['schema']} (e.g. querying 'release_year' which doesn't exist)")
    print(f"Syntax Err:  {stats['errors']['syntax']}")
    print(f"Semantic Err:{stats['errors']['semantic']} (Logic mismatch)")

if __name__ == "__main__":
    # Load gold standard from file
    gold_map = load_gold_standard()
    if not gold_map:
        print("Failed to load gold standard. Exiting.")
        exit(1)
    
    # Load database
    conn = load_db()
    if conn:
        for name, fname in PREDICTION_FILES:
            evaluate(name, fname, conn, gold_map)
        conn.close()