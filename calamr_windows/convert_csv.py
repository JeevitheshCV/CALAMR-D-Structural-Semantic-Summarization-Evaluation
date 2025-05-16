# convert_csv.py
import pandas as pd
import json
import os

# CSV input path (update if needed)
csv_path = r"D:\assignments\independent_study\dataset\archive (4)\cnn_dailymail\test_2.csv"

# Output path
output_json = os.path.join("corpus", "input_docs.json")
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

records = []
for idx, row in df.iterrows():
    records.append({
        "id": f"doc_{idx:03d}",
        "body": str(row["article"]),
        "summary": str(row["highlights"])
    })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"[✓] Saved {len(records)} records to {output_json}")
