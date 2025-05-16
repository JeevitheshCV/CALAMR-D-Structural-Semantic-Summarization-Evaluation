# calamr_step_3_score_amrs.py
import json
import os
import numpy as np

path = os.path.join("corpus", "calamr_scores.json")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

scores = [x["score"] for x in data]
print("\n==== CALAMR Summary ====")
print(f"Total: {len(scores)}")
print(f"Average Cosine Similarity: {np.mean(scores):.4f}")
print(f"Standard Deviation:        {np.std(scores):.4f}")
print("=========================\n")
