# optimize_weights.py
import os
import json
import numpy as np
from itertools import product
from scipy.stats import pearsonr

# Load files
with open("corpus/human_scores.json", "r", encoding="utf-8") as f:
    human_scores = {x["id"]: x["human_score"] for x in json.load(f)}

with open("corpus/calamr_plus_d_outputs.json", "r", encoding="utf-8") as f:
    model_scores = {x["id"]: x for x in json.load(f)}

# Merge by ID
data = []
for _id in human_scores:
    if _id in model_scores:
        m = model_scores[_id]
        data.append({
            "id": _id,
            "human_score": human_scores[_id],
            "sem_mean": m["sem_mean"],
            "coverage": m["coverage"],
            "depth_penalty": m["depth_penalty"],
            "hallucination_penalty": m["hallucination_penalty"]
        })

print(f"[✓] Aligned {len(data)} records for evaluation")

# Predict using weights
def predict(w, x):
    return (
        w[0] * x["sem_mean"] +
        w[1] * x["coverage"] +
        w[2] * x["depth_penalty"] -
        w[3] * x["hallucination_penalty"]
    )

# Grid search
best_corr = -1
best_weights = None
weight_range = np.arange(0.0, 1.05, 0.1)

for w1, w2, w3 in product(weight_range, repeat=3):
    if w1 + w2 + w3 > 1:
        continue
    w4 = 1.0  # hallucination penalty fixed to subtractive

    preds = [predict((w1, w2, w3, w4), x) for x in data]
    targets = [x["human_score"] for x in data]
    if len(preds) >= 2:
        corr, _ = pearsonr(preds, targets)
        if corr > best_corr:
            best_corr = corr
            best_weights = (w1, w2, w3, w4)

# Final output
if best_weights is not None:
    print("\n=== Optimal CALAMR+D Weights ===")
    print(f"Semantic Similarity (α1):       {best_weights[0]:.2f}")
    print(f"Coverage (α2):                 {best_weights[1]:.2f}")
    print(f"Depth Penalty (α3):            {best_weights[2]:.2f}")
    print(f"Hallucination Penalty (α4):    {best_weights[3]:.2f} (subtracted)")
    print(f"Best Pearson Correlation:      {best_corr:.4f}")
else:
    print("\n[!] Not enough data to compute correlation.")
    print("    Need at least 2 records with both CALAMR+D and human scores.")

