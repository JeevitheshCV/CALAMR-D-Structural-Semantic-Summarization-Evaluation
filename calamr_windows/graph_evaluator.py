# graph_evaluator.py
import os
import json
from collections import defaultdict

try:
    from smatch import smatch_score
    has_smatch = True
except ImportError:
    has_smatch = False
    print("[!] 'smatch' module not found. Only concept overlap will be used.")

# Paths
input_path = "corpus/parsed_amrs.json"
output_path = "corpus/graph_eval_results.json"
os.makedirs("corpus", exist_ok=True)

# Load AMRs
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

def extract_concepts(amr):
    tokens = set()
    for line in amr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.update(line.replace("(", " ").replace(")", " ").split())
    return tokens

for entry in data:
    doc_id = entry["id"]
    body_amr = entry["body_amr"]
    summary_amr = entry["summary_amr"]

    try:
        body_concepts = extract_concepts(body_amr)
        summary_concepts = extract_concepts(summary_amr)
        intersection = body_concepts & summary_concepts
        union = body_concepts | summary_concepts
        jaccard = len(intersection) / len(union) if union else 0.0

        # Optional: run Smatch
        smatch_f1 = None
        if has_smatch:
            smatch_f1 = next(smatch_score.compute_f_score(body_amr, summary_amr))

        results.append({
            "id": doc_id,
            "concept_overlap": round(jaccard, 4),
            "common_concepts": len(intersection),
            "body_concepts": len(body_concepts),
            "summary_concepts": len(summary_concepts),
            "smatch_f1": round(smatch_f1, 4) if smatch_f1 is not None else None
        })

    except Exception as e:
        results.append({
            "id": doc_id,
            "error": str(e)
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[✓] Saved graph-based scores for {len(results)} documents → {output_path}")
