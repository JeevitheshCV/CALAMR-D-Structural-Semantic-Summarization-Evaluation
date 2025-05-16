
# convert_json.py
import json
import os
import statistics

# Input/output file paths
input_path = r"D:\assignments\independent_study\project_CALAMR\project_CALAMR\CALAMR\dataset\model_annotations\records_M0.json"
output_docs_path = "corpus/input_docs.json"
output_scores_path = "corpus/human_scores.json"

# Ensure output directory exists
os.makedirs("corpus", exist_ok=True)

docs = []
scores = []

# Load JSON array from input
with open(input_path, "r", encoding="utf-8") as infile:
    all_items = json.load(infile)

for item in all_items:
    summary_id = item["id"]
    summary = item["decoded"]
    references = item["references"]

    # Combine references into article
    article = " ".join(references)

    # Prepare document entry
    docs.append({
        "id": summary_id,
        "body": article,
        "summary": summary
    })

    # Gather all annotations
    annotations = item["expert_annotations"] + item["turker_annotations"]

    # Extract and average individual scores
    coherence_scores = [a["coherence"] for a in annotations]
    consistency_scores = [a["consistency"] for a in annotations]
    fluency_scores = [a["fluency"] for a in annotations]
    relevance_scores = [a["relevance"] for a in annotations]

    # Compute means
    coherence_mean = statistics.mean(coherence_scores)
    consistency_mean = statistics.mean(consistency_scores)
    fluency_mean = statistics.mean(fluency_scores)
    relevance_mean = statistics.mean(relevance_scores)

    # Final overall score (average of 4 dimensions)
    human_score = statistics.mean([
        coherence_mean,
        consistency_mean,
        fluency_mean,
        relevance_mean
    ])

    # Append scores entry
    scores.append({
        "id": summary_id,
        "coherence": coherence_mean,
        "consistency": consistency_mean,
        "fluency": fluency_mean,
        "relevance": relevance_mean,
        "human_score": human_score
    })

# Save documents and scores
with open(output_docs_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=4)

with open(output_scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=4)

print(f"[✓] Created {len(docs)} summaries and {len(scores)} detailed human score entries.")
