# # calamr_plus_d_step_5_score.py
# import os
# import json
# import numpy as np
#
# # Paths
# alignment_path = os.path.join("corpus", "alignment_with_discourse.json")
# flow_path = os.path.join("corpus", "flow_results.json")
# output_path = os.path.join("corpus", "calamr_plus_d_outputs.json")
# os.makedirs("corpus", exist_ok=True)
#
# # Load data
# with open(alignment_path, "r", encoding="utf-8") as f:
#     align_data = json.load(f)
#
# with open(flow_path, "r", encoding="utf-8") as f:
#     flow_data = json.load(f)
#
# # Organize by ID
# from collections import defaultdict
#
# cosine_by_id = defaultdict(list)
# for entry in align_data:
#     cosine_by_id[entry["id"]].append(entry["cosine_similarity"])
#
# flow_by_id = {x["id"]: x for x in flow_data}
#
# # Composite scoring
# results = []
# for doc_id, cosines in cosine_by_id.items():
#     sem_mean = np.mean(cosines)
#     flow = flow_by_id.get(doc_id, {})
#
#     coverage = flow.get("alignment_ratio", 0)
#     halluc_penalty = flow.get("hallucination_penalty", 0)
#     avg_depth = flow.get("avg_body_depth_matched", 0)
#     depth_penalty = 1.0 - (avg_depth / 6.0) if avg_depth is not None else 0.0
#
#     composite_score = (
#         (0.5 * sem_mean) +
#         (0.3 * coverage) +
#         (0.2 * depth_penalty) -
#         halluc_penalty
#     )
#
#     results.append({
#         "id": doc_id,
#         "sem_mean": round(sem_mean, 4),
#         "coverage": round(coverage, 4),
#         "depth_penalty": round(depth_penalty, 4),
#         "hallucination_penalty": round(halluc_penalty, 4),
#         "composite_score": round(composite_score, 4)
#     })
#
# # Save
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2)
#
# print(f"[✓] CALAMR+D scores saved to {output_path}")


# calamr_plus_d_step_5_score.py
import os
import json
import numpy as np
from collections import defaultdict

# Paths
alignment_path = os.path.join("corpus", "alignment_with_discourse.json")
flow_path = os.path.join("corpus", "flow_results.json")
output_path = os.path.join("corpus", "calamr_plus_d_outputs.json")
os.makedirs("corpus", exist_ok=True)

# Load data
with open(alignment_path, "r", encoding="utf-8") as f:
    align_data = json.load(f)

with open(flow_path, "r", encoding="utf-8") as f:
    flow_data = json.load(f)

# Organize by ID
cosine_by_id = defaultdict(list)
for entry in align_data:
    cosine_by_id[entry["id"]].append(entry["cosine_similarity"])

flow_by_id = {x["id"]: x for x in flow_data}

# Composite scoring
results = []

print(f"[•] Scoring {len(cosine_by_id)} documents...")

for doc_id, cosines in cosine_by_id.items():
    sem_mean = np.mean(cosines)
    flow = flow_by_id.get(doc_id, {})

    coverage = flow.get("alignment_ratio", 0)
    halluc_penalty = flow.get("hallucination_penalty", 0)
    avg_depth = flow.get("avg_body_depth_matched", 0)
    depth_penalty = 1.0 - (avg_depth / 6.0) if avg_depth is not None else 0.0

    composite_score = (
        (0.5 * sem_mean) +
        (0.3 * coverage) +
        (0.2 * depth_penalty) -
        halluc_penalty
    )

    results.append({
        "id": doc_id,
        "sem_mean": round(sem_mean, 4),
        "coverage": round(coverage, 4),
        "depth_penalty": round(depth_penalty, 4),
        "hallucination_penalty": round(halluc_penalty, 4),
        "composite_score": round(composite_score, 4)
    })

    print(f"[{doc_id}] Score: {composite_score:.4f}")
    print(f"   • Sem. Mean: {sem_mean:.2f}")
    print(f"   • Coverage: {coverage:.2f}")
    print(f"   • Depth Penalty: {depth_penalty:.2f}")
    print(f"   • Halluc. Penalty: {halluc_penalty:.2f}")

# Save
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

avg_score = np.mean([x["composite_score"] for x in results])
print(f"[🎯] CALAMR+D scores saved to {output_path}")
print(f"[📈] Average Composite Score: {avg_score:.4f}")
