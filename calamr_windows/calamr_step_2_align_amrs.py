# # calamr_step_2_align_amrs.py
# import os
# import json
# from sentence_transformers import SentenceTransformer, util
# from tqdm import tqdm
#
# input_path = os.path.join("corpus", "parsed_amrs.json")
# output_path = os.path.join("corpus", "calamr_scores.json")
#
# # Load data
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
# results = []
#
# print(f"[•] Computing similarity for {len(data)} documents...")
#
# for entry in tqdm(data):
#     doc_id = entry["id"]
#     body_amr = entry["body_amr"]
#     summary_amr = entry["summary_amr"]
#
#     try:
#         vec_body = model.encode(body_amr, convert_to_tensor=True, device=device)
#         vec_summary = model.encode(summary_amr, convert_to_tensor=True, device=device)
#         score = util.cos_sim(vec_body, vec_summary).item()
#
#         results.append({
#             "id": doc_id,
#             "score": round(score, 4)
#         })
#     except Exception as e:
#         print(f"[!] Failed {doc_id}: {e}")
#
# # Save results
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2)
#
# print(f"[✓] Scores saved to {output_path}")


# calamr_step_2_align_amrs.py
import os
import json
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

input_path = os.path.join("corpus", "parsed_amrs.json")
output_path = os.path.join("corpus", "calamr_scores.json")

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
results = []

print(f"[•] Loaded {len(data)} documents for alignment")
print(f"[•] Using model: all-MiniLM-L6-v2 on device: {device}")

for entry in tqdm(data, desc="Computing similarity"):
    doc_id = entry["id"]
    body_amr = entry["body_amr"]
    summary_amr = entry["summary_amr"]

    try:
        vec_body = model.encode(body_amr, convert_to_tensor=True, device=device)
        vec_summary = model.encode(summary_amr, convert_to_tensor=True, device=device)
        score = util.cos_sim(vec_body, vec_summary).item()

        results.append({
            "id": doc_id,
            "score": round(score, 4)
        })

        print(f"[✓] {doc_id} → Cosine Similarity: {score:.4f}")

    except Exception as e:
        print(f"[!] Failed alignment for {doc_id}: {e}")

# Save results
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[🎯] Scores saved to {output_path}")
