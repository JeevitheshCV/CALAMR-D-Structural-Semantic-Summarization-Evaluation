# # calamr_plus_d_step_3_align_with_discourse.py
# import os
# import json
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, util
#
# # Paths
# amr_path = os.path.join("corpus", "parsed_amrs_structured.json")
# disc_path = os.path.join("corpus", "discourse_tags.json")
# output_path = os.path.join("corpus", "alignment_with_discourse.json")
# os.makedirs("corpus", exist_ok=True)
#
# # Load data
# with open(amr_path, "r", encoding="utf-8") as f:
#     amr_data = json.load(f)
#
# with open(disc_path, "r", encoding="utf-8") as f:
#     disc_data = json.load(f)
#
# # SBERT model
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
#
# def get_weight(b_depth, s_depth, b_rel, s_rel):
#     weight = 1.0
#     if b_depth == 0 or s_depth == 0:
#         weight *= 1.2
#     elif b_rel == s_rel:
#         weight *= 1.1
#     elif "reason" in [b_rel, s_rel] and "result" in [b_rel, s_rel]:
#         weight *= 1.05
#     elif b_rel in ["elaboration", "example"] or s_rel in ["elaboration", "example"]:
#         weight *= 0.9
#     return weight
#
# all_alignments = []
#
# for doc in tqdm(amr_data, desc="Aligning docs"):
#     doc_id = doc["id"]
#     body_amrs = doc["body_amrs"]
#     summary_amrs = doc["summary_amrs"]
#
#     body_meta = disc_data[doc_id]["body"]
#     summary_meta = disc_data[doc_id]["summary"]
#
#     for i, s_amr in enumerate(summary_amrs):
#         s_vec = model.encode(s_amr, convert_to_tensor=True, device=device)
#         s_info = summary_meta[i]
#         s_depth = s_info["depth"]
#         s_rel = s_info["relation_to_prev"]
#
#         for j, b_amr in enumerate(body_amrs):
#             b_vec = model.encode(b_amr, convert_to_tensor=True, device=device)
#             b_info = body_meta[j]
#             b_depth = b_info["depth"]
#             b_rel = b_info["relation_to_prev"]
#
#             cos_sim = util.cos_sim(s_vec, b_vec).item()
#             disc_weight = get_weight(b_depth, s_depth, b_rel, s_rel)
#             score = cos_sim * disc_weight
#
#             all_alignments.append({
#                 "id": doc_id,
#                 "summary_index": i,
#                 "body_index": j,
#                 "cosine_similarity": round(cos_sim, 4),
#                 "discourse_weight": round(disc_weight, 3),
#                 "adjusted_score": round(score, 4),
#                 "summary_depth": s_depth,
#                 "body_depth": b_depth,
#                 "summary_relation": s_rel,
#                 "body_relation": b_rel
#             })
#
# # Save all alignments
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(all_alignments, f, indent=2)
#
# print(f"[✓] Discourse-aware alignments saved to {output_path}")



# calamr_plus_d_step_3_align_with_discourse.py
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Paths
amr_path = os.path.join("corpus", "parsed_amrs_structured.json")
disc_path = os.path.join("corpus", "discourse_tags.json")
output_path = os.path.join("corpus", "alignment_with_discourse.json")
os.makedirs("corpus", exist_ok=True)

# Load data
with open(amr_path, "r", encoding="utf-8") as f:
    amr_data = json.load(f)

with open(disc_path, "r", encoding="utf-8") as f:
    disc_data = json.load(f)

# SBERT model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def get_weight(b_depth, s_depth, b_rel, s_rel):
    weight = 1.0
    if b_depth == 0 or s_depth == 0:
        weight *= 1.2
    elif b_rel == s_rel:
        weight *= 1.1
    elif "reason" in [b_rel, s_rel] and "result" in [b_rel, s_rel]:
        weight *= 1.05
    elif b_rel in ["elaboration", "example"] or s_rel in ["elaboration", "example"]:
        weight *= 0.9
    return weight

all_alignments = []

for doc in tqdm(amr_data, desc="Aligning docs"):
    doc_id = doc["id"]
    body_amrs = doc["body_amrs"]
    summary_amrs = doc["summary_amrs"]

    body_meta = disc_data[doc_id]["body"]
    summary_meta = disc_data[doc_id]["summary"]

    for i, s_amr in enumerate(summary_amrs):
        s_vec = model.encode(s_amr, convert_to_tensor=True, device=device)
        s_info = summary_meta[i]
        s_depth = s_info["depth"]
        s_rel = s_info["relation_to_prev"]

        for j, b_amr in enumerate(body_amrs):
            b_vec = model.encode(b_amr, convert_to_tensor=True, device=device)
            b_info = body_meta[j]
            b_depth = b_info["depth"]
            b_rel = b_info["relation_to_prev"]

            cos_sim = util.cos_sim(s_vec, b_vec).item()
            disc_weight = get_weight(b_depth, s_depth, b_rel, s_rel)
            score = cos_sim * disc_weight

            all_alignments.append({
                "id": doc_id,
                "summary_index": i,
                "body_index": j,
                "cosine_similarity": round(cos_sim, 4),
                "discourse_weight": round(disc_weight, 3),
                "adjusted_score": round(score, 4),
                "summary_depth": s_depth,
                "body_depth": b_depth,
                "summary_relation": s_rel,
                "body_relation": b_rel
            })

# Save all alignments
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_alignments, f, indent=2)

print(f"[✓] Discourse-aware alignments saved to {output_path}")
