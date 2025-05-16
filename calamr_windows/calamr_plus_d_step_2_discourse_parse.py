# # calamr_plus_d_step_2_discourse_parse.py
# import os
# import json
#
# # Paths
# input_path = os.path.join("corpus", "parsed_amrs_structured.json")
# output_path = os.path.join("corpus", "discourse_tags.json")
# os.makedirs("corpus", exist_ok=True)
#
# # Heuristic rule-based relation tagging
# def tag_relation(prev, curr):
#     curr = curr.lower()
#     if "because" in curr or "due to" in curr:
#         return "reason"
#     elif "so" in curr or "therefore" in curr:
#         return "result"
#     elif "but" in curr or "however" in curr:
#         return "contrast"
#     elif "for example" in curr:
#         return "example"
#     elif "also" in curr or "and" in curr:
#         return "addition"
#     else:
#         return "elaboration"
#
# # Build discourse graph for a list of sentences
# def build_graph(sents):
#     graph = []
#     for i, sent in enumerate(sents):
#         if i == 0:
#             relation = "root"
#             parent = None
#             depth = 0
#         else:
#             relation = tag_relation(sents[i - 1], sent)
#             parent = i - 1
#             depth = graph[parent]["depth"] + 1
#
#         graph.append({
#             "index": i,
#             "sentence": sent,
#             "relation_to_prev": relation,
#             "parent": parent,
#             "depth": depth
#         })
#     return graph
#
# # Load sentence-level AMRs
# with open(input_path, "r", encoding="utf-8") as f:
#     structured_data = json.load(f)
#
# tagged_data = {}
#
# for doc in structured_data:
#     doc_id = doc["id"]
#     body_tags = build_graph(doc["body_sents"])
#     summary_tags = build_graph(doc["summary_sents"])
#
#     tagged_data[doc_id] = {
#         "body": body_tags,
#         "summary": summary_tags
#     }
#
# # Save output
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(tagged_data, f, indent=2)
#
# print(f"[✓] Discourse tags saved to {output_path}")

# calamr_plus_d_step_2_discourse_parse.py
import os
import json
from collections import Counter

# Paths
input_path = os.path.join("corpus", "parsed_amrs_structured.json")
output_path = os.path.join("corpus", "discourse_tags.json")
os.makedirs("corpus", exist_ok=True)

# Heuristic rule-based relation tagging
def tag_relation(prev, curr):
    curr = curr.lower()
    if "because" in curr or "due to" in curr:
        return "reason"
    elif "so" in curr or "therefore" in curr:
        return "result"
    elif "but" in curr or "however" in curr:
        return "contrast"
    elif "for example" in curr:
        return "example"
    elif "also" in curr or "and" in curr:
        return "addition"
    else:
        return "elaboration"

# Build discourse graph for a list of sentences
def build_graph(sents):
    graph = []
    for i, sent in enumerate(sents):
        if i == 0:
            relation = "root"
            parent = None
            depth = 0
        else:
            relation = tag_relation(sents[i - 1], sent)
            parent = i - 1
            depth = graph[parent]["depth"] + 1

        graph.append({
            "index": i,
            "sentence": sent,
            "relation_to_prev": relation,
            "parent": parent,
            "depth": depth
        })
    return graph

# Load sentence-level AMRs
with open(input_path, "r", encoding="utf-8") as f:
    structured_data = json.load(f)

tagged_data = {}
rel_counter = Counter()

print(f"[•] Processing {len(structured_data)} documents for discourse tagging...")

for doc in structured_data:
    doc_id = doc["id"]
    body_tags = build_graph(doc["body_sents"])
    summary_tags = build_graph(doc["summary_sents"])

    for tag in body_tags + summary_tags:
        rel_counter[tag["relation_to_prev"]] += 1

    tagged_data[doc_id] = {
        "body": body_tags,
        "summary": summary_tags
    }

    print(f"[✓] Tagged discourse for {doc_id}")

# Save output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tagged_data, f, indent=2)

print(f"[🎯] Discourse tags saved to {output_path}")
print(f"[📊] Relation distribution: {dict(rel_counter)}")
