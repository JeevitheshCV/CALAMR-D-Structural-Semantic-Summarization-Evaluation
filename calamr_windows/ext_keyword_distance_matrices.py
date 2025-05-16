# step2_generate_keyword_distance_matrices.py

import json
import os
import penman
import networkx as nx
from tqdm import tqdm

# Paths
amr_path = "corpus/parsed_amrs.json"
keywords_path = "corpus/amr_keywords.json"
output_path = "corpus/amr_keyword_distance_matrices.json"

# Load files
with open(amr_path, "r", encoding="utf-8") as f:
    amr_data = json.load(f)

with open(keywords_path, "r", encoding="utf-8") as f:
    keyword_data = json.load(f)

# Build a lookup for AMRs
amr_lookup = {entry["id"]: entry for entry in amr_data}

# Output list
output = []

# Normalize concept label (e.g., "run-01" -> "run")
def normalize_concept(c):
    return c.lower().split("-")[0]

# Build graph from penman triples
def build_nx_graph(amr_str):
    graph = penman.decode(amr_str)
    G = nx.Graph()

    for var, _, concept in graph.instances():
        G.add_node(var, label=normalize_concept(concept))

    for src, role, tgt in graph.edges():
        G.add_edge(src, tgt, role=role)

    return G, graph

# Main loop
for entry in tqdm(keyword_data, desc="Processing documents"):
    doc_id = entry["id"]
    keywords = entry["keywords"]

    if doc_id not in amr_lookup:
        print(f"[!] Missing AMR for {doc_id}")
        continue

    doc_amrs = amr_lookup[doc_id]
    matrices = {}

    for key in ["body_amr", "summary_amr"]:
        amr_str = doc_amrs.get(key, "")
        if not amr_str.strip():
            print(f"[!] Missing {key} in {doc_id}")
            continue

        try:
            G, pen_graph = build_nx_graph(amr_str)

            # Map keywords to AMR node variables
            kw_to_var = {}
            for var, data in G.nodes(data=True):
                label = data["label"]
                for kw in keywords:
                    if kw.lower() == label:
                        kw_to_var[kw] = var

            # Initialize matrix
            K = len(keywords)
            matrix = [[0 for _ in range(K)] for _ in range(K)]

            for i in range(K):
                for j in range(K):
                    k1, k2 = keywords[i], keywords[j]
                    if k1 in kw_to_var and k2 in kw_to_var:
                        try:
                            path_len = nx.shortest_path_length(G, source=kw_to_var[k1], target=kw_to_var[k2])
                            matrix[i][j] = path_len
                        except nx.NetworkXNoPath:
                            matrix[i][j] = 0  # no path
                    else:
                        matrix[i][j] = 0  # one or both keywords not found

            matrices[key] = matrix

        except Exception as e:
            print(f"[ERROR] Failed for {doc_id} ({key}): {e}")

    # Store output
    output.append({
        "id": doc_id,
        "keywords": keywords,
        "body_matrix": matrices.get("body_amr", []),
        "summary_matrix": matrices.get("summary_amr", [])
    })

# Save results
os.makedirs("corpus", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[✓] Keyword-distance matrices saved → {output_path}")
