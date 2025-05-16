# import json
# import os
# import networkx as nx
# from tqdm import tqdm
#
# # Load keyword distance matrices
# with open("corpus/amr_keyword_distance_matrices.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# output_scores = []
#
# def build_graph_from_matrix(keywords, matrix):
#     G = nx.Graph()
#     for i, kw in enumerate(keywords):
#         G.add_node(i, label=kw)
#
#     for i in range(len(keywords)):
#         for j in range(i + 1, len(keywords)):
#             if matrix[i][j] > 0:
#                 G.add_edge(i, j)
#     return G
#
# def compute_edge_overlap(G1, G2):
#     e1 = set(G1.edges())
#     e2 = set(G2.edges())
#     if not e1 and not e2:
#         return 1.0
#     return len(e1 & e2) / len(e1 | e2)
#
# def compute_edit_distance(G1, G2):
#     try:
#         ged = nx.graph_edit_distance(G1, G2, timeout=2)
#         max_edits = max(G1.number_of_edges(), G2.number_of_edges()) or 1
#         return 1 - min(ged, max_edits) / max_edits
#     except Exception:
#         return 0.0
#
# def compute_neighborhood_overlap(G1, G2):
#     nodes = set(G1.nodes()) & set(G2.nodes())
#     if not nodes:
#         return 0.0
#
#     total_overlap = 0
#     for n in nodes:
#         n1 = set(G1.neighbors(n)) if G1.has_node(n) else set()
#         n2 = set(G2.neighbors(n)) if G2.has_node(n) else set()
#         if not n1 and not n2:
#             total_overlap += 1
#         elif n1 or n2:
#             total_overlap += len(n1 & n2) / len(n1 | n2)
#     return total_overlap / len(nodes)
#
# def count_isolated_nodes(G):
#     return len([n for n in G.nodes() if G.degree(n) == 0])
#
# # Main processing
# for entry in tqdm(data, desc="Computing graph similarity scores"):
#     doc_id = entry["id"]
#     keywords = entry["keywords"]
#     body_matrix = entry.get("body_matrix", [])
#     summary_matrix = entry.get("summary_matrix", [])
#
#     if not body_matrix or not summary_matrix:
#         continue
#
#     G_body = build_graph_from_matrix(keywords, body_matrix)
#     G_summary = build_graph_from_matrix(keywords, summary_matrix)
#
#     edge_score = compute_edge_overlap(G_body, G_summary)
#     edit_score = compute_edit_distance(G_body, G_summary)
#     neighborhood_score = compute_neighborhood_overlap(G_body, G_summary)
#
#     # Weighted Final Score
#     weighted_score = (
#         0.2 * edge_score +
#         0.2 * edit_score +
#         0.6 * neighborhood_score
#     )
#
#     # Sparsity-aware adjustment
#     iso_body = count_isolated_nodes(G_body)
#     iso_summary = count_isolated_nodes(G_summary)
#     total_nodes = len(keywords)
#     isolation_penalty = (iso_body + iso_summary) / (2 * total_nodes)
#     penalty_weight = 0.2
#
#     adjusted_score = round(weighted_score * (1 - penalty_weight * isolation_penalty), 4)
#
#     output_scores.append({
#         "id": doc_id,
#         "edge_overlap_score": round(edge_score, 4),
#         "edit_distance_score": round(edit_score, 4),
#         "neighborhood_overlap_score": round(neighborhood_score, 4),
#         "Final_score": round(weighted_score, 4),
#         "isolated_nodes_body": iso_body,
#         "isolated_nodes_summary": iso_summary
#     })
#
# # Save output
# output_path = "corpus/graph_similarity_scores.json"
# os.makedirs("corpus", exist_ok=True)
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output_scores, f, indent=2)
#
# print(f"[✓] Graph similarity scores saved to {output_path}")


# -------------------------------------------------------------------------------------------------------------

# compare_graphs_similarity.py

import json
import os
import numpy as np
from tqdm import tqdm

# Paths
edge_weights_path = "corpus/edge_weights.json"
output_path = "corpus/graph_similarity_scores.json"

# Load edge weights
with open(edge_weights_path, "r", encoding="utf-8") as f:
    edge_data = json.load(f)

# Metric: CALAMR-style flow coverage score (final score only)
def calamr_flow_score(mat1, mat2):
    n = len(mat1)
    total_1 = sum(max(row) for row in mat1 if any(row))
    total_2 = sum(max(row) for row in mat2 if any(row))

    coverage_1 = sum(
        max(mat2[j][i] for j in range(n)) if any(row) else 0
        for i, row in enumerate(mat1)
    )
    coverage_2 = sum(
        max(mat1[i][j] for i in range(n)) if any(col) else 0
        for j, col in enumerate(zip(*mat2))
    )

    cfc = coverage_1 / (total_1 + 1e-8)
    cfy = coverage_2 / (total_2 + 1e-8)
    cf = 2 * cfc * cfy / (cfc + cfy + 1e-8)
    return round(cf, 4)

# Metric: Cosine similarity between flattened matrices
def cosine_similarity_score(mat1, mat2):
    v1 = np.array(mat1).flatten()
    v2 = np.array(mat2).flatten()
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return round(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 4)

# Metric: Jaccard similarity on presence/absence of connections
def jaccard_connections(mat1, mat2, threshold=0.1):
    edges1 = {(i, j) for i in range(len(mat1)) for j in range(len(mat1)) if mat1[i][j] > threshold}
    edges2 = {(i, j) for i in range(len(mat2)) for j in range(len(mat2)) if mat2[i][j] > threshold}
    if not edges1 and not edges2:
        return 1.0
    inter = len(edges1 & edges2)
    union = len(edges1 | edges2)
    return round(inter / union, 4) if union else 0.0

# Metric: Inverse of average difference
def inverse_avg_diff(mat1, mat2):
    total = 0
    count = 0
    for i in range(len(mat1)):
        for j in range(len(mat1)):
            total += abs(mat1[i][j] - mat2[i][j])
            count += 1
    avg_diff = total / count if count else 1
    return round(1 - avg_diff, 4)

# Main loop
results = []
for doc in tqdm(edge_data, desc="Comparing graphs"):
    doc_id = doc["id"]
    mat1 = np.array(doc["body_weights"])
    mat2 = np.array(doc["summary_weights"])

    results.append({
        "id": doc_id,
        "cosine_similarity": cosine_similarity_score(mat1, mat2),
        "flow_coverage": calamr_flow_score(mat1, mat2),
    })

# Save results
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[✓] Graph similarity scores saved to → {output_path}")

