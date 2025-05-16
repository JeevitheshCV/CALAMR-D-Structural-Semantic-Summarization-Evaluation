# # step3_visualize_graphs_fixed_layout.py
#
# import os
# import json
# import networkx as nx
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # Load the keyword-distance matrices
# with open("corpus/amr_keyword_distance_matrices.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # Output directories
# body_dir = "graphs/improved_graphs/body"
# summary_dir = "graphs/improved_graphs/summary"
# os.makedirs(body_dir, exist_ok=True)
# os.makedirs(summary_dir, exist_ok=True)
#
# # Build graph from matrix
# def build_graph_from_matrix(keywords, matrix):
#     G = nx.Graph()
#     for i, kw in enumerate(keywords):
#         G.add_node(i, label=kw)
#
#     for i in range(len(keywords)):
#         for j in range(i + 1, len(keywords)):
#             try:
#                 depth = int(matrix[i][j])
#                 if depth > 0:
#                     G.add_edge(i, j, depth=depth)
#             except:
#                 continue
#     return G
#
# # Draw and save graph
# def draw_and_save(G, keywords, save_path, title=""):
#     pos = nx.kamada_kawai_layout(G)  # Handles sparse graphs better
#     labels = nx.get_node_attributes(G, 'label')
#     edge_labels = nx.get_edge_attributes(G, 'depth')
#
#     plt.figure(figsize=(min(15, 1.5 * len(G.nodes)), 12))
#     nx.draw(G, pos,
#             labels=labels,
#             with_labels=True,
#             node_color='skyblue',
#             node_size=1200,
#             font_size=9,
#             font_weight='bold',
#             edge_color='gray',
#             width=1.5)
#
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
#     plt.title(title)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#
# # Visualize graphs
# for entry in tqdm(data, desc="Visualizing graphs"):
#     doc_id = entry["id"]
#     keywords = entry["keywords"]
#
#     body_matrix = entry.get("body_matrix", [])
#     summary_matrix = entry.get("summary_matrix", [])
#
#     if body_matrix:
#         G_body = build_graph_from_matrix(keywords, body_matrix)
#         draw_and_save(G_body, keywords, os.path.join(body_dir, f"{doc_id}.png"), title=f"{doc_id} - Article Graph")
#
#     if summary_matrix:
#         G_summary = build_graph_from_matrix(keywords, summary_matrix)
#         draw_and_save(G_summary, keywords, os.path.join(summary_dir, f"{doc_id}.png"), title=f"{doc_id} - Summary Graph")
#
# print("[✓] All graph visualizations successfully saved.")

#-------------------------------------------------------------------------------------------------------------


# # step3_generate_keyword_graph_plots.py
#
# import json
# import os
# import networkx as nx
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # Paths
# input_path = "corpus/amr_keyword_embedding_matrices.json"
# body_dir = "graphs/improved_graphs/body"
# summary_dir = "graphs/improved_graphs/summary"
# os.makedirs(body_dir, exist_ok=True)
# os.makedirs(summary_dir, exist_ok=True)
#
# # Load the matrices
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
#
# def build_graph(keywords, matrix):
#     G = nx.DiGraph()
#     for kw in keywords:
#         G.add_node(kw)
#
#     K = len(keywords)
#     for i in range(K):
#         for j in range(K):
#             emb = matrix[i][j]
#             if emb:
#                 weight = float(np.linalg.norm(emb))
#                 if weight > 0:
#                     G.add_edge(keywords[i], keywords[j], weight=weight)
#     return G
#
#
# def draw_and_save_graph(G, filepath, title=""):
#     plt.figure(figsize=(10, 8))
#     pos = nx.spring_layout(G, seed=42)
#     edge_weights = nx.get_edge_attributes(G, "weight")
#
#     # Draw
#     nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#AED6F1")
#     nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="#2E86C1", width=2)
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
#     nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.2f}" for e, w in edge_weights.items()}, font_size=8)
#
#     plt.title(title, fontsize=12)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.close()
#
#
# # Main loop
# for entry in tqdm(data, desc="Generating and saving graph plots"):
#     doc_id = entry["id"]
#     keywords = entry["keywords"]
#     body_matrix = entry.get("body_matrix", [])
#     summary_matrix = entry.get("summary_matrix", [])
#
#     # Body Graph Plot
#     if body_matrix:
#         G_body = build_graph(keywords, body_matrix)
#         body_path = os.path.join(body_dir, f"{doc_id}.png")
#         draw_and_save_graph(G_body, body_path, title=f"Body Graph: {doc_id}")
#
#     # Summary Graph Plot
#     if summary_matrix:
#         G_summary = build_graph(keywords, summary_matrix)
#         summary_path = os.path.join(summary_dir, f"{doc_id}.png")
#         draw_and_save_graph(G_summary, summary_path, title=f"Summary Graph: {doc_id}")
#
# print("[✓] Graph plots saved in 'graphs/improved_graphs/body' and 'summary'.")


#-------------------------------------------------------------------------------------------------------------------
#
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from sklearn.decomposition import PCA
#
# # Load SBERT
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Input/output paths
# input_path = "corpus/amr_keyword_embedding_matrices_hybrid.json"
# body_dir = "graphs/improved_graphs/body"
# summary_dir = "graphs/improved_graphs/summary"
# os.makedirs(body_dir, exist_ok=True)
# os.makedirs(summary_dir, exist_ok=True)
#
# def build_graph(keywords, matrix):
#     G = nx.DiGraph()
#     for kw in keywords:
#         G.add_node(kw)
#
#     K = len(keywords)
#     for i in range(K):
#         for j in range(K):
#             emb = matrix[i][j]
#             if emb:
#                 weight = float(np.linalg.norm(emb))
#                 G.add_edge(keywords[i], keywords[j], weight=weight)
#     return G
#
# def get_semantic_layout(keywords):
#     embeddings = [sbert_model.encode(kw) for kw in keywords]
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(embeddings)
#     return {kw: tuple(reduced[i]) for i, kw in enumerate(keywords)}
#
# def draw_and_save_graph(G, layout, filepath, title=""):
#     plt.figure(figsize=(10, 8))
#
#     edge_weights = nx.get_edge_attributes(G, "weight")
#
#     nx.draw_networkx_nodes(G, layout, node_color="#AED6F1", node_size=800)
#     nx.draw_networkx_edges(G, layout, edge_color="#2E86C1", arrows=True, width=2)
#     nx.draw_networkx_labels(G, layout, font_size=10, font_weight="bold")
#     nx.draw_networkx_edge_labels(
#         G,
#         layout,
#         edge_labels={e: f"{w:.2f}" for e, w in edge_weights.items()},
#         font_size=7
#     )
#
#     plt.title(title, fontsize=12)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.close()
#
# # Main loop
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# for entry in tqdm(data, desc="Drawing SBERT-positioned graphs"):
#     doc_id = entry["id"]
#     keywords = entry["keywords"]
#     layout = get_semantic_layout(keywords)
#
#     for kind, out_dir in [("body_matrix", body_dir), ("summary_matrix", summary_dir)]:
#         matrix = entry.get(kind)
#         if matrix:
#             G = build_graph(keywords, matrix)
#             out_path = os.path.join(out_dir, f"{doc_id}.png")
#             draw_and_save_graph(G, layout, out_path, title=f"{kind.replace('_', ' ').title()}: {doc_id}")
#
# print("[✓] Semantic-positioned graph plots saved.")
#
# --------------------------------------------------------------------------------------------------------------------

# generate_edge_weighted_graphs.py

import os
import json
import penman
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Paths
embedding_path = "corpus/amr_keyword_embedding_matrices.json"
amr_path = "corpus/parsed_amrs.json"
output_weights_path = "corpus/edge_weights.json"
body_dir = "graphs/improved_graphs/body"
summary_dir = "graphs/improved_graphs/summary"
os.makedirs(body_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# Load data
with open(embedding_path, "r", encoding="utf-8") as f:
    embedding_data = json.load(f)
with open(amr_path, "r", encoding="utf-8") as f:
    amr_data = json.load(f)
amr_lookup = {entry["id"]: entry for entry in amr_data}

# SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Utilities
def normalize(c): return c.lower().split("-")[0]

def extract_named_concepts(graph):
    concepts = {}
    for t in graph.triples:
        if t[1] == ":instance":
            concepts[t[0]] = normalize(t[2])
    for t in graph.triples:
        if t[1] == ":name" and t[2] in concepts:
            parts = []
            for op in [":op1", ":op2", ":op3", ":op4", ":op5"]:
                parts += [tt[2].strip('"').lower() for tt in graph.triples if tt[0] == t[2] and tt[1] == op]
            if parts:
                concepts[t[0]] = " ".join(parts)
    return concepts

def find_reference_phrase(amr_str, kw1, kw2):
    try:
        graph = penman.decode(amr_str)
        var_to_concept = extract_named_concepts(graph)
        G = nx.Graph()
        for var in var_to_concept:
            G.add_node(var, label=var_to_concept[var])
        for src, role, tgt in graph.edges():
            G.add_edge(src, tgt, role=role)
        var1 = next((v for v, l in G.nodes(data="label") if l == kw1), None)
        var2 = next((v for v, l in G.nodes(data="label") if l == kw2), None)
        if var1 and var2 and nx.has_path(G, var1, var2):
            path = nx.shortest_path(G, var1, var2)
            parts = []
            for i in range(len(path) - 1):
                src = path[i]
                tgt = path[i+1]
                role = G[src][tgt].get("role", ":related").lstrip(":")
                parts.append(f"{G.nodes[src]['label']} is the {role} of {G.nodes[tgt]['label']}")
            return " -> ".join(parts)
    except:
        return None
    return None

def cosine(vec1, vec2):
    return float(cosine_similarity([vec1], [vec2])[0][0])


def draw_graph(doc_id, keywords, matrix, out_path, title=""):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    G = nx.Graph()
    for kw in keywords:
        G.add_node(kw)

    K = len(keywords)
    for i in range(K):
        for j in range(K):
            w = matrix[i][j]
            if w and w > 0.05:
                G.add_edge(keywords[i], keywords[j], weight=w)

    if not G.edges:
        print(f"[!] Empty graph skipped: {doc_id}")
        return

    # Edge length = inverse similarity
    ε = 1e-5
    for u, v, d in G.edges(data=True):
        d["length"] = 1.0 / (d["weight"] + ε)

    connected_nodes = set(n for e in G.edges for n in e)
    disconnected_nodes = set(G.nodes) - connected_nodes

    pos_connected = nx.kamada_kawai_layout(G.subgraph(connected_nodes), weight="length")

    # Place disconnected nodes in a neat outer circle
    angle_step = 2 * np.pi / max(1, len(disconnected_nodes))
    cluster_radius = 1.2 * max(np.linalg.norm(p) for p in pos_connected.values())
    for idx, node in enumerate(disconnected_nodes):
        angle = idx * angle_step
        x = cluster_radius * np.cos(angle)
        y = cluster_radius * np.sin(angle)
        pos_connected[node] = [x, y]

    pos = pos_connected
    weights = nx.get_edge_attributes(G, "weight")

    # Edge styling: thin & elegant
    edge_widths = [0.5 + 1.5 * w for w in weights.values()]
    edge_color = "#2980B9"

    # Label all edges with decent weight
    edge_labels = {e: f"{w:.2f}" for e, w in weights.items() if w >= 0.5}

    # Drawing
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color="#AED6F1", node_size=900)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_color, alpha=0.85)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1)
    )

    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=300)
    plt.close()




# Main processing loop
final_output = []
for entry in tqdm(embedding_data, desc="Generating graphs with reference embeddings"):
    doc_id = entry["id"]
    keywords = entry["keywords"]
    body_amr = amr_lookup.get(doc_id, {}).get("body_amr", "")

    weights_body = []
    weights_summary = []

    K = len(keywords)
    observed_body = entry.get("body_matrix", [])
    observed_summary = entry.get("summary_matrix", [])

    for matrix_type, observed_matrix, store_matrix in [
        ("body", observed_body, weights_body),
        ("summary", observed_summary, weights_summary)
    ]:
        weight_matrix = [[0.0 for _ in range(K)] for _ in range(K)]

        for i in range(K):
            for j in range(K):
                vec = observed_matrix[i][j]
                if vec:
                    ref_phrase = find_reference_phrase(body_amr, keywords[i], keywords[j])
                    if ref_phrase:
                        try:
                            ref_vec = sbert_model.encode(ref_phrase)
                            weight_matrix[i][j] = cosine(vec, ref_vec)
                        except:
                            continue
        store_matrix.extend(weight_matrix)

        out_path = os.path.join(body_dir if matrix_type == "body" else summary_dir, f"{doc_id}.png")
        draw_graph(doc_id, keywords, weight_matrix, out_path, title=f"{matrix_type.title()} Graph: {doc_id}")

    final_output.append({
        "id": doc_id,
        "keywords": keywords,
        "body_weights": weights_body,
        "summary_weights": weights_summary
    })

# Save edge weights
with open(output_weights_path, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2)

print(f"[✓] Edge-weighted matrices + graphs saved → {output_weights_path}")
