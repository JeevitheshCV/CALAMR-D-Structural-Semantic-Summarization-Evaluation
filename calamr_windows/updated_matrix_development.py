# step2_generate_keyword_embedding_matrices_hybrid.py

import json
import os
import penman
import networkx as nx
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
amr_path = "corpus/parsed_amrs.json"
keywords_path = "corpus/amr_keywords.json"
output_path = "corpus/amr_keyword_embedding_matrices.json"

# Load files
with open(amr_path, "r", encoding="utf-8") as f:
    amr_data = json.load(f)

with open(keywords_path, "r", encoding="utf-8") as f:
    keyword_data = json.load(f)

amr_lookup = {entry["id"]: entry for entry in amr_data}
output = []

def normalize_concept(concept):
    return concept.lower().split("-")[0]

def extract_named_concepts(graph):
    var_to_concept = {}
    for triple in graph.triples:
        src, role, tgt = triple
        if role == ":instance":
            var_to_concept[src] = normalize_concept(tgt)

    for triple in graph.triples:
        src, role, tgt = triple
        if role == ":name" and tgt in var_to_concept:
            name_parts = []
            for op_role in [":op1", ":op2", ":op3", ":op4", ":op5"]:
                for t in graph.triples:
                    if t[0] == tgt and t[1] == op_role:
                        name_parts.append(t[2].strip('"').lower())
            if name_parts:
                var_to_concept[src] = " ".join(name_parts)
    return var_to_concept

def build_nx_graph(graph, var_to_concept):
    G = nx.Graph()
    for var in var_to_concept:
        G.add_node(var, label=var_to_concept[var])
    for src, role, tgt in graph.triples:
        if role.startswith(":"):
            G.add_edge(src, tgt, role=role)
    return G

def build_keyword_embedding_matrix(amr_str, keywords):
    graph = penman.decode(amr_str)
    var_to_concept = extract_named_concepts(graph)
    nx_graph = build_nx_graph(graph, var_to_concept)

    # Match keyword → AMR variable
    keyword_to_var = {}
    for var, concept in var_to_concept.items():
        for kw in keywords:
            if concept == kw:
                keyword_to_var[kw] = var

    K = len(keywords)
    kw_index = {kw: i for i, kw in enumerate(keywords)}
    matrix = [[None for _ in range(K)] for _ in range(K)]

    for i in range(K):
        for j in range(K):
            kw1, kw2 = keywords[i], keywords[j]
            var1 = keyword_to_var.get(kw1)
            var2 = keyword_to_var.get(kw2)

            if not var1 or not var2:
                # Try to find any node labeled as the keyword
                var1 = next((v for v, l in nx_graph.nodes(data="label") if l == kw1), None)
                var2 = next((v for v, l in nx_graph.nodes(data="label") if l == kw2), None)

            if var1 and var2 and nx.has_path(nx_graph, var1, var2):
                try:
                    path = nx.shortest_path(nx_graph, var1, var2)
                    phrase_parts = []
                    for k in range(len(path) - 1):
                        src_var = path[k]
                        tgt_var = path[k+1]
                        src_label = nx_graph.nodes[src_var].get("label", "concept")
                        tgt_label = nx_graph.nodes[tgt_var].get("label", "concept")
                        role = nx_graph[src_var][tgt_var].get("role", ":related")
                        phrase_parts.append(f"{src_label} is the {role[1:]} of {tgt_label}")
                    phrase = " -> ".join(phrase_parts)
                    embedding = sbert_model.encode(phrase).tolist()
                    matrix[i][j] = embedding
                except Exception:
                    continue
    return matrix

# Main loop
for entry in tqdm(keyword_data, desc="Building dense SBERT embedding matrices"):
    doc_id = entry["id"]
    keywords = [kw.lower() for kw in entry["keywords"]]

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
            matrix = build_keyword_embedding_matrix(amr_str, keywords)
            matrices[key] = matrix
        except Exception as e:
            print(f"[ERROR] {doc_id} ({key}): {e}")

    output.append({
        "id": doc_id,
        "keywords": keywords,
        "body_matrix": matrices.get("body_amr", []),
        "summary_matrix": matrices.get("summary_amr", [])
    })

# Save results
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"[✓] Hybrid SBERT-embedded matrices saved → {output_path}")

