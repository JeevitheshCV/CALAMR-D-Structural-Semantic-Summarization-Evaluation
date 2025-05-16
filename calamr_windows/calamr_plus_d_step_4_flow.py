# # calamr_plus_d_step_4_flow.py
# import os
# import json
# import networkx as nx
# from collections import defaultdict
#
# input_path = os.path.join("corpus", "alignment_with_discourse.json")
# output_path = os.path.join("corpus", "flow_results.json")
# os.makedirs("corpus", exist_ok=True)
#
# # Load alignment data
# with open(input_path, "r", encoding="utf-8") as f:
#     all_alignments = json.load(f)
#
# # Group by document ID
# doc_alignments = defaultdict(list)
# for a in all_alignments:
#     doc_alignments[a["id"]].append(a)
#
# results = []
#
# for doc_id, edges in doc_alignments.items():
#     summary_nodes = {e["summary_index"] for e in edges}
#     body_nodes = {e["body_index"] for e in edges}
#     num_sum = max(summary_nodes) + 1
#     num_body = max(body_nodes) + 1
#
#     G = nx.DiGraph()
#     G.add_node("s")
#     G.add_node("t")
#
#     for i in range(num_sum):
#         G.add_edge("s", f"s_{i}", capacity=1.0)
#     for j in range(num_body):
#         G.add_edge(f"b_{j}", "t", capacity=1.0)
#
#     for e in edges:
#         if e["adjusted_score"] > 0.2:
#             G.add_edge(f"s_{e['summary_index']}", f"b_{e['body_index']}", capacity=e["adjusted_score"])
#
#     try:
#         flow_value, flow_dict = nx.maximum_flow(G, "s", "t")
#     except Exception as ex:
#         print(f"[!] Max flow failed for {doc_id}: {ex}")
#         continue
#
#     aligned_summary_ids = set()
#     matched_depths = []
#
#     for i in range(num_sum):
#         s_node = f"s_{i}"
#         for b_node, fval in flow_dict.get(s_node, {}).items():
#             if fval > 0:
#                 aligned_summary_ids.add(i)
#                 for e in edges:
#                     if e["summary_index"] == i and f"b_{e['body_index']}" == b_node:
#                         matched_depths.append(e["body_depth"])
#                         break
#
#     total_sum = num_sum
#     unaligned = [i for i in range(total_sum) if i not in aligned_summary_ids]
#     coverage = len(aligned_summary_ids) / total_sum if total_sum > 0 else 0
#     avg_depth = sum(matched_depths) / len(matched_depths) if matched_depths else None
#     halluc_penalty = len(unaligned) / total_sum if total_sum > 0 else 0
#
#     results.append({
#         "id": doc_id,
#         "flow_score": round(flow_value, 3),
#         "summary_sentences": total_sum,
#         "aligned": len(aligned_summary_ids),
#         "unaligned": unaligned,
#         "alignment_ratio": round(coverage, 3),
#         "avg_body_depth_matched": round(avg_depth, 3) if avg_depth else None,
#         "hallucination_penalty": round(halluc_penalty, 3)
#     })
#
# # Save all results
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2)
#
# print(f"[✓] Flow alignment results saved to {output_path}")


# calamr_plus_d_step_4_flow.py
import os
import json
import networkx as nx
from collections import defaultdict

input_path = os.path.join("corpus", "alignment_with_discourse.json")
output_path = os.path.join("corpus", "flow_results.json")
os.makedirs("corpus", exist_ok=True)

# Load alignment data
with open(input_path, "r", encoding="utf-8") as f:
    all_alignments = json.load(f)

# Group by document ID
doc_alignments = defaultdict(list)
for a in all_alignments:
    doc_alignments[a["id"]].append(a)

results = []

print(f"[•] Running flow alignment for {len(doc_alignments)} documents...")

for doc_id, edges in doc_alignments.items():
    summary_nodes = {e["summary_index"] for e in edges}
    body_nodes = {e["body_index"] for e in edges}
    num_sum = max(summary_nodes) + 1
    num_body = max(body_nodes) + 1

    G = nx.DiGraph()
    G.add_node("s")
    G.add_node("t")

    for i in range(num_sum):
        G.add_edge("s", f"s_{i}", capacity=1.0)
    for j in range(num_body):
        G.add_edge(f"b_{j}", "t", capacity=1.0)

    for e in edges:
        if e["adjusted_score"] > 0.2:
            G.add_edge(f"s_{e['summary_index']}", f"b_{e['body_index']}", capacity=e["adjusted_score"])

    try:
        flow_value, flow_dict = nx.maximum_flow(G, "s", "t")
    except Exception as ex:
        print(f"[!] Max flow failed for {doc_id}: {ex}")
        continue

    aligned_summary_ids = set()
    matched_depths = []

    for i in range(num_sum):
        s_node = f"s_{i}"
        for b_node, fval in flow_dict.get(s_node, {}).items():
            if fval > 0:
                aligned_summary_ids.add(i)
                for e in edges:
                    if e["summary_index"] == i and f"b_{e['body_index']}" == b_node:
                        matched_depths.append(e["body_depth"])
                        break

    total_sum = num_sum
    unaligned = [i for i in range(total_sum) if i not in aligned_summary_ids]
    coverage = len(aligned_summary_ids) / total_sum if total_sum > 0 else 0
    avg_depth = sum(matched_depths) / len(matched_depths) if matched_depths else None
    halluc_penalty = len(unaligned) / total_sum if total_sum > 0 else 0

    results.append({
        "id": doc_id,
        "flow_score": round(flow_value, 3),
        "summary_sentences": total_sum,
        "aligned": len(aligned_summary_ids),
        "unaligned": unaligned,
        "alignment_ratio": round(coverage, 3),
        "avg_body_depth_matched": round(avg_depth, 3) if avg_depth else None,
        "hallucination_penalty": round(halluc_penalty, 3)
    })

    print(f"[✓] {doc_id} → Max flow: {flow_value:.2f} | Aligned: {len(aligned_summary_ids)}/{total_sum}")
    if unaligned:
        print(f"    • Unaligned summary indices: {unaligned}")
    print(f"    • Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")

# Save all results
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[🎯] Flow alignment results saved to {output_path}")
