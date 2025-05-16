# # STEP 1: Imports
# import os
# import json
# import penman
# from graphviz import Digraph
#
# # Paths
# parsed_amrs_path = 'corpus/parsed_amrs_structured.json'
# structured_docs_path = 'corpus/structured_docs.json'
# output_json_path = 'corpus/amr+disclosuretags_combined_graph.json'
# body_graph_dir = 'graphs/amr+d/body'
# summary_graph_dir = 'graphs/amr+d/summary'
#
# # Create directories
# os.makedirs(body_graph_dir, exist_ok=True)
# os.makedirs(summary_graph_dir, exist_ok=True)
# os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#
# # Load files
# with open(parsed_amrs_path, 'r', encoding='utf-8') as f:
#     parsed_amrs = {doc['id']: doc for doc in json.load(f)}
#
# with open(structured_docs_path, 'r', encoding='utf-8') as f:
#     structured_docs = {doc['id']: doc for doc in json.load(f)}
#
# # Helper to detect discourse relations
# def detect_relation(sent1, sent2):
#     s2 = sent2.lower()
#     if any(word in s2 for word in ['because', 'since', 'due to']):
#         return ':cause'
#     if any(word in s2 for word in ['but', 'however', 'although', 'though']):
#         return ':contrast'
#     if any(word in s2 for word in ['also', 'moreover', 'furthermore']):
#         return ':addition'
#     if any(word in s2 for word in ['thus', 'therefore', 'hence']):
#         return ':result'
#     return ':elaboration'
#
# # STEP 2: Helper to safely merge AMRs in hierarchical rooted structure
# def merge_amrs_hierarchical(amrs_to_merge, sentences, id_prefix):
#     graphs = []
#     for amr_text in amrs_to_merge:
#         try:
#             clean_amr = "\n".join(
#                 line for line in amr_text.splitlines()
#                 if not line.strip().startswith("#")
#             )
#             g = penman.decode(clean_amr)
#             graphs.append(g)
#         except Exception as e:
#             print(f"[!] Skipping bad AMR due to decode error: {e}")
#
#     if not graphs:
#         return ""
#
#     merged_triples = []
#     root_node = f"root-{id_prefix}"
#     merged_triples.append((root_node, ":instance", "ROOT"))
#
#     prev_node = root_node
#     for idx, g in enumerate(graphs):
#         g_root = g.top
#
#         # Validate and add triples safely
#         for h, r, t in g.triples:
#             if not r.startswith(":"):
#                 print(f"[!] Skipping malformed triple: ({h}, {r}, {t})")
#                 continue
#             merged_triples.append((h, r, t))
#
#         # Attach child to parent with a discourse relation
#         if idx == 0:
#             merged_triples.append((root_node, ":discourse", g_root))
#         else:
#             relation = detect_relation(sentences[idx - 1], sentences[idx])
#             merged_triples.append((prev_node, relation, g_root))
#
#         prev_node = g_root
#
#     merged_graph = penman.Graph(merged_triples, top=root_node)
#     return penman.encode(merged_graph)
#
# # STEP 3: Main Processing
# final_entries = []
#
# for doc_id, doc in parsed_amrs.items():
#     print(f"Processing {doc_id}...")
#
#     body_sents = structured_docs[doc_id]['body_sents']
#     body_amrs = doc['body_amrs']
#
#     summary_sents = structured_docs[doc_id]['summary_sents']
#     summary_amrs = doc['summary_amrs']
#
#     # Body merging
#     body_merge_amrs = []
#     for i, sent in enumerate(body_sents):
#         if i < len(body_amrs):
#             body_merge_amrs.append(body_amrs[i])
#
#     merged_body = merge_amrs_hierarchical(body_merge_amrs, body_sents, id_prefix=f"{doc_id}_body")
#
#     # Summary merging
#     summary_merge_amrs = []
#     for i, sent in enumerate(summary_sents):
#         if i < len(summary_amrs):
#             summary_merge_amrs.append(summary_amrs[i])
#
#     merged_summary = merge_amrs_hierarchical(summary_merge_amrs, summary_sents, id_prefix=f"{doc_id}_summary")
#
#     final_entries.append({
#         'id': doc_id,
#         'body_amr': merged_body,
#         'summary_amr': merged_summary
#     })
#
# # Save final JSON
# with open(output_json_path, 'w', encoding='utf-8') as f:
#     json.dump(final_entries, f, indent=2)
#
# print(f"[✓] Saved merged AMRs to {output_json_path}")
#
# # STEP 4: Visualization
# def render_amr_graph(amr_string, title, save_path):
#     try:
#         clean_amr = "\n".join(
#             line for line in amr_string.splitlines()
#             if not line.strip().startswith("#")
#         ).strip()
#
#         g = penman.decode(clean_amr)
#         dot = Digraph(comment=title, format='png')
#         dot.attr(label=title, labelloc='t', fontsize='20')
#
#         for h, r, t in g.triples:
#             if r.startswith(":"):
#                 if r == ':instance':
#                     dot.node(h, f"{h} / {t}", shape='ellipse', style='filled', fillcolor='lightblue')
#                 else:
#                     dot.edge(h, t, label=r)
#             else:
#                 print(f"[!] Skipped malformed relation: {h} {r} {t}")
#
#         dot.render(save_path, cleanup=True)
#         print(f"[✓] Saved Graph: {save_path}.png")
#     except Exception as e:
#         print(f"[!] Failed to render {title}: {e}")
#
# # Render all final graphs
# for entry in final_entries:
#     doc_id = entry['id']
#
#     body_path = os.path.join(body_graph_dir, f"{doc_id}_body")
#     summary_path = os.path.join(summary_graph_dir, f"{doc_id}_summary")
#
#     render_amr_graph(entry['body_amr'], f"{doc_id} Body", body_path)
#     render_amr_graph(entry['summary_amr'], f"{doc_id} Summary", summary_path)
#
# print("[✓] All graphs rendered and saved!")

import os
import json
import penman
from graphviz import Digraph

# Paths
input_json_path = 'corpus/amr+disclosuretags_combined_graph.json'  # <-- path to your JSON
body_graph_dir = 'graphs/amr+d/body'
summary_graph_dir = 'graphs/amr+d/summary'

# Create output directories
os.makedirs(body_graph_dir, exist_ok=True)
os.makedirs(summary_graph_dir, exist_ok=True)

# Load combined AMR JSON
with open(input_json_path, 'r', encoding='utf-8') as f:
    combined_amrs = json.load(f)


# Function to render a graph
def render_amr_graph(amr_string, title, save_path):
    try:
        g = penman.decode(amr_string)
        dot = Digraph(comment=title, format='png')
        dot.attr(label=title, labelloc='t', fontsize='20')

        for h, r, t in g.triples:
            if r == ':instance':
                dot.node(h, f"{h} / {t}", shape='ellipse', style='filled', fillcolor='lightblue')
            else:
                dot.edge(h, t, label=r)

        dot.render(save_path, cleanup=True)
        print(f"[✓] Saved graph: {save_path}.png")
    except Exception as e:
        print(f"[!] Failed to render {title}: {e}")


# Render graphs
for entry in combined_amrs:
    doc_id = entry['id']

    # Render body graph
    body_amr = entry['body_amr']
    body_save_path = os.path.join(body_graph_dir, f"{doc_id}_body")
    render_amr_graph(body_amr, f"{doc_id} Body", body_save_path)

    # Render summary graph
    summary_amr = entry['summary_amr']
    summary_save_path = os.path.join(summary_graph_dir, f"{doc_id}_summary")
    render_amr_graph(summary_amr, f"{doc_id} Summary", summary_save_path)

print("[✓] All graphs from JSON rendered successfully!")
