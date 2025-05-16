# # visualize_amr_graphs.py
# import os
# import json
# import penman
# import pydot
#
# # Config
# input_path = "corpus/parsed_amrs.json"
# output_dir = "visuals/amr_graphs"
# os.makedirs(output_dir, exist_ok=True)
#
# # Load parsed AMRs
# with open(input_path, "r", encoding="utf-8") as f:
#     amrs = json.load(f)
#
# def render_amr(amr_string, title, filename):
#     try:
#         graph = penman.decode(amr_string)
#         dot = pydot.Dot(graph_type='digraph', rankdir='LR', label=title, fontsize=16, labelloc="t")
#
#         for triple in graph.triples:
#             if triple[1] == ':instance':
#                 node = pydot.Node(triple[0], label=f"{triple[0]} / {triple[2]}", shape='ellipse', style='filled', fillcolor='lightblue')
#                 dot.add_node(node)
#             else:
#                 dot.add_edge(pydot.Edge(triple[0], triple[2], label=triple[1]))
#
#         dot.write_png(filename)
#         print(f"[✓] Saved: {filename}")
#     except Exception as e:
#         print(f"[!] Failed to render: {e}")
#
# # Interactive or manual select
# def render_by_id(target_id):
#     for entry in amrs:
#         if entry["id"] == target_id:
#             body_path = os.path.join(output_dir, f"{target_id}_body.png")
#             summary_path = os.path.join(output_dir, f"{target_id}_summary.png")
#             render_amr(entry["body_amr"], f"{target_id} – Body", body_path)
#             render_amr(entry["summary_amr"], f"{target_id} – Summary", summary_path)
#             break
#     else:
#         print(f"[!] ID {target_id} not found.")
#
# # Example usage (modify or call from CLI wrapper)
# if __name__ == "__main__":
#     target = input("Enter document ID to visualize (e.g., doc_001): ").strip()
#     render_by_id(target)

# visualize_amr_graphs.py
import os
import json
import penman
import pydot

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# visualize_amr_graphs.py
import os
import json
import penman
from graphviz import Digraph

# Ensure Graphviz is reachable
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Paths
input_path = "corpus/parsed_amrs.json"
body_dir = "graphs/amr/body"
summary_dir = "graphs/amr/summary"
os.makedirs(body_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# Load parsed AMRs
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def render_amr(amr_str, title, path):
    try:
        graph = penman.decode(amr_str)
        dot = Digraph(comment=title, format='png')
        dot.attr(label=title, labelloc='t', fontsize='20')

        for h, r, t in graph.triples:
            if r == ":instance":
                dot.node(h, f"{h} / {t}", shape='ellipse', style='filled', fillcolor='lightblue')
            else:
                dot.edge(h, t, label=r)

        dot.render(path, cleanup=True)
        print(f"[✓] Saved: {path}.png")
    except Exception as e:
        print(f"[!] Failed to render {title}: {e}")

# Loop and render
for entry in data:
    doc_id = entry["id"]
    body_amr = entry.get("body_amr", "").strip()
    summary_amr = entry.get("summary_amr", "").strip()

    if not body_amr or not summary_amr:
        print(f"[!] Skipping empty AMR: {doc_id}")
        continue

    render_amr(body_amr, f"Body: {doc_id}", os.path.join(body_dir, doc_id))
    render_amr(summary_amr, f"Summary: {doc_id}", os.path.join(summary_dir, doc_id))
