# visualize_discourse_graphs.py
import os
import json
from graphviz import Digraph

# ✅ Point to Graphviz binary
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Paths
discourse_path = "corpus/discourse_tags.json"
sents_path = "corpus/structured_docs.json"
output_dir = "graphs/discourse"
os.makedirs(output_dir, exist_ok=True)

# Load discourse structure
with open(discourse_path, "r", encoding="utf-8") as f:
    discourse_data = json.load(f)

# Optionally load sentence text
sent_texts = {}
if os.path.exists(sents_path):
    with open(sents_path, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            sent_texts[entry["id"]] = {
                "body": entry["body_sents"],
                "summary": entry["summary_sents"]
            }

def render_discourse(doc_id, side, structure):
    dot = Digraph(format="png")
    dot.attr(rankdir="TB", label=f"{side.capitalize()} Discourse: {doc_id}", fontsize="18", labelloc="t")

    texts = sent_texts.get(doc_id, {}).get(side, [])

    for node in structure:
        idx = node["index"]
        rel = node["relation_to_prev"]
        depth = node.get("depth", "-")
        preview = texts[idx][:60] + "..." if idx < len(texts) else ""

        label = f"{idx}: {rel}\n{preview}"
        dot.node(str(idx), label, shape="box", style="filled", fillcolor="lightyellow")

    for node in structure:
        if node["parent"] is not None:
            dot.edge(str(node["parent"]), str(node["index"]))

    out_path = os.path.join(output_dir, f"{doc_id}_{side}.png")
    try:
        dot.render(out_path, cleanup=True)
        print(f"[✓] Rendered {side} for {doc_id}")
    except Exception as e:
        print(f"[!] Failed {side} {doc_id}: {e}")

# Main loop
for doc_id, entry in discourse_data.items():
    try:
        if "body" in entry:
            render_discourse(doc_id, "body", entry["body"])
        if "summary" in entry:
            render_discourse(doc_id, "summary", entry["summary"])
    except Exception as e:
        print(f"[!] Error rendering {doc_id}: {e}")
