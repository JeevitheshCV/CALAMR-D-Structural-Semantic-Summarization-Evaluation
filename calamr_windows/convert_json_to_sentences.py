# convert_json_to_sentences.py
import os
import json
import spacy

nlp = spacy.load("en_core_web_sm")

input_path = os.path.join("corpus", "input_docs.json")
output_path = os.path.join("corpus", "structured_docs.json")
os.makedirs("corpus", exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

structured = []

for entry in raw_docs:
    doc_id = entry["id"]
    body_doc = nlp(entry["body"])
    summary_doc = nlp(entry["summary"])

    body_sents = [sent.text.strip() for sent in body_doc.sents if sent.text.strip()]
    summary_sents = [sent.text.strip() for sent in summary_doc.sents if sent.text.strip()]

    structured.append({
        "id": doc_id,
        "body_sents": body_sents,
        "summary_sents": summary_sents
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured, f, indent=2)

print(f"[✓] Saved structured sentences to {output_path}")
