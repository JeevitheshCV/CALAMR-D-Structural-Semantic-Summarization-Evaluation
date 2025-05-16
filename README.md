# CALAMR+D: A Discourse-Aware Graph Alignment Framework for Summarization Evaluation

**CALAMR+D** is a symbolic, interpretable evaluation framework designed to assess the quality of **abstractive summaries** using both semantic structure (via AMR) and discourse structure. It extends the original **CALAMR** framework by integrating **discourse parsing**, **hierarchical graph alignment**, and **hallucination detection**.

---

## 📌 Project Purpose

Traditional evaluation metrics like **ROUGE** or **METEOR** rely on lexical overlap, failing to measure true semantic or rhetorical faithfulness in abstractive summarization.

**CALAMR+D solves this by:**
- Parsing the source and summary into **Abstract Meaning Representation (AMR)** graphs.
- Extracting **discourse relations** between sentences (e.g., reason, contrast, elaboration).
- Aligning content using **Sentence-BERT cosine similarity** and **discourse weighting**.
- Optimizing **max-flow matching** between summary and source nodes.
- Detecting **hallucinated content** (summary information not grounded in source).

---

## 🧠 Key Features

- ✅ Semantic Graph Alignment using AMR parsing
- ✅ Sentence Embedding with SBERT for alignment
- ✅ Discourse-Aware Scoring using rhetorical relation depth
- ✅ Flow Network Optimization for global sentence mapping
- ✅ Composite Scoring: semantic + discourse + hallucination penalty
- ✅ Full Visualizations: AMR graphs, discourse trees

---

## 📂 Repository Structure

```bash
calamr_plus_d/
├── corpus/                     # Preprocessed input and output files
│   ├── input_docs.json         # Structured source-summary pairs
│   ├── parsed_amrs.json        # AMR graphs for source/summary
│   ├── discourse_tags.json     # Discourse relations and tree data
│   ├── flow_results.json       # Alignment & max-flow results
│   ├── calamr_plus_d_outputs.json # Final composite scores
├── graphs/                     # Graph visualizations (.png)
│   ├── body/                   # AMR graphs for body
│   ├── summary/                # AMR graphs for summary
│   ├── discourse/              # Discourse tree visualizations
├── src/                        # Source code for pipeline
│   ├── convert_json.py
│   ├── parse_amr.py
│   ├── discourse_parser.py
│   ├── build_discourse_graph.py
│   ├── align_amrs_with_discourse.py
│   ├── flow_alignment.py
│   ├── calamr_plus_d_score.py
│   ├── graph_evaluator.py
│   ├── visualize_amr_graphs.py
│   ├── visualize_discourse_graphs.py
├── requirements.txt
├── README.md
