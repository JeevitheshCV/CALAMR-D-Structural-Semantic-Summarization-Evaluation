# 🧠 CALAMR+D: Discourse-Aware Structural Semantic Summarization Evaluation

**CALAMR+D** is a novel, graph-based evaluation framework for **abstractive summarization**, designed to move beyond traditional string-matching metrics like ROUGE and BLEU. By integrating **Abstract Meaning Representation (AMR)** and **discourse parsing**, CALAMR+D provides a fine-grained, interpretable analysis of summary quality—focusing on **faithfulness**, **semantic alignment**, and **hallucination detection**.

> This is an enhanced version of [CALAMR](https://aclanthology.org/2024.lrec-main.507), enriched with **discourse-aware alignment** and advanced **semantic flow optimization**.

---

## 🚀 Motivation: Why CALAMR+D?

Traditional metrics struggle with:
- Lexical variation (e.g., paraphrases)
- Semantic drift or hallucinations
- Disregard for rhetorical/structural importance

**CALAMR+D** aims to resolve these limitations via:
- Parsing **AMR graphs** for rich semantic meaning
- Using **discourse trees** to weigh rhetorical significance
- Performing **semantic alignment** with discourse-aware weighting
- Leveraging **flow optimization** to detect hallucinations
- Scoring based on **semantic fidelity**, **coverage**, **centrality**, and **penalties**

---

## 🧩 Pipeline Overview

The evaluation framework consists of five modular stages:

### 1️⃣ AMR Parsing

python calamr_windows/calamr_step_1_parse_amr.py

- Parses each sentence with SPRING AMR Parser
- Outputs AMR graphs in JSON

### 2️⃣ Discourse Parsing

python calamr_windows/calamr_plus_d_step_2_discourse_parse.py

- Annotates sentences with rhetorical relations
- Builds discourse trees and computes depth (for salience)

### 3️⃣ Semantic Alignment (Discourse-Aware)

python calamr_windows/calamr_plus_d_step_3_align_with_discourse.py

- Embeds AMRs via Sentence-BERT
- Adjusts cosine similarities using discourse weights

### 4️⃣ Flow Network Optimization
python calamr_windows/calamr_plus_d_step_4_flow.py

- Constructs bipartite alignment flow graphs
- Applies max-flow optimization to filter weak/hallucinated alignments

### 5️⃣ Scoring

python calamr_windows/calamr_plus_d_step_5_score.py

- Calculates final composite score using:
  - ✅ Semantic Similarity
  - 📈 Coverage
  - 📉 Discourse Depth Penalty
  - ⚠️ Hallucination Penalty

---

## 📐 Scoring Formula


Composite Score = Semantic Mean + Coverage + (1 - Depth Penalty) - Hallucination Penalty


Where:
- **Semantic Mean** = Avg. similarity of aligned sentence pairs
- **Coverage** = % of summary sentences that align
- **Depth Penalty** = Avg. discourse depth (lower is better)
- **Hallucination Penalty** = % of unaligned summary content

---

## 📁 Directory Structure


CALAMR-D-Structural-Semantic-Summarization-Evaluation/
├── calamr_windows/              # Core pipeline (Steps 1-5)
├── dataset/                     # Input summaries & sources
├── models/                      # SPRING model directory
├── outputs/                     # AMRs, discourse trees, scores
├── utils/                       # Preprocessing helpers
├── requirements.txt             # Python dependencies
└── README.md                    # You are here!


---

## ⚙️ Installation & Setup

### 1. Clone the Repo

git clone https://github.com/JeevitheshCV/CALAMR-D-Structural-Semantic-Summarization-Evaluation.git
cd CALAMR-D-Structural-Semantic-Summarization-Evaluation


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Download AMR Parser Model

python -m amrlib.download_model model_parse_xfm_bart_large-v0_1_0

> Place the downloaded model inside the `models/` directory.

---

## 📊 Output Files

For each input document pair:
- `parsed_amrs.json`: Sentence-level AMRs
- `discourse_tree.json`: Discourse depth and relations
- `alignment_graph.json`: Semantic + discourse flow alignment
- `score_output.json` or `.csv`: Composite scoring breakdown

Visualization tools: `networkx`, `pygraphviz`, custom dashboards

---

## 🔍 Use Cases

- 📏 Evaluate outputs of BART, T5, Pegasus, etc.
- 🧪 Detect hallucinated or unsupported summary content
- 🧭 Visualize rhetorical & semantic alignment
- 🔬 Inspect training/validation datasets for summarization

---

## 🧪 Experimental Results

On CNN/DailyMail evaluation set:
- Strong central discourse alignment
- Effective hallucination flagging
- Higher correlation with human judgments vs ROUGE
- Intuitive scoring breakdowns and visualization potential

---

## 📚 References

- Opitz et al., *CALAMR*, LREC 2024  
- Zhang et al., *BERTScore*, ICLR 2020  
- Lin, C.-Y., *ROUGE*, ACL 2004  
- [SPRING AMR Parser](https://github.com/SapienzaNLP/spring)

---

## 👤 Author

**Jeevithesh C V**  
📧 jeevithesh.cv07@gmail.com  
