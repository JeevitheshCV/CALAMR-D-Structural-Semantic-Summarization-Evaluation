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


\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{color}
\geometry{margin=1in}

\title{\textbf{CALAMR+D: Discourse-Aware Structural Semantic Summarization Evaluation}}
\author{Jeevithesh C V}
\date{}

\begin{document}
\maketitle

\section*{Abstract}
\textbf{CALAMR+D} is a structural and semantic evaluation framework for abstractive summarization that integrates Abstract Meaning Representation (AMR) parsing with discourse structure analysis. It extends the CALAMR methodology by incorporating hierarchical rhetorical roles, centrality estimation, and hallucination detection through flow-based alignment. This document describes the full pipeline, implementation steps, mathematical scoring metrics, and project structure.

\section{Introduction}
Conventional summarization evaluation metrics such as ROUGE and BLEU rely heavily on lexical overlap, which fails to assess paraphrased or semantically equivalent summaries. CALAMR+D aims to go beyond surface-level evaluation by using:
\begin{itemize}
  \item \textbf{AMR Graphs}: Representing the semantic structure of each sentence.
  \item \textbf{Discourse Trees}: Capturing rhetorical relations like \textit{reason}, \textit{elaboration}, and \textit{summary}.
  \item \textbf{Max-Flow Alignment}: Matching summary nodes to source nodes using semantic + discourse weights.
\end{itemize}

\section{Pipeline Execution}
The evaluation process consists of five sequential stages:

\begin{enumerate}
  \item \texttt{calamr\_step\_1\_parse\_amr.py} \\
        Parses source and summary sentences into AMR graphs using the SPRING parser.

  \item \texttt{calamr\_plus\_d\_step\_2\_discourse\_parse.py} \\
        Assigns rhetorical roles and constructs hierarchical discourse trees.

  \item \texttt{calamr\_plus\_d\_step\_3\_align\_with\_discourse.py} \\
        Calculates cosine similarity (via Sentence-BERT) and adjusts with discourse-based weights.

  \item \texttt{calamr\_plus\_d\_step\_4\_flow.py} \\
        Builds and optimizes a bipartite flow graph to determine alignments.

  \item \texttt{calamr\_plus\_d\_step\_5\_score.py} \\
        Computes composite scores including hallucination penalty and discourse centrality.
\end{enumerate}

\section{Scoring Methodology}
The final evaluation score is defined as:

\[
\text{Composite Score} = \text{Semantic Mean} + \text{Coverage} + (1 - \text{Depth Penalty}) - \text{Hallucination Penalty}
\]

\begin{itemize}
  \item \textbf{Semantic Mean}: Average cosine similarity of aligned sentence pairs.
  \item \textbf{Coverage}: Fraction of summary nodes matched with source nodes.
  \item \textbf{Depth Penalty}: Penalization based on discourse depth (peripheral nodes score lower).
  \item \textbf{Hallucination Penalty}: Penalty for unaligned or unsupported summary content.
\end{itemize}

\section{Repository Structure}
\begin{verbatim}
CALAMR-D-Structural-Semantic-Summarization-Evaluation/
├── calamr_windows/         # All pipeline execution scripts
├── dataset/                # Input texts (CNN/DailyMail format)
├── models/                 # AMR parsing models (e.g., SPRING)
├── outputs/                # Result graphs, scores, visualizations
├── utils/                  # Helper scripts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
\end{verbatim}

\section{Setup and Installation}
\subsection*{Clone the Repository}
\begin{lstlisting}[language=bash]
git clone https://github.com/JeevitheshCV/CALAMR-D-Structural-Semantic-Summarization-Evaluation.git
cd CALAMR-D-Structural-Semantic-Summarization-Evaluation
\end{lstlisting}

\subsection*{Install Dependencies}
\begin{lstlisting}[language=bash]
pip install -r requirements.txt
\end{lstlisting}

\subsection*{Download AMR Model}
\begin{lstlisting}[language=python]
import amrlib
amrlib.download_model('model_parse_xfm_bart_large-v0_1_0')
\end{lstlisting}

\section{Experimental Results}
Tested on the CNN/DailyMail dataset, CALAMR+D shows:
\begin{itemize}
  \item Higher alignment accuracy for core content.
  \item Zero hallucination in selected model-generated summaries.
  \item Improved interpretability and correlation with human judgment over ROUGE.
\end{itemize}

\section{Output Artifacts}
Each run produces:
\begin{itemize}
  \item AMR Graphs (.json)
  \item Discourse Trees (.json)
  \item Flow Network Visualizations (.png)
  \item Composite Scores (.csv / .json)
\end{itemize}

\section{References}
\begin{itemize}
  \item Opitz et al., \textit{CALAMR: Component Alignment for Abstract Meaning Representation}, LREC 2024.
  \item Zhang et al., \textit{BERTScore: Evaluating Text Generation with BERT}, ICLR 2020.
  \item Lin, C.-Y., \textit{ROUGE: A Package for Automatic Evaluation of Summaries}, ACL 2004.
  \item SapienzaNLP, \textit{SPRING: Semantic Parser for AMR}, GitHub.
\end{itemize}

\section{Author}
\textbf{Jeevithesh C V} \\
Email: \texttt{jeevithesh.cv@example.com}

\section{License}
This project is licensed under the \textbf{MIT License}.

\end{document}

