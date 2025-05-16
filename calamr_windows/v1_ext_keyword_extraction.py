# ext_keywords_extraction.py

import json
import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Paths
input_path = "corpus/input_docs_2.json"
output_path = "corpus/keywords.json"

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    docs = json.load(f)

# Prepare text corpus
texts = [item["body"] for item in docs]

# --------------------
# 1. Multipart TF-IDF
# --------------------
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)
tfidf_keywords = []
for row in tfidf_matrix:
    scores = zip(vectorizer.get_feature_names_out(), row.toarray().flatten())
    sorted_kws = sorted(scores, key=lambda x: -x[1])
    tfidf_keywords.append([kw for kw, score in sorted_kws if score > 0])

# --------------------
# 2. Topic Modeling (LDA via sklearn)
# --------------------
# count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
count_vectorizer = CountVectorizer(stop_words='english', min_df=1, max_df=1.0)

count_data = count_vectorizer.fit_transform(texts)
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(count_data)
feature_names = count_vectorizer.get_feature_names_out()

lda_keywords = []
for topic_dist in lda_model.transform(count_data):
    topic_idx = np.argmax(topic_dist)
    topic = lda_model.components_[topic_idx]
    top_indices = topic.argsort()[::-1]
    lda_keywords.append([feature_names[i] for i in top_indices if topic[i] > 0])

# --------------------
# 3. KeyBERT
# --------------------
kw_model = KeyBERT()
keybert_keywords = []
for doc in texts:
    try:
        kw = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=100)
        keybert_keywords.append([phrase for phrase, score in kw if score > 0])
    except Exception:
        keybert_keywords.append([])

# Keyword count diagnostics
lengths = {
    "tfidf": Counter(),
    "lda": Counter(),
    "keybert": Counter()
}

# Save all outputs
results = []
for idx, item in enumerate(docs):
    tfidf_list = tfidf_keywords[idx]
    lda_list = lda_keywords[idx]
    keybert_list = keybert_keywords[idx]

    lengths["tfidf"][len(tfidf_list)] += 1
    lengths["lda"][len(lda_list)] += 1
    lengths["keybert"][len(keybert_list)] += 1

    results.append({
        "id": item["id"],
        "tfidf": tfidf_list,
        "lda": lda_list,
        "keybert": keybert_list
    })

# Print keyword count distributions
print("TF-IDF keyword count distribution:", dict(lengths["tfidf"]))
print("LDA keyword count distribution:", dict(lengths["lda"]))
print("KeyBERT keyword count distribution:", dict(lengths["keybert"]))

# Save to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"[✓] Extracted keywords using 3 methods for {len(docs)} documents.")