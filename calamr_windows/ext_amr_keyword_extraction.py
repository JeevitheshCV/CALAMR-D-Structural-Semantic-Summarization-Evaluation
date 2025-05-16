# # ext_amr_topic_keywords.py
#
# import json
# import os
# import penman
# import re
# import nltk
# from collections import defaultdict
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
#
# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()
#
# # Paths
# input_path = "corpus/parsed_amrs.json"
# output_path = "corpus/amr_topic_keywords.json"
#
# # Load parsed AMRs
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# concept_docs = []
# id_list = []
# concepts_per_doc = {}
#
# # Utility: Strip AMR suffixes like -01
# strip_suffix = lambda c: re.sub(r"-\d{2}$", "", c.lower())
#
# # Step 1: Extract concepts from AMR graph structure
# for entry in data:
#     doc_id = entry.get("id")
#     body_amr = entry.get("body_amr", "")
#     try:
#         graph = penman.decode(body_amr)
#         useful_concepts = set()
#
#         # Build variable → concept map
#         var_to_concept = {}
#         for var, _, concept in graph.instances():
#             var_to_concept[var] = strip_suffix(concept)
#
#         # Build edge maps
#         child_map = defaultdict(list)
#         for source, role, target in graph.triples:
#             child_map[source].append((role, target))
#
#         # Rule 1: Add root concept
#         for var, role, concept in graph.instances():
#             if var == graph.top:
#                 useful_concepts.add(strip_suffix(concept))
#
#         # Rule 2: ARG concepts with children
#         for source, role, target in graph.triples:
#             if role.startswith(":ARG") and target in child_map:
#                 if target in var_to_concept:
#                     useful_concepts.add(var_to_concept[target])
#
#         # Rule 3: Extract :opN values from :name structures
#         for var, concept in var_to_concept.items():
#             for role, target in child_map.get(var, []):
#                 if role == ":name" and target in child_map:
#                     for subrole, literal in child_map[target]:
#                         if subrole.startswith(":op") and literal.startswith('"'):
#                             literal_value = literal.strip('"').lower()
#                             if len(literal_value) > 1:
#                                 useful_concepts.add(literal_value)
#
#     except Exception as e:
#         print(f"[ERROR] AMR decode failed for {doc_id}: {e}")
#         useful_concepts = set()
#
#     if useful_concepts:
#         concepts_list = list(useful_concepts)
#         concept_docs.append(" ".join(concepts_list))
#         id_list.append(doc_id)
#         concepts_per_doc[doc_id] = concepts_list
#         print(f"[{doc_id}] Extracted: {concepts_list}")
#     else:
#         print(f"[!] Skipped {doc_id}, no keywords extracted")
#
# # Step 2: Filter out empty docs
# filtered_docs = []
# filtered_ids = []
# filtered_concepts = []
#
# for i, doc in enumerate(concept_docs):
#     if doc.strip():
#         filtered_docs.append(doc)
#         filtered_ids.append(id_list[i])
#         filtered_concepts.append(concepts_per_doc[id_list[i]])
#
# if not filtered_docs:
#     print("[!] No valid documents to process for LDA.")
#     exit()
#
# # Replace originals
# concept_docs = filtered_docs
# id_list = filtered_ids
# concepts_per_doc = dict(zip(filtered_ids, filtered_concepts))
#
# # Step 3: LDA modeling
# vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', lowercase=True)
# X = vectorizer.fit_transform(concept_docs)
# feature_names = vectorizer.get_feature_names_out()
#
# lda = LatentDirichletAllocation(n_components=10, random_state=42)
# lda.fit(X)
# topic_word_matrix = lda.components_
#
# # Step 4: Score and refine concepts
# keywords_output = []
#
# for idx, doc_id in enumerate(id_list):
#     doc_concepts = concepts_per_doc[doc_id]
#     topic_dist = lda.transform(X[idx])[0]
#     weighted_topic = topic_dist @ topic_word_matrix
#     word_scores = dict(zip(feature_names, weighted_topic))
#
#     # Score only available concepts
#     doc_scores = {c: word_scores[c] for c in doc_concepts if c in word_scores}
#     ranked = sorted(doc_scores.items(), key=lambda x: -x[1])
#
#     # Lemmatize and deduplicate
#     seen = set()
#     final_keywords = []
#     for word, _ in ranked:
#         lemma = lemmatizer.lemmatize(word)
#         if lemma not in seen:
#             final_keywords.append(lemma)
#             seen.add(lemma)
#
#     keywords_output.append({
#         "id": doc_id,
#         "keywords": final_keywords
#     })
#
# # Step 5: Save to file
# os.makedirs("corpus", exist_ok=True)
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(keywords_output, f, indent=2)
#
# print(f"[✓] Refined AMR topic keywords saved → {output_path}")


import json
import json
import penman
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# DOMAIN-SPECIFIC BLACKLIST
DOMAIN_BLACKLIST = {'fc', 'sub', 'and'}

def strip_suffix(concept):
    return re.sub(r"-\d{2}$", "", concept.lower())

def is_valid_keyword(word):
    word_lower = word.lower()
    if word_lower in DOMAIN_BLACKLIST:
        return False

    doc = nlp(word_lower)
    for token in doc:
        if (token.pos_ in {'CCONJ', 'SCONJ', 'ADJ', 'ADP', 'DET', 'PART', 'INTJ', 'PRON', 'AUX'}
            or token.is_stop
            or len(token.text) <= 2):
            return False
    return True

def extract_keywords_from_amr(body_amr):
    keywords = set()
    try:
        graph = penman.decode(body_amr)

        var_to_concept = {var: strip_suffix(concept) for var, _, concept in graph.instances()}

        child_map = {}
        for source, role, target in graph.triples:
            child_map.setdefault(source, []).append((role, target))

        # Explicit root concept checking
        root_var = graph.top
        root_concept = var_to_concept.get(root_var, "")
        if is_valid_keyword(root_concept):
            keywords.add(root_concept)

        # Named entities and literals explicitly checked
        for source, edges in child_map.items():
            for role, target in edges:
                if role == ':name':
                    for subrole, literal in child_map.get(target, []):
                        if subrole.startswith(":op") and literal.startswith('"'):
                            literal_val = literal.strip('"').lower()
                            if is_valid_keyword(literal_val):
                                keywords.add(literal_val)

        # ARG concepts explicitly checked
        for source, role, target in graph.triples:
            if role.startswith(':ARG') and target in var_to_concept:
                concept = var_to_concept[target]
                if is_valid_keyword(concept):
                    keywords.add(concept)

    except Exception as e:
        print(f"[Error parsing AMR]: {e}")

    return list(keywords)

# Final semantic cleaning & duplicates removal
def final_keyword_filtering(keywords):
    final_keywords = set()
    doc = nlp(" ".join(keywords))
    seen_lemmas = set()

    for token in doc:
        lemma = lemmatizer.lemmatize(token.text.lower())
        if lemma not in seen_lemmas:
            final_keywords.add(lemma)
            seen_lemmas.add(lemma)

    return list(final_keywords)

def process_amr_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords_output = []
    for entry in data:
        doc_id = entry.get("id")
        body_amr = entry.get("body_amr", "")

        # Initial extraction with robust domain-specific filtering
        raw_keywords = extract_keywords_from_amr(body_amr)

        # Final cleanup: deduplicate and standardize lemmas
        keywords = final_keyword_filtering(raw_keywords)

        if keywords:
            print(f"[{doc_id}] Final Keywords: {keywords}")
        else:
            print(f"[{doc_id}] No meaningful keywords extracted")

        keywords_output.append({
            "id": doc_id,
            "keywords": keywords
        })

    # Save the refined keyword set
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keywords_output, f, indent=2)

    print(f"[✔] Refined keywords saved to: {output_path}")

if __name__ == "__main__":
    input_path = "corpus/parsed_amrs.json"
    output_path = "corpus/amr_keywords.json"
    process_amr_file(input_path, output_path)
