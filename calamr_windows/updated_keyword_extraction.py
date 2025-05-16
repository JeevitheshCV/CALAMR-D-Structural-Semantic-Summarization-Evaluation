import json
import penman
import re
import nltk
import spacy
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load NLP tools
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Domain-specific stoplist
DOMAIN_BLACKLIST = {'fc', 'sub', 'and'}

# Roles we consider semantically meaningful
IMPORTANT_ROLES = {
    ':ARG0', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5',
    ':location', ':time', ':manner', ':cause', ':purpose', ':destination'
}

def strip_suffix(concept):
    return re.sub(r"-\d{2}$", "", concept.lower())

def is_valid_keyword(word):
    word_lower = word.lower()
    if word_lower in DOMAIN_BLACKLIST:
        return False

    doc = nlp(word_lower)
    for token in doc:
        if (token.pos_ in {'CCONJ', 'SCONJ', 'ADJ', 'ADP', 'DET', 'PART', 'INTJ', 'PRON', 'AUX'}
            or token.is_stop or len(token.text) <= 2):
            return False
    return True

def extract_keywords_from_amr(body_amr):
    keywords = set()
    try:
        graph = penman.decode(body_amr)
        var_to_concept = {var: strip_suffix(concept) for var, _, concept in graph.instances()}
        child_map = {}
        target_counter = Counter()

        for source, role, target in graph.triples:
            child_map.setdefault(source, []).append((role, target))
            if isinstance(target, str):
                target_counter[target] += 1

        # Add root concept
        root_var = graph.top
        root_concept = var_to_concept.get(root_var, "")
        if is_valid_keyword(root_concept):
            keywords.add(root_concept)

        # Collect concepts from important roles
        for source, role, target in graph.triples:
            if role in IMPORTANT_ROLES and target in var_to_concept:
                concept = var_to_concept[target]
                if is_valid_keyword(concept):
                    keywords.add(concept)

        # Collect literals in all roles
        for source, edges in child_map.items():
            for role, target in edges:
                if isinstance(target, str) and target.startswith('"'):
                    literal_val = target.strip('"').lower()
                    if is_valid_keyword(literal_val):
                        keywords.add(literal_val)

        # Reentrant concepts: add if used multiple times
        for var, count in target_counter.items():
            if count > 1 and var in var_to_concept:
                concept = var_to_concept[var]
                if is_valid_keyword(concept):
                    keywords.add(concept)

    except Exception as e:
        print(f"[Error parsing AMR]: {e}")

    return list(keywords)

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

        raw_keywords = extract_keywords_from_amr(body_amr)
        keywords = final_keyword_filtering(raw_keywords)

        if keywords:
            print(f"[{doc_id}] Final Keywords: {keywords}")
        else:
            print(f"[{doc_id}] No meaningful keywords extracted")

        keywords_output.append({
            "id": doc_id,
            "keywords": keywords
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keywords_output, f, indent=2)

    print(f"[✔] Refined keywords saved to: {output_path}")

if __name__ == "__main__":
    input_path = "corpus/parsed_amrs.json"
    output_path = "corpus/amr_keywords.json"
    process_amr_file(input_path, output_path)
