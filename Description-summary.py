import pandas as pd
import re
import torch
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords


nltk.download("stopwords")

# model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# stopwords
stop_words = set(stopwords.words("english"))
turkish_stopwords = {
    "ve", "ile", "için", "ama", "fakat", "çünkü", "veya", "bir", "bu", "şu", "daha", "gibi", "olan",
    "en", "çok", "az", "hiç", "mı", "mi", "mu", "mü", "da", "de", "ki", "ya", "ise", "olarak", "sadece"
}
extra_english_stopwords = {
    "also", "just", "still", "though", "even", "yet", "however", "there", "where", "whose", "which", "such",
    "like", "about", "into", "through", "among", "each", "every", "whether", "been", "being", "because",
    "around", "without", "across", "while", "although", "before", "after"
}
all_stopwords = stop_words.union(turkish_stopwords).union(extra_english_stopwords)

# ing,yüklem
turkish_verb_suffixes = ["mek", "mak", "yor", "acak", "ecek", "miş", "mış", "muş", "müş"]
english_verb_suffixes = ["ing", "ed", "ize", "ise", "s", "es", "d", "ly"]
verb_suffixes = turkish_verb_suffixes + english_verb_suffixes

def is_verb(word):
    return any(word.lower().endswith(suffix) for suffix in verb_suffixes)

# normalize
def normalize_word(word):
    word = word.lower()
    for suffix in verb_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[: -len(suffix)]
            break
    return word

# hybrid keyword 
def extract_hybrid_keywords(title, desc, top_n=5):
    desc_clean = re.sub(r"[^\w\s]", " ", desc)
    words = desc_clean.split()
    candidate_words = [
        w for w in words if w.lower() not in all_stopwords and not is_verb(w) and len(w) > 2
    ]
    unique_words = list(dict.fromkeys(candidate_words))
    if not unique_words:
        return ""

    # TF-IDF 
    tfidf_vectorizer = TfidfVectorizer().fit([title] + unique_words)
    tfidf_vectors = tfidf_vectorizer.transform([title] + unique_words)
    tfidf_similarities = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:]).flatten()

    # Embedding similarity
    word_embeddings = model.encode(unique_words, convert_to_tensor=True)
    title_embedding = model.encode(title, convert_to_tensor=True)
    embedding_similarities = torch.nn.functional.cosine_similarity(title_embedding, word_embeddings).cpu().tolist()

    # hybrid score
    hybrid_scores = [
        0.5 * tfidf + 0.5 * embed
        for tfidf, embed in zip(tfidf_similarities, embedding_similarities)
    ]

    # best words 
    top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)

    # filtering normalized words
    seen_roots = set()
    final_words = []
    for i in top_indices:
        word = unique_words[i]
        root = normalize_word(word)
        if root not in seen_roots:
            seen_roots.add(root)
            final_words.append(word)
        if len(final_words) == top_n:
            break

    return " | ".join(final_words)


df = pd.read_excel("dsdsd.xlsx") 
df["hybrid_keywords"] = df.apply(lambda row: extract_hybrid_keywords(row["Title"], row["desc"]), axis=1)


df.to_excel("dsds.xlsx", index=False)
print("Bitti. Dosya: dsds.xlsx")
