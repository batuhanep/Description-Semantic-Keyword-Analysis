import pandas as pd
import re
import random
import torch
import nltk
from rank_bm25 import BM25Okapi  # TF-IDF yerine BM25 için eklendi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

nltk.download("stopwords")

# MODELLER
model = SentenceTransformer('BAAI/bge-m3')
#model = SentenceTransformer('sentence-transformers/LaBSE')
#model = SentenceTransformer('intfloat/multilingual-e5-large')
#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
#model = SentenceTransformer("all-mpnet-base-v2")

# stopwords
stop_words = set(stopwords.words("english"))
turkish_stopwords = {
    "ve", "ile", "için", "ama", "fakat", "çünkü", "veya", "bir", "bu", "şu", "daha", "gibi", "olan", "cihazı", "değildir","gramdır","Paket","gönderilecektir",
    "en","üzere","ürünlerdeki","tarafından","göre","kargonun", "çok", "az", "hiç", "mı", "mi", "mu", "mü", "da", "de", "ki", "ya", "ise", "olarak", "sadece"
}
extra_english_stopwords = {
    "also", "just", "still", "though", "even", "yet", "however", "there", "where", "whose", "which", "such",
    "like", "about", "into", "through", "among", "each", "every", "whether", "been", "being", "because",
    "around", "without", "across", "while", "although", "before", "after"
}
all_stopwords = stop_words.union(turkish_stopwords).union(extra_english_stopwords)

# fiiller
turkish_verb_suffixes = ["mek", "mak", "yor", "acak", "ecek", "miş", "mış", "muş", "müş"]
english_verb_suffixes = ["ing", "ed", "ize", "ise", "s", "es", "d", "ly"]
verb_suffixes = turkish_verb_suffixes + english_verb_suffixes

def is_verb(word):
    return any(word.lower().endswith(suffix) for suffix in verb_suffixes)

# Basit normalize işlemi: lowercase + son ekleri at
def normalize_word(word):
    word = word.lower()
    for suffix in verb_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[: -len(suffix)]
            break
    return word

# hibrit anahtar kelime seçimi
def extract_hybrid_keywords(title, desc, top_n=5):
    desc_clean = re.sub(r"[^\w\s]", " ", desc)
    words = desc_clean.split()
    candidate_words = [
        w for w in words if w.lower() not in all_stopwords and not is_verb(w) and len(w) > 2
    ]
    unique_words = list(dict.fromkeys(candidate_words))
    if not unique_words:
        return ""

    # BM25 similarity (TF-IDF'in yerine)
    tokenized_corpus = [doc.split(" ") for doc in unique_words]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = title.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    # Embedding similarity
    word_embeddings = model.encode(unique_words, convert_to_tensor=True)
    title_embedding = model.encode(title, convert_to_tensor=True)
    embedding_similarities = torch.nn.functional.cosine_similarity(title_embedding, word_embeddings).cpu().tolist()

    # Hibrit skor
    hybrid_scores = [
        0.2 * bm25 + 0.8 * embed
        for bm25, embed in zip(bm25_scores, embedding_similarities)
    ]

    # En yüksek skora sahip kelimeleri sırala
    top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)

    # Normalize edilmiş köklere göre tekrarları filtrele
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

    random.shuffle(final_words)
    return " | ".join(final_words)

df = pd.read_excel("/Users/batuhanozdemir/Desktop/datasicence-ops-software/Datascience/desci.xlsx")  # kendi dosya adını gir
df['Title'] = df['Title'].fillna('')
df['desc'] = df['desc'].fillna('')
df["hybrid_keywords"] = df.apply(lambda row: extract_hybrid_keywords(row["Title"], row["desc"]), axis=1)

df.to_excel("descbm25.xlsx", index=False)
print("Bitti. Dosya: dsds.xlsx")
