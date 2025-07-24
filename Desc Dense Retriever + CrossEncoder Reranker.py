import pandas as pd
import re
import torch
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, CrossEncoder
from TurkishStemmer import TurkishStemmer

nltk.download("stopwords")

dense_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
rerank_model = CrossEncoder("cross-encoder/stsb-roberta-base")

stemmer = TurkishStemmer()

stop_words = set(stopwords.words("english"))
turkish_stopwords = {
    "ve", "ile", "için", "bir", "bu", "şu", "da", "de", "gibi", "olan", "ama", "veya",
    "çok", "en", "mi", "mı", "mu", "mü", "çünkü", "fakat", "ise", "olarak", "sadece", "hiç"
}
extra_english_stopwords = {
    "also", "just", "still", "though", "even", "yet", "however", "there", "where", "whose", "which", "such",
    "like", "about", "into", "through", "among", "each", "every", "whether", "been", "being", "because",
    "around", "without", "across", "while", "although", "before", "after", "another", "the", "an", "a"
}
all_stopwords = stop_words.union(turkish_stopwords).union(extra_english_stopwords)

verb_suffixes = ["mek", "mak", "yor", "acak", "ecek", "miş", "mış", "muş", "müş", "ing", "ed", "ize", "ise", "ly", "s", "es", "d", "en"]

def is_verb(word):
    return any(word.lower().endswith(suffix) for suffix in verb_suffixes)

def normalize_word(word):
    return stemmer.stem(word.lower())

def rerank_dense_keywords(title, desc, top_n=5, preselect=20):
    desc_clean = re.sub(r"[^\w\s]", " ", str(desc))
    words = desc_clean.split()
    candidate_words = [
        w for w in words if w.lower() not in all_stopwords and not is_verb(w) and len(w) > 2
    ]
    unique_words = list(dict.fromkeys(candidate_words))
    if not unique_words:
        return ""

    word_embeds = dense_model.encode(unique_words, convert_to_tensor=True)
    title_embed = dense_model.encode(title, convert_to_tensor=True)
    sims = torch.nn.functional.cosine_similarity(title_embed, word_embeds).cpu().tolist()

    top_dense_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:preselect]
    dense_candidates = [unique_words[i] for i in top_dense_indices]

    pairs = [(title, word) for word in dense_candidates]
    scores = rerank_model.predict(pairs)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    seen_roots = set()
    final_words = []
    for i in sorted_indices:
        word = dense_candidates[i]
        root = normalize_word(word)
        if root not in seen_roots:
            seen_roots.add(root)
            final_words.append(word)
        if len(final_words) == top_n:
            break

    return " | ".join(final_words)

df = pd.read_excel("desc1.xlsx")
results = []
total = len(df)

for idx, row in df.iterrows():
    title = row["Title"]
    desc = row["desc"]
    keywords = rerank_dense_keywords(title, desc)
    results.append(keywords)
    print(f"%{int((idx + 1) / total * 100)} tamamlandı... ({idx + 1}/{total})")

df["reranked_keywords"] = results
df.to_excel("desc1_rerankeddsds_keywords.xlsx", index=False)
print("✅ Bitti! Çıktı: desc1_reranked_kdsdseywords.xlsx")
