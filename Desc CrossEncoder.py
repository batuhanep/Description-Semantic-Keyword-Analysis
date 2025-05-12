import pandas as pd
import re
import torch
from sentence_transformers import CrossEncoder
from nltk.corpus import stopwords
import nltk

# NLTK veri indir
nltk.download("stopwords")

# MODEL yükle
model = CrossEncoder("cross-encoder/stsb-roberta-base")

# STOPWORDS + EKLER
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

# Fiil ekleri
turkish_verb_suffixes = ["mek", "mak", "yor", "acak", "ecek", "miş", "mış", "muş", "müş"]
english_verb_suffixes = ["ing", "ed", "ize", "ise", "ly", "s", "es", "d", "en"]
verb_suffixes = turkish_verb_suffixes + english_verb_suffixes

def is_verb(word):
    return any(word.lower().endswith(suffix) for suffix in verb_suffixes)

def normalize_word(word):
    word = word.lower()
    for suffix in verb_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            word = word[: -len(suffix)]
            break
    return word

# CrossEncoder ile anahtar kelime çıkarımı
def sbert_crossencoder_keywords(title, desc, top_n=5):
    desc_clean = re.sub(r"[^\w\s]", " ", str(desc))
    words = desc_clean.split()
    candidate_words = [
        w for w in words if w.lower() not in all_stopwords and not is_verb(w) and len(w) > 2
    ]
    unique_words = list(dict.fromkeys(candidate_words))

    if not unique_words:
        return ""

    pairs = [(title, word) for word in unique_words]
    scores = model.predict(pairs)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

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

# Excel dosyasını oku
df = pd.read_excel("desc1.xlsx")  # Girdi dosyan buraya

# Anahtar kelimeleri sırayla üret ve ilerleme göster
results = []
total = len(df)

for idx, row in df.iterrows():
    title = row["Title"]
    desc = row["desc"]
    keywords = sbert_crossencoder_keywords(title, desc)
    results.append(keywords)

    # Yüzde hesapla ve yazdır
    percent = int((idx + 1) / total * 100)
    print(f"%{percent} tamamlandı... ({idx + 1}/{total})")

# Sonuçları ekle ve kaydet
df["crossencoder_keywords"] = results
df.to_excel("sdsd.xlsx", index=False)
print("✅ Bitti! Çıktı: dsdsd.xlsx")
