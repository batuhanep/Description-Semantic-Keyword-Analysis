# Hybrid Keyword Extractor

This script automates the process of extracting relevant keywords from product descriptions. It analyzes the description text and selects the most important words based on their statistical and semantic relevance to the product title, using a hybrid TF-IDF and SBERT model.

## Features

- **Hybrid Scoring:** Combines TF-IDF (keyword frequency) and SBERT (semantic meaning) to identify the most relevant keywords.
- **Smart Filtering:** Excludes common stopwords in both English and Turkish, and filters out words that appear to be verbs based on their suffixes.
- **Root Word Deduplication:** Ensures keyword diversity by preventing variations of the same word (e.g., "color" and "colors") from being selected.
- **Multi-language Ready:** Built with support for both English and Turkish, and easily extendable.

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Sentence-Transformers
- PyTorch
