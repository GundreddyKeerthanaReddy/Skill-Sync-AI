"""TF-IDF feature extraction helpers."""
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b[\w+#.-]+\b", sublinear_tf=True)
