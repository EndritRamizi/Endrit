from __future__ import annotations

import re
import os
import string
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF

import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


# -----------------------------
# 0) NLTK setup (safe)
# -----------------------------
def ensure_nltk() -> None:
    """Download only what's needed if missing."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


# -----------------------------
# 1) Load data
# -----------------------------
def load_reviews(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Review" not in df.columns:
        raise ValueError(f"Expected column 'Review' but got: {df.columns.tolist()}")
    df = df.dropna(subset=["Review"]).copy()
    df["Review"] = df["Review"].astype(str)
    return df


# -----------------------------
# 2) Preprocess
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NUM_RE = re.compile(r"\b\d+\b")

def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NUM_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text, lemmatizer, stopwords):
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in stopwords]
    return tokens


def preprocess_corpus(texts: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Returns:
      cleaned_docs: list of cleaned+lemmatized docs (strings)
      tokenized_docs: list of token lists (for Word2Vec)
    """
    lemmatizer = None

    stopwords = set(ENGLISH_STOP_WORDS)

    tokenized_docs: List[List[str]] = []
    cleaned_docs: List[str] = []

    for t in texts:
        t2 = clean_text(t)
        toks = tokenize_and_lemmatize(t2, lemmatizer, stopwords)
        tokenized_docs.append(toks)
        cleaned_docs.append(" ".join(toks))

    return cleaned_docs, tokenized_docs


# -----------------------------
# 3) Vectorization (2 methods)
#   A) TF-IDF
#   B) Word2Vec average embeddings
# -----------------------------
def build_tfidf(docs: List[str], max_features: int = 5000):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    X = tfidf.fit_transform(docs)
    return tfidf, X

def build_word2vec_doc_vectors(tokenized_docs: List[List[str]], vector_size: int = 100, window: int = 5) -> np.ndarray:
    """
    Train Word2Vec on your corpus and return one vector per document
    by averaging word vectors.
    """
    # Train Word2Vec
    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=3,
        workers=1,
        sg=1,  # skip-gram
        epochs=10,
    )

    doc_vecs = np.zeros((len(tokenized_docs), vector_size), dtype=np.float32)

    for i, toks in enumerate(tokenized_docs):
        vecs = [w2v.wv[w] for w in toks if w in w2v.wv]
        if vecs:
            doc_vecs[i] = np.mean(vecs, axis=0)
        else:
            doc_vecs[i] = np.zeros(vector_size, dtype=np.float32)

    return doc_vecs


# -----------------------------
# 4) Topic Modeling (2 methods)
#   A) LDA on Count vectors
#   B) NMF on TF-IDF
# -----------------------------
def fit_lda(docs: List[str], n_topics: int = 5, max_features: int = 5000):
    count_vec = CountVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
    )
    X = count_vec.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=42,
        max_iter=20,
    )
    lda.fit(X)
    return count_vec, X, lda

def fit_nmf(tfidf_matrix, n_topics: int = 5):
    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400,
    )
    W = nmf.fit_transform(tfidf_matrix)
    return nmf, W


def top_words_per_topic(model, feature_names: List[str], n_top_words: int = 10) -> Dict[int, List[str]]:
    topic_words: Dict[int, List[str]] = {}
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[::-1][:n_top_words]
        topic_words[topic_idx] = [feature_names[i] for i in top_idx]
    return topic_words


# -----------------------------
# 5) Run pipeline
# -----------------------------
def main():
    # Update this path if needed
    csv_path = "tripadvisor_hotel_reviews.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Couldn't find {csv_path}. Put the CSV in the same folder as this script "
            f"or change csv_path."
        )

    df = load_reviews(csv_path)
    raw_texts = df["Review"].tolist()

    # Preprocess
    docs, tokenized_docs = preprocess_corpus(raw_texts)

    # Vectorize 1: TF-IDF
    tfidf_vec, X_tfidf = build_tfidf(docs)

    # Vectorize 2: Word2Vec doc vectors (not used by LDA/NMF directly, but fulfills requirement)
    doc_embeddings = build_word2vec_doc_vectors(tokenized_docs)
    print(f"Word2Vec doc-embeddings shape: {doc_embeddings.shape}")

    # Topic model 1: LDA
    count_vec, X_count, lda = fit_lda(docs, n_topics=5)
    lda_topics = top_words_per_topic(lda, count_vec.get_feature_names_out(), n_top_words=10)

        # ---------- LDA STATISTICS ----------
    lda_doc_topics = lda.transform(X_count)

    lda_strength = lda_doc_topics.mean(axis=0)

    lda_stats = pd.DataFrame({
        "topic_id": range(len(lda_strength)),
        "mean_probability": lda_strength
    }).sort_values(by="mean_probability", ascending=False)

    lda_dominant = lda_doc_topics.argmax(axis=1)

    lda_percent = (
        pd.Series(lda_dominant)
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    lda_percent.columns = ["topic_id", "percentage_of_reviews"]

    lda_stats.to_csv("lda_topic_stats.csv", index=False)
    lda_percent.to_csv("lda_dominance.csv", index=False)

    print("\nSaved: lda_topic_stats.csv, lda_dominance.csv")

    
    print("\n=== LDA Topics (top words) ===")
    for k, words in lda_topics.items():
        print(f"Topic {k}: {', '.join(words)}")

    # Topic model 2: NMF
    nmf, W = fit_nmf(X_tfidf, n_topics=5)
    nmf_topics = top_words_per_topic(nmf, tfidf_vec.get_feature_names_out(), n_top_words=10)

        # ---------- NMF STATISTICS ----------
    nmf_strength = W.mean(axis=0)

    nmf_stats = pd.DataFrame({
        "topic_id": range(len(nmf_strength)),
        "mean_weight": nmf_strength
    }).sort_values(by="mean_weight", ascending=False)

    nmf_dominant = W.argmax(axis=1)

    nmf_percent = (
        pd.Series(nmf_dominant)
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    nmf_percent.columns = ["topic_id", "percentage_of_reviews"]

    nmf_stats.to_csv("nmf_topic_stats.csv", index=False)
    nmf_percent.to_csv("nmf_dominance.csv", index=False)

    print("\nSaved: nmf_topic_stats.csv, nmf_dominance.csv")


    print("\n=== NMF Topics (top words) ===")
    for k, words in nmf_topics.items():
        print(f"Topic {k}: {', '.join(words)}")

    # Optional: save topics for your report
    out = []
    for k, words in lda_topics.items():
        out.append({"model": "LDA", "topic": k, "top_words": ", ".join(words)})
    for k, words in nmf_topics.items():
        out.append({"model": "NMF", "topic": k, "top_words": ", ".join(words)})
    pd.DataFrame(out).to_csv("topics_output.csv", index=False)
    print("\nSaved: topics_output.csv")


if __name__ == "__main__":
    main()
