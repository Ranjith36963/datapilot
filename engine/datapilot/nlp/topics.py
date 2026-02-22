"""
Topic modeling — find themes/topics in text data.

Uses sklearn LDA/NMF with TF-IDF vectorization.
Optionally uses gensim for coherence scoring.
"""


import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result

logger = setup_logging("datapilot.topics")


def extract_topics(
    file_path: str,
    text_column: str,
    n_topics: int = 5,
    method: str = "lda",
) -> dict:
    """
    Topic modeling.

    Methods: lda (Latent Dirichlet Allocation), nmf (Non-negative Matrix Factorization).
    """
    try:
        df = load_data(file_path)
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found"}

        texts = df[text_column].dropna().astype(str).tolist()
        if len(texts) < n_topics:
            return {"status": "error", "message": f"Need at least {n_topics} documents"}

        logger.info(f"Extracting {n_topics} topics via {method} from {len(texts)} docs")

        if method == "nmf":
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
            X = vectorizer.fit_transform(texts)
            model = NMF(n_components=n_topics, random_state=42, max_iter=500)
        else:  # lda
            vectorizer = CountVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
            X = vectorizer.fit_transform(texts)
            model = LatentDirichletAllocation(
                n_components=n_topics, random_state=42, max_iter=20, learning_method="batch",
            )

        model.fit(X)
        feature_names = vectorizer.get_feature_names_out()

        topics = []
        for idx, component in enumerate(model.components_):
            top_idx = component.argsort()[::-1][:10]
            top_words = [
                {"word": feature_names[i], "weight": round(float(component[i]), 4)}
                for i in top_idx
            ]
            label = " / ".join([w["word"] for w in top_words[:3]])
            topics.append({
                "topic_id": idx,
                "top_words": top_words,
                "suggested_label": label.title(),
            })

        # Document-topic assignments
        doc_topics = model.transform(X)
        dominant = doc_topics.argmax(axis=1).tolist()

        # Topic distribution
        for topic in topics:
            tid = topic["topic_id"]
            cnt = dominant.count(tid)
            topic["document_count"] = cnt
            topic["document_pct"] = round(cnt / len(dominant) * 100, 2)

        # Coherence score (simple approximation: average top-word co-occurrence)
        coherence = None
        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel
            tokenized = [t.lower().split() for t in texts]
            dictionary = Dictionary(tokenized)
            topic_words = [[feature_names[i] for i in comp.argsort()[::-1][:10]] for comp in model.components_]
            cm = CoherenceModel(topics=topic_words, texts=tokenized, dictionary=dictionary, coherence="c_v")
            coherence = round(float(cm.get_coherence()), 4)
        except (ImportError, Exception):
            pass

        result = {
            "status": "success",
            "n_topics": n_topics,
            "method": method,
            "topics": topics,
            "document_topics": dominant,
            "coherence_score": coherence,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def assign_topics(df: pd.DataFrame, text_column: str, model_path: str | None = None) -> pd.DataFrame:
    """Assign dominant topic to each document. Returns df with topic column."""
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
    lda.fit(X)
    doc_topics = lda.transform(X)
    df_out = df.copy()
    dominant = doc_topics.argmax(axis=1)
    # Map back including NaN rows
    topic_col = pd.Series(index=df.index, dtype=float)
    non_null_idx = df[text_column].dropna().index
    topic_col.loc[non_null_idx] = dominant
    df_out["topic"] = topic_col
    return df_out


def topics_and_upload(file_path: str, output_name: str = "topics.json", **kwargs) -> dict:
    """Convenience function: extract_topics + upload."""
    result = extract_topics(file_path, **kwargs)
    upload_result(result, output_name)
    return result
