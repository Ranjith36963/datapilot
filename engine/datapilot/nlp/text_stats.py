"""
Text statistics — word count, readability, keywords, bigrams.

Uses nltk for tokenization, textstat for readability scores,
sklearn TF-IDF for keyword extraction.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.text_stats")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "up", "out", "off", "over", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "what", "which",
    "who", "whom",
}


def _tokenize(text: str) -> List[str]:
    """Simple word tokenizer."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def analyze_text(text: str) -> dict:
    """Text statistics for a single text string."""
    try:
        words = _tokenize(text)
        sents = _sentences(text)

        word_count = len(words)
        char_count = len(text)
        sent_count = len(sents)
        avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 2)
        avg_sent_len = round(word_count / max(sent_count, 1), 2)

        # Readability
        readability: Dict[str, Any] = {}
        try:
            import textstat
            fre = textstat.flesch_reading_ease(text)
            fkg = textstat.flesch_kincaid_grade(text)
            if fre >= 80:
                interp = "Easy to read"
            elif fre >= 60:
                interp = "Standard"
            elif fre >= 40:
                interp = "Difficult"
            else:
                interp = "Very difficult / College level"
            readability = {
                "flesch_reading_ease": round(fre, 2),
                "flesch_kincaid_grade": round(fkg, 2),
                "interpretation": interp,
            }
        except ImportError:
            # Manual Flesch approximation
            syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in words)
            if word_count > 0 and sent_count > 0:
                fre = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllables / word_count)
                readability = {"flesch_reading_ease": round(fre, 2), "note": "Approximation"}

        # Top words (excluding stopwords)
        filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        top_words = [{"word": w, "count": c} for w, c in Counter(filtered).most_common(15)]

        # Top bigrams
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)]
        top_bigrams = [{"bigram": b, "count": c} for b, c in Counter(bigrams).most_common(10)]

        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sent_count,
            "avg_word_length": avg_word_len,
            "avg_sentence_length": avg_sent_len,
            "readability": readability,
            "top_words": top_words,
            "top_bigrams": top_bigrams,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_text_stats(file_path: str, text_column: str) -> dict:
    """Text stats aggregated over an entire column."""
    try:
        df = load_data(file_path)
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found"}

        texts = df[text_column].dropna().astype(str)
        logger.info(f"Batch text stats on {len(texts)} texts")

        word_counts = []
        char_counts = []
        all_words: List[str] = []

        for text in texts:
            words = _tokenize(text)
            word_counts.append(len(words))
            char_counts.append(len(text))
            all_words.extend([w for w in words if w not in _STOPWORDS and len(w) > 2])

        top_words = [{"word": w, "count": c} for w, c in Counter(all_words).most_common(20)]

        import numpy as np
        result = {
            "status": "success",
            "total_texts": len(texts),
            "avg_word_count": round(float(np.mean(word_counts)), 2) if word_counts else 0,
            "avg_char_count": round(float(np.mean(char_counts)), 2) if char_counts else 0,
            "min_word_count": int(min(word_counts)) if word_counts else 0,
            "max_word_count": int(max(word_counts)) if word_counts else 0,
            "top_words_overall": top_words,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def extract_keywords(text: str, n: int = 10) -> list:
    """Extract key terms via TF-IDF on sentences or simple frequency."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        sents = _sentences(text)
        if len(sents) < 2:
            words = _tokenize(text)
            filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
            return [{"keyword": w, "score": c} for w, c in Counter(filtered).most_common(n)]

        tfidf = TfidfVectorizer(stop_words="english", max_features=100)
        X = tfidf.fit_transform(sents)
        scores = X.sum(axis=0).A1
        feature_names = tfidf.get_feature_names_out()
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [{"keyword": w, "score": round(float(s), 4)} for w, s in ranked[:n]]
    except Exception:
        words = _tokenize(text)
        filtered = [w for w in words if w not in _STOPWORDS and len(w) > 2]
        return [{"keyword": w, "score": c} for w, c in Counter(filtered).most_common(n)]


def text_stats_and_upload(file_path: str, output_name: str = "text_stats.json", **kwargs) -> dict:
    """Convenience function: batch_text_stats + upload."""
    result = batch_text_stats(file_path, **kwargs)
    upload_result(result, output_name)
    return result
