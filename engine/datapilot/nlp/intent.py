"""
Intent detector — classify user intent (complaint, inquiry, feedback, etc.).

Uses sklearn TF-IDF + classifier for labeled data, keyword heuristics for unlabeled.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ..utils.helpers import load_data, save_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.intent")


# ---------------------------------------------------------------------------
# Keyword-based heuristic fallback
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS = {
    "complaint": ["bad", "terrible", "worst", "horrible", "poor", "angry", "frustrated",
                   "unacceptable", "disappointed", "awful", "useless", "broken", "issue", "problem"],
    "inquiry": ["how", "what", "when", "where", "can you", "could you", "please tell",
                "information", "details", "question", "wondering", "curious"],
    "feedback": ["suggest", "recommendation", "improve", "better", "could be",
                 "would like", "wish", "opinion", "think", "feel"],
    "request": ["please", "need", "want", "require", "help me", "assist",
                "looking for", "send", "provide", "give"],
    "praise": ["great", "excellent", "amazing", "wonderful", "love", "fantastic",
               "awesome", "perfect", "best", "thank", "happy", "satisfied"],
}


def _heuristic_intent(text: str) -> Dict[str, Any]:
    """Keyword-based intent detection."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in text_lower)

    total = sum(scores.values()) or 1
    normalized = {k: round(v / total, 4) for k, v in scores.items()}
    best = max(scores, key=scores.get)  # type: ignore
    confidence = round(scores[best] / total, 4) if total else 0.0

    return {
        "text": text[:100],
        "intent": best if scores[best] > 0 else "unknown",
        "confidence": confidence,
        "all_scores": normalized,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_intent(text: str, intents: Optional[List[str]] = None) -> dict:
    """Classify a single text into an intent category."""
    try:
        return _heuristic_intent(text)
    except Exception as e:
        return {"status": "error", "message": str(e)}


def train_intent_classifier(file_path: str, text_column: str, intent_column: str) -> dict:
    """Train a custom intent classifier on labeled data."""
    try:
        df = load_data(file_path)
        for col in [text_column, intent_column]:
            if col not in df.columns:
                return {"status": "error", "message": f"Column '{col}' not found"}

        clean = df.dropna(subset=[text_column, intent_column])
        texts = clean[text_column].astype(str).tolist()
        labels = clean[intent_column].astype(str).tolist()

        logger.info(f"Training intent classifier: {len(texts)} samples, {len(set(labels))} intents")

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])

        cv_scores = cross_val_score(pipe, texts, labels, cv=min(5, len(set(labels))), scoring="accuracy")
        pipe.fit(texts, labels)

        from pathlib import Path
        model_path = str(Path(file_path).parent / "intent_model.pkl")
        joblib.dump(pipe, model_path)

        result = {
            "status": "success",
            "n_samples": len(texts),
            "n_intents": len(set(labels)),
            "intents": sorted(set(labels)),
            "cv_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "model_path": model_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_classify_intent(
    file_path: str,
    text_column: str,
    model_path: Optional[str] = None,
) -> dict:
    """Classify intents for an entire column."""
    try:
        df = load_data(file_path)
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found"}

        texts = df[text_column].dropna().astype(str)
        logger.info(f"Batch intent classification: {len(texts)} texts")

        if model_path:
            pipe = joblib.load(model_path)
            preds = pipe.predict(texts.tolist())
            probs = pipe.predict_proba(texts.tolist()) if hasattr(pipe, "predict_proba") else None
            intent_list = preds.tolist()
            low_conf = 0
            if probs is not None:
                max_probs = probs.max(axis=1)
                low_conf = int((max_probs < 0.5).sum())
        else:
            intent_list = []
            low_conf = 0
            for text in texts:
                r = _heuristic_intent(text)
                intent_list.append(r["intent"])
                if r["confidence"] < 0.3:
                    low_conf += 1

        dist = pd.Series(intent_list).value_counts().to_dict()

        # Save
        from pathlib import Path
        df_out = df.copy()
        intent_col = []
        idx = 0
        for i, row in df.iterrows():
            if pd.notna(row[text_column]):
                intent_col.append(intent_list[idx])
                idx += 1
            else:
                intent_col.append("")
        df_out["intent"] = intent_col
        out_path = str(Path(file_path).parent / "intent_results.csv")
        save_data(df_out, out_path)

        result = {
            "status": "success",
            "intent_distribution": dist,
            "low_confidence_count": low_conf,
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def intent_and_upload(file_path: str, output_name: str = "intents.json", **kwargs) -> dict:
    """Convenience function: batch_classify_intent + upload."""
    result = batch_classify_intent(file_path, **kwargs)
    upload_result(result, output_name)
    return result
