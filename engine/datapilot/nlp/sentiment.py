"""
Sentiment analysis — detect positive/negative/neutral in text.

Uses VADER (best for social media/reviews) and TextBlob (general purpose).
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.helpers import load_data, save_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.sentiment")


def analyze_sentiment(text: str, method: str = "vader") -> dict:
    """Analyze sentiment of a single text string."""
    try:
        if method == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            return {
                "text": text[:100],
                "method": "vader",
                "sentiment": label,
                "scores": {
                    "positive": round(scores["pos"], 4),
                    "negative": round(scores["neg"], 4),
                    "neutral": round(scores["neu"], 4),
                    "compound": round(compound, 4),
                },
                "confidence": round(abs(compound), 4),
            }

        elif method == "textblob":
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return {
                "text": text[:100],
                "method": "textblob",
                "sentiment": label,
                "scores": {
                    "positive": round(max(polarity, 0), 4),
                    "negative": round(abs(min(polarity, 0)), 4),
                    "neutral": round(1 - abs(polarity), 4),
                    "compound": round(polarity, 4),
                },
                "confidence": round(abs(polarity), 4),
            }
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

    except ImportError as e:
        return {"status": "error", "message": f"Library not installed: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_sentiment(file_path: str, text_column: str, method: str = "vader") -> dict:
    """Sentiment analysis for an entire column."""
    try:
        df = load_data(file_path)
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found"}

        logger.info(f"Batch sentiment ({method}) on {len(df)} rows")

        results = []
        for text in df[text_column].dropna().astype(str):
            r = analyze_sentiment(text, method)
            results.append(r)

        # Distribution
        labels = [r.get("sentiment", "unknown") for r in results]
        total = len(labels)
        dist = {}
        for lab in ["positive", "negative", "neutral"]:
            cnt = labels.count(lab)
            dist[lab] = {"count": cnt, "pct": round(cnt / total * 100, 2) if total else 0}

        compounds = [r.get("scores", {}).get("compound", 0) for r in results]
        avg_compound = round(sum(compounds) / len(compounds), 4) if compounds else 0

        # Most extreme
        sorted_by_compound = sorted(results, key=lambda x: x.get("scores", {}).get("compound", 0))
        most_negative = sorted_by_compound[:3]
        most_positive = sorted_by_compound[-3:][::-1]

        # Save with sentiment columns
        from pathlib import Path
        df_out = df.copy()
        sentiments = []
        scores_list = []
        idx = 0
        for i, row in df.iterrows():
            if pd.notna(row[text_column]):
                sentiments.append(results[idx].get("sentiment", ""))
                scores_list.append(results[idx].get("scores", {}).get("compound", 0))
                idx += 1
            else:
                sentiments.append("")
                scores_list.append(0)
        df_out["sentiment"] = sentiments
        df_out["sentiment_score"] = scores_list
        out_path = str(Path(file_path).parent / "sentiment_results.csv")
        save_data(df_out, out_path)

        result = {
            "status": "success",
            "total_rows": total,
            "distribution": dist,
            "average_compound": avg_compound,
            "most_positive": [{"text": r.get("text", ""), "score": r.get("scores", {}).get("compound", 0)} for r in most_positive],
            "most_negative": [{"text": r.get("text", ""), "score": r.get("scores", {}).get("compound", 0)} for r in most_negative],
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def sentiment_and_upload(file_path: str, output_name: str = "sentiment.json", **kwargs) -> dict:
    """Convenience function: batch_sentiment + upload."""
    result = batch_sentiment(file_path, **kwargs)
    upload_result(result, output_name)
    return result
