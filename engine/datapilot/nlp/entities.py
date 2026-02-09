"""
Entity extractor — extract named entities (people, companies, locations).

Uses spaCy for NER. Falls back to regex patterns if spaCy unavailable.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.helpers import load_data, save_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.entities")


# ---------------------------------------------------------------------------
# spaCy NER
# ---------------------------------------------------------------------------

_NLP = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                _NLP = spacy.load("en_core_web_md") if spacy.util.is_package("en_core_web_md") else None
        except ImportError:
            _NLP = None
    return _NLP


def _spacy_ner(text: str) -> List[Dict[str, Any]]:
    """Extract entities using spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        return []
    doc = nlp(text[:100000])  # cap input size
    return [
        {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"[\+]?[\d\s\-().]{7,20}")
_PCT_RE = re.compile(r"\b\d+\.?\d*\s*%")


def _regex_ner(text: str) -> List[Dict[str, Any]]:
    """Fallback regex-based entity extraction."""
    entities = []
    for m in _DATE_RE.finditer(text):
        entities.append({"text": m.group(), "type": "DATE", "start": m.start(), "end": m.end()})
    for m in _MONEY_RE.finditer(text):
        entities.append({"text": m.group(), "type": "MONEY", "start": m.start(), "end": m.end()})
    for m in _EMAIL_RE.finditer(text):
        entities.append({"text": m.group(), "type": "EMAIL", "start": m.start(), "end": m.end()})
    for m in _PCT_RE.finditer(text):
        entities.append({"text": m.group(), "type": "PERCENT", "start": m.start(), "end": m.end()})
    return entities


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict:
    """Named Entity Recognition on a single text."""
    try:
        entities = _spacy_ner(text)
        if not entities:
            entities = _regex_ner(text)

        counts: Dict[str, int] = Counter(e["type"] for e in entities)

        return {
            "text": text[:100],
            "entities": entities[:100],
            "entity_counts": dict(counts),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def batch_extract_entities(file_path: str, text_column: str) -> dict:
    """NER for an entire column."""
    try:
        df = load_data(file_path)
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found"}

        texts = df[text_column].dropna().astype(str)
        logger.info(f"Batch NER on {len(texts)} texts")

        all_entities: List[Dict] = []
        for text in texts:
            ents = _spacy_ner(text)
            if not ents:
                ents = _regex_ner(text)
            all_entities.extend(ents)

        total = len(all_entities)

        # Summary by type
        type_counter: Dict[str, Counter] = {}
        for ent in all_entities:
            t = ent["type"]
            if t not in type_counter:
                type_counter[t] = Counter()
            type_counter[t][ent["text"]] += 1

        entity_summary = {}
        for t, counter in type_counter.items():
            entity_summary[t] = [{"name": n, "count": c} for n, c in counter.most_common(20)]

        most_people = entity_summary.get("PERSON", [])[:10]
        most_orgs = entity_summary.get("ORG", [])[:10]
        most_locs = entity_summary.get("GPE", [])[:10]

        # Save with entities column
        from pathlib import Path
        out_path = str(Path(file_path).parent / "entities_results.csv")
        df_out = df.copy()
        ent_col = []
        idx = 0
        for i, row in df.iterrows():
            if pd.notna(row[text_column]):
                ents = _spacy_ner(str(row[text_column])) or _regex_ner(str(row[text_column]))
                ent_col.append("; ".join(f"{e['text']}({e['type']})" for e in ents[:10]))
            else:
                ent_col.append("")
        df_out["entities"] = ent_col
        save_data(df_out, out_path)

        result = {
            "status": "success",
            "total_entities": total,
            "entity_summary": entity_summary,
            "most_mentioned_people": most_people,
            "most_mentioned_organizations": most_orgs,
            "most_mentioned_locations": most_locs,
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def entities_and_upload(file_path: str, output_name: str = "entities.json", **kwargs) -> dict:
    """Convenience function: batch_extract_entities + upload."""
    result = batch_extract_entities(file_path, **kwargs)
    upload_result(result, output_name)
    return result
