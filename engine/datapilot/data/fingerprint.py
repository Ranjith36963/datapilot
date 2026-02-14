"""
Dataset fingerprinting — 3-layer domain detection with explainability.

Detects the business domain of a dataset (finance, healthcare, retail, etc.)
using a multi-layer approach: keyword matching → column pattern analysis → LLM confirmation.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from ..utils.helpers import setup_logging
from ..utils.serializer import safe_json_serialize

logger = setup_logging("datapilot.fingerprint")


# ---------------------------------------------------------------------------
# D3: Dataset Understanding (LLM-generated)
# ---------------------------------------------------------------------------

@dataclass
class DatasetUnderstanding:
    """LLM-generated understanding of a dataset."""
    domain: str                    # "telecom customer churn"
    domain_short: str              # "Telecom"
    target_column: str | None      # "churn"
    target_type: str | None        # "classification"
    key_observations: list[str]    # ["The churn rate is ~14.5%", ...]
    suggested_questions: list[str] # ["What drives churn?", ...]
    data_quality_notes: list[str]  # ["High correlation between X and Y"]
    confidence: float              # 0.9 if LLM responded, 0.0 if failed
    provider_used: str             # "gemini" or "groq"


def build_data_snapshot(df, filename: str, profile: dict) -> str:
    """Build a concise text snapshot of the dataset for LLM consumption. Target: <2000 tokens."""
    raise NotImplementedError("D3: build_data_snapshot not yet implemented")


def understand_dataset(df, filename: str, profile: dict, llm_provider) -> DatasetUnderstanding | None:
    """LLM reads the data snapshot and returns structured understanding."""
    raise NotImplementedError("D3: understand_dataset not yet implemented")


# ---------------------------------------------------------------------------
# Domain definitions and patterns
# ---------------------------------------------------------------------------

# Domain-specific keyword patterns for layer 1 & 2 detection
DOMAIN_PATTERNS = {
    "finance": {
        "keywords": {
            # Column name patterns
            "revenue", "profit", "cost", "price", "sales", "income", "expense",
            "transaction", "payment", "account", "balance", "interest", "loan",
            "credit", "debit", "cash", "budget", "roi", "margin", "earnings",
            "dividend", "stock", "equity", "asset", "liability", "tax", "fee",
            # Filename patterns
            "financial", "accounting", "treasury", "invoice", "billing",
        },
        "value_patterns": [
            r"^\$?\d+\.?\d*$",  # Currency values
            r"^\d{4}-\d{2}-\d{2}$",  # Dates (common in transactions)
        ],
        "common_targets": ["revenue", "profit", "sales", "price", "default", "churn"],
    },
    "healthcare": {
        "keywords": {
            "patient", "diagnosis", "treatment", "medication", "doctor", "nurse",
            "hospital", "clinic", "visit", "admission", "discharge", "symptom",
            "disease", "condition", "procedure", "prescription", "dosage", "vital",
            "blood_pressure", "heart_rate", "temperature", "weight", "height",
            "age", "gender", "medical", "health", "clinical", "icd", "cpt",
        },
        "value_patterns": [
            r"^\d{3}-\d{2}-\d{4}$",  # Patient IDs (SSN-like)
            r"^[A-Z]\d{2}",  # ICD codes
        ],
        "common_targets": ["diagnosis", "outcome", "readmission", "mortality", "severity"],
    },
    "retail": {
        "keywords": {
            "product", "sku", "inventory", "stock", "store", "shelf", "warehouse",
            "supplier", "vendor", "category", "brand", "quantity", "unit", "reorder",
            "discount", "promotion", "clearance", "markup", "wholesale", "retail",
            "backorder", "turnover", "shrinkage", "planogram", "merchandising",
        },
        "value_patterns": [
            r"^[A-Z0-9]{6,12}$",  # SKU patterns
        ],
        "common_targets": ["sales", "quantity_sold", "stock_level", "reorder", "category"],
    },
    "ecommerce": {
        "keywords": {
            "order", "cart", "checkout", "customer", "user", "session", "click",
            "view", "impression", "conversion", "bounce", "add_to_cart", "purchase",
            "shipping", "delivery", "fulfillment", "return", "refund", "review",
            "rating", "wishlist", "recommendation", "search", "keyword", "affiliate",
            "utm", "campaign", "channel", "funnel", "abandonment",
        },
        "value_patterns": [
            r"^[A-Z0-9-]{10,20}$",  # Order IDs
            r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # IP addresses
        ],
        "common_targets": ["purchase", "conversion", "churn", "lifetime_value", "order_value"],
    },
    "hr": {
        "keywords": {
            "employee", "staff", "hire", "termination", "salary", "wage", "compensation",
            "bonus", "benefits", "department", "title", "role", "position", "manager",
            "supervisor", "performance", "rating", "review", "appraisal", "training",
            "certification", "attendance", "leave", "vacation", "sick_day", "overtime",
            "turnover", "retention", "promotion", "demotion", "headcount", "fte",
        },
        "value_patterns": [],
        "common_targets": ["salary", "performance_rating", "turnover", "promotion", "tenure"],
    },
    "marketing": {
        "keywords": {
            "campaign", "lead", "prospect", "conversion", "ctr", "cpc", "cpm", "roas",
            "impression", "reach", "engagement", "like", "share", "comment", "follower",
            "subscriber", "email", "open_rate", "click_rate", "unsubscribe", "segment",
            "audience", "targeting", "demographic", "persona", "channel", "attribution",
            "touchpoint", "journey", "funnel", "awareness", "consideration", "intent",
        },
        "value_patterns": [
            r"^\d+%$",  # Percentage metrics
        ],
        "common_targets": ["conversion", "lead_score", "churn", "engagement", "lifetime_value"],
    },
}


# Note: FingerprintResult dataclass removed — function now returns flat dict
# to match backend Pydantic models. Return format:
# {
#     "status": "success",
#     "domain": str,
#     "confidence": float,
#     "explainability": {
#         "reasons": List[str],
#         "column_matches": List[str],
#         "layer": str,
#         "llm_reasoning": Optional[str]
#     },
#     "suggested_target": Optional[str],
#     "layer_scores": Dict[str, float]
# }


# ---------------------------------------------------------------------------
# Layer 1: Keyword matching
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> Set[str]:
    """Extract normalized keywords from text (filename or column names)."""
    # Lowercase, split on non-alphanumeric, filter short tokens
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return {t for t in tokens if len(t) > 2}


def _layer1_keyword_match(
    filename: str,
    column_names: List[str],
) -> Tuple[Optional[str], float, List[str]]:
    """Layer 1: Match keywords in filename and column names.

    Returns (domain, confidence, matched_patterns).
    """
    # Extract keywords from filename (remove extension if present)
    filename_stem = Path(filename).stem if '.' in filename else filename
    file_keywords = _extract_keywords(filename_stem)

    # Extract keywords from column names
    col_keywords = set()
    for col in column_names:
        col_keywords.update(_extract_keywords(col))

    all_keywords = file_keywords | col_keywords

    # Count matches per domain
    domain_scores: Dict[str, Tuple[int, List[str]]] = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        matches = all_keywords & patterns["keywords"]
        if matches:
            domain_scores[domain] = (len(matches), list(matches))

    if not domain_scores:
        return None, 0.0, []

    # Pick domain with most matches
    best_domain = max(domain_scores, key=lambda d: domain_scores[d][0])
    match_count, matched = domain_scores[best_domain]

    # Calculate confidence based on match count and uniqueness
    total_keywords = len(all_keywords)
    confidence = min(match_count / max(total_keywords, 1), 1.0)

    # Boost confidence if matches are exclusive to this domain
    other_domains = [d for d in domain_scores if d != best_domain]
    if not other_domains or match_count > sum(domain_scores[d][0] for d in other_domains):
        confidence = min(confidence * 1.2, 1.0)

    return best_domain, confidence, matched


# ---------------------------------------------------------------------------
# Layer 2: Column pattern analysis
# ---------------------------------------------------------------------------

def _match_value_pattern(series: pd.Series, pattern: str) -> float:
    """Return fraction of non-null values matching the regex pattern."""
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return 0.0

    regex = re.compile(pattern)
    matches = clean.apply(lambda v: bool(regex.match(v))).sum()
    return matches / len(clean)


def _layer2_column_analysis(
    df: pd.DataFrame,
    column_names: List[str],
) -> Tuple[Optional[str], float, List[str]]:
    """Layer 2: Analyze column value patterns and distributions.

    Returns (domain, confidence, matched_patterns).
    """
    domain_scores: Dict[str, Tuple[float, List[str]]] = {}

    for domain, patterns in DOMAIN_PATTERNS.items():
        score = 0.0
        matched = []

        # Check value patterns
        for col in column_names[:20]:  # Sample first 20 columns for performance
            if col not in df.columns:
                continue

            series = df[col]
            for pattern in patterns["value_patterns"]:
                match_rate = _match_value_pattern(series, pattern)
                if match_rate > 0.3:  # At least 30% match
                    score += match_rate
                    matched.append(f"{col}~{pattern[:20]}")

        # Check for common target columns
        for target in patterns["common_targets"]:
            # Match with fuzzy logic (contains or similar)
            for col in column_names:
                col_clean = col.lower().replace("_", "").replace(" ", "")
                target_clean = target.lower().replace("_", "")
                if target_clean in col_clean or col_clean in target_clean:
                    score += 0.5
                    matched.append(f"target:{col}")
                    break

        if score > 0:
            domain_scores[domain] = (score, matched)

    if not domain_scores:
        return None, 0.0, []

    # Pick domain with highest score
    best_domain = max(domain_scores, key=lambda d: domain_scores[d][0])
    score, matched = domain_scores[best_domain]

    # Normalize confidence to 0-1 range
    confidence = min(score / 3.0, 1.0)  # Max score ~3 (multiple patterns + targets)

    return best_domain, confidence, matched


# ---------------------------------------------------------------------------
# Layer 3: LLM-based detection
# ---------------------------------------------------------------------------

FINGERPRINT_SYSTEM_PROMPT = (
    "You are a dataset domain classifier. "
    "Given column names, sample values, and basic statistics, "
    "determine which business domain this dataset belongs to. "
    "Consider the semantic meaning of columns and typical patterns in each domain. "
    "Respond ONLY with valid JSON: "
    '{"domain": "<finance|healthcare|retail|ecommerce|hr|marketing|general>", '
    '"confidence": <0.0-1.0>, "reasoning": "<one_sentence>", '
    '"suggested_target": "<column_name_or_null>"}'
)

FINGERPRINT_PROMPT_TEMPLATE = """\
Dataset overview:
- Rows: {rows}
- Columns: {columns}

Column summary:
{column_summary}

Sample values:
{sample_values}

Classify this dataset into one of these domains:
- finance: Financial transactions, accounting, revenue, pricing
- healthcare: Medical records, patient data, diagnoses, treatments
- retail: Product inventory, stores, stock levels, merchandising
- ecommerce: Online orders, customers, carts, conversions, web analytics
- hr: Employee data, salaries, performance, hiring, turnover
- marketing: Campaigns, leads, engagement, conversion funnels, attribution
- general: None of the above / mixed domain

Respond with JSON only."""


def _build_column_summary(df: pd.DataFrame) -> str:
    """Build a concise summary of columns for LLM."""
    lines = []
    for col in df.columns[:30]:  # Limit to first 30 columns
        dtype = str(df[col].dtype)
        unique = df[col].nunique()
        missing = df[col].isna().sum()
        lines.append(f"- {col} ({dtype}, {unique} unique, {missing} nulls)")

    if len(df.columns) > 30:
        lines.append(f"... and {len(df.columns) - 30} more columns")

    return "\n".join(lines)


def _build_sample_values(df: pd.DataFrame) -> str:
    """Build a sample of column values for LLM."""
    lines = []
    for col in df.columns[:15]:  # Limit to first 15 columns
        samples = df[col].dropna().head(3).tolist()
        samples_str = ", ".join(str(s)[:30] for s in samples)
        lines.append(f"- {col}: {samples_str}")

    if len(df.columns) > 15:
        lines.append(f"... and {len(df.columns) - 15} more columns")

    return "\n".join(lines)


def _layer3_llm_detection(
    df: pd.DataFrame,
    llm_provider,
) -> Tuple[Optional[str], float, str, Optional[str]]:
    """Layer 3: Use LLM to detect domain.

    Returns (domain, confidence, reasoning, suggested_target).
    """
    if llm_provider is None:
        return None, 0.0, "No LLM provider available", None

    try:
        # Build prompt
        column_summary = _build_column_summary(df)
        sample_values = _build_sample_values(df)

        prompt = FINGERPRINT_PROMPT_TEMPLATE.format(
            rows=len(df),
            columns=len(df.columns),
            column_summary=column_summary,
            sample_values=sample_values,
        )

        # Check if provider has fingerprint_dataset method
        if hasattr(llm_provider, "fingerprint_dataset"):
            response = llm_provider.fingerprint_dataset(prompt)
        else:
            # Fallback: use generate_narrative with system prompt override
            logger.warning("LLM provider missing fingerprint_dataset method, skipping layer 3")
            return None, 0.0, "LLM provider doesn't support fingerprinting", None

        if not response:
            return None, 0.0, "LLM returned empty response", None

        # Parse response
        result = response if isinstance(response, dict) else json.loads(response)

        domain = result.get("domain", "general")
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "LLM classification")
        suggested_target = result.get("suggested_target")

        # Validate domain
        valid_domains = {"finance", "healthcare", "retail", "ecommerce", "hr", "marketing", "general"}
        if domain not in valid_domains:
            domain = "general"
            confidence *= 0.5

        return domain, confidence, reasoning, suggested_target

    except Exception as e:
        logger.warning(f"Layer 3 LLM detection failed: {e}")
        return None, 0.0, f"LLM error: {str(e)}", None


# ---------------------------------------------------------------------------
# Main fingerprinting function
# ---------------------------------------------------------------------------

def fingerprint_dataset(
    df: pd.DataFrame,
    filename: str,
    profile: dict,
    llm_provider=None,
) -> dict:
    """
    Detect the business domain of a dataset using 3-layer detection.

    Layer 1: Keyword matching in filename and column names
    Layer 2: Column value pattern analysis and target detection
    Layer 3: LLM-based semantic classification

    Args:
        df: Loaded pandas DataFrame
        filename: Original filename (for layer 1 keyword matching)
        profile: Profile dict from profile_data() (used for layer 2)
        llm_provider: Optional LLM provider for layer 3 detection (FailoverProvider recommended)

    Returns:
        Dict with structure:
        {
            "status": "success",
            "domain": "finance" | "healthcare" | "retail" | "ecommerce" | "hr" | "marketing" | "general",
            "confidence": 0.0-1.0,
            "explainability": {
                "reasons": ["reason1", "reason2"],
                "column_matches": ["col1", "col2"],
                "layer": "keyword" | "column" | "llm" | "none",
                "llm_reasoning": str | None
            },
            "suggested_target": str | None,
            "layer_scores": {"keyword": 0.X, "column": 0.X, "llm": 0.X}
        }

        Or {"status": "error", "message": str} on failure.
    """
    try:
        logger.info(f"Fingerprinting {filename}: {df.shape[0]} rows x {df.shape[1]} cols")

        column_names = df.columns.tolist()

        # Layer 1: Keyword matching
        l1_domain, l1_conf, l1_matches = _layer1_keyword_match(filename, column_names)
        logger.info(f"Layer 1: domain={l1_domain}, confidence={l1_conf:.2f}, matches={len(l1_matches)}")

        # Layer 2: Column pattern analysis
        l2_domain, l2_conf, l2_matches = _layer2_column_analysis(df, column_names)
        logger.info(f"Layer 2: domain={l2_domain}, confidence={l2_conf:.2f}, matches={len(l2_matches)}")

        # Layer 3: LLM detection (only if previous layers are weak)
        l3_domain, l3_conf, l3_reasoning, l3_target = None, 0.0, "", None
        if llm_provider and (not l1_domain or l1_conf < 0.7):
            l3_domain, l3_conf, l3_reasoning, l3_target = _layer3_llm_detection(df, llm_provider)
            logger.info(f"Layer 3: domain={l3_domain}, confidence={l3_conf:.2f}")

        # Aggregate results
        layer_scores = {
            "keyword": l1_conf,
            "column": l2_conf,
            "llm": l3_conf,
        }

        # Decision logic: prioritize layers by confidence
        candidates = []
        if l1_domain and l1_conf > 0.0:
            candidates.append((l1_domain, l1_conf, "keyword", l1_matches))
        if l2_domain and l2_conf > 0.0:
            candidates.append((l2_domain, l2_conf, "column", l2_matches))
        if l3_domain and l3_conf > 0.0:
            candidates.append((l3_domain, l3_conf, "llm", [l3_reasoning]))

        if not candidates:
            # No detection — default to general
            return safe_json_serialize({
                "status": "success",
                "domain": "general",
                "confidence": 0.0,
                "explainability": {
                    "reasons": ["No clear domain signals detected"],
                    "column_matches": [],
                    "layer": "none",
                    "llm_reasoning": None,
                },
                "suggested_target": None,
                "layer_scores": layer_scores,
            })

        # Pick highest confidence layer
        best_domain, best_conf, best_layer, best_matches = max(candidates, key=lambda x: x[1])

        # If multiple layers agree, boost confidence
        domain_votes = [d for d, c, l, m in candidates if d == best_domain]
        if len(domain_votes) >= 2:
            best_conf = min(best_conf * 1.2, 1.0)

        # Build explainability dict
        reasons = []
        column_matches = []
        llm_reasoning = None

        if best_layer == "keyword":
            # Separate keyword matches (from filename/column names) vs column names
            keyword_matches = [m for m in best_matches if m not in column_names]
            col_name_matches = [m for m in best_matches if m in column_names]

            if keyword_matches:
                reasons.append(f"Found domain keywords: {', '.join(keyword_matches[:5])}")
            if col_name_matches:
                column_matches.extend(col_name_matches[:10])
                reasons.append(f"Column names match '{best_domain}' domain patterns")

        elif best_layer == "column":
            # Extract column names from pattern matches
            for match in best_matches:
                if ":" in match:
                    col_name = match.split(":")[0]
                    if col_name in column_names:
                        column_matches.append(col_name)
                elif "~" in match:
                    col_name = match.split("~")[0]
                    if col_name in column_names:
                        column_matches.append(col_name)

            reasons.append(f"Column value patterns match '{best_domain}' domain")
            if column_matches:
                reasons.append(f"Found {len(column_matches)} relevant columns")

        else:  # llm
            llm_reasoning = l3_reasoning or f"LLM classified dataset as '{best_domain}' domain"
            reasons.append(f"LLM-based classification: {best_domain}")

        # Find suggested target
        suggested_target = l3_target
        if not suggested_target and best_domain in DOMAIN_PATTERNS:
            # Try to find target from common targets
            for target in DOMAIN_PATTERNS[best_domain]["common_targets"]:
                for col in column_names:
                    col_clean = col.lower().replace("_", "").replace(" ", "")
                    target_clean = target.lower().replace("_", "")
                    if target_clean in col_clean:
                        suggested_target = col
                        break
                if suggested_target:
                    break

        logger.info(f"Final fingerprint: {best_domain} (confidence={best_conf:.2f}, layer={best_layer})")

        return safe_json_serialize({
            "status": "success",
            "domain": best_domain,
            "confidence": round(best_conf, 2),
            "explainability": {
                "reasons": reasons,
                "column_matches": list(set(column_matches))[:10],  # Dedupe and limit
                "layer": best_layer,
                "llm_reasoning": llm_reasoning,
            },
            "suggested_target": suggested_target,
            "layer_scores": layer_scores,
        })

    except Exception as e:
        logger.error(f"Fingerprinting failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
