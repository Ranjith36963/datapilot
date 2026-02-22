"""
Dataset fingerprinting — LLM-first dataset understanding.

D3: LLM freely classifies any dataset — no hardcoded domains.
No keyword dicts, no 3-layer detection — LLM or nothing.
"""

import json
from dataclasses import dataclass

import pandas as pd

from ..utils.helpers import setup_logging

logger = setup_logging("datapilot.fingerprint")

# PII column name patterns for redaction
_PII_PATTERNS = {
    "email", "e_mail",
    "phone", "phone_number", "telephone", "mobile",
    "ssn", "social_security", "national_id",
    "first_name", "last_name", "full_name", "surname",
    "address", "street", "zip", "postal",
    "date_of_birth", "dob", "birth_date",
    "passport", "drivers_license", "license_number",
    "credit_card", "card_number", "account_number",
    "password", "secret",
}

# Keep for backward compat — imported by gemini.py / groq.py fingerprint_dataset()
FINGERPRINT_SYSTEM_PROMPT = (
    "You are a dataset domain classifier. "
    "Given column names, sample values, and basic statistics, "
    "determine which business domain this dataset belongs to. "
    "Respond ONLY with valid JSON."
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_pii_column(col_name: str) -> bool:
    """Check if column name suggests PII data."""
    col_lower = col_name.lower().replace("_", "").replace(" ", "")
    for pattern in _PII_PATTERNS:
        if pattern.replace("_", "") in col_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# build_data_snapshot
# ---------------------------------------------------------------------------

def build_data_snapshot(df: pd.DataFrame, filename: str, profile: dict = None) -> str:
    """Build a concise text snapshot of the dataset for LLM consumption.

    Target: <2000 tokens (~8000 chars).

    Args:
        df: The DataFrame to summarize.
        filename: Original filename.
        profile: Optional profile dict (currently unused; reserved for future).

    Returns:
        A compact text summary suitable for LLM context windows.
    """
    lines: list[str] = []

    # Header
    lines.append(f"Dataset: {filename}")
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    if df.empty or len(df.columns) == 0:
        lines.append("(Empty dataset — no columns)")
        return "\n".join(lines)

    # Decide how many columns to show
    max_cols = 30 if len(df.columns) > 50 else len(df.columns)
    cols_to_show = list(df.columns[:max_cols])

    lines.append("Columns:")
    for col in cols_to_show:
        dtype = str(df[col].dtype)
        col_line = f"  - {col} ({dtype})"

        if _is_pii_column(col):
            col_line += ": [REDACTED]"
        elif pd.api.types.is_bool_dtype(df[col]):
            vc = df[col].value_counts()
            col_line += f" — values: {dict(vc.head(5))}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()
            col_line += f" — min: {col_min}, max: {col_max}, mean: {col_mean:.2f}"
        else:
            top_vals = df[col].value_counts().head(5)
            if len(top_vals) > 0:
                vals_str = ", ".join(str(v) for v in top_vals.index)
                col_line += f" — top values: {vals_str}"

        lines.append(col_line)

    if len(df.columns) > 50:
        remaining = len(df.columns) - 30
        lines.append(f"  ... and {remaining} more columns")

    lines.append("")

    # Missing values summary
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        lines.append("Missing values:")
        for col in cols_to_show:
            nc = int(null_counts.get(col, 0))
            if nc > 0:
                pct = round(nc / len(df) * 100, 1)
                lines.append(f"  - {col}: {nc} ({pct}%)")
        lines.append("")

    # Profile enrichment (top correlations if available)
    if profile and isinstance(profile, dict):
        corr_data = profile.get("correlations") or profile.get("correlation_matrix")
        if isinstance(corr_data, dict):
            pairs = []
            for k, v in corr_data.items():
                if isinstance(v, (int, float)) and abs(v) > 0.3:
                    pairs.append((k, round(v, 3)))
            if pairs:
                pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                lines.append("Notable correlations:")
                for name, val in pairs[:5]:
                    lines.append(f"  - {name}: {val}")
                lines.append("")

    # Sample rows (first 5, limited to first 10 columns)
    lines.append("Sample rows:")
    sample = df.head(5)
    sample_cols = cols_to_show[:10]
    for _, row in sample.iterrows():
        parts = []
        for col in sample_cols:
            val = row[col]
            if _is_pii_column(col):
                val = "[REDACTED]"
            parts.append(f"{col}={val}")
        lines.append("  " + ", ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# understand_dataset
# ---------------------------------------------------------------------------

def _try_understand(snapshot: str, provider) -> DatasetUnderstanding | None:
    """Attempt to get understanding from a single provider."""
    try:
        response = provider.understand_dataset(snapshot)
    except Exception as e:
        logger.warning(f"Provider raised during understand_dataset: {e}")
        return None

    if response is None:
        return None

    # String response — try JSON parse
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM returned unparseable string response")
            return None

    if not isinstance(response, dict):
        logger.warning(f"LLM returned non-dict type: {type(response)}")
        return None

    if "domain" not in response:
        logger.warning("LLM response missing 'domain' field")
        return None

    # Prefer _provider_used injected by FailoverProvider, fallback to provider.name
    provider_name = response.pop("_provider_used", None) or getattr(provider, "name", "DataPilot AI")

    try:
        return DatasetUnderstanding(
            domain=response.get("domain", "general"),
            domain_short=response.get("domain_short", "General"),
            target_column=response.get("target_column"),
            target_type=response.get("target_type"),
            key_observations=response.get("key_observations", []),
            suggested_questions=response.get("suggested_questions", []),
            data_quality_notes=response.get("data_quality_notes", []),
            confidence=float(response.get("confidence", 0.9)),
            provider_used=provider_name,
        )
    except Exception as e:
        logger.warning(f"Failed to construct DatasetUnderstanding: {e}")
        return None


def understand_dataset(
    df: pd.DataFrame,
    filename: str,
    profile: dict,
    llm_provider,
    fallback_provider=None,
) -> DatasetUnderstanding | None:
    """LLM reads the data snapshot and returns structured understanding.

    Two-layer failover: primary → fallback. No deterministic fallback.

    Args:
        df: The DataFrame.
        filename: Original filename.
        profile: Profile dict from profile_data().
        llm_provider: Primary LLM provider (must have understand_dataset method).
        fallback_provider: Optional fallback LLM provider.

    Returns:
        DatasetUnderstanding or None if both providers fail.
    """
    snapshot = build_data_snapshot(df, filename, profile)

    # Try primary
    result = _try_understand(snapshot, llm_provider)
    if result is not None:
        return result

    # Try fallback
    if fallback_provider is not None:
        result = _try_understand(snapshot, fallback_provider)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# fingerprint_dataset — backward-compatible wrapper
# ---------------------------------------------------------------------------

def fingerprint_dataset(
    df: pd.DataFrame,
    filename: str,
    profile: dict,
    llm_provider=None,
) -> dict:
    """Backward-compatible wrapper around understand_dataset.

    Returns a flat dict with status, domain, confidence, etc.
    """
    try:
        if llm_provider is None:
            return {
                "status": "success",
                "domain": "general",
                "domain_short": "General",
                "confidence": 0.0,
            }

        understanding = understand_dataset(df, filename, profile, llm_provider)

        if understanding is None:
            return {
                "status": "success",
                "domain": "general",
                "domain_short": "General",
                "confidence": 0.0,
            }

        return {
            "status": "success",
            "domain": understanding.domain,
            "domain_short": understanding.domain_short,
            "confidence": understanding.confidence,
            "target_column": understanding.target_column,
            "target_type": understanding.target_type,
            "provider_used": understanding.provider_used,
        }
    except Exception as e:
        logger.error(f"fingerprint_dataset failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
