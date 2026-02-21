"""
LLM Provider — abstract base class for all LLM integrations.

Defines the interface that Ollama, Claude, and OpenAI providers implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class RoutingResult:
    """Result of routing a user question to a skill.

    route_method values:
      - "keyword"     — matched via keyword pattern table
      - "smart_first" — routed to smart_query (LLM-generated pandas code)
      - "semantic"    — matched via semantic intent detection (no LLM)
      - "llm"         — routed by LLM provider
      - "fallback"    — no match found, fell back to profile_data
    """
    skill_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    route_method: str = "llm"


@dataclass
class NarrativeResult:
    """Result of generating a narrative from analysis output."""
    text: str
    key_points: List[str]
    suggestions: List[str]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def route_question(
        self,
        question: str,
        data_context: Dict[str, Any],
        skill_catalog: str,
    ) -> RoutingResult:
        """
        Route a user question to the appropriate analysis skill.

        Args:
            question: Natural language question from the user.
            data_context: Info about the loaded dataset (columns, types, shape).
            skill_catalog: String description of available skills.

        Returns:
            RoutingResult with skill name, parameters, and confidence.
        """
        ...

    @abstractmethod
    def generate_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: Optional[str] = None,
        skill_name: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> NarrativeResult:
        """
        Generate a human-readable narrative from analysis results.

        Args:
            analysis_result: Raw output from a skill function.
            question: The original user question (for context).
            skill_name: Name of the skill that produced the results.
            conversation_context: Summary of previous Q&A pairs for continuity.

        Returns:
            NarrativeResult with text, key points, and suggestions.
        """
        ...

    @abstractmethod
    def suggest_chart(
        self,
        data_context: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest chart types and parameters for the data.

        Returns:
            Dict with a "suggestions" key containing a list of suggestion dicts.
            Each suggestion has: chart_type, x, y (nullable), hue (nullable),
            title, and reason (one-line explanation of why this chart is useful).
        """
        ...

    def generate_chart_insight(self, chart_summary: Dict[str, Any]) -> str:
        """
        Generate a one-sentence insight from chart summary data.

        Default implementation returns empty string. Override in subclasses.
        """
        return ""

    def stream_response(
        self,
        question: str,
        data_context: Dict[str, Any],
    ) -> str:
        """
        Stream a response to a question. Default non-streaming implementation.

        Override in subclasses for true streaming support.
        """
        routing = self.route_question(question, data_context, "")
        return f"I would use the '{routing.skill_name}' skill to answer this."

    def fingerprint_dataset(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Classify a dataset's business domain based on a formatted prompt.

        Args:
            prompt: Formatted prompt with dataset overview, column summary, and samples.
                   Should be created using FINGERPRINT_PROMPT_TEMPLATE from
                   datapilot.data.fingerprint module.

        Returns:
            Dict with keys: domain, confidence, reasoning, suggested_target
            or None if classification fails.

        Expected JSON response format:
            {
              "domain": "<finance|healthcare|retail|ecommerce|hr|marketing|general>",
              "confidence": <0.0-1.0>,
              "reasoning": "<one_sentence>",
              "suggested_target": "<column_name_or_null>"
            }

        Default implementation returns None. Override in subclasses.
        """
        return None

    def understand_dataset(self, snapshot: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a dataset snapshot and return structured understanding.

        Returns:
            Dict with domain, domain_short, target_column, target_type,
            key_observations, suggested_questions, data_quality_notes.
            Or None if analysis fails.
        """
        return None

    def generate_plan(self, prompt: str) -> Optional[str]:
        """
        Generate an analysis plan from a prompt.

        Returns:
            JSON string with title and steps, or None.
        """
        return None

    def generate_summary(self, prompt: str) -> Optional[str]:
        """
        Generate a summary of analysis results.

        Returns:
            Summary text string, or None.
        """
        return None


def smart_fallback_suggestions(data_context: Dict[str, Any]) -> Dict[str, Any]:
    """Build diverse, quality-aware fallback chart suggestions from column metadata.

    Uses n_unique, null_pct, and n_rows to pick analytically useful columns
    and diverse chart types. Always returns 4-5 suggestions.
    """
    cols = data_context.get("columns", [])
    n_rows = data_context.get("n_rows", 0)

    # Classify columns, skip IDs (n_unique == n_rows) and high-null (>50%)
    numeric: List[Dict[str, Any]] = []
    categorical: List[Dict[str, Any]] = []
    datetime_cols: List[Dict[str, Any]] = []

    for c in cols:
        # Skip likely ID columns: integer with all unique values
        dtype = c.get("dtype", "")
        is_int = "int" in dtype
        if is_int and c.get("n_unique", 0) == n_rows and n_rows > 10:
            continue
        # Skip high-null columns
        if c.get("null_pct", 0) > 50:
            continue

        st = c.get("semantic_type", "")
        if st == "numeric":
            numeric.append(c)
        elif st in ("categorical", "boolean"):
            categorical.append(c)
        elif st == "datetime":
            datetime_cols.append(c)

    # Sort numeric by variance proxy: prefer columns with moderate n_unique
    # (not too few = constant, not too many = possibly continuous ID-like)
    numeric.sort(key=lambda c: -min(c.get("n_unique", 1), n_rows // 2))
    # Sort categorical by cardinality: prefer low-cardinality for hue/pie
    categorical.sort(key=lambda c: c.get("n_unique", 999))

    suggestions: List[Dict[str, Any]] = []

    # 1. Histogram of best numeric column
    if numeric:
        col = numeric[0]
        suggestions.append({
            "chart_type": "histogram",
            "x": col["name"], "y": None, "hue": None,
            "title": f"Distribution of {col['name']}",
            "reason": "Understand the spread and shape of this numeric variable.",
        })

    # 2. Bar chart of best categorical column
    if categorical:
        col = categorical[0]
        suggestions.append({
            "chart_type": "bar",
            "x": col["name"], "y": None, "hue": None,
            "title": f"Frequency of {col['name']}",
            "reason": "See how observations are distributed across categories.",
        })

    # 3. Scatter of two numeric columns (colored by categorical if available)
    if len(numeric) >= 2:
        hue = categorical[0]["name"] if categorical and categorical[0].get("n_unique", 0) <= 10 else None
        suggestions.append({
            "chart_type": "scatter",
            "x": numeric[0]["name"], "y": numeric[1]["name"], "hue": hue,
            "title": f"{numeric[0]['name']} vs {numeric[1]['name']}",
            "reason": "Reveal relationships and clusters between numeric variables.",
        })

    # 4. Box plot of numeric by categorical
    if numeric and categorical:
        cat = next((c for c in categorical if c.get("n_unique", 0) <= 10), None)
        if cat:
            num = numeric[1] if len(numeric) > 1 else numeric[0]
            suggestions.append({
                "chart_type": "box",
                "x": cat["name"], "y": num["name"], "hue": None,
                "title": f"{num['name']} by {cat['name']}",
                "reason": "Compare distributions across groups and spot outliers.",
            })

    # 5. Pie chart for low-cardinality categorical
    if categorical:
        low_card = next((c for c in categorical if 2 <= c.get("n_unique", 0) <= 8), None)
        if low_card and not any(s["chart_type"] == "bar" and s["x"] == low_card["name"] for s in suggestions):
            suggestions.append({
                "chart_type": "pie",
                "x": low_card["name"], "y": None, "hue": None,
                "title": f"Proportion of {low_card['name']}",
                "reason": "Quickly see the relative share of each category.",
            })

    # 6. Heatmap if 3+ numeric columns
    if len(numeric) >= 3:
        suggestions.append({
            "chart_type": "heatmap",
            "x": None, "y": None, "hue": None,
            "title": "Correlation Heatmap",
            "reason": "Discover which numeric variables are related to each other.",
        })

    # 7. Line chart if datetime exists
    if datetime_cols and numeric:
        suggestions.append({
            "chart_type": "line",
            "x": datetime_cols[0]["name"], "y": numeric[0]["name"], "hue": None,
            "title": f"{numeric[0]['name']} over Time",
            "reason": "Spot trends and patterns over time.",
        })

    # Ensure at least one suggestion even for edge cases
    if not suggestions and cols:
        suggestions.append({
            "chart_type": "histogram",
            "x": cols[0]["name"], "y": None, "hue": None,
            "title": f"Distribution of {cols[0]['name']}",
            "reason": "A starting point to explore this column's values.",
        })

    # Cap at 5 for UI readability
    return {"suggestions": suggestions[:5], "source": "fallback"}
