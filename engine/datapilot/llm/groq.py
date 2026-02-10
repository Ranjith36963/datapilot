"""
Groq LLM Provider — Groq API integration.

Uses the OpenAI Python SDK with Groq's OpenAI-compatible endpoint.
Requires GROQ_API_KEY environment variable.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..utils.config import Config
from .provider import LLMProvider, NarrativeResult, RoutingResult

logger = logging.getLogger("datapilot.llm.groq")

MAX_NARRATION_ROWS = 20
MAX_NARRATION_CHARS = 4000


# Keys that carry no analytical value and waste token budget in narration prompts.
# Defense-in-depth: analyst.py also strips these before calling providers.
_NARRATION_EXCLUDED_KEYS = {
    "chart_base64", "image_base64", "chart_path", "chart_html_path",
}


def _truncate_for_narration(result: Dict[str, Any]) -> str:
    """Truncate analysis result to keep narration prompts small.

    Strips excluded keys (base64 blobs, file paths), caps list values at
    MAX_NARRATION_ROWS items, and caps the final JSON at MAX_NARRATION_CHARS.
    """
    truncated: Dict[str, Any] = {}
    for key, value in result.items():
        if key in _NARRATION_EXCLUDED_KEYS:
            continue
        if isinstance(value, list) and len(value) > MAX_NARRATION_ROWS:
            truncated[key] = value[:MAX_NARRATION_ROWS]
            truncated[f"_{key}_truncated"] = f"{MAX_NARRATION_ROWS} of {len(value)} items shown"
        elif isinstance(value, dict):
            inner: Dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(v, list) and len(v) > MAX_NARRATION_ROWS:
                    inner[k] = v[:MAX_NARRATION_ROWS]
                    inner[f"_{k}_truncated"] = f"{MAX_NARRATION_ROWS} of {len(v)} items shown"
                else:
                    inner[k] = v
            truncated[key] = inner
        else:
            truncated[key] = value

    result_str = json.dumps(truncated, default=str)
    if len(result_str) > MAX_NARRATION_CHARS:
        result_str = result_str[:MAX_NARRATION_CHARS] + "... (truncated)"
    return result_str


class GroqProvider(LLMProvider):
    """LLM provider using Groq's API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.GROQ_MODEL
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client pointed at Groq."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. "
                    "Set it via environment variable or pass api_key to GroqProvider."
                )
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def route_question(
        self,
        question: str,
        data_context: Dict[str, Any],
        skill_catalog: str,
    ) -> RoutingResult:
        """Route a question using Groq."""
        client = self._get_client()

        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', c.get('dtype', 'unknown'))})"
            for c in data_context.get("columns", [])
        ) or "unknown columns"

        system = (
            "You are a data analysis routing assistant. "
            "Pick the best analysis skill for the user's question. "
            "Respond with JSON only."
        )

        prompt = (
            f"Dataset columns: {columns_info}\n"
            f"Dataset shape: {data_context.get('shape', 'unknown')}\n\n"
            f"Available skills:\n{skill_catalog}\n\n"
            f"User question: {question}\n\n"
            "Respond with JSON: "
            '{"skill": "<name>", "params": {}, "confidence": <0-1>, "reasoning": "<why>"}'
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0,
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text)

            return RoutingResult(
                skill_name=result.get("skill", "profile_data"),
                parameters=result.get("params", {}),
                confidence=float(result.get("confidence", 0.8)),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Groq routing failed: {e}")
            return RoutingResult(
                skill_name="profile_data",
                parameters={},
                confidence=0.1,
                reasoning=f"Fallback: {e}",
            )

    def generate_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> NarrativeResult:
        """Generate narrative using Groq."""
        client = self._get_client()

        result_str = _truncate_for_narration(analysis_result)
        logger.info(f"Narration prompt size: {len(result_str)} chars")

        system = (
            "You are a senior data analyst explaining results to a business colleague. "
            "Speak naturally and conversationally — not like a report, but like a smart colleague "
            "walking someone through what they found and why it matters. "
            "IMPORTANT: Only cite numbers that appear verbatim in the analysis results. "
            "Never invent or estimate statistics. If no numbers are available, "
            "describe what was done without fabricating data. "
            "Format numbers cleanly for business users. Round decimals to 1-2 places. "
            "Convert proportions to percentages (0.946 → 94.6%). Never show more than 2 decimal places. "
            "When describing chart results where y is binary (0/1 or True/False), describe proportions "
            "as 'rate' or 'percentage', not 'mean'. For example, say 'churn rate of 46%' not 'mean churn of 0.46'. "
            "Narrative style rules: "
            "- Explain WHY findings matter, not just WHAT they are. Connect numbers to business implications. "
            "- When discussing correlations, explain what the relationship means practically "
            "(e.g., 'as X increases, Y tends to...' and why that matters for decisions). "
            "- When discussing classification, explain which features drive predictions and suggest what actions to take. "
            "- Keep narratives to 3-5 sentences — concise but insightful. "
            "- Never start with 'The analysis reveals...' or 'The results show...' — vary your openings. "
            "- End with a forward-looking insight or recommendation when possible. "
            "Respond ONLY with valid JSON: "
            '{"text": "<3-5 sentence narrative>", "key_points": ["<point1>", "<point2>", ...], '
            '"suggestions": ["<follow-up question 1>", "<follow-up question 2>"]}'
        )

        skill_hint = f"\nAnalysis type: {skill_name}" if skill_name else ""
        question_line = f"\nUser asked: {question}" if question else ""

        prompt = (
            f"Analysis results:\n{result_str}\n"
            f"{skill_hint}{question_line}\n\n"
            "Write a natural narrative summarizing the key findings. "
            "Only reference numbers present in the results above. Do not invent statistics. "
            "3-5 key points. 2-3 follow-up questions as suggestions. "
            "When suggesting follow-up questions, only reference column names from "
            "the _dataset_columns list in the results. Use exact column names including spaces."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text)

            return NarrativeResult(
                text=result.get("text", "Analysis complete."),
                key_points=result.get("key_points", []),
                suggestions=result.get("suggestions", []),
            )
        except Exception as e:
            logger.warning(f"Groq narrative failed: {e}")
            return NarrativeResult(
                text="",
                key_points=[],
                suggestions=[],
            )

    def suggest_chart(
        self,
        data_context: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Suggest 3-5 ranked charts using Groq."""
        client = self._get_client()

        columns_info = "\n".join(
            f"- {c['name']} ({c.get('semantic_type', 'unknown')}, {c.get('n_unique', '?')} unique)"
            for c in data_context.get("columns", [])
        )

        allowed_types = "histogram, bar, scatter, line, box, violin, heatmap, pie, area, strip"

        system = (
            "You are a data visualization expert. "
            "Suggest 3-5 ranked chart ideas for this dataset. "
            "Respond with a JSON array only."
        )

        prompt = (
            f"Dataset: {data_context.get('shape', 'unknown')}\n"
            f"Columns:\n{columns_info}\n\n"
            f"Allowed chart types: {allowed_types}\n\n"
            "Suggest 3-5 charts ranked by insight value. Use exact column names from the list above.\n"
            "Respond with a JSON array: "
            '[{"chart_type": "<type>", "x": "<column_name>", "y": "<column_name_or_null>", '
            '"hue": "<column_name_or_null>", "title": "<descriptive title>", '
            '"reason": "<one-line explanation of why this chart is useful>"}, ...]\n'
            "Use null (not the string \"null\") for optional fields."
        )

        allowed = {"histogram", "bar", "scatter", "line", "box", "violin", "heatmap", "pie", "area", "strip"}

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            suggestions_raw = json.loads(text)

            # Normalise: if the LLM returned a dict with a "suggestions" key, unwrap it
            if isinstance(suggestions_raw, dict) and "suggestions" in suggestions_raw:
                suggestions_raw = suggestions_raw["suggestions"]

            if not isinstance(suggestions_raw, list):
                suggestions_raw = [suggestions_raw]

            suggestions: List[Dict[str, Any]] = []
            for item in suggestions_raw:
                # Clean up "null" strings → None
                for key in ("x", "y", "hue"):
                    if isinstance(item.get(key), str) and item[key].lower() == "null":
                        item[key] = None

                # Validate chart_type against allowed list
                if item.get("chart_type") not in allowed:
                    item["chart_type"] = "histogram"

                # Ensure reason field exists
                if "reason" not in item:
                    item["reason"] = ""

                suggestions.append(item)

            return {"suggestions": suggestions}
        except Exception as e:
            logger.warning(f"Groq chart suggestion failed: {e}")
            # Smart fallback: build 2 default suggestions using actual column names
            cols = data_context.get("columns", [])
            numeric_col = None
            categorical_col = None
            second_numeric_col = None
            for c in cols:
                if c.get("semantic_type") == "numeric" and numeric_col is None:
                    numeric_col = c["name"]
                elif c.get("semantic_type") == "numeric" and second_numeric_col is None:
                    second_numeric_col = c["name"]
                elif c.get("semantic_type") == "categorical" and categorical_col is None:
                    categorical_col = c["name"]

            fallback: List[Dict[str, Any]] = [
                {
                    "chart_type": "histogram",
                    "x": numeric_col,
                    "y": None,
                    "hue": None,
                    "title": f"Distribution of {numeric_col}" if numeric_col else "Data Distribution",
                    "reason": "Histograms are a good starting point to understand value distributions.",
                },
            ]
            if numeric_col and second_numeric_col:
                fallback.append({
                    "chart_type": "scatter",
                    "x": numeric_col,
                    "y": second_numeric_col,
                    "hue": categorical_col,
                    "title": f"{numeric_col} vs {second_numeric_col}",
                    "reason": "Scatter plots reveal relationships between numeric variables.",
                })
            elif numeric_col and categorical_col:
                fallback.append({
                    "chart_type": "box",
                    "x": categorical_col,
                    "y": numeric_col,
                    "hue": None,
                    "title": f"{numeric_col} by {categorical_col}",
                    "reason": "Box plots show how a numeric variable varies across categories.",
                })

            return {"suggestions": fallback}
