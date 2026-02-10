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
            "You are a friendly, expert data analyst explaining results to a business person. "
            "Write conversationally — like a colleague sharing insights over coffee. "
            "IMPORTANT: Only cite numbers that appear verbatim in the analysis results. "
            "Never invent or estimate statistics. If no numbers are available, "
            "describe what was done without fabricating data. Be concise but insightful. "
            "Format numbers cleanly for business users. Round decimals to 1-2 places. "
            "Convert proportions to percentages (0.946 → 94.6%). Never show more than 2 decimal places. "
            "When describing chart results where y is binary (0/1 or True/False), describe proportions "
            "as 'rate' or 'percentage', not 'mean'. For example, say 'churn rate of 46%' not 'mean churn of 0.46'. "
            "Respond ONLY with valid JSON: "
            '{"text": "<2-4 sentence narrative>", "key_points": ["<point1>", "<point2>", ...], '
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
        """Suggest a chart using Groq."""
        client = self._get_client()

        columns_info = "\n".join(
            f"- {c['name']} ({c.get('semantic_type', 'unknown')}, {c.get('n_unique', '?')} unique)"
            for c in data_context.get("columns", [])
        )

        allowed_types = "histogram, bar, scatter, line, box, violin, heatmap, pie, area, strip"

        system = (
            "You are a data visualization expert. "
            "Suggest the single best chart for this dataset. "
            "Respond with JSON only."
        )

        prompt = (
            f"Dataset: {data_context.get('shape', 'unknown')}\n"
            f"Columns:\n{columns_info}\n\n"
            f"Allowed chart types: {allowed_types}\n\n"
            "Pick the most insightful chart. Use exact column names from the list above.\n"
            "Respond with JSON: "
            '{"chart_type": "<type>", "x": "<column_name>", "y": "<column_name_or_null>", '
            '"hue": "<column_name_or_null>", "title": "<descriptive title>"}\n'
            "Use null (not the string \"null\") for optional fields."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text)

            # Clean up "null" strings → None
            for key in ("x", "y", "hue"):
                if isinstance(result.get(key), str) and result[key].lower() == "null":
                    result[key] = None

            # Validate chart_type against allowed list
            allowed = {"histogram", "bar", "scatter", "line", "box", "violin", "heatmap", "pie", "area", "strip"}
            if result.get("chart_type") not in allowed:
                result["chart_type"] = "histogram"

            return result
        except Exception as e:
            logger.warning(f"Groq chart suggestion failed: {e}")
            # Smart fallback: pick first numeric column for histogram
            cols = data_context.get("columns", [])
            x_col = None
            for c in cols:
                if c.get("semantic_type") == "numeric":
                    x_col = c["name"]
                    break
            return {
                "chart_type": "histogram",
                "x": x_col,
                "y": None,
                "hue": None,
                "title": f"Distribution of {x_col}" if x_col else "Data Distribution",
            }
