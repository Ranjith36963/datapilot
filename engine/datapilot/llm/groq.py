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


def _truncate_for_narration(result: Dict[str, Any]) -> str:
    """Truncate analysis result to keep narration prompts small.

    Caps any list values at MAX_NARRATION_ROWS items and the final
    JSON string at MAX_NARRATION_CHARS characters.
    """
    truncated: Dict[str, Any] = {}
    for key, value in result.items():
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
            "Use specific numbers from the results. Be concise but insightful. "
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
            "Include specific numbers. 3-5 key points. 2-3 follow-up questions as suggestions."
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

        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', 'unknown')})"
            for c in data_context.get("columns", [])
        )

        prompt = (
            f"Dataset columns: {columns_info}\n"
            "Suggest the best chart. Respond with JSON: "
            '{"chart_type": "<type>", "x": "<col>", "y": "<col_or_null>", '
            '"hue": "<col_or_null>", "title": "<title>"}'
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return {"chart_type": "histogram", "x": None, "y": None, "title": "Data Distribution"}
