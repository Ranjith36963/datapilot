"""
Claude LLM Provider — Anthropic API integration.

Uses the Anthropic Python SDK with tool_use for structured routing.
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..utils.config import Config
from .provider import LLMProvider, NarrativeResult, RoutingResult

logger = logging.getLogger("datapilot.llm.claude")


class ClaudeProvider(LLMProvider):
    """LLM provider using Anthropic's Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it via environment variable or pass api_key to ClaudeProvider."
                )
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def route_question(
        self,
        question: str,
        data_context: Dict[str, Any],
        skill_catalog: str,
    ) -> RoutingResult:
        """Route a question using Claude with tool_use."""
        client = self._get_client()

        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', c.get('dtype', 'unknown'))})"
            for c in data_context.get("columns", [])
        ) or "unknown columns"

        system = (
            "You are a data analysis routing assistant. "
            "Given a user question and dataset context, pick the best analysis skill. "
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
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
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
            logger.warning(f"Claude routing failed: {e}")
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
        """Generate narrative using Claude."""
        client = self._get_client()

        result_str = json.dumps(analysis_result, default=str)
        if len(result_str) > 8000:
            result_str = result_str[:8000] + "... (truncated)"

        prompt = f"Analysis results:\n{result_str}\n\n"
        if question:
            prompt += f"Original question: {question}\n\n"
        prompt += (
            "Write a concise narrative summary for a non-technical audience. "
            "Include key findings and suggested next steps. "
            "Respond with JSON: "
            '{"text": "<narrative>", "key_points": ["..."], "suggestions": ["..."]}'
        )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system="You are a data analyst writing clear insights.",
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
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
            logger.warning(f"Claude narrative failed: {e}")
            return NarrativeResult(
                text="Analysis complete. See raw results for details.",
                key_points=[],
                suggestions=[],
            )

    def suggest_chart(
        self,
        data_context: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Suggest a chart using Claude."""
        client = self._get_client()

        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', 'unknown')})"
            for c in data_context.get("columns", [])
        )

        prompt = (
            f"Dataset columns: {columns_info}\n"
            "Suggest the best chart for exploring this data. "
            "Respond with JSON: "
            '{"chart_type": "<type>", "x": "<col>", "y": "<col_or_null>", '
            '"hue": "<col_or_null>", "title": "<title>"}'
        )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return {"chart_type": "histogram", "x": None, "y": None, "title": "Data Distribution"}
