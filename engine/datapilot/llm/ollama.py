"""
Ollama LLM Provider — local-first LLM integration.

Connects to Ollama server at localhost:11434 (configurable).
Default model: llama3.2. Uses structured prompts for function calling.
"""

import json
import logging
from typing import Any

from ..utils.config import Config
from .provider import LLMProvider, NarrativeResult, RoutingResult

logger = logging.getLogger("datapilot.llm.ollama")


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama server."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
    ):
        self.host = (host or Config.OLLAMA_HOST).rstrip("/")
        self.model = model or Config.OLLAMA_MODEL
        self._session = None

    def _request(self, prompt: str, system: str | None = None) -> str:
        """Send a request to the Ollama API and return the response text."""
        import urllib.error
        import urllib.request

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "")
        except urllib.error.URLError as e:
            logger.error(f"Ollama connection failed: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            ) from e

    def route_question(
        self,
        question: str,
        data_context: dict[str, Any],
        skill_catalog: str,
    ) -> RoutingResult:
        """Route a question to the best skill using Ollama."""
        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', c.get('dtype', 'unknown'))})"
            for c in data_context.get("columns", [])
        ) or "unknown columns"

        system = (
            "You are a data analysis routing assistant. "
            "Given a user question and dataset info, pick the best analysis skill. "
            "Respond ONLY with valid JSON: "
            '{"skill": "<skill_name>", "params": {<params>}, "confidence": <0-1>, "reasoning": "<why>"}'
        )

        prompt = (
            f"Dataset columns: {columns_info}\n"
            f"Dataset shape: {data_context.get('shape', 'unknown')}\n\n"
            f"Available skills:\n{skill_catalog}\n\n"
            f"User question: {question}\n\n"
            "Pick the best skill and parameters. Respond with JSON only."
        )

        try:
            response = self._request(prompt, system=system)
            # Parse JSON from response (handle markdown code blocks)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text)

            return RoutingResult(
                skill_name=result.get("skill", "profile_data"),
                parameters=result.get("params", {}),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ConnectionError, Exception) as e:
            logger.warning(f"Routing failed, falling back to profile_data: {e}")
            return RoutingResult(
                skill_name="profile_data",
                parameters={},
                confidence=0.1,
                reasoning=f"Fallback due to error: {e}",
            )

    def generate_narrative(
        self,
        analysis_result: dict[str, Any],
        question: str | None = None,
        skill_name: str | None = None,
        conversation_context: str | None = None,
    ) -> NarrativeResult:
        """Generate a human-readable narrative from analysis results."""
        system = (
            "You are a data analyst writing insights for a non-technical audience. "
            "Be concise, highlight key findings, and suggest next steps."
        )

        # Truncate large results to fit context
        result_str = json.dumps(analysis_result, default=str)
        if len(result_str) > 4000:
            result_str = result_str[:4000] + "... (truncated)"

        prompt = f"Analysis results:\n{result_str}\n\n"
        if question:
            prompt += f"Original question: {question}\n\n"
        prompt += (
            "Write a brief narrative summary. Include:\n"
            "1. A 1-2 sentence headline summary\n"
            "2. 3-5 key points as bullet points\n"
            "3. 2-3 suggested next steps\n\n"
            "Respond with JSON: "
            '{"text": "<narrative>", "key_points": ["..."], "suggestions": ["..."]}'
        )

        try:
            response = self._request(prompt, system=system)
            text = response.strip()
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
            logger.warning(f"Narrative generation failed: {e}")
            return NarrativeResult(
                text="Analysis complete. See the raw results for details.",
                key_points=[],
                suggestions=["Review the raw analysis output"],
            )

    def suggest_chart(
        self,
        data_context: dict[str, Any],
        analysis_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Suggest a chart type based on data context."""
        columns_info = ", ".join(
            f"{c['name']} ({c.get('semantic_type', 'unknown')})"
            for c in data_context.get("columns", [])
        )

        system = "You are a data visualization expert. Suggest the best chart."
        prompt = (
            f"Dataset columns: {columns_info}\n"
            "Suggest a chart. Respond with JSON: "
            '{"chart_type": "<type>", "x": "<col>", "y": "<col_or_null>", '
            '"hue": "<col_or_null>", "title": "<title>"}'
        )

        try:
            response = self._request(prompt, system=system)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return {"chart_type": "histogram", "x": None, "y": None, "title": "Data Distribution"}

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False
