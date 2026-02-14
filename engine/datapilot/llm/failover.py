"""
Failover LLM Provider — task-aware routing across multiple providers.

Routes each LLM task (routing, narrative, chart_suggest, chart_insight)
to the best provider for that task, with automatic failover.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .provider import LLMProvider, NarrativeResult, RoutingResult

logger = logging.getLogger("datapilot.llm.failover")

# Default task routing: which provider to try first for each task type.
# Key = task name, Value = ordered list of provider names to try.
DEFAULT_TASK_ROUTING = {
    "routing": ["groq", "gemini"],  # Groq is faster for routing
    "narrative": ["gemini", "groq"],  # Gemini has lower hallucination
    "chart_suggest": ["gemini", "groq"],  # Gemini has better JSON output
    "chart_insight": ["groq", "gemini"],  # Groq is faster for short text
    "fingerprint": ["gemini", "groq"],  # Gemini has better JSON + accuracy
    "understand": ["gemini", "groq"],   # D3: dataset understanding
    "plan": ["gemini", "groq"],         # D3: analysis plan generation
    "summary": ["gemini", "groq"],      # D3: results summary
}


class FailoverProvider(LLMProvider):
    """Task-aware LLM provider with automatic failover.

    Routes each task type to the best provider, falling back to
    alternatives if the primary fails. Logs which provider handled
    each request for debugging.
    """

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        task_routing: Optional[Dict[str, List[str]]] = None,
    ):
        self.providers = providers
        self.task_routing = task_routing or DEFAULT_TASK_ROUTING
        available = list(providers.keys())
        logger.info(f"FailoverProvider initialized with providers: {available}")

    def _get_provider_order(self, task: str) -> List[Tuple[str, LLMProvider]]:
        """Get ordered list of (name, provider) for a task, filtered to available providers."""
        order = self.task_routing.get(task, list(self.providers.keys()))
        result = []
        for name in order:
            if name in self.providers:
                result.append((name, self.providers[name]))
        # Add any providers not in the routing (safety net)
        for name, provider in self.providers.items():
            if name not in [r[0] for r in result]:
                result.append((name, provider))
        return result

    def route_question(
        self,
        question: str,
        data_context: Dict[str, Any],
        skill_catalog: str,
    ) -> Optional[RoutingResult]:
        """Route question using task-aware provider order."""
        for name, provider in self._get_provider_order("routing"):
            try:
                start = time.time()
                result = provider.route_question(question, data_context, skill_catalog)
                if result and result.skill_name:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Routing handled by {name} ({elapsed}ms)")
                    result.route_method = f"llm:{name}"
                    return result
            except Exception as e:
                logger.warning(f"Routing failed with {name}: {e}")
        return None  # All providers failed — caller falls back to keyword routing

    def generate_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: Optional[str] = None,
        skill_name: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> Optional[NarrativeResult]:
        """Generate narrative using task-aware provider order."""
        for name, provider in self._get_provider_order("narrative"):
            try:
                start = time.time()
                result = provider.generate_narrative(
                    analysis_result, question, skill_name, conversation_context
                )
                if result and result.text and len(result.text) > 30:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Narrative handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Narrative failed with {name}: {e}")
        return None  # All failed — caller falls back to template narrative

    def suggest_chart(
        self,
        data_context: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Suggest charts using task-aware provider order."""
        for name, provider in self._get_provider_order("chart_suggest"):
            try:
                start = time.time()
                result = provider.suggest_chart(data_context, analysis_result)
                if result and result.get("suggestions"):
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Chart suggest handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Chart suggest failed with {name}: {e}")
        return {"suggestions": []}  # All failed — return empty

    def generate_chart_insight(self, chart_summary: Dict[str, Any]) -> str:
        """Generate chart insight using task-aware provider order."""
        for name, provider in self._get_provider_order("chart_insight"):
            try:
                start = time.time()
                result = provider.generate_chart_insight(chart_summary)
                if result and len(result) > 10:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Chart insight handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Chart insight failed with {name}: {e}")
        return ""  # All failed — return empty string

    def fingerprint_dataset(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Classify dataset domain using task-aware provider order (simple interface).

        Args:
            prompt: Formatted prompt from fingerprint.py

        Returns:
            Dict with domain, confidence, reasoning, suggested_target
            or None if all providers fail
        """
        for name, provider in self._get_provider_order("fingerprint"):
            # Check if provider has fingerprint_dataset method
            if not hasattr(provider, "fingerprint_dataset"):
                logger.debug(f"Provider {name} does not support fingerprint_dataset")
                continue

            try:
                start = time.time()
                result = provider.fingerprint_dataset(prompt)
                if result and "domain" in result and "confidence" in result:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Fingerprint (simple) handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Fingerprint (simple) failed with {name}: {e}")

        return None  # All providers failed

    def understand_dataset(self, snapshot: str) -> Optional[Dict[str, Any]]:
        """Understand dataset using task-aware provider order."""
        for name, provider in self._get_provider_order("understand"):
            if not hasattr(provider, "understand_dataset"):
                continue
            try:
                start = time.time()
                result = provider.understand_dataset(snapshot)
                if result and isinstance(result, dict) and "domain" in result:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Understand handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Understand failed with {name}: {e}")
        return None

    def generate_plan(self, prompt: str) -> Optional[str]:
        """Generate analysis plan using task-aware provider order."""
        for name, provider in self._get_provider_order("plan"):
            if not hasattr(provider, "generate_plan"):
                continue
            try:
                start = time.time()
                result = provider.generate_plan(prompt)
                if result and isinstance(result, str) and len(result) > 10:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Plan handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Plan failed with {name}: {e}")
        return None

    def generate_summary(self, prompt: str) -> Optional[str]:
        """Generate summary using task-aware provider order."""
        for name, provider in self._get_provider_order("summary"):
            if not hasattr(provider, "generate_summary"):
                continue
            try:
                start = time.time()
                result = provider.generate_summary(prompt)
                if result and isinstance(result, str) and len(result) > 10:
                    elapsed = round((time.time() - start) * 1000)
                    logger.info(f"Summary handled by {name} ({elapsed}ms)")
                    return result
            except Exception as e:
                logger.warning(f"Summary failed with {name}: {e}")
        return None
