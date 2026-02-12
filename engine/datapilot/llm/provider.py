"""
LLM Provider — abstract base class for all LLM integrations.

Defines the interface that Ollama, Claude, and OpenAI providers implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class RoutingResult:
    """Result of routing a user question to a skill."""
    skill_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    route_method: str = "llm"  # "keyword" or "llm"


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
