"""
LLM integration layer — provider ABC and implementations.
"""

from .claude import ClaudeProvider
from .failover import FailoverProvider
from .gemini import GeminiProvider
from .groq import GroqProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .prompts import build_skill_catalog, get_skill_function, get_skill_names
from .provider import LLMProvider, NarrativeResult, RoutingResult

__all__ = [
    "LLMProvider",
    "RoutingResult",
    "NarrativeResult",
    "OllamaProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "GroqProvider",
    "GeminiProvider",
    "FailoverProvider",
    "build_skill_catalog",
    "get_skill_function",
    "get_skill_names",
]
