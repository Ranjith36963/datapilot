"""
LLM integration layer — provider ABC and implementations.
"""

from .provider import LLMProvider, RoutingResult, NarrativeResult
from .ollama import OllamaProvider
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .groq import GroqProvider
from .gemini import GeminiProvider
from .failover import FailoverProvider
from .prompts import build_skill_catalog, get_skill_function, get_skill_names

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
