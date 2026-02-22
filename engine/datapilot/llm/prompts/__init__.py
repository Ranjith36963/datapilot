"""
LLM prompts subpackage — modular prompt templates.

Organized by task type:
- base: Core prompts (routing, narratives, charts, skill catalog)
"""

# Import from base module
from .base import (
    _CATALOG_CACHE,
    _SKILL_REGISTRY,
    CHART_PROMPT_TEMPLATE,
    CHART_SYSTEM_PROMPT,
    NARRATION_EXCLUDED_KEYS,
    NARRATIVE_PROMPT_TEMPLATE,
    NARRATIVE_SYSTEM_PROMPT,
    ROUTING_PROMPT_TEMPLATE,
    ROUTING_SYSTEM_PROMPT,
    build_skill_catalog,
    build_skill_registry,
    format_chart_prompt,
    format_narrative_prompt,
    format_routing_prompt,
    get_skill_function,
    get_skill_names,
)

__all__ = [
    # Internal state (used by tests)
    "_CATALOG_CACHE",
    "_SKILL_REGISTRY",
    # Base prompts (routing, narratives, charts)
    "ROUTING_SYSTEM_PROMPT",
    "ROUTING_PROMPT_TEMPLATE",
    "NARRATIVE_SYSTEM_PROMPT",
    "NARRATIVE_PROMPT_TEMPLATE",
    "CHART_SYSTEM_PROMPT",
    "CHART_PROMPT_TEMPLATE",
    "NARRATION_EXCLUDED_KEYS",
    "format_routing_prompt",
    "format_narrative_prompt",
    "format_chart_prompt",
    "build_skill_catalog",
    "build_skill_registry",
    "get_skill_function",
    "get_skill_names",
]
