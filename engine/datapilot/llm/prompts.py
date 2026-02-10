"""
LLM Prompts — system prompts, skill catalog, and templates.

The skill catalog is auto-generated from function docstrings at import time.
"""

import inspect
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("datapilot.llm.prompts")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Keys that carry no analytical value for narration (base64 blobs, server paths).
# Used by providers (groq.py, etc.) and analyst.py to strip before LLM calls.
NARRATION_EXCLUDED_KEYS = frozenset({
    "chart_base64", "image_base64", "chart_path", "chart_html_path",
})

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ROUTING_SYSTEM_PROMPT = (
    "You are a data analysis routing assistant for DataPilot. "
    "Given a user question, dataset context, and available analysis skills, "
    "pick the single best skill to answer the question. "
    "Respond ONLY with valid JSON: "
    '{"skill": "<skill_name>", "params": {<params>}, '
    '"confidence": <0.0-1.0>, "reasoning": "<one_sentence>"}'
)

NARRATIVE_SYSTEM_PROMPT = (
    "You are a data analyst writing clear, actionable insights "
    "for a non-technical business audience. Be concise. "
    "Respond ONLY with valid JSON: "
    '{"text": "<narrative>", "key_points": ["..."], "suggestions": ["..."]}'
)

CHART_SYSTEM_PROMPT = (
    "You are a data visualization expert. "
    "Suggest the best chart type for exploring the given data. "
    "Respond ONLY with valid JSON: "
    '{"chart_type": "<type>", "x": "<column_or_null>", '
    '"y": "<column_or_null>", "hue": "<column_or_null>", "title": "<title>"}'
)

# ---------------------------------------------------------------------------
# Routing prompt template
# ---------------------------------------------------------------------------

ROUTING_PROMPT_TEMPLATE = """\
Dataset columns: {columns_info}
Dataset shape: {shape}

Available skills:
{skill_catalog}

User question: {question}

Pick the best skill and fill in the parameters. Respond with JSON only."""

# ---------------------------------------------------------------------------
# Narrative prompt template
# ---------------------------------------------------------------------------

NARRATIVE_PROMPT_TEMPLATE = """\
Analysis results:
{result_json}

{question_line}Write a concise narrative summary. Include:
1. A 1-2 sentence headline summary
2. 3-5 key points
3. 2-3 suggested next steps

Respond with JSON only."""

# ---------------------------------------------------------------------------
# Chart suggestion prompt template
# ---------------------------------------------------------------------------

CHART_PROMPT_TEMPLATE = """\
Dataset columns: {columns_info}

{analysis_context}Suggest the best chart for exploring this data. Respond with JSON only."""

# ---------------------------------------------------------------------------
# Skill catalog — auto-generated from engine function docstrings
# ---------------------------------------------------------------------------

# Registry: list of (skill_name, callable, description, param_hints)
_SKILL_REGISTRY: List[Tuple[str, object, str, Dict]] = []
_CATALOG_CACHE: Optional[str] = None


def _extract_first_line(docstring: Optional[str]) -> str:
    """Extract the first non-empty line from a docstring."""
    if not docstring:
        return "No description available."
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.rstrip(".")
    return "No description available."


def _extract_params(func) -> Dict[str, str]:
    """Extract parameter names and their annotations from a function."""
    params = {}
    try:
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if name in ("df", "data", "self", "cls"):
                continue
            annotation = ""
            if param.annotation != inspect.Parameter.empty:
                annotation = getattr(param.annotation, "__name__", str(param.annotation))
            default = ""
            if param.default != inspect.Parameter.empty:
                default = f" = {param.default!r}"
            params[name] = f"{annotation}{default}".strip()
    except (ValueError, TypeError):
        pass
    return params


def build_skill_registry() -> List[Tuple[str, object, str, Dict]]:
    """Build the skill registry from datapilot's exported functions.

    Returns list of (name, callable, description, param_hints).
    """
    global _SKILL_REGISTRY

    if _SKILL_REGISTRY:
        return _SKILL_REGISTRY

    try:
        import datapilot
    except ImportError:
        logger.warning("Cannot import datapilot — skill catalog will be empty")
        return []

    all_names = getattr(datapilot, "__all__", [])
    registry = []

    for name in all_names:
        # Skip upload variants and utility functions
        if name.endswith("_and_upload") or name in (
            "load_data", "save_data", "upload_result", "safe_json_serialize",
            "format_for_narrative", "create_executive_summary_data",
            "create_detailed_findings_data",
            # Low-level helpers that return raw DataFrames, not skill dicts
            "correlation_matrix",
        ):
            continue

        func = getattr(datapilot, name, None)
        if func is None or not inspect.isfunction(func):
            continue

        desc = _extract_first_line(func.__doc__)
        params = _extract_params(func)
        registry.append((name, func, desc, params))

    _SKILL_REGISTRY = registry
    return registry


def build_skill_catalog() -> str:
    """Build a human-readable skill catalog string for LLM prompts.

    Format:
        - skill_name(param1, param2): Description
    """
    global _CATALOG_CACHE

    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    registry = build_skill_registry()
    if not registry:
        return "(No skills available)"

    lines = []
    for name, _func, desc, params in registry:
        param_list = ", ".join(
            f"{p}: {t}" if t else p for p, t in params.items()
        )
        lines.append(f"- {name}({param_list}): {desc}")

    _CATALOG_CACHE = "\n".join(lines)
    return _CATALOG_CACHE


def get_skill_function(skill_name: str):
    """Look up a skill function by name. Returns None if not found."""
    registry = build_skill_registry()
    for name, func, _desc, _params in registry:
        if name == skill_name:
            return func
    return None


def get_skill_names() -> List[str]:
    """Return list of all available skill names."""
    return [name for name, *_ in build_skill_registry()]


def format_routing_prompt(
    question: str,
    columns_info: str,
    shape: str,
    skill_catalog: Optional[str] = None,
) -> str:
    """Format a routing prompt with dataset context."""
    catalog = skill_catalog or build_skill_catalog()
    return ROUTING_PROMPT_TEMPLATE.format(
        columns_info=columns_info,
        shape=shape,
        skill_catalog=catalog,
        question=question,
    )


def format_narrative_prompt(
    result_json: str,
    question: Optional[str] = None,
) -> str:
    """Format a narrative generation prompt."""
    question_line = f"Original question: {question}\n\n" if question else ""
    return NARRATIVE_PROMPT_TEMPLATE.format(
        result_json=result_json,
        question_line=question_line,
    )


def format_chart_prompt(
    columns_info: str,
    analysis_result: Optional[str] = None,
) -> str:
    """Format a chart suggestion prompt."""
    context = ""
    if analysis_result:
        context = f"Recent analysis:\n{analysis_result}\n\n"
    return CHART_PROMPT_TEMPLATE.format(
        columns_info=columns_info,
        analysis_context=context,
    )
