# LLM Providers & Failover Architecture

## Overview

DataPilot uses a **task-aware multi-LLM architecture**. Two free-tier providers (Groq and Gemini) are assigned to tasks based on their strengths, with automatic failover and deterministic fallbacks.

## Providers

### Groq (Llama 3.3 70B)
- **SDK:** `openai` with `base_url="https://api.groq.com/openai/v1"`
- **Model:** `llama-3.3-70b-versatile`
- **Strengths:** Speed (~300 tok/s), strong reasoning, rich storytelling
- **Weaknesses:** Inconsistent JSON output, higher hallucination risk
- **Free tier:** 14,400 RPD, 30 RPM
- **File:** `engine/datapilot/llm/groq.py`

### Gemini (Flash 2.0)
- **SDK:** `google-genai` (NOT google-generativeai — that's the old package)
- **Model:** `gemini-2.0-flash`
- **Strengths:** Reliable JSON output, lower hallucination, multimodal
- **Weaknesses:** Slower, less creative narratives
- **Free tier:** 1,500 RPD, 15 RPM
- **File:** `engine/datapilot/llm/gemini.py`

### Alternative Providers
- **Ollama** (`llm/ollama.py`) — Local inference, no API key needed
- **Claude** (`llm/claude.py`) — Anthropic Claude
- **OpenAI** (`llm/openai.py`) — GPT models

## Task-Aware Routing

Each LLM task routes to the provider best suited for it:

| Task | Primary | Fallback | Rationale |
|------|---------|----------|-----------|
| **Routing** | Groq | Gemini | Speed matters — user is waiting |
| **Narratives** | Gemini | Groq | Lower hallucination risk — user reads this |
| **Chart suggestions** | Gemini | Groq | Reliable JSON array output |
| **Chart insights** | Groq | Gemini | Short creative text, speed matters |
| **Fingerprinting** | Gemini | Groq | Accurate JSON, runs once per upload |
| **Understanding** (D3) | Gemini | Groq | Structured JSON, quality matters |
| **Plan generation** (D3) | Gemini | Groq | Ordered JSON array, structure matters |
| **Summary** (D3) | Gemini | Groq | Quality synthesis, runs once |

## FailoverProvider

**File:** `engine/datapilot/llm/failover.py`

The `FailoverProvider` wraps all LLM calls with automatic failover:

```python
DEFAULT_TASK_ROUTING = {
    "routing":       ["groq", "gemini"],
    "narrative":     ["gemini", "groq"],
    "chart_suggest": ["gemini", "groq"],
    "chart_insight": ["groq", "gemini"],
    "fingerprint":   ["gemini", "groq"],
}
```

### How failover works

1. Look up the task's provider order (e.g., `narrative → [gemini, groq]`)
2. Try the primary provider
3. If it fails (exception, timeout, None response) → try the next provider
4. If ALL providers fail → return `None`
5. The caller's deterministic fallback handles the `None` case

### Fallback layers

**D1 tasks (routing, narratives, charts):** 3 layers
```
Primary provider → Fallback provider → Deterministic fallback (no LLM)
```

**D3 tasks (understanding, plan, summary):** 2 layers
```
Primary provider → Fallback provider → "AI analysis unavailable"
```

D3 intentionally has no deterministic fallback — hardcoded rules for understanding a dataset would encode biases. The LLM decides or nothing happens.

### Response metadata

Every response includes which provider handled the request:

```python
{
    "provider_used": "gemini",
    "task": "narrative",
    "latency_ms": 340
}
```

## Provider Configuration

### Key availability

The system adapts to available API keys:

| Keys Available | Behavior |
|---|---|
| Both Groq + Gemini | Full task-aware routing |
| Groq only | Groq handles everything |
| Gemini only | Gemini handles everything |
| Neither | Full deterministic fallback |

### Environment Variables

```bash
GROQ_API_KEY=gsk_...          # Groq API key
GEMINI_API_KEY=AIza...        # Google Gemini API key
GEMINI_MODEL=gemini-2.0-flash # Gemini model (optional)
```

## LLMProvider ABC

**File:** `engine/datapilot/llm/provider.py`

All providers implement the `LLMProvider` abstract base class:

```python
class LLMProvider(ABC):
    @abstractmethod
    def route_question(self, question, data_context, skill_catalog) -> RoutingResult | None

    @abstractmethod
    def generate_narrative(self, skill_name, result, context) -> NarrativeResult | None

    @abstractmethod
    def suggest_chart(self, data_context) -> list[dict] | None

    @abstractmethod
    def generate_chart_insight(self, chart_type, data_summary) -> str | None
```

D3 will add: `understand_dataset()`, `generate_plan()`, `generate_summary()`

## Prompt Templates

**Package:** `engine/datapilot/llm/prompts/`

```
prompts/
├── __init__.py              Re-exports all prompts
├── base.py                  Core prompts (routing, narrative, chart, skill catalog)
└── fingerprint_prompts.py   Domain detection prompts
```

Key prompts:
- **ROUTING_SYSTEM_PROMPT** — 6-rule system prompt for skill routing
- **NARRATIVE_SYSTEM_PROMPT** — "Only cite numbers that appear verbatim in the analysis results"
- **CHART_SYSTEM_PROMPT** — Chart suggestion with JSON schema
- **Skill catalog** — Auto-generated from function docstrings at import time

## Hallucination Defense (3 Layers)

1. **`chart_summary()`** in `viz/charts.py` — Structured data summary (actual min, max, mean)
2. **`_sanitize_for_narration()`** in `core/analyst.py` — Strips base64/paths, injects column names
3. **Narration prompt** — "Only cite numbers that appear verbatim in the analysis results"

## Testing

LLM calls are always mocked in tests — never hit real APIs in CI.

```bash
python -m pytest tests/unit/test_failover.py -v         # Failover logic
python -m pytest tests/unit/test_gemini_provider.py -v   # Gemini mocks
```
