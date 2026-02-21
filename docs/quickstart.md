# Quick Start Guide

Get DataPilot running in under 3 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Groq API key](https://console.groq.com) (free tier available)
- [Gemini API key](https://aistudio.google.com/apikey) (optional, free tier available)

## 1. Clone and install

```bash
git clone https://github.com/Ranjith36963/datapilot.git
cd datapilot

# Python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Frontend
cd frontend && npm install && cd ..
```

## 2. Set your API keys

```bash
export GROQ_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here  # optional but recommended
```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

**LLM behavior by key availability:**
- Both keys: Full task-aware routing (Groq for speed tasks, Gemini for accuracy tasks)
- Groq only: Groq handles all LLM tasks
- Gemini only: Gemini handles all LLM tasks
- Neither: Full deterministic fallback (no LLM, still functional)

## 3. Start the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Verify: open [http://localhost:8000/docs](http://localhost:8000/docs)

On startup, the backend:
- Initializes SQLite session store (`backend/data/sessions.db`)
- Cleans up expired sessions (>24h)
- Recovers any existing sessions from the database

## 4. Start the frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## 5. Use it

1. **Upload** a CSV or Excel file on the home page
2. **Explore** — click "Start Exploring" to open the chat interface
3. **Ask questions** like:
   - "Give me an overview of the data"
   - "What are the strongest correlations?"
   - "Are there any outliers?"
   - "Which features predict [target column]?"
4. **Visualize** — switch to the chart builder tab
5. **Export** — generate a PDF/Word/PowerPoint report

Sessions persist across server restarts — your data and analysis history are saved in SQLite.

## Using Python directly

```python
from datapilot import Analyst

analyst = Analyst("your_data.csv")
result = analyst.ask("What patterns do you see?")
print(result.text)
```

## Using other LLM providers

Set the environment variable before starting the backend:

```bash
# Ollama (local, no API key needed)
export DATAPILOT_LLM_PROVIDER=ollama
# Make sure Ollama is running: ollama serve && ollama pull llama3.2

# Claude
export DATAPILOT_LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
export DATAPILOT_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

## Docker alternative

```bash
docker compose up --build
```

This starts backend (:8000) and frontend (:3000) together. Set API keys in your `.env` file first.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | — | Groq API key for Llama 3.3 70B |
| `GEMINI_API_KEY` | No | — | Google Gemini API key for Flash 2.0 |
| `DATAPILOT_DB_PATH` | No | `backend/data/sessions.db` | Custom SQLite database path |
| `DATAPILOT_LLM_PROVIDER` | No | Auto-detect | Force specific LLM provider |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server address |

\* At least one LLM API key recommended. System works without any keys using deterministic fallbacks.
