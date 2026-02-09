# Quick Start Guide

Get DataPilot running in under 3 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) (for local LLM)

## 1. Clone and install

```bash
git clone https://github.com/veritly/datapilot.git
cd datapilot

# Python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Frontend
cd frontend && npm install && cd ..
```

## 2. Start Ollama

```bash
ollama serve
ollama pull llama3.2
```

## 3. Start the backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Verify: open [http://localhost:8000/docs](http://localhost:8000/docs)

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

## Using Python directly

```python
from datapilot import Analyst

analyst = Analyst("your_data.csv")
result = analyst.ask("What patterns do you see?")
print(result.text)
```

## Using Claude or OpenAI instead of Ollama

Set the environment variable before starting the backend:

```bash
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

This starts backend (:8000), frontend (:3000), and Ollama (:11434) together.
