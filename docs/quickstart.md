# Quick Start Guide

Get DataPilot running in under 3 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Groq API key](https://console.groq.com) (free tier available)

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

## 2. Set your Groq API key

```bash
export GROQ_API_KEY=your_key_here
```

Or create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
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

This starts backend (:8000) and frontend (:3000) together. Set `GROQ_API_KEY` in your `.env` file first.
