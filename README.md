# DataPilot

**AI-powered data analysis.** Upload a dataset, ask questions in natural language, get insights.

DataPilot routes your questions to **34 analysis skills** spanning statistics, machine learning, NLP, visualization, and more — powered by multi-LLM failover (Groq + Gemini).

## Screenshots

| Upload | Data Preview |
|--------|-------------|
| ![Upload](docs/screenshots/upload.png) | ![Preview](docs/screenshots/preview.png) |

| Explore & Chat | Auto-Pilot |
|---------------|------------|
| ![Explore](docs/screenshots/explore.png) | ![Autopilot](docs/screenshots/autopilot.png) |

| Visualize | Export |
|-----------|--------|
| ![Visualize](docs/screenshots/visualize.png) | ![Export](docs/screenshots/export.png) |

## Features

- **Natural Language Q&A** — Ask "What predicts churn?" and DataPilot picks the right analysis
- **34 Analysis Skills** — Profiling, correlations, hypothesis tests, classification, regression, clustering, time series, NLP, and more
- **Semantic-First Routing** — Keyword overrides → Semantic embeddings → Smart query (LLM) → Profile fallback
- **Multi-LLM Support** — Groq + Gemini (task-aware failover), Ollama, Claude, or OpenAI
- **Interactive Chat** — Streaming responses with key points and follow-up suggestions
- **AI Dataset Understanding** — LLM classifies domain, identifies target columns, suggests questions
- **Auto-Pilot** — LLM generates and executes a full analysis plan automatically
- **Chart Builder** — 13 chart types with AI-suggested visualizations and manual controls
- **Report Export** — PDF, Word, and PowerPoint reports from your analysis history
- **Dark/Light Mode** — Clean, modern interface built with Next.js and Tailwind

## Quick Start

**Backend:**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
cd backend && uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**API keys:**
```bash
export GROQ_API_KEY=your_key_here    # Free at https://console.groq.com
export GEMINI_API_KEY=your_key_here  # Free at https://aistudio.google.com
```

Open [http://localhost:3000](http://localhost:3000) and upload a CSV.

> **Note:** DataPilot works without API keys — keyword routing, semantic matching, and all 34 analysis skills run locally. API keys add LLM-powered narratives, smart_query routing, dataset understanding, and auto-pilot.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend   │────▶│   FastAPI API     │────▶│  DataPilot      │
│   Next.js    │◀────│   17 endpoints    │◀────│  Engine         │
│   :3000      │     │   + WebSocket     │     │  34 skills      │
└──────────────┘     │   :8000           │     └────────┬────────┘
                     └────────┬─────────┘              │
                              │                   ┌────▼────────────┐
                     ┌────────▼─────────┐         │  4-Tier Router  │
                     │  Session Manager │         │  Keywords →     │
                     │  SQLite + Cache  │         │  Semantic →     │
                     └──────────────────┘         │  LLM → Profile  │
                                                  └────┬────────────┘
                                                       │
                                                  ┌────▼────────────┐
                                                  │  LLM Failover   │
                                                  │  Groq + Gemini  │
                                                  └─────────────────┘
```

**Engine modules:**

| Module | Skills | Examples |
|--------|--------|----------|
| `data/` | Profiling, validation, cleaning | `profile_data`, `validate_data`, `curate_dataframe` |
| `analysis/` | Stats, ML, time series, queries | `describe_data`, `classify`, `forecast`, `smart_query` |
| `nlp/` | Sentiment, topics, entities | `analyze_sentiment`, `extract_topics` |
| `viz/` | 13 chart types + dashboards | `create_chart`, `auto_chart` |
| `export/` | Reports | PDF, DOCX, PPTX generation |

## Python API

```python
from datapilot import Analyst

analyst = Analyst("sales.csv", llm="groq")
result = analyst.ask("Which features predict revenue?")
print(result.text)           # narrative summary
print(result.key_points)     # bullet points
print(result.suggestions)    # follow-up questions

# Direct skill access
analyst.profile()
analyst.correlations(target="revenue")
analyst.classify(target="churn")
analyst.chart(chart_type="scatter", x="price", y="sales")
analyst.export("report.pdf")
```

## REST API

All endpoints require an `x-session-id` header (returned by upload).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a dataset (CSV, Excel, JSON, Parquet) |
| `GET` | `/api/preview` | Preview rows |
| `GET` | `/api/profile` | Full dataset profile |
| `POST` | `/api/fingerprint/{id}` | AI domain understanding |
| `GET` | `/api/autopilot/{id}` | Auto-pilot analysis plan |
| `POST` | `/api/ask` | Ask a natural-language question |
| `POST` | `/api/analyze` | Run a specific skill directly |
| `GET` | `/api/history` | Analysis history |
| `POST` | `/api/chart/create` | Create a chart |
| `GET` | `/api/chart/suggest` | AI chart suggestions |
| `POST` | `/api/export/{fmt}` | Generate PDF/DOCX/PPTX report |
| `GET` | `/api/export/download/{file}` | Download generated report |
| `GET` | `/api/narrative` | Get LLM narrative for results |
| `GET` | `/api/sessions` | List all sessions |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/health` | Health check |
| `WS` | `/api/ws/chat` | Streaming chat |

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (free at [console.groq.com](https://console.groq.com)) |
| `GEMINI_API_KEY` | — | Gemini API key (free at [aistudio.google.com](https://aistudio.google.com)) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `DATAPILOT_LLM_PROVIDER` | `groq` | LLM provider: `groq`, `ollama`, `claude`, `openai` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL (if using Ollama) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name (if using Ollama) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for frontend |

## Tech Stack

**Engine:** Python 3.13 · pandas · scikit-learn · XGBoost · LightGBM · statsmodels · SHAP · sentence-transformers · matplotlib · seaborn · Plotly

**Backend:** FastAPI · Pydantic v2 · uvicorn · WebSockets · SQLite (WAL)

**Frontend:** Next.js 16 · React 19 · TypeScript · Tailwind CSS 4 · lucide-react · next-themes · react-dropzone

**LLM:** Groq (Llama 3.3 70B) + Gemini (Flash 2.0) task-aware failover · Ollama · Anthropic Claude · OpenAI GPT

## License

[MIT](LICENSE)
