# DataPilot

**AI-powered data analysis.** Upload a dataset, ask questions in natural language, get insights.

DataPilot routes your questions to **81+ analysis skills** spanning statistics, machine learning, NLP, visualization, and more — powered by local LLMs via Ollama.

## Features

- **Natural Language Q&A** — Ask "What predicts churn?" and DataPilot picks the right analysis
- **81 Analysis Skills** — Profiling, correlations, hypothesis tests, classification, regression, clustering, time series, NLP, and more
- **LLM Routing** — Ollama (local/free), Claude, or OpenAI to interpret questions
- **Interactive Chat** — Streaming responses with key points and follow-up suggestions
- **Chart Builder** — AI-suggested visualizations with manual controls
- **Report Export** — PDF, Word, and PowerPoint reports from your analysis history
- **Dark/Light Mode** — Clean, modern interface built with Next.js and Tailwind

## Quick Start

### Option 1: Docker (recommended)

```bash
git clone https://github.com/veritly/datapilot.git
cd datapilot

# Pull the default LLM model
ollama pull llama3.2

# Start all services
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) and upload a CSV.

### Option 2: Manual Setup

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

**Ollama:**
```bash
ollama serve
ollama pull llama3.2
```

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend   │────▶│   FastAPI API     │────▶│  DataPilot      │
│   Next.js    │◀────│   16 endpoints    │◀────│  Engine         │
│   :3000      │     │   + WebSocket     │     │  81+ skills     │
└──────────────┘     │   :8000           │     └────────┬────────┘
                     └────────┬─────────┘              │
                              │                        │
                     ┌────────▼─────────┐     ┌────────▼────────┐
                     │  Session Manager │     │  LLM Provider   │
                     │  In-memory state │     │  Ollama/Claude/ │
                     └──────────────────┘     │  OpenAI         │
                                              └─────────────────┘
```

**Engine modules:**

| Module | Skills | Examples |
|--------|--------|----------|
| `data/` | Profiling, validation, cleaning | `profile_data`, `validate_data`, `curate_dataframe` |
| `analysis/` | Stats, ML, time series | `describe_data`, `classify`, `forecast` |
| `nlp/` | Sentiment, topics, entities | `analyze_sentiment`, `extract_topics` |
| `viz/` | Charts and dashboards | `create_chart`, `auto_chart` |
| `export/` | Reports | PDF, DOCX, PPTX generation |

## Python API

```python
from datapilot import Analyst

analyst = Analyst("sales.csv", llm="ollama")
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
| `POST` | `/api/ask` | Ask a natural-language question |
| `POST` | `/api/analyze` | Run a specific skill directly |
| `POST` | `/api/chart/create` | Create a chart |
| `GET` | `/api/chart/suggest` | AI chart suggestion |
| `POST` | `/api/export/{fmt}` | Generate PDF/DOCX/PPTX report |
| `WS` | `/api/ws/chat` | Streaming chat |

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATAPILOT_LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `claude`, `openai` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for Claude) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for frontend |

## Development

```bash
make dev-backend    # Start FastAPI with hot reload
make dev-frontend   # Start Next.js dev server
make test           # Run tests
make lint           # Run linters
make lint-fix       # Auto-fix lint issues
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Tech Stack

**Engine:** Python 3.10+ · pandas · scikit-learn · XGBoost · LightGBM · statsmodels · SHAP · TextBlob · matplotlib · seaborn · Plotly

**Backend:** FastAPI · Pydantic v2 · uvicorn · WebSockets

**Frontend:** Next.js 16 · TypeScript · Tailwind CSS 4 · lucide-react · next-themes · react-dropzone

**LLM:** Ollama (local) · Anthropic Claude · OpenAI GPT

## License

[MIT](LICENSE)
