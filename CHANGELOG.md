# Changelog

## [0.1.0] - 2026-02-08

### Added
- **Engine**: 114 analysis functions across 29 modules (data, stats, ML, NLP, viz, export)
- **LLM Layer**: Abstract provider with Groq (default), Ollama, Claude, and OpenAI implementations
- **Analyst Class**: Natural-language interface routing questions to 81 skills
- **FastAPI Backend**: 16 REST endpoints + WebSocket streaming
- **Next.js Frontend**: Upload, Explore (chat), Visualize (chart builder), Export pages
- **Docker**: Full-stack deployment with docker-compose (backend + frontend)
- **CI/CD**: GitHub Actions test workflow on pull requests
