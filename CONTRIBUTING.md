# Contributing to DataPilot

Thank you for your interest in contributing to DataPilot!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Ranjith36963/datapilot.git
   cd datapilot
   ```

2. Install Python dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[all,dev]"
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend && npm install
   ```

4. Start development servers:
   ```bash
   make dev-backend   # Terminal 1: FastAPI at :8000
   make dev-frontend   # Terminal 2: Next.js at :3000
   ```

## Running Tests

```bash
make test         # Run all tests
make test-cov     # With coverage report
```

## Code Style

- **Python**: We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- **TypeScript**: ESLint with Next.js config
- Run `make lint` before submitting a PR

## Adding a New Analysis Skill

1. Add your function to the appropriate module in `engine/datapilot/`
2. Follow the contract: return `{"status": "success|error", ...}`
3. Add the function name to `engine/datapilot/__init__.py` exports
4. Write tests in `tests/unit/`
5. The skill will be auto-registered in the LLM routing catalog

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass (`make test`)
4. Ensure linting passes (`make lint`)
5. Submit a PR with a clear description of changes

## Reporting Issues

Use [GitHub Issues](https://github.com/Ranjith36963/datapilot/issues) to report bugs or request features. Include:

- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python/Node version and OS
