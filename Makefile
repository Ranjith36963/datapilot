.PHONY: dev test lint build clean install

# Development
dev:
	docker compose up --build

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# Testing
test:
	cd engine && python -m pytest ../tests/ -v --tb=short

test-cov:
	cd engine && python -m pytest ../tests/ -v --cov=datapilot --cov-report=html

# Linting
lint:
	ruff check engine/ backend/
	cd frontend && npm run lint

lint-fix:
	ruff check --fix engine/ backend/

# Type checking
typecheck:
	mypy engine/datapilot/ --ignore-missing-imports

# Build
build:
	docker compose build

build-frontend:
	cd frontend && npm run build

# Install
install:
	pip install -e ".[all,dev]"

install-frontend:
	cd frontend && npm install

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf frontend/.next frontend/out
	rm -rf dist build *.egg-info

