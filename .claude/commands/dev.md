Start the DataPilot development environment.

Execute: `docker compose up --build`

This starts:
- Backend API on http://localhost:8000
- Frontend on http://localhost:3000

Requires `GROQ_API_KEY` environment variable (set in `.env` or export manually).

If Docker is not available, start services individually:
1. `cd backend && uvicorn app.main:app --reload --port 8000`
2. `cd frontend && npm run dev`
