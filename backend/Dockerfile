FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies (all have pre-built wheels)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine and backend
COPY engine/ /app/engine/
COPY backend/ /app/backend/

# Add engine to Python path
ENV PYTHONPATH="/app/engine:${PYTHONPATH}"
ENV DATAPILOT_TEMP_DIR="/tmp/datapilot"

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
