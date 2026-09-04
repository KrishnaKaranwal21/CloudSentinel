# SkyGuardAI backend-ml
FROM python:3.12-slim

WORKDIR /srv

# System deps for scikit-learn/shap wheels compiling on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Trained artifacts are committed to the repo (see artifacts/), so the
# image is servable immediately without needing dublin.csv at build time.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Most PaaS platforms (Render, Railway, Fly.io, Cloud Run) inject $PORT.
# Falls back to 8000 for plain `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
