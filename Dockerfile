FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application ──────────────────────────────────────────────────────────
COPY . .

# ── Create __init__ files so Python packages resolve ─────────────────────────
RUN touch env/__init__.py grader/__init__.py tasks/__init__.py server/__init__.py

# ── HuggingFace Spaces expects port 7860 ─────────────────────────────────────
ENV PORT=7860
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
