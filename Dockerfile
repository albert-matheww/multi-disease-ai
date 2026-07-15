FROM python:3.11-slim

WORKDIR /app

# System deps needed by matplotlib/reportlab/pyarrow wheels at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data, models, and reports are generated at runtime (see docker-compose.yml
# for a typical first-run command); create the directories so the app can
# start even before the pipeline has been run once.
RUN mkdir -p data/raw data/processed models reports/figures reports/generated

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
