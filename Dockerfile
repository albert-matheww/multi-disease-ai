FROM python:3.11-slim

# System deps needed by matplotlib/reportlab/pyarrow wheels at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Hugging Face Spaces executes the container as uid 1000
# ("user"); matching that here means the same image runs identically on a Space,
# under `docker compose`, or on any other container host.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TABPFN_MODEL_CACHE_DIR=/home/user/app/.tabpfn_models

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Directories the pipeline/app write to at runtime (processed data, model
# bundles, generated PDFs, the prediction-history SQLite DB, TabPFN's weight
# cache). Created up front so the app starts even before the pipeline is run.
RUN mkdir -p data/raw data/processed models reports/figures reports/generated .tabpfn_models

# Optional: bake TabPFN's gated pretrained weights into the image so the running
# container needs neither network nor a token. Pass `--build-arg TABPFN_TOKEN=...`
# (or a BuildKit secret) at build time; the step is skipped and never fails when
# the arg is absent.
ARG TABPFN_TOKEN=""
RUN if [ -n "$TABPFN_TOKEN" ]; then \
        TABPFN_TOKEN="$TABPFN_TOKEN" python -c "import numpy as np; from tabpfn import TabPFNClassifier; \
TabPFNClassifier(n_estimators=1).fit(np.random.rand(24, 3), np.random.randint(0, 2, 24)); \
print('TabPFN weights cached into image')" || echo 'TabPFN weight prefetch skipped'; \
    fi

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
