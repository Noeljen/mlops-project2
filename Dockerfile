FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /artifacts

RUN pip install uv

# Preinstall the CPU-only PyTorch wheel to avoid CUDA downloads in Codespaces.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1+cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system .
COPY main.py ./main.py
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "main.py"]
CMD ["--run-name", "best_local", "--learning-rate", "3e-5", "--weight-decay", "1e-4", "--train-batch-size", "64", "--warmup-steps", "0", "--checkpoint-dir", "/artifacts"]
