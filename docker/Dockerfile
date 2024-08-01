FROM python:3.11.8-slim-bookworm AS builder

ARG CONFIG

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local' \
  POETRY_VERSION=1.8.1

# System deps:
RUN apt update && apt install -y --no-install-recommends --fix-missing python3-opencv && \
    python3 -m pip install poetry

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY image-acquisition/pyproject.toml image-acquisition/poetry.lock* ./

# Create requirements.txt
RUN poetry export --only=main --output=requirements.txt

# Project initialization:
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runner
FROM python:3.11.8-slim-bookworm AS runner

WORKDIR /app

ENV VDTL_KAFKA_QUEUE_SIZE=6500000 \
  VDTL_MAN_TOPIC="add.stream" \
  VDTL_POLL_TIME=0.1 \
  VDTL_QUEUE_TIMEOUT=0.1 \
  VDTL_UPDATE_RATE=0.04 \
  VDTL_ADD_CONSUMER_SLEEP=0.1 \
  VDTL_POLL_TIME=0.1 \
  VDTL_API_URL="back-end:8080" \
  VDTL_KAFKA_URL="kafka:9092" \ 
  VDTL_IS_KILLABLE=1 \
  VDTL_STREAM_WIDTH=480 \
  VDTL_STREAM_HEIGHT=480

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY --from=builder /app/wheels wheels
RUN python -m pip install --no-cache /app/wheels/*

# Creating folders, and files for a project:
COPY image-acquisition/app .

CMD ["python3", "main.py"]