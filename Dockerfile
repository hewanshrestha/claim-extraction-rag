# --- Stage 1: The Builder ---
FROM python:3.13-slim AS builder

# Install uv for the build phase
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies into the project folder
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# --- Stage 2: The Runtime ---
FROM python:3.13-slim

# IMPORTANT: We must also have uv in the runtime stage to use 'uv run'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the environment and the code
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Ensure Python output is sent straight to terminal logs
ENV PYTHONUNBUFFERED=1

# Standard ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501
