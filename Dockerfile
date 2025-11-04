FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system build dependencies
RUN apt-get update && apt-get install -y \
  git \
  build-essential \
  portaudio19-dev \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user for running the application
RUN useradd -m appuser

# Set up virtual environment and install Poetry
ENV VENV_PATH=/app/.venv
RUN python3 -m venv $VENV_PATH \
  && $VENV_PATH/bin/pip install -U pip setuptools \
  && $VENV_PATH/bin/pip install poetry

# Set additional Python environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

# Set working directory and copy configuration files with proper ownership
WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml poetry.toml README.md ./

# Copy source files with proper permissions
COPY --chown=appuser:appuser .env main.py inbound_main.py inbound_voice_main.py outbound_live_main.py outbound_main.py dev-antonym-442118-a2-d5ac21d57c4d.json empty_log_config.json ./

# Install dependencies using Poetry (ensuring container’s system environment is used)
RUN $VENV_PATH/bin/poetry install --no-interaction --no-ansi

# Update PATH to use the virtual environment
ENV PATH="/app/.venv/bin:${PATH}"

# Expose required port for the application
EXPOSE 5503

# Switch to non-root user
USER appuser

# Start with Bash (or change 