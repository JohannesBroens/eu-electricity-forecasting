FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
COPY .env.example .env.example

# Install dependencies
RUN uv sync --no-dev --frozen

# Create data and output directories
RUN mkdir -p data/raw data/processed data/backtest_cache output

# Default: run the pipeline (validate only, no backtest)
CMD ["uv", "run", "python", "scripts/run_pipeline.py", "--validate"]
