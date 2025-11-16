FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed \
    models/artifacts models/checkpoints \
    experiments/runs experiments/results \
    reports/figures reports/tables \
    notebooks

# Expose ports
EXPOSE 8000 8888

# Default command
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]