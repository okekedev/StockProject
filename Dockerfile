FROM python:3.11-slim

WORKDIR /app

# Only non-sensitive configuration at build time
ARG DEBUG=False

# Set non-sensitive environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBUG=$DEBUG

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/stock_data/aiplus_cache /app/stock_data/aiplus_predictions && \
    chmod -R 777 /app/stock_data

# Expose the port
EXPOSE 8050

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8050 || exit 1

# Command to run the application
# All secrets will be provided at runtime via environment variables
CMD ["python", "main.py"]