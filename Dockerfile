FROM python:3.11-slim

WORKDIR /app

# Allow overriding environment variables during build
ARG DASH_USERNAME=sundai
ARG DASH_PASSWORD_HASH=scrypt:32768:8:1$ZKCxkyqwu6ECGDP5$0fc9eca9ca1d67df5a514caba195b6842eb36e4a3ba1f7cc5fcbbe1d132200effc50eb067d5e8ffae966dedc2fece75e405cea0d7de0969df20224c5f763ef69
ARG SECRET_KEY=b82d1eb7fd57cde0482f0e1da0916546ba453ee1dd6460e7
ARG GEMINI_API_KEY=AIzaSyCrr6OzYwYvuiorPvmAAkYwb0lHQI8U7Wo
ARG NEWS_API_KEY=9b73205028734f2181dcda4f1b892d66
# Set DEBUG explicitly to False for production
ARG DEBUG=False

# Set environment variables - these can be overridden at runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DASH_USERNAME=$DASH_USERNAME \
    DASH_PASSWORD_HASH=$DASH_PASSWORD_HASH \
    SECRET_KEY=$SECRET_KEY \
    GEMINI_API_KEY=$GEMINI_API_KEY \
    NEWS_API_KEY=$NEWS_API_KEY \
    # Force debug mode off for stability in production
    DEBUG=False

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

# Check if main.py handles debug mode through environment variables
# If not, modify the command to explicitly set debug=False
CMD ["python", "main.py"]