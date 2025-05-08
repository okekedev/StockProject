#!/bin/bash
# Start the Sundai Stocks Docker container

# Set default environment variables
DASH_USERNAME="${DASH_USERNAME:-sundai}"
DASH_PASSWORD_HASH="${DASH_PASSWORD_HASH:-scrypt:32768:8:1$ZKCxkyqwu6ECGDP5$0fc9eca9ca1d67df5a514caba195b6842eb36e4a3ba1f7cc5fcbbe1d132200effc50eb067d5e8ffae966dedc2fece75e405cea0d7de0969df20224c5f763ef69}"
SECRET_KEY="${SECRET_KEY:-b82d1eb7fd57cde0482f0e1da0916546ba453ee1dd6460e7}"
GEMINI_API_KEY="${GEMINI_API_KEY:-AIzaSyCrr6OzYwYvuiorPvmAAkYwb0lHQI8U7Wo}"
NEWS_API_KEY="${NEWS_API_KEY:-9b73205028734f2181dcda4f1b892d66}"

# Create host directories if they don't exist
mkdir -p ./stock_data
mkdir -p ./stock_data/aiplus_cache
mkdir -p ./stock_data/aiplus_predictions

# Start the container
docker run -d \
  --name sundai-stocks \
  -p 8050:8050 \
  -v "$(pwd)/stock_data:/app/stock_data" \
  -e DASH_USERNAME="$DASH_USERNAME" \
  -e DASH_PASSWORD_HASH="$DASH_PASSWORD_HASH" \
  -e SECRET_KEY="$SECRET_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e NEWS_API_KEY="$NEWS_API_KEY" \
  sundai-stocks

echo "Sundai Stocks container is now running on http://localhost:8050"
echo "Login with username: $DASH_USERNAME and password: (as configured)"