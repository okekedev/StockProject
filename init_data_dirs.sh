#!/bin/bash
# Initialize the data directory structure for Sundai Stocks

# Create main data directories
mkdir -p /app/stock_data
mkdir -p /app/stock_data/aiplus_cache
mkdir -p /app/stock_data/aiplus_predictions

# Set permissions
chmod -R 777 /app/stock_data

# Create empty placeholder files in each directory to ensure the directory structure is maintained
touch /app/stock_data/.placeholder
touch /app/stock_data/aiplus_cache/.placeholder
touch /app/stock_data/aiplus_predictions/.placeholder

echo "Data directory structure initialized successfully."