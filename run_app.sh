#!/bin/bash
# Run the Sundai Stocks application

# Determine if we're using Docker or local
USE_DOCKER=true

# Default username and password
DEFAULT_USERNAME="sundai"
DEFAULT_PASSWORD="Stocks123!"

# Check command-line arguments
for arg in "$@"; do
  case $arg in
    --local)
      USE_DOCKER=false
      shift
      ;;
    --setup-sample-data)
      SETUP_SAMPLE_DATA=true
      shift
      ;;
    --help)
      echo "Sundai Stocks Application Runner"
      echo "--------------------------------"
      echo "Usage: ./run_app.sh [options]"
      echo ""
      echo "Options:"
      echo "  --local             Run locally instead of in Docker"
      echo "  --setup-sample-data Generate sample data before running"
      echo "  --help              Show this help message and exit"
      echo ""
      echo "Default: Uses Docker with sample data"
      exit 0
      ;;
  esac
done

# Ensure directories exist
mkdir -p stock_data
mkdir -p stock_data/aiplus_cache
mkdir -p stock_data/aiplus_predictions

# Setup sample data if requested
if [ "$SETUP_SAMPLE_DATA" = true ]; then
  echo "Setting up sample data..."
  python setup_sample_data.py
  echo "Sample data setup complete."
fi

if [ "$USE_DOCKER" = true ]; then
  echo "Starting Sundai Stocks in Docker container..."
  
  # Check if the container already exists
  if [ "$(docker ps -aq -f name=sundai-stocks)" ]; then
    echo "Stopping and removing existing container..."
    docker stop sundai-stocks
    docker rm sundai-stocks
  fi
  
  # Build the image if needed
  if ! docker image inspect sundai-stocks &> /dev/null; then
    echo "Building Docker image..."
    docker-compose build
  fi
  
  # Start the container
  docker-compose up -d
  
  # Show status
  echo "Sundai Stocks is running at: http://localhost:8050"
  echo "Login with username: $DEFAULT_USERNAME and password: $DEFAULT_PASSWORD"
  echo ""
  echo "To view logs: docker logs -f sundai-stocks"
  echo "To stop: docker-compose down"
else
  echo "Starting Sundai Stocks locally..."
  
  # Check if virtual environment exists, if not create one
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
  else
    source venv/bin/activate
  fi
  
  # Set environment variables
  export DASH_USERNAME="$DEFAULT_USERNAME"
  export DASH_PASSWORD_HASH="scrypt:32768:8:1$ZKCxkyqwu6ECGDP5$0fc9eca9ca1d67df5a514caba195b6842eb36e4a3ba1f7cc5fcbbe1d132200effc50eb067d5e8ffae966dedc2fece75e405cea0d7de0969df20224c5f763ef69"
  export SECRET_KEY="b82d1eb7fd57cde0482f0e1da0916546ba453ee1dd6460e7"
  export GEMINI_API_KEY="AIzaSyCrr6OzYwYvuiorPvmAAkYwb0lHQI8U7Wo"
  export NEWS_API_KEY="9b73205028734f2181dcda4f1b892d66"
  
  # Run the application
  python main.py
fi