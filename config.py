"""
Configuration settings for the Stock Analysis application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# File paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "stock_data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
INPUT_FILE = os.path.join(DATA_DIR, "nasdaq_screener.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "stock_symbols.csv")
TECH_DATA_FILE = os.path.join(DATA_DIR, "stock_data_technical.csv")
PREDICTION_OUTPUT = os.path.join(DATA_DIR, "top_10_upward_picks.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "test_results.csv")
TEST_STOCK_ACCURACY = os.path.join(DATA_DIR, "test_stock_accuracy.csv")

# Script paths
DOWNLOAD_SCRIPT = os.path.join(ROOT_DIR, "modules", "stock_download.py")
TECHNICAL_SCRIPT = os.path.join(ROOT_DIR, "modules", "fetch_technical_data.py")
TEST_SCRIPT = os.path.join(ROOT_DIR, "modules", "test_model.py")
PREDICT_SCRIPT = os.path.join(ROOT_DIR, "modules", "predict_stocks.py")

# Get API keys from environment variables
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Print warning if API keys are not set
if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY environment variable is not set.")
    
    
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set.")