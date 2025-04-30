"""
Configuration settings for the Stock Analysis application.
"""
import os

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

# Script paths - Update these to use the correct locations
DOWNLOAD_SCRIPT = os.path.join(ROOT_DIR, "modules", "stock_download.py")
TECHNICAL_SCRIPT = os.path.join(ROOT_DIR, "modules", "fetch_technical_data.py")
TEST_SCRIPT = os.path.join(ROOT_DIR, "modules", "test_model.py")
PREDICT_SCRIPT = os.path.join(ROOT_DIR, "modules", "predict_stocks.py")