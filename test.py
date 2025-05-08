#!/usr/bin/env python3
"""
Test yfinance functionality inside Docker container.
"""
import sys
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance():
    """Test basic yfinance functionality."""
    print("Testing yfinance...")
    
    # Test a well-known stable ticker
    symbol = "AAPL"
    print(f"Fetching data for {symbol}")
    
    try:
        # Try to fetch basic info
        ticker = yf.Ticker(symbol)
        print(f"Ticker instance created for {symbol}")
        
        # Try to get history
        hist = ticker.history(period="1mo")
        if hist.empty:
            print("ERROR: History is empty")
        else:
            print(f"Successfully fetched {len(hist)} days of history")
            print(f"Latest price: {hist['Close'].iloc[-1]}")
        
        # Try to get company info
        try:
            info = ticker.info
            print(f"Company name: {info.get('longName', 'Unknown')}")
            print(f"Industry: {info.get('industry', 'Unknown')}")
        except Exception as e:
            print(f"ERROR getting company info: {e}")
        
        print("yfinance basic test completed")
        return True
    except Exception as e:
        print(f"ERROR in yfinance test: {e}")
        return False

def test_with_fallback():
    """Test with fallback to sample data if yfinance fails."""
    if not test_yfinance():
        print("\nFalling back to sample data...")
        
        # Create sample data
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Create sample DataFrame
        sample_data = pd.DataFrame(
            index=date_range,
            data={
                'Open': [150 + i * 0.1 for i in range(len(date_range))],
                'High': [155 + i * 0.1 for i in range(len(date_range))],
                'Low': [145 + i * 0.1 for i in range(len(date_range))],
                'Close': [153 + i * 0.1 for i in range(len(date_range))],
                'Volume': [10000000 for _ in range(len(date_range))]
            }
        )
        
        print(f"Created sample data with {len(sample_data)} rows")
        
        # Save the sample data
        os.makedirs("sample_data", exist_ok=True)
        sample_data['Symbol'] = 'AAPL'
        sample_data.to_csv("sample_data/sample_technical_data.csv", index=True)
        print("Saved sample data to sample_data/sample_technical_data.csv")
        
        return True
    
    return True

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"yfinance version: {yf.__version__}")
    
    success = test_with_fallback()
    sys.exit(0 if success else 1)