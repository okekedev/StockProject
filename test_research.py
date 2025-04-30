"""
Test script for the Research Department using News API and Gemini.

This script tests the stock_news module by fetching news and analyzing
sentiment for sample stocks.
"""
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the Gemini API key is set
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = "AIzaSyCrr6OzYwYvuiorPvmAAkYwb0lHQI8U7Wo"
    print("Setting GEMINI_API_KEY directly for testing")

# Ensure the News API key is set
os.environ['NEWS_API_KEY'] = "9b73205028734f2181dcda4f1b892d66"
print("News API Key set for testing")

try:
    # Import the stock news module
    from modules.stock_news import get_stock_analysis
except ImportError:
    print("Error: Could not import the stock_news module.")
    print("Make sure you've created modules/stock_news.py")
    exit(1)

def test_stock_news():
    """Test the stock news module with sample stocks."""
    print("\n" + "="*50)
    print("RESEARCH DEPARTMENT TEST")
    print("="*50)
    
    # Sample stocks to test
    test_stocks = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"Testing with samples: {', '.join(test_stocks)}")
    print("\nRunning tests...")
    
    results = {}
    
    # Test each stock
    for i, symbol in enumerate(test_stocks):
        print(f"\nFetching data for {symbol} ({i+1}/{len(test_stocks)})...")
        
        start_time = time.time()
        result = get_stock_analysis(symbol)
        end_time = time.time()
        
        # Add performance metrics
        result['fetch_time_seconds'] = round(end_time - start_time, 2)
        results[symbol] = result
        
        # Print a summary
        news_count = len(result.get('news_items', []))
        sentiment = result.get('sentiment', 'unknown').upper()
        
        print(f"  - Found {news_count} news items")
        print(f"  - Sentiment: {sentiment}")
        print(f"  - Analysis took {result['fetch_time_seconds']}s")
        
        # Respect API rate limits
        if i < len(test_stocks) - 1:
            print("  - Waiting 2 seconds to respect rate limits...")
            time.sleep(2)
    
    # Save results to a file for inspection
    output_file = "research_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "-"*50)
    print(f"Test complete! Results saved to {output_file}")
    print("-"*50)
    
    # Print a summary report
    print("\nSUMMARY REPORT:")
    print("-"*50)
    for symbol, result in results.items():
        news_count = len(result.get('news_items', []))
        sentiment = result.get('sentiment', 'unknown').upper()
        fetch_time = result.get('fetch_time_seconds', 0)
        
        print(f"{symbol}: {news_count} news items, {sentiment} sentiment, {fetch_time}s")
        
        # Print first news headline if available
        if news_count > 0:
            headline = result['news_items'][0].get('headline', 'No headline')
            print(f"  Latest headline: {headline[:60]}{'...' if len(headline) > 60 else ''}")
    
    print("-"*50)
    print(f"Test ran on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

if __name__ == "__main__":
    test_stock_news()