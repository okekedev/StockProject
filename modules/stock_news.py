"""
Stock news and sentiment analysis module using yfinance and Gemini API.
"""
import os
import json
import yfinance as yf
from datetime import datetime, timedelta
import google.generativeai as genai
import pandas as pd

# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with the API key."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables!")
        return False
        
    genai.configure(api_key=api_key)
    return True

def get_stock_news(symbol, days_lookback=7):
    """
    Get recent news for a stock using yfinance.
    
    Args:
        symbol (str): Stock symbol
        days_lookback (int): Number of days to look back for news
        
    Returns:
        list: List of news items with date, headline, and link
    """
    try:
        # Create a Ticker object
        ticker = yf.Ticker(symbol)
        
        # Get news items
        news_items = []
        
        # Get news data from yfinance
        news = ticker.news
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_lookback)
        
        # Process news items
        if news:
            for item in news:
                # Convert timestamp to datetime
                news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                # Only include news from the specified lookback period
                if news_date >= cutoff_date:
                    news_items.append({
                        'date': news_date.strftime('%Y-%m-%d'),
                        'headline': item.get('title', 'No headline available'),
                        'summary': item.get('summary', 'No summary available')[:100] + '...',
                        'url': item.get('link', '#')
                    })
        
        # Get company information for context
        info = ticker.info
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Get recent stock price movement
        hist = ticker.history(period='1mo')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            month_ago_price = hist['Close'].iloc[0]
            price_change_pct = ((current_price - month_ago_price) / month_ago_price) * 100
        else:
            price_change_pct = 0
            
        context = {
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'price_change_pct': round(price_change_pct, 2),
            'current_price': round(current_price, 2) if 'current_price' in locals() else 'Unknown'
        }
        
        return news_items, context
    
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return [], {'company_name': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}

def analyze_sentiment(news_items, context, symbol):
    """
    Analyze sentiment of news items using Gemini API.
    
    Args:
        news_items (list): List of news items
        context (dict): Stock context information
        symbol (str): Stock symbol
        
    Returns:
        dict: Sentiment analysis results
    """
    if not configure_gemini():
        return {
            'symbol': symbol,
            'sentiment': 'unknown',
            'sentiment_reasoning': 'Gemini API not available',
            'impact_summary': 'Unable to analyze sentiment due to API configuration issues.'
        }
    
    # If no news items found
    if not news_items:
        return {
            'symbol': symbol,
            'sentiment': 'neutral',
            'sentiment_reasoning': 'No recent news found for analysis',
            'impact_summary': 'Without recent news, sentiment analysis is inconclusive. Consider technical indicators instead.'
        }
    
    try:
        # Format news for the prompt
        news_text = "\n".join([
            f"Date: {item['date']}, Headline: {item['headline']}, Summary: {item['summary']}"
            for item in news_items[:5]  # Limit to 5 news items to keep prompt size reasonable
        ])
        
        # Create a prompt for Gemini
        prompt = f"""
        Analyze the sentiment of these recent news items about {context['company_name']} ({symbol}):
        
        Company Information:
        - Sector: {context['sector']}
        - Industry: {context['industry']}
        - Recent price change: {context['price_change_pct']}% in the last month
        
        Recent News:
        {news_text}
        
        Please provide the following in JSON format:
        1. Overall sentiment (positive, negative, or neutral)
        2. Reasoning behind the sentiment assessment
        3. A summary of how these news items might impact the stock's near-term performance
        
        Format as:
        {{
            "sentiment": "positive/negative/neutral",
            "sentiment_reasoning": "explanation",
            "impact_summary": "potential impact on stock"
        }}
        """
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON portion
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            result = {
                'sentiment': 'neutral',
                'sentiment_reasoning': 'Unable to determine sentiment from analysis',
                'impact_summary': 'Analysis inconclusive. Consider additional research.'
            }
        
        # Add the symbol and news items
        result['symbol'] = symbol
        result['news_items'] = news_items
        
        return result
    
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {e}")
        return {
            'symbol': symbol,
            'sentiment': 'error',
            'sentiment_reasoning': f'Error during analysis: {str(e)}',
            'impact_summary': 'Analysis failed. Please try again later.',
            'news_items': news_items
        }

def get_stock_analysis(symbol):
    """
    Main function to get stock news and analyze sentiment.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Analysis results including news and sentiment
    """
    # Get news and context
    news_items, context = get_stock_news(symbol)
    
    # Analyze sentiment
    result = analyze_sentiment(news_items, context, symbol)
    
    return result

# Test function
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    result = get_stock_analysis(symbol)
    print(json.dumps(result, indent=2))