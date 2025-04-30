"""
Stock news and sentiment analysis module using News API and Gemini API.
"""
import os
import json
import yfinance as yf
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with the API key."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Set the API key directly for testing
        api_key = "AIzaSyCrr6OzYwYvuiorPvmAAkYwb0lHQI8U7Wo"
        os.environ['GEMINI_API_KEY'] = api_key
        print("Using hardcoded API key for testing")
        
    genai.configure(api_key=api_key)
    return True if api_key else False

def get_stock_news(symbol, days_lookback=3):
    """
    Get recent news for a stock using News API.
    
    Args:
        symbol (str): Stock symbol
        days_lookback (int): Number of days to look back for news
        
    Returns:
        list: List of news items with date, headline, and link
    """
    try:
        # Get News API key from environment or use default
        news_api_key = os.environ.get('NEWS_API_KEY', '9b73205028734f2181dcda4f1b892d66')

        # Get company info from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get('longName', symbol)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Calculate date for query (NewsAPI uses format: 2025-04-30)
        from_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
        
        # Construct URL - added sortBy=publishedAt to get newest articles first
        url = f"https://newsapi.org/v2/everything?q={company_name}&from={from_date}&sortBy=publishedAt&apiKey={news_api_key}"
        
        print(f"Fetching news for {company_name} from {from_date} using News API")
        
        # Make request
        response = requests.get(url)
        news_data = response.json()
        
        news_items = []
        if news_data.get('status') == 'ok' and 'articles' in news_data:
            articles = news_data['articles']
            print(f"Found {len(articles)} news articles using News API (using top 10 most recent)")
            
            for article in articles[:10]:  # Limit to 10 most recent articles
                news_date = datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
                
                news_items.append({
                    'date': news_date.strftime('%Y-%m-%d'),
                    'headline': article.get('title', 'No headline available'),
                    'summary': article.get('description', 'No summary available')[:100] + '...' if article.get('description') else 'No summary available',
                    'url': article.get('url', '#')
                })
        else:
            print(f"Error or no articles from News API: {news_data.get('message', 'Unknown error')}")
        
        # Get recent stock price movement
        hist = ticker.history(period='1mo')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            month_ago_price = hist['Close'].iloc[0]
            price_change_pct = ((current_price - month_ago_price) / month_ago_price) * 100
        else:
            price_change_pct = 0
            current_price = 0
            
        context = {
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'price_change_pct': round(price_change_pct, 2),
            'current_price': round(current_price, 2) if current_price != 0 else 'Unknown'
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
            'impact_summary': 'Unable to analyze sentiment due to API configuration issues.',
            'news_items': news_items
        }
    
    # If no news items found
    if not news_items:
        return {
            'symbol': symbol,
            'sentiment': 'neutral',
            'sentiment_reasoning': f'No recent news found for {context["company_name"]} ({symbol})',
            'impact_summary': 'Without recent news, sentiment analysis is inconclusive. Consider technical indicators instead.',
            'news_items': news_items
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

def get_stock_analysis(symbol, days_lookback=3):
    """
    Main function to get stock news and analyze sentiment.
    
    Args:
        symbol (str): Stock symbol
        days_lookback (int): Number of days to look back for news
        
    Returns:
        dict: Analysis results including news and sentiment
    """
    # Get news and context
    news_items, context = get_stock_news(symbol, days_lookback=days_lookback)
    
    # Analyze sentiment
    result = analyze_sentiment(news_items, context, symbol)
    
    return result

# Test function
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    result = get_stock_analysis(symbol)
    print(json.dumps(result, indent=2))