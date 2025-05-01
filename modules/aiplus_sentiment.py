"""
AI+ Sentiment Analysis Module

This module handles fetching news data, analyzing sentiment, and extracting key insights
for the AI+ enhanced prediction system. It extends the regular news analysis with
deeper context understanding and multi-source integration.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
from bs4 import BeautifulSoup
import yfinance as yf
import time
import re
import config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
NEWS_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_news_cache")
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with the API key."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Try from config if not in environment
        api_key = config.GEMINI_API_KEY
        
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        print("Warning: No Gemini API key found")
        return False


class AIplusSentimentAnalyzer:
    """
    Class for fetching news, analyzing sentiment and extracting insights
    for AI-powered stock predictions.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.cache = {}
        self.last_fetch_time = {}
        self.api_initialized = configure_gemini()
        self.news_api_key = os.environ.get('NEWS_API_KEY', config.NEWS_API_KEY)
    
    def get_news_sentiment(self, symbol, days_lookback=7, force_refresh=False):
        """
        Get news sentiment analysis for a stock.
        
        Args:
            symbol (str): Stock symbol
            days_lookback (int): Number of days to look back for news
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            dict: News sentiment analysis results
        """
        cache_key = f"{symbol}_{days_lookback}"
        cache_file = os.path.join(NEWS_CACHE_DIR, f"{cache_key}.json")
        
        # Check if we have fresh cached data
        if not force_refresh and cache_key in self.cache:
            # Use cached data if it exists and is recent (within 6 hours)
            if (datetime.now() - self.last_fetch_time.get(cache_key, datetime(1970, 1, 1))).total_seconds() < 21600:
                return self.cache[cache_key]
        
        # Try to load from disk cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is fresh (within 6 hours)
                if 'timestamp' in cache_data:
                    cache_time = datetime.fromisoformat(cache_data['timestamp'])
                    if (datetime.now() - cache_time).total_seconds() < 21600:
                        self.cache[cache_key] = cache_data
                        self.last_fetch_time[cache_key] = cache_time
                        return cache_data
            except Exception as e:
                print(f"Error loading cache for {symbol}: {e}")
        
        try:
            # Get stock info
            stock_info = self._get_stock_info(symbol)
            
            # Fetch news from multiple sources
            news_items = self._fetch_news_data(symbol, stock_info, days_lookback)
            
            if not news_items:
                # No news found
                result = {
                    "symbol": symbol,
                    "company_name": stock_info.get("company_name", symbol),
                    "days_analyzed": days_lookback,
                    "news_count": 0,
                    "sentiment_score": 0,
                    "sentiment": "neutral",
                    "news_items": [],
                    "key_developments": [],
                    "sentiment_drivers": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the results
                self.cache[cache_key] = result
                self.last_fetch_time[cache_key] = datetime.now()
                
                # Save to disk cache
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                
                return result
            
            # Analyze sentiment using multiple methods
            sentiment_results = self._analyze_sentiment(news_items, stock_info)
            
            # Extract key developments
            key_developments = self._extract_key_developments(news_items, sentiment_results)
            
            # Overall result
            result = {
                "symbol": symbol,
                "company_name": stock_info.get("company_name", symbol),
                "sector": stock_info.get("sector", "Unknown"),
                "industry": stock_info.get("industry", "Unknown"),
                "days_analyzed": days_lookback,
                "news_count": len(news_items),
                "sentiment_score": sentiment_results.get("sentiment_score", 0),
                "sentiment": sentiment_results.get("sentiment", "neutral"),
                "sentiment_summary": sentiment_results.get("sentiment_summary", ""),
                "news_items": news_items[:10],  # Limit to top 10 news items
                "key_developments": key_developments,
                "sentiment_drivers": sentiment_results.get("sentiment_drivers", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the results
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = datetime.now()
            
            # Save to disk cache
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            error_msg = f"Error analyzing news sentiment for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    def _get_stock_info(self, symbol):
        """
        Get basic stock information.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock information
        """
        try:
            # Get company info from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            stock_info = {
                "symbol": symbol,
                "company_name": info.get('longName', symbol),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "website": info.get('website', ''),
                "business_summary": info.get('longBusinessSummary', ''),
                "market_cap": info.get('marketCap', 0),
                "current_price": info.get('currentPrice', 0)
            }
            
            # Get recent stock price movement
            hist = ticker.history(period='1mo')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                month_ago_price = hist['Close'].iloc[0]
                price_change_pct = ((current_price - month_ago_price) / month_ago_price) * 100
                stock_info["price_change_pct"] = price_change_pct
            
            return stock_info
            
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return {"symbol": symbol, "company_name": symbol}
    
    def _fetch_news_data(self, symbol, stock_info, days_lookback):
        """
        Fetch news data from multiple sources.
        
        Args:
            symbol (str): Stock symbol
            stock_info (dict): Stock information
            days_lookback (int): Number of days to look back
            
        Returns:
            list: News items
        """
        all_news = []
        
        # Use company name for better search results
        company_name = stock_info.get("company_name", symbol)
        
        # Add ticker symbol for specificity
        search_query = f"{company_name} {symbol} stock"
        
        # Calculate date for query
        from_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
        
        # Fetch from News API
        if self.news_api_key:
            try:
                # Construct URL
                url = f"https://newsapi.org/v2/everything?q={search_query}&from={from_date}&sortBy=publishedAt&apiKey={self.news_api_key}"
                
                # Make request
                response = requests.get(url)
                news_data = response.json()
                
                if news_data.get('status') == 'ok' and 'articles' in news_data:
                    articles = news_data['articles']
                    
                    for article in articles:
                        news_date = datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
                        
                        news_item = {
                            'date': news_date.strftime('%Y-%m-%d'),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'headline': article.get('title', 'No headline available'),
                            'summary': article.get('description', 'No summary available'),
                            'url': article.get('url', '#'),
                            'content': article.get('content', '')
                        }
                        
                        all_news.append(news_item)
            except Exception as e:
                print(f"Error fetching news from News API: {e}")
        
        # Supplement with Yahoo Finance news
        try:
            # Get news from Yahoo Finance via yfinance
            ticker = yf.Ticker(symbol)
            yahoo_news = ticker.news
            
            for item in yahoo_news:
                # Convert timestamp to datetime
                news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                
                # Skip if older than lookback period
                if news_date < datetime.now() - timedelta(days=days_lookback):
                    continue
                
                news_item = {
                    'date': news_date.strftime('%Y-%m-%d'),
                    'source': 'Yahoo Finance',
                    'headline': item.get('title', 'No headline available'),
                    'summary': item.get('summary', 'No summary available'),
                    'url': item.get('link', '#'),
                    'content': ''  # Yahoo doesn't provide full content
                }
                
                all_news.append(news_item)
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        # Sort by date (newest first) and remove duplicates
        all_news.sort(key=lambda x: x['date'], reverse=True)
        
        # Remove duplicates (based on headlines)
        seen_headlines = set()
        unique_news = []
        
        for item in all_news:
            headline = item['headline'].lower()
            if headline not in seen_headlines:
                seen_headlines.add(headline)
                unique_news.append(item)
        
        return unique_news
    
    def _analyze_sentiment(self, news_items, stock_info):
        """
        Analyze sentiment of news items using multiple methods.
        
        Args:
            news_items (list): News items
            stock_info (dict): Stock information
            
        Returns:
            dict: Sentiment analysis results
        """
        # Use Gemini API for sentiment analysis if available
        if self.api_initialized:
            return self._analyze_sentiment_with_gemini(news_items, stock_info)
        else:
            # Fallback to rule-based analysis
            return self._analyze_sentiment_rule_based(news_items, stock_info)
    
    def _analyze_sentiment_with_gemini(self, news_items, stock_info):
        """
        Analyze sentiment using Gemini API.
        
        Args:
            news_items (list): News items
            stock_info (dict): Stock information
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Limit to most recent 10 news items to respect context limits
            recent_news = news_items[:10]
            
            # Format news for the prompt
            news_text = "\n".join([
                f"Date: {item['date']}, Source: {item['source']}\nHeadline: {item['headline']}\nSummary: {item['summary']}"
                for item in recent_news
            ])
            
            # Create a prompt for Gemini
            company_info = f"Company: {stock_info.get('company_name', stock_info.get('symbol'))}\n"
            company_info += f"Sector: {stock_info.get('sector', 'Unknown')}\n"
            company_info += f"Industry: {stock_info.get('industry', 'Unknown')}\n"
            
            if 'price_change_pct' in stock_info:
                company_info += f"Recent price change: {stock_info['price_change_pct']:.2f}% over the past month\n"
            
            prompt = f"""
            Please analyze the sentiment of these news articles about {stock_info.get('company_name', stock_info.get('symbol'))} ({stock_info.get('symbol')}).

            Company Information:
            {company_info}
            
            Recent News Articles:
            {news_text}
            
            Please provide the following in JSON format:
            1. Overall sentiment (positive, negative, neutral, or mixed)
            2. A sentiment score from -100 (very negative) to +100 (very positive)
            3. A brief reasoning for the sentiment assessment
            4. Key sentiment drivers (the main factors influencing sentiment)
            5. A short sentiment summary (2-3 sentences)
            
            Format as:
            {{
                "sentiment": "positive/negative/neutral/mixed",
                "sentiment_score": number,
                "sentiment_reasoning": "explanation",
                "sentiment_drivers": ["driver1", "driver2", "driver3"],
                "sentiment_summary": "2-3 sentence summary"
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
                    'sentiment_score': 0,
                    'sentiment_reasoning': 'Unable to determine sentiment from analysis',
                    'sentiment_drivers': [],
                    'sentiment_summary': 'Analysis inconclusive.'
                }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing sentiment with Gemini: {e}")
            # Fallback to rule-based analysis
            return self._analyze_sentiment_rule_based(news_items, stock_info)
    
    def _analyze_sentiment_rule_based(self, news_items, stock_info):
        """
        Analyze sentiment using rule-based methods.
        
        Args:
            news_items (list): News items
            stock_info (dict): Stock information
            
        Returns:
            dict: Sentiment analysis results
        """
        # Simple rule-based sentiment analysis using keyword matching
        positive_keywords = [
            'up', 'rise', 'gain', 'growth', 'profit', 'bullish', 'outperform',
            'beat', 'exceed', 'positive', 'strong', 'success', 'win', 'soar',
            'jump', 'rally', 'improve', 'upgrade', 'opportunity', 'innovation'
        ]
        
        negative_keywords = [
            'down', 'fall', 'drop', 'decline', 'loss', 'bearish', 'underperform',
            'miss', 'below', 'negative', 'weak', 'fail', 'lose', 'plunge', 
            'crash', 'sink', 'struggle', 'downgrade', 'risk', 'concern', 'worry',
            'lawsuit', 'investigation', 'recall', 'fine', 'penalty', 'cut', 'layoff'
        ]
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Count sentiment indicators in headlines and summaries
        for item in news_items:
            headline = item['headline'].lower()
            summary = item['summary'].lower()
            
            item_positive = 0
            item_negative = 0
            
            # Check headline and summary for sentiment keywords
            for keyword in positive_keywords:
                if keyword in headline or keyword in summary:
                    item_positive += 1
            
            for keyword in negative_keywords:
                if keyword in headline or keyword in summary:
                    item_negative += 1
            
            # Determine sentiment for this news item
            if item_positive > item_negative:
                positive_count += 1
            elif item_negative > item_positive:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate sentiment score (-100 to +100)
        total_items = len(news_items)
        if total_items > 0:
            sentiment_score = ((positive_count - negative_count) / total_items) * 100
        else:
            sentiment_score = 0
        
        # Determine overall sentiment
        if sentiment_score > 30:
            sentiment = "positive"
        elif sentiment_score < -30:
            sentiment = "negative"
        elif positive_count > 0 and negative_count > 0:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        # Extract potential sentiment drivers
        sentiment_drivers = []
        seen_topics = set()
        
        for item in news_items[:5]:  # Consider most recent 5 items for drivers
            headline = item['headline']
            
            # Extract key phrases (simplistic approach)
            key_phrases = re.findall(r'\b[A-Z][a-z]+ [A-Za-z]+(?: [A-Za-z]+)?\b', headline)
            
            for phrase in key_phrases:
                if phrase.lower() not in seen_topics:
                    seen_topics.add(phrase.lower())
                    sentiment_drivers.append(phrase)
        
        # Limit to top 3 drivers
        sentiment_drivers = sentiment_drivers[:3]
        
        # Generate sentiment summary
        company_name = stock_info.get('company_name', stock_info.get('symbol'))
        
        if sentiment == "positive":
            sentiment_summary = f"News sentiment for {company_name} is predominantly positive, suggesting favorable market perception. The company has received encouraging coverage recently."
        elif sentiment == "negative":
            sentiment_summary = f"News sentiment for {company_name} is mostly negative, indicating concerns in the market. Recent coverage highlights challenges the company is facing."
        elif sentiment == "mixed":
            sentiment_summary = f"News sentiment for {company_name} is mixed, with both positive and negative coverage. The market appears to have conflicting views about the company's prospects."
        else:
            sentiment_summary = f"News sentiment for {company_name} is relatively neutral. Recent coverage does not strongly indicate positive or negative market perception."
        
        return {
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'sentiment_reasoning': f"Based on {total_items} news items: {positive_count} positive, {negative_count} negative, {neutral_count} neutral",
            'sentiment_drivers': sentiment_drivers,
            'sentiment_summary': sentiment_summary
        }
    
    def _extract_key_developments(self, news_items, sentiment_results):
        """
        Extract key developments from news items.
        
        Args:
            news_items (list): News items
            sentiment_results (dict): Sentiment analysis results
            
        Returns:
            list: Key developments
        """
        key_developments = []
        
        # Keywords indicating significant developments
        key_event_patterns = [
            r'acqui(?:re|sition)',
            r'partner(?:ship)?',
            r'launch(?:es|ed)?',
            r'announc(?:e|ed|es)',
            r'releas(?:e|ed|es)',
            r'approv(?:e|al|ed)',
            r'deal',
            r'contract',
            r'invest(?:ment)?',
            r'quarterly results',
            r'earnings',
            r'guidance',
            r'forecast',
            r'outlook',
            r'CEO',
            r'executive',
            r'lawsuit',
            r'regulatory',
            r'patent',
            r'dividend',
            r'restructur(?:e|ing)',
            r'layoffs?'
        ]
        
        pattern = re.compile('|'.join(key_event_patterns), re.IGNORECASE)
        
        # Look through news items
        for item in news_items:
            headline = item['headline']
            summary = item['summary']
            date = item['date']
            
            # Check if headline or summary contains key event patterns
            if pattern.search(headline) or pattern.search(summary):
                # Determine sentiment for this item
                item_sentiment = "neutral"
                
                # Simple sentiment check based on keywords
                positive_matches = len(re.findall(r'gain|growth|profit|up|bullish|beat|exceed|positive', headline.lower() + " " + summary.lower()))
                negative_matches = len(re.findall(r'loss|drop|decline|down|bearish|miss|below|negative', headline.lower() + " " + summary.lower()))
                
                if positive_matches > negative_matches:
                    item_sentiment = "positive"
                elif negative_matches > positive_matches:
                    item_sentiment = "negative"
                
                development = {
                    'date': date,
                    'headline': headline,
                    'sentiment': item_sentiment,
                    'url': item['url']
                }
                
                key_developments.append(development)
        
        # Sort by date (newest first) and limit to top 5
        key_developments.sort(key=lambda x: x['date'], reverse=True)
        return key_developments[:5]


# Function to get sentiment analysis for a specific stock
def get_aiplus_sentiment(symbol, days_lookback=7, force_refresh=False):
    """
    Get sentiment analysis for a stock.
    
    Args:
        symbol (str): Stock symbol
        days_lookback (int): Number of days to look back for news
        force_refresh (bool): Whether to force refresh data
        
    Returns:
        dict: Sentiment analysis results
    """
    analyzer = AIplusSentimentAnalyzer()
    return analyzer.get_news_sentiment(symbol, days_lookback, force_refresh)


# Test function
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    days_lookback = 7
    
    result = get_aiplus_sentiment(symbol, days_lookback, force_refresh=True)
    print(json.dumps(result, indent=2))