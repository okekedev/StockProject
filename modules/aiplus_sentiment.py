"""
AI+ Sentiment Analyzer - Streamlined Implementation

This module fetches and analyzes news sentiment for stocks with improved reliability
and efficiency through better data validation and error handling.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
import yfinance as yf
import re
import config
from dotenv import load_dotenv

# Load environment variables and setup constants
load_dotenv()
NEWS_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_news_cache")
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

# Keyword lists for rule-based analysis
POSITIVE_KEYWORDS = [
    'up', 'rise', 'gain', 'growth', 'profit', 'bullish', 'outperform', 'beat', 
    'exceed', 'positive', 'strong', 'success', 'win', 'soar', 'jump', 'rally', 
    'improve', 'upgrade', 'opportunity', 'innovation'
]

NEGATIVE_KEYWORDS = [
    'down', 'fall', 'drop', 'decline', 'loss', 'bearish', 'underperform', 'miss', 
    'below', 'negative', 'weak', 'fail', 'lose', 'plunge', 'crash', 'sink', 
    'struggle', 'downgrade', 'risk', 'concern', 'worry', 'lawsuit', 'investigation', 
    'recall', 'fine', 'penalty', 'cut', 'layoff'
]

# Key event patterns for identifying important news
KEY_EVENT_PATTERNS = [
    r'acqui(?:re|sition)', r'partner(?:ship)?', r'launch(?:es|ed)?', r'announc(?:e|ed|es)',
    r'releas(?:e|ed|es)', r'approv(?:e|al|ed)', r'deal', r'contract', r'invest(?:ment)?',
    r'quarterly results', r'earnings', r'guidance', r'forecast', r'outlook', r'CEO',
    r'executive', r'lawsuit', r'regulatory', r'patent', r'dividend', r'restructur(?:e|ing)',
    r'layoffs?'
]

class SentimentAnalyzer:
    """Handles news retrieval and sentiment analysis for stocks."""
    
    def __init__(self):
        """Initialize the analyzer with APIs and caching."""
        # API keys and initialization
        self.news_api_key = os.environ.get('NEWS_API_KEY', config.NEWS_API_KEY)
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY', config.GEMINI_API_KEY)
        self.api_initialized = self._init_api()
        
        # Caching system
        self.cache = {}
        self.last_fetch_time = {}
        
        # Compile regex patterns once for efficiency
        self.event_pattern = re.compile('|'.join(KEY_EVENT_PATTERNS), re.IGNORECASE)
        self.pos_sentiment_pattern = re.compile(r'gain|growth|profit|up|bullish|beat|exceed|positive', re.IGNORECASE)
        self.neg_sentiment_pattern = re.compile(r'loss|drop|decline|down|bearish|miss|below|negative', re.IGNORECASE)
    
    def _init_api(self):
        """Initialize the Gemini API if possible."""
        if not self.gemini_api_key:
            return False
            
        try:
            genai.configure(api_key=self.gemini_api_key)
            return True
        except Exception as e:
            print(f"Failed to initialize Gemini API: {e}")
            return False
    
    def get_sentiment(self, symbol, days_lookback=7, force_refresh=False):
        """
        Main method to get sentiment analysis for a stock.
        
        Args:
            symbol (str): Stock symbol
            days_lookback (int): Number of days to look back for news
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            dict: Sentiment analysis results
        """
        # Create unique cache key
        cache_key = f"{symbol}_{days_lookback}"
        cache_file = os.path.join(NEWS_CACHE_DIR, f"{cache_key}.json")
        
        # Check memory cache first (valid for 6 hours)
        if not force_refresh and self._check_memory_cache(cache_key):
            return self.cache[cache_key]
        
        # Then check disk cache
        if not force_refresh and self._check_disk_cache(cache_file):
            return self.cache[cache_key]
        
        try:
            # Get stock information (company name, sector, etc.)
            stock_info = self._get_stock_info(symbol)
            
            # Fetch news articles from multiple sources
            news_items = self._fetch_news(symbol, stock_info, days_lookback)
            
            # No news case - return neutral sentiment
            if not news_items:
                return self._create_empty_result(symbol, stock_info, days_lookback)
            
            # Analyze sentiment (using Gemini if available, else rule-based)
            sentiment_data = self._analyze_sentiment(news_items, stock_info)
            
            # Extract key developments from news
            key_developments = self._extract_key_developments(news_items)
            
            # Build complete result
            result = {
                "symbol": symbol,
                "company_name": stock_info.get("company_name", symbol),
                "sector": stock_info.get("sector", "Unknown"),
                "industry": stock_info.get("industry", "Unknown"),
                "days_analyzed": days_lookback,
                "news_count": len(news_items),
                "sentiment_score": sentiment_data.get("sentiment_score", 0),
                "sentiment": sentiment_data.get("sentiment", "neutral"),
                "sentiment_summary": sentiment_data.get("sentiment_summary", ""),
                "sentiment_reasoning": sentiment_data.get("sentiment_reasoning", ""),
                "news_items": news_items[:10],  # Limit to 10 most recent items
                "key_developments": key_developments,
                "sentiment_drivers": sentiment_data.get("sentiment_drivers", []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to caches
            self._save_to_cache(cache_key, cache_file, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error analyzing news sentiment for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg, "symbol": symbol}
    
    def _check_memory_cache(self, cache_key):
        """Check if valid data exists in memory cache."""
        if cache_key in self.cache:
            last_fetch = self.last_fetch_time.get(cache_key, datetime(1970, 1, 1))
            if (datetime.now() - last_fetch).total_seconds() < 21600:  # 6 hours
                return True
        return False
    
    def _check_disk_cache(self, cache_file):
        """Check if valid data exists in disk cache."""
        if not os.path.exists(cache_file):
            return False
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            if 'timestamp' in cache_data:
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < 21600:  # 6 hours
                    self.cache[cache_data['symbol'] + f"_{cache_data['days_analyzed']}"] = cache_data
                    self.last_fetch_time[cache_data['symbol'] + f"_{cache_data['days_analyzed']}"] = cache_time
                    return True
        except Exception as e:
            print(f"Error reading cache file: {e}")
            
        return False
    
    def _save_to_cache(self, cache_key, cache_file, data):
        """Save data to both memory and disk cache."""
        # Save to memory cache
        self.cache[cache_key] = data
        self.last_fetch_time[cache_key] = datetime.now()
        
        # Save to disk cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving to cache file: {e}")
    
    def _get_stock_info(self, symbol):
        """Get basic information about a stock."""
        try:
            # Get data from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                "symbol": symbol,
                "company_name": info.get('longName', symbol),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0),
                "current_price": info.get('currentPrice', 0)
            }
            
            # Calculate recent price movement
            hist = ticker.history(period='1mo')
            if not hist.empty:
                price_start = hist['Close'].iloc[0] 
                price_end = hist['Close'].iloc[-1]
                if price_start > 0:
                    price_change = ((price_end - price_start) / price_start) * 100
                    stock_info["price_change_pct"] = price_change
            
            return stock_info
            
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return {"symbol": symbol, "company_name": symbol}
    
    def _fetch_news(self, symbol, stock_info, days_lookback):
        """Fetch news from multiple sources and deduplicate results."""
        all_news = []
        company_name = stock_info.get("company_name", symbol)
        
        # Date range for news
        from_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
        
        # Try News API first
        if self.news_api_key:
            search_query = f"{company_name} {symbol} stock"
            try:
                url = f"https://newsapi.org/v2/everything?q={search_query}&from={from_date}&sortBy=publishedAt&apiKey={self.news_api_key}"
                response = requests.get(url, timeout=10)  # Add timeout
                
                if response.status_code == 200:
                    news_data = response.json()
                    if news_data.get('status') == 'ok' and 'articles' in news_data:
                        # Process each article
                        for article in news_data['articles']:
                            try:
                                news_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                                news_item = {
                                    'date': news_date.strftime('%Y-%m-%d'),
                                    'source': article.get('source', {}).get('name', 'Unknown'),
                                    'headline': article.get('title', 'No headline available'),
                                    'summary': article.get('description', 'No summary available'),
                                    'url': article.get('url', '#')
                                }
                                all_news.append(news_item)
                            except Exception as e:
                                print(f"Error processing article: {e}")
                                continue
            except Exception as e:
                print(f"Error fetching from News API: {e}")
        
        # Also try Yahoo Finance
        try:
            ticker = yf.Ticker(symbol)
            yahoo_news = ticker.news
            
            for item in yahoo_news:
                try:
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
                        'url': item.get('link', '#')
                    }
                    all_news.append(news_item)
                except Exception as e:
                    print(f"Error processing Yahoo news item: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
        
        # Deduplicate news items
        return self._deduplicate_news(all_news)
    
    def _deduplicate_news(self, news_items):
        """Sort by date and remove duplicate headlines."""
        # Sort by date, newest first
        news_items.sort(key=lambda x: x.get('date', '1970-01-01'), reverse=True)
        
        # Remove duplicates based on headline
        seen_headlines = set()
        unique_news = []
        
        for item in news_items:
            # Normalize headline for comparison
            headline = item.get('headline', '').lower().strip()
            if headline and headline not in seen_headlines:
                seen_headlines.add(headline)
                unique_news.append(item)
        
        return unique_news
    
    def _analyze_sentiment(self, news_items, stock_info):
        """Analyze sentiment using Gemini API or rule-based method."""
        if self.api_initialized:
            # Try AI-based analysis first
            try:
                ai_result = self._analyze_with_gemini(news_items, stock_info)
                if ai_result and 'sentiment' in ai_result:
                    return ai_result
            except Exception as e:
                print(f"Error with Gemini analysis, falling back to rule-based: {e}")
        
        # Fall back to rule-based analysis
        return self._analyze_rule_based(news_items, stock_info)
    
    def _analyze_with_gemini(self, news_items, stock_info):
        """Analyze sentiment using Gemini API."""
        # Limit to most recent 10 news items to respect context limits
        recent_news = news_items[:10]
        
        # Format news for the prompt
        news_text = "\n".join([
            f"Date: {item['date']}, Source: {item['source']}\nHeadline: {item['headline']}\nSummary: {item['summary']}"
            for item in recent_news
        ])
        
        # Format company info
        company_name = stock_info.get('company_name', stock_info.get('symbol'))
        symbol = stock_info.get('symbol')
        company_info = (
            f"Company: {company_name}\n"
            f"Symbol: {symbol}\n"
            f"Sector: {stock_info.get('sector', 'Unknown')}\n"
            f"Industry: {stock_info.get('industry', 'Unknown')}\n"
        )
        
        if 'price_change_pct' in stock_info:
            company_info += f"Recent price change: {stock_info['price_change_pct']:.2f}% over the past month\n"
        
        # Craft prompt for sentiment analysis
        prompt = f"""
        Please analyze the sentiment of these news articles about {company_name} ({symbol}).

        Company Information:
        {company_info}
        
        Recent News Articles:
        {news_text}
        
        Provide detailed sentiment analysis in JSON format with these fields:
        1. sentiment: The overall sentiment (positive, negative, neutral, or mixed)
        2. sentiment_score: A score from -100 (very negative) to +100 (very positive)
        3. sentiment_reasoning: Reasoning for your assessment
        4. sentiment_drivers: Array of 2-4 key factors driving sentiment
        5. sentiment_summary: 2-3 sentence summary of the sentiment analysis
        
        JSON Format:
        {{
            "sentiment": "positive/negative/neutral/mixed",
            "sentiment_score": number,
            "sentiment_reasoning": "explanation",
            "sentiment_drivers": ["driver1", "driver2", "driver3"],
            "sentiment_summary": "2-3 sentence summary"
        }}
        """
        
        # Generate response with safety settings
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract JSON portion from response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            try:
                result = json.loads(json_str)
                # Validate required fields or set defaults
                if 'sentiment' not in result:
                    result['sentiment'] = 'neutral'
                if 'sentiment_score' not in result:
                    result['sentiment_score'] = 0
                return result
            except json.JSONDecodeError:
                print("Failed to parse Gemini response as JSON")
        
        # Return default in case of problems
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0,
            'sentiment_reasoning': 'Unable to analyze sentiment properly',
            'sentiment_drivers': [],
            'sentiment_summary': 'Analysis inconclusive due to technical issues.'
        }
    
    def _analyze_rule_based(self, news_items, stock_info):
        """Rule-based sentiment analysis using keyword matching."""
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Analyze sentiment for each news item
        for item in news_items:
            text = (item.get('headline', '') + ' ' + item.get('summary', '')).lower()
            
            pos_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
            neg_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
            
            if pos_count > neg_count:
                positive_count += 1
            elif neg_count > pos_count:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate sentiment score and overall sentiment
        total_items = len(news_items)
        sentiment_score = 0
        
        if total_items > 0:
            sentiment_score = ((positive_count - negative_count) / total_items) * 100
        
        # Determine overall sentiment
        if sentiment_score > 30:
            sentiment = "positive"
        elif sentiment_score < -30:
            sentiment = "negative"
        elif positive_count > 0 and negative_count > 0:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        # Extract topics that might be driving sentiment
        sentiment_drivers = self._extract_topics(news_items[:5])
        
        # Generate summary based on sentiment
        company_name = stock_info.get('company_name', stock_info.get('symbol'))
        
        summary_templates = {
            "positive": f"News sentiment for {company_name} is predominantly positive, suggesting favorable market perception. The company has received encouraging coverage recently.",
            "negative": f"News sentiment for {company_name} is mostly negative, indicating concerns in the market. Recent coverage highlights challenges the company is facing.",
            "mixed": f"News sentiment for {company_name} is mixed, with both positive and negative coverage. The market appears to have conflicting views about the company's prospects.",
            "neutral": f"News sentiment for {company_name} is relatively neutral. Recent coverage does not strongly indicate positive or negative market perception."
        }
        
        return {
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'sentiment_reasoning': f"Based on {total_items} news items: {positive_count} positive, {negative_count} negative, {neutral_count} neutral",
            'sentiment_drivers': sentiment_drivers,
            'sentiment_summary': summary_templates.get(sentiment, summary_templates["neutral"])
        }
    
    def _extract_topics(self, news_items):
        """Extract key topics from news headlines."""
        topics = []
        seen_topics = set()
        
        # Simple pattern to extract potential topics
        topic_pattern = re.compile(r'\b[A-Z][a-z]+ [A-Za-z]+(?: [A-Za-z]+)?\b')
        
        for item in news_items:
            headline = item.get('headline', '')
            matches = topic_pattern.findall(headline)
            
            for match in matches:
                if match.lower() not in seen_topics:
                    seen_topics.add(match.lower())
                    topics.append(match)
                    if len(topics) >= 3:  # Limit to top 3
                        return topics
        
        return topics
    
    def _extract_key_developments(self, news_items):
        """Extract key developments from news items."""
        developments = []
        
        for item in news_items:
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            date = item.get('date', '')
            
            # Check if this is a key development
            if self.event_pattern.search(headline) or self.event_pattern.search(summary):
                # Determine sentiment for this item
                combined_text = (headline + " " + summary).lower()
                pos_matches = len(self.pos_sentiment_pattern.findall(combined_text))
                neg_matches = len(self.neg_sentiment_pattern.findall(combined_text))
                
                sentiment = "neutral"
                if pos_matches > neg_matches:
                    sentiment = "positive" 
                elif neg_matches > pos_matches:
                    sentiment = "negative"
                
                developments.append({
                    'date': date,
                    'headline': headline,
                    'sentiment': sentiment,
                    'url': item.get('url', '#')
                })
                
                # Limit to top 5 developments
                if len(developments) >= 5:
                    break
        
        return developments
    
    def _create_empty_result(self, symbol, stock_info, days_lookback):
        """Create a default neutral result when no news is found."""
        company_name = stock_info.get("company_name", symbol)
        
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": stock_info.get("sector", "Unknown"),
            "industry": stock_info.get("industry", "Unknown"),
            "days_analyzed": days_lookback,
            "news_count": 0,
            "sentiment_score": 0,
            "sentiment": "neutral",
            "sentiment_summary": f"No recent news found for {company_name}. The analysis is inconclusive due to lack of data.",
            "sentiment_reasoning": "No news articles found for analysis",
            "news_items": [],
            "key_developments": [],
            "sentiment_drivers": [],
            "timestamp": datetime.now().isoformat()
        }


# Module-level function for external use
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
    analyzer = SentimentAnalyzer()
    return analyzer.get_sentiment(symbol, days_lookback, force_refresh)


# Test code
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Testing sentiment analysis for {symbol}...")
    result = get_aiplus_sentiment(symbol, force_refresh=True)
    print(json.dumps(result, indent=2))