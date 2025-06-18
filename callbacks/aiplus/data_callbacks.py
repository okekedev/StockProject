"""
Data fetch callbacks for the AI+ tab with updated local storage.
"""
from dash import html, callback, Output, Input, State
import pandas as pd
import os
import json
import config

# Import modules
from modules.aiplus_technical import get_aiplus_technical_data
from modules.aiplus_sentiment import get_aiplus_sentiment
from callbacks.aiplus.status_callbacks import update_data_readiness


# Callback to populate the stock dropdown
@callback(
    Output('aiplus-stock-dropdown', 'options'),
    Input('tabs', 'value')
)
def populate_aiplus_dropdown(tab):
    """
    Populate the stock dropdown for AI+ analysis.
    
    Args:
        tab (str): Current active tab
        
    Returns:
        list: Dropdown options
    """
    if tab != 'aiplus':
        return []
    
    try:
        # Load all available stocks
        symbols_file = config.OUTPUT_FILE
        if not os.path.exists(symbols_file):
            return []
        
        symbols_df = pd.read_csv(symbols_file)
        
        # Create dropdown options
        options = [
            {
                'label': symbol,
                'value': symbol
            }
            for symbol in symbols_df['Symbol'].tolist()
        ]
        
        return options
    except Exception as e:
        print(f"Error loading symbols: {e}")
        return []


# Technical data fetch callback
@callback(
    Output('tech-data-status', 'children'),
    Output('tech-data-status', 'className'),
    Input('fetch-tech-button', 'n_clicks'),
    State('aiplus-stock-dropdown', 'value'),
    State('tech-timeframe-dropdown', 'value'),
    State('tabs', 'value'),
    prevent_initial_call=True
)
def fetch_technical_data(n_clicks, symbol, timeframe, tab):
    """
    Fetch technical data for AI+ analysis.
    
    Args:
        n_clicks (int): Number of button clicks
        symbol (str): Selected stock symbol
        timeframe (str): Selected timeframe
        tab (str): Current tab
        
    Returns:
        tuple: (status_message, status_class)
    """
    if n_clicks == 0 or tab != 'aiplus' or not symbol:
        return "", "bank-status"
    
    try:
        # Show loading status
        status_message = html.Div([
            html.P("Fetching technical data...", className="bank-text"),
            html.Div(className="bank-loading"),
        ])
        
        # Force refresh to get newest data
        technical_data = get_aiplus_technical_data(symbol, timeframe, force_refresh=True)
        
        if 'error' in technical_data:
            return html.P(f"Error: {technical_data['error']}", className="bank-text"), "bank-status status-error"
        
        # Extract key info for display
        current_price = technical_data.get('current_price', 0)
        price_change = technical_data.get('price_change_pct', 0)
        volatility = technical_data.get('volatility', 0)
        
        # Get TSMN value
        tsmn = technical_data.get('tsmn', {})
        tsmn_value = tsmn.get('value', 0) if isinstance(tsmn, dict) else 0
        tsmn_signal = tsmn.get('signal', 'neutral') if isinstance(tsmn, dict) else 'neutral'
        
        # Create success status message
        status_message = html.Div([
            html.P(f"Technical analysis complete for {symbol}:", className="bank-text"),
            html.Div([
                html.Span(f"Price: ${current_price:.2f} ", className="bank-value"),
                html.Span(
                    f"({price_change:.2f}%)", 
                    className=f"{'positive' if price_change > 0 else 'negative' if price_change < 0 else 'neutral'}"
                )
            ]),
            html.Div([
                html.Span("TSMN Signal: ", className="bank-label"),
                html.Span(
                    f"{tsmn_signal.upper()} ({tsmn_value:.1f})", 
                    className=f"{'positive' if tsmn_value > 0 else 'negative' if tsmn_value < 0 else 'neutral'}"
                )
            ]),
            html.P(f"Volatility: {volatility:.1f}%", className="bank-text")
        ])
        
        # Save data locally - ensure directory exists
        aiplus_cache_dir = os.path.join(config.DATA_DIR, "aiplus_cache")
        os.makedirs(aiplus_cache_dir, exist_ok=True)
        
        # Save technical data to local cache
        tech_cache_file = os.path.join(aiplus_cache_dir, f"{symbol}_tech.json")
        cache_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": technical_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(tech_cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Check readiness
        update_data_readiness(symbol)
        
        return status_message, "bank-status status-ready"
        
    except Exception as e:
        error_message = f"Error fetching technical data: {str(e)}"
        print(error_message)
        return html.P(error_message, className="bank-text"), "bank-status status-error"


# News data fetch callback
@callback(
    Output('news-data-status', 'children'),
    Output('news-data-status', 'className'),
    Input('fetch-news-button', 'n_clicks'),
    State('aiplus-stock-dropdown', 'value'),
    State('news-timeframe-dropdown', 'value'),
    State('tabs', 'value'),
    prevent_initial_call=True
)
def fetch_news_data(n_clicks, symbol, days_lookback, tab):
    """
    Fetch news data for AI+ analysis.
    
    Args:
        n_clicks (int): Number of button clicks
        symbol (str): Selected stock symbol
        days_lookback (int): Number of days to look back
        tab (str): Current tab
        
    Returns:
        tuple: (status_message, status_class)
    """
    if n_clicks == 0 or tab != 'aiplus' or not symbol:
        return "", "bank-status"
    
    try:
        # Show loading status
        status_message = html.Div([
            html.P("Fetching news data...", className="bank-text"),
            html.Div(className="bank-loading"),
        ])
        
        # Force refresh to get newest data
        sentiment_data = get_aiplus_sentiment(symbol, days_lookback, force_refresh=True)
        
        if 'error' in sentiment_data:
            return html.P(f"Error: {sentiment_data['error']}", className="bank-text"), "bank-status status-error"
        
        # Extract key info for display
        company_name = sentiment_data.get('company_name', symbol)
        sentiment = sentiment_data.get('sentiment', 'neutral')
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        news_count = sentiment_data.get('news_count', 0)
        
        # Get key developments
        key_developments = sentiment_data.get('key_developments', [])
        development_text = ""
        
        if key_developments:
            development_items = []
            for dev in key_developments[:2]:  # Show top 2 developments
                headline = dev.get('headline', '')
                date = dev.get('date', '')
                sentiment = dev.get('sentiment', 'neutral')
                
                # Format as list item with sentiment color
                item = html.Li([
                    html.Span(f"{date}: ", className="bank-date"),
                    html.Span(
                        headline, 
                        className=f"bank-news-headline {sentiment}"
                    )
                ])
                development_items.append(item)
            
            development_text = html.Ul(development_items, className="bank-news-list")
        
        # Create success status message
        status_message = html.Div([
            html.P(f"News analysis complete for {company_name}:", className="bank-text"),
            html.Div([
                html.Span("Sentiment: ", className="bank-label"),
                html.Span(
                    f"{sentiment.upper()} ({sentiment_score:.1f})", 
                    className=f"{'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'}"
                )
            ]),
            html.P(f"Analyzed {news_count} news items from past {days_lookback} days", className="bank-text"),
            html.Div([
                html.P("Key Developments:", className="bank-label") if key_developments else None,
                development_text
            ]) if key_developments else None
        ])
        
        # Save data locally - ensure directory exists
        aiplus_cache_dir = os.path.join(config.DATA_DIR, "aiplus_cache")
        os.makedirs(aiplus_cache_dir, exist_ok=True)
        
        # Save news data to local cache
        news_cache_file = os.path.join(aiplus_cache_dir, f"{symbol}_news.json")
        cache_data = {
            "symbol": symbol,
            "days_lookback": days_lookback,
            "data": sentiment_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(news_cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Check readiness
        update_data_readiness(symbol)
        
        return status_message, "bank-status status-ready"
        
    except Exception as e:
        error_message = f"Error fetching news data: {str(e)}"
        print(error_message)
        return html.P(error_message, className="bank-text"), "bank-status status-error"