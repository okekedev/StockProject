"""
Status update callbacks for the AI+ tab with local storage.
"""
from dash import html, callback, Output, Input
import os
import json
import pandas as pd
import config


def update_data_readiness(symbol):
    """
    Update the data readiness status based on available data.
    
    Args:
        symbol (str): Stock symbol
    """
    try:
        # Ensure the cache directory exists
        aiplus_cache_dir = os.path.join(config.DATA_DIR, "aiplus_cache")
        os.makedirs(aiplus_cache_dir, exist_ok=True)
        
        # Check if both technical and news data are available locally
        tech_file = os.path.join(aiplus_cache_dir, f"{symbol}_tech.json")
        news_file = os.path.join(aiplus_cache_dir, f"{symbol}_news.json")
        
        tech_ready = os.path.exists(tech_file)
        news_ready = os.path.exists(news_file)
        
        # Create readiness state file
        readiness_file = os.path.join(aiplus_cache_dir, "readiness.json")
        
        # Update readiness state
        if tech_ready and news_ready:
            readiness_state = {
                "ready": True,
                "tech_ready": True,
                "news_ready": True,
                "symbol": symbol,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            readiness_state = {
                "ready": False,
                "tech_ready": tech_ready,
                "news_ready": news_ready,
                "symbol": symbol,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        # Save readiness state to local storage
        with open(readiness_file, 'w') as f:
            json.dump(readiness_state, f)
            
    except Exception as e:
        print(f"Error updating data readiness: {e}")


# Data readiness status callback
@callback(
    Output('data-readiness-status', 'children'),
    Output('data-readiness-status', 'className'),
    Input('tech-data-status', 'children'),
    Input('news-data-status', 'children'),
    Input('aiplus-stock-dropdown', 'value'),
)
def update_readiness_status(tech_status, news_status, symbol):
    """
    Update the data readiness status.
    
    Args:
        tech_status: Technical data status
        news_status: News data status
        symbol (str): Selected stock symbol
        
    Returns:
        tuple: (readiness_message, readiness_class)
    """
    if not symbol:
        return html.P("Select a stock symbol and fetch data to begin.", className="bank-text"), "bank-status-large"
    
    try:
        # Set up paths for cache files
        aiplus_cache_dir = os.path.join(config.DATA_DIR, "aiplus_cache")
        os.makedirs(aiplus_cache_dir, exist_ok=True)
        
        readiness_file = os.path.join(aiplus_cache_dir, "readiness.json")
        
        # Try to load readiness state from local file
        if os.path.exists(readiness_file):
            with open(readiness_file, 'r') as f:
                readiness = json.load(f)
                
            # Check if readiness is for current symbol
            if readiness.get('symbol') != symbol:
                # Different symbol, check data files
                tech_file = os.path.join(aiplus_cache_dir, f"{symbol}_tech.json")
                news_file = os.path.join(aiplus_cache_dir, f"{symbol}_news.json")
                
                tech_ready = os.path.exists(tech_file)
                news_ready = os.path.exists(news_file)
                
                # Update readiness status for the new symbol
                update_data_readiness(symbol)
                
                if tech_ready and news_ready:
                    return html.Div([
                        html.P("All data ready for AI analysis!", className="bank-success"),
                        html.P(f"Technical and news data available for {symbol}. You can now proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-ready"
                elif tech_ready:
                    return html.Div([
                        html.P("Technical data ready, news data needed.", className="bank-warning"),
                        html.P(f"Please fetch news data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
                elif news_ready:
                    return html.Div([
                        html.P("News data ready, technical data needed.", className="bank-warning"),
                        html.P(f"Please fetch technical data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
                else:
                    return html.Div([
                        html.P("Data collection required.", className="bank-warning"),
                        html.P(f"Please fetch both technical and news data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
            
            # Current symbol
            if readiness.get('ready', False):
                return html.Div([
                    html.P("All data ready for AI analysis!", className="bank-success"),
                    html.P(f"Technical and news data available for {symbol}. You can now proceed with AI analysis.", className="bank-text")
                ]), "bank-status-large status-ready"
            else:
                # Not ready, show what's missing
                tech_ready = readiness.get('tech_ready', False)
                news_ready = readiness.get('news_ready', False)
                
                if tech_ready:
                    return html.Div([
                        html.P("Technical data ready, news data needed.", className="bank-warning"),
                        html.P(f"Please fetch news data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
                elif news_ready:
                    return html.Div([
                        html.P("News data ready, technical data needed.", className="bank-warning"),
                        html.P(f"Please fetch technical data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
                else:
                    return html.Div([
                        html.P("Data collection required.", className="bank-warning"),
                        html.P(f"Please fetch both technical and news data for {symbol} to proceed with AI analysis.", className="bank-text")
                    ]), "bank-status-large status-pending"
        else:
            # No readiness file yet
            # Check if the files exist directly
            tech_file = os.path.join(aiplus_cache_dir, f"{symbol}_tech.json")
            news_file = os.path.join(aiplus_cache_dir, f"{symbol}_news.json")
            
            tech_ready = os.path.exists(tech_file)
            news_ready = os.path.exists(news_file)
            
            # Create an initial readiness file
            update_data_readiness(symbol)
            
            if tech_ready and news_ready:
                return html.Div([
                    html.P("All data ready for AI analysis!", className="bank-success"),
                    html.P(f"Technical and news data available for {symbol}. You can now proceed with AI analysis.", className="bank-text")
                ]), "bank-status-large status-ready"
            elif tech_ready:
                return html.Div([
                    html.P("Technical data ready, news data needed.", className="bank-warning"),
                    html.P(f"Please fetch news data for {symbol} to proceed with AI analysis.", className="bank-text")
                ]), "bank-status-large status-pending"
            elif news_ready:
                return html.Div([
                    html.P("News data ready, technical data needed.", className="bank-warning"),
                    html.P(f"Please fetch technical data for {symbol} to proceed with AI analysis.", className="bank-text")
                ]), "bank-status-large status-pending"
            else:
                return html.Div([
                    html.P("Data collection required.", className="bank-warning"),
                    html.P(f"Please fetch both technical and news data for {symbol} to proceed with AI analysis.", className="bank-text")
                ]), "bank-status-large status-pending"
            
    except Exception as e:
        print(f"Error updating readiness status: {e}")
        return html.Div([
            html.P("Error checking data readiness.", className="bank-error"),
            html.P(str(e), className="bank-text")
        ]), "bank-status-large status-error"