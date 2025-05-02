"""
Utility functions for AI+ callbacks.
"""
from dash import html
import os
import json
import config

# Constants
AIPLUS_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_cache")
os.makedirs(AIPLUS_CACHE_DIR, exist_ok=True)

# Helper function to get indicator interpretation
def get_indicator_interpretation(indicator, value):
    """
    Get interpretation of technical indicator value.
    
    Args:
        indicator (str): Indicator name
        value: Indicator value
        
    Returns:
        html component: Interpretation with appropriate styling
    """
    try:
        if indicator == 'RSI_14':
            if value > 70:
                return html.Span("Overbought", className="negative")
            elif value < 30:
                return html.Span("Oversold", className="positive")
            elif value > 60:
                return html.Span("Bullish", className="positive")
            elif value < 40:
                return html.Span("Bearish", className="negative")
            else:
                return html.Span("Neutral", className="neutral")
        
        elif 'MACD' in indicator:
            if indicator == 'MACD':
                if value > 0:
                    return html.Span("Bullish", className="positive")
                else:
                    return html.Span("Bearish", className="negative")
            elif indicator == 'MACD_Signal':
                return html.Span("Signal Line", className="neutral")
            elif indicator == 'MACD_Histogram':
                if value > 0:
                    return html.Span("Bullish Momentum", className="positive")
                else:
                    return html.Span("Bearish Momentum", className="negative")
        
        elif indicator == 'BB_Percent':
            if value > 1:
                return html.Span("Above Upper Band", className="negative")
            elif value < 0:
                return html.Span("Below Lower Band", className="positive")
            elif value > 0.8:
                return html.Span("Near Upper Band", className="negative")
            elif value < 0.2:
                return html.Span("Near Lower Band", className="positive")
            else:
                return html.Span("Within Bands", className="neutral")
        
        elif 'MA_' in indicator:
            return html.Span("Moving Average", className="neutral")
        
        elif indicator == 'ADX':
            if value > 30:
                return html.Span("Strong Trend", className="positive")
            elif value > 20:
                return html.Span("Developing Trend", className="neutral")
            else:
                return html.Span("No Trend", className="negative")
        
        else:
            return html.Span("--", className="neutral")
    
    except Exception as e:
        print(f"Error interpreting indicator {indicator}: {e}")
        return html.Span("--", className="neutral")


# Helper function to get human-readable horizon display
def get_horizon_display(horizon):
    """
    Convert prediction horizon code to display text.
    
    Args:
        horizon (str): Prediction horizon code
        
    Returns:
        str: Human-readable horizon text
    """
    horizon_display = {
        '1d': 'Next Day',
        '2d': 'Next 2 Days',
        '1w': 'Next Week',
        '1mo': 'Next Month'
    }
    return horizon_display.get(horizon, horizon)


# Function to save data to cache
def save_to_cache(filename, data):
    """
    Save data to the cache directory.
    
    Args:
        filename (str): Cache filename
        data (dict): Data to cache
        
    Returns:
        bool: Success status
    """
    try:
        filepath = os.path.join(AIPLUS_CACHE_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving to cache: {e}")
        return False


# Function to load data from cache
def load_from_cache(filename):
    """
    Load data from the cache directory.
    
    Args:
        filename (str): Cache filename
        
    Returns:
        dict: Cached data or None if not found
    """
    try:
        filepath = os.path.join(AIPLUS_CACHE_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading from cache: {e}")
        return None