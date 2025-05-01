"""
Callback functions for the AI+ tab in the Stock Analysis Dashboard.
"""
from dash import html, callback, Output, Input, State
import pandas as pd
import os
import time
import json
import config

# Import AI+ modules
from modules.aiplus_technical import get_aiplus_technical_data
from modules.aiplus_sentiment import get_aiplus_sentiment
from modules.aiplus_predictor import generate_aiplus_prediction, get_aiplus_performance

# Constants
AIPLUS_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_cache")
os.makedirs(AIPLUS_CACHE_DIR, exist_ok=True)


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
        
        # Save data to cache
        cache_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_tech.json")
        with open(cache_file, 'w') as f:
            json.dump({
                "symbol": symbol,
                "timeframe": timeframe,
                "data": technical_data,
                "timestamp": pd.Timestamp.now().isoformat()
            }, f)
        
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
        
        # Save data to cache
        cache_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_news.json")
        with open(cache_file, 'w') as f:
            json.dump({
                "symbol": symbol,
                "days_lookback": days_lookback,
                "data": sentiment_data,
                "timestamp": pd.Timestamp.now().isoformat()
            }, f)
        
        # Check readiness
        update_data_readiness(symbol)
        
        return status_message, "bank-status status-ready"
        
    except Exception as e:
        error_message = f"Error fetching news data: {str(e)}"
        print(error_message)
        return html.P(error_message, className="bank-text"), "bank-status status-error"


def update_data_readiness(symbol):
    """Update the data readiness status based on available data."""
    try:
        # Check if both technical and news data are available
        tech_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_tech.json")
        news_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_news.json")
        
        tech_ready = os.path.exists(tech_file)
        news_ready = os.path.exists(news_file)
        
        # Update global readiness state
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
            
        # Save readiness state
        with open(os.path.join(AIPLUS_CACHE_DIR, "readiness.json"), 'w') as f:
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
        # Try to load readiness state
        readiness_file = os.path.join(AIPLUS_CACHE_DIR, "readiness.json")
        
        if os.path.exists(readiness_file):
            with open(readiness_file, 'r') as f:
                readiness = json.load(f)
            
            # Check if readiness is for current symbol
            if readiness.get('symbol') != symbol:
                # Different symbol, check data files
                tech_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_tech.json")
                news_file = os.path.join(AIPLUS_CACHE_DIR, f"{symbol}_news.json")
                
                tech_ready = os.path.exists(tech_file)
                news_ready = os.path.exists(news_file)
                
                if tech_ready and news_ready:
                    return html.Div([
                        html.P("All data ready for AI analysis!", className="bank-success"),
                        html.P(f"Technical and news data available for {symbol}. You can now proceed to the 'Think & Predict' tab.", className="bank-text")
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
                    html.P(f"Technical and news data available for {symbol}. You can now proceed to the 'Think & Predict' tab.", className="bank-text")
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


# AI Analysis button callback
@callback(
    Output('analysis-output', 'children'),
    Input('begin-analysis-button', 'n_clicks'),
    State('aiplus-stock-dropdown', 'value'),
    State('tech-timeframe-dropdown', 'value'),
    State('news-timeframe-dropdown', 'value'),
    State('tabs', 'value'),
    prevent_initial_call=True
)
def generate_ai_analysis(n_clicks, symbol, tech_timeframe, news_days, tab):
    """
    Generate AI analysis based on collected data.
    
    Args:
        n_clicks (int): Number of button clicks
        symbol (str): Selected stock symbol
        tech_timeframe (str): Selected technical timeframe
        news_days (int): Number of days for news analysis
        tab (str): Current tab
        
    Returns:
        html component: Analysis output
    """
    if n_clicks == 0 or tab != 'aiplus' or not symbol:
        return ""
    
    try:
        # Check if data is ready
        readiness_file = os.path.join(AIPLUS_CACHE_DIR, "readiness.json")
        
        if os.path.exists(readiness_file):
            with open(readiness_file, 'r') as f:
                readiness = json.load(f)
            
            if not readiness.get('ready', False) or readiness.get('symbol') != symbol:
                return html.Div([
                    html.P("Data not ready for analysis.", className="bank-error"),
                    html.P(f"Please ensure both technical and news data are fetched for {symbol} before generating analysis.", className="bank-text")
                ], className="bank-notification bank-error-notification")
        else:
            return html.Div([
                html.P("Data not ready for analysis.", className="bank-error"),
                html.P(f"Please ensure both technical and news data are fetched for {symbol} before generating analysis.", className="bank-text")
            ], className="bank-notification bank-error-notification")
        
        # Generate AI analysis
        prediction = generate_aiplus_prediction(symbol, tech_timeframe, news_days, force_refresh=True)
        
        if 'error' in prediction:
            return html.Div([
                html.P("Error generating analysis.", className="bank-error"),
                html.P(prediction['error'], className="bank-text")
            ], className="bank-notification bank-error-notification")
        
        # Extract data for display
        company_name = prediction.get('company_name', symbol)
        current_price = prediction.get('current_price', 0)
        
        pred_data = prediction.get('prediction', {})
        signal = pred_data.get('signal', 'neutral')
        pred_change = pred_data.get('predicted_change_pct', 0)
        confidence = pred_data.get('confidence', 'low')
        target_low = pred_data.get('target_price_low', 0)
        target_mid = pred_data.get('target_price_mid', 0)
        target_high = pred_data.get('target_price_high', 0)
        recommendation = pred_data.get('recommendation', 'Hold')
        
        tech_contrib = prediction.get('technical_contribution', {})
        tech_weight = tech_contrib.get('weight', 0)
        tech_signal = tech_contrib.get('signal', 'neutral')
        tsmn_value = tech_contrib.get('tsmn_value', 0)
        
        sent_contrib = prediction.get('sentiment_contribution', {})
        sent_weight = sent_contrib.get('weight', 0)
        sentiment = sent_contrib.get('sentiment', 'neutral')
        sentiment_score = sent_contrib.get('sentiment_score', 0)
        
        analysis = prediction.get('analysis', {})
        tech_summary = analysis.get('technical_summary', '')
        sent_summary = analysis.get('sentiment_summary', '')
        enhanced_analysis = analysis.get('enhanced_analysis', '')
        
        # Format the prediction report
        return html.Div([
            # Header with company and current price
            html.Div([
                html.Div(className="bank-emblem", style={"float": "right", "width": "40px", "height": "40px"}),
                html.H4(f"AI+ ANALYSIS: {company_name} ({symbol})", className="bank-report-title"),
                html.P(f"Analysis Date: {pd.Timestamp.now().strftime('%B %d, %Y')}", className="bank-report-date"),
                html.Div([
                    html.Span("Current Price: ", className="bank-label"),
                    html.Span(f"${current_price:.2f}", className="bank-value")
                ]),
                html.Div([
                    html.Span("CONFIDENTIAL", className="confidential-stamp")
                ], className="stamp-container"),
            ], className="bank-report-header"),
            
            # Prediction summary
            html.Div([
                html.H5("PREDICTION SUMMARY", className="bank-section-heading"),
                html.Div([
                    html.Div([
                        html.P("Signal:", className="bank-label"),
                        html.P(
                            signal.upper(), 
                            className=f"bank-value large-value {'positive' if signal == 'bullish' else 'negative' if signal == 'bearish' else 'neutral'}"
                        )
                    ], className="prediction-signal"),
                    html.Div([
                        html.P("Predicted Change:", className="bank-label"),
                        html.P(
                            f"{pred_change:.2f}%", 
                            className=f"bank-value large-value {'positive' if pred_change > 0 else 'negative' if pred_change < 0 else 'neutral'}"
                        )
                    ], className="prediction-change"),
                    html.Div([
                        html.P("Confidence:", className="bank-label"),
                        html.P(
                            confidence.upper(),
                            className=f"bank-value {'high-confidence' if confidence == 'high' else 'medium-confidence' if confidence == 'medium' else 'low-confidence'}"
                        )
                    ], className="prediction-confidence"),
                ], className="prediction-summary-row"),
                html.Div([
                    html.P("Target Price Range:", className="bank-label"),
                    html.Div([
                        html.Span(f"${target_low:.2f}", className="target-low"),
                        html.Span(" → ", className="target-arrow"),
                        html.Span(f"${target_mid:.2f}", className="target-mid"),
                        html.Span(" → ", className="target-arrow"),
                        html.Span(f"${target_high:.2f}", className="target-high")
                    ], className="target-range")
                ], className="prediction-targets"),
                html.Div([
                    html.P("Recommendation:", className="bank-label"),
                    html.P(
                        recommendation,
                        className=f"bank-recommendation {'positive' if 'buy' in recommendation.lower() else 'negative' if 'sell' in recommendation.lower() else 'neutral'}"
                    )
                ], className="prediction-recommendation")
            ], className="bank-prediction-summary bank-card"),
            
            # Analysis details
            html.Div([
                html.H5("ANALYSIS DETAILS", className="bank-section-heading"),
                
                # Technical and sentiment contribution visualization
                html.Div([
                    html.H6("Signal Contributions", className="bank-subsection-heading"),
                    html.Div([
                        html.Div([
                            html.P(f"Technical ({tech_weight*100:.0f}%)", className="bank-label"),
                            html.Div([
                                html.Div(
                                    className="contribution-bar technical",
                                    style={"width": f"{abs(tsmn_value)/2:.1f}%", "marginLeft": "50%" if tsmn_value > 0 else "auto", "marginRight": "50%" if tsmn_value < 0 else "auto"}
                                )
                            ], className="contribution-container"),
                            html.P(
                                f"{tech_signal.upper()} ({tsmn_value:.1f})", 
                                className=f"{'positive' if tsmn_value > 0 else 'negative' if tsmn_value < 0 else 'neutral'}"
                            )
                        ], className="contribution-row"),
                        html.Div([
                            html.P(f"Sentiment ({sent_weight*100:.0f}%)", className="bank-label"),
                            html.Div([
                                html.Div(
                                    className="contribution-bar sentiment",
                                    style={"width": f"{abs(sentiment_score)/2:.1f}%", "marginLeft": "50%" if sentiment_score > 0 else "auto", "marginRight": "50%" if sentiment_score < 0 else "auto"}
                                )
                            ], className="contribution-container"),
                            html.P(
                                f"{sentiment.upper()} ({sentiment_score:.1f})", 
                                className=f"{'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'}"
                            )
                        ], className="contribution-row")
                    ], className="signal-contributions")
                ]),
                
                # Enhanced analysis
                html.Div([
                    html.H6("AI+ Enhanced Analysis", className="bank-subsection-heading"),
                    html.P(enhanced_analysis, className="bank-text enhanced-analysis"),
                ], className="enhanced-analysis-section"),
                
                # Technical analysis summary
                html.Div([
                    html.H6("Technical Analysis", className="bank-subsection-heading"),
                    html.P(tech_summary, className="bank-text tech-summary"),
                ], className="technical-summary-section"),
                
                # Sentiment analysis summary
                html.Div([
                    html.H6("Sentiment Analysis", className="bank-subsection-heading"),
                    html.P(sent_summary, className="bank-text sent-summary"),
                ], className="sentiment-summary-section"),
                
                # Key developments
                html.Div([
                    html.H6("Key Developments", className="bank-subsection-heading"),
                    html.Ul([
                        html.Li([
                            html.Span(f"{dev.get('date', '')}: ", className="bank-date"),
                            html.Span(
                                dev.get('headline', ''), 
                                className=f"bank-news-headline {dev.get('sentiment', 'neutral')}"
                            )
                        ]) for dev in sent_contrib.get('key_developments', [])[:5]
                    ], className="bank-news-list") if sent_contrib.get('key_developments', []) else html.P("No significant developments found.", className="bank-text")
                ], className="key-developments-section"),
                
                # Key indicators
                html.Div([
                    html.H6("Key Technical Indicators", className="bank-subsection-heading"),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Indicator"),
                                html.Th("Value"),
                                html.Th("Interpretation")
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td("TSMN"),
                                html.Td(f"{tsmn_value:.1f}"),
                                html.Td(
                                    "Bullish" if tsmn_value > 20 else "Bearish" if tsmn_value < -20 else "Neutral",
                                    className=f"{'positive' if tsmn_value > 20 else 'negative' if tsmn_value < -20 else 'neutral'}"
                                )
                            ]),
                            *[
                                html.Tr([
                                    html.Td(indicator),
                                    html.Td(f"{value:.1f}" if isinstance(value, (int, float)) else str(value)),
                                    html.Td(get_indicator_interpretation(indicator, value))
                                ]) for indicator, value in tech_contrib.get('key_indicators', {}).items()
                            ]
                        ])
                    ], className="bank-table")
                ], className="key-indicators-section")
            ], className="bank-analysis-details bank-card"),
            
            # Footer
            html.Div([
                html.P(
                    "DISCLAIMER: This AI+ analysis is provided for informational purposes only. "
                    "All investment decisions should be made with appropriate consideration of risk "
                    "tolerance and financial objectives. Past performance is not indicative of future results.",
                    className="bank-disclaimer"
                ),
                html.P(
                    f"Generated by AI+ Advanced Prediction Department on {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')}",
                    className="bank-generation-info"
                )
            ], className="bank-report-footer")
        ], className="bank-ai-report slide-in")
        
    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        return html.Div([
            html.P("Error generating AI analysis.", className="bank-error"),
            html.P(str(e), className="bank-text")
        ], className="bank-notification bank-error-notification")


# Helper function to get indicator interpretation
def get_indicator_interpretation(indicator, value):
    """
    Get interpretation of technical indicator value.
    
    Args:
        indicator (str): Indicator name
        value: Indicator value
        
    Returns:
        str: Interpretation
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


# Performance metrics callback
@callback(
    Output('performance-metrics', 'children'),
    Input('tabs', 'value')
)
def update_performance_metrics(tab):
    """
    Update performance metrics display.
    
    Args:
        tab (str): Current tab
        
    Returns:
        html component: Performance metrics
    """
    if tab != 'aiplus':
        return ""
    
    try:
        # Get performance metrics
        metrics = get_aiplus_performance()
        
        if 'error' in metrics:
            return html.Div([
                html.P("No performance data available.", className="bank-warning"),
                html.P("AI+ predictions must be evaluated against actual market performance to generate metrics.", className="bank-text")
            ], className="bank-notification bank-warning-notification")
        
        # Extract overall metrics
        overall = metrics.get('overall', {})
        total_predictions = overall.get('total_predictions', 0)
        correct_predictions = overall.get('correct_predictions', 0)
        accuracy = overall.get('accuracy', 0)
        avg_error = overall.get('average_error', 0)
        
        # Format the performance report
        if total_predictions > 0:
            return html.Div([
                html.Div([
                    html.H5("PERFORMANCE METRICS", className="bank-section-heading"),
                    html.Div([
                        html.Div([
                            html.P("Overall Accuracy:", className="bank-label"),
                            html.P(
                                f"{accuracy:.1f}%", 
                                className=f"bank-value large-value {'high-accuracy' if accuracy >= 70 else 'medium-accuracy' if accuracy >= 55 else 'low-accuracy'}"
                            )
                        ], className="metric-box"),
                        html.Div([
                            html.P("Predictions:", className="bank-label"),
                            html.P(
                                f"{correct_predictions} / {total_predictions}",
                                className="bank-value"
                            )
                        ], className="metric-box"),
                        html.Div([
                            html.P("Avg. Error:", className="bank-label"),
                            html.P(
                                f"{avg_error:.2f}%",
                                className="bank-value"
                            )
                        ], className="metric-box")
                    ], className="metrics-row"),
                    
                    # By confidence level
                    html.Div([
                        html.H6("Performance by Confidence Level", className="bank-subsection-heading"),
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Confidence"),
                                    html.Th("Accuracy"),
                                    html.Th("Predictions")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(conf.upper()),
                                    html.Td(
                                        f"{data.get('accuracy', 0):.1f}%",
                                        className=f"{'high-accuracy' if data.get('accuracy', 0) >= 70 else 'medium-accuracy' if data.get('accuracy', 0) >= 55 else 'low-accuracy'}"
                                    ),
                                    html.Td(f"{data.get('correct_predictions', 0)} / {data.get('total_predictions', 0)}")
                                ]) for conf, data in metrics.get('by_confidence', {}).items()
                            ])
                        ], className="bank-table")
                    ]),
                    
                    # Recent predictions
                    html.Div([
                        html.H6("Recent Predictions", className="bank-subsection-heading"),
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Symbol"),
                                    html.Th("Date"),
                                    html.Th("Prediction"),
                                    html.Th("Actual"),
                                    html.Th("Result")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(pred.get('symbol', '')),
                                    html.Td(pred.get('prediction_date', '')),
                                    html.Td(
                                        f"{pred.get('predicted_change_pct', 0):.2f}%",
                                        className=f"{'positive' if pred.get('predicted_change_pct', 0) > 0 else 'negative' if pred.get('predicted_change_pct', 0) < 0 else 'neutral'}"
                                    ),
                                    html.Td(
                                        f"{pred.get('actual_change_pct', 0):.2f}%",
                                        className=f"{'positive' if pred.get('actual_change_pct', 0) > 0 else 'negative' if pred.get('actual_change_pct', 0) < 0 else 'neutral'}"
                                    ),
                                    html.Td(
                                        "✓" if pred.get('correct_direction', False) else "✗",
                                        className=f"{'success' if pred.get('correct_direction', False) else 'failure'}"
                                    )
                                ]) for pred in metrics.get('recent_predictions', [])[:10]
                            ])
                        ], className="bank-table")
                    ])
                ], className="bank-performance-metrics bank-card")
            ], className="bank-metrics-container slide-in")
        else:
            return html.Div([
                html.P("No performance data available yet.", className="bank-warning"),
                html.P("AI+ prediction performance will be shown here once predictions have been evaluated against actual market performance.", className="bank-text")
            ], className="bank-notification bank-warning-notification")
            
    except Exception as e:
        print(f"Error updating performance metrics: {e}")
        return html.Div([
            html.P("Error loading performance metrics.", className="bank-error"),
            html.P(str(e), className="bank-text")
        ], className="bank-notification bank-error-notification")