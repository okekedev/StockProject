"""
Updated Research Department that uses yfinance for stock news.
"""
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
import json
import config
from datetime import datetime, timedelta

# Import the stock news module
from modules.stock_news import get_stock_analysis

# Function to get layout
def get_layout():
    """Generate layout for the Research Department."""
    return html.Div([
        html.Div([
            html.H4("Intelligence Research Department", className="bank-section-title")
        ], className="bank-card-header"),
        
        html.Div([
            html.Div([
                html.Div(className="bank-emblem", style={"float": "right", "width": "40px", "height": "40px"}),
                html.H5("MARKET INTELLIGENCE BULLETIN", className="bank-forecast-title"),
                html.P(
                    f"Report Date: {datetime.now().strftime('%B %d, %Y')}",
                    className="bank-forecast-date"
                ),
            ], className="bank-forecast-header"),
            
            html.P(
                "Our Intelligence Research Department provides comprehensive analysis of market news "
                "and sentiment for selected investment opportunities. This department examines "
                "public information sources to validate technical predictions with qualitative insights.",
                className="bank-text"
            ),
            
            html.Div([
                html.Div([
                    html.Span("CONFIDENTIAL", className="confidential-stamp")
                ], className="stamp-container"),
                
                html.P(
                    "Select securities with positive forecasts for intelligence gathering:",
                    className="bank-label"
                ),
                
                # Dropdown to select stocks
                dcc.Dropdown(
                    id='research-stocks-dropdown',
                    multi=True,
                    placeholder="Select securities for intelligence analysis...",
                    className="bank-dropdown"
                ),
                
                html.Button(
                    "Request Intelligence Report", 
                    id='research-button', 
                    n_clicks=0, 
                    className="bank-button",
                    style={"margin-top": "20px", "margin-bottom": "20px"}
                ),
                
                # Loading indicator
                dcc.Loading(
                    id="research-loading",
                    type="default",
                    children=html.Div(id="research-output")
                )
            ], className="bank-card-content")
        ], className="bank-card"),
    ], id='research-content', className="bank-section")

# Callback to populate dropdown with stocks that have positive predictions
@callback(
    Output('research-stocks-dropdown', 'options'),
    Input('tabs', 'value')
)
def populate_research_dropdown(tab):
    """
    Populate the stock dropdown with securities that have positive predictions.
    
    Args:
        tab (str): Current active tab.
        
    Returns:
        list: Dropdown options.
    """
    if tab != 'research':
        return []
    
    try:
        # Load the prediction results
        predictions_file = config.PREDICTION_OUTPUT
        if not os.path.exists(predictions_file):
            return []
        
        predictions_df = pd.read_csv(predictions_file)
        
        # Filter for positive predictions
        positive_predictions = predictions_df[predictions_df['percent_predicted_change'] > 0]
        
        if positive_predictions.empty:
            return []
        
        # Create dropdown options with labels showing symbol and predicted change
        options = [
            {
                'label': f"{row['Symbol']} (+{row['percent_predicted_change']:.2f}%)",
                'value': row['Symbol']
            }
            for _, row in positive_predictions.iterrows()
        ]
        
        return options
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return []

# Callback to handle research button click
@callback(
    Output('research-output', 'children'),
    Input('research-button', 'n_clicks'),
    State('research-stocks-dropdown', 'value'),
    State('tabs', 'value')
)
def generate_research_report(n_clicks, selected_stocks, tab):
    """
    Generate a research report for selected stocks.
    
    Args:
        n_clicks (int): Number of button clicks.
        selected_stocks (list): Selected stock symbols.
        tab (str): Current active tab.
        
    Returns:
        html component: Research report.
    """
    if n_clicks == 0 or tab != 'research' or not selected_stocks:
        return html.P("Select securities and request intelligence to generate a report.", className="bank-text")
    
    # Create a vintage-style research report
    report_sections = []
    
    # Process each selected stock
    for i, symbol in enumerate(selected_stocks):
        # Get stock analysis using yahoo finance
        news_data = get_stock_analysis(symbol)
        
        # Create a report section for this stock
        section = html.Div([
            html.Div([
                html.H5(f"SECURITY: {symbol}", className="research-stock-title"),
                
                # Sentiment indicator
                html.Div([
                    html.Span("MARKET SENTIMENT:", className="sentiment-label"),
                    html.Span(
                        news_data.get('sentiment', 'Unknown').upper(),
                        className=f"sentiment-value {news_data.get('sentiment', 'unknown').lower()}-sentiment"
                    )
                ], className="sentiment-indicator"),
                
                # News items
                html.Div([
                    html.H6("INTELLIGENCE FINDINGS:", className="research-subtitle"),
                    html.Ul([
                        html.Li([
                            html.Span(f"{item.get('date', 'N/A')}: ", className="news-date"),
                            html.Span(item.get('headline', 'No headline available'), className="news-headline"),
                            html.P(item.get('summary', ''), className="news-summary"),
                            html.A("Source", href=item.get('url', '#'), target="_blank", className="news-link")
                        ], className="news-item")
                        for item in news_data.get('news_items', [])
                    ], className="news-list") if news_data.get('news_items') else html.P("No significant news items found.", className="bank-text")
                ], className="news-section"),
                
                # Sentiment reasoning
                html.Div([
                    html.H6("SENTIMENT ANALYSIS:", className="research-subtitle"),
                    html.P(news_data.get('sentiment_reasoning', 'No sentiment analysis available.'), className="bank-text")
                ], className="sentiment-section"),
                
                # Impact summary
                html.Div([
                    html.H6("MARKET IMPACT ASSESSMENT:", className="research-subtitle"),
                    html.P(news_data.get('impact_summary', 'No impact assessment available.'), className="bank-text")
                ], className="impact-section"),
                
                # Department stamp
                html.Div([
                    html.Div([
                        html.Span("VERIFIED", className="verified-stamp")
                    ], className="stamp-container")
                ], className="stamp-section")
            ], className="research-stock-section")
        ], className="research-section slide-in", style={"--animation-order": i})
        
        report_sections.append(section)
    
    # Create the full report
    report = html.Div([
        html.Div([
            html.H5("MARKET INTELLIGENCE REPORT", className="report-title"),
            html.P(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", className="report-date"),
            html.P("CLASSIFICATION: CONFIDENTIAL - FOR INTERNAL USE ONLY", className="report-classification")
        ], className="report-header"),
        
        html.Div(report_sections, className="report-body"),
        
        html.Div([
            html.P(
                "DISCLAIMER: This intelligence report contains information gathered from public sources. "
                "The Research Department provides this information for consideration alongside technical analysis, "
                "but does not guarantee accuracy or completeness. All investment decisions remain the "
                "responsibility of the investor.",
                className="report-disclaimer"
            ),
            html.Div([
                html.Span("DEPARTMENT OF MARKET INTELLIGENCE", className="department-name"),
                html.Div(className="department-seal")
            ], className="report-footer-content")
        ], className="report-footer")
    ], className="research-report")
    
    return report