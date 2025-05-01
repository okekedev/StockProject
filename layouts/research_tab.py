"""
Updated Research Department that uses News API for stock news with expandable dropdowns.
"""
from dash import html, dcc, callback, Output, Input, State, ALL, MATCH
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
    Generate a research report for selected stocks with expandable dropdowns.
    
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
    
    # Process each selected stock - use past 3 days of news
    for i, symbol in enumerate(selected_stocks):
        # Get stock analysis using News API and Gemini
        news_data = get_stock_analysis(symbol, days_lookback=3)
        
        # Determine sentiment class for styling
        sentiment = news_data.get('sentiment', 'unknown').lower()
        sentiment_class = {
            'positive': 'positive-sentiment',
            'negative': 'negative-sentiment',
            'neutral': 'neutral-sentiment'
        }.get(sentiment, 'unknown-sentiment')
        
        # Create a collapsible section for this stock
        section = html.Div([
            # Header - always visible with improved layout
            html.Div([
                html.Div([
                    html.H5(f"SECURITY: {symbol}", style={"display": "inline-block", "margin": "0"}),
                    html.Span(
                        news_data.get('sentiment', 'Unknown').upper(),
                        className=f"sentiment-badge {sentiment_class}",
                        style={
                            "padding": "4px 10px", 
                            "borderRadius": "4px",
                            "marginLeft": "15px"
                        }
                    )
                ], style={"display": "flex", "alignItems": "center"}),
                
                # Improved expand/collapse button
                html.Button(
                    "▼",
                    id=f'stock-expand-button-{i}',  # Fixed ID with index
                    n_clicks=0,
                    className="bank-expand-button",
                    style={
                        "position": "absolute",
                        "right": "15px",
                        "top": "15px",
                        "backgroundColor": "transparent",
                        "border": "1px solid var(--bank-border)",
                        "borderRadius": "4px",
                        "fontSize": "16px",
                        "cursor": "pointer",
                        "padding": "2px 10px",
                        "zIndex": "10"
                    }
                )
            ], className="stock-header", style={
                "backgroundColor": "var(--bank-cream)", 
                "padding": "15px", 
                "borderRadius": "4px 4px 0 0", 
                "borderBottom": "1px solid var(--bank-border)",
                "position": "relative",
                "cursor": "pointer"
            }),
            
            # Collapsible content with fixed ID
            html.Div([
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
                ], className="news-section")
            ], 
            id=f'stock-content-{i}',  # Fixed ID with index
            className="stock-content",
            style={"display": "none", "padding": "15px", "backgroundColor": "white", "borderRadius": "0 0 4px 4px", "marginBottom": "20px", "borderTop": "none"}
            )
        ], className="research-section slide-in", style={"--animation-order": i, "marginBottom": "15px", "border": "1px solid var(--bank-border)", "borderRadius": "4px"})
        
        report_sections.append(section)
    
    # Create the full report with accordion-style dropdowns
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

# Define callbacks for a fixed number of potential stocks (10)
for i in range(10):
    @callback(
        Output(f'stock-content-{i}', 'style'),
        Output(f'stock-expand-button-{i}', 'children'),
        Input(f'stock-expand-button-{i}', 'n_clicks'),
        prevent_initial_call=True
    )
    def toggle_content(n_clicks, i=i):  # Capture i in closure
        if n_clicks and n_clicks % 2 == 1:  # Odd clicks - show content
            return {
                "display": "block", 
                "padding": "15px", 
                "backgroundColor": "white", 
                "borderRadius": "0 0 4px 4px", 
                "marginBottom": "20px", 
                "borderTop": "none"
            }, "▲"
        else:  # Even clicks or initial state - hide content
            return {
                "display": "none", 
                "padding": "15px", 
                "backgroundColor": "white", 
                "borderRadius": "0 0 4px 4px", 
                "marginBottom": "20px", 
                "borderTop": "none"
            }, "▼"