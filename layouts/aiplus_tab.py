"""
AI+ Prediction tab that combines technical analysis with news data using Gemini API.
"""
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
import json
import config
from datetime import datetime, timedelta
import google.generativeai as genai
from modules.stock_news import get_stock_news
from modules.data_loader import load_data

# Function to get layout
def get_layout():
    """Generate layout for the AI+ Department."""
    return html.Div([
        html.Div([
            html.H4("AI+ Advanced Prediction Department", className="bank-section-title")
        ], className="bank-card-header"),
        
        html.Div([
            html.Div([
                html.Div(className="bank-emblem", style={"float": "right", "width": "40px", "height": "40px"}),
                html.H5("ADVANCED AI MARKET PREDICTION", className="bank-forecast-title"),
                html.P(
                    f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}",
                    className="bank-forecast-date"
                ),
            ], className="bank-forecast-header"),
            
            html.P(
                "Our Advanced AI+ Department combines technical indicators with news sentiment analysis "
                "to provide deeper market insights. The system uses iterative learning to improve "
                "prediction accuracy over time.",
                className="bank-text"
            ),
            
            # AI+ Subtabs
            dcc.Tabs(
                id='aiplus-tabs',
                className='bank-subtabs',
                value='gather-data',
                children=[
                    dcc.Tab(
                        label='1. Gather Information', 
                        value='gather-data',
                        className='bank-subtab',
                        selected_className='bank-subtab--selected'
                    ),
                    dcc.Tab(
                        label='2. Think & Predict', 
                        value='think-predict',
                        className='bank-subtab',
                        selected_className='bank-subtab--selected'
                    ),
                    dcc.Tab(
                        label='3. Review Performance', 
                        value='review',
                        className='bank-subtab',
                        selected_className='bank-subtab--selected'
                    )
                ]
            ),
            
            # Content for the selected subtab
            html.Div(id='aiplus-subtab-content', className="bank-subtab-content")
            
        ], className="bank-card"),
    ], id='aiplus-content', className="bank-section")

# Callback to switch between AI+ subtabs
@callback(
    Output('aiplus-subtab-content', 'children'),
    Input('aiplus-tabs', 'value')
)
def render_aiplus_subtab(subtab):
    """Render the content for the selected AI+ subtab."""
    if subtab == 'gather-data':
        return render_gather_data_tab()
    elif subtab == 'think-predict':
        return render_think_predict_tab()
    elif subtab == 'review':
        return render_review_tab()
    
    return html.P("Select a tab to continue.", className="bank-text")

def render_gather_data_tab():
    """Render the Gather Information tab content."""
    return html.Div([
        html.Div([
            html.Span("DATA COLLECTION", className="confidential-stamp")
        ], className="stamp-container"),
        
        # Stock selection
        html.P(
            "Select a security for advanced AI analysis:",
            className="bank-label"
        ),
        dcc.Dropdown(
            id='aiplus-stock-dropdown',
            placeholder="Select a security to analyze...",
            className="bank-dropdown"
        ),
        
        # Data range selectors
        html.Div([
            html.Div([
                html.H6("Technical Data Parameters", className="bank-card-subtitle"),
                html.P("Select the timeframe for technical indicator analysis:", className="bank-text"),
                dcc.Dropdown(
                    id='tech-timeframe-dropdown',
                    options=[
                        {'label': '1 Month', 'value': '1mo'},
                        {'label': '3 Months', 'value': '3mo'},
                        {'label': '6 Months', 'value': '6mo'},
                        {'label': '1 Year', 'value': '1y'}
                    ],
                    value='1mo',
                    className="bank-dropdown"
                ),
                
                html.Button(
                    "Fetch Technical Data", 
                    id='fetch-tech-button', 
                    n_clicks=0, 
                    className="bank-button",
                    style={"marginTop": "15px"}
                ),
                
                html.Div(id='tech-data-status', className="bank-status")
            ], className="bank-card-half"),
            
            html.Div([
                html.H6("News Data Parameters", className="bank-card-subtitle"),
                html.P("Select the timeframe for news sentiment analysis:", className="bank-text"),
                dcc.Dropdown(
                    id='news-timeframe-dropdown',
                    options=[
                        {'label': '3 Days', 'value': 3},
                        {'label': '7 Days', 'value': 7},
                        {'label': '14 Days', 'value': 14},
                        {'label': '30 Days', 'value': 30}
                    ],
                    value=7,
                    className="bank-dropdown"
                ),
                
                html.Button(
                    "Fetch News Data", 
                    id='fetch-news-button', 
                    n_clicks=0, 
                    className="bank-button",
                    style={"marginTop": "15px"}
                ),
                
                html.Div(id='news-data-status', className="bank-status")
            ], className="bank-card-half")
        ], className="bank-card-row"),
        
        # Overall status
        html.Div([
            html.H6("Data Readiness Status", className="bank-card-subtitle"),
            html.Div(id='data-readiness-status', className="bank-status-large")
        ], className="bank-card-footer"),
        
    ], className="gather-data-container")

def render_think_predict_tab():
    """Render the Think & Predict tab content."""
    return html.Div([
        html.Div([
            html.Span("AI ANALYSIS", className="confidential-stamp")
        ], className="stamp-container"),
        
        html.P(
            "Generate an advanced AI prediction using both technical and news data:",
            className="bank-label"
        ),
        
        html.Button(
            "Begin AI Analysis", 
            id='begin-analysis-button', 
            n_clicks=0, 
            className="bank-button bank-button-large"
        ),
        
        # Loading indicator
        dcc.Loading(
            id="analysis-loading",
            type="default",
            children=html.Div(id="analysis-output")
        )
    ], className="think-predict-container")

def render_review_tab():
    """Render the Review Performance tab content."""
    return html.Div([
        html.Div([
            html.Span("PERFORMANCE REVIEW", className="confidential-stamp")
        ], className="stamp-container"),
        
        html.P(
            "Historical Prediction Performance:",
            className="bank-label"
        ),
        
        # Placeholder for performance metrics
        html.Div(id='performance-metrics', className="performance-container")
    ], className="review-container")

# Additional callbacks for AI+ functionality
@callback(
    Output('aiplus-stock-dropdown', 'options'),
    Input('tabs', 'value')
)
def populate_aiplus_dropdown(tab):
    """Populate the stock dropdown for AI+ analysis."""
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
