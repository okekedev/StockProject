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
            
            # Single column layout
            html.Div([
              
                
                # Stock selection
                html.Div([
                    html.H6("Data Collection", className="bank-card-subtitle"),
                    
                    html.P(
                        "Select a security for advanced AI analysis:",
                        className="bank-label"
                    ),
                    dcc.Dropdown(
                        id='aiplus-stock-dropdown',
                        placeholder="Select a security to analyze...",
                        className="bank-dropdown"
                    ),
                ], className="bank-section-container"),
                
                # Technical Data Parameters
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
                ], className="bank-section-container"),
                
                # News Data Parameters
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
                ], className="bank-section-container"),
                
                # Prediction Parameters
                html.Div([
                    html.H6("Prediction Parameters", className="bank-card-subtitle"),
                    
                    # Prediction Horizon selection
                    html.P("Select the timeframe for price prediction:", className="bank-text"),
                    dcc.Dropdown(
                        id='prediction-horizon-dropdown',
                        options=[
                            {'label': 'Next Day', 'value': '1d'},
                            {'label': 'Next 2 Days', 'value': '2d'},
                            {'label': 'Next Week', 'value': '1w'},
                            {'label': 'Next Month', 'value': '1mo'}
                        ],
                        value='1d',
                        className="bank-dropdown"
                    ),
                    
                    # Prediction Forecast Time
                    html.P("Select forecast time for predictions:", className="bank-text", style={"marginTop": "15px"}),
                    dcc.Dropdown(
                        id='forecast-time-dropdown',
                        options=[
                            {'label': 'End of Day', 'value': 'eod'},
                            {'label': 'End of Week', 'value': 'eow'},
                            {'label': 'End of Month', 'value': 'eom'},
                            {'label': 'Market Open', 'value': 'open'},
                            {'label': 'Market Close', 'value': 'close'}
                        ],
                        value='close',
                        className="bank-dropdown"
                    ),
                ], className="bank-section-container"),
                
                # Data Readiness Status
                html.Div([
                    html.H6("Data Readiness Status", className="bank-card-subtitle"),
                    html.Div(id='data-readiness-status', className="bank-status-large")
                ], className="bank-section-container"),
                
                # Analysis Button
                html.Div([
                    html.Button(
                        "Begin AI Analysis", 
                        id='begin-analysis-button', 
                        n_clicks=0, 
                        className="bank-button bank-button-large",
                        style={"marginTop": "15px", "marginBottom": "15px"}
                    ),
                ], className="bank-section-container", style={"textAlign": "center"}),
                
                # Analysis Results
                html.Div([
                    html.H6("Analysis Results", className="bank-card-subtitle"),
                    
                    # Loading indicator and analysis output
                    dcc.Loading(
                        id="analysis-loading",
                        type="default",
                        children=html.Div(id="analysis-output", className="analysis-output-container")
                    ),
                ], className="bank-section-container"),
                
                # Performance Metrics
                html.Div([
                    html.H6("Performance Metrics", className="bank-card-subtitle"),
                    html.Div(id='performance-metrics', className="performance-container")
                ], className="bank-section-container"),
                
            ], className="bank-combined-view")
        ], className="bank-card"),
    ], id='aiplus-content', className="bank-section")

# No callback is needed here since it's handled elsewhere in the app