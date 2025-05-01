"""
Predict stocks tab layout and callbacks for the Stock Analysis Dashboard.
"""
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import config
from utils.script_runner import run_script
from modules.data_loader import load_data
from modules.date_utils import get_next_business_day

# Layout for the Predict Stocks tab with 1940s bank theme
def get_layout():
    """Generate layout with next business day."""
    next_business_day = get_next_business_day()
    
    return html.Div([
        html.Div([
            html.H4("Market Forecasting Bureau", className="bank-section-title")
        ], className="bank-card-header"),
        
        html.Div([
            html.Div([
                html.Div(className="bank-emblem", style={"float": "right", "width": "50px", "height": "50px"}),
                html.H5("OFFICIAL MARKET PREDICTION", className="bank-forecast-title"),
                html.P(
                    f"For Trading Date: {next_business_day}",
                    className="bank-forecast-date"
                ),
            ], className="bank-forecast-header"),
            
            html.P(
                "Our proprietary model generates forecasts for market movements based on "
                "technical analysis and historical performance patterns. This department "
                "produces official recommendations for your investment consideration.",
                className="bank-text"
            ),
            
            html.Div([
                html.P(
                    "IMPORTANT NOTICE: For most accurate next-day predictions, fetch stock data "
                    "after market close (6-7 PM EST) to ensure the latest trading day's data is incorporated "
                    "into the analysis bureau's calculations.",
                    className="bank-notice"
                ),
                
                html.Button(
                    "Generate Official Forecast", 
                    id='predict-button', 
                    n_clicks=0, 
                    className="bank-button stamp-button"
                ),
                
                html.Div(id='predict-table', className="bank-forecast-results slide-in")
            ], className="bank-card-content")
        ], className="bank-card"),
    ], id='predict-content', className="bank-section")

# Callback to handle predict stocks button click
@callback(
    Output('predict-table', 'children'),
    Input('predict-button', 'n_clicks'),
    State('tabs', 'value')
)
def update_predict(n_clicks, tab):
    """
    Handle predict stocks button click to run the predict_stocks.py script.
    
    Args:
        n_clicks (int): Number of button clicks.
        tab (str): Current active tab.
        
    Returns:
        html component: Prediction results display.
    """
    if n_clicks == 0 or tab != 'predict':
        return ""
    
    # Run the script
    success, output = run_script(config.PREDICT_SCRIPT)
    
    if success:
        # Load the prediction results
        predictions_df = load_data(config.PREDICTION_OUTPUT)
        
        if not predictions_df.empty:
            target_date = predictions_df['Target_Date'].iloc[0]
            
            # Format the predictions in a vintage stock ticker style
            return html.Div([
                html.Div([
                    html.Div([
                        html.Span("CONFIDENTIAL", className="confidential-stamp")
                    ], className="stamp-container"),
                    
                    html.H5(f"Top Market Opportunities for {target_date}", className="bank-results-subtitle"),
                    
                    html.P(
                        "The following securities have been identified by our analysis bureau as "
                        "having the highest probability of upward price movement. Recommendations "
                        "are weighted by historical model accuracy for each security.",
                        className="bank-results-description"
                    ),
                    
                    # Prediction table with vintage styling
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Rank"),
                                html.Th("Security"),
                                html.Th("Current Price"),
                                html.Th("Predicted Movement"),
                                html.Th("Model Confidence"),
                                html.Th("Recommendation"),
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"#{i+1}", className="rank-cell"),
                                html.Td(row['Symbol'], className="symbol-cell"),
                                html.Td(f"${row['Close']:.2f}", className="price-cell"),
                                html.Td(
                                    f"{row['percent_predicted_change']:.2f}%", 
                                    className=f"movement-cell {'positive' if row['percent_predicted_change'] > 0 else 'negative'}"
                                ),
                                html.Td(
                                    f"{row['Accuracy_%']:.1f}%", 
                                    className=f"accuracy-cell {get_accuracy_class(row['Accuracy_%'])}"
                                ),
                                html.Td(
                                    get_recommendation(row['percent_predicted_change'], row['Accuracy_%']),
                                    className="recommendation-cell"
                                ),
                            ]) for i, row in predictions_df.iterrows()
                        ])
                    ], className="bank-table prediction-table"),
                    
                    html.Div([
                        html.P(
                            "DISCLAIMER: These forecasts represent our analysis bureau's best predictions "
                            "based on available data and proprietary algorithms. Past performance is not "
                            "indicative of future results. All investment decisions should be made with "
                            "appropriate consideration of risk tolerance and financial objectives.",
                            className="bank-disclaimer"
                        ),
                        
                        html.Div([
                            html.Span("AUTHORIZED BY CENTRAL ANALYSIS DEPARTMENT", className="auth-text"),
                            html.Div(className="auth-signature")
                        ], className="authorization")
                    ], className="forecast-footer")
                ], className="forecast-document")
            ], className="forecast-container slide-in")
        
        return html.Div([
            html.P(
                "No predictions generated. Please ensure you have completed the Test Model process "
                "to generate accuracy data for weighted recommendations.",
                className="bank-error"
            )
        ], className="bank-notification bank-error-notification slide-in")
    
    return html.Div([
        html.P(
            "Error during forecast generation.",
            className="bank-error"
        ),
        html.Pre(
            output,
            className="bank-code"
        )
    ], className="bank-notification bank-error-notification slide-in")

# Helper function to get CSS class based on accuracy percentage
def get_accuracy_class(accuracy):
    """Return CSS class based on accuracy level."""
    if accuracy >= 70:
        return "high-accuracy"
    elif accuracy >= 60:
        return "medium-accuracy"
    else:
        return "low-accuracy"

# Helper function to get recommendation based on predicted change and accuracy
def get_recommendation(predicted_change, accuracy):
    """Return recommendation text based on prediction and accuracy."""
    if predicted_change <= 0:
        return "AVOID"
    
    if accuracy >= 70 and predicted_change >= 2.0:
        return "STRONG BUY"
    elif accuracy >= 60 and predicted_change >= 1.0:
        return "BUY"
    elif accuracy >= 55:
        return "MODERATE BUY"
    else:
        return "SPECULATIVE"