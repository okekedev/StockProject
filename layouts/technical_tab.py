"""
Technical data tab layout and callbacks for the Stock Analysis Dashboard.
"""
from dash import html, dcc, callback, Output, Input, State
import config
from utils.script_runner import run_script
from modules.data_loader import load_data

# Layout for the Technical Data tab with 1940s bank theme
layout = html.Div([
    html.Div([
        html.H4("Technical Data Procurement", className="bank-section-title")
    ], className="bank-card-header"),
    
    html.Div([
        html.P(
            "This service will acquire technical market data for your selected stocks. "
            "The system will retrieve historical price information, volatility metrics, and market sentiment "
            "indicators necessary for financial analysis.",
            className="bank-text"
        ),
        
        html.Div([
            html.P(
                "Please ensure you have completed the Stock Selection process before proceeding.",
                className="bank-text bank-info-panel"
            ),
            
            html.Button(
                [
                    html.Span("Commence Data Fetching", className="button-text"),
                ], 
                id='technical-button', 
                n_clicks=0, 
                className="bank-button"
            ),
            
            html.Div(id='technical-output', className="bank-output fade-in")
        ], className="bank-card-content")
    ], className="bank-card"),
], id='technical-content', className="bank-section")

# Callback to handle fetch technical data button click
@callback(
    Output('technical-output', 'children'),
    Input('technical-button', 'n_clicks'),
    State('tabs', 'value')
)
def update_technical(n_clicks, tab):
    """
    Handle fetch technical data button click to run the fetch_technical_data.py script.
    
    Args:
        n_clicks (int): Number of button clicks.
        tab (str): Current active tab.
        
    Returns:
        str: Status message.
    """
    if n_clicks == 0 or tab != 'technical':
        return ""
    
    # Run the script
    success, output = run_script(config.TECHNICAL_SCRIPT)
    
    if success:
        # Load the technical data to check row count
        tech_df = load_data(config.TECH_DATA_FILE)
        
        if not tech_df.empty:
            # Count unique symbols
            unique_symbols = tech_df['Symbol'].nunique() if 'Symbol' in tech_df.columns else 0
            
            # Get date range
            date_range = ""
            if 'Date' in tech_df.columns:
                min_date = tech_df['Date'].min()
                max_date = tech_df['Date'].max()
                if min_date and max_date:
                    date_range = f"Data spans from {min_date.strftime('%B %d, %Y')} to {max_date.strftime('%B %d, %Y')}"
            
            return html.Div([
                html.P(
                    f"Technical data procurement complete: {len(tech_df)} records acquired for {unique_symbols} securities.", 
                    className="bank-success"
                ),
                html.P(
                    date_range,
                    className="bank-text"
                ),
                html.P(
                    f"Data has been filed in your records at {config.TECH_DATA_FILE}",
                    className="bank-text"
                ),
                
                # Technical indicators acquired
                html.Div([
                    html.P("Technical Indicators Acquired:", className="bank-label"),
                    html.Ul([
                        html.Li("Price Movement", className="bank-list-item"),
                        html.Li("Volume Analysis", className="bank-list-item"),
                        html.Li("Relative Strength Indicators", className="bank-list-item"),
                        html.Li("Volatility Metrics", className="bank-list-item"),
                        html.Li("TSMN (Temporal-Spectral Momentum Nexus)", className="bank-list-item"),
                    ], className="bank-list")
                ], className="bank-indicator-list")
            ], className="bank-notification bank-success-notification slide-in")
        
        return html.Div([
            html.P(
                "No technical data acquired. Please ensure your stock symbols file contains valid securities.",
                className="bank-error"
            )
        ], className="bank-notification bank-error-notification slide-in")
    
    return html.Div([
        html.P(
            "Error during technical data acquisition.",
            className="bank-error"
        ),
        html.P(
            output,
            className="bank-code"
        )
    ], className="bank-notification bank-error-notification slide-in")