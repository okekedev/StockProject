"""
Download tab layout and callbacks for the Stock Analysis Dashboard.
"""
from dash import html, dcc, callback, Output, Input, State
import config
from utils.script_runner import run_script
from modules.data_loader import load_data

# Layout for the Download tab with 1940s bank theme
layout = html.Div([
    html.Div([
        html.H4("NASDAQ Stock Data Acquisition", className="bank-section-title")
    ], className="bank-card-header"),
    
    html.Div([
        html.P(
            "This service allows you to download the latest stock data from the NASDAQ exchange. "
            "The data will be saved to your local records for further analysis.",
            className="bank-text"
        ),
        
        html.Div([
            html.Button(
                [
                    html.Span("Initiate Download", className="button-text"),
                ], 
                id='download-button', 
                n_clicks=0, 
                className="bank-button"
            ),
            
            html.Div(id='download-output', className="bank-output fade-in")
        ], className="bank-card-content")
    ], className="bank-card"),
], id='download-content', className="bank-section")

# Callback to handle download button click
@callback(
    Output('download-output', 'children'),
    Input('download-button', 'n_clicks'),
    State('tabs', 'value')
)
def handle_download(n_clicks, tab):
    """
    Handle download button click to run the stock_download.py script.
    
    Args:
        n_clicks (int): Number of button clicks.
        tab (str): Current active tab.
        
    Returns:
        str: Status message.
    """
    if n_clicks == 0 or tab != 'download':
        return ""
    
    # Show loading indicator
    loading_html = html.Div([
        html.Div(className="bank-loading"),
        html.P("Downloading data from NASDAQ...", className="bank-text")
    ], className="bank-loading-container")
    
    # Run the script
    success, output = run_script(config.DOWNLOAD_SCRIPT)
    
    if success:
        # Load the newly downloaded data to check row count
        screener_df = load_data(config.INPUT_FILE)
        
        if not screener_df.empty:
            return html.Div([
                html.P(
                    f"Successfully acquired {len(screener_df)} records from NASDAQ exchange.", 
                    className="bank-success"
                ),
                html.P(
                    f"Data has been filed in your records at {config.INPUT_FILE}",
                    className="bank-text"
                ),
                html.Div([
                    html.P("Data acquisition timestamp:", className="bank-label"),
                    html.P(screener_df['Data_Date'].iloc[0] if 'Data_Date' in screener_df.columns else "Not available", 
                           className="bank-value stamp")
                ], className="bank-timestamp")
            ], className="bank-notification bank-success-notification slide-in")
        
        return html.Div([
            html.P(
                "No data acquired. Please verify NASDAQ service availability.",
                className="bank-error"
            )
        ], className="bank-notification bank-error-notification slide-in")
    
    return html.Div([
        html.P(
            "Error during data acquisition.",
            className="bank-error"
        ),
        html.P(
            output,
            className="bank-code"
        )
    ], className="bank-notification bank-error-notification slide-in")