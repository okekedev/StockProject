"""
Main application entry point for the Stock Analysis Dashboard.
"""
import os
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

# Import configuration
import config

# Import data loader
from modules.data_loader import load_data

# Import tab layouts
from layouts import download_tab
from layouts import technical_tab
from layouts import selection_tab
from layouts import test_tab 
from layouts import predict_tab
from layouts import research_tab
from layouts import aiplus_tab


# Load data
screener_df = load_data(config.INPUT_FILE, [
    'Symbol', 'Name', 'Last Sale', 'Volume', 'Market Cap', 
    'Sector', 'Industry', 'Data_Date', 'IPO Year'
])
symbols_df = load_data(config.OUTPUT_FILE, ['Symbol'])
tech_df = load_data(config.TECH_DATA_FILE, [
    'Symbol', 'Date', 'Close', 'Price_Change', 'TSMN', 'Close_Volatility', 
    'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment', 'MarketCap'
])

# Create Dash app with assets folder for CSS files
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'),
    suppress_callback_exceptions=True  # Add this line
)

# Define app layout with 1940s bank theme
app.layout = html.Div([
    # Bank Header with Emblem
    html.Header([
        html.Div(className="bank-emblem"),
        html.H1("Sundai Stocks", className="bank-title"),
        html.P("Financial Intelligence Using Math + AI", className="bank-subtitle")
    ], className="bank-header"),
    
    # Main Container
    html.Div([
        # Navigation Tabs
        dcc.Tabs([
            dcc.Tab(label='Stock Download', value='download', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(label='Screener', value='selection', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(label='Fetch Data', value='technical', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(label='Test Predictions', value='test', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(label='Predict Trends', value='predict', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(label='Research', value='research', className="bank-tab", selected_className="bank-tab--selected"),
            dcc.Tab(
    label='AI+ Prediction', 
    value='aiplus', 
    className="bank-tab", 
    selected_className="bank-tab--selected",
    # Same styling as your other tabs
)
        ], value='download', id='tabs', className="bank-tabs"),
        
        # Tab content containers with bank theme styling
        html.Div([
            download_tab.layout,
            selection_tab.get_layout(),
            technical_tab.layout,
            test_tab.get_layout(),
            predict_tab.get_layout(),
            research_tab.get_layout(),
        ], className="bank-content"),
        
        # Bank Footer
        html.Footer([
            html.P("© 2025 Stock Analysis Bureau"),
            html.P("All market data is for informational purposes only")
        ], className="bank-footer")
    ], className="bank-container"),
], className="bank-app texture-bg")

# Tab switching callback
@callback(
    [Output('download-content', 'style'),
     Output('selection-content', 'style'),
     Output('technical-content', 'style'),
     Output('test-content', 'style'),
     Output('predict-content', 'style'),
     Output('research-content', 'style'),
     Output('aiplus-content', 'style')],  # Add this output
    Input('tabs', 'value')
)
def toggle_tab_content(tab):
    """
    Toggle visibility of tab content based on selected tab.
    
    Args:
        tab (str): Selected tab value.
        
    Returns:
        tuple: Display style for each tab content div.
    """
    styles = {'display': 'none'}
    current_style = {'display': 'block', 'animation': 'slide-in 0.4s ease-out forwards'}
    
    return (
        current_style if tab == 'download' else styles,
        current_style if tab == 'selection' else styles,
        current_style if tab == 'technical' else styles,
        current_style if tab == 'test' else styles,
        current_style if tab == 'predict' else styles,
        current_style if tab == 'research' else styles,
        current_style if tab == 'aiplus' else styles,  # Add this line
    )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)