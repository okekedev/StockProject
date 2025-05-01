"""
Test model tab layout and callbacks for the Stock Analysis Dashboard.
"""
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import config
from utils.script_runner import run_script
from modules.data_loader import load_data
from modules.date_utils import get_date_options, get_end_date_options

def get_layout():
    """Generate layout with date options."""
    # Generate date options for start date
    start_options, start_value = get_date_options(
        datetime(2020, 1, 2), 
        datetime.now() - timedelta(days=3)
    )
    
    # Generate date options for end date based on start date
    end_options, end_value = [], None
    if start_value:
        end_options, end_value = get_end_date_options(
            start_value,
            datetime.now() - timedelta(days=3)
        )
    
    return html.Div([
        html.Div([
            html.H4("Model Testing & Verification", className="bank-section-title")
        ], className="bank-card-header"),
        
        html.Div([
            html.P(
                "This department verifies the accuracy of our proprietary prediction model "
                "by backtesting against historical market data. Select a date range to "
                "evaluate model performance.",
                className="bank-text"
            ),
            
            html.Div([
                # Date selection form
                html.Div([
                    html.Label(
                        "Target Start Date (Tuesday to Friday, up to 3 days before today):",
                        className="bank-label"
                    ),
                    dcc.Dropdown(
                        id='test-start-date',
                        options=start_options,
                        value=start_value,
                        placeholder="Select start date for testing",
                        clearable=False,
                        className="bank-dropdown"
                    ),
                    
                    html.Label(
                        "Target End Date (up to 30 days from start, Tuesday to Friday):",
                        className="bank-label",
                        style={"marginTop": "15px"}
                    ),
                    dcc.Dropdown(
                        id='test-end-date',
                        options=end_options,
                        value=end_value,
                        placeholder="Select end date for testing",
                        clearable=False,
                        className="bank-dropdown"
                    ),
                    
                    html.Button(
                        "Run Model Test", 
                        id='test-button', 
                        n_clicks=0, 
                        className="bank-button",
                        style={"marginTop": "20px"}
                    ),
                ], className="bank-form"),
                
                # Results section
                html.Div([
                    html.Div(id='test-success-rate', className="bank-results-header"),
                    html.Div(id='test-table', className="bank-results-content slide-in")
                ], className="bank-results-container")
            ], className="bank-card-content")
        ], className="bank-card"),
    ], id='test-content', className="bank-section")

# Callback to update end date options when start date changes
@callback(
    [Output('test-end-date', 'options'),
     Output('test-end-date', 'value')],
    Input('test-start-date', 'value')
)
def update_test_end_date_options(start_date):
    """
    Update end date options based on selected start date.
    
    Args:
        start_date (str): Selected start date.
        
    Returns:
        tuple: (options, value) for end date dropdown.
    """
    if not start_date:
        return [], None
    
    max_end_date = datetime.now() - timedelta(days=3)
    options, default_value = get_end_date_options(start_date, max_end_date)
    
    return options, default_value

# Callback to handle test model button click
@callback(
    [Output('test-success-rate', 'children'),
     Output('test-table', 'children')],
    Input('test-button', 'n_clicks'),
    State('test-start-date', 'value'),
    State('test-end-date', 'value'),
    State('tabs', 'value')
)
def update_test(n_clicks, start_date, end_date, tab):
    """
    Handle test model button click to run the test_model.py script.
    
    Args:
        n_clicks (int): Number of button clicks.
        start_date (str): Selected start date.
        end_date (str): Selected end date.
        tab (str): Current active tab.
        
    Returns:
        tuple: (success_rate, table_content) to display.
    """
    if n_clicks == 0 or tab != 'test':
        return "", ""
    
    if not start_date or not end_date:
        return html.P(
            "Please select both a start date and an end date.", 
            className="bank-error"
        ), ""
    
    # Validate date range
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date_dt >= end_date_dt:
        return html.P(
            "Start date must be before end date.", 
            className="bank-error"
        ), ""
    
    # Run the script with date arguments
    success, output = run_script(config.TEST_SCRIPT, [start_date, end_date])
    
    if success:
        # Load the test results
        test_df = load_data(config.TEST_OUTPUT)
        stock_accuracy_df = load_data(config.TEST_STOCK_ACCURACY)
        
        if not test_df.empty:
            success_count = test_df['Trend_Success'].sum()
            total_count = len(test_df)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            # Success rate display with vintage adding machine style
            success_rate_display = html.Div([
                html.Div([
                    html.Span("MODEL VERIFICATION RESULTS", className="bank-results-title"),
                    html.Div([
                        html.Span("Test Period:", className="result-label"),
                        html.Span(f"{start_date} to {end_date}", className="result-value")
                    ], className="result-line"),
                    html.Div([
                        html.Span("Successful Predictions:", className="result-label"),
                        html.Span(f"{success_count} of {total_count}", className="result-value")
                    ], className="result-line"),
                    html.Div([
                        html.Span("Success Rate:", className="result-label"),
                        html.Span(f"{success_rate:.2f}%", className="result-value highlight")
                    ], className="result-line"),
                ], className="bank-results-summary")
            ], className="bank-results-header-content stamp")
            
            # Table content
            if not stock_accuracy_df.empty:
                # Per-stock accuracy table
                accuracy_table = html.Div([
                    html.H5("Individual Security Performance", className="bank-table-title"),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Security"),
                                html.Th("Successful"),
                                html.Th("Total"),
                                html.Th("Accuracy")
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(row['Symbol']),
                                html.Td(f"{row['Successful_Trends']:.0f}"),
                                html.Td(f"{row['Total_Predictions']:.0f}"),
                                html.Td(f"{row['Accuracy_%']:.2f}%", 
                                       className="accuracy-cell" + (" high-accuracy" if row['Accuracy_%'] >= 60 else ""))
                            ]) for i, row in stock_accuracy_df.head(10).iterrows()
                        ])
                    ], className="bank-table")
                ], className="bank-table-container slide-in")
                
                # Full test results
                test_results = html.Div([
                    html.H5("Detailed Test Results", className="bank-table-title"),
                    html.Div([
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Symbol"),
                                    html.Th("Date"),
                                    html.Th("Target"),
                                    html.Th("Close"),
                                    html.Th("Predicted %"),
                                    html.Th("Actual %"),
                                    html.Th("Success")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(row['Symbol']),
                                    html.Td(row['Prediction_Date']),
                                    html.Td(row['Target_Date']),
                                    html.Td(f"${row['Close']:.2f}"),
                                    html.Td(f"{row['Predicted_%_Change']:.2f}%", 
                                           className=("positive" if row['Predicted_%_Change'] > 0 else "negative")),
                                    html.Td(f"{row['Actual_%_Change']:.2f}%",
                                           className=("positive" if row['Actual_%_Change'] > 0 else "negative")),
                                    html.Td("✓" if row['Trend_Success'] else "✗",
                                           className=("success" if row['Trend_Success'] else "failure"))
                                ]) for i, row in test_df.head(15).iterrows()
                            ])
                        ], className="bank-table")
                    ], className="bank-table-scroll")
                ], className="bank-table-container slide-in")
                
                return success_rate_display, html.Div([accuracy_table, test_results])
            
            return success_rate_display, html.P("No detailed results available.", className="bank-text")
        
        return html.P(
            "No test results available. Please ensure your technical data is properly loaded.",
            className="bank-error"
        ), ""
    
    return html.P(
        "Error during model testing.",
        className="bank-error"
    ), html.Pre(output, className="bank-code")