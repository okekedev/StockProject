import pandas as pd
import os
import subprocess
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta

# File paths
INPUT_FILE = "./stock_data/nasdaq_screener.csv"
OUTPUT_FILE = "./stock_data/stock_symbols.csv"
TECH_DATA_FILE = "./stock_data/stock_data_technical.csv"
PREDICTION_OUTPUT = "./stock_data/top_10_upward_picks.csv"
TEST_OUTPUT = "./stock_data/test_results.csv"
TEST_STOCK_ACCURACY = "./stock_data/test_stock_accuracy.csv"

# Load or initialize data
def load_data(file_path, default_columns=None):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Ensure Date column is Timestamp for technical data
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()
    elif default_columns:
        return pd.DataFrame(columns=default_columns)
    return pd.DataFrame()

# Global DataFrames
screener_df = load_data(INPUT_FILE, ['Symbol', 'Name', 'Last Sale', 'Volume', 'Market Cap', 'Sector', 'Industry', 'Data_Date'])
symbols_df = load_data(OUTPUT_FILE, ['Symbol'])
tech_df = load_data(TECH_DATA_FILE, ['Symbol', 'Date', 'Close', 'Price_Change', 'TSMN', 'Close_Volatility', 'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment', 'MarketCap'])

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Criteria for dropdowns
criteria = {
    'IPO Year': sorted(screener_df['IPO Year'].dropna().astype(int).unique().tolist()),
    'Last Sale': sorted(screener_df['Last Sale'].dropna().unique().tolist()),
    'Market Cap': sorted(screener_df['Market Cap'].dropna().unique().tolist()),
    'Sector': sorted(screener_df['Sector'].dropna().unique().tolist()),
    'Industry': sorted(screener_df['Industry'].dropna().unique().tolist()),
    'Volume': sorted(screener_df['Volume'].dropna().astype(int).unique().tolist())
}

# Function to generate date options for Test Model tab, excluding weekends and Mondays, limiting to 3 days before tomorrow
def get_date_options(start_date, end_date, default_date=None):
    # Convert start_date and end_date to date objects if they are datetime
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    
    # Ensure start_date is not after end_date
    if start_date > end_date:
        return [], None
    
    # Limit end_date to 3 days before tomorrow
    today = datetime.now().date()
    max_date = today - timedelta(days=3)  # 3 days before tomorrow
    end_date = min(end_date, max_date)
    
    # Generate business days (excludes weekends) and exclude Mondays (weekday 0)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only (excludes weekends)
    valid_dates = [date for date in dates if date.weekday() != 0]  # Exclude Mondays (weekday 0)
    options = [
        {'label': date.strftime('%Y-%m-%d'), 'value': date.strftime('%Y-%m-%d'), 'disabled': False}
        for date in valid_dates
    ]
    if not options:  # If no valid dates, return empty options
        return [], None
    if default_date:
        if isinstance(default_date, datetime):
            default_date = default_date.date()
        default_value = default_date.strftime('%Y-%m-%d')
    else:
        default_value = valid_dates[-1].strftime('%Y-%m-%d') if len(valid_dates) > 0 else None  # Default to most recent valid day
    return options, default_value

# Function to generate end date options for Test Model tab based on start date (up to 2 weeks)
def get_end_date_options(start_date, max_end_date):
    if not start_date:
        return [], None
    start_date = pd.to_datetime(start_date)
    max_end_date = pd.to_datetime(max_end_date)
    # Limit to 2 weeks (14 days) from start date or max_end_date, whichever is earlier
    two_weeks_later = start_date + timedelta(days=14)
    end_date_limit = min(max_end_date, two_weeks_later)
    if start_date >= end_date_limit:
        return [], None
    dates = pd.date_range(start=start_date + timedelta(days=1), end=end_date_limit, freq='B')
    valid_dates = [date for date in dates if date.weekday() != 0]  # Exclude Mondays
    options = [
        {'label': date.strftime('%Y-%m-%d'), 'value': date.strftime('%Y-%m-%d'), 'disabled': False}
        for date in valid_dates
    ]
    if not options:  # If no valid dates, return empty options
        return [], None
    default_value = valid_dates[-1].strftime('%Y-%m-%d')  # Default to last valid date
    return options, default_value

# Initial layout with all components, hidden by default
app.layout = html.Div([
    html.H2("Stock Analysis Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Stock Download (As Needed)', value='download'),
        dcc.Tab(label='Stock Selection: Step 1', value='selection'),
        dcc.Tab(label='Fetch Technical Data: Step 2', value='technical'),
        dcc.Tab(label='Test Model: Step 3', value='test'),
        dcc.Tab(label='Predict Stocks: Step 4', value='predict'),
    ], value='download', id='tabs'),
    # Stock Download View
    html.Div([
        html.H4("Download NASDAQ Stock Screener"),
        html.Button("Run Download", id='download-button', n_clicks=0),
        html.Div(id='download-output')
    ], id='download-content', style={'display': 'none'}),
    # Stock Selection View
    html.Div([
        html.H4("Stock Selection"),
        dcc.RadioItems(
            options=[
                {'label': 'Manual Selection', 'value': 'manual'},
                {'label': 'Stock Screener', 'value': 'screener'}
            ],
            value='manual',
            id='selection-mode'
        ),
        html.Div(
            dcc.Dropdown(
                id='manual-dropdown',
                options=[{'label': s, 'value': s} for s in sorted(screener_df['Symbol'].astype(str).dropna().unique()) if s],
                multi=True,
                placeholder="Select stock symbols..."
            ),
            id='manual-content',
            style={'display': 'block'}
        ),
        html.Div([
            dcc.Dropdown(id='min-ipo', options=[''] + criteria['IPO Year'], placeholder="Min IPO Year"),
            dcc.Dropdown(id='max-ipo', options=[''] + criteria['IPO Year'], placeholder="Max IPO Year"),
            dcc.Dropdown(id='min-price', options=[''] + [str(x) for x in criteria['Last Sale']], placeholder="Min Price"),
            dcc.Dropdown(id='max-price', options=[''] + [str(x) for x in criteria['Last Sale']], placeholder="Max Price"),
            dcc.Dropdown(id='min-cap', options=[''] + [str(x) for x in criteria['Market Cap']], placeholder="Min Market Cap"),
            dcc.Dropdown(id='max-cap', options=[''] + [str(x) for x in criteria['Market Cap']], placeholder="Max Market Cap"),
            dcc.Dropdown(id='min-volume', options=[''] + [str(x) for x in criteria['Volume']], placeholder="Min Volume"),
            dcc.Dropdown(id='max-volume', options=[''] + [str(x) for x in criteria['Volume']], placeholder="Max Volume"),
            dcc.Dropdown(id='sector', options=[''] + criteria['Sector'], placeholder="Sector"),
            dcc.Dropdown(id='industry', options=[''] + criteria['Industry'], placeholder="Industry")
        ], id='screener-content', style={'display': 'none'}),
        html.Button("Run Selection", id='save-selection', n_clicks=0),
        html.Div(id='selection-output')
    ], id='selection-content', style={'display': 'none'}),
    # Fetch Technical Data View
    html.Div([
        html.H4("Fetch Technical Data"),
        html.Button("Run Fetch", id='technical-button', n_clicks=0),
        html.Div(id='technical-output')  # Simplified for completion notification
    ], id='technical-content', style={'display': 'none'}),
    # Test Model View
    html.Div([
        html.H4("Test Model Results"),
        html.Label("Select Target Start Date (Tuesday to Friday, up to 3 days before tomorrow):"),
        dcc.Dropdown(
            id='test-start-date',
            options=get_date_options(datetime(2020, 1, 2), datetime.now() + timedelta(days=1))[0],
            value=get_date_options(datetime(2020, 1, 2), datetime.now() + timedelta(days=1))[1],
            placeholder="Select start date for testing",
            clearable=False
        ),
        html.Label("Select Target End Date (up to 2 weeks from start, Tuesday to Friday):"),
        dcc.Dropdown(
            id='test-end-date',
            options=get_end_date_options(datetime.now() + timedelta(days=1), datetime.now() + timedelta(days=1))[0],
            value=get_end_date_options(datetime.now() + timedelta(days=1), datetime.now() + timedelta(days=1))[1],
            placeholder="Select end date for testing",
            clearable=False
        ),
        html.Button("Run Test", id='test-button', n_clicks=0),
        html.P(id='test-success-rate'),
        html.Pre(id='test-table')
    ], id='test-content', style={'display': 'none'}),
    # Predict Stocks View
    html.Div([
        html.H4("Predict Stock Changes"),
        html.P(f"Predicting for the next business day: {((datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d'))}"),
        html.Button("Run Predictions", id='predict-button', n_clicks=0),
        html.Pre(id='predict-table')  # Simplified to show table only, no graph
    ], id='predict-content', style={'display': 'none'}),
])

# Callback to update end date options for Test Model based on start date
@app.callback(
    [Output('test-end-date', 'options'),
     Output('test-end-date', 'value')],
    Input('test-start-date', 'value')
)
def update_test_end_date_options(start_date):
    if not start_date:
        return [], None
    max_end_date = datetime.now() - timedelta(days=2)  # 3 days before tomorrow
    options, default_value = get_end_date_options(start_date, max_end_date)
    return options, default_value

# Stock Download Callback
@app.callback(
    Output('download-output', 'children'),
    Input('download-button', 'n_clicks'),
    State('tabs', 'value')
)
def handle_download(n_clicks, tab):
    if n_clicks > 0 and tab == 'download':
        script_path = os.path.join(os.path.dirname(__file__), "stock_download.py")
        if not os.path.exists(script_path):
            return "Error: stock_download.py not found."
        
        try:
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Script output: {result.stdout}")
            
            global screener_df
            screener_df = load_data(INPUT_FILE)
            
            if not screener_df.empty:
                return f"Downloaded {len(screener_df)} records. Data saved to {INPUT_FILE}"
            return "No data downloaded. Ensure the download was successful."
        
        except subprocess.CalledProcessError as e:
            print(f"Script error: {e.stderr}")
            return f"Failed to run stock_download.py: {e.stderr}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Unexpected error running stock_download.py: {str(e)}"
    return ""

# Stock Selection Callbacks
@app.callback(
    [Output('manual-content', 'style'),
     Output('screener-content', 'style')],
    Input('selection-mode', 'value')
)
def update_selection_mode(mode):
    if mode == 'manual':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('selection-output', 'children'),
    Input('save-selection', 'n_clicks'),
    State('selection-mode', 'value'),
    State('manual-dropdown', 'value'),
    State('min-ipo', 'value'),
    State('max-ipo', 'value'),
    State('min-price', 'value'),
    State('max-price', 'value'),
    State('min-cap', 'value'),
    State('max-cap', 'value'),
    State('min-volume', 'value'),
    State('max-volume', 'value'),
    State('sector', 'value'),
    State('industry', 'value')
)
def save_selection(n_clicks, mode, manual_symbols, min_ipo, max_ipo, min_price, max_price, min_cap, max_cap, min_volume, max_volume, sector, industry):
    if n_clicks == 0:
        return ""
    
    filtered_df = screener_df.copy()
    if mode == 'manual':
        if not manual_symbols:
            return "Please select at least one symbol."
        filtered_df = pd.DataFrame({'Symbol': manual_symbols})
    else:
        if min_ipo:
            filtered_df = filtered_df[filtered_df['IPO Year'] >= int(min_ipo)]
        if max_ipo:
            filtered_df = filtered_df[filtered_df['IPO Year'] <= int(max_ipo)]
        if min_price:
            filtered_df = filtered_df[filtered_df['Last Sale'] >= float(min_price)]
        if max_price:
            filtered_df = filtered_df[filtered_df['Last Sale'] <= float(max_price)]
        if min_cap:
            filtered_df = filtered_df[filtered_df['Market Cap'] >= float(min_cap)]
        if max_cap:
            filtered_df = filtered_df[filtered_df['Market Cap'] <= float(max_cap)]
        if min_volume:
            filtered_df = filtered_df[filtered_df['Volume'] >= int(min_volume)]
        if max_volume:
            filtered_df = filtered_df[filtered_df['Volume'] <= int(max_volume)]
        if sector:
            filtered_df = filtered_df[filtered_df['Sector'] == sector]
        if industry:
            filtered_df = filtered_df[filtered_df['Industry'] == industry]
        filtered_df = filtered_df[['Symbol']]
    
    if filtered_df.empty:
        return "No stocks match your criteria."
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    filtered_df.to_csv(OUTPUT_FILE, index=False)
    return f"Saved {len(filtered_df)} symbols to {OUTPUT_FILE}"

# Predict Stocks Callback
@app.callback(
    Output('predict-table', 'children'),
    Input('predict-button', 'n_clicks'),
    State('tabs', 'value')
)
def update_predict(n_clicks, tab):
    if n_clicks == 0 or tab != 'predict':
        return ""
    
    # Run the entire predict_stocks.py script (no date arguments since it predicts for the next day)
    script_path = os.path.join(os.path.dirname(__file__), "predict_stocks.py")
    if not os.path.exists(script_path):
        return "Error: predict_stocks.py not found."
    
    try:
        # Use subprocess to run the script
        result = subprocess.run(
            ['python', script_path],  # No date arguments needed
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Script output: {result.stdout}")
        
        # Load the prediction results from the CSV
        predictions_df = load_data(PREDICTION_OUTPUT)
        
        if not predictions_df.empty:
            target_date = predictions_df['Target_Date'].iloc[0]  # Get the target date from the predictions
            return html.Pre(f"Top 10 Predictions for {target_date}:\n" + predictions_df.to_string(index=False))
        return "No predictions generated. Check if data is available."
    
    except subprocess.CalledProcessError as e:
        print(f"Script error: {e.stderr}")
        return f"Failed to run predict_stocks.py: {e.stderr}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Unexpected error running predict_stocks.py: {str(e)}"

# Test Model Callbacks
@app.callback(
    [Output('test-success-rate', 'children'),
     Output('test-table', 'children')],
    Input('test-button', 'n_clicks'),
    State('test-start-date', 'value'),
    State('test-end-date', 'value'),
    State('tabs', 'value')
)
def update_test(n_clicks, start_date, end_date, tab):
    if n_clicks == 0 or tab != 'test':
        return "", ""
    
    if not start_date or not end_date:
        return "Please select both a start date and an end date.", ""
    
    # Validate date range
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date_dt >= end_date_dt:
        return "Start date must be before end date.", ""
    
    # Run the entire test_model.py script with the date range as arguments
    script_path = os.path.join(os.path.dirname(__file__), "test_model.py")
    if not os.path.exists(script_path):
        return "", "Error: test_model.py not found."
    
    try:
        # Use subprocess to run the script with start_date and end_date
        result = subprocess.run(
            ['python', script_path, start_date, end_date],  # Pass both dates as command-line arguments
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Script output: {result.stdout}")
        
        # Load the test results from the CSV
        test_df = load_data(TEST_OUTPUT)
        # Load the per-stock accuracy results
        stock_accuracy_df = load_data(TEST_STOCK_ACCURACY)
        
        if not test_df.empty:
            success_rate = f"Successful Trend Predictions: {test_df['Trend_Success'].sum()} out of {len(test_df)} ({(test_df['Trend_Success'].mean() * 100):.2f}%)"
            if not stock_accuracy_df.empty:
                stock_accuracy_output = f"\nPer-Stock Accuracy:\n{stock_accuracy_df.to_string(index=False)}"
            else:
                stock_accuracy_output = "\nPer-Stock Accuracy: Not available."
            table_content = html.Pre(
                f"Test Results for {start_date} to {end_date}:\n" +
                test_df.to_string(index=False) + stock_accuracy_output
            )
        else:
            success_rate = "No success rate available."
            table_content = html.P("No test results available.")
        
        return success_rate, table_content
    
    except subprocess.CalledProcessError as e:
        print(f"Script error: {e.stderr}")
        return "", f"Failed to run test_model.py: {e.stderr}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "", f"Unexpected error running test_model.py: {str(e)}"

# Fetch Technical Data Callbacks
@app.callback(
    Output('technical-output', 'children'),
    Input('technical-button', 'n_clicks'),
    State('tabs', 'value')
)
def update_technical(n_clicks, tab):
    if n_clicks == 0 or tab != 'technical':
        return ""
    
    # Run the entire fetch_technical_data.py script
    script_path = os.path.join(os.path.dirname(__file__), "fetch_technical_data.py")
    if not os.path.exists(script_path):
        return "Error: fetch_technical_data.py not found."
    
    try:
        # Use subprocess to run the script
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Script output: {result.stdout}")
        
        # Reload tech_df with the new data
        global tech_df
        tech_df = load_data(TECH_DATA_FILE)
        
        if not tech_df.empty:
            return f"Total records: {len(tech_df)}"
        return "No technical data available. Ensure stock_symbols.csv has valid symbols."
    
    except subprocess.CalledProcessError as e:
        print(f"Script error: {e.stderr}")
        return f"Failed to run fetch_technical_data.py: {e.stderr}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Unexpected error running fetch_technical_data.py: {str(e)}"

# Tab Switching Callback
@app.callback(
    [Output('download-content', 'style'),
     Output('selection-content', 'style'),
     Output('predict-content', 'style'),
     Output('test-content', 'style'),
     Output('technical-content', 'style')],
    Input('tabs', 'value')
)
def toggle_tab_content(tab):
    styles = {'display': 'none'}
    current_style = {'display': 'block'}
    return (
        current_style if tab == 'download' else styles,
        current_style if tab == 'selection' else styles,
        current_style if tab == 'predict' else styles,
        current_style if tab == 'test' else styles,
        current_style if tab == 'technical' else styles
    )

if __name__ == "__main__":
    app.run_server(debug=True)