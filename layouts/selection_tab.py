"""
Selection tab layout and callbacks for the Stock Analysis Dashboard.
"""
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
import re
import config
from modules.data_loader import load_data, get_criteria_from_data

# Function to convert dollar amounts to float
def convert_dollar_to_float(value):
    """
    Convert a dollar amount string to float.
    
    Args:
        value: The value to convert.
        
    Returns:
        float: Converted value or NaN if conversion failed.
    """
    if pd.isna(value):
        return pd.NA
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove dollar signs, commas, etc.
        try:
            # Remove dollar sign, commas and other non-numeric characters except decimal point
            clean_value = re.sub(r'[^\d.-]', '', value)
            return float(clean_value)
        except:
            pass
    
    return pd.NA

# Function to get layout
def get_layout():
    """Generate layout based on current data."""
    # Load the data
    screener_df = load_data(config.INPUT_FILE)
    
    # Convert data types for proper filtering
    if not screener_df.empty:
        # Convert price column to numeric, handling dollar signs
        if 'Last Sale' in screener_df.columns:
            screener_df['Last Sale'] = screener_df['Last Sale'].apply(convert_dollar_to_float)
        
        # Convert market cap column to numeric
        if 'Market Cap' in screener_df.columns:
            screener_df['Market Cap'] = pd.to_numeric(screener_df['Market Cap'], errors='coerce')
            
        # Convert volume column to numeric
        if 'Volume' in screener_df.columns:
            screener_df['Volume'] = pd.to_numeric(screener_df['Volume'], errors='coerce')
    
    # Get criteria from data
    criteria = get_criteria_from_data(screener_df)
    
    # Display range information about the data
    price_range = ""
    market_cap_range = ""
    volume_range = ""
    
    if not screener_df.empty:
        if 'Last Sale' in screener_df.columns and not screener_df['Last Sale'].isna().all():
            min_price = screener_df['Last Sale'].min()
            max_price = screener_df['Last Sale'].max()
            price_range = f"(Range: ${min_price:.2f} - ${max_price:.2f})"
            
        if 'Market Cap' in screener_df.columns and not screener_df['Market Cap'].isna().all():
            min_cap = screener_df['Market Cap'].min()
            max_cap = screener_df['Market Cap'].max()
            market_cap_range = f"(Range: {min_cap:.0f} - {max_cap:.0f})"
            
        if 'Volume' in screener_df.columns and not screener_df['Volume'].isna().all():
            min_vol = screener_df['Volume'].min()
            max_vol = screener_df['Volume'].max()
            volume_range = f"(Range: {min_vol:.0f} - {max_vol:.0f})"
    
    return html.Div([
        html.H4("Stock Selection", className="bank-section-title"),
        html.Div([
            html.P(f"Available Stocks: {len(screener_df) if not screener_df.empty else 0} records", className="bank-text"),
            html.P("Use the Manual Selection to pick specific stocks or use the Stock Screener to filter by criteria.", className="bank-text")
        ], className="bank-info-panel"),
        
        dcc.RadioItems(
            options=[
                {'label': 'Manual Selection', 'value': 'manual'},
                {'label': 'Stock Screener', 'value': 'screener'}
            ],
            value='manual',
            id='selection-mode',
            className="bank-radio"
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
            # IPO Year - keep as dropdown
            html.Label("IPO Year Range:", className="bank-label"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='min-ipo', options=[''] + criteria['IPO Year'], placeholder="Min IPO Year"), width=6),
                dbc.Col(dcc.Dropdown(id='max-ipo', options=[''] + criteria['IPO Year'], placeholder="Max IPO Year"), width=6),
            ], className="bank-form-row"),
            
            # Price Range - numeric input
            html.Label(f"Price Range ($): {price_range}", className="bank-label"),
            dbc.Row([
                dbc.Col(dcc.Input(id='min-price', type='number', placeholder="Min Price", min=0, step=0.01, className="bank-input"), width=6),
                dbc.Col(dcc.Input(id='max-price', type='number', placeholder="Max Price", min=0, step=0.01, className="bank-input"), width=6),
            ], className="bank-form-row"),
            
            # Market Cap Range - numeric input
            html.Label(f"Market Cap Range ($): {market_cap_range}", className="bank-label"),
            dbc.Row([
                dbc.Col(dcc.Input(id='min-cap', type='number', placeholder="Min Market Cap", min=0, className="bank-input"), width=6),
                dbc.Col(dcc.Input(id='max-cap', type='number', placeholder="Max Market Cap", min=0, className="bank-input"), width=6),
            ], className="bank-form-row"),
            
            # Volume Range - numeric input
            html.Label(f"Volume Range: {volume_range}", className="bank-label"),
            dbc.Row([
                dbc.Col(dcc.Input(id='min-volume', type='number', placeholder="Min Volume", min=0, className="bank-input"), width=6),
                dbc.Col(dcc.Input(id='max-volume', type='number', placeholder="Max Volume", min=0, className="bank-input"), width=6),
            ], className="bank-form-row"),
            
            # Sector and Industry - keep as dropdown
            html.Label("Sector:", className="bank-label"),
            dcc.Dropdown(id='sector', options=[''] + criteria['Sector'], placeholder="Sector"),
            html.Label("Industry:", className="bank-label"),
            dcc.Dropdown(id='industry', options=[''] + criteria['Industry'], placeholder="Industry")
        ], id='screener-content', style={'display': 'none'}, className="bank-form"),
        html.Button("Run Selection", id='save-selection', n_clicks=0, className="bank-button"),
        html.Div(id='selection-output', className="bank-output")
    ], id='selection-content', className="bank-section")

# Callback to toggle between manual and screener mode
@callback(
    [Output('manual-content', 'style'),
     Output('screener-content', 'style')],
    Input('selection-mode', 'value')
)
def update_selection_mode(mode):
    """
    Update display based on selection mode.
    
    Args:
        mode (str): Selected mode ('manual' or 'screener').
        
    Returns:
        tuple: (manual_style, screener_style) display styles.
    """
    if mode == 'manual':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback to handle stock selection
@callback(
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
    State('industry', 'value'),
    State('tabs', 'value')
)
def save_selection(n_clicks, mode, manual_symbols, min_ipo, max_ipo, min_price, max_price, 
                   min_cap, max_cap, min_volume, max_volume, sector, industry, tab):
    """
    Save selected stocks to CSV based on criteria.
    
    Args:
        n_clicks (int): Number of button clicks.
        mode (str): Selection mode ('manual' or 'screener').
        manual_symbols (list): Manually selected symbols.
        min_ipo, max_ipo (str): IPO year range.
        min_price, max_price (float): Price range.
        min_cap, max_cap (float): Market cap range.
        min_volume, max_volume (int): Volume range.
        sector (str): Selected sector.
        industry (str): Selected industry.
        tab (str): Current active tab.
        
    Returns:
        str: Status message.
    """
    if n_clicks == 0 or tab != 'selection':
        return ""
    
    # Load the data fresh each time to ensure we have the latest
    screener_df = load_data(config.INPUT_FILE)
    
    # Convert data types for proper filtering
    if not screener_df.empty:
        # Convert price column to numeric, handling dollar signs
        if 'Last Sale' in screener_df.columns:
            screener_df['Last Sale'] = screener_df['Last Sale'].apply(convert_dollar_to_float)
        
        # Convert market cap column to numeric
        if 'Market Cap' in screener_df.columns:
            screener_df['Market Cap'] = pd.to_numeric(screener_df['Market Cap'], errors='coerce')
            
        # Convert volume column to numeric
        if 'Volume' in screener_df.columns:
            screener_df['Volume'] = pd.to_numeric(screener_df['Volume'], errors='coerce')
    
    filtered_df = screener_df.copy()
    
    if mode == 'manual':
        if not manual_symbols:
            return html.P("Please select at least one symbol.", className="bank-error")
        filtered_df = pd.DataFrame({'Symbol': manual_symbols})
    else:
        # Build filter conditions list to track what's being applied
        filter_conditions = []
        
        # Apply filters based on selected criteria
        if min_ipo and 'IPO Year' in filtered_df.columns:
            filter_conditions.append(f"IPO Year >= {min_ipo}")
            filtered_df = filtered_df[filtered_df['IPO Year'] >= int(min_ipo)]
        if max_ipo and 'IPO Year' in filtered_df.columns:
            filter_conditions.append(f"IPO Year <= {max_ipo}")
            filtered_df = filtered_df[filtered_df['IPO Year'] <= int(max_ipo)]
        
        # Only apply price filters if price column exists and has non-null values
        has_price_data = 'Last Sale' in filtered_df.columns and not filtered_df['Last Sale'].isna().all()
        if min_price is not None and has_price_data:
            filter_conditions.append(f"Price >= ${min_price}")
            filtered_df = filtered_df[filtered_df['Last Sale'] >= float(min_price)]
        if max_price is not None and has_price_data:
            filter_conditions.append(f"Price <= ${max_price}")
            filtered_df = filtered_df[filtered_df['Last Sale'] <= float(max_price)]
            
        # Only apply market cap filters if column exists and has non-null values
        has_market_cap_data = 'Market Cap' in filtered_df.columns and not filtered_df['Market Cap'].isna().all()
        if min_cap is not None and has_market_cap_data:
            filter_conditions.append(f"Market Cap >= ${min_cap}")
            filtered_df = filtered_df[filtered_df['Market Cap'] >= float(min_cap)]
        if max_cap is not None and has_market_cap_data:
            filter_conditions.append(f"Market Cap <= ${max_cap}")
            filtered_df = filtered_df[filtered_df['Market Cap'] <= float(max_cap)]
            
        # Only apply volume filters if column exists and has non-null values
        has_volume_data = 'Volume' in filtered_df.columns and not filtered_df['Volume'].isna().all()
        if min_volume is not None and has_volume_data:
            filter_conditions.append(f"Volume >= {min_volume}")
            filtered_df = filtered_df[filtered_df['Volume'] >= float(min_volume)]
        if max_volume is not None and has_volume_data:
            filter_conditions.append(f"Volume <= {max_volume}")
            filtered_df = filtered_df[filtered_df['Volume'] <= float(max_volume)]
            
        if sector and 'Sector' in filtered_df.columns:
            filter_conditions.append(f"Sector = {sector}")
            filtered_df = filtered_df[filtered_df['Sector'] == sector]
        if industry and 'Industry' in filtered_df.columns:
            filter_conditions.append(f"Industry = {industry}")
            filtered_df = filtered_df[filtered_df['Industry'] == industry]
        
        # Drop rows with NaN after filtering
        filtered_df = filtered_df.dropna(subset=['Symbol'])
        
        # Extract only the Symbol column for saving
        filtered_df = filtered_df[['Symbol']]
    
    if filtered_df.empty:
        if mode == 'screener':
            # Provide more detailed info about why no stocks matched
            if filter_conditions:
                applied_filters = "\n".join([f"- {f}" for f in filter_conditions])
                return html.Div([
                    html.P("No stocks match your criteria.", className="bank-error"),
                    html.P("Applied filters:", className="bank-text"),
                    html.Pre(applied_filters, className="bank-code"),
                    html.P("Try widening your filter ranges or using fewer filters.", className="bank-text")
                ], className="bank-notification bank-error-notification")
            else:
                return html.P("No stocks match your criteria. Please apply at least one filter.", className="bank-error")
        else:
            return html.P("No stocks selected. Please choose at least one stock symbol.", className="bank-error")
    
    # Save to CSV, overwriting existing file
    if os.path.exists(config.OUTPUT_FILE):
        os.remove(config.OUTPUT_FILE)
    filtered_df.to_csv(config.OUTPUT_FILE, index=False)
    
    return html.Div([
        html.P(f"Saved {len(filtered_df)} symbols to {config.OUTPUT_FILE}", className="bank-success"),
        html.P(f"Sample stocks selected: {', '.join(filtered_df['Symbol'].head(5).tolist())}...", className="bank-text")
    ], className="bank-notification bank-success-notification slide-in")