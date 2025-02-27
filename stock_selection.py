import pandas as pd
import os
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

# File paths
INPUT_FILE = "./stock_data/nasdaq_screener.csv"
OUTPUT_FILE = "./stock_data/stock_symbols.csv"

def load_data():
    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        # Ensure Symbol column is string and drop NaN
        df['Symbol'] = df['Symbol'].astype(str).replace('nan', '').dropna()
        return df
    else:
        print(f"Error: {INPUT_FILE} not found.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    exit()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Criteria for dropdowns
criteria = {
    'IPO Year': sorted(df['IPO Year'].dropna().astype(int).unique().tolist()),
    'Last Sale': sorted(df['Last Sale'].dropna().unique().tolist()),
    'Market Cap': sorted(df['Market Cap'].dropna().unique().tolist()),
    'Sector': sorted(df['Sector'].dropna().unique().tolist()),
    'Industry': sorted(df['Industry'].dropna().unique().tolist())
}

app.layout = html.Div([
    html.H3("Stock Selector"),
    dcc.RadioItems(
        options=[
            {'label': 'Manual Selection', 'value': 'manual'},
            {'label': 'Stock Screener', 'value': 'screener'}
        ],
        value='manual',
        id='mode-selector'
    ),
    html.Div(
        dcc.Dropdown(
            id='manual-dropdown',
            options=[{'label': s, 'value': s} for s in sorted(df['Symbol']) if s],  # Filter out empty strings
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
        dcc.Dropdown(id='sector', options=[''] + criteria['Sector'], placeholder="Sector"),
        dcc.Dropdown(id='industry', options=[''] + criteria['Industry'], placeholder="Industry")
    ], id='screener-content', style={'display': 'none'}),
    html.Button("Save Selection", id='save-button', n_clicks=0),
    html.Div(id='output-message')
])

@app.callback(
    [Output('manual-content', 'style'),
     Output('screener-content', 'style')],
    Input('mode-selector', 'value')
)
def update_selection_mode(mode):
    if mode == 'manual':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('output-message', 'children'),
    Input('save-button', 'n_clicks'),
    State('mode-selector', 'value'),
    State('manual-dropdown', 'value'),
    State('min-ipo', 'value'),
    State('max-ipo', 'value'),
    State('min-price', 'value'),
    State('max-price', 'value'),
    State('min-cap', 'value'),
    State('max-cap', 'value'),
    State('sector', 'value'),
    State('industry', 'value')
)
def save_selection(n_clicks, mode, manual_symbols, min_ipo, max_ipo, min_price, max_price, min_cap, max_cap, sector, industry):
    if n_clicks == 0:
        return ""
    
    filtered_df = df.copy()
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

if __name__ == "__main__":
    app.run_server(debug=True)