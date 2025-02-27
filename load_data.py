import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# Load data
df = pd.read_csv('./stock_data/stock_data_technical.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create target
df['Target_Price_Change_1'] = df['Price_Change'].shift(-1)
df = df.dropna(subset=['Target_Price_Change_1'])

# Features
features = [
    'TSMN', 'Close_Volatility', 'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment',
    'Close', 'MarketCap', 'Price_Change', 'Close_Diff_1'
]

# Function to predict price change for one day ahead
def predict_one_day(symbol, df, features, prediction_date='2025-02-25', train_start_date='2024-02-01'):
    symbol_data = df[df['Symbol'] == symbol]
    if symbol_data.empty:
        print(f"No data found for {symbol}")
        return None
    
    prediction_date = pd.to_datetime(prediction_date)
    train_start_date = pd.to_datetime(train_start_date)
    train_data = symbol_data[(symbol_data['Date'] >= train_start_date) & (symbol_data['Date'] <= prediction_date)]
    if len(train_data) < 10:
        print(f"Insufficient data for {symbol} from {train_start_date} to {prediction_date}: only {len(train_data)} days")
        return None
    
    # Training data
    X_train = train_data[features].fillna(train_data[features].median())
    y_train = train_data['Target_Price_Change_1']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = HistGradientBoostingRegressor(random_state=42, max_iter=300, max_depth=3, learning_rate=0.05)
    model.fit(X_train_scaled, y_train)
    
    # Prepare features for prediction
    pred_day_data = symbol_data[symbol_data['Date'] == prediction_date].copy()
    if pred_day_data.empty:
        print(f"No data for {symbol} on {prediction_date}")
        return None
    
    X_pred = pred_day_data[features].fillna(pred_day_data[features].median())
    X_pred_scaled = scaler.transform(X_pred)
    
    # Predict and scale
    pred_change = model.predict(X_pred_scaled)[0]
    volatility = train_data['Close_Volatility'].tail(3).mean()  # 3-day average
    pred_change = pred_change * volatility  # No cap
    
    # Actual next day
    next_day = symbol_data[symbol_data['Date'] == prediction_date + timedelta(days=1)]
    actual_change = next_day['Price_Change'].iloc[0] if not next_day.empty else None
    
    # Last 3 days for context
    recent_data = symbol_data[symbol_data['Date'] <= prediction_date].tail(3)
    result = pd.DataFrame({
        'Date': recent_data['Date'].tolist() + [prediction_date + timedelta(days=1)],
        'Price_Change': recent_data['Price_Change'].tolist() + [actual_change],
        'Predicted_Price_Change': [None] * 3 + [pred_change],
        'Type': ['Actual'] * 3 + ['Predicted']
    })
    
    # Accuracy check
    if actual_change is not None:
        trend_correct = (actual_change > 0) == (pred_change > 0)
        error = abs(actual_change - pred_change)
        print(f"Trend Correct: {trend_correct}, Absolute Error: {error:.4f}")
    else:
        print(f"Prediction for {symbol} on {prediction_date + timedelta(days=1)}: {pred_change:.4f}")
    
    return result

# Load symbols
symbols_df = pd.read_csv('./stock_data/stock_symbols.csv')
symbols = symbols_df['Symbol'].tolist()

# Predict for Feb 26 based on Feb 25
prediction_date = '2025-02-25'
for symbol in symbols:
    result = predict_one_day(symbol, df, features, prediction_date)
    if result is not None:
        print(f"\n{symbol} - Last 3 Days Actual and Next Day Prediction (Feb 26):")
        print(result)