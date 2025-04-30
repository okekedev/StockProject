import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import sys
import os

# File paths
TECH_DATA_FILE = "./stock_data/stock_data_technical.csv"
TEST_STOCK_ACCURACY = "./stock_data/test_stock_accuracy.csv"

# Load historical data
df = pd.read_csv(TECH_DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Load accuracy data
def load_accuracy_data():
    if os.path.exists(TEST_STOCK_ACCURACY):
        try:
            accuracy_df = pd.read_csv(TEST_STOCK_ACCURACY)
            # Ensure necessary columns are present
            if 'Symbol' in accuracy_df.columns and 'Accuracy_%' in accuracy_df.columns:
                return accuracy_df[['Symbol', 'Accuracy_%']]
            else:
                print(f"Warning: {TEST_STOCK_ACCURACY} does not contain required columns 'Symbol' and 'Accuracy_%'. Using default accuracy of 50%.")
                return pd.DataFrame(columns=['Symbol', 'Accuracy_%'])
        except Exception as e:
            print(f"Error loading {TEST_STOCK_ACCURACY}: {e}. Using default accuracy of 50%.")
            return pd.DataFrame(columns=['Symbol', 'Accuracy_%'])
    else:
        print(f"Warning: {TEST_STOCK_ACCURACY} not found. Using default accuracy of 50%.")
        return pd.DataFrame(columns=['Symbol', 'Accuracy_%'])

accuracy_df = load_accuracy_data()

# Create target (shifted price change for training, will be NaN for the last row)
df['Target_Price_Change_1'] = df['Price_Change'].shift(-1)
# Do not drop NaN here; we'll handle it later to allow predictions on the latest date

# Features
features = [
    'TSMN', 'Close_Volatility', 'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment',
    'Close', 'MarketCap', 'Price_Change'
]

# Function to predict price change for one day ahead
def predict_one_day(symbol, df, features, prediction_date, target_date, train_start_date='2020-01-01'):
    symbol_data = df[df['Symbol'] == symbol].copy()
    if symbol_data.empty:
        return None
    
    prediction_date = pd.to_datetime(prediction_date)
    train_start_date = pd.to_datetime(train_start_date)
    
    # Determine the most recent date available in the data
    max_date = symbol_data['Date'].max()
    
    # If prediction_date is in the future, use the most recent date for prediction
    if prediction_date > max_date:
        pred_date = max_date
    else:
        pred_date = prediction_date
    
    # Training data up to and including the prediction date (or most recent date)
    train_data = symbol_data[(symbol_data['Date'] >= train_start_date) & 
                             (symbol_data['Date'] <= pred_date)]
    if len(train_data) < 10:
        return None
    
    # Drop NaN in Target_Price_Change_1 for training purposes
    train_data = train_data.dropna(subset=['Target_Price_Change_1'])
    if len(train_data) < 10:
        return None
    
    # Training data
    X_train = train_data[features].fillna(train_data[features].median())
    y_train = train_data['Target_Price_Change_1']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = HistGradientBoostingRegressor(random_state=42, max_iter=200, max_depth=3, learning_rate=0.05)
    model.fit(X_train_scaled, y_train)
    
    # Prepare features for prediction using the most recent date
    pred_day_data = symbol_data[symbol_data['Date'] == pred_date].copy()
    if pred_day_data.empty:
        return None
    
    X_pred = pred_day_data[features].fillna(pred_day_data[features].median())
    X_pred_scaled = scaler.transform(X_pred)
    
    # Predict and scale
    pred_change = model.predict(X_pred_scaled)[0]
    volatility = pred_day_data['Close_Volatility'].values[0]
    close_price = pred_day_data['Close'].iloc[0]
    pred_change = pred_change * min(volatility, 3)  # Cap at 3
    
    # Calculate predicted percentage change
    percent_change = (pred_change / close_price) * 100 if close_price != 0 else 0
    
    return {
        'Symbol': symbol,
        'Prediction_Date': pred_date.strftime('%Y-%m-%d'),
        'Target_Date': target_date.strftime('%Y-%m-%d'),
        'Close': close_price,
        'Predicted_Price_Change': pred_change,
        'percent_predicted_change': percent_change
    }

def run_predictions():
    # Load symbols
    symbols_df = pd.read_csv('./stock_data/stock_symbols.csv')
    symbols = symbols_df['Symbol'].tolist()
    
    # Determine the next business day for prediction
    today = datetime.now().date()
    target_date = today + timedelta(days=1)
    
    # Adjust target_date to the next business day (skip weekends)
    while target_date.weekday() in [5, 6]:  # Saturday or Sunday
        target_date += timedelta(days=1)
    
    # Prediction date is the day before the target date
    prediction_date = target_date - timedelta(days=1)
    
    # Make predictions for the next business day
    predictions = []
    for symbol in symbols:
        result = predict_one_day(symbol, df, features, prediction_date, target_date)
        if result is not None:
            predictions.append(result)
    
    # Combine all predictions
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        
        # Merge with accuracy data
        if not accuracy_df.empty:
            predictions_df = predictions_df.merge(accuracy_df, on='Symbol', how='left')
            # Fill missing accuracy with a default value (e.g., 50%)
            predictions_df['Accuracy_%'] = predictions_df['Accuracy_%'].fillna(50.0)
        else:
            # If no accuracy data is available, use a default accuracy of 50%
            predictions_df['Accuracy_%'] = 50.0
        
        # Calculate the new metric: percent_predicted_change * (Accuracy_% / 100)
        predictions_df['weighted_movement'] = predictions_df['percent_predicted_change'] * (predictions_df['Accuracy_%'] / 100)
        
        # Sort by weighted_movement (descending) and select top 10
        predictions_df = predictions_df.sort_values(by='weighted_movement', ascending=False)
        top_picks_df = predictions_df.head(10)
        
        # Save all results to CSV, overwriting the file
        output_path = './stock_data/top_10_upward_picks.csv'
        if os.path.exists(output_path):
            os.remove(output_path)  # Overwrite existing file
        top_picks_df.to_csv(output_path, index=False)
        
        # Print all results to terminal
        print(f"Top 10 Predictions for {target_date.strftime('%Y-%m-%d')} (using data from {prediction_date.strftime('%Y-%m-%d')}):")
        print("\nDetailed prediction results:")
        print(top_picks_df[['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_Price_Change', 'percent_predicted_change', 'Accuracy_%', 'weighted_movement']].to_string(index=False))
    else:
        # Save an empty CSV if no predictions
        output_path = './stock_data/top_10_upward_picks.csv'
        if os.path.exists(output_path):
            os.remove(output_path)
        pd.DataFrame(columns=['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_Price_Change', 'percent_predicted_change']).to_csv(output_path, index=False)
        print(f"Predictions for {target_date.strftime('%Y-%m-%d')} (using data from {prediction_date.strftime('%Y-%m-%d')}):")
        print("No predictions generated due to data issues.")

if __name__ == "__main__":
    run_predictions()