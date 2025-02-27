import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import sys
import os

# Load data
df = pd.read_csv('./stock_data/stock_data_technical.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Create target (shifted price change for training, will be NaN for the last row)
df['Target_Price_Change_1'] = df['Price_Change'].shift(-1)
# Do not drop NaN here; we'll handle it later to allow predictions on the latest date

# Features
features = [
    'TSMN', 'Close_Volatility', 'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment',
    'Close', 'MarketCap', 'Price_Change'
]

# Function to predict price change for one day ahead
def predict_one_day(symbol, df, features, prediction_date, train_start_date='2020-01-01'):
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
    
    # Get actual change for the next day
    actual_change = symbol_data[symbol_data['Date'] == pred_date + timedelta(days=1)]['Price_Change']
    actual_change = actual_change.iloc[0] if not actual_change.empty else np.nan
    
    # Calculate actual percentage change
    actual_percent_change = (actual_change / close_price) * 100 if close_price != 0 and not pd.isna(actual_change) else np.nan
    
    return {
        'Symbol': symbol,
        'Prediction_Date': pred_date.strftime('%Y-%m-%d'),
        'Target_Date': (pred_date + timedelta(days=1)).strftime('%Y-%m-%d'),
        'Close': close_price,
        'Predicted_Price_Change': pred_change,
        'percent_predicted_change': percent_change,
        'Actual_%_Change': actual_percent_change
    }

def run_predictions(target_start_date, target_end_date):
    # Load symbols
    symbols_df = pd.read_csv('./stock_data/stock_symbols.csv')
    symbols = symbols_df['Symbol'].tolist()
    
    # Convert target dates to prediction dates (day before)
    target_start_date = pd.to_datetime(target_start_date)
    target_end_date = pd.to_datetime(target_end_date)
    prediction_start_date = target_start_date - timedelta(days=1)
    prediction_end_date = target_end_date - timedelta(days=1)
    
    # Generate prediction dates (business days only)
    dates = pd.date_range(start=prediction_start_date, end=prediction_end_date, freq='B')  # Business days
    
    # Make predictions for each date
    all_predictions = []
    for pred_date in dates:
        target_date = pred_date + timedelta(days=1)
        predictions = []
        for symbol in symbols:
            result = predict_one_day(symbol, df, features, pred_date)
            if result is not None:
                predictions.append(result)
        
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            all_predictions.append(predictions_df)
    
    # Combine all predictions
    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)
        upward_predictions = final_df[final_df['percent_predicted_change'] > 0]
        total_days = len(final_df['Prediction_Date'].unique())
        upward_days = len(upward_predictions['Prediction_Date'].unique())
        
        # Save all results to CSV, overwriting the file
        output_path = './stock_data/top_10_upward_picks.csv'
        if os.path.exists(output_path):
            os.remove(output_path)  # Overwrite existing file
        final_df.to_csv(output_path, index=False)
        
        # Print all results to terminal
        print(f"Predictions for target dates {target_start_date.strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')} (using data from {prediction_start_date.strftime('%Y-%m-%d')} to {prediction_end_date.strftime('%Y-%m-%d')}):")
        print(f"Total prediction days: {total_days}")
        print(f"Days with upward predictions: {upward_days} out of {total_days} ({(upward_days / total_days * 100):.2f}%)")
        print("\nDetailed prediction results:")
        print(final_df[['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_Price_Change', 'percent_predicted_change', 'Actual_%_Change']].to_string(index=False))
    else:
        # Save an empty CSV if no predictions
        output_path = './stock_data/top_10_upward_picks.csv'
        if os.path.exists(output_path):
            os.remove(output_path)
        pd.DataFrame(columns=['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_Price_Change', 'percent_predicted_change', 'Actual_%_Change']).to_csv(output_path, index=False)
        print(f"Predictions for target dates {target_start_date.strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')} (using data from {prediction_start_date.strftime('%Y-%m-%d')} to {prediction_end_date.strftime('%Y-%m-%d')}):")
        print("No predictions generated due to data issues.")

if __name__ == "__main__":
    # Get date range from command-line arguments or default
    target_start_date = '2025-02-27'  # Default target start date
    target_end_date = '2025-02-28'    # Default target end date
    if len(sys.argv) > 1:
        target_start_date = sys.argv[1]
    if len(sys.argv) > 2:
        target_end_date = sys.argv[2]
    
    run_predictions(target_start_date, target_end_date)