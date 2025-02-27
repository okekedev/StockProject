import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os
import sys

# Load data
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    raise FileNotFoundError(f"Data file not found: {file_path}")

# Create target and features
def prepare_data(df):
    df['Target_Price_Change_1'] = df['Price_Change'].shift(-1)
    # Do not drop NaN here; we'll handle it in predict_one_day
    features = [
        'TSMN', 'Close_Volatility', 'RSI_5', 'RSI_20', 'SP500', 'Volume', 'Market_Sentiment',
        'Close', 'MarketCap', 'Price_Change'
    ]
    return df, features

# Function to predict price change for one day ahead
def predict_one_day(symbol, df, features, prediction_date, train_start_date='2020-01-01'):
    symbol_data = df[df['Symbol'] == symbol].copy()
    if symbol_data.empty:
        return None
    
    prediction_date = pd.to_datetime(prediction_date)
    train_start_date = pd.to_datetime(train_start_date)
    
    # Training data up to and including prediction date
    train_data = symbol_data[(symbol_data['Date'] >= train_start_date) & 
                             (symbol_data['Date'] <= prediction_date)]
    if len(train_data) < 10:
        return None
    
    # Prediction day data
    pred_day_data = symbol_data[symbol_data['Date'] == prediction_date]
    if pred_day_data.empty:
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
    
    # Features for prediction
    X_pred = pred_day_data[features].fillna(pred_day_data[features].median())
    X_pred_scaled = scaler.transform(X_pred)
    
    # Predict and scale
    pred_change = model.predict(X_pred_scaled)[0]
    volatility = pred_day_data['Close_Volatility'].values[0]
    close_price = pred_day_data['Close'].iloc[0]
    pred_change = pred_change * min(volatility, 3)
    
    # Get actual change
    actual_change = symbol_data[symbol_data['Date'] == prediction_date + timedelta(days=1)]['Price_Change']
    actual_change = actual_change.iloc[0] if not actual_change.empty else np.nan
    
    # Calculate actual percentage change
    actual_percent_change = (actual_change / close_price) * 100 if close_price != 0 and not pd.isna(actual_change) else np.nan
    
    percent_change = (pred_change / close_price) * 100 if close_price != 0 else 0
    
    return {
        'Symbol': symbol,
        'Prediction_Date': prediction_date.strftime('%Y-%m-%d'),
        'Close': close_price,
        'Predicted_%_Change': percent_change,
        'Actual_%_Change': actual_percent_change
    }

def run_test(target_start_date, target_end_date):
    # Load technical data
    data_file = './stock_data/stock_data_technical.csv'
    df = load_data(data_file)
    
    # Prepare data
    df, features = prepare_data(df)
    
    # Load symbols
    symbols_file = './stock_data/stock_symbols.csv'
    symbols_df = pd.read_csv(symbols_file)
    symbols = symbols_df['Symbol'].tolist()
    
    # Convert target dates to prediction dates (day before)
    target_start_date = pd.to_datetime(target_start_date)
    target_end_date = pd.to_datetime(target_end_date)
    prediction_start_date = target_start_date - timedelta(days=1)
    prediction_end_date = target_end_date - timedelta(days=1)
    
    # Generate prediction dates (exclude Mondays and weekends)
    dates = pd.date_range(start=prediction_start_date, end=prediction_end_date, freq='B')  # Business days
    valid_dates = [date for date in dates if date.weekday() != 0]  # Exclude Mondays
    
    # Make predictions for each valid date
    all_predictions = []
    for pred_date in valid_dates:
        target_date = pred_date + timedelta(days=1)
        # Ensure target_date is not a Monday or weekend (must have data)
        if target_date.weekday() in [0, 5, 6]:  # Skip if target is Mon, Sat, or Sun
            continue
        
        predictions = []
        for symbol in symbols:
            result = predict_one_day(symbol, df, features, pred_date)
            if result is not None:
                predictions.append(result)
        
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            predictions_df['Target_Date'] = target_date.strftime('%Y-%m-%d')
            predictions_df['Trend_Success'] = (
                ((predictions_df['Predicted_%_Change'] > 0) & (predictions_df['Actual_%_Change'] > 0)) |
                ((predictions_df['Predicted_%_Change'] < 0) & (predictions_df['Actual_%_Change'] < 0))
            )
            all_predictions.append(predictions_df)
    
    # Combine all predictions
    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)
        final_df = final_df.dropna(subset=['Actual_%_Change'])  # Drop rows where actual change is NaN
        if final_df.empty:
            # Save empty CSVs if no valid predictions after dropping NaN
            output_path = './stock_data/test_results.csv'
            if os.path.exists(output_path):
                os.remove(output_path)
            pd.DataFrame(columns=['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_%_Change', 'Actual_%_Change', 'Trend_Success']).to_csv(output_path, index=False)
            per_stock_accuracy_path = './stock_data/test_stock_accuracy.csv'
            if os.path.exists(per_stock_accuracy_path):
                os.remove(per_stock_accuracy_path)
            pd.DataFrame(columns=['Symbol', 'Successful_Trends', 'Total_Predictions', 'Accuracy_%']).to_csv(per_stock_accuracy_path, index=False)
            print(f"Predictions for target dates {target_start_date.strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')} (using data from {prediction_start_date.strftime('%Y-%m-%d')} to {prediction_end_date.strftime('%Y-%m-%d')}):")
            print("No valid predictions generated due to missing actual data.")
            return
        
        successful_trends = final_df['Trend_Success'].sum()
        total_predictions = len(final_df)
        success_rate = (successful_trends / total_predictions * 100) if total_predictions > 0 else 0
        
        # Calculate per-stock accuracy
        per_stock_accuracy = final_df.groupby('Symbol').agg({
            'Trend_Success': ['sum', 'count']
        }).reset_index()
        per_stock_accuracy.columns = ['Symbol', 'Successful_Trends', 'Total_Predictions']
        per_stock_accuracy['Accuracy_%'] = (per_stock_accuracy['Successful_Trends'] / per_stock_accuracy['Total_Predictions'] * 100).round(2)
        
        # Save all results to CSV, overwriting the file
        output_path = './stock_data/test_results.csv'
        if os.path.exists(output_path):
            os.remove(output_path)  # Overwrite existing file
        final_df.to_csv(output_path, index=False)
        
        # Save per-stock accuracy to a separate CSV for UI display
        per_stock_accuracy_path = './stock_data/test_stock_accuracy.csv'
        if os.path.exists(per_stock_accuracy_path):
            os.remove(per_stock_accuracy_path)
        per_stock_accuracy.to_csv(per_stock_accuracy_path, index=False)
        
        # Print all results to terminal
        print(f"Predictions for target dates {target_start_date.strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')} (using data from {prediction_start_date.strftime('%Y-%m-%d')} to {prediction_end_date.strftime('%Y-%m-%d')}):")
        print(f"Total predictions made: {total_predictions}")
        print(f"Overall successful trend predictions: {successful_trends} out of {total_predictions} ({success_rate:.2f}%)")
        print("\nPer-Stock Accuracy:")
        print(per_stock_accuracy[['Symbol', 'Successful_Trends', 'Total_Predictions', 'Accuracy_%']].to_string(index=False))
        print("\nDetailed prediction results:")
        print(final_df[['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_%_Change', 'Actual_%_Change', 'Trend_Success']].to_string(index=False))
    else:
        # Save an empty CSV if no predictions
        output_path = './stock_data/test_results.csv'
        if os.path.exists(output_path):
            os.remove(output_path)
        pd.DataFrame(columns=['Symbol', 'Prediction_Date', 'Target_Date', 'Close', 'Predicted_%_Change', 'Actual_%_Change', 'Trend_Success']).to_csv(output_path, index=False)
        # Save an empty per-stock accuracy CSV
        per_stock_accuracy_path = './stock_data/test_stock_accuracy.csv'
        if os.path.exists(per_stock_accuracy_path):
            os.remove(per_stock_accuracy_path)
        pd.DataFrame(columns=['Symbol', 'Successful_Trends', 'Total_Predictions', 'Accuracy_%']).to_csv(per_stock_accuracy_path, index=False)
        print(f"Predictions for target dates {target_start_date.strftime('%Y-%m-%d')} to {target_end_date.strftime('%Y-%m-%d')} (using data from {prediction_start_date.strftime('%Y-%m-%d')} to {prediction_end_date.strftime('%Y-%m-%d')}):")
        print("No predictions generated due to data issues.")

def main():
    # Check for command-line arguments for date range
    target_start_date = '2025-02-20'  # Default target start date
    target_end_date = '2025-02-21'    # Default target end date
    if len(sys.argv) > 1:
        target_start_date = sys.argv[1]
    if len(sys.argv) > 2:
        target_end_date = sys.argv[2]
    
    run_test(target_start_date, target_end_date)

if __name__ == "__main__":
    main()