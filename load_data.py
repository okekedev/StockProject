import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load the data (using your specified path)
df = pd.read_csv('./stock_data/stock_data_technical.csv')

# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Create target variable: 1-day percentage change in Close (regression)
df['Target_Pct_Change'] = df['Close'].pct_change(periods=-1) * 100  # (Close_t+1 - Close_t) / Close_t * 100

# Drop rows with NaN in Target (first row or where data is incomplete)
df = df.dropna(subset=['Target_Pct_Change'])

# Feature engineering: reduce Close's direct impact, enhance TSMN
features = ['TSMN', 'RSI_20', 'RSI_5', 'Volume', 'MarketCap', 'SP500']
# Log-transform Close and Volume for non-linear scaling, reduce dominance
df['Log_Close'] = np.log(df['Close'] + 1)  # Avoid log(0) with +1
df['Log_Volume'] = np.log(df['Volume'] + 1)
df['TSMN_Diff'] = df['TSMN'].diff().fillna(0)  # Add TSMN difference for momentum
df['SP500_Zscore'] = (df['SP500'] - df['SP500'].rolling(window=5).mean()) / df['SP500'].rolling(window=5).std().replace(0, 1).fillna(1)

# Updated features list with engineered features
updated_features = ['TSMN', 'TSMN_Diff', 'RSI_20', 'RSI_5', 'Log_Close', 'Log_Volume', 'MarketCap', 'SP500', 'SP500_Zscore']
X = df[updated_features]
y = df['Target_Pct_Change']

# Split data: train on 2020-01-01 to 2025-01-31, test on February 2025 (2025-02-01 to 2025-02-28)
train_mask = df['Date'] < pd.to_datetime('2025-02-01')
test_mask = (df['Date'] >= pd.to_datetime('2025-02-01')) & (df['Date'] <= pd.to_datetime('2025-02-28'))

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Handle any remaining NaN values (fill with median)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features for better model performance (CPU-efficient)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest Regressor (CPU-efficient, non-linear model)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # n_jobs=-1 uses all CPU cores for efficiency

# Train the model
rf.fit(X_train_scaled, y_train)

# Predict on test set (February 2025)
y_pred = rf.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error on February 2025 percentage change: {mse:.2f}")
print(f"Mean Absolute Error on February 2025 percentage change: {mae:.2f}%")

# Feature importance (from Random Forest)
feature_importance = pd.DataFrame({'feature': updated_features, 'importance': rf.feature_importances_})
print("\nFeature Importance (Random Forest):")
print(feature_importance.sort_values('importance', ascending=False))

# Save the model for future use (optional, using joblib for CPU efficiency)
import joblib
joblib.dump(rf, './stock_data/rf_model_pct_change.joblib')
joblib.dump(scaler, './stock_data/scaler_pct_change.joblib')

# Validate on February 2025 GOOG sample (example from your data)
goog_sample = pd.DataFrame({
    'Date': ['2025-02-13', '2025-02-14', '2025-02-18', '2025-02-19', '2025-02-20', '2025-02-21', '2025-02-24'],
    'Open': [185.92999267578125, 186.8300018310547, 187.44000244140625, 186.18499755859375, 186.5, 187.2899932861328, 183.8000030517578],
    'High': [187.99000549316406, 188.14999389648438, 187.77999877929688, 187.36000061035156, 187.1199951171875, 187.47000122070312, 185.08999633789062],
    'Low': [184.8800048828125, 186.11000061035156, 183.5800018310547, 185.5, 184.60000610351562, 181.1300048828125, 180.8800048828125],
    'Close': [187.8800048828125, 186.8699951171875, 185.8000030517578, 187.1300048828125, 186.63999938964844, 181.5800018310547, 181.19000244140625],
    'Volume': [12729300, 12714200, 19796000, 13120500, 12063800, 19520800, 18734000],
    'Symbol': ['GOOG'] * 7,
    'StockSplits': [0.0] * 7,
    'MarketCap': [2149422727168] * 7,
    'SP500': [6115.06982421875, 6114.6298828125, 6129.580078125, 6144.14990234375, 6117.52001953125, 6013.1298828125, 5983.25],
    'RSI_20': [28.192821880469268, 48.14812489142015, 33.56172687254136, 50.39998209642702, 59.527579868078, 14.843759313218769, 15.94727116380335],
    'RSI_5': [43.087213657151054, 44.13320308196149, 40.55314396307466, 39.82747436255319, 39.11913185867285, 36.39250376970907, 33.87324064154856],
    'TSMN': [-0.13477538884428347, -0.11418062232684016, -0.07902765470713928, -0.03291979615105654, 0.04024207516661034, 0.08241478443420577, 0.09349540281809228]
})

goog_sample['Date'] = pd.to_datetime(goog_sample['Date'])

# Engineer features for GOOG sample
goog_sample['Log_Close'] = np.log(goog_sample['Close'] + 1)
goog_sample['Log_Volume'] = np.log(goog_sample['Volume'] + 1)
goog_sample['TSMN_Diff'] = goog_sample['TSMN'].diff().fillna(0)
goog_sample['SP500_Zscore'] = (goog_sample['SP500'] - goog_sample['SP500'].rolling(window=5).mean()) / goog_sample['SP500'].rolling(window=5).std().replace(0, 1).fillna(1)

# Calculate features for prediction
updated_features = ['TSMN', 'TSMN_Diff', 'RSI_20', 'RSI_5', 'Log_Close', 'Log_Volume', 'MarketCap', 'SP500', 'SP500_Zscore']
X_sample = goog_sample[updated_features].fillna(goog_sample[updated_features].median())

# Scale features using the same scaler
X_sample_scaled = scaler.transform(X_sample)

# Predict 1-day percentage change for GOOG February 2025
predicted_pct_changes = rf.predict(X_sample_scaled)

# Calculate actual 1-day percentage changes for validation
actual_pct_changes = goog_sample['Close'].pct_change(periods=-1) * 100
actual_pct_changes = actual_pct_changes.dropna()[:-1]  # Remove last value (no next day), align with predictions

print("\nGOOG February 2025 Predicted vs Actual Percentage Changes (%):")
print(pd.DataFrame({
    'Date': goog_sample['Date'][:-1],
    'Actual_Pct_Change': actual_pct_changes,
    'Predicted_Pct_Change': predicted_pct_changes
}))

# Calculate errors for the sample
mse_sample = mean_squared_error(actual_pct_changes, predicted_pct_changes)
mae_sample = mean_absolute_error(actual_pct_changes, predicted_pct_changes)
print(f"\nMean Squared Error for GOOG Feb 2025 percentage change: {mse_sample:.2f}%^2")
print(f"Mean Absolute Error for GOOG Feb 2025 percentage change: {mae_sample:.2f}%")