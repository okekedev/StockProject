import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import time  # For retry logic

def fetch_technical_data_to_csv(symbols, start_date="2020-01-01", end_date=None, output_file="stock_data_technical.csv"):
    """
    Fetch technical data (OHLCV, splits, MarketCap, SP500, and indicators) using yfinance, save to CSV, excluding dividends, 
    quarterly fundamentals, and S&P 500 correlation, with 20-day and 5-day RSIs and the Temporal-Spectral Momentum Nexus (TSMN) for short-term analysis.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = pd.DataFrame()
    
    # Buffer for technical indicators (20 days for RSI_20, 5 days for RSI_5 and TSMN)
    buffer_days = 20  # Maintain 20 days to support 20-day RSI and custom indicators
    fetch_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    # Fetch S&P 500 (included in output, no correlation calculated)
    try:
        sp500 = yf.Ticker('^GSPC').history(start=fetch_start, end=end_date).reset_index()
        sp500['Date'] = pd.to_datetime(sp500['Date']).dt.tz_localize(None)
        sp500 = sp500[['Date', 'Close']].rename(columns={'Close': 'SP500'})  # Rename immediately
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        sp500 = pd.DataFrame(columns=['Date', 'SP500'])  # Empty DataFrame as fallback
    
    for symbol in symbols:
        print(f"Fetching technical data for {symbol}...")
        stock = yf.Ticker(symbol)
        
        # Historical OHLCV data with retry and adjusted date range for delisted stocks
        max_retries = 3
        retry_delay = 5  # seconds
        for attempt in range(max_retries):
            try:
                df_history = stock.history(start=fetch_start, end=end_date)
                if df_history.empty:
                    # Try a narrower date range if initial fetch fails (e.g., start from 2020-01-01)
                    adjusted_start = "2020-01-01"
                    adjusted_fetch_start = (datetime.strptime(adjusted_start, '%Y-%m-%d') - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
                    df_history = stock.history(start=adjusted_fetch_start, end=end_date)
                    if df_history.empty:
                        print(f"No historical data for {symbol} after retry with adjusted range")
                        break
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for {symbol}, skipping...")
                    continue
        
        if df_history.empty:
            continue
        
        df_history = df_history.reset_index()
        df_history['Date'] = pd.to_datetime(df_history['Date']).dt.tz_localize(None)
        df_history = df_history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_history['Symbol'] = symbol
        
        # Remove dividends (no longer included)
        
        # Stock splits
        splits = stock.splits.reset_index()
        splits.columns = ['Date', 'StockSplits']
        splits['Date'] = pd.to_datetime(splits['Date']).dt.tz_localize(None)
        df_history = df_history.merge(splits, on='Date', how='left').fillna({'StockSplits': 0})
        
        # Fundamentals (keeping only MarketCap, added before custom indicators)
        info = stock.info
        fundamentals = {
            'MarketCap': info.get('marketCap', np.nan),
        }
        for key, value in fundamentals.items():
            df_history[key] = value  # Add MarketCap to df_history before calculations
        
        # Merge S&P 500 (included in output, before TSMN calculation)
        df_history = df_history.merge(sp500, on='Date', how='left').ffill().infer_objects(copy=False)
        
        # Technical indicators (use 'Close' for stock price, keeping 20-day and 5-day RSIs and TSMN)
        delta = df_history['Close'].diff()
        # 5-day RSI for short-term analysis
        gain_5 = (delta.where(delta > 0, 0)).rolling(window=5).mean()
        loss_5 = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
        rs_5 = gain_5 / loss_5.replace(0, np.nan)
        df_history['RSI_5'] = 100 - (100 / (1 + rs_5)).fillna(50)  # Fill NaN with 50 (neutral RSI) for initial periods
        # 20-day RSI for longer-term context within short-term focus
        gain_20 = (delta.where(delta > 0, 0)).rolling(window=20).mean()
        loss_20 = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
        rs_20 = gain_20 / loss_20.replace(0, np.nan)
        df_history['RSI_20'] = 100 - (100 / (1 + rs_20)).fillna(50)  # Fill NaN with 50 (neutral RSI) for initial periods
        
        # Custom Indicator: Temporal-Spectral Momentum Nexus (TSMN)
        # Temporal Momentum Gradient
        momentum_gradient = df_history['RSI_5'] - df_history['RSI_20']
        temp_gradient = momentum_gradient.diff(periods=5).fillna(0)
        temp_gradient_smoothed = temp_gradient.rolling(window=3).mean().fillna(0)
        
        # Spectral Oscillations (simplified to return scalar using sum of cosine-weighted differences)
        close_diff = df_history['Close'].diff().fillna(0)
        vol_diff = df_history['Volume'].diff().fillna(0)
        spectral_close = close_diff.rolling(window=5).apply(lambda x: np.sum(x ** 2 * np.cos(2 * np.pi * 2 / 5))).fillna(0)  # Use fixed frequency for scalar output
        spectral_vol = vol_diff.rolling(window=5).apply(lambda x: np.sum(x ** 2 * np.cos(2 * np.pi * 2 / 5))).fillna(0)  # Use fixed frequency for scalar output
        spectral_energy = (spectral_close + spectral_vol) / (spectral_close.rolling(window=5).sum().replace(0, 1) + spectral_vol.rolling(window=5).sum().replace(0, 1))
        
        # Stochastic Market Adjustment
        vol_weight = df_history['Volume'] / df_history['Volume'].rolling(window=5).sum().replace(0, 1)
        market_cap_weight = np.log(1 + df_history['MarketCap'] / df_history['MarketCap'].rolling(window=5).sum().replace(0, 1))
        weighted_spectral = spectral_energy * vol_weight * market_cap_weight
        
        sp500_mean = df_history['SP500'].rolling(window=5).mean().fillna(df_history['SP500'].mean())
        sp500_std = df_history['SP500'].rolling(window=5).std().replace(0, 1).fillna(1)
        sp500_zscore = (df_history['SP500'] - sp500_mean) / sp500_std
        sp500_zscore = np.clip(sp500_zscore, -3, 3)  # Cap at ±3 for stability
        market_stability = np.exp(-np.abs(sp500_zscore) / 3)
        
        # TSMN Formula (ensure scalar output)
        tsmn = temp_gradient_smoothed * weighted_spectral * market_stability
        df_history['TSMN'] = tsmn.rolling(window=5).mean().fillna(0)  # 5-day EMA for short-term focus
        
        # Trim to requested start_date
        df_history = df_history[df_history['Date'] >= start_date]
        
        # Append to all_data
        all_data = pd.concat([all_data, df_history], ignore_index=True)
    
    if all_data.empty:
        print("No data fetched.")
        return
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    all_data.to_csv(output_file, index=False)
    print(f"Technical data saved to {output_file}")
    
    return all_data

def main():
    symbols = ['AAPL', 'MSFT', 'GOOG']
    start_date = "2020-01-01"
    output_dir = "./stock_data"
    output_file = os.path.join(output_dir, "stock_data_technical.csv")
    
    data = fetch_technical_data_to_csv(symbols, start_date=start_date, output_file=output_file)

if __name__ == "__main__":
    main()