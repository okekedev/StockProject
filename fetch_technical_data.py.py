import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import time

def fetch_technical_data_to_csv(symbols_file, start_date="2020-01-01", end_date=None, output_file="stock_data_technical.csv"):
    """
    Fetch technical data (OHLCV, splits, MarketCap, SP500, and enhanced indicators) using yfinance, save to CSV,
    excluding dividends and quarterly fundamentals, with multiple RSIs, TSMN with spectral analysis, price derivatives,
    volatility, market sentiment, and daily price change for short-term analysis.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Read stock symbols from CSV
    try:
        symbols_df = pd.read_csv(symbols_file)
        if 'Symbol' not in symbols_df.columns:
            raise ValueError("CSV must contain a 'Symbol' column")
        symbols = symbols_df['Symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from {symbols_file}: {symbols}")
    except Exception as e:
        print(f"Error reading symbols from {symbols_file}: {e}")
        return
    
    all_data = pd.DataFrame()
    
    # Buffer for technical indicators (60 days for RSI_60, 5 days for TSMN)
    buffer_days = 60
    fetch_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    # Fetch S&P 500
    try:
        sp500 = yf.Ticker('^GSPC').history(start=fetch_start, end=end_date).reset_index()
        sp500['Date'] = pd.to_datetime(sp500['Date']).dt.tz_localize(None)
        sp500 = sp500[['Date', 'Close']].rename(columns={'Close': 'SP500'})
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        sp500 = pd.DataFrame(columns=['Date', 'SP500'])
    
    for symbol in symbols:
        print(f"Fetching technical data for {symbol}...")
        stock = yf.Ticker(symbol)
        
        # Historical OHLCV data with retry
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                df_history = stock.history(start=fetch_start, end=end_date)
                if df_history.empty:
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
        
        # Stock splits
        splits = stock.splits.reset_index()
        splits.columns = ['Date', 'StockSplits']
        splits['Date'] = pd.to_datetime(splits['Date']).dt.tz_localize(None)
        df_history = df_history.merge(splits, on='Date', how='left').fillna({'StockSplits': 0})
        
        # Fundamentals (MarketCap only)
        info = stock.info
        df_history['MarketCap'] = info.get('marketCap', np.nan)
        
        # Merge S&P 500
        df_history = df_history.merge(sp500, on='Date', how='left').ffill().infer_objects(copy=False)
        
        # Technical indicators
        delta = df_history['Close'].diff()
        
        # Multiple RSI indicators
        def calculate_rsi(data, periods=[5, 10, 20, 30, 60]):
            rsi_values = {}
            for period in periods:
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss.replace(0, 1e-10)
                rsi = 100 - (100 / (1 + rs))
                rsi_values[f'RSI_{period}'] = rsi.fillna(50)
            return pd.DataFrame(rsi_values)
        
        rsi_df = calculate_rsi(df_history)
        df_history = pd.concat([df_history, rsi_df], axis=1)
        
        # Price derivatives and daily price change
        df_history['Close_Diff_1'] = df_history['Close'].diff().fillna(0)
        df_history['Price_Change'] = df_history['Close'].diff().fillna(0)  # Daily price change
        df_history['Close_Diff_2'] = df_history['Close_Diff_1'].diff().fillna(0)
        
        # Volatility
        df_history['Close_Volatility'] = df_history['Close'].rolling(window=5).std().fillna(df_history['Close'].std())
        
        # Enhanced TSMN with spectral analysis
        momentum_gradient = df_history['RSI_5'] - df_history['RSI_20']
        temp_gradient = momentum_gradient.diff(periods=5).fillna(0)
        temp_gradient_smoothed = temp_gradient.rolling(window=3).mean().fillna(0)
        
        close_diff = df_history['Close'].diff().fillna(0)
        vol_diff = df_history['Volume'].diff().fillna(0)
        from scipy.fft import fft
        spectral_close = np.abs(fft(close_diff.rolling(window=5).mean().fillna(0)))[:3].mean()
        spectral_vol = np.abs(fft(vol_diff.rolling(window=5).mean().fillna(0)))[:3].mean()
        spectral_energy = (spectral_close + spectral_vol) / 2
        
        vol_weight = df_history['Volume'] / df_history['Volume'].rolling(window=5).sum().replace(0, 1)
        market_cap_weight = np.log(1 + df_history['MarketCap'] / df_history['MarketCap'].rolling(window=5).sum().replace(0, 1))
        weighted_spectral = spectral_energy * vol_weight * market_cap_weight
        
        sp500_mean = df_history['SP500'].rolling(window=5).mean().fillna(df_history['SP500'].mean())
        sp500_std = df_history['SP500'].rolling(window=5).std().replace(0, 1).fillna(1)
        sp500_zscore = (df_history['SP500'] - sp500_mean) / sp500_std
        sp500_zscore = np.clip(sp500_zscore, -3, 3)
        market_stability = np.exp(-np.abs(sp500_zscore) / 3)
        
        df_history['TSMN_Spectral'] = temp_gradient_smoothed * weighted_spectral * market_stability
        df_history['TSMN'] = df_history['TSMN_Spectral'].rolling(window=5).mean().fillna(0)
        
        # Market sentiment
        df_history['Market_Sentiment'] = (df_history['SP500'].pct_change().fillna(0) + np.log(df_history['Volume'] + 1).diff().fillna(0)) / 2
        
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

def create_sample_symbols_csv(output_file="./stock_data/stock_symbols.csv"):
    """Create a sample CSV with 10 stock symbols if it doesn't exist."""
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'PYPL', 'INTC', 'AMD']
    if not os.path.exists(output_file):
        pd.DataFrame({'Symbol': symbols}).to_csv(output_file, index=False)
        print(f"Created sample symbols file at {output_file}")
    else:
        print(f"Symbols file already exists at {output_file}")

def main():
    start_date = "2020-01-01"
    output_dir = "./stock_data"
    symbols_file = os.path.join(output_dir, "stock_symbols.csv")
    output_file = os.path.join(output_dir, "stock_data_technical.csv")
    
    # Create sample symbols CSV if it doesn't exist
    create_sample_symbols_csv(symbols_file)
    
    # Fetch data using symbols from CSV
    data = fetch_technical_data_to_csv(symbols_file, start_date=start_date, output_file=output_file)

if __name__ == "__main__":
    main()