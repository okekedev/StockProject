"""
Simplified AI+ Technical Data Module with fixed type handling

This module handles fetching and basic processing of stock data for AI analysis,
focusing on preparing clean raw data rather than calculating complex indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import config

# Constants
DATA_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

def get_aiplus_technical_data(symbol, timeframe='1mo', force_refresh=False):
    """
    Get technical data for AI analysis of a stock.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Time period - '1mo', '3mo', '6mo', '1y'
        force_refresh (bool): Whether to force refresh data
        
    Returns:
        dict: Technical data results
    """
    # Create cache key and file path
    cache_key = f"{symbol}_{timeframe}"
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_key}.json")
    
    # Use cached data if available and not forcing refresh
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is fresh (within 4 hours)
            if 'timestamp' in cache_data:
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < 14400:
                    return cache_data
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
    
    try:
        # Get company info using Ticker
        stock = yf.Ticker(symbol)
        try:
            info = stock.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
        except Exception as e:
            print(f"Warning: Could not get company info for {symbol}: {e}")
            company_name = symbol
            sector = "Unknown"
            industry = "Unknown"
        
        # Convert timeframe to period parameter
        if timeframe == '1mo':
            period = "1mo"
            days = 30
        elif timeframe == '3mo':
            period = "3mo"
            days = 90
        elif timeframe == '6mo':
            period = "6mo"
            days = 180
        elif timeframe == '1y':
            period = "1y"
            days = 365
        else:
            period = "1mo"
            days = 30
        
        # Get historical data with period parameter
        df = yf.download(symbol, period=period, progress=False)
        
        if df.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Fix timezone issues by resetting index and removing timezone
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Basic metrics - ensure they're converted to Python native types
        current_price = float(df['Close'].iloc[-1])
        start_price = float(df['Close'].iloc[0])  
        price_change = float((current_price - start_price) / start_price * 100)
        
        # Calculate volatility
        daily_returns = df['Close'].pct_change().dropna()
        volatility = float(daily_returns.std() * (252 ** 0.5) * 100)  # Annualized

        # Calculate basic indicators
        basic_indicators = {}
        
        # Moving averages
        for window in [20, 50, 200]:
            if len(df) >= window:
                try:
                    ma_value = float(df['Close'].rolling(window=window).mean().iloc[-1])
                    basic_indicators[f'MA_{window}'] = ma_value
                except Exception as e:
                    print(f"Error calculating MA_{window}: {e}")
        
        # Basic RSI
        if len(df) >= 14:
            try:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean().replace(0, 1e-10)
                rs = avg_gain / avg_loss
                basic_indicators['RSI_14'] = float(100 - (100 / (1 + rs.iloc[-1])))
            except Exception as e:
                print(f"Error calculating RSI: {e}")
        
        # Convert date to strings
        dates = [date.strftime('%Y-%m-%d') for date in df['Date']]
        
        # Prepare price data with explicit type conversion and error handling
        price_data = {}
        try:
            # Process each series individually with proper error handling
            price_data['close'] = []
            for x in df['Close'].values:
                try:
                    price_data['close'].append(float(x))
                except:
                    price_data['close'].append(None)
                    
            price_data['open'] = []
            for x in df['Open'].values:
                try:
                    price_data['open'].append(float(x))
                except:
                    price_data['open'].append(None)
                    
            price_data['high'] = []
            for x in df['High'].values:
                try:
                    price_data['high'].append(float(x))
                except:
                    price_data['high'].append(None)
                    
            price_data['low'] = []
            for x in df['Low'].values:
                try:
                    price_data['low'].append(float(x))
                except:
                    price_data['low'].append(None)
        except Exception as e:
            print(f"Error converting price data: {e}")
            # Fallback using direct values
            price_data = {
                'close': df['Close'].values.tolist(),
                'open': df['Open'].values.tolist(),
                'high': df['High'].values.tolist(),
                'low': df['Low'].values.tolist(),
            }
        
        # Handle volume data with error handling
        volume_data = {}
        if 'Volume' in df.columns:
            try:
                volume_data['volume'] = []
                for x in df['Volume'].values:
                    try:
                        if pd.isna(x):
                            volume_data['volume'].append(0)
                        else:
                            volume_data['volume'].append(float(x))
                    except:
                        volume_data['volume'].append(0)
            except Exception as e:
                print(f"Error converting volume data: {e}")
                # Fallback with direct values
                volume_data['volume'] = df['Volume'].values.tolist()
        
        # Print debug information for troubleshooting
        print(f"Processing {symbol}: data shapes - price:{len(price_data['close'])}, dates:{len(dates)}")
        
        # Compile result
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "timeframe": timeframe,
            "current_price": current_price,
            "price_change_pct": price_change,
            "volatility": volatility,
            "basic_indicators": basic_indicators,
            "price_data": price_data,
            "volume_data": volume_data,
            "dates": dates,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            print(f"Warning: Could not cache results for {symbol}: {e}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error fetching technical data for {symbol}: {str(e)}"
        print(error_msg)
        return {"error": error_msg}