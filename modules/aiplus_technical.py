"""
Comprehensive Technical Analysis Module with Dynamic Indicator Calculation

This module handles fetching stock data and calculating all technical indicators
dynamically, ensuring indicators are always available for analysis.
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

class TechnicalIndicators:
    """
    Class for calculating common technical indicators from price data.
    These calculations are used as fallbacks when pre-calculated indicators
    aren't available from the data source.
    """
    
    @staticmethod
    def ensure_dataframe(df):
        """
        Ensures input is a valid DataFrame with required columns.
        
        Args:
            df: Input data that could be a DataFrame or dict
            
        Returns:
            pd.DataFrame: Properly formatted DataFrame
        """
        if isinstance(df, dict):
            # Convert price dictionary to DataFrame
            if 'close' in df and 'dates' in df:
                price_df = pd.DataFrame({
                    'Close': df['close'],
                    'Date': df['dates']
                })
                if 'open' in df:
                    price_df['Open'] = df['open']
                if 'high' in df:
                    price_df['High'] = df['high']
                if 'low' in df:
                    price_df['Low'] = df['low']
                if 'volume' in df:
                    price_df['Volume'] = df['volume']
                
                price_df['Date'] = pd.to_datetime(price_df['Date'])
                price_df.set_index('Date', inplace=True)
                return price_df
            return pd.DataFrame()  # Empty DataFrame if invalid dict
        
        # If already a DataFrame, ensure it has required columns
        if isinstance(df, pd.DataFrame):
            if 'Close' not in df.columns:
                raise ValueError("DataFrame must contain 'Close' prices")
            return df
        
        return pd.DataFrame()  # Empty DataFrame for any other type
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            df (pd.DataFrame): Historical price data with 'Close' column
            period (int): RSI period (typically 14)
            
        Returns:
            float: The most recent RSI value
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Need at least period+1 data points to calculate RSI
            if len(df) < period + 1:
                return 50  # Return neutral RSI if not enough data
                
            # Calculate price changes
            delta = df['Close'].diff().dropna()
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = -losses  # Make losses positive for calculations
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Return the most recent RSI value, or 50 (neutral) if calculation failed
            final_rsi = rsi.iloc[-1]
            if pd.isna(final_rsi):
                return 50
                
            return float(final_rsi)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return 50
    
    @staticmethod
    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        
        Args:
            df (pd.DataFrame): Historical price data with 'Close' column
            fast_period (int): Fast EMA period (typically 12)
            slow_period (int): Slow EMA period (typically 26)
            signal_period (int): Signal line period (typically 9)
            
        Returns:
            tuple: (MACD line value, Signal line value, Histogram value)
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Need enough data for the slow period calculation
            if len(df) < slow_period + signal_period:
                return 0, 0, 0  # Return neutral values if not enough data
                
            # Calculate EMAs
            ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line, signal line, and histogram
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            # Get latest values
            latest_macd = float(macd_line.iloc[-1])
            latest_signal = float(signal_line.iloc[-1])
            latest_histogram = float(histogram.iloc[-1])
            
            # Check for NaN values
            if pd.isna(latest_macd) or pd.isna(latest_signal) or pd.isna(latest_histogram):
                return 0, 0, 0
                
            return latest_macd, latest_signal, latest_histogram
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return 0, 0, 0
    
    @staticmethod
    def calculate_bollinger_bands(df, period=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            df (pd.DataFrame): Historical price data with 'Close' column
            period (int): Moving average period (typically 20)
            num_std (int): Number of standard deviations (typically 2)
            
        Returns:
            tuple: (Upper Band, Middle Band, Lower Band, %B)
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Need at least 'period' data points
            if len(df) < period:
                latest_price = df['Close'].iloc[-1]
                return float(latest_price * 1.05), float(latest_price), float(latest_price * 0.95), 0.5
                
            # Calculate middle band (SMA)
            middle_band = df['Close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std_dev = df['Close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)
            
            # Calculate %B (position within the bands)
            latest_price = df['Close'].iloc[-1]
            latest_upper = upper_band.iloc[-1]
            latest_middle = middle_band.iloc[-1]
            latest_lower = lower_band.iloc[-1]
            
            # Handle potential division by zero
            band_width = latest_upper - latest_lower
            if band_width <= 0:
                percent_b = 0.5  # Default to middle
            else:
                percent_b = (latest_price - latest_lower) / band_width
                
            # Check for NaN values and provide defaults if needed
            if pd.isna(latest_upper) or pd.isna(latest_middle) or pd.isna(latest_lower) or pd.isna(percent_b):
                latest_price = df['Close'].iloc[-1]
                return float(latest_price * 1.05), float(latest_price), float(latest_price * 0.95), 0.5
                
            return float(latest_upper), float(latest_middle), float(latest_lower), float(percent_b)
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            latest_price = df['Close'].iloc[-1] if len(df) > 0 else 0
            return float(latest_price * 1.05), float(latest_price), float(latest_price * 0.95), 0.5
    
    @staticmethod
    def calculate_adx(df, period=14):
        """
        Calculate the Average Directional Index (ADX).
        
        Args:
            df (pd.DataFrame): Historical price data with High, Low, Close columns
            period (int): ADX period (typically 14)
            
        Returns:
            tuple: (ADX, +DI, -DI)
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Check for required columns
            required_columns = ['High', 'Low', 'Close']
            for column in required_columns:
                if column not in df.columns:
                    return 20, 0, 0  # Return default values if missing columns
                    
            # Need enough data points
            if len(df) < period + 1:
                return 20, 0, 0  # Return default values if not enough data
                
            # True Range
            df['H-L'] = df['High'] - df['Low']
            df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
            df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=period).mean()
            
            # Plus Directional Movement (+DM)
            df['H-PH'] = df['High'] - df['High'].shift(1)
            df['PL-L'] = df['Low'].shift(1) - df['Low']
            df['+DM'] = np.where((df['H-PH'] > df['PL-L']) & (df['H-PH'] > 0), df['H-PH'], 0)
            df['-DM'] = np.where((df['PL-L'] > df['H-PH']) & (df['PL-L'] > 0), df['PL-L'], 0)
            
            # Smoothed +DM and -DM
            df['+DM'] = df['+DM'].rolling(window=period).mean()
            df['-DM'] = df['-DM'].rolling(window=period).mean()
            
            # Directional Indicators
            df['+DI'] = 100 * (df['+DM'] / df['ATR'].replace(0, 1e-10))
            df['-DI'] = 100 * (df['-DM'] / df['ATR'].replace(0, 1e-10))
            
            # Directional Index
            df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']).replace(0, 1e-10)
            
            # Average Directional Index
            df['ADX'] = df['DX'].rolling(window=period).mean()
            
            # Get latest values
            latest_adx = df['ADX'].iloc[-1]
            latest_plus_di = df['+DI'].iloc[-1]
            latest_minus_di = df['-DI'].iloc[-1]
            
            # Handle NaN values
            if pd.isna(latest_adx) or pd.isna(latest_plus_di) or pd.isna(latest_minus_di):
                return 20, 0, 0
                
            return float(latest_adx), float(latest_plus_di), float(latest_minus_di)
                
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return 20, 0, 0
    
    @staticmethod
    def calculate_moving_averages(df, periods=[20, 50, 200]):
        """
        Calculate Simple Moving Averages for multiple periods.
        
        Args:
            df (pd.DataFrame): Historical price data with 'Close' column
            periods (list): List of periods to calculate
            
        Returns:
            dict: Dictionary of moving averages by period
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            result = {}
            
            for period in periods:
                # Skip if not enough data
                if len(df) < period:
                    result[f'MA_{period}'] = float(df['Close'].iloc[-1])  # Use current price as fallback
                    continue
                    
                # Calculate SMA
                ma = df['Close'].rolling(window=period).mean().iloc[-1]
                
                # Handle NaN
                if pd.isna(ma):
                    ma = float(df['Close'].iloc[-1])  # Use current price as fallback
                    
                result[f'MA_{period}'] = float(ma)
                
            return result
        except Exception as e:
            print(f"Error calculating Moving Averages: {e}")
            result = {}
            for period in periods:
                result[f'MA_{period}'] = 0
            return result
    
    @staticmethod
    def calculate_volatility(df, period=20, annualized=True):
        """
        Calculate price volatility.
        
        Args:
            df (pd.DataFrame): Historical price data with 'Close' column
            period (int): Period for volatility calculation
            annualized (bool): Whether to annualize the volatility
            
        Returns:
            float: Volatility as a percentage
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Need at least 'period' data points
            if len(df) < period:
                return 20  # Return default moderate volatility if not enough data
                
            # Calculate daily returns
            returns = df['Close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.rolling(window=period).std().iloc[-1]
            
            # Handle NaN
            if pd.isna(volatility):
                return 20
                
            # Annualize if requested (approximately 252 trading days in a year)
            if annualized:
                volatility = volatility * np.sqrt(252)
                
            # Convert to percentage
            volatility = volatility * 100
            
            return float(volatility)
        except Exception as e:
            print(f"Error calculating Volatility: {e}")
            return 20
    
    @staticmethod
    def calculate_tsmn(df):
        """
        Calculate the Temporal-Spectral Momentum Nexus (TSMN) indicator.
        This is a custom momentum indicator combining various technical factors.
        
        Args:
            df (pd.DataFrame): Historical price data
            
        Returns:
            dict: TSMN indicator values and signal
        """
        try:
            # Ensure we have a DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Need at least 30 data points for meaningful calculation
            if len(df) < 30:
                return {
                    "value": 0,
                    "signal": "neutral",
                    "strength": "weak"
                }
            
            # Calculate RSI values for different periods
            rsi_5 = TechnicalIndicators.calculate_rsi(df, period=5)
            rsi_14 = TechnicalIndicators.calculate_rsi(df, period=14) 
            rsi_20 = TechnicalIndicators.calculate_rsi(df, period=20)
            
            # Calculate price momentum
            close_series = df['Close']
            price_momentum = {}
            for period in [5, 10, 20]:
                if len(df) >= period:
                    momentum = (close_series.iloc[-1] / close_series.iloc[-period] - 1) * 100
                    price_momentum[f'{period}d'] = float(momentum)
            
            # Calculate volatility factor
            volatility = TechnicalIndicators.calculate_volatility(df)
            vol_factor = min(1.5, max(0.5, 20 / volatility))  # Higher volatility = lower factor
            
            # Calculate volume factor if volume data exists
            volume_factor = 1.0
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].iloc[-5:].mean()
                avg_volume = df['Volume'].iloc[-20:].mean()
                if avg_volume > 0:
                    volume_factor = min(1.5, max(0.5, recent_volume / avg_volume))
            
            # Calculate momentum gradient (difference between fast and slow RSI)
            momentum_gradient = rsi_5 - rsi_20
            
            # Calculate temporal changes in momentum
            temporal_factor = momentum_gradient / 10  # Scale factor
            
            # Combine factors for TSMN calculation
            tsmn_raw = momentum_gradient * volume_factor * vol_factor * (1 + 0.2 * temporal_factor)
            
            # Scale to -100 to +100 range
            tsmn_value = max(min(tsmn_raw * 5, 100), -100)
            
            # Determine signal and strength
            if tsmn_value > 60:
                signal = "bullish"
                strength = "strong"
            elif tsmn_value > 20:
                signal = "bullish"
                strength = "moderate"
            elif tsmn_value < -60:
                signal = "bearish"
                strength = "strong"
            elif tsmn_value < -20:
                signal = "bearish"
                strength = "moderate"
            else:
                signal = "neutral"
                strength = "weak"
            
            return {
                "value": float(tsmn_value),
                "signal": signal,
                "strength": strength,
                "components": {
                    "momentum_gradient": float(momentum_gradient),
                    "volume_factor": float(volume_factor),
                    "volatility_factor": float(vol_factor),
                    "rsi_5": float(rsi_5),
                    "rsi_20": float(rsi_20)
                }
            }
        except Exception as e:
            print(f"Error calculating TSMN: {e}")
            return {
                "value": 0,
                "signal": "neutral",
                "strength": "weak"
            }
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all essential technical indicators from price data.
        
        Args:
            df: DataFrame or dict with price data
            
        Returns:
            dict: All calculated technical indicators
        """
        try:
            # Ensure we have a proper DataFrame
            df = TechnicalIndicators.ensure_dataframe(df)
            
            # Validate data
            if df.empty or len(df) < 20:
                return {
                    "indicators_calculated": False,
                    "reason": f"Insufficient price data (need at least 20 points)"
                }
                
            # RSI
            rsi_14 = TechnicalIndicators.calculate_rsi(df)
            
            # MACD
            macd, macd_signal, macd_hist = TechnicalIndicators.calculate_macd(df)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_percent = TechnicalIndicators.calculate_bollinger_bands(df)
            
            # ADX
            adx, plus_di, minus_di = TechnicalIndicators.calculate_adx(df)
            
            # Moving Averages
            moving_averages = TechnicalIndicators.calculate_moving_averages(df)
            
            # Volatility
            volatility = TechnicalIndicators.calculate_volatility(df)
            
            # TSMN
            tsmn = TechnicalIndicators.calculate_tsmn(df)
            
            # Compile all indicators
            indicators = {
                "indicators_calculated": True,
                "standard_indicators": {
                    "RSI_14": rsi_14,
                    "MACD": macd,
                    "MACD_Signal": macd_signal,
                    "MACD_Histogram": macd_hist,
                    "BB_Upper": bb_upper,
                    "BB_Middle": bb_middle,
                    "BB_Lower": bb_lower,
                    "BB_Percent": bb_percent,
                    "ADX": adx,
                    "+DI": plus_di,
                    "-DI": minus_di
                },
                "moving_averages": moving_averages,
                "volatility": volatility,
                "tsmn": tsmn
            }
            
            # Add moving averages to standard indicators for compatibility
            indicators["standard_indicators"].update(moving_averages)
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {
                "indicators_calculated": False,
                "reason": f"Error: {str(e)}"
            }


def get_aiplus_technical_data(symbol, timeframe='1mo', force_refresh=False):
    """
    Get technical data for AI analysis of a stock.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Time period - '1mo', '3mo', '6mo', '1y'
        force_refresh (bool): Whether to force refresh data
        
    Returns:
        dict: Technical data results with all indicators
    """
    # Create cache key and file path
    cache_key = f"{symbol}_{timeframe}"
    cache_file = os.path.join(DATA_CACHE_DIR, f"{symbol}_tech.json")
    
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
            market_cap = info.get('marketCap', 0)
        except Exception as e:
            print(f"Warning: Could not get company info for {symbol}: {e}")
            company_name = symbol
            sector = "Unknown"
            industry = "Unknown"
            market_cap = 0
        
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
        
        # Convert date to strings for JSON compatibility
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
                
        # Create combined price data for indicator calculation
        price_df = pd.DataFrame({
            'Date': dates,
            'Close': price_data['close'],
            'Open': price_data['open'],
            'High': price_data['high'],
            'Low': price_data['low']
        })
        if 'volume' in volume_data:
            price_df['Volume'] = volume_data['volume']
            
        # Set date as index
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        indicator_df = price_df.set_index('Date')
        
        # Calculate all technical indicators dynamically
        indicators = TechnicalIndicators.calculate_all_indicators(indicator_df)
        
        # Extract calculated indicators
        if indicators.get('indicators_calculated', False):
            standard_indicators = indicators.get('standard_indicators', {})
            tsmn = indicators.get('tsmn', {})
            volatility = indicators.get('volatility', 20)
            
            # Generate technical summary based on indicators
            technical_summary = generate_technical_summary(
                price_change, volatility, standard_indicators, tsmn
            )
        else:
            standard_indicators = {}
            tsmn = {"value": 0, "signal": "neutral", "strength": "weak"}
            volatility = 20
            technical_summary = "Insufficient data to generate technical summary."
        
        # Compile result
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "timeframe": timeframe,
            "current_price": current_price,
            "price_change_pct": price_change,
            "volatility": volatility,
            "standard_indicators": standard_indicators,
            "tsmn": tsmn,
            "price_data": price_data,
            "volume_data": volume_data,
            "dates": dates,
            "technical_summary": technical_summary,
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


def generate_technical_summary(price_change, volatility, indicators, tsmn):
    """
    Generate a human-readable summary of technical indicators.
    
    Args:
        price_change (float): Percentage price change
        volatility (float): Volatility percentage
        indicators (dict): Technical indicators
        tsmn (dict): TSMN indicator
        
    Returns:
        str: Technical summary
    """
    try:
        # Get key indicators
        rsi = indicators.get('RSI_14', None)
        macd = indicators.get('MACD', None)
        macd_signal = indicators.get('MACD_Signal', None)
        bb_percent = indicators.get('BB_Percent', None)
        tsmn_value = tsmn.get('value', 0)
        
        summary_parts = []
        
        # Price trend summary
        if price_change > 5:
            summary_parts.append(f"Price action is strongly bullish with a {price_change:.1f}% increase in this timeframe.")
        elif price_change > 1:
            summary_parts.append(f"Price action is moderately bullish with a {price_change:.1f}% increase in this timeframe.")
        elif price_change > -1:
            summary_parts.append(f"Price action is neutral with a small {abs(price_change):.1f}% change in this timeframe.")
        elif price_change > -5:
            summary_parts.append(f"Price action is moderately bearish with a {abs(price_change):.1f}% decrease in this timeframe.")
        else:
            summary_parts.append(f"Price action is strongly bearish with a {abs(price_change):.1f}% decrease in this timeframe.")
        
        # RSI interpretation
        if rsi is not None:
            if rsi > 70:
                summary_parts.append(f"RSI at {rsi:.1f} indicates overbought conditions.")
            elif rsi < 30:
                summary_parts.append(f"RSI at {rsi:.1f} indicates oversold conditions.")
            else:
                summary_parts.append(f"RSI at {rsi:.1f} is in neutral territory.")
        
        # MACD interpretation
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                if macd > 0:
                    summary_parts.append("MACD is positive and above signal line, suggesting bullish momentum.")
                else:
                    summary_parts.append("MACD is below zero but above signal line, suggesting potential trend reversal.")
            else:
                if macd < 0:
                    summary_parts.append("MACD is negative and below signal line, indicating bearish momentum.")
                else:
                    summary_parts.append("MACD is above zero but below signal line, suggesting potential weakening momentum.")
        
        # Bollinger Bands
        if bb_percent is not None:
            if bb_percent > 1:
                summary_parts.append("Price is above the upper Bollinger Band, indicating strong upward momentum or potential reversal.")
            elif bb_percent > 0.8:
                summary_parts.append("Price is near the upper Bollinger Band, suggesting strong momentum.")
            elif bb_percent < 0:
                summary_parts.append("Price is below the lower Bollinger Band, indicating strong downward momentum or potential reversal.")
            elif bb_percent < 0.2:
                summary_parts.append("Price is near the lower Bollinger Band, suggesting strong selling pressure.")
            else:
                summary_parts.append("Price is within the Bollinger Bands, indicating average volatility.")
        
        # Volatility assessment
        if volatility > 50:
            summary_parts.append(f"Volatility is extremely high at {volatility:.1f}%.")
        elif volatility > 30:
            summary_parts.append(f"Volatility is high at {volatility:.1f}%.")
        elif volatility > 15:
            summary_parts.append(f"Volatility is moderate at {volatility:.1f}%.")
        else:
            summary_parts.append(f"Volatility is low at {volatility:.1f}%.")
        
        # TSMN interpretation
        if tsmn_value != 0:
            if tsmn_value > 60:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} shows strong bullish momentum.")
            elif tsmn_value > 20:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} suggests moderate bullish momentum.")
            elif tsmn_value < -60:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} shows strong bearish pressure.")
            elif tsmn_value < -20:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} suggests moderate bearish pressure.")
            else:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} indicates neutral momentum.")
        
        # Combine all parts
        return " ".join(summary_parts)
        
    except Exception as e:
        print(f"Error generating technical summary: {e}")
        return "Unable to generate technical summary due to an error."


# Test code
if __name__ == "__main__":
    import sys
    
    # Get symbol from command line or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Testing technical analysis for {symbol}...")
    
    # Force refresh to get latest data
    result = get_aiplus_technical_data(symbol, force_refresh=True)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Symbol: {result['symbol']}")
        print(f"Company: {result['company_name']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Price Change: {result['price_change_pct']:.2f}%")
        print(f"Volatility: {result['volatility']:.2f}%")
        print("\nTSMN Indicator:")
        print(f"Value: {result['tsmn']['value']:.2f}")
        print(f"Signal: {result['tsmn']['signal']}")
        print(f"Strength: {result['tsmn']['strength']}")
        print("\nKey Technical Indicators:")
        for key, value in result['standard_indicators'].items():
            print(f"{key}: {value}")
        print("\nTechnical Summary:")
        print(result['technical_summary'])