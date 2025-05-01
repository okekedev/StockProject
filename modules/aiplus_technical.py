"""
AI+ Technical Data Module

This module handles fetching, processing and analyzing technical stock data
for the AI+ enhanced prediction system. It extends the regular technical
analysis with additional indicators and features specifically designed
for machine learning applications.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
import config

# Constants
DATA_CACHE_DIR = os.path.join(config.DATA_DIR, "aiplus_cache")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


class AIplusTechnicalAnalyzer:
    """
    Class for fetching and analyzing technical stock data with enhanced
    indicators for AI-powered predictions.
    """
    
    def __init__(self):
        """Initialize the technical analyzer."""
        self.cache = {}
        self.indicators = {}
        self.last_fetch_time = {}
    
    def get_technical_data(self, symbol, timeframe='1mo', force_refresh=False):
        """
        Fetch technical data for a symbol with specified timeframe.
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Time period - '1mo', '3mo', '6mo', '1y'
            force_refresh (bool): Whether to force refresh data from source
            
        Returns:
            dict: Technical analysis results
        """
        cache_key = f"{symbol}_{timeframe}"
        cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_key}.json")
        
        # Check if we have fresh cached data
        if not force_refresh and cache_key in self.cache:
            # Use cached data if it exists and is recent (within 4 hours)
            if (datetime.now() - self.last_fetch_time.get(cache_key, datetime(1970, 1, 1))).total_seconds() < 14400:
                return self.cache[cache_key]
        
        # Try to load from disk cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is fresh (within 4 hours)
                if 'timestamp' in cache_data:
                    cache_time = datetime.fromisoformat(cache_data['timestamp'])
                    if (datetime.now() - cache_time).total_seconds() < 14400:
                        self.cache[cache_key] = cache_data
                        self.last_fetch_time[cache_key] = cache_time
                        return cache_data
            except Exception as e:
                print(f"Error loading cache for {symbol}: {e}")
        
        # Fetch fresh data if no valid cache exists
        try:
            # Download stock data
            stock = yf.Ticker(symbol)
            
            # Convert timeframe to pandas date offset
            if timeframe == '1mo':
                period = '1mo'
                days = 30
            elif timeframe == '3mo':
                period = '3mo'
                days = 90
            elif timeframe == '6mo':
                period = '6mo'
                days = 180
            elif timeframe == '1y':
                period = '1y'
                days = 365
            else:
                period = '1mo'
                days = 30
            
            # Get historical data with buffer for calculations
            buffer_days = 60  # Additional days for calculating indicators
            start_date = (datetime.now() - timedelta(days=days+buffer_days)).strftime('%Y-%m-%d')
            
            # Fetch historical data
            df = stock.history(start=start_date)
            
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Process the data
            result = self._process_technical_data(df, symbol, timeframe)
            
            # Cache the results
            self.cache[cache_key] = result
            self.last_fetch_time[cache_key] = datetime.now()
            
            # Save to disk cache
            result['timestamp'] = datetime.now().isoformat()
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching technical data for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    def _process_technical_data(self, df, symbol, timeframe):
        """
        Process raw price data to calculate technical indicators.
        
        Args:
            df (DataFrame): Raw price data
            symbol (str): Stock symbol
            timeframe (str): Time period
            
        Returns:
            dict: Technical analysis results
        """
        # Ensure index is datetime and sorted
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Basic price metrics
        current_price = df['Close'].iloc[-1]
        
        # Define start_date based on the timeframe
        if timeframe == '1mo':
            days_back = 30
        elif timeframe == '3mo':
            days_back = 90
        elif timeframe == '6mo':
            days_back = 180
        elif timeframe == '1y':
            days_back = 365
        else:
            days_back = 30
        
        # Calculate start_date from the current date
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Trim the buffer before calculating returns
        if timeframe == '1mo':
            actual_df = df.iloc[-30:]
        elif timeframe == '3mo':
            actual_df = df.iloc[-90:]
        elif timeframe == '6mo':
            actual_df = df.iloc[-180:]
        elif timeframe == '1y':
            actual_df = df.iloc[-365:]
        else:
            actual_df = df.iloc[-30:]
        
        # Calculate returns
        start_price = actual_df['Close'].iloc[0]
        price_change = (current_price - start_price) / start_price * 100
        
        # Calculate volatility
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
        
        # Calculate standard technical indicators
        indicators = {}
        
        # Moving Averages
        for window in [10, 20, 50, 200]:
            if len(df) >= window:
                indicators[f'MA_{window}'] = df['Close'].rolling(window=window).mean().iloc[-1]
        
        # Calculate RSI
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI_14'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Calculate MACD
        if len(df) >= 26:
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['MACD'] = macd_line.iloc[-1]
            indicators['MACD_Signal'] = signal_line.iloc[-1]
            indicators['MACD_Histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
        
        # Bollinger Bands
        if len(df) >= 20:
            rolling_mean = df['Close'].rolling(window=20).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            
            indicators['BB_Upper'] = rolling_mean.iloc[-1] + (rolling_std.iloc[-1] * 2)
            indicators['BB_Middle'] = rolling_mean.iloc[-1]
            indicators['BB_Lower'] = rolling_mean.iloc[-1] - (rolling_std.iloc[-1] * 2)
            
            # Calculate %B (relative position within the bands)
            indicators['BB_Percent'] = (df['Close'].iloc[-1] - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
        
        # Calculate trend strength indicators
        
        # Average Directional Index (ADX)
        if len(df) >= 14:
            try:
                # Calculate +DI and -DI
                high_diff = df['High'].diff()
                low_diff = df['Low'].diff().abs()
                
                plus_dm = high_diff.copy()
                plus_dm[plus_dm < 0] = 0
                plus_dm[(high_diff <= 0) | (high_diff <= low_diff)] = 0
                
                minus_dm = low_diff.copy()
                minus_dm[minus_dm < 0] = 0
                minus_dm[(low_diff <= 0) | (low_diff <= high_diff)] = 0
                
                tr = pd.DataFrame()
                tr['h-l'] = df['High'] - df['Low']
                tr['h-pc'] = (df['High'] - df['Close'].shift()).abs()
                tr['l-pc'] = (df['Low'] - df['Close'].shift()).abs()
                tr['tr'] = tr.max(axis=1)
                
                atr = tr['tr'].rolling(14).mean()
                
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
                
                dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                adx = dx.rolling(14).mean()
                
                indicators['ADX'] = adx.iloc[-1]
                indicators['+DI'] = plus_di.iloc[-1]
                indicators['-DI'] = minus_di.iloc[-1]
            except Exception as e:
                print(f"Error calculating ADX: {e}")
        
        # AI+ Enhanced Indicators
        
        # Price Momentum Strength
        price_momentum = {}
        for period in [5, 10, 20, 60]:
            if len(df) >= period:
                momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-period] - 1) * 100
                price_momentum[f'momentum_{period}d'] = momentum
        
        # Volatility trend
        volatility_trend = {}
        for period in [10, 20]:
            if len(df) >= period * 2:
                recent_vol = df['Close'].iloc[-period:].pct_change().std() * (252 ** 0.5) * 100
                previous_vol = df['Close'].iloc[-period*2:-period].pct_change().std() * (252 ** 0.5) * 100
                volatility_trend[f'vol_change_{period}d'] = recent_vol - previous_vol
        
        # Volume analysis
        volume_analysis = {}
        if 'Volume' in df.columns:
            # Calculate OBV (On-Balance Volume)
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            volume_analysis['OBV'] = obv.iloc[-1]
            
            # Volume trend
            avg_volume_10d = df['Volume'].iloc[-10:].mean()
            avg_volume_30d = df['Volume'].iloc[-30:].mean()
            volume_analysis['volume_ratio_10d_30d'] = avg_volume_10d / avg_volume_30d if avg_volume_30d > 0 else 1.0
            
            # Price-volume relationship
            price_volume_corr = df['Close'].iloc[-20:].pct_change().corr(df['Volume'].iloc[-20:].pct_change())
            volume_analysis['price_volume_corr_20d'] = price_volume_corr
        
        # Market context
        market_context = {}
        try:
            # Get S&P 500 data
            sp500 = yf.Ticker('^GSPC').history(start=start_date)
            
            if not sp500.empty:
                # Calculate beta
                if len(df) > 20 and len(sp500) > 20:
                    stock_returns = df['Close'].pct_change().dropna()
                    market_returns = sp500['Close'].pct_change().dropna()
                    
                    # Align dates
                    aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
                    if len(aligned_data) > 1:  # Need at least 2 points for regression
                        beta, alpha, r_value, p_value, std_err = stats.linregress(
                            aligned_data.iloc[:, 1],  # Market returns
                            aligned_data.iloc[:, 0]   # Stock returns
                        )
                        market_context['beta'] = beta
                        market_context['r_squared'] = r_value ** 2
                
                # Relative strength vs S&P 500
                stock_performance = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
                market_performance = (sp500['Close'].iloc[-1] / sp500['Close'].iloc[0]) - 1
                market_context['relative_strength'] = stock_performance - market_performance
        except Exception as e:
            print(f"Error calculating market context: {e}")
        
        # AI+ Prediction Metrics
        
        # Calculate the TSMN (Temporal-Spectral Momentum Nexus) indicator
        # This is your custom indicator combining multiple signal types
        tsmn = self._calculate_tsmn(df)
        
        # Compile results
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "price_change_pct": price_change,
            "volatility": volatility,
            "standard_indicators": indicators,
            "price_momentum": price_momentum,
            "volatility_trend": volatility_trend,
            "volume_analysis": volume_analysis,
            "market_context": market_context,
            "tsmn": tsmn,
            "pattern_detection": self._detect_patterns(df),
            "technical_summary": self._generate_technical_summary(
                current_price, price_change, volatility, indicators, 
                price_momentum, tsmn
            )
        }
        
        return result
    
    def _calculate_tsmn(self, df):
        """
        Calculate the Temporal-Spectral Momentum Nexus indicator.
        
        Args:
            df (DataFrame): Price and volume data
            
        Returns:
            dict: TSMN data
        """
        try:
            # Ensure minimum data length
            if len(df) < 60:
                return {"error": "Insufficient data for TSMN calculation"}
            
            # Get price data
            close = df['Close']
            
            # Calculate RSI values
            rsi_5 = self._calculate_rsi(close, 5)
            rsi_14 = self._calculate_rsi(close, 14)
            rsi_20 = self._calculate_rsi(close, 20)
            
            # Calculate momentum gradient
            momentum_gradient = rsi_5.iloc[-1] - rsi_20.iloc[-1]
            
            # Calculate temporal gradient (rate of change of momentum)
            rsi_5_change = rsi_5.diff(periods=5).iloc[-1]
            
            # Incorporate volume information if available
            volume_factor = 1.0
            if 'Volume' in df.columns:
                # Calculate volume ratio compared to moving average
                volume = df['Volume']
                volume_ma = volume.rolling(window=20).mean()
                volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
                
                # Bound volume factor between 0.5 and 1.5
                volume_factor = min(max(volume_ratio, 0.5), 1.5)
            
            # Spectral component - use price velocities at different frequencies
            diff_1d = close.diff(periods=1).fillna(0)
            diff_5d = close.diff(periods=5).fillna(0)
            diff_20d = close.diff(periods=20).fillna(0)
            
            # Normalize by recent volatility
            vol_20d = close.pct_change().rolling(window=20).std().iloc[-1]
            if vol_20d > 0:
                spec_1d = diff_1d.iloc[-1] / (close.iloc[-1] * vol_20d)
                spec_5d = diff_5d.iloc[-1] / (close.iloc[-1] * vol_20d * (5**0.5))
                spec_20d = diff_20d.iloc[-1] / (close.iloc[-1] * vol_20d * (20**0.5))
            else:
                spec_1d = spec_5d = spec_20d = 0
            
            # Add spectral components with higher weight for shorter term
            spectral_factor = (0.5 * spec_1d + 0.3 * spec_5d + 0.2 * spec_20d)
            
            # Calculate final TSMN
            tsmn_raw = momentum_gradient * (1 + 0.2 * rsi_5_change) * volume_factor * (1 + spectral_factor)
            
            # Normalize to -100 to +100 scale (approximately)
            tsmn_normalized = max(min(tsmn_raw * 5, 100), -100)
            
            # TSMN signal interpretation
            signal = "neutral"
            strength = "weak"
            
            if tsmn_normalized > 60:
                signal = "strong_buy"
                strength = "strong"
            elif tsmn_normalized > 20:
                signal = "buy"
                strength = "moderate"
            elif tsmn_normalized < -60:
                signal = "strong_sell"
                strength = "strong"
            elif tsmn_normalized < -20:
                signal = "sell"
                strength = "moderate"
            
            return {
                "value": tsmn_normalized,
                "signal": signal,
                "strength": strength,
                "components": {
                    "momentum_gradient": momentum_gradient,
                    "rsi_change": rsi_5_change,
                    "volume_factor": volume_factor,
                    "spectral_factor": spectral_factor
                }
            }
            
        except Exception as e:
            print(f"Error calculating TSMN: {e}")
            return {"error": str(e)}
    
    def _calculate_rsi(self, series, window):
        """Calculate RSI for a price series."""
        delta = series.diff().dropna()
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=window).mean()
        avg_loss = down.abs().rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    def _detect_patterns(self, df):
        """
        Detect technical chart patterns in the price data.
        
        Args:
            df (DataFrame): Price data
            
        Returns:
            dict: Detected patterns and confidence levels
        """
        patterns = {}
        
        # Minimum data required
        if len(df) < 30:
            return {"error": "Insufficient data for pattern detection"}
        
        # Get OHLC data for recent period
        recent_df = df.iloc[-30:]
        
        try:
            # Head and Shoulders pattern
            # Simplified detection: look for 3 peaks with middle one higher
            
            # Double Top pattern
            rolling_max = recent_df['High'].rolling(5).max()
            potential_tops = (recent_df['High'] == rolling_max) & (recent_df['High'] > recent_df['High'].shift(1)) & (recent_df['High'] > recent_df['High'].shift(-1))
            top_indexes = potential_tops[potential_tops].index.tolist()
            
            if len(top_indexes) >= 2:
                # Check if the tops are separated and at similar price levels
                date_diffs = [(top_indexes[i] - top_indexes[i-1]).days for i in range(1, len(top_indexes))]
                price_diffs = [abs(recent_df.loc[top_indexes[i], 'High'] - recent_df.loc[top_indexes[i-1], 'High']) / recent_df.loc[top_indexes[i-1], 'High'] for i in range(1, len(top_indexes))]
                
                if any(diff >= 5 and diff <= 20 for diff in date_diffs) and any(diff <= 0.03 for diff in price_diffs):
                    patterns['double_top'] = {
                        'detected': True,
                        'confidence': 'medium',
                        'implication': 'bearish'
                    }
            
            # Bullish Engulfing pattern
            bullish_engulfing = (
                (recent_df['Close'] > recent_df['Open']) &  # Current candle is bullish
                (recent_df['Close'].shift(1) < recent_df['Open'].shift(1)) &  # Previous candle is bearish
                (recent_df['Close'] > recent_df['Open'].shift(1)) &  # Current close > previous open
                (recent_df['Open'] < recent_df['Close'].shift(1))  # Current open < previous close
            )
            
            if bullish_engulfing.any():
                patterns['bullish_engulfing'] = {
                    'detected': True,
                    'confidence': 'high' if bullish_engulfing.iloc[-5:].any() else 'medium',
                    'implication': 'bullish'
                }
            
            # Bearish Engulfing pattern
            bearish_engulfing = (
                (recent_df['Close'] < recent_df['Open']) &  # Current candle is bearish
                (recent_df['Close'].shift(1) > recent_df['Open'].shift(1)) &  # Previous candle is bullish
                (recent_df['Close'] < recent_df['Open'].shift(1)) &  # Current close < previous open
                (recent_df['Open'] > recent_df['Close'].shift(1))  # Current open > previous close
            )
            
            if bearish_engulfing.any():
                patterns['bearish_engulfing'] = {
                    'detected': True,
                    'confidence': 'high' if bearish_engulfing.iloc[-5:].any() else 'medium',
                    'implication': 'bearish'
                }
            
            # Detect potential support/resistance levels
            sup_res_levels = self._detect_support_resistance(recent_df)
            if sup_res_levels:
                patterns['support_resistance'] = sup_res_levels
            
        except Exception as e:
            print(f"Error in pattern detection: {e}")
            patterns['error'] = str(e)
        
        return patterns
    
    def _detect_support_resistance(self, df):
        """
        Detect support and resistance levels.
        
        Args:
            df (DataFrame): Price data
            
        Returns:
            dict: Support and resistance levels
        """
        # Use pivot points for support/resistance detection
        levels = {}
        
        # Calculate typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Previous period high, low, close
        prev_high = df['High'].iloc[-2]
        prev_low = df['Low'].iloc[-2]
        prev_close = df['Close'].iloc[-2]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        
        # Current price
        current_price = df['Close'].iloc[-1]
        
        # Determine closest levels
        price_levels = [s2, s1, pivot, r1, r2]
        level_names = ["S2", "S1", "Pivot", "R1", "R2"]
        
        # Calculate distance to each level
        distances = [abs(level - current_price) / current_price * 100 for level in price_levels]
        
        # Find nearest level
        nearest_idx = distances.index(min(distances))
        nearest_level = level_names[nearest_idx]
        nearest_value = price_levels[nearest_idx]
        
        # Determine if price is near support or resistance
        is_support = nearest_idx < 2  # S2 or S1
        is_resistance = nearest_idx > 2  # R1 or R2
        
        # Return levels
        return {
            'levels': {
                'S2': s2,
                'S1': s1,
                'Pivot': pivot,
                'R1': r1,
                'R2': r2
            },
            'nearest': {
                'level': nearest_level,
                'value': nearest_value,
                'distance_pct': distances[nearest_idx]
            },
            'interpretation': {
                'is_near_support': is_support,
                'is_near_resistance': is_resistance,
                'at_level': distances[nearest_idx] < 0.5  # Within 0.5% of a level
            }
        }
    
    def _generate_technical_summary(self, price, price_change, volatility, 
                                   indicators, momentum, tsmn):
        """
        Generate a human-readable summary of technical analysis.
        
        Args:
            Various technical indicators and metrics
            
        Returns:
            str: Technical analysis summary
        """
        summary_parts = []
        
        # Price trend summary
        if price_change > 5:
            price_trend = f"strongly bullish with a {price_change:.1f}% increase"
        elif price_change > 1:
            price_trend = f"moderately bullish with a {price_change:.1f}% increase"
        elif price_change > -1:
            price_trend = f"neutral with a small {abs(price_change):.1f}% change"
        elif price_change > -5:
            price_trend = f"moderately bearish with a {abs(price_change):.1f}% decrease"
        else:
            price_trend = f"strongly bearish with a {abs(price_change):.1f}% decrease"
        
        summary_parts.append(f"Price action is {price_trend} in this timeframe.")
        
        # RSI interpretation
        if 'RSI_14' in indicators:
            rsi = indicators['RSI_14']
            if rsi > 70:
                summary_parts.append(f"RSI at {rsi:.1f} indicates overbought conditions.")
            elif rsi < 30:
                summary_parts.append(f"RSI at {rsi:.1f} indicates oversold conditions.")
            else:
                summary_parts.append(f"RSI at {rsi:.1f} is in neutral territory.")
        
        # MACD interpretation
        if 'MACD' in indicators and 'MACD_Signal' in indicators:
            macd = indicators['MACD']
            signal = indicators['MACD_Signal']
            
            if macd > signal:
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
        if 'BB_Percent' in indicators:
            bb_pct = indicators['BB_Percent']
            
            if bb_pct > 1:
                summary_parts.append("Price is above the upper Bollinger Band, indicating strong upward momentum or potential reversal.")
            elif bb_pct > 0.8:
                summary_parts.append("Price is near the upper Bollinger Band, suggesting strong momentum.")
            elif bb_pct < 0:
                summary_parts.append("Price is below the lower Bollinger Band, indicating strong downward momentum or potential reversal.")
            elif bb_pct < 0.2:
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
        if isinstance(tsmn, dict) and 'value' in tsmn:
            tsmn_value = tsmn['value']
            signal = tsmn.get('signal', 'neutral')
            
            if signal == 'strong_buy':
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} shows strong bullish momentum.")
            elif signal == 'buy':
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} suggests moderate bullish momentum.")
            elif signal == 'strong_sell':
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} shows strong bearish pressure.")
            elif signal == 'sell':
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} suggests moderate bearish pressure.")
            else:
                summary_parts.append(f"TSMN indicator at {tsmn_value:.1f} indicates neutral momentum.")
        
        # Combine all parts
        return " ".join(summary_parts)


# Function to get technical data for a specific stock
def get_aiplus_technical_data(symbol, timeframe='1mo', force_refresh=False):
    """
    Get comprehensive technical analysis for a stock.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Time period - '1mo', '3mo', '6mo', '1y'
        force_refresh (bool): Whether to force refresh data
        
    Returns:
        dict: Technical analysis results
    """
    analyzer = AIplusTechnicalAnalyzer()
    return analyzer.get_technical_data(symbol, timeframe, force_refresh)


# Test function
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    timeframe = "1mo"
    
    result = get_aiplus_technical_data(symbol, timeframe, force_refresh=True)
    print(json.dumps(result, indent=2))