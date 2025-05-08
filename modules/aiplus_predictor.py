"""
Enhanced AI+ Predictor Module that leverages Gemini's mathematical expertise with raw data

This module sends raw stock price and news data to Gemini AI, letting it perform its own
analysis without being constrained by predetermined indicators or formulas.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import config

# Import our modules
from modules.aiplus_technical import get_aiplus_technical_data
from modules.aiplus_sentiment import get_aiplus_sentiment

# Load environment variables
load_dotenv()

# Constants
PREDICTIONS_DIR = os.path.join(config.DATA_DIR, "aiplus_predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Prediction horizon mapping for display
PREDICTION_HORIZON_TEXT = {
    '1d': 'next day',
    '2d': 'next two days',
    '1w': 'next week',
    '1mo': 'next month'
}

# Configure Gemini API
def configure_gemini():
    """Configure the Gemini API with the API key."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Try from config if not in environment
        api_key = config.GEMINI_API_KEY
        
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        print("Warning: No Gemini API key found")
        return False


class AIplusPredictor:
    """
    Class for sending raw technical and sentiment data to advanced AI for prediction.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.api_initialized = configure_gemini()
        self.predictions = {}
        self.performance_history = {}
    
    def generate_prediction(self, symbol, technical_timeframe='1mo', news_days=7, prediction_horizon='1d', force_refresh=False):
        """
        Generate an AI-enhanced stock prediction using raw data.
        
        Args:
            symbol (str): Stock symbol
            technical_timeframe (str): Timeframe for technical data
            news_days (int): Days for news analysis
            prediction_horizon (str): Timeframe for prediction (1d, 2d, 1w, 1mo)
            force_refresh (bool): Whether to force refresh data
            
        Returns:
            dict: Prediction results
        """
        # Create prediction key for caching
        pred_key = f"{symbol}_{technical_timeframe}_{news_days}_{prediction_horizon}"
        cache_file = os.path.join(PREDICTIONS_DIR, f"{pred_key}.json")
        
        # Check for existing predictions
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_pred = json.load(f)
                
                # Check if prediction is fresh (within 12 hours)
                if 'timestamp' in cached_pred:
                    pred_time = datetime.fromisoformat(cached_pred['timestamp'])
                    if (datetime.now() - pred_time).total_seconds() < 43200:  # 12 hours
                        return cached_pred
            except Exception as e:
                print(f"Error loading cached prediction: {e}")
        
        try:
            # Step 1: Get technical analysis data
            technical_data = get_aiplus_technical_data(symbol, technical_timeframe, force_refresh)
            
            # Step 2: Get news sentiment data
            sentiment_data = get_aiplus_sentiment(symbol, news_days, force_refresh)
            
            # If any data fetch failed, return the error
            if 'error' in technical_data:
                return {'error': f"Technical data error: {technical_data['error']}"}
                
            if 'error' in sentiment_data:
                return {'error': f"Sentiment data error: {sentiment_data['error']}"}
            
            # Step 3: Process the data with advanced AI analysis
            prediction_results = self._generate_ai_enhanced_prediction(
                symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon
            )
            
            # Store the prediction
            self.predictions[pred_key] = prediction_results
            
            # Save to disk
            with open(cache_file, 'w') as f:
                json.dump(prediction_results, f)
            
            return prediction_results
            
        except Exception as e:
            error_msg = f"Error generating prediction for {symbol}: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    def _generate_ai_enhanced_prediction(self, symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon):
        """
        Use advanced AI to analyze raw data and predict stock movements.
        
        Args:
            symbol (str): Stock symbol
            technical_data (dict): Technical analysis data
            sentiment_data (dict): Sentiment analysis data
            technical_timeframe (str): Timeframe for technical data
            prediction_horizon (str): Timeframe for prediction
            
        Returns:
            dict: Enhanced AI prediction
        """
        if not self.api_initialized:
            # If AI API not available, fall back to simple weighted combination
            return self._generate_simple_prediction(symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon)
        
        try:
            # Extract key data for the prompt
            current_price = technical_data.get('current_price', 0)
            company_name = technical_data.get('company_name', symbol)
            sector = technical_data.get('sector', 'Unknown')
            industry = technical_data.get('industry', 'Unknown')
            
            # Format timeframes for AI prompt
            ai_timeframe = {
                '1mo': 'one month',
                '3mo': 'three months',
                '6mo': 'six months',
                '1y': 'one year'
            }.get(technical_timeframe, technical_timeframe)
            
            prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
            
            # Get raw price data
            price_data = technical_data.get('price_data', {})
            volume_data = technical_data.get('volume_data', {})
            dates = technical_data.get('dates', [])
            
            # Extract just a few basic indicators for context
            basic_indicators = technical_data.get('basic_indicators', {})
            
            # Prepare raw data in a format for the AI to analyze
            # For timeframes up to 3 months, include complete dataset
            # For longer timeframes, sample representative points
            price_points = []
            
            # Determine if we need to sample (only for long timeframes to manage token limits)
            use_sampling = len(dates) > 60  # Only sample if more than 60 days of data
            
            # Get indices - either all indices or a representative sample for long timeframes
            indices = self._get_representative_indices(len(dates)) if use_sampling else range(len(dates))
            
            for idx in indices:
                if idx < len(dates):
                    date_point = dates[idx]
                    price_point = {
                        'date': date_point,
                        'close': price_data.get('close', [])[idx] if idx < len(price_data.get('close', [])) else None,
                        'open': price_data.get('open', [])[idx] if idx < len(price_data.get('open', [])) else None,
                        'high': price_data.get('high', [])[idx] if idx < len(price_data.get('high', [])) else None,
                        'low': price_data.get('low', [])[idx] if idx < len(price_data.get('low', [])) else None,
                        'volume': volume_data.get('volume', [])[idx] if idx < len(volume_data.get('volume', [])) else None
                    }
                    price_points.append(price_point)
            
            # Get news data
            news_items = sentiment_data.get('news_items', [])
            
            # Create prompt for Gemini with raw data
            prompt = f"""
            You are an expert financial analyst with deep expertise in quantitative finance, technical analysis, and market psychology.

            Please analyze the following data for {company_name} ({symbol}) and provide a detailed price movement prediction for the {prediction_horizon_text}.

            COMPANY INFORMATION:
            - Current Price: ${current_price:.2f}
            - Sector: {sector}
            - Industry: {industry}
            
            RAW PRICE DATA (SAMPLE POINTS ACROSS {ai_timeframe}):
            """
            
            # Add price data points
            sampling_notice = " (SAMPLED POINTS)" if use_sampling else " (COMPLETE DATASET)"
            prompt += sampling_notice + "\n"
            
            for point in price_points:
                close_price = point.get('close', 'N/A')
                open_price = point.get('open', 'N/A')
                high_price = point.get('high', 'N/A')
                low_price = point.get('low', 'N/A')
                
                prompt += f"- {point['date']}: Open ${open_price}, High ${high_price}, Low ${low_price}, Close ${close_price}"
                
                volume = point.get('volume')
                if volume is not None:
                    prompt += f", Volume {volume}"
                    
                prompt += "\n"
            
            # Add a few basic indicators for context
            if basic_indicators:
                prompt += "\nBASIC TECHNICAL INDICATORS (FOR CONTEXT):\n"
                for indicator, value in basic_indicators.items():
                    prompt += f"- {indicator}: {value}\n"
            
            # Add news data
            prompt += "\nRECENT NEWS:\n"
            for i, item in enumerate(news_items[:5]):  # Limit to 5 most recent news items
                date = item.get('date', 'N/A')
                headline = item.get('headline', 'No headline available')
                summary = item.get('summary', 'No summary available')
                prompt += f"- {date}: {headline}\n  Summary: {summary}\n"
            
            # Add additional instructions
            prompt += """
            PREDICTION REQUIREMENTS:
            1. Analyze the raw price data yourself, applying your expertise in technical analysis and market behavior.
            2. Consider market context, sector trends, and how news might impact the stock.
            3. Provide a detailed price movement prediction for this stock for the specified timeframe with a percentage change forecast.
            4. Explain the key factors driving this prediction, detailing how much weight you assign to technical versus sentiment factors.
            5. Provide a confidence level (high, medium, or low) and explain why.
            6. Give a specific target price range (low, mid, high) based on your prediction.
            7. Make a clear investment recommendation (e.g., Strong Buy, Buy, Hold, Sell, etc.) with clear reasoning.
            
            IMPORTANT ANALYSIS GUIDANCE:
            - Perform your own analysis on the raw data rather than relying on predetermined indicators.
            - Extract patterns, trends, and correlations from the raw price and volume history.
            - Calculate your own technical indicators if needed (RSI, MACD, Bollinger Bands, etc.).
            - Identify chart patterns in the data (head and shoulders, double tops/bottoms, etc.).
            - Draw upon your full knowledge of financial markets, recent market conditions, and sector trends.
            - Integrate both technical patterns and news sentiment in your prediction.
            - Base predictions on probability distributions that account for market volatility.
            - Consider the specific time horizon for your forecast carefully in your analysis.
            
            Format your response as JSON with these keys:
            {
              "predicted_change_pct": number,
              "confidence": "high/medium/low",
              "target_price_low": number,
              "target_price_mid": number, 
              "target_price_high": number,
              "recommendation": "string",
              "technical_contribution": {
                "weight": number,
                "signal": "bullish/bearish/neutral",
                "explanation": "string",
                "patterns_identified": ["string"]
              },
              "sentiment_contribution": {
                "weight": number,
                "explanation": "string"
              },
              "enhanced_analysis": "string (2-3 paragraphs of detailed analysis)"
            }
            """
            
            # Initialize Gemini model with the latest version
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Generate response with structured output
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more deterministic output
                    top_p=0.95,
                    top_k=40,
                    candidate_count=1,
                )
            )
            
            response_text = response.text
            
            # Extract JSON portion
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                try:
                    ai_result = json.loads(json_str)
                except json.JSONDecodeError:
                    print("Failed to parse AI response as JSON, falling back to simple prediction")
                    print(f"Response: {response_text}")
                    return self._generate_simple_prediction(symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon)
            else:
                print("No JSON found in AI response, falling back to simple prediction")
                return self._generate_simple_prediction(symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon)
            
            # Calculate signal based on predicted change
            signal = "bullish" if ai_result.get('predicted_change_pct', 0) > 0 else "bearish" if ai_result.get('predicted_change_pct', 0) < 0 else "neutral"
            
            # Compile final prediction
            prediction = {
                "symbol": symbol,
                "company_name": company_name,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat(),
                "prediction": {
                    "signal": signal,
                    "predicted_change_pct": ai_result.get('predicted_change_pct', 0),
                    "confidence": ai_result.get('confidence', 'medium'),
                    "target_price_low": ai_result.get('target_price_low', current_price * 0.98),
                    "target_price_mid": ai_result.get('target_price_mid', current_price),
                    "target_price_high": ai_result.get('target_price_high', current_price * 1.02),
                    "data_timeframe": technical_timeframe,
                    "prediction_horizon": prediction_horizon,
                    "prediction_horizon_text": prediction_horizon_text,
                    "recommendation": ai_result.get('recommendation', 'Hold')
                },
                "technical_contribution": {
                    "weight": ai_result.get('technical_contribution', {}).get('weight', 0.6),
                    "signal": ai_result.get('technical_contribution', {}).get('signal', signal),
                    "explanation": ai_result.get('technical_contribution', {}).get('explanation', ''),
                    "patterns_identified": ai_result.get('technical_contribution', {}).get('patterns_identified', []),
                    "basic_indicators": basic_indicators
                },
                "sentiment_contribution": {
                    "weight": ai_result.get('sentiment_contribution', {}).get('weight', 0.4),
                    "explanation": ai_result.get('sentiment_contribution', {}).get('explanation', ''),
                    "news_items": news_items[:5]  # Include top 5 news items
                },
                "analysis": {
                    "enhanced_analysis": ai_result.get('enhanced_analysis', '')
                }
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error in AI-enhanced prediction: {e}")
            # Fall back to simpler prediction method
            return self._generate_simple_prediction(symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon)
    
    def _get_representative_indices(self, length):
        """
        Get representative indices to sample data points.
        Includes first, last, and evenly spaced points in between.
        
        Args:
            length (int): Length of the array
            
        Returns:
            list: Indices to sample
        """
        if length <= 10:
            return list(range(length))
        
        # Always include first and last points
        indices = [0, length-1]
        
        # Add some evenly spaced points in between
        num_points = 8  # 8 points + first + last = 10 total
        step = length // (num_points + 1)
        
        for i in range(1, num_points + 1):
            idx = i * step
            if idx < length - 1:  # Avoid duplicating the last point
                indices.append(idx)
        
        return sorted(indices)
    
    def _generate_simple_prediction(self, symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon):
        """
        Generate a simplified prediction as fallback when AI is unavailable.
        This is a very basic implementation that should be used only when the primary AI method fails.
        
        Args:
            symbol (str): Stock symbol
            technical_data (dict): Technical analysis data
            sentiment_data (dict): Sentiment analysis data
            technical_timeframe (str): Timeframe for technical data
            prediction_horizon (str): Timeframe for prediction
            
        Returns:
            dict: Combined prediction
        """
        # Extract technical signals
        current_price = technical_data.get('current_price', 0)
        price_change = technical_data.get('price_change_pct', 0)
        volatility = technical_data.get('volatility', 20)
        basic_indicators = technical_data.get('basic_indicators', {})
        
        # Get RSI if available
        rsi = basic_indicators.get('RSI_14', 50)
        
        # Get prediction horizon text for display
        prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
        
        # Extract news data
        news_items = sentiment_data.get('news_items', [])
        
        # Simple heuristic for price direction
        # - If price has been rising and RSI < 70, likely to continue
        # - If price has been falling and RSI > 30, likely to continue
        # - If price rising and RSI > 70, potential reversal (overbought)
        # - If price falling and RSI < 30, potential reversal (oversold)
        predicted_change_pct = 0
        confidence = "low"
        signal = "neutral"
        
        if price_change > 0 and rsi < 70:
            # Uptrend likely to continue
            predicted_change_pct = min(price_change * 0.5, volatility * 0.25)
            confidence = "medium"
            signal = "bullish"
        elif price_change < 0 and rsi > 30:
            # Downtrend likely to continue
            predicted_change_pct = max(price_change * 0.5, -volatility * 0.25)
            confidence = "medium"
            signal = "bearish"
        elif price_change > 0 and rsi > 70:
            # Potential reversal from overbought
            predicted_change_pct = -volatility * 0.2
            confidence = "low"
            signal = "bearish"
        elif price_change < 0 and rsi < 30:
            # Potential reversal from oversold
            predicted_change_pct = volatility * 0.2
            confidence = "low"
            signal = "bullish"
        else:
            # Neutral case
            predicted_change_pct = price_change * 0.1
            confidence = "low"
            signal = "neutral"
        
        # Adjust for timeframe
        if prediction_horizon == '2d':
            predicted_change_pct *= 1.5
        elif prediction_horizon == '1w':
            predicted_change_pct *= 3
        elif prediction_horizon == '1mo':
            predicted_change_pct *= 5
        
        # Generate recommendation
        if predicted_change_pct > 3:
            recommendation = "Buy"
        elif predicted_change_pct > 1:
            recommendation = "Accumulate"
        elif predicted_change_pct > -1:
            recommendation = "Hold"
        elif predicted_change_pct > -3:
            recommendation = "Reduce"
        else:
            recommendation = "Sell"
        
        # Calculate target prices
        target_mid = current_price * (1 + predicted_change_pct / 100)
        target_low = current_price * (1 + (predicted_change_pct - volatility * 0.2) / 100)
        target_high = current_price * (1 + (predicted_change_pct + volatility * 0.2) / 100)
        
        # Compile final prediction
        prediction = {
            "symbol": symbol,
            "company_name": technical_data.get('company_name', sentiment_data.get('company_name', symbol)),
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "signal": signal,
                "predicted_change_pct": predicted_change_pct,
                "confidence": confidence,
                "target_price_low": target_low,
                "target_price_mid": target_mid,
                "target_price_high": target_high,
                "data_timeframe": technical_timeframe,
                "prediction_horizon": prediction_horizon,
                "prediction_horizon_text": prediction_horizon_text,
                "recommendation": recommendation
            },
            "technical_contribution": {
                "weight": 0.8,
                "signal": signal,
                "explanation": "Basic technical analysis suggests a " + signal + " outlook based on recent price action and RSI.",
                "patterns_identified": [],
                "basic_indicators": basic_indicators
            },
            "sentiment_contribution": {
                "weight": 0.2,
                "explanation": "Limited news sentiment analysis included in this fallback prediction.",
                "news_items": news_items[:3]  # Include top 3 news items
            },
            "analysis": {
                "enhanced_analysis": f"This is a fallback prediction generated using basic heuristics since AI analysis was unavailable. The {signal} outlook is based primarily on the recent price movement of {price_change:.2f}% and current RSI of {rsi:.1f}. Given the {prediction_horizon_text} timeframe, we estimate a {predicted_change_pct:.2f}% move with {confidence} confidence."
            }
        }
        
        return prediction
    
    def track_prediction_performance(self, symbol, prediction_data, actual_price, prediction_horizon='1d'):
        """
        Track the performance of a prediction.
        
        Args:
            symbol (str): Stock symbol
            prediction_data (dict): Original prediction data
            actual_price (float): Actual price at target date
            prediction_horizon (str, optional): Timeframe used for prediction.
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Extract prediction details
            pred_timestamp = datetime.fromisoformat(prediction_data.get('timestamp'))
            pred_price = prediction_data.get('current_price', 0)
            pred_change_pct = prediction_data.get('prediction', {}).get('predicted_change_pct', 0)
            pred_signal = prediction_data.get('prediction', {}).get('signal', 'neutral')
            confidence = prediction_data.get('prediction', {}).get('confidence', 'low')
            
            # Calculate actual change
            actual_change_pct = ((actual_price - pred_price) / pred_price) * 100 if pred_price > 0 else 0
            
            # Determine if prediction was correct
            correct_direction = (pred_change_pct > 0 and actual_change_pct > 0) or (pred_change_pct < 0 and actual_change_pct < 0)
            
            # Calculate error
            error_pct = abs(pred_change_pct - actual_change_pct)
            
            # Create performance record
            performance = {
                "symbol": symbol,
                "prediction_date": pred_timestamp.strftime('%Y-%m-%d'),
                "evaluation_date": datetime.now().strftime('%Y-%m-%d'),
                "prediction_horizon": prediction_horizon,
                "predicted_change_pct": pred_change_pct,
                "actual_change_pct": actual_change_pct,
                "correct_direction": correct_direction,
                "error_pct": error_pct,
                "confidence": confidence
            }
            
            # Store in performance history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
            
            self.performance_history[symbol].append(performance)
            
            # Save performance history
            self._save_performance_history()
            
            return performance
            
        except Exception as e:
            print(f"Error tracking prediction performance: {e}")
            return {"error": str(e)}
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            performance_file = os.path.join(PREDICTIONS_DIR, "performance_history.json")
            with open(performance_file, 'w') as f:
                json.dump(self.performance_history, f)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def load_performance_history(self):
        """Load performance history from disk."""
        try:
            performance_file = os.path.join(PREDICTIONS_DIR, "performance_history.json")
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    self.performance_history = json.load(f)
            return self.performance_history
        except Exception as e:
            print(f"Error loading performance history: {e}")
            return {}
    
    def get_performance_metrics(self, symbol=None):
        """
        Get performance metrics for predictions.
        
        Args:
            symbol (str, optional): Symbol to get metrics for, or None for all
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Load performance history
            self.load_performance_history()
            
            # Filter for symbol if specified
            if symbol:
                history = self.performance_history.get(symbol, [])
                symbols = [symbol]
            else:
                history = []
                symbols = list(self.performance_history.keys())
                for sym in symbols:
                    history.extend(self.performance_history[sym])
            
            if not history:
                return {"error": "No performance history found"}
            
            # Calculate overall metrics
            total_predictions = len(history)
            correct_directions = sum(1 for item in history if item.get('correct_direction', False))
            accuracy = (correct_directions / total_predictions) * 100 if total_predictions > 0 else 0
            
            avg_error = sum(item.get('error_pct', 0) for item in history) / total_predictions if total_predictions > 0 else 0
            
            # Calculate metrics by confidence level
            confidence_metrics = {}
            for conf in ["high", "medium", "low"]:
                conf_history = [item for item in history if item.get('confidence') == conf]
                conf_total = len(conf_history)
                if conf_total > 0:
                    conf_correct = sum(1 for item in conf_history if item.get('correct_direction', False))
                    conf_accuracy = (conf_correct / conf_total) * 100
                    confidence_metrics[conf] = {
                        "total_predictions": conf_total,
                        "correct_predictions": conf_correct,
                        "accuracy": conf_accuracy
                    }
            
            # Calculate metrics by symbol if multiple symbols
            symbol_metrics = {}
            if len(symbols) > 1:
                for sym in symbols:
                    sym_history = self.performance_history.get(sym, [])
                    sym_total = len(sym_history)
                    if sym_total > 0:
                        sym_correct = sum(1 for item in sym_history if item.get('correct_direction', False))
                        sym_accuracy = (sym_correct / sym_total) * 100
                        symbol_metrics[sym] = {
                            "total_predictions": sym_total,
                            "correct_predictions": sym_correct,
                            "accuracy": sym_accuracy
                        }
            
            # Calculate metrics by prediction horizon
            horizon_metrics = {}
            for horizon in ['1d', '2d', '1w', '1mo']:
                horizon_history = [item for item in history if item.get('prediction_horizon') == horizon]
                horizon_total = len(horizon_history)
                if horizon_total > 0:
                    horizon_correct = sum(1 for item in horizon_history if item.get('correct_direction', False))
                    horizon_accuracy = (horizon_correct / horizon_total) * 100
                    horizon_metrics[horizon] = {
                        "total_predictions": horizon_total,
                        "correct_predictions": horizon_correct,
                        "accuracy": horizon_accuracy
                    }
            
            # Compile metrics
            metrics = {
                "overall": {
                    "total_predictions": total_predictions,
                    "correct_predictions": correct_directions,
                    "accuracy": accuracy,
                    "average_error": avg_error
                },
                "by_confidence": confidence_metrics,
                "by_symbol": symbol_metrics if len(symbols) > 1 else {},
                "by_horizon": horizon_metrics,
                "recent_predictions": sorted(history, key=lambda x: x.get('prediction_date', ''), reverse=True)[:10]
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error getting performance metrics: {e}")
            return {"error": str(e)}


# Function to generate prediction for a stock
def generate_aiplus_prediction(symbol, technical_timeframe='1mo', news_days=7, prediction_horizon='1d', force_refresh=False):
    """
    Generate an enhanced stock prediction using AI+.
    
    Args:
        symbol (str): Stock symbol
        technical_timeframe (str): Timeframe for technical data
        news_days (int): Days for news analysis
        prediction_horizon (str): Timeframe for prediction (1d, 2d, 1w, 1mo)
        force_refresh (bool): Whether to force refresh data
        
    Returns:
        dict: Prediction results
    """
    predictor = AIplusPredictor()
    return predictor.generate_prediction(symbol, technical_timeframe, news_days, prediction_horizon, force_refresh)


# Function to get performance metrics
def get_aiplus_performance(symbol=None):
    """
    Get performance metrics for AI+ predictions.
    
    Args:
        symbol (str, optional): Symbol to get metrics for, or None for all
        
    Returns:
        dict: Performance metrics
    """
    predictor = AIplusPredictor()
    return predictor.get_performance_metrics(symbol)