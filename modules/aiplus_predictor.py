"""
Enhanced AI+ Predictor Module that leverages Gemini's mathematical expertise

This module combines technical analysis and news sentiment to generate
enhanced stock price predictions using advanced AI capabilities.
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
    Class for combining technical and sentiment data to make enhanced predictions
    using advanced AI capabilities.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.api_initialized = configure_gemini()
        self.predictions = {}
        self.performance_history = {}
    
    def generate_prediction(self, symbol, technical_timeframe='1mo', news_days=7, prediction_horizon='1d', force_refresh=False):
        """
        Generate an enhanced stock prediction using both technical and news data with advanced AI analysis.
        
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
        Use advanced AI to analyze and predict stock movements.
        
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
            price_change = technical_data.get('price_change_pct', 0)
            volatility = technical_data.get('volatility', 0)
            
            # Get technical indicators
            indicators = technical_data.get('standard_indicators', {})
            tsmn = technical_data.get('tsmn', {})
            tsmn_value = tsmn.get('value', 0) if isinstance(tsmn, dict) else 0
            
            # Get sentiment data
            sentiment = sentiment_data.get('sentiment', 'neutral')
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            
            # Format timeframes for AI prompt
            ai_timeframe = {
                '1mo': 'one month',
                '3mo': 'three months',
                '6mo': 'six months',
                '1y': 'one year'
            }.get(technical_timeframe, technical_timeframe)
            
            prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
            
            # Create prompt for Gemini with raw technical and sentiment data
            prompt = f"""
            You are an expert financial analyst with deep expertise in quantitative finance, technical analysis, and market psychology.

            Please analyze the following data for {symbol} and provide a detailed price movement prediction for the {prediction_horizon_text}.

            TECHNICAL ANALYSIS DATA:
            - Current Price: ${current_price:.2f}
            - Price Change ({ai_timeframe}): {price_change:.2f}%
            - Volatility (annualized): {volatility:.2f}%
            - TSMN Value: {tsmn_value:.2f}
            """
            
            # Add technical indicators
            prompt += "\nTECHNICAL INDICATORS:\n"
            for indicator, value in indicators.items():
                prompt += f"- {indicator}: {value}\n"
            
            # Add more technical data if available
            if 'price_momentum' in technical_data:
                prompt += "\nPRICE MOMENTUM:\n"
                for period, value in technical_data['price_momentum'].items():
                    prompt += f"- {period}: {value:.2f}%\n"
            
            if 'volume_analysis' in technical_data:
                prompt += "\nVOLUME ANALYSIS:\n"
                for metric, value in technical_data['volume_analysis'].items():
                    prompt += f"- {metric}: {value}\n"
            
            if 'pattern_detection' in technical_data:
                patterns = technical_data['pattern_detection']
                if isinstance(patterns, dict) and patterns:
                    prompt += "\nPATTERN DETECTION:\n"
                    for pattern, data in patterns.items():
                        if pattern != 'error' and isinstance(data, dict):
                            if data.get('detected', False):
                                prompt += f"- {pattern}: Detected with {data.get('confidence', 'unknown')} confidence\n"
            
            # Add sentiment data
            prompt += f"""
            SENTIMENT ANALYSIS:
            - Overall Sentiment: {sentiment}
            - Sentiment Score: {sentiment_score:.2f}
            """
            
            # Add key developments from news
            key_developments = sentiment_data.get('key_developments', [])
            if key_developments:
                prompt += "\nKEY DEVELOPMENTS:\n"
                for dev in key_developments[:3]:  # Top 3 developments
                    date = dev.get('date', '')
                    headline = dev.get('headline', '')
                    prompt += f"- {date}: {headline}\n"
            
            # Add additional context
            prompt += f"""
            PREDICTION REQUIREMENTS:
            1. Provide a detailed price movement prediction for {symbol} for the {prediction_horizon_text} with a percentage change forecast.
            2. Explain the key factors driving this prediction, detailing how much weight you assign to technical versus sentiment factors.
            3. Provide a confidence level (high, medium, or low) and explain why.
            4. Give a specific target price range (low, mid, high) based on your prediction.
            5. Make a clear investment recommendation (e.g., Strong Buy, Buy, Hold, Sell, etc.) with clear reasoning.
            
            IMPORTANT ANALYSIS GUIDANCE:
            - Use your advanced mathematical expertise to analyze the correlations between these indicators.
            - Draw upon your full knowledge of financial markets, recent market conditions, and sector trends.
            - Integrate both technical signals and sentiment analysis in your prediction.
            - Base predictions on probability distributions that account for market volatility.
            - Consider the specific time horizon ({prediction_horizon_text}) carefully in your analysis.
            
            Format your response as JSON with these keys:
            {
                "predicted_change_pct": number,
                "confidence": "high/medium/low",
                "target_price_low": number,
                "target_price_mid": number, 
                "target_price_high": number,
                "recommendation": "string",
                "technical_contribution": {
                    "weight": number (0-1),
                    "signal": "bullish/bearish/neutral",
                    "explanation": "string"
                },
                "sentiment_contribution": {
                    "weight": number (0-1),
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
                "company_name": technical_data.get('company_name', sentiment_data.get('company_name', symbol)),
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
                    "key_indicators": technical_data.get('standard_indicators', {})
                },
                "sentiment_contribution": {
                    "weight": ai_result.get('sentiment_contribution', {}).get('weight', 0.4),
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "explanation": ai_result.get('sentiment_contribution', {}).get('explanation', ''),
                    "key_developments": sentiment_data.get('key_developments', []),
                    "sentiment_drivers": sentiment_data.get('sentiment_drivers', [])
                },
                "analysis": {
                    "technical_summary": technical_data.get('technical_summary', ''),
                    "sentiment_summary": sentiment_data.get('sentiment_summary', ''),
                    "enhanced_analysis": ai_result.get('enhanced_analysis', '')
                }
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error in AI-enhanced prediction: {e}")
            # Fall back to simpler prediction method
            return self._generate_simple_prediction(symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon)
    
    def _generate_simple_prediction(self, symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon):
        """
        Generate a simplified prediction using weighted approach when AI is unavailable.
        
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
        tsmn = technical_data.get('tsmn', {})
        tsmn_value = tsmn.get('value', 0) if isinstance(tsmn, dict) else 0
        technical_indicators = technical_data.get('standard_indicators', {})
        
        # Extract sentiment signals
        sentiment = sentiment_data.get('sentiment', 'neutral')
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        
        # Technical weighting (higher during high volatility periods)
        volatility = technical_data.get('volatility', 20)
        tech_weight = min(max(0.6, volatility / 50), 0.8)  # 0.6-0.8 range based on volatility
        
        # Sentiment weighting (inversely related to technical weight)
        sent_weight = 1 - tech_weight
        
        # Calculate combined signal (-100 to +100 scale)
        combined_signal = (tsmn_value * tech_weight) + (sentiment_score * sent_weight)
        
        # Normalize to percent change range based on volatility and prediction horizon
        if prediction_horizon == '1d':
            # For 1-day prediction, use a smaller fraction of volatility
            max_move = volatility / 30  # e.g., 30% volatility -> 1% max daily move
        elif prediction_horizon == '2d':
            # For 2-day prediction, slightly larger
            max_move = volatility / 25  # e.g., 30% volatility -> 1.2% max 2-day move
        elif prediction_horizon == '1w':
            # For 1-week prediction, larger
            max_move = volatility / 15  # e.g., 30% volatility -> 2% max weekly move
        else:
            # Original scaling for monthly timeframes
            max_move = volatility / 10  # e.g., 30% volatility -> 3% max move
            
        predicted_change_pct = combined_signal * max_move / 100
        
        # Calculate confidence level based on signal consensus
        tech_signal = 1 if tsmn_value > 0 else -1 if tsmn_value < 0 else 0
        sent_signal = 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0
        
        confidence = "low"
        if tech_signal == sent_signal and tech_signal != 0:
            # Both signals agree and aren't neutral
            confidence = "high" if abs(combined_signal) > 40 else "medium"
        elif tech_signal != 0 and sent_signal != 0:
            # Both signals are non-neutral but disagree
            confidence = "low"
        else:
            # One or both signals are neutral
            non_neutral_signal = tech_signal if tech_signal != 0 else sent_signal
            confidence = "medium" if abs(combined_signal) > 40 else "low"
        
        # Generate prediction recommendation
        if predicted_change_pct > 3:
            recommendation = "Strong Buy" if confidence == "high" else "Buy"
        elif predicted_change_pct > 1:
            recommendation = "Buy" if confidence == "high" else "Accumulate"
        elif predicted_change_pct > 0:
            recommendation = "Hold with Positive Bias"
        elif predicted_change_pct > -1:
            recommendation = "Hold"
        elif predicted_change_pct > -3:
            recommendation = "Hold with Negative Bias" if confidence != "high" else "Reduce"
        else:
            recommendation = "Sell" if confidence == "high" else "Reduce"
        
        # Calculate target prices
        target_low = current_price * (1 + (predicted_change_pct - (volatility / 30)) / 100)
        target_high = current_price * (1 + (predicted_change_pct + (volatility / 30)) / 100)
        target_mid = current_price * (1 + predicted_change_pct / 100)
        
        # Get prediction horizon text for display
        prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
        
        # Compile final prediction
        prediction = {
            "symbol": symbol,
            "company_name": technical_data.get('company_name', sentiment_data.get('company_name', symbol)),
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "signal": "bullish" if predicted_change_pct > 0 else "bearish" if predicted_change_pct < 0 else "neutral",
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
                "weight": tech_weight,
                "signal": "bullish" if tsmn_value > 0 else "bearish" if tsmn_value < 0 else "neutral",
                "tsmn_value": tsmn_value,
                "key_indicators": self._extract_key_indicators(technical_indicators)
            },
            "sentiment_contribution": {
                "weight": sent_weight,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "key_developments": sentiment_data.get('key_developments', []),
                "sentiment_drivers": sentiment_data.get('sentiment_drivers', [])
            },
            "analysis": {
                "technical_summary": technical_data.get('technical_summary', ''),
                "sentiment_summary": sentiment_data.get('sentiment_summary', ''),
                "enhanced_analysis": self._generate_fallback_analysis(symbol, technical_data, sentiment_data, predicted_change_pct, confidence, prediction_horizon)
            }
        }
        
        return prediction
    
    def _extract_key_indicators(self, indicators):
        """
        Extract the most important technical indicators.
        
        Args:
            indicators (dict): All technical indicators
            
        Returns:
            dict: Key indicators
        """
        key_indicators = {}
        
        # RSI
        if 'RSI_14' in indicators:
            key_indicators['RSI_14'] = indicators['RSI_14']
        
        # MACD
        if 'MACD' in indicators and 'MACD_Signal' in indicators:
            key_indicators['MACD'] = indicators['MACD']
            key_indicators['MACD_Signal'] = indicators['MACD_Signal']
        
        # Bollinger Bands
        if 'BB_Percent' in indicators:
            key_indicators['BB_Percent'] = indicators['BB_Percent']
        
        # ADX (trend strength)
        if 'ADX' in indicators:
            key_indicators['ADX'] = indicators['ADX']
        
        # Moving Averages
        for ma in ['MA_20', 'MA_50', 'MA_200']:
            if ma in indicators:
                key_indicators[ma] = indicators[ma]
        
        return key_indicators
    
    def _generate_fallback_analysis(self, symbol, technical_data, sentiment_data, predicted_change_pct, confidence, prediction_horizon):
        """
        Generate a fallback analysis when AI service is not available.
        
        Args:
            symbol (str): Stock symbol
            technical_data (dict): Technical analysis data
            sentiment_data (dict): Sentiment analysis data
            predicted_change_pct (float): Predicted percentage change
            confidence (str): Confidence level
            prediction_horizon (str): Timeframe for prediction
            
        Returns:
            str: Fallback analysis
        """
        # Get prediction horizon text for display
        prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
        
        # Get technical and sentiment summaries
        tech_summary = technical_data.get('technical_summary', '')
        sent_summary = sentiment_data.get('sentiment_summary', '')
        
        # Create fallback analysis
        fallback_analysis = f"Based on our combined technical and sentiment analysis, we anticipate a {predicted_change_pct:.2f}% move for {symbol} in the {prediction_horizon_text}. "
        
        if predicted_change_pct > 0:
            fallback_analysis += "The positive outlook is "
        elif predicted_change_pct < 0:
            fallback_analysis += "The negative outlook is "
        else:
            fallback_analysis += "The neutral outlook is "
        
        fallback_analysis += f"supported with {confidence} confidence based on the correlation between technical indicators and sentiment analysis. "
        
        # Add technical and sentiment context
        if tech_summary:
            fallback_analysis += f"From a technical perspective, {tech_summary} "
        
        if sent_summary:
            fallback_analysis += f"Sentiment analysis reveals that {sent_summary}"
        
        return fallback_analysis
    
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