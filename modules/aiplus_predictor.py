"""
AI+ Predictor Module

This module combines technical analysis and news sentiment to generate
enhanced stock price predictions using both data streams.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import config
from dotenv import load_dotenv

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
    Class for combining technical and sentiment data to make enhanced predictions.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.api_initialized = configure_gemini()
        self.predictions = {}
        self.performance_history = {}
    
    def generate_prediction(self, symbol, technical_timeframe='1mo', news_days=7, prediction_horizon='1d', force_refresh=False):
        """
        Generate an enhanced stock prediction using both technical and news data.
        
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
            
            # Step 3: Process the data for prediction
            # This combines both data sources and enhances with AI analysis
            prediction_results = self._generate_combined_prediction(
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
    
    def _generate_combined_prediction(self, symbol, technical_data, sentiment_data, technical_timeframe, prediction_horizon):
        """
        Combine technical and sentiment data for enhanced prediction.
        
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
        
        # Convert sentiment to numeric value
        sentiment_value = 0
        if sentiment == 'positive':
            sentiment_value = 1
        elif sentiment == 'negative':
            sentiment_value = -1
        elif sentiment == 'mixed':
            sentiment_value = sentiment_score / 100  # Scale between -1 and 1
        
        # Combine the signals using weighted approach
        
        # Technical weighting (higher during high volatility periods)
        volatility = technical_data.get('volatility', 20)
        tech_weight = min(max(0.6, volatility / 50), 0.8)  # 0.6-0.8 range based on volatility
        
        # Sentiment weighting (inversely related to technical weight)
        sent_weight = 1 - tech_weight
        
        # Calculate combined signal (-100 to +100 scale)
        combined_signal = (tsmn_value * tech_weight) + (sentiment_score * sent_weight)
        
        # Normalize to percent change range based on volatility and prediction horizon
        # Higher volatility = larger potential moves
        # Shorter timeframes = smaller expected moves
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
        
        # Generate prediction recommendation based on prediction horizon
        recommendation = self._generate_recommendation(
            predicted_change_pct, confidence, technical_data, sentiment_data, prediction_horizon
        )
        
        # Use LLM to enhance analysis if available
        enhanced_analysis = ""
        if self.api_initialized:
            enhanced_analysis = self._generate_llm_analysis(
                symbol, technical_data, sentiment_data, 
                predicted_change_pct, confidence, prediction_horizon
            )
        
        # Calculate target prices
        target_low = current_price * (1 + (predicted_change_pct - (volatility / 30)) / 100)
        target_high = current_price * (1 + (predicted_change_pct + (volatility / 30)) / 100)
        
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
                "target_price_mid": current_price * (1 + predicted_change_pct / 100),
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
                "enhanced_analysis": enhanced_analysis
            }
        }
        
        return prediction
    
    def _generate_recommendation(self, predicted_change_pct, confidence, technical_data, sentiment_data, prediction_horizon):
        """
        Generate investment recommendation based on prediction.
        
        Args:
            predicted_change_pct (float): Predicted percentage change
            confidence (str): Confidence level
            technical_data (dict): Technical analysis data
            sentiment_data (dict): Sentiment analysis data
            prediction_horizon (str): Timeframe for prediction
            
        Returns:
            str: Investment recommendation
        """
        # Extract additional context
        volatility = technical_data.get('volatility', 20)
        market_cap = technical_data.get('market_cap', 0)
        
        # Adjust thresholds based on prediction horizon
        if prediction_horizon == '1d':
            # Daily thresholds are smaller
            strong_threshold = 1.0  # 1% for daily is significant
            moderate_threshold = 0.5  # 0.5%
            small_threshold = 0.2  # 0.2%
            neg_moderate_threshold = -0.5  # -0.5%
            neg_strong_threshold = -1.0  # -1.0%
        elif prediction_horizon == '2d':
            # 2-day thresholds
            strong_threshold = 1.5  # 1.5%
            moderate_threshold = 0.8  # 0.8%
            small_threshold = 0.3  # 0.3%
            neg_moderate_threshold = -0.8  # -0.8%
            neg_strong_threshold = -1.5  # -1.5%
        elif prediction_horizon == '1w':
            # Weekly thresholds
            strong_threshold = 2.5  # 2.5%
            moderate_threshold = 1.2  # 1.2%
            small_threshold = 0.5  # 0.5%
            neg_moderate_threshold = -1.2  # -1.2%
            neg_strong_threshold = -2.5  # -2.5%
        else:
            # Monthly thresholds (original)
            strong_threshold = 5.0  # 5%
            moderate_threshold = 2.0  # 2%
            small_threshold = 0.0  # 0%
            neg_moderate_threshold = -2.0  # -2%
            neg_strong_threshold = -5.0  # -5%
        
        # Base recommendation on predicted change and confidence
        if predicted_change_pct > strong_threshold:
            if confidence == "high":
                return "Strong Buy"
            elif confidence == "medium":
                return "Buy"
            else:
                return "Speculative Buy"
        elif predicted_change_pct > moderate_threshold:
            if confidence == "high":
                return "Buy"
            elif confidence == "medium":
                return "Accumulate"
            else:
                return "Speculative Buy"
        elif predicted_change_pct > small_threshold:
            if confidence == "high":
                return "Accumulate"
            else:
                return "Hold with Positive Bias"
        elif predicted_change_pct > neg_moderate_threshold:
            if confidence == "high":
                return "Hold with Negative Bias"
            else:
                return "Hold"
        elif predicted_change_pct > neg_strong_threshold:
            if confidence == "high":
                return "Reduce"
            elif confidence == "medium":
                return "Hold with Negative Bias"
            else:
                return "Hold"
        else:
            if confidence == "high":
                return "Sell"
            elif confidence == "medium":
                return "Reduce"
            else:
                return "Hold with Negative Bias"
    
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
    
    def _generate_llm_analysis(self, symbol, technical_data, sentiment_data, predicted_change_pct, confidence, prediction_horizon):
        """
        Generate enhanced analysis using LLM.
        
        Args:
            symbol (str): Stock symbol
            technical_data (dict): Technical analysis data
            sentiment_data (dict): Sentiment analysis data
            predicted_change_pct (float): Predicted percentage change
            confidence (str): Confidence level
            prediction_horizon (str): Timeframe for prediction
            
        Returns:
            str: Enhanced analysis
        """
        try:
            # Extract key information for the prompt
            company_name = technical_data.get('company_name', sentiment_data.get('company_name', symbol))
            tech_summary = technical_data.get('technical_summary', '')
            sent_summary = sentiment_data.get('sentiment_summary', '')
            
            # Get key developments
            key_developments = sentiment_data.get('key_developments', [])
            news_highlights = "\n".join([
                f"- {dev.get('date', '')}: {dev.get('headline', '')}" 
                for dev in key_developments[:3]
            ])
            
            # Technical indicators to highlight
            indicators = technical_data.get('standard_indicators', {})
            tsmn = technical_data.get('tsmn', {})
            tsmn_value = tsmn.get('value', 0) if isinstance(tsmn, dict) else 0
            volatility = technical_data.get('volatility', 20)
            
            rsi = indicators.get('RSI_14', 'N/A')
            macd = indicators.get('MACD', 'N/A')
            bb_percent = indicators.get('BB_Percent', 'N/A')
            
            indicator_highlights = f"""
            TSMN Value: {tsmn_value}
            Volatility: {volatility}%
            RSI (14): {rsi}
            MACD: {macd}
            Bollinger %B: {bb_percent}
            """
            
            # Get prediction horizon text for display
            prediction_horizon_text = PREDICTION_HORIZON_TEXT.get(prediction_horizon, 'specified period')
            
            # Craft the prompt
            prompt = f"""
            Please provide a concise analysis of {company_name} ({symbol}) based on the following information:
            
            TECHNICAL ANALYSIS SUMMARY:
            {tech_summary}
            
            SENTIMENT ANALYSIS SUMMARY:
            {sent_summary}
            
            TECHNICAL INDICATORS:
            {indicator_highlights}
            
            RECENT NEWS HIGHLIGHTS:
            {news_highlights}
            
            PREDICTION HORIZON: {prediction_horizon_text}
            
            PREDICTION:
            - Predicted Change for {prediction_horizon_text}: {predicted_change_pct:.2f}%
            - Confidence: {confidence}
            
            Generate 2-3 paragraphs of insightful analysis combining both technical and fundamental factors,
            focusing specifically on the {prediction_horizon_text} outlook.
            Focus on synthesizing information rather than repeating it. Include insights about potential catalysts
            and risks. Be specific about this stock, mentioning the company and industry by name.
            Conclude with a balanced perspective on the investment opportunity.
            
            Do not include any disclaimers or reminders that you are an AI assistant.
            """
            
            # Initialize Gemini model and generate analysis
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            analysis = response.text.strip()
            
            return analysis
            
        except Exception as e:
            print(f"Error generating enhanced analysis: {e}")
            return "Enhanced analysis unavailable. Please refer to the technical and sentiment summaries above."
    
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


# Test function
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    
    # Generate prediction
    prediction = generate_aiplus_prediction(symbol, prediction_horizon='1d', force_refresh=True)
    
    print(json.dumps(prediction, indent=2))
    
    # Get performance metrics
    performance = get_aiplus_performance()
    print(json.dumps(performance, indent=2))