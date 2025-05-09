"""
AI+ Predictor Module - Streamlined Version

Combines technical analysis and news sentiment to generate enhanced stock price predictions.
"""

import os
import json
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import config
from dotenv import load_dotenv

# Import modules
from modules.aiplus_technical import get_aiplus_technical_data
from modules.aiplus_sentiment import get_aiplus_sentiment

# Load environment variables and setup
load_dotenv()
PREDICTIONS_DIR = os.path.join(config.DATA_DIR, "aiplus_predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Mapping for clearer display
HORIZON_TEXT = {'1d': 'next day', '2d': 'next two days', '1w': 'next week', '1mo': 'next month'}

class AIplusPredictor:
    """Core predictor class that handles all prediction functionality."""
    
    def __init__(self):
        """Initialize the predictor with API and history."""
        api_key = os.environ.get('GEMINI_API_KEY', config.GEMINI_API_KEY)
        self.api_initialized = bool(api_key)
        if self.api_initialized:
            genai.configure(api_key=api_key)
        
        self.performance_history = self._load_performance_history()
    
    def generate_prediction(self, symbol, tech_timeframe='1mo', news_days=7, pred_horizon='1d', force_refresh=False):
        """Generate an enhanced prediction combining technical and sentiment analysis."""
        # Try to load from cache first if not forcing refresh
        pred_key = f"{symbol}_{tech_timeframe}_{news_days}_{pred_horizon}"
        cache_file = os.path.join(PREDICTIONS_DIR, f"{pred_key}.json")
        
        if not force_refresh and self._is_valid_cache(cache_file):
            return self._load_from_cache(cache_file)
            
        try:
            # Get fresh data from both sources
            tech_data = get_aiplus_technical_data(symbol, tech_timeframe, force_refresh)
            sent_data = get_aiplus_sentiment(symbol, news_days, force_refresh)
            
            # Validate both data sources
            for data, source in [(tech_data, "Technical"), (sent_data, "Sentiment")]:
                if not data or 'error' in data:
                    error_msg = data.get('error', f"Failed to retrieve {source} data")
                    return {'error': f"{source} data error: {error_msg}"}
                    
            # Process data and generate prediction
            result = self._process_prediction(symbol, tech_data, sent_data, tech_timeframe, pred_horizon)
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(result, f)
                
            return result
            
        except Exception as e:
            return {"error": f"Error generating prediction for {symbol}: {str(e)}"}
    
    def _is_valid_cache(self, cache_file):
        """Check if cache file exists and is recent (within 12 hours)."""
        if not os.path.exists(cache_file):
            return False
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            if 'timestamp' in cached_data:
                pred_time = datetime.fromisoformat(cached_data['timestamp'])
                if (datetime.now() - pred_time).total_seconds() < 43200:  # 12 hours
                    return True
        except:
            pass
            
        return False
        
    def _load_from_cache(self, cache_file):
        """Load prediction from cache file."""
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached prediction: {e}")
            return None
    
    def _process_prediction(self, symbol, tech_data, sent_data, tech_timeframe, pred_horizon):
        """Process technical and sentiment data into a prediction."""
        # Extract key metrics
        current_price = tech_data.get('current_price', 0)
        volatility = tech_data.get('volatility', 20)
        tsmn = tech_data.get('tsmn', {})
        tsmn_value = tsmn.get('value', 0) if isinstance(tsmn, dict) else 0
        sentiment_score = sent_data.get('sentiment_score', 0)
        
        # Data validation - handle missing or invalid values
        if current_price <= 0:
            current_price = 1  # Default to avoid division by zero
        if not isinstance(volatility, (int, float)) or volatility <= 0:
            volatility = 20  # Default to reasonable volatility
        if not isinstance(tsmn_value, (int, float)):
            tsmn_value = 0
        if not isinstance(sentiment_score, (int, float)):
            sentiment_score = 0
        
        # Calculate signal weights - technical vs sentiment
        tech_weight = min(max(0.6, volatility / 50), 0.8)  # 0.6-0.8 range based on volatility
        sent_weight = 1 - tech_weight
        
        # Calculate combined signal (-100 to +100 scale)
        combined_signal = (tsmn_value * tech_weight) + (sentiment_score * sent_weight)
        
        # Set max percentage move based on timeframe and volatility
        timeframe_factors = {'1d': 30, '2d': 25, '1w': 15, '1mo': 10}
        max_move = volatility / timeframe_factors.get(pred_horizon, 30)
        
        # Calculate predicted change
        predicted_change_pct = combined_signal * max_move / 100
        
        # Determine confidence level
        tech_direction = 1 if tsmn_value > 20 else -1 if tsmn_value < -20 else 0
        sent_direction = 1 if sentiment_score > 30 else -1 if sentiment_score < -30 else 0
        
        confidence = self._calculate_confidence(tech_direction, sent_direction, combined_signal)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(predicted_change_pct, confidence, pred_horizon)
        
        # Calculate target prices
        target_mid = current_price * (1 + predicted_change_pct / 100)
        target_range = volatility / 30  # Range as percentage
        target_low = current_price * (1 + (predicted_change_pct - target_range) / 100)
        target_high = current_price * (1 + (predicted_change_pct + target_range) / 100)
        
        # Get advanced analysis if API is available
        enhanced_analysis = ""
        if self.api_initialized:
            enhanced_analysis = self._generate_enhanced_analysis(
                symbol, tech_data, sent_data, predicted_change_pct, 
                confidence, pred_horizon
            )
        
        # Determine signal based on predicted change
        signal = "bullish" if predicted_change_pct > 0.1 else "bearish" if predicted_change_pct < -0.1 else "neutral"
        
        # Assemble the prediction result
        return {
            "symbol": symbol,
            "company_name": tech_data.get('company_name', sent_data.get('company_name', symbol)),
            "current_price": current_price,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "signal": signal,
                "predicted_change_pct": predicted_change_pct,
                "confidence": confidence,
                "target_price_low": target_low,
                "target_price_mid": target_mid,
                "target_price_high": target_high,
                "data_timeframe": tech_timeframe,
                "prediction_horizon": pred_horizon,
                "prediction_horizon_text": HORIZON_TEXT.get(pred_horizon, 'specified period'),
                "recommendation": recommendation
            },
            "technical_contribution": {
                "weight": tech_weight,
                "signal": "bullish" if tsmn_value > 0 else "bearish" if tsmn_value < 0 else "neutral",
                "tsmn_value": tsmn_value,
                "key_indicators": self._extract_key_indicators(tech_data.get('standard_indicators', {}))
            },
            "sentiment_contribution": {
                "weight": sent_weight,
                "sentiment": sent_data.get('sentiment', 'neutral'),
                "sentiment_score": sentiment_score,
                "key_developments": sent_data.get('key_developments', []),
                "sentiment_drivers": sent_data.get('sentiment_drivers', [])
            },
            "analysis": {
                "technical_summary": tech_data.get('technical_summary', ''),
                "sentiment_summary": sent_data.get('sentiment_summary', ''),
                "enhanced_analysis": enhanced_analysis
            }
        }
    
    def _calculate_confidence(self, tech_direction, sent_direction, combined_signal):
        """Calculate confidence level based on signal agreement and strength."""
        # Both signals agree and aren't neutral
        if tech_direction == sent_direction and tech_direction != 0:
            return "high" if abs(combined_signal) > 40 else "medium"
        
        # Both signals are non-neutral but disagree
        if tech_direction != 0 and sent_direction != 0 and tech_direction != sent_direction:
            return "low"
        
        # One signal is strong, the other is neutral
        if tech_direction != 0 or sent_direction != 0:
            return "medium" if abs(combined_signal) > 30 else "low"
        
        # Both signals are neutral
        return "low"
    
    def _generate_recommendation(self, pred_change, confidence, horizon):
        """Generate investment recommendation based on prediction parameters."""
        # Set thresholds based on timeframe
        thresholds = {
            # Format: (strong_pos, moderate_pos, small_pos, moderate_neg, strong_neg)
            '1d': (1.0, 0.5, 0.2, -0.5, -1.0),
            '2d': (1.5, 0.8, 0.3, -0.8, -1.5),
            '1w': (2.5, 1.2, 0.5, -1.2, -2.5),
            '1mo': (5.0, 2.0, 0.8, -2.0, -5.0)
        }
        
        t = thresholds.get(horizon, thresholds['1d'])
        
        # Determine recommendation based on thresholds and confidence
        if pred_change > t[0]:  # Strong positive
            if confidence == "high": return "Strong Buy"
            if confidence == "medium": return "Buy"
            return "Speculative Buy"
            
        elif pred_change > t[1]:  # Moderate positive
            if confidence == "high": return "Buy"
            if confidence == "medium": return "Accumulate"
            return "Speculative Buy"
            
        elif pred_change > t[2]:  # Small positive
            if confidence == "high": return "Accumulate"
            return "Hold with Positive Bias"
            
        elif pred_change > t[3]:  # Neutral to moderate negative
            if confidence == "high" and pred_change < 0: 
                return "Hold with Negative Bias"
            return "Hold"
            
        elif pred_change > t[4]:  # Moderate negative
            if confidence == "high": return "Reduce"
            if confidence == "medium": return "Hold with Negative Bias"
            return "Hold"
            
        else:  # Strong negative
            if confidence == "high": return "Sell"
            if confidence == "medium": return "Reduce"
            return "Hold with Negative Bias"
    
    def _extract_key_indicators(self, indicators):
        """Extract most important technical indicators for display."""
        key_metrics = {}
        
        # Priority indicators to display
        priority_indicators = ['RSI_14', 'MACD', 'MACD_Signal', 'BB_Percent', 'ADX']
        
        # Add available priority indicators
        for indicator in priority_indicators:
            if indicator in indicators:
                key_metrics[indicator] = indicators[indicator]
        
        # Add up to 3 moving averages if available
        for ma in ['MA_20', 'MA_50', 'MA_200']:
            if ma in indicators:
                key_metrics[ma] = indicators[ma]
        
        return key_metrics
    
    def _generate_enhanced_analysis(self, symbol, tech_data, sent_data, pred_change, confidence, horizon):
        """Generate detailed analysis using LLM if available."""
        try:
            # Get key data for prompt
            company_name = tech_data.get('company_name', sent_data.get('company_name', symbol))
            tech_summary = tech_data.get('technical_summary', '')
            sent_summary = sent_data.get('sentiment_summary', '')
            
            # Format indicators for prompt
            indicators = tech_data.get('standard_indicators', {})
            tsmn = tech_data.get('tsmn', {}).get('value', 'N/A') 
            volatility = tech_data.get('volatility', 'N/A')
            
            # Format key developments
            developments = sent_data.get('key_developments', [])
            news_text = ""
            if developments:
                news_items = [f"- {dev.get('date', '')}: {dev.get('headline', '')}" 
                             for dev in developments[:3]]
                news_text = "\n".join(news_items)
            
            # Get timeframe text
            horizon_text = HORIZON_TEXT.get(horizon, horizon)
            
            # Create the prompt
            prompt = f"""
            Please provide a concise analysis for {company_name} ({symbol}) with focus on the {horizon_text}:
            
            TECHNICAL SUMMARY: {tech_summary}
            
            SENTIMENT SUMMARY: {sent_summary}
            
            KEY INDICATORS:
            - TSMN Value: {tsmn}
            - Volatility: {volatility}%
            - RSI (14): {indicators.get('RSI_14', 'N/A')}
            - MACD: {indicators.get('MACD', 'N/A')}
            - Bollinger %B: {indicators.get('BB_Percent', 'N/A')}
            
            RECENT NEWS:
            {news_text}
            
            PREDICTION: {pred_change:.2f}% change for {horizon_text} with {confidence} confidence
            
            Provide 2-3 paragraphs analyzing {company_name}'s outlook specifically for the {horizon_text},
            focusing on synthesizing the data above. Include insights about potential catalysts and risks.
            Be specific about this stock and industry. Conclude with investment perspective.
            """
            
            # Generate the analysis
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating enhanced analysis: {e}")
            return "Enhanced analysis unavailable."
    
    def track_prediction(self, symbol, prediction_data, actual_price, pred_horizon='1d'):
        """Track prediction performance against actual results."""
        try:
            # Extract key data
            timestamp = prediction_data.get('timestamp')
            pred_price = prediction_data.get('current_price', 0)
            pred_change = prediction_data.get('prediction', {}).get('predicted_change_pct', 0)
            confidence = prediction_data.get('prediction', {}).get('confidence', 'low')
            
            # Calculate actual change
            if pred_price <= 0:  # Avoid division by zero
                return {"error": "Invalid current price in prediction data"}
                
            actual_change = ((actual_price - pred_price) / pred_price) * 100
            
            # Create performance record
            record = {
                "symbol": symbol,
                "prediction_date": datetime.fromisoformat(timestamp).strftime('%Y-%m-%d') if timestamp else datetime.now().strftime('%Y-%m-%d'),
                "evaluation_date": datetime.now().strftime('%Y-%m-%d'),
                "prediction_horizon": pred_horizon,
                "predicted_change_pct": pred_change,
                "actual_change_pct": actual_change,
                "correct_direction": (pred_change > 0 and actual_change > 0) or (pred_change < 0 and actual_change < 0),
                "error_pct": abs(pred_change - actual_change),
                "confidence": confidence
            }
            
            # Update history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []
                
            self.performance_history[symbol].append(record)
            
            # Save to disk
            self._save_performance_history()
            
            return record
            
        except Exception as e:
            return {"error": f"Error tracking prediction: {str(e)}"}
    
    def get_performance_metrics(self, symbol=None):
        """Get performance metrics for predictions."""
        try:
            # Get relevant history
            if symbol:
                history = self.performance_history.get(symbol, [])
                symbols = [symbol] if history else []
            else:
                symbols = list(self.performance_history.keys())
                history = []
                for sym in symbols:
                    history.extend(self.performance_history[sym])
            
            if not history:
                return {"error": "No performance history found"}
            
            # Calculate overall metrics
            total = len(history)
            correct = sum(1 for item in history if item.get('correct_direction', False))
            accuracy = (correct / total * 100) if total > 0 else 0
            avg_error = sum(item.get('error_pct', 0) for item in history) / total if total > 0 else 0
            
            # Build detailed metrics by confidence, symbol, and horizon
            confidence_metrics = self._calculate_group_metrics(history, 'confidence', ['high', 'medium', 'low'])
            horizon_metrics = self._calculate_group_metrics(history, 'prediction_horizon', ['1d', '2d', '1w', '1mo'])
            
            # Symbol metrics only if multiple symbols
            symbol_metrics = {}
            if len(symbols) > 1:
                symbol_metrics = self._calculate_group_metrics(
                    history, 'symbol', symbols
                )
            
            return {
                "overall": {
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "accuracy": accuracy,
                    "average_error": avg_error
                },
                "by_confidence": confidence_metrics,
                "by_symbol": symbol_metrics,
                "by_horizon": horizon_metrics,
                "recent_predictions": sorted(history, key=lambda x: x.get('prediction_date', ''), reverse=True)[:10]
            }
            
        except Exception as e:
            return {"error": f"Error calculating performance metrics: {str(e)}"}
    
    def _calculate_group_metrics(self, history, group_key, group_values):
        """Calculate metrics for a specific grouping (confidence, horizon, etc.)."""
        group_metrics = {}
        
        for value in group_values:
            group_items = [item for item in history if item.get(group_key) == value]
            total = len(group_items)
            
            if total > 0:
                correct = sum(1 for item in group_items if item.get('correct_direction', False))
                group_metrics[value] = {
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "accuracy": (correct / total * 100)
                }
                
        return group_metrics
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        try:
            file_path = os.path.join(PREDICTIONS_DIR, "performance_history.json")
            with open(file_path, 'w') as f:
                json.dump(self.performance_history, f)
        except Exception as e:
            print(f"Error saving performance history: {e}")
    
    def _load_performance_history(self):
        """Load performance history from disk."""
        try:
            file_path = os.path.join(PREDICTIONS_DIR, "performance_history.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading performance history: {e}")
        return {}


# Exported functions for module use
def generate_aiplus_prediction(symbol, technical_timeframe='1mo', news_days=7, prediction_horizon='1d', force_refresh=False):
    """Generate an enhanced stock prediction using AI+."""
    predictor = AIplusPredictor()
    return predictor.generate_prediction(symbol, technical_timeframe, news_days, prediction_horizon, force_refresh)

def get_aiplus_performance(symbol=None):
    """Get performance metrics for AI+ predictions."""
    predictor = AIplusPredictor()
    return predictor.get_performance_metrics(symbol)


# Test code
if __name__ == "__main__":
    # Test with a sample stock
    symbol = "AAPL"
    prediction = generate_aiplus_prediction(symbol, prediction_horizon='1d', force_refresh=True)
    print(json.dumps(prediction, indent=2))