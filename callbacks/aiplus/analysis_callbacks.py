"""
Analysis generation callbacks for the AI+ tab.
"""
from dash import html, callback, Output, Input, State
import pandas as pd
import os

# Import modules
from modules.aiplus_predictor import generate_aiplus_prediction
from callbacks.aiplus.utils import AIPLUS_CACHE_DIR, get_indicator_interpretation, load_from_cache


# AI Analysis button callback
@callback(
    Output('analysis-output', 'children'),
    Input('begin-analysis-button', 'n_clicks'),
    State('aiplus-stock-dropdown', 'value'),
    State('tech-timeframe-dropdown', 'value'),
    State('news-timeframe-dropdown', 'value'),
    State('prediction-horizon-dropdown', 'value'),
    State('tabs', 'value'),
    prevent_initial_call=True
)
def generate_ai_analysis(n_clicks, symbol, tech_timeframe, news_days, prediction_horizon, tab):
    """
    Generate AI analysis based on collected data.
    
    Args:
        n_clicks (int): Number of button clicks
        symbol (str): Selected stock symbol
        tech_timeframe (str): Selected technical timeframe
        news_days (int): Number of days for news analysis
        prediction_horizon (str): Timeframe for prediction
        tab (str): Current tab
        
    Returns:
        html component: Analysis output
    """
    if n_clicks == 0 or tab != 'aiplus' or not symbol:
        return ""
    
    try:
        # Check if data is ready
        readiness = load_from_cache("readiness.json")
        
        if not readiness or not readiness.get('ready', False) or readiness.get('symbol') != symbol:
            return html.Div([
                html.P("Data not ready for analysis.", className="bank-error"),
                html.P(f"Please ensure both technical and news data are fetched for {symbol} before generating analysis.", className="bank-text")
            ], className="bank-notification bank-error-notification")
        
        # Generate AI analysis
        prediction = generate_aiplus_prediction(
            symbol, tech_timeframe, news_days, prediction_horizon, force_refresh=True
        )
        
        if 'error' in prediction:
            return html.Div([
                html.P("Error generating analysis.", className="bank-error"),
                html.P(prediction['error'], className="bank-text")
            ], className="bank-notification bank-error-notification")
        
        # Create and return the analysis report
        return create_analysis_report(prediction)
        
    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        return html.Div([
            html.P("Error generating AI analysis.", className="bank-error"),
            html.P(str(e), className="bank-text")
        ], className="bank-notification bank-error-notification")


def create_analysis_report(prediction):
    """
    Create the analysis report HTML structure.
    
    Args:
        prediction (dict): Prediction data
        
    Returns:
        html component: Formatted analysis report
    """
    # Extract data for display
    symbol = prediction.get('symbol', '')
    company_name = prediction.get('company_name', symbol)
    current_price = prediction.get('current_price', 0)
    
    pred_data = prediction.get('prediction', {})
    signal = pred_data.get('signal', 'neutral')
    pred_change = pred_data.get('predicted_change_pct', 0)
    confidence = pred_data.get('confidence', 'low')
    target_low = pred_data.get('target_price_low', 0)
    target_mid = pred_data.get('target_price_mid', 0)
    target_high = pred_data.get('target_price_high', 0)
    recommendation = pred_data.get('recommendation', 'Hold')
    prediction_horizon_text = pred_data.get('prediction_horizon_text', 'specified period')
    
    tech_contrib = prediction.get('technical_contribution', {})
    tech_weight = tech_contrib.get('weight', 0)
    tech_signal = tech_contrib.get('signal', 'neutral')
    tsmn_value = tech_contrib.get('tsmn_value', 0)
    
    sent_contrib = prediction.get('sentiment_contribution', {})
    sent_weight = sent_contrib.get('weight', 0)
    sentiment = sent_contrib.get('sentiment', 'neutral')
    sentiment_score = sent_contrib.get('sentiment_score', 0)
    
    analysis = prediction.get('analysis', {})
    tech_summary = analysis.get('technical_summary', '')
    sent_summary = analysis.get('sentiment_summary', '')
    enhanced_analysis = analysis.get('enhanced_analysis', '')
    
    # Format the prediction report
    return html.Div([
        # Header with company and current price
        html.Div([
            html.Div(className="bank-emblem", style={"float": "right", "width": "40px", "height": "40px"}),
            html.H4(f"AI+ ANALYSIS: {company_name} ({symbol})", className="bank-report-title"),
            html.P(f"Analysis Date: {pd.Timestamp.now().strftime('%B %d, %Y')}", className="bank-report-date"),
            html.Div([
                html.Span("Current Price: ", className="bank-label"),
                html.Span(f"${current_price:.2f}", className="bank-value")
            ]),
            html.Div([
                html.Span("CONFIDENTIAL", className="confidential-stamp")
            ], className="stamp-container"),
        ], className="bank-report-header"),
        
        # Prediction summary
        html.Div([
            html.H5(f"PREDICTION SUMMARY FOR {prediction_horizon_text.upper()}", className="bank-section-heading"),
            html.Div([
                html.Div([
                    html.P("Signal:", className="bank-label"),
                    html.P(
                        signal.upper(), 
                        className=f"bank-value large-value {'positive' if signal == 'bullish' else 'negative' if signal == 'bearish' else 'neutral'}"
                    )
                ], className="prediction-signal"),
                html.Div([
                    html.P("Predicted Change:", className="bank-label"),
                    html.P(
                        f"{pred_change:.2f}%", 
                        className=f"bank-value large-value {'positive' if pred_change > 0 else 'negative' if pred_change < 0 else 'neutral'}"
                    )
                ], className="prediction-change"),
                html.Div([
                    html.P("Confidence:", className="bank-label"),
                    html.P(
                        confidence.upper(),
                        className=f"bank-value {'high-confidence' if confidence == 'high' else 'medium-confidence' if confidence == 'medium' else 'low-confidence'}"
                    )
                ], className="prediction-confidence"),
            ], className="prediction-summary-row"),
            html.Div([
                html.P("Target Price Range:", className="bank-label"),
                html.Div([
                    html.Span(f"${target_low:.2f}", className="target-low"),
                    html.Span(" → ", className="target-arrow"),
                    html.Span(f"${target_mid:.2f}", className="target-mid"),
                    html.Span(" → ", className="target-arrow"),
                    html.Span(f"${target_high:.2f}", className="target-high")
                ], className="target-range")
            ], className="prediction-targets"),
            html.Div([
                html.P("Recommendation:", className="bank-label"),
                html.P(
                    recommendation,
                    className=f"bank-recommendation {'positive' if 'buy' in recommendation.lower() else 'negative' if 'sell' in recommendation.lower() else 'neutral'}"
                )
            ], className="prediction-recommendation")
        ], className="bank-prediction-summary bank-card"),
        
        # Analysis details
        html.Div([
            html.H5("ANALYSIS DETAILS", className="bank-section-heading"),
            
            # Technical and sentiment contribution visualization
            html.Div([
                html.H6("Signal Contributions", className="bank-subsection-heading"),
                html.Div([
                    html.Div([
                        html.P(f"Technical ({tech_weight*100:.0f}%)", className="bank-label"),
                        html.Div([
                            html.Div(
                                className="contribution-bar technical",
                                style={"width": f"{abs(tsmn_value)/2:.1f}%", "marginLeft": "50%" if tsmn_value > 0 else "auto", "marginRight": "50%" if tsmn_value < 0 else "auto"}
                            )
                        ], className="contribution-container"),
                        html.P(
                            f"{tech_signal.upper()} ({tsmn_value:.1f})", 
                            className=f"{'positive' if tsmn_value > 0 else 'negative' if tsmn_value < 0 else 'neutral'}"
                        )
                    ], className="contribution-row"),
                    html.Div([
                        html.P(f"Sentiment ({sent_weight*100:.0f}%)", className="bank-label"),
                        html.Div([
                            html.Div(
                                className="contribution-bar sentiment",
                                style={"width": f"{abs(sentiment_score)/2:.1f}%", "marginLeft": "50%" if sentiment_score > 0 else "auto", "marginRight": "50%" if sentiment_score < 0 else "auto"}
                            )
                        ], className="contribution-container"),
                        html.P(
                            f"{sentiment.upper()} ({sentiment_score:.1f})", 
                            className=f"{'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'}"
                        )
                    ], className="contribution-row")
                ], className="signal-contributions")
            ]),
            
            # Enhanced analysis
            html.Div([
                html.H6(f"AI+ Enhanced Analysis for {prediction_horizon_text}", className="bank-subsection-heading"),
                html.P(enhanced_analysis, className="bank-text enhanced-analysis"),
            ], className="enhanced-analysis-section"),
            
            # Technical analysis summary
            html.Div([
                html.H6("Technical Analysis", className="bank-subsection-heading"),
                html.P(tech_summary, className="bank-text tech-summary"),
            ], className="technical-summary-section"),
            
            # Sentiment analysis summary
            html.Div([
                html.H6("Sentiment Analysis", className="bank-subsection-heading"),
                html.P(sent_summary, className="bank-text sent-summary"),
            ], className="sentiment-summary-section"),
            
            # Key developments
            html.Div([
                html.H6("Key Developments", className="bank-subsection-heading"),
                html.Ul([
                    html.Li([
                        html.Span(f"{dev.get('date', '')}: ", className="bank-date"),
                        html.Span(
                            dev.get('headline', ''), 
                            className=f"bank-news-headline {dev.get('sentiment', 'neutral')}"
                        )
                    ]) for dev in sent_contrib.get('key_developments', [])[:5]
                ], className="bank-news-list") if sent_contrib.get('key_developments', []) else html.P("No significant developments found.", className="bank-text")
            ], className="key-developments-section"),
            
            # Key indicators
            html.Div([
                html.H6("Key Technical Indicators", className="bank-subsection-heading"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Indicator"),
                            html.Th("Value"),
                            html.Th("Interpretation")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td("TSMN"),
                            html.Td(f"{tsmn_value:.1f}"),
                            html.Td(
                                "Bullish" if tsmn_value > 20 else "Bearish" if tsmn_value < -20 else "Neutral",
                                className=f"{'positive' if tsmn_value > 20 else 'negative' if tsmn_value < -20 else 'neutral'}"
                            )
                        ]),
                        *[
                            html.Tr([
                                html.Td(indicator),
                                html.Td(f"{value:.1f}" if isinstance(value, (int, float)) else str(value)),
                                html.Td(get_indicator_interpretation(indicator, value))
                            ]) for indicator, value in tech_contrib.get('key_indicators', {}).items()
                        ]
                    ])
                ], className="bank-table")
            ], className="key-indicators-section")
        ], className="bank-analysis-details bank-card"),
        
        # Footer
        html.Div([
            html.P(
                "DISCLAIMER: This AI+ analysis is provided for informational purposes only. "
                "All investment decisions should be made with appropriate consideration of risk "
                "tolerance and financial objectives. Past performance is not indicative of future results.",
                className="bank-disclaimer"
            ),
            html.P(
                f"Generated by AI+ Advanced Prediction Department on {pd.Timestamp.now().strftime('%B %d, %Y at %H:%M')}",
                className="bank-generation-info"
            )
        ], className="bank-report-footer")
    ], className="bank-ai-report slide-in")