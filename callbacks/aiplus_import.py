"""
Central import file for AI+ callbacks.

This file ensures all AI+ callbacks are imported and registered
when the application starts up. Import this file in main.py.
"""

# Import callback modules to ensure all are registered
from callbacks.aiplus.data_callbacks import fetch_technical_data, fetch_news_data, populate_aiplus_dropdown
from callbacks.aiplus.status_callbacks import update_readiness_status, update_data_readiness
from callbacks.aiplus.analysis_callbacks import generate_ai_analysis
from callbacks.aiplus.metrics_callbacks import update_performance_metrics