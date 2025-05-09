"""
Modified metrics_callbacks.py to hide the performance metrics section
"""
from dash import html, callback, Output, Input

# Performance metrics callback - return empty content to hide the section
@callback(
    Output('performance-metrics', 'children'),
    Input('tabs', 'value')
)
def update_performance_metrics(tab):
    """
    Empty implementation to hide the performance metrics section.
    
    Args:
        tab (str): Current tab
        
    Returns:
        html component: Empty div
    """
    # Return empty content regardless of tab value to hide this section
    return html.Div(style={"display": "none"})
