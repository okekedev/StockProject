"""
Performance metrics callbacks for the AI+ tab.
"""
from dash import html, callback, Output, Input
from modules.aiplus_predictor import get_aiplus_performance
from callbacks.aiplus.utils import get_horizon_display


# Performance metrics callback
@callback(
    Output('performance-metrics', 'children'),
    Input('tabs', 'value')
)
def update_performance_metrics(tab):
    """
    Update performance metrics display.
    
    Args:
        tab (str): Current tab
        
    Returns:
        html component: Performance metrics
    """
    if tab != 'aiplus':
        return ""
    
    try:
        # Get performance metrics
        metrics = get_aiplus_performance()
        
        if 'error' in metrics:
            return html.Div([
                html.P("No performance data available.", className="bank-warning"),
                html.P("AI+ predictions must be evaluated against actual market performance to generate metrics.", className="bank-text")
            ], className="bank-notification bank-warning-notification")
        
        # Extract overall metrics
        overall = metrics.get('overall', {})
        total_predictions = overall.get('total_predictions', 0)
        correct_predictions = overall.get('correct_predictions', 0)
        accuracy = overall.get('accuracy', 0)
        avg_error = overall.get('average_error', 0)
        
        # Format the performance report
        if total_predictions > 0:
            return create_performance_report(metrics, total_predictions, correct_predictions, accuracy, avg_error)
        else:
            return html.Div([
                html.P("No performance data available yet.", className="bank-warning"),
                html.P("AI+ prediction performance will be shown here once predictions have been evaluated against actual market performance.", className="bank-text")
            ], className="bank-notification bank-warning-notification")
            
    except Exception as e:
        print(f"Error updating performance metrics: {e}")
        return html.Div([
            html.P("Error loading performance metrics.", className="bank-error"),
            html.P(str(e), className="bank-text")
        ], className="bank-notification bank-error-notification")


def create_performance_report(metrics, total_predictions, correct_predictions, accuracy, avg_error):
    """
    Create the performance metrics report HTML structure.
    
    Args:
        metrics (dict): Performance metrics data
        total_predictions (int): Total number of predictions
        correct_predictions (int): Correct predictions count
        accuracy (float): Overall accuracy percentage
        avg_error (float): Average error percentage
        
    Returns:
        html component: Formatted performance report
    """
    return html.Div([
        html.Div([
            html.H5("PERFORMANCE METRICS", className="bank-section-heading"),
            html.Div([
                html.Div([
                    html.P("Overall Accuracy:", className="bank-label"),
                    html.P(
                        f"{accuracy:.1f}%", 
                        className=f"bank-value large-value {'high-accuracy' if accuracy >= 70 else 'medium-accuracy' if accuracy >= 55 else 'low-accuracy'}"
                    )
                ], className="metric-box"),
                html.Div([
                    html.P("Predictions:", className="bank-label"),
                    html.P(
                        f"{correct_predictions} / {total_predictions}",
                        className="bank-value"
                    )
                ], className="metric-box"),
                html.Div([
                    html.P("Avg. Error:", className="bank-label"),
                    html.P(
                        f"{avg_error:.2f}%",
                        className="bank-value"
                    )
                ], className="metric-box")
            ], className="metrics-row"),
            
            # By confidence level
            html.Div([
                html.H6("Performance by Confidence Level", className="bank-subsection-heading"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Confidence"),
                            html.Th("Accuracy"),
                            html.Th("Predictions")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(conf.upper()),
                            html.Td(
                                f"{data.get('accuracy', 0):.1f}%",
                                className=f"{'high-accuracy' if data.get('accuracy', 0) >= 70 else 'medium-accuracy' if data.get('accuracy', 0) >= 55 else 'low-accuracy'}"
                            ),
                            html.Td(f"{data.get('correct_predictions', 0)} / {data.get('total_predictions', 0)}")
                        ]) for conf, data in metrics.get('by_confidence', {}).items()
                    ])
                ], className="bank-table")
            ]),
            
            # By prediction horizon
            html.Div([
                html.H6("Performance by Prediction Horizon", className="bank-subsection-heading"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Horizon"),
                            html.Th("Accuracy"),
                            html.Th("Predictions")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(get_horizon_display(horizon)),
                            html.Td(
                                f"{data.get('accuracy', 0):.1f}%",
                                className=f"{'high-accuracy' if data.get('accuracy', 0) >= 70 else 'medium-accuracy' if data.get('accuracy', 0) >= 55 else 'low-accuracy'}"
                            ),
                            html.Td(f"{data.get('correct_predictions', 0)} / {data.get('total_predictions', 0)}")
                        ]) for horizon, data in metrics.get('by_horizon', {}).items()
                    ])
                ], className="bank-table")
            ]),
            
            # Recent predictions
            html.Div([
                html.H6("Recent Predictions", className="bank-subsection-heading"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Symbol"),
                            html.Th("Date"),
                            html.Th("Horizon"),
                            html.Th("Prediction"),
                            html.Th("Actual"),
                            html.Th("Result")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(pred.get('symbol', '')),
                            html.Td(pred.get('prediction_date', '')),
                            html.Td(get_horizon_display(pred.get('prediction_horizon', '1d'))),
                            html.Td(
                                f"{pred.get('predicted_change_pct', 0):.2f}%",
                                className=f"{'positive' if pred.get('predicted_change_pct', 0) > 0 else 'negative' if pred.get('predicted_change_pct', 0) < 0 else 'neutral'}"
                            ),
                            html.Td(
                                f"{pred.get('actual_change_pct', 0):.2f}%",
                                className=f"{'positive' if pred.get('actual_change_pct', 0) > 0 else 'negative' if pred.get('actual_change_pct', 0) < 0 else 'neutral'}"
                            ),
                            html.Td(
                                "✓" if pred.get('correct_direction', False) else "✗",
                                className=f"{'success' if pred.get('correct_direction', False) else 'failure'}"
                            )
                        ]) for pred in metrics.get('recent_predictions', [])[:10]
                    ])
                ], className="bank-table")
            ])
        ], className="bank-performance-metrics bank-card")
    ], className="bank-metrics-container slide-in")