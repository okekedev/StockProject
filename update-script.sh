#!/bin/bash
# Script to update the AI+ module with improved styling

# Create a backup directory
mkdir -p ./backups
BACKUP_DATE=$(date +"%Y%m%d%H%M%S")
BACKUP_DIR="./backups/aiplus_backup_$BACKUP_DATE"
mkdir -p $BACKUP_DIR

echo "Creating backups of original files..."
# Backup the original files
cp ./assets/css/aiplus_styles.css $BACKUP_DIR/ 2>/dev/null || echo "No aiplus_styles.css file found"
cp ./callbacks/aiplus/analysis_callbacks.py $BACKUP_DIR/ 2>/dev/null || echo "No analysis_callbacks.py file found"
cp ./callbacks/aiplus/metrics_callbacks.py $BACKUP_DIR/ 2>/dev/null || echo "No metrics_callbacks.py file found"

# Create or append styles to CSS file
echo "Updating CSS styles..."
CSS_DIR="./assets/css"
AIPLUS_CSS="$CSS_DIR/aiplus_styles.css"

# If the directory doesn't exist, create it
mkdir -p $CSS_DIR

# Create or append to the aiplus_styles.css
cat << 'EOF' >> $AIPLUS_CSS

/**
 * Updated AI+ Styles
 * This CSS will improve the recommendation text color and remove the performance metrics section styling
 */

/* Change the recommendation text color for better contrast on dark background */
.prediction-recommendation {
    text-align: center;
    padding: 1rem;
    background-color: var(--bank-green-light);
    color: white; /* Changed from the default dark color to white for better contrast */
    font-weight: 700;
    font-size: 1.3rem;
    border-radius: 4px;
    box-shadow: 0 2px 4px var(--bank-shadow);
}

/* Hide the performance metrics section */
.bank-metrics-container {
    display: none !important; /* Hide the entire metrics container */
}

/* Additional styling for recommendations */
.bank-recommendation.positive,
.bank-recommendation.negative,
.bank-recommendation.neutral {
    color: white !important; /* Force white color for all recommendations */
}
EOF

echo "CSS styles updated."

# Update the analysis_callbacks.py file if it exists
ANALYSIS_FILE="./callbacks/aiplus/analysis_callbacks.py"
if [ -f "$ANALYSIS_FILE" ]; then
    echo "Updating analysis_callbacks.py..."
    # Add style property to the recommendation text
    sed -i 's/className="bank-recommendation"/className="bank-recommendation" style={"color": "white"}/g' $ANALYSIS_FILE
    echo "analysis_callbacks.py updated."
else
    echo "analysis_callbacks.py not found. Skipping."
fi

# Update the metrics_callbacks.py file if it exists
METRICS_FILE="./callbacks/aiplus/metrics_callbacks.py"
if [ -f "$METRICS_FILE" ]; then
    echo "Updating metrics_callbacks.py..."
    # Replace the function implementation to return an empty div
    cat << 'EOF' > $METRICS_FILE
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
EOF
    echo "metrics_callbacks.py updated."
else
    echo "metrics_callbacks.py not found. Skipping."
fi

echo "Update completed. Backups saved to $BACKUP_DIR"