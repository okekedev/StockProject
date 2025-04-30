"""
Data loading functions for the Stock Analysis application.
"""
import pandas as pd
from datetime import datetime

def load_data(file_path, default_columns=None):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        default_columns (list, optional): Default columns if file is empty or not found.
        
    Returns:
        pandas.DataFrame: The loaded data or an empty DataFrame with default columns.
    """
    if not file_path:
        return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()
    
    try:
        if pd.io.common.file_exists(file_path):
            df = pd.read_csv(file_path)
            # Ensure Date column is Timestamp for technical data if it exists
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
        else:
            print(f"Warning: File not found - {file_path}")
            return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file - {file_path}")
        return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame(columns=default_columns) if default_columns else pd.DataFrame()

def get_criteria_from_data(df):
    """
    Extract criteria options from the DataFrame for use in filters.
    
    Args:
        df (pandas.DataFrame): The data to extract criteria from.
        
    Returns:
        dict: Dictionary of criteria options.
    """
    criteria = {}
    
    # Add IPO Year criteria if column exists
    if 'IPO Year' in df.columns:
        criteria['IPO Year'] = sorted(df['IPO Year'].dropna().astype(int).unique().tolist())
    else:
        criteria['IPO Year'] = []
    
    # Add Sector criteria if column exists
    if 'Sector' in df.columns:
        criteria['Sector'] = sorted(df['Sector'].dropna().unique().tolist())
    else:
        criteria['Sector'] = []
    
    # Add Industry criteria if column exists
    if 'Industry' in df.columns:
        criteria['Industry'] = sorted(df['Industry'].dropna().unique().tolist())
    else:
        criteria['Industry'] = []
    
    return criteria