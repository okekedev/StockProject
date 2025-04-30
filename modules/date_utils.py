"""
Date utility functions for the Stock Analysis application.
"""
import pandas as pd
from datetime import datetime, timedelta

def get_date_options(start_date, end_date, default_date=None):
    """
    Generate date options for Test Model tab, excluding weekends and Mondays.
    
    Args:
        start_date (datetime or str): The start date.
        end_date (datetime or str): The end date.
        default_date (datetime or str, optional): The default selected date.
        
    Returns:
        tuple: (options, default_value) for dropdown.
    """
    # Convert to datetime objects if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Convert to date objects if they are datetime
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    
    # Ensure start_date is not after end_date
    if start_date > end_date:
        return [], None
    
    # Limit end_date to 3 days before tomorrow
    today = datetime.now().date()
    max_date = today - timedelta(days=3)  # 3 days before tomorrow
    end_date = min(end_date, max_date)
    
    # Generate business days (excludes weekends) and exclude Mondays (weekday 0)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    valid_dates = [date for date in dates if date.weekday() != 0]  # Exclude Mondays
    
    options = [
        {'label': date.strftime('%Y-%m-%d'), 'value': date.strftime('%Y-%m-%d'), 'disabled': False}
        for date in valid_dates
    ]
    
    if not options:  # If no valid dates, return empty options
        return [], None
    
    if default_date:
        if isinstance(default_date, datetime):
            default_date = default_date.date()
        default_value = default_date.strftime('%Y-%m-%d')
    else:
        # Default to most recent valid day
        default_value = valid_dates[-1].strftime('%Y-%m-%d') if valid_dates else None
    
    return options, default_value

def get_end_date_options(start_date, max_end_date, max_days=30):
    """
    Generate end date options for Test Model tab based on start date (up to max_days).
    
    Args:
        start_date (str or datetime): The start date.
        max_end_date (str or datetime): The maximum possible end date.
        max_days (int, optional): Maximum days from start date. Defaults to 30.
        
    Returns:
        tuple: (options, default_value) for dropdown.
    """
    if not start_date:
        return [], None
    
    # Convert to datetime objects if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(max_end_date, str):
        max_end_date = datetime.strptime(max_end_date, '%Y-%m-%d')
    
    # Limit to max_days from start date or max_end_date, whichever is earlier
    end_date_limit = min(max_end_date, start_date + timedelta(days=max_days))
    
    if start_date >= end_date_limit:
        return [], None
    
    # Generate dates, excluding weekends and Mondays
    dates = pd.date_range(start=start_date + timedelta(days=1), end=end_date_limit, freq='B')
    valid_dates = [date for date in dates if date.weekday() != 0]  # Exclude Mondays
    
    options = [
        {'label': date.strftime('%Y-%m-%d'), 'value': date.strftime('%Y-%m-%d'), 'disabled': False}
        for date in valid_dates
    ]
    
    if not options:  # If no valid dates, return empty options
        return [], None
    
    # Default to last valid date
    default_value = valid_dates[-1].strftime('%Y-%m-%d') if valid_dates else None
    
    return options, default_value

def get_next_business_day():
    """
    Get the next business day after today.
    
    Returns:
        str: Next business day in YYYY-MM-DD format.
    """
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() in [5, 6]:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    
    return next_day.strftime('%Y-%m-%d')