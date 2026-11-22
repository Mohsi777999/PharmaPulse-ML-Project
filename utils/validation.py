"""
PharmaPulse - Validation Utility Module
Provides defensive checks for user selections, filters, dates, and horizon constraints.
"""

from datetime import datetime

def validate_horizon(horizon_days, min_horizon=1, max_horizon=180):
    """
    Validates that the forecasting horizon falls within acceptable simulation bounds.
    """
    try:
        h = int(horizon_days)
        if h < min_horizon or h > max_horizon:
            return False, f"Horizon must be between {min_horizon} and {max_horizon} days."
        return True, h
    except (ValueError, TypeError):
        return False, "Horizon must be an integer."

def validate_date_range(start_date, end_date):
    """
    Checks that start_date <= end_date.
    """
    if start_date > end_date:
        return False, "Start date cannot be after end date."
    return True, "Valid date range."

def validate_medicine_id(med_id, valid_ids):
    """
    Verifies medicine_id exists in the valid catalog.
    """
    if med_id not in valid_ids:
        return False, f"Medicine ID {med_id} not found in catalog."
    return True, "Valid medicine ID."
