"""
Deterministic analytical functions for production data.
These functions perform all numeric computation.
They never call the LLM and never query the database directly.
They receive DataFrames and return structured result dicts.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional
from datetime import timedelta, date
from src.database import get_production_data
from src.utils import resolve_date_range

def compute_production_trends(
        well_id: str,
        date_expression: str = 'last_60_days',
        metric: str = None
) -> dict:
    """
    Compute the production trend for a single well over a date range.
    Compares the recent half of the window to the prior half.

    Returns a dict with:
        trend_direction: 'declining' | 'stable' | 'improving'
        slope:           bbl/day change (negative = declining)
        pct_change:      % change recent vs prior window
        recent_mean:     average in recent window
        prior_mean:      average in prior window
        confidence:      'high' | 'medium' | 'low'
        data_points:     number of rows used
        is_significant:  True if p-value < 0.05
    """
    if metric is None:
        from src.database import get_schema_info
        schema = get_schema_info()
        prod_cols = [c['column_name'] for c in schema['production_schema']]

        for candidate in ['oil_prod_bbl', 'oil_produced_bbl', 'oil_production_bbl',
                          'daily_oil_bbl', 'oil_bbl']:
            if candidate in prod_cols:
                metric = candidate
                break

            if metric is None:
                return {
                'status': 'column_not_found',
                'available_columns': prod_cols,
                'message': 'Could not find an oil production column. '
                           'Set metric= explicitly.'
            }

    start, end = resolve_date_range(date_expression)
    df = get_production_data([well_id], start, end, fields=[metric].sort_values('date'))

    if len(df) < 6:
        return { 
            'status': 'insufficient_data',
            'data_points': len(df), 
            'minimum_required': 6
        }
    # split into prior and recent halves
    mid = len(df) // 2
    prior = df.iloc[:mid]
    recent = df.iloc[mid:]

    recent_mean = recent[metric].mean()
    prior_mean = prior[metric].mean()
    pct_change = ((recent_mean - prior_mean) / prior_mean) * 100 if prior_mean != 0 else np.inf

    # linear regression over the full window to get slope
    x = np.arange(len(df))
    y = df[metric].values
    slope, _, r_value, p_value, _ = stats.linregress(x, y)

    # classify trend direction
    if pct_change < -5 and p_value < 0.05:
        direction = 'declining'
    elif pct_change > 5 and p_value < 0.05:
        direction = 'improving'
    else:
        direction = 'stable'
    
    confidence = ('high' if p_value < 0.01 and len(df) > 21 
                  else 'medium' if p_value < 0.05 else 'low')
    
    return {
        'well_id': well_id,
        'trend_direction': direction,
        'metric': metric,
        'date_range': [str(start), str(end)],
        'slope': round(float(slope), 4),
        'pct_change': round(pct_change, 2),
        'recent_mean': round(recent_mean, 2),
        'prior_mean': round(prior_mean, 2),
        'p_value': round(p_value, 4),
        'r_squred': round(r_value**2, 4),
        'confidence': confidence,
        'data_points': len(df),
        'is_significant': p_value < 0.05
    }

