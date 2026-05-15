"""
Deterministic analytical functions for production data.
These functions perform all numeric computation.
They never call the LLM and never query the database directly.
They receive DataFrames and return structured result dicts.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.database import get_production_data, _production_columns
from src.utils import resolve_date_range

# The canonical oil column name in the real database
# Detected once at import time from the live schema
def _detect_oil_column() -> str:
    """Find the oil production column in the real schema."""
    candidates = [
        'oil_produced_bbl',    # your real column
        'oil_prod_bbl',        # synthetic data name
        'oil_production_bbl',
        'daily_oil_bbl',
    ]
    real_cols = _production_columns()
    for c in candidates:
        if c in real_cols:
            return c
    raise RuntimeError(
        f'No oil production column found. '
        f'Available columns: {sorted(real_cols)}'
    )


def _production_trend(
        well_id: str,
        date_expression: str = 'last_60_days',
        metric: str = None
) -> dict:
    """
    Compute the production trend for one well over a date window.
    Compares the recent half vs the prior half of the window.

    Args:
        well_id:         Well identifier, e.g. 'A001-W01'.
        date_expression: Time window. Default 'last_60_days'.
        metric:          Column to analyse. Auto-detected if None.
                         Auto-detect finds 'oil_produced_bbl' in your schema.

    Returns dict with:
        trend_direction: 'declining' | 'stable' | 'improving'
        slope:           units/day change
        pct_change:      % change recent vs prior
        recent_mean, prior_mean, p_value, confidence, data_points
    """

    if metric is None:
        metric = _detect_oil_column()

    start, end = resolve_date_range(date_expression)
    df = get_production_data(
        [well_id], start, end,
        fields=[metric]
    ).sort_values('date')

    if len(df) < 6:
        return {
            'status':           'insufficient_data',
            'well_id':          well_id,
            'data_points':      len(df),
            'minimum_required': 6
        }

    mid         = len(df) // 2
    prior_mean  = df.iloc[:mid][metric].mean()
    recent_mean = df.iloc[mid:][metric].mean()
    pct_change  = ((recent_mean - prior_mean) / prior_mean) * 100

    x = np.arange(len(df))
    y = df[metric].values
    slope, _, r_val, p_val, _ = stats.linregress(x, y)

    direction = ('declining' if pct_change < -5 and p_val < 0.05 else
                 'improving' if pct_change >  5 and p_val < 0.05 else
                 'stable')

    confidence = ('high'   if p_val < 0.01 and len(df) >= 21 else
                  'medium' if p_val < 0.05 else 'low')

    return {
        'well_id':          well_id,
        'metric':           metric,
        'date_range':       [str(start), str(end)],
        'trend_direction':  direction,
        'slope':            round(float(slope), 4),
        'pct_change':       round(pct_change, 2),
        'recent_mean':      round(recent_mean, 2),
        'prior_mean':       round(prior_mean, 2),
        'p_value':          round(p_val, 4),
        'r_squared':        round(r_val**2, 4),
        'confidence':       confidence,
        'is_significant':   bool(p_val < 0.05),
        'data_points':      len(df),
    }


