"""
LangChain tool wrappers for production data queries.
These functions are decorated with @tool so the LLM can call them.
They translate LLM-friendly string inputs into typed Python calls
against the database functions in src/database.py.
"""
from email.policy import default

from altair import value
from langchain_core.tools import tool
from datetime import date
from typing import Optional, Union

from narwhals import col
from src.database import (
    get_production_data, get_equipment_health, get_joined_data,
    get_production_summary, get_latest_equipment_status, get_schema_info
) 

from src.utils import resolve_date_range, parse_list_param, parse_optional_list
import json

# ── Helper: convert DataFrame to LLM-readable JSON summary ─────────
def _df_to_summary(df, extra:dict = None)->str:
    """
    Convert a DataFrame to a compact JSON string the LLM can read.
    Never passes raw DataFrames to the LLM context.
    """
    if df.empty:
        return json.dumps({'status':'no_data', 'row_count':'0'})
    
    numerica_cols = df.select_dtypes('number').columns.tolist()
    summary = {
        'row_count': len(df),
        'wells':     df['well_id'].unique().tolist() if 'well_id' in 
                     df.columns else [],
        'date_range': [str(df['date'].min()), str(df['date'].max())] if 'date' in
                      df.columns else [],
        'columns':    df.columns.tolist(),
        'statistics': {
            col: {
                'mean':  round(float(df[col].mean()), 3),
                'min':   round(float(df[col].min()),  3),
                'max':   round(float(df[col].max()),  3),
                'latest': round(float(df[col].iloc[-1]), 3)
            }
            for col in numerica_cols
        }
    }
    if extra :
        summary.update(extra)
    return json.dumps(summary, default=str)


def _parse_list(value: Union[list, str]) -> list:
    """
    Coerce a value to a list.
    Some LLMs serialise lists arguments as JSON strings. This handles both test cases.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.load(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            # Treat as a single comma-separated string
            return [v.strip() for v in value.split(',')]
    return [value]

# ── Tool 1: Production data ───────────────────────────────────────
@tool
def fetch_production_data(
    well_ids: list[str],
    date_expression: str = 'last_30_days',
    fields: Optional[list[str]] = None
) -> str:
    """
    Fetch daily production records for one or more wells.

    Use when the user asks about: oil production, gas production,
    water cut, downtime, or any production metric over a time period.

    Args:
        well_ids: Well identifiers. Format: 'A001-W01'.
            Use inspect_database_schema to get valid well IDs first.
            Examples: ['A001-W01'] or ['A001-W01', 'A002-W01']

        date_expression: One of:
            'last_7_days', 'last_14_days', 'last_30_days' (default),
            'last_60_days', 'last_90_days',
            'last_month', 'last_quarter', 'ytd'

        fields: Columns to return (optional). Available columns:
            oil_produced_bbl, gas_produced_mcf, water_cut_pct,
            downtime_hours, asset_id
            (well_id and date are always included)

    Returns:
        JSON with row_count, date_range, and statistics per column.
    """
# ── Type guards (handles LLM JSON-string serialisation) ──────────
    well_ids = _parse_list(well_ids)
    if fields is not None:
        fields = parse_optional_list(fields)

    start, end = resolve_date_range(date_expression)
    df = get_production_data(well_ids, start, end, fields)

    if df.empty:
        return json.dumps({
            'status': 'no_data',
            'wells': well_ids,
            'date_expression': date_expression,
            'message': f'No production data found. Check well IDs and date range.'
        })
    
    return _df_to_summary(df)

# ── Tool 2: Equipment Health ─────────────────────────────────────────
@tool

def fetch_equipment_health(
    well_ids: list[str],
    date_expression: str = 'last_30_days',
    equipment_types: list = None
) -> str:
    """
    Fetch equipment sensor readings and failure risk scores for wells.

    Use when the user asks about: equipment condition, vibration,
    temperature, pressure, failure risk, or maintenance needs.

    Args:
        well_ids: Well identifiers. Format: 'A001-W01'.
        date_expression: Time range. Default 'last_30_days'.
        equipment_types: Optional filter. Values: 'Compressor', 'Valve', 'Pump'.

    Available columns:
        equipment_type, vibration_level, temperature_c,
        pressure_psi, failure_risk_score

    Returns:
        JSON with row_count, date_range, and statistics per column.
    """

    well_ids        = parse_list_param(well_ids, 'well_ids')
    equipment_types = parse_optional_list(equipment_types)

    
    start, end = resolve_date_range(date_expression)
    df = get_equipment_health(well_ids, start, end, equipment_types)

    if df.empty:
        return json.dumps({
            'status': 'no_data',
            'wells': well_ids,
            'message': 'No equipment data found. Check well IDs and date range.'
        })

    return _df_to_summary(df)

# ── Tool 3: Schema Inspector ─────────────────────────────────────────
@tool
def inspect_database_schema() -> str:
    """
    Return all available well IDs, asset IDs, date ranges,
    and column names for both tables.

    ALWAYS call this tool first when:
    - The user refers to a well by a name you don't recognise
    - The user asks what data is available
    - You are unsure whether a well ID exists

    Returns a JSON object with:
        available_wells, available_assets, date_range,
        production_columns, equipment_columns
    """
    
    info = get_schema_info()
    return json.dumps(info, default=str)

# ── Tool 4: Production Summary ───────────────────────────────────────
@tool
def fetch_production_summary(
    well_ids:           list,
    date_expression:   str = 'last_30_days',
    group_by:           str = 'well_id'
) -> str:
    """
    Return aggregated production statistics (averages, totals).

    Use when the user asks for averages, totals, or comparisons
    across wells or time periods rather than raw daily data.

    Args:
        well_ids: Well identifiers.
        date_expression: Time range.
        group_by: 'well_id' (default), 'month', or 'well_month'
    """
    well_ids = parse_list_param(well_ids, 'well_ids')

    start, end = resolve_date_range(date_expression)
    df = get_production_summary(well_ids, start, end, group_by)

    if df.empty:
        return json.dumps({'status': 'no_data', 'wells': well_ids})

    # Summary tool returns the full table (it's already aggregated)
    return df.to_json(orient='records', date_format='iso', default_handler=str)


# ── Tool 5: Latest Equipment Status ─────────────────────────────────
@tool
def fetch_latest_equipment_status(
    well_ids: list = None
) -> str:
    """
    Return the most recent equipment reading for each well.

    Use when the user asks about current equipment condition,
    risk rankings, or which wells need immediate attention.
    Returns one row per well with the latest sensor values.

    Args:
        well_ids: Optional list of wells. If None, returns all wells.
    """
    well_ids = parse_optional_list(well_ids)

    df = get_latest_equipment_status(well_ids)

    if df.empty:
        return json.dumps({'status': 'no_data'})

    return df.to_json(orient='records', date_format='iso', default_handler=str)



