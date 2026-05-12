"""
LangChain tool wrappers for production data queries.
These functions are decorated with @tool so the LLM can call them.
They translate LLM-friendly string inputs into typed Python calls
against the database functions in src/database.py.
"""
from altair import value
from langchain_core.tools import tool
from datetime import date
from typing import Optional, Union
from src.database import (
    get_production_data, get_equipment_health,
    get_latest_equipment_status, get_schema_info,
    get_production_summary
) 

from src.utils import resolve_date_range
import json

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


@tool
def fetch_production_data(
    well_ids: list[str],
    date_expression: str = 'last_30_days',
    fields: Optional[list[str]] = None
) -> str:
    """
    Fetch daily production records for one or more wells.
    Use this tool when the user asks about production volumes,
    oil output, gas output, downtime, or operational efficiency
    for specific wells over a time period.

    Args:
        well_ids: List of well identifiers, e.g. ['WELL_A', 'WELL_B'].
        date_expression: Time range using standard expressions:
            'last_30_days', 'last_60_days', 'last_90_days',
            'last_month', 'last_quarter', 'ytd'.
            Default is 'last_30_days'.
        fields: Optional list of specific columns to return.
            Available: oil_produced_bbl, gas_prod_mcf, downtime_hrs,
            op_efficiency_pct, water_cut_pct, pressure_psi.
    """
    well_ids = _parse_list(well_ids)
    if fields is not None:
        fields = _parse_list(fields)

    start, end = resolve_date_range(date_expression)
    df = get_production_data(well_ids, start, end, fields)

    if df.empty:
        return f'No production data found for {well_ids} in the specified {date_expression} range.'

    # Return a compact text summary, not a raw DataFrame.
    # The LLM reads this text to understand what was retrieved.
    summary = {
        'wells':      df['well_id'].unique().tolist(),
        'date_range': [str(df['date'].min()), str(df['date'].max())],
        'row_count':  len(df),
        'columns':    df.columns.tolist(),
        'sample_stats': {
            col: {'mean': round(df[col].mean(), 2),
                  'min':  round(df[col].min(),  2),
                  'max':  round(df[col].max(),  2)}
            for col in df.select_dtypes('number').columns
        }
    }
    return json.dumps(summary, default=str)


@tool

def fetch_equipment_health(
    well_ids: list[str],
    date_expression: str = 'last_30_days'
) -> str:
    """
    Fetch quipment sensor readings and failure risk scores for wells.
    Use this tool when the user asks about equipment condition, 
    Vibration levels, temperature, pressure readings, or failure
    risk for specific wells.
    
    Args:
        well_ids: list of well identifiers, e.g. ['WELL_A', 'WELL_B'].  
        date_expression: Time range expression, default 'last_30_days'.
    """
    well_ids = _parse_list(well_ids)
    
    start, end = resolve_date_range(date_expression)
    df = get_equipment_health(well_ids, start, end)

    if df.empty:
        return f'No equipment health data found for {well_ids} in the specified {date_expression} range.'
    
    summary = {
        'wells':    df['well_id'].unique().tolist(),
        'date_range': [str(df['date'].min()), str(df['date'].max())],
        'row_count': len(df),
        'sample stats': {
            col: {'mean': round(df[col].mean(), 2),
                  'min':  round(df[col].min(),  2),
                  'max':  round(df[col].max(),  2)}
            for col in df.select_dtypes('number').columns
        }

    }
    return json.dumps(summary, default=str)

@tool
def inspect_database_schema() -> str:
    """
    Return metadata about the database: available wells, date ranges,
    and column names for both tables.
    Use this tool first if the user asks a general question and you
    need to know what wells or date ranges are available before
    deciding which other tools to call.
    """
    info = get_schema_info()
    return json.dumps(info, default=str)
