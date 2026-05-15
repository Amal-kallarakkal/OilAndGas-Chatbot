"""
DuckDB data access layer.
Schema is loaded dynamically from the database at runtime.
No column name is hardcoded anywhere in this file.
"""
import functools
from sys import maxsize

import duckdb
import pandas as pd
import functools
from contextlib import contextmanager
from typing import Optional, List
from datetime import date
from src.config import cfg


@contextmanager
def get_connection(read_only: bool = True):
    """
    Context manager for DuckDB connections.
    Usage:
        with get_connection() as conn:
            df = conn.execute(query).fetchdf()
    The connection is always closed after the with-block exits,
    even if an exception is raised inside it.
    """
    conn = duckdb.connect(cfg.DUCKDB_PATH, read_only=True)
    try:
        yield conn
    finally:
        conn.close()

# ── Column allowlist ─────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _production_columns() -> frozenset:
    """Load real production column names from the database at runtime."""
    with get_connection() as conn:
        result = conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'production'
            ORDER BY ordinal_position
        """).fetchdf()
    return frozenset(result['column_name'].tolist())


@functools.lru_cache(maxsize=1)
def _equipment_columns() -> frozenset:
    """Load real equipment column names from the database at runtime."""
    with get_connection() as conn:
        result = conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'equipment_health'
            ORDER  BY ordinal_position
        """).fetchdf()
    return frozenset(result['column_name'].tolist())

def get_production_data(
    well_ids:   List[str],
    start_date: date,
    end_date:   date,
    fields:     Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieve daily production records for one or more wells.

    Real columns available:
        date, asset_id, well_id, oil_produced_bbl,
        gas_produced_mcf, water_cut_pct, downtime_hours

    Returns pd.DataFrame sorted by (well_id, date).
    Returns empty DataFrame with correct columns if no data found.
    """
    # ── Input validation ─────────────────────────────────────────
    if not well_ids:
        raise ValueError('well_ids must be a non-empty list.')
    if start_date > end_date:
        raise ValueError('start_date must not be after end_date.')
    
    VALID_COLS = _production_columns()  

    # Validate and filter field names against allowlist
    if fields is None:
        fields = list(VALID_COLS)
    else:
        # Drop invalid names, always keep well_id and date
        valid = {f for f in fields if f in VALID_COLS}
        valid.update({'well_id', 'date'})
        fields = list(valid)

    col_str = ', '.join(fields)

    # ── Build placeholders for well_ids list ────────────────────
    # DuckDB does not support list binding for IN clauses directly,
    # so we build one ? per well_id and bind them individually.
    placeholders = ', '.join(['?'] * len(well_ids))
    
    params = well_ids + [start_date, end_date]

    query = f'''
        SELECT {col_str}
        FROM   production
        WHERE  well_id IN ({placeholders})
          AND  date BETWEEN ? AND ?
        ORDER  BY well_id, date
    '''


    # ── Execute ──────────────────────────────────────────────────
    with get_connection() as conn:
        df = conn.execute(query, params).fetchdf()

    # ── Guarantee correct schema on empty result ─────────────────
    if df.empty:
        return pd.DataFrame(columns=fields)
    return df


def get_equipment_health(
    well_ids:        List[str],
    start_date:      date,
    end_date:        date,
    equipment_types: Optional[List[str]] = None,
    fields:          Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieve equipment sensor data for one or more wells.

    Real columns available:
        date, asset_id, well_id, equipment_type,
        vibration_level, temperature_c, pressure_psi, failure_risk_score

    equipment_types: Optional filter on 'Compressor', 'Valve', 'Pump'.
    """
    if not well_ids:
        raise ValueError('well_ids must be a non-empty list.')
    if start_date > end_date:
        raise ValueError('start_date must not be after end_date.')
    
    VALID_COLS = _equipment_columns()

    if fields is None:
        fields = list(VALID_COLS)
    else:
        valid = {f for f in fields if f in VALID_COLS}
        valid.update({'well_id', 'date'})
        fields = list(valid)

    col_str = ', '.join(fields)
    well_placeholders = ', '.join(['?'] * len(well_ids))
    params = well_ids + [start_date, end_date]

    # Build optional equipment_type filter
    equip_filter = ''
    if equipment_types:
        eq_placeholders = ', '.join(['?'] * len(equipment_types))
        equip_filter = f'AND equipment_type IN ({eq_placeholders})'
        params += equipment_types

    query = f'''
        SELECT {col_str}
        FROM   equipment_health
        WHERE  well_id IN ({well_placeholders})
          AND  date BETWEEN ? AND ?
          {equip_filter}
        ORDER  BY well_id, date
    '''

    with get_connection() as conn:
        df = conn.execute(query, params).fetchdf()

    if df.empty:
        return pd.DataFrame(columns=fields)
    return df

def get_joined_data(
    well_ids:   List[str],
    start_date: date,
    end_date:   date,
    join_type:  str = 'inner'
) -> pd.DataFrame:
    """
    Join production_data and equipment_health on (well_id, date).
    Returns all columns from both tables with no duplicates.
    """
    if join_type not in ('inner', 'left'):
        raise ValueError("join_type must be 'inner' or 'left'.")

    well_placeholders = ', '.join(['?'] * len(well_ids))
    params = well_ids + [start_date, end_date]

    query = f'''
        SELECT
            p.date,
            p.asset_id,
            p.well_id,
            p.oil_produced_bbl,
            p.gas_produced_mcf,
            p.water_cut_pct,
            p.downtime_hours,
            e.equipment_type,
            e.vibration_level,
            e.temperature_c,
            e.pressure_psi,
            e.failure_risk_score
        FROM production p
        {join_type.upper()} JOIN equipment_health e
            ON  p.well_id = e.well_id
            AND p.date    = e.date
        WHERE p.well_id IN ({well_placeholders})
          AND p.date BETWEEN ? AND ?
        ORDER BY p.well_id, p.date
    '''

    with get_connection() as conn:
        df = conn.execute(query, params).fetchdf()

    return df
def get_schema_info() -> dict:
    """
    Returns metadata about the database: available wells, date ranges,
    and column schemas for both tables.
    Used by the Query Planning Agent to build valid query plans.

    Returns:
        dict with keys: available_wells, date_range, production_schema,
                        equipment_schema
    """
    with get_connection() as conn:

        wells = conn.execute(
            'SELECT DISTINCT well_id FROM production ORDER BY well_id'
        ).fetchdf()['well_id'].tolist()

        assets = conn.execute(
            'SELECT DISTINCT asset_id FROM production ORDER BY asset_id'
        ).fetchdf()['asset_id'].tolist()


        date_range = conn.execute('''
            SELECT MIN(date) AS min_date, MAX(date) AS max_date
            FROM production
        ''').fetchone()

        prod_schema = conn.execute('''
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'production'
            ORDER BY ordinal_position
        ''').fetchdf().to_dict('records')

        equip_schema = conn.execute('''
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'equipment_health'
            ORDER BY ordinal_position
        ''').fetchdf().to_dict('records')

    return {
        'available_wells':   wells,
        'available_assets':     assets,
        'date_range':        {'min_date': str(date_range[0]),
                              'max_date': str(date_range[1])},
        'production_schema':  prod_schema,
        'equipment_schema':   equip_schema,
    }

def get_latest_equipment_status(
    well_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Return the most recent equipment row per well."""
    if well_ids:
        well_ph = ', '.join(['?'] * len(well_ids))
        where   = f'WHERE well_id IN ({well_ph})'
        params  = well_ids
    else:
        where, params = '', []

    query = f'''
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY well_id ORDER BY date DESC
                   ) AS rn
            FROM equipment_health {where}
        )
        WHERE rn = 1
        ORDER BY well_id
    '''
    with get_connection() as conn:
        return conn.execute(query, params).fetchdf()

def get_production_summary(
    well_ids:   List[str],
    start_date: date,
    end_date:   date,
    group_by:   str = 'well_id'
) -> pd.DataFrame:
    """
    Aggregated production stats grouped by well, month, or both.
    Uses real column names: oil_produced_bbl, gas_produced_mcf,
    downtime_hours, water_cut_pct.
    """
    VALID_GROUPS = {'well_id', 'month', 'well_month'}
    if group_by not in VALID_GROUPS:
        raise ValueError(f'group_by must be one of {VALID_GROUPS}')

    well_ph = ', '.join(['?'] * len(well_ids))
    params  = well_ids + [start_date, end_date]

    # Build SELECT and GROUP BY clauses based on grouping strategy
    if group_by == 'well_id':
        select_group = 'well_id'
        group_clause = 'GROUP BY well_id'
    elif group_by == 'month':
        select_group = "DATE_TRUNC('month', date) AS month"
        group_clause = "GROUP BY DATE_TRUNC('month', date)"
    else:  # well_month
        select_group = "well_id, DATE_TRUNC('month', date) AS month"
        group_clause = "GROUP BY well_id, DATE_TRUNC('month', date)"

    query = f'''
        SELECT
            {select_group},
            ROUND(AVG(oil_produced_bbl),  2) AS avg_oil_bbl,
            ROUND(SUM(oil_produced_bbl),  2) AS total_oil_bbl,
            ROUND(AVG(gas_produced_mcf),  2) AS avg_gas_mcf,
            ROUND(AVG(water_cut_pct),     2) AS avg_water_cut_pct,
            ROUND(AVG(downtime_hours),    3) AS avg_downtime_hours,
            COUNT(*)                         AS record_count
        FROM production 
        WHERE well_id IN ({well_ph}) AND date BETWEEN ? AND ?
        {group_clause}
        ORDER BY 1
    '''
    with get_connection() as conn:
        return conn.execute(query, params).fetchdf()
    

def get_latest_equipment_status(
    well_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Return the most recent equipment health record for each well.
    Used by the Risk Assessment Agent to compute current risk scores.

    Args:
        well_ids: Wells to include. If None, returns all wells.

    Returns:
        pd.DataFrame with one row per well, latest date.
    """
    if well_ids:
        well_ph = ', '.join(['?'] * len(well_ids))
        where   = f'WHERE well_id IN ({well_ph})'
        params  = well_ids
    else:
        where  = ''
        params = []

    # QUALIFY with ROW_NUMBER: DuckDB supports this window function
    # to select the latest row per group without a self-join.
    query = f'''
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY well_id
                       ORDER BY date DESC
                   ) AS rn
            FROM equipment_health
            {where}
        ) sub
        WHERE rn = 1
        ORDER BY well_id
    '''

    with get_connection() as conn:
        df = conn.execute(query, params).fetchdf()

    # Drop the helper column before returning
    return df.drop(columns=['rn'], errors='ignore')

