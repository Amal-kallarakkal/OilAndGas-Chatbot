"""
DuckDB data access layer.
All database interactions in this project go through this module.
No agent, tool, or graph node queries DuckDB directly.
"""
import duckdb
import pandas as pd
from contextlib import contextmanager
from typing import Optional, List
from datetime import date
from src.config import cfg


@contextmanager
def get_connection():
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

PRODUCTION_COLUMNS = {
    'well_id', 'date',
    'oil_prod_bbl',          # synthetic data name
    'oil_produced_bbl',      # your real data name  ← ADD THIS
    'gas_prod_mcf',
    'gas_produced_mcf',      # your real data name  ← ADD THIS (if applicable)
    'downtime_hrs',
    'op_efficiency_pct',
    'water_cut_pct',
    'reservoir_pressure_psi',
}

EQUIPMENT_COLUMNS = {
    'well_id', 'date', 'equipment_type', 'asset_id', 'temperature_c',
    'vibration_level', 'pressure_psi', 'failure_risk_score'
}

def get_production_data(
    well_ids:   List[str],
    start_date: date,
    end_date:   date,
    fields:     Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Retrieve daily production records for one or more wells
    over a date range.

    Args:
        well_ids:   List of well identifiers, e.g. ['WELL_A', 'WELL_B'].
        start_date: First date (inclusive) of the query range.
        end_date:   Last date (inclusive) of the query range.
        fields:     Columns to return. Defaults to all columns.
                    Invalid field names are silently dropped.

    Returns:
        pd.DataFrame sorted by (well_id, date).
        Returns empty DataFrame with correct columns if no data found.
    """
    # ── Input validation ─────────────────────────────────────────
    if not well_ids:
        raise ValueError('well_ids must be a non-empty list.')
    if start_date > end_date:
        raise ValueError('start_date must not be after end_date.')

    # Validate and filter field names against allowlist
    if fields is None:
        fields = list(PRODUCTION_COLUMNS)
    else:
        # Drop invalid names, always keep well_id and date
        valid = {f for f in fields if f in PRODUCTION_COLUMNS}
        valid.update({'well_id', 'date'})
        fields = list(valid)

    col_str = ', '.join(fields)

    # ── Build placeholders for well_ids list ────────────────────
    # DuckDB does not support list binding for IN clauses directly,
    # so we build one ? per well_id and bind them individually.
    placeholders = ', '.join(['?'] * len(well_ids))

    query = f'''
        SELECT {col_str}
        FROM   production
        WHERE  well_id IN ({placeholders})
          AND  date BETWEEN ? AND ?
        ORDER  BY well_id, date
    '''

    params = well_ids + [start_date, end_date]

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
    Retrieve equipment sensor readings and failure risk scores.

    Args:
        well_ids:        List of well identifiers.
        start_date:      Start of query range (inclusive).
        end_date:        End of query range (inclusive).
        equipment_types: Optional filter, e.g. ['pump', 'compressor'].
                         If None, all equipment types are returned.
        fields:          Columns to return. Defaults to all columns.

    Returns:
        pd.DataFrame sorted by (well_id, date).
    """
    if not well_ids:
        raise ValueError('well_ids must be a non-empty list.')
    if start_date > end_date:
        raise ValueError('start_date must not be after end_date.')

    if fields is None:
        fields = list(EQUIPMENT_COLUMNS)
    else:
        valid = {f for f in fields if f in EQUIPMENT_COLUMNS}
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
    Return production and equipment records joined on (well_id, date).
    Used by the Correlation Agent to compute cross-signal relationships.

    Args:
        well_ids:   Wells to include.
        start_date: Start of range (inclusive).
        end_date:   End of range (inclusive).
        join_type:  'inner' (default) or 'left'.
                    Use 'inner' for correlation (requires both signals).
                    Use 'left' to retain production rows with missing equipment.

    Returns:
        pd.DataFrame with columns from both tables.
        Duplicate column names are disambiguated: p_date, e_date.
    """
    if join_type not in ('inner', 'left'):
        raise ValueError("join_type must be 'inner' or 'left'.")

    well_placeholders = ', '.join(['?'] * len(well_ids))
    params = well_ids + [start_date, end_date]

    query = f'''
        SELECT
            p.well_id,
            p.date,
            p.oil_produced_bbl,
            p.gas_prod_mcf,
            p.downtime_hrs,
            p.water_cut_pct,
            e.equipment_type,
            e.vibration_level,
            e.temperature_c,
            e.pressure_psi      AS equip_pressure_psi,
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
        'date_range':        {'min_date': str(date_range[0]),
                              'max_date': str(date_range[1])},
        'production_schema':  prod_schema,
        'equipment_schema':   equip_schema,
    }

def get_production_summary(
    well_ids:   List[str],
    start_date: date,
    end_date:   date,
    group_by:   str = 'well_id'   # 'well_id' | 'month' | 'well_month'
) -> pd.DataFrame:
    """
    Return aggregated production statistics over a date range.

    Args:
        well_ids:   Wells to include.
        start_date: Start of range.
        end_date:   End of range.
        group_by:   Grouping strategy.
                    'well_id'    -> one row per well (default)
                    'month'      -> one row per month across all wells
                    'well_month' -> one row per well per month

    Returns:
        pd.DataFrame with avg, total, min, max per group.
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
            ROUND(MIN(oil_produced_bbl),  2) AS min_oil_bbl,
            ROUND(MAX(oil_produced_bbl),  2) AS max_oil_bbl,
            ROUND(AVG(gas_produced_mcf),  2) AS avg_gas_mcf,
            ROUND(AVG(downtime_hours),  3) AS avg_downtime_hrs,
            COUNT(*) AS record_count
        FROM production
        WHERE well_id IN ({well_ph})
          AND date BETWEEN ? AND ?
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

