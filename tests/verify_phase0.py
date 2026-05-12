
# tests/test_phase1_database.py
import pytest
from datetime import date
from src.database import (
    get_production_data, get_equipment_health,
    get_joined_data, get_schema_info,
    get_production_summary, get_latest_equipment_status
)
from src.utils import resolve_date_range

# ─── Fixtures ────────────────────────────────────────────────────────
START = date(2025, 7, 1)
END   = date(2026, 1, 1)
WELLS = ['WELL_A', 'WELL_B']


# ─── get_production_data ─────────────────────────────────────────────
def test_production_returns_dataframe():
    df = get_production_data(WELLS, START, END)
    assert not df.empty
    assert 'oil_prod_bbl' in df.columns
    assert 'well_id' in df.columns

def test_production_respects_date_range():
    narrow_start = date(2025, 10, 1)
    narrow_end   = date(2025, 10, 31)
    df = get_production_data(['WELL_A'], narrow_start, narrow_end)
    assert df['date'].min() >= narrow_start
    assert df['date'].max() <= narrow_end

def test_production_field_selection():
    df = get_production_data(WELLS, START, END,
                             fields=['oil_prod_bbl', 'downtime_hrs'])
    # well_id and date always included
    assert 'well_id' in df.columns
    assert 'date'    in df.columns
    assert 'oil_prod_bbl' in df.columns
    # Gas prod was NOT requested
    assert 'gas_prod_mcf' not in df.columns

def test_production_unknown_well_returns_empty():
    df = get_production_data(['WELL_UNKNOWN'], START, END)
    assert df.empty
    # Empty DataFrame still has the correct columns
    assert 'oil_prod_bbl' in df.columns

def test_production_invalid_date_raises():
    with pytest.raises(ValueError):
        get_production_data(WELLS, END, START)   # start > end


# ─── get_equipment_health ────────────────────────────────────────────
def test_equipment_returns_dataframe():
    df = get_equipment_health(WELLS, START, END)
    assert not df.empty
    assert 'vibration_mm_s' in df.columns

def test_equipment_type_filter():
    df = get_equipment_health(['WELL_A'], START, END,
                             equipment_types=['pump'])
    if not df.empty:
        assert all(df['equipment_type'] == 'pump')


# ─── get_joined_data ─────────────────────────────────────────────────
def test_join_contains_both_tables_columns():
    df = get_joined_data(WELLS, START, END)
    # Production columns
    assert 'oil_prod_bbl'    in df.columns
    # Equipment columns
    assert 'vibration_mm_s'  in df.columns
    assert 'failure_risk_score' in df.columns

def test_inner_join_no_nulls_in_key_columns():
    df = get_joined_data(WELLS, START, END, join_type='inner')
    assert df['oil_prod_bbl'].isna().sum() == 0
    assert df['vibration_mm_s'].isna().sum() == 0


# ─── get_schema_info ─────────────────────────────────────────────────
def test_schema_info_structure():
    info = get_schema_info()
    assert 'available_wells' in info
    assert 'date_range'      in info
    assert len(info['available_wells']) == 7   # 7 wells in synthetic data
    assert 'WELL_A' in info['available_wells']


# ─── get_production_summary ──────────────────────────────────────────
def test_summary_well_grouping():
    df = get_production_summary(WELLS, START, END, group_by='well_id')
    assert len(df) == 2   # One row per well
    assert 'avg_oil_bbl' in df.columns
    assert 'total_oil_bbl' in df.columns

def test_summary_month_grouping():
    df = get_production_summary(WELLS, START, END, group_by='month')
    assert len(df) >= 6   # 6+ months of data


# ─── get_latest_equipment_status ─────────────────────────────────────
def test_latest_equipment_one_row_per_well():
    df = get_latest_equipment_status()
    assert len(df) == 7   # 7 wells
    assert df['well_id'].nunique() == 7
    assert 'rn' not in df.columns   # Helper column must be dropped


# ─── resolve_date_range ──────────────────────────────────────────────
def test_last_30_days():
    from datetime import date, timedelta
    start, end = resolve_date_range('last_30_days')
    assert (end - start).days == 30
    assert end == date.today()

def test_invalid_expression_raises():
    with pytest.raises(ValueError):
        resolve_date_range('last_gibberish')
    