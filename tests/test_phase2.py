import pytest, json
from datetime import date
from src.database import (
    _production_columns, _equipment_columns, get_schema_info,
    get_production_data, get_equipment_health,
    get_latest_equipment_status, get_production_summary
)
from src.tools.production_tools import (
    fetch_production_data, fetch_equipment_health,
    inspect_database_schema, fetch_production_summary,
    fetch_latest_equipment_status
)
from src.utils import parse_list_param, parse_optional_list
from src.analytics.production_analytics import (
    compute_production_trend, _detect_oil_column
)

REAL_WELL   = 'A001-W01'
FAKE_WELL   = 'WELL_DOES_NOT_EXIST'
DATE_START  = date(2023, 1, 1)
DATE_END    = date(2023, 3, 31)

# ── Schema registry tests ────────────────────────────────────────────
class TestSchemaRegistry:

    def test_production_columns_contains_real_names(self):
        cols = _production_columns()
        # These are your REAL column names from the CSV
        assert 'oil_produced_bbl'  in cols, 'Missing oil_produced_bbl'
        assert 'gas_produced_mcf'  in cols, 'Missing gas_produced_mcf'
        assert 'water_cut_pct'     in cols, 'Missing water_cut_pct'
        assert 'downtime_hours'    in cols, 'Missing downtime_hours'
        assert 'asset_id'          in cols, 'Missing asset_id'

    def test_production_columns_no_phantom_names(self):
        cols = _production_columns()
        # These are the WRONG synthetic names from the Phase 1 doc
        assert 'oil_prod_bbl'   not in cols, 'Phantom column: oil_prod_bbl'
        assert 'gas_prod_mcf'   not in cols, 'Phantom column: gas_prod_mcf'
        assert 'downtime_hrs'   not in cols, 'Phantom column: downtime_hrs'

    def test_equipment_columns_contains_real_names(self):
        cols = _equipment_columns()
        assert 'vibration_level'     in cols, 'Missing vibration_level'
        assert 'temperature_c'       in cols
        assert 'pressure_psi'        in cols
        assert 'failure_risk_score'  in cols
        assert 'equipment_type'      in cols

    def test_equipment_columns_no_phantom_names(self):
        cols = _equipment_columns()
        assert 'vibration_mm_s'      not in cols
        assert 'op_efficiency_pct'   not in cols
        assert 'reservoir_pressure_psi' not in cols

    def test_schema_info_returns_real_wells(self):
        info = get_schema_info()
        assert REAL_WELL in info['available_wells']
        assert len(info['available_wells']) > 0
        assert 'A001' in info['available_assets']

    def test_oil_column_detection(self):
        col = _detect_oil_column()
        assert col == 'oil_produced_bbl'


# ── Database function tests ──────────────────────────────────────────
class TestDatabaseFunctions:

    def test_production_returns_real_columns(self):
        df = get_production_data([REAL_WELL], DATE_START, DATE_END)
        assert 'oil_produced_bbl' in df.columns
        assert 'gas_produced_mcf' in df.columns
        assert 'downtime_hours'   in df.columns
        assert 'oil_prod_bbl' not in df.columns  # phantom name must be absent

    def test_production_field_selection_real_names(self):
        df = get_production_data(
            [REAL_WELL], DATE_START, DATE_END,
            fields=['oil_produced_bbl', 'downtime_hours']
        )
        assert 'oil_produced_bbl' in df.columns
        assert 'downtime_hours'   in df.columns
        assert 'gas_produced_mcf' not in df.columns

    def test_phantom_field_silently_dropped(self):
        # Requesting a non-existent column should not crash
        # It should be silently ignored
        df = get_production_data(
            [REAL_WELL], DATE_START, DATE_END,
            fields=['oil_produced_bbl', 'DOES_NOT_EXIST']
        )
        assert 'oil_produced_bbl' in df.columns
        assert 'DOES_NOT_EXIST'   not in df.columns

    def test_equipment_returns_vibration_level(self):
        df = get_equipment_health([REAL_WELL], DATE_START, DATE_END)
        assert 'vibration_level' in df.columns
        assert 'vibration_mm_s'  not in df.columns  # phantom

    def test_production_summary_uses_real_columns(self):
        df = get_production_summary([REAL_WELL], DATE_START, DATE_END)
        assert 'avg_oil_bbl'         in df.columns
        assert 'avg_downtime_hours'   in df.columns
        assert 'avg_efficiency_pct'   not in df.columns  # does not exist


# ── Type guard tests ─────────────────────────────────────────────────
class TestTypeGuards:

    def test_parse_list_from_native_list(self):
        assert parse_list_param(['A001-W01'], 'w') == ['A001-W01']

    def test_parse_list_from_json_string(self):
        # This is what NVIDIA model sends
        result = parse_list_param('["A001-W01"]', 'w')
        assert result == ['A001-W01']

    def test_parse_list_from_json_multiple(self):
        result = parse_list_param('["A001-W01", "A001-W02"]', 'w')
        assert result == ['A001-W01', 'A001-W02']

    def test_parse_list_from_comma_string(self):
        result = parse_list_param('A001-W01, A001-W02', 'w')
        assert result == ['A001-W01', 'A001-W02']

    def test_parse_optional_list_none_returns_none(self):
        assert parse_optional_list(None)   is None
        assert parse_optional_list([])     is None
        assert parse_optional_list('[]')   is None

    def test_parse_optional_list_with_value(self):
        result = parse_optional_list('["Pump"]')
        assert result == ['Pump']


# ── Tool wrapper tests (no LLM) ─────────────────────────────────────
class TestToolWrappers:

    def test_fetch_production_returns_json(self):
        result = fetch_production_data.invoke({
            'well_ids': [REAL_WELL],
            'date_expression': 'last_30_days'
        })
        parsed = json.loads(result)
        assert 'row_count'  in parsed
        assert 'statistics' in parsed
        # Oil column must use REAL name
        assert 'oil_produced_bbl' in parsed['statistics']
        assert 'oil_prod_bbl' not in parsed['statistics']

    def test_fetch_production_json_string_well_ids(self):
        # Simulate what NVIDIA model actually sends
        result = fetch_production_data.invoke({
            'well_ids': f'["{REAL_WELL}"]',  # JSON string, not list
            'date_expression': 'last_30_days'
        })
        parsed = json.loads(result)
        # Must succeed, not crash or return no_data
        assert parsed.get('status') != 'no_data', (
            f'Type guard failed: received no_data for real well via JSON string. '
            f'parse_list_param is not being called.'
        )

    def test_fetch_production_unknown_well_returns_no_data(self):
        result = fetch_production_data.invoke({
            'well_ids': [FAKE_WELL],
            'date_expression': 'last_30_days'
        })
        parsed = json.loads(result)
        assert parsed['status'] == 'no_data'

    def test_fetch_equipment_returns_vibration_level(self):
        result = fetch_equipment_health.invoke({
            'well_ids': [REAL_WELL],
            'date_expression': 'last_30_days'
        })
        parsed = json.loads(result)
        assert 'vibration_level' in parsed['statistics']
        assert 'vibration_mm_s'  not in parsed['statistics']

    def test_inspect_schema_returns_real_wells(self):
        result = json.loads(inspect_database_schema.invoke({}))
        assert REAL_WELL in result['available_wells']
        


# ── Analytics tests (no LLM) ─────────────────────────────────────────
class TestAnalytics:

    def test_trend_uses_real_column(self):
        result = compute_production_trend(REAL_WELL, 'last_60_days')
        assert 'status' not in result or result['status'] != 'column_not_found'
        assert result.get('metric') == 'oil_produced_bbl'

    def test_trend_unknown_well_returns_insufficient(self):
        result = compute_production_trend(FAKE_WELL, 'last_30_days')
        assert result['status'] == 'insufficient_data'

    def test_trend_direction_field_present(self):
        result = compute_production_trend(REAL_WELL, 'last_60_days')
        if 'status' not in result:
            assert result['trend_direction'] in ('declining', 'stable', 'improving')
            assert 'p_value'     in result
            assert 'confidence'  in result

