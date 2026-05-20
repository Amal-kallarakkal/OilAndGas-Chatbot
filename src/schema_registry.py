from dataclasses import dataclass
from functools import lru_cache

from src.database import get_schema_info


# ── Nested schema containers ────────────────────────────────────────

@dataclass(frozen=True)
class TableSchema:
    columns: list[str]
    numeric_columns: list[str]


@dataclass(frozen=True)
class SchemaRegistry:
    well_ids: list[str]
    asset_ids: list[str]
    oil_column: str
    date_range: dict

    production: TableSchema
    equipment: TableSchema


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_columns(schema_records: list[dict]) -> list[str]:
    return [c['column_name'] for c in schema_records]


def _extract_numeric_columns(schema_records: list[dict]) -> list[str]:

    numeric_markers = (
        'INTEGER',
        'BIGINT',
        'SMALLINT',
        'FLOAT',
        'DOUBLE',
        'REAL',
        'DECIMAL',
        'NUMERIC',
    )

    cols = []

    for c in schema_records:

        dtype = c['data_type'].upper()

        if any(marker in dtype for marker in numeric_markers):
            cols.append(c['column_name'])

    return cols


def _detect_oil_column(prod_columns: list[str]) -> str:

    candidates = [
        'oil_produced_bbl',
        'oil_prod_bbl',
        'oil_production_bbl',
        'daily_oil_bbl',
    ]

    for col in candidates:
        if col in prod_columns:
            return col

    raise RuntimeError(
        f'No oil production column found. '
        f'Available columns: {prod_columns}'
    )


# ── Main registry loader ────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_registry() -> SchemaRegistry:

    info = get_schema_info()

    prod_columns = _extract_columns(
        info['production_schema']
    )

    equip_columns = _extract_columns(
        info['equipment_schema']
    )

    return SchemaRegistry(

        well_ids=info['available_wells'],

        asset_ids=info['available_assets'],

        oil_column=_detect_oil_column(prod_columns),

        date_range=info['date_range'],

        production=TableSchema(
            columns=prod_columns,
            numeric_columns=_extract_numeric_columns(
                info['production_schema']
            )
        ),

        equipment=TableSchema(
            columns=equip_columns,
            numeric_columns=_extract_numeric_columns(
                info['equipment_schema']
            )
        )
    )


# Global singleton-style registry
registry = load_registry()