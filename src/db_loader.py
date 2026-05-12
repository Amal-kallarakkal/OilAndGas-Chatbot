import duckdb
import os
from src.config import cfg


def load_datasets():
    conn = duckdb.connect(cfg.DUCKDB_PATH)

    # ----Productions table----
    conn.execute('DROP TABLE IF EXISTS production')
    conn.execute('''
        CREATE TABLE production (
            well_id VARCHAR  NOT NULL,
            date    DATE     NOT NULL, 
            oil_prod_bbl  double,
            gas_prod_mcf double,
            downtime_hrs double,op_efficiency_pct    DOUBLE,
            water_cut_pct        DOUBLE,
            PRIMARY KEY (well_id, date)                    
        )
    ''')
    conn.execute("""
                    CREATE OR REPLACE TABLE production AS
                    SELECT *
                    FROM read_csv_auto(
                        'data/production_data.csv',
                        header=True
                    )
                """)

    # ----Equipment_health table----
    conn.execute('DROP TABLE IF EXISTS equipment_health')
    conn.execute('''
        CREATE TABLE equipment_health (
            well_id              VARCHAR  NOT NULL,
            date                 DATE     NOT NULL, 
            equipment_type       VARCHAR,
            vibration_mm_s       DOUBLE,
            temperature_c        DOUBLE,
            pressure_psi         DOUBLE,
            failure_risk_score   DOUBLE,
            PRIMARY KEY (well_id, date)                   
        )
    ''')
    conn.execute("""
                    CREATE OR REPLACE TABLE equipment_health AS
                    SELECT *
                    FROM read_csv_auto(
                        'data/equipment_health.csv',
                        header=True
                    )
                """)
    # ----verify data load----
    prod_count = conn.execute('SELECT COUNT(*) FROM production').fetchone()[0]
    equip_count = conn.execute('SELECT COUNT(*) FROM equipment_health').fetchone()[0]

    print(f'Production records loaded: {prod_count}')
    print(f'Equipment health records loaded: {equip_count}')

    conn.close()
    return prod_count, equip_count

if __name__ == "__main__":
    load_datasets()