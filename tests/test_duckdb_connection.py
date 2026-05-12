import duckdb
from src.config import cfg
import os

def test_duckdb_connection():
    os.makedirs(os.path.dirname(cfg.DUCKDB_PATH), exist_ok=True)

    conn = duckdb.connect(cfg.DUCKDB_PATH)

    # verify duckdb version
    version = conn.execute('SELECT version()').fetchone()[0]
    result = conn.execute('SELECT 1 + 1').fetchone()[0]

    print(f'DuckDB Version: {version}')
    print(f'Test Query Result (1 + 1): {result}')
    assert result == 2, 'DuckDB connection test failed: 1 + 1 did not equal 2'

    conn.close()
    print('PASS: DuckDB connection successful and test query executed. created at', cfg.DUCKDB_PATH)

if __name__ == "__main__":
    test_duckdb_connection()