"""Local DuckDB utilities shared by preprocessing and settlement checks."""
from pathlib import Path

import duckdb


def connect():
    con = duckdb.connect()
    con.execute("SET TimeZone='Asia/Seoul'")
    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=2")
    return con


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def read_view(con, name, path):
    path = Path(path).resolve()
    if path.suffix == '.parquet':
        reader = f"read_parquet({literal(path.as_posix())})"
    elif path.suffix == '.csv':
        reader = f"read_csv_auto({literal(path.as_posix())}, header=true, sample_size=-1)"
    else:
        raise ValueError(f'Expected CSV or Parquet: {path}')
    con.execute(f'CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {reader}')


def reject(con, sql, message):
    if con.execute(sql).fetchone()[0]:
        raise ValueError(message)


def check_keys(con, table, keys):
    cols = ', '.join(f'"{key}"' for key in keys)
    nulls = ' OR '.join(f'"{key}" IS NULL' for key in keys)
    reject(con, f'SELECT count(*) FROM {table} WHERE {nulls}', f'{table}: null key')
    reject(con, f'SELECT count(*)-count(DISTINCT ({cols})) FROM {table}',
           f'{table}: duplicate key {keys}')


def check_time(con, table, hourly=True):
    schema = dict((r[0], r[1]) for r in con.execute(f'DESCRIBE {table}').fetchall())
    if schema.get('ts') != 'TIMESTAMP WITH TIME ZONE':
        raise ValueError(f'{table}: ts must include a timezone')
    reject(con, f'SELECT count(*) FROM {table} WHERE ts IS NULL OR NOT isfinite(ts)',
           f'{table}: invalid timestamp')
    if hourly:
        reject(con, f"SELECT count(*) FROM {table} WHERE ts != date_trunc('hour', ts)",
               f'{table}: timestamps must be on the hour')


def check_finite(con, table, column, nonnegative=False):
    predicate = f'"{column}" IS NULL OR NOT isfinite("{column}")'
    if nonnegative:
        predicate += f' OR "{column}" < 0'
    reject(con, f'SELECT count(*) FROM {table} WHERE {predicate}',
           f'{table}: invalid {column}')


def check_sites(con):
    check_keys(con, 'sites', ['site_id'])
    expected = {f'PV{i:03}' for i in range(1, 254)}
    if {r[0] for r in con.execute('SELECT site_id FROM sites').fetchall()} != expected:
        raise ValueError('sites: expected PV001 through PV253 exactly once')
    check_finite(con, 'sites', 'capacity_kw', nonnegative=True)
    reject(con, 'SELECT count(*) FROM sites WHERE capacity_kw = 0', 'sites: zero capacity')


def prepare_prices(con, raw='raw_prices', name='prices'):
    check_keys(con, raw, ['ts'])
    check_time(con, raw)
    check_finite(con, raw, 'smp', nonnegative=True)
    if con.execute(f'SELECT count(*) FROM {raw}').fetchone()[0] == 0:
        raise ValueError('prices: empty time axis')
    k = "CASE WHEN date_part('hour', ts) BETWEEN 11 AND 15 THEN 3 ELSE 1 END"
    columns = [r[0] for r in con.execute(f'DESCRIBE {raw}').fetchall()]
    if 'k' in columns:
        reject(con, f'SELECT count(*) FROM {raw} WHERE k IS DISTINCT FROM ({k})',
               'prices: k disagrees with KST hour')
    con.execute(f'CREATE OR REPLACE TEMP VIEW {name} AS SELECT ts, smp, {k} AS k FROM {raw}')


def export(con, query, path):
    con.execute(f'COPY ({query}) TO {literal(Path(path).as_posix())} '
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
