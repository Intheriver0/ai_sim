"""Read-only local audit of the delivered OIBC files; export aggregate metadata only."""
from pathlib import Path
import hashlib
import json
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "OIBC_2026_DATA/handover 2/01_participant_dataset"
OUT = ROOT / "data/dataset_audit.json"
con = duckdb.connect()
con.execute("SET memory_limit='1GB'")
con.execute("SET threads=2")
con.execute("SET TimeZone='Asia/Seoul'")


def query(sql):
    result = con.execute(sql)
    names = [item[0] for item in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


files = {}
for path in sorted(DATA.rglob("*")):
    if path.suffix not in {".csv", ".parquet"} or path.name.startswith("."):
        continue
    rel = path.relative_to(DATA).as_posix()
    name = rel.replace("/", "_").replace(".", "_")
    source = path.as_posix().replace("'", "''")
    reader = f"read_parquet('{source}')" if path.suffix == ".parquet" else f"read_csv_auto('{source}', header=true)"
    con.execute(f'CREATE VIEW "{name}" AS SELECT * FROM {reader}')
    schema = query(f'DESCRIBE "{name}"')
    cols = {row['column_name']: row['column_type'] for row in schema}
    keys = [c for c in ("site_id", "wa_station", "wb_station", "round", "ts") if c in cols]
    if rel == "sites.csv":
        keys = ["site_id"]
    key_sql = ', '.join(f'"{c}"' for c in keys)
    summary = query(f'SELECT count(*) AS rows, count(*) - count(DISTINCT ({key_sql})) AS duplicate_keys FROM "{name}"')[0]
    if "ts" in cols:
        summary.update(query(f'SELECT min(ts) AS start, max(ts) AS end, count(DISTINCT ts) AS unique_timestamps, count(*) FILTER (WHERE date_part(\'minute\', ts) != 0 OR date_part(\'second\', ts) != 0) AS off_hour_rows FROM "{name}"')[0])
    if "round" in cols:
        summary["rounds"] = query(f'SELECT round, count(*) AS rows FROM "{name}" GROUP BY round ORDER BY round')
    for col in ("site_id", "wa_station", "wb_station"):
        if col in cols:
            summary[f"unique_{col}"] = query(f'SELECT count(DISTINCT "{col}") AS n FROM "{name}"')[0]['n']
    stats = []
    for col, typ in cols.items():
        numeric = typ in {"DOUBLE", "FLOAT", "BIGINT", "INTEGER", "SMALLINT", "TINYINT"}
        expr = f'count(*) FILTER (WHERE "{col}" IS NULL) AS nulls'
        if numeric:
            expr += f', count(*) FILTER (WHERE isnan("{col}")) AS nans, count(*) FILTER (WHERE isinf("{col}")) AS infinities, min("{col}") AS min, max("{col}") AS max'
        stats.append({"column": col, "type": typ, **query(f'SELECT {expr} FROM "{name}"')[0]})
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    files[rel] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest(), "keys": keys, **summary, "columns": stats}
    print(f"Checked {rel}: {summary['rows']:,} rows", flush=True)

checks = {}
checks['sites'] = query('SELECT count(*) AS rows, sum(capacity_kw) AS capacity_kw, count(*) FILTER (WHERE capacity_kw <= 0) AS invalid_capacity, median(wa_dist) AS median_wa_dist_m, max(wa_dist) AS max_wa_dist_m, median(wb_dist) AS median_wb_dist_m, max(wb_dist) AS max_wb_dist_m FROM sites_csv')[0]
checks['reference_groups'] = query('SELECT current_group, count(*) AS sites, sum(capacity_kw) AS capacity_kw FROM sites_csv GROUP BY current_group ORDER BY current_group')
checks['site_id_set_exact'] = {r[0] for r in con.execute('SELECT site_id FROM sites_csv').fetchall()} == {f'PV{i:03}' for i in range(1, 254)}
for name in ('smp_csv', 'train_smp_csv'):
    checks[name] = query(f'SELECT count(*) FILTER (WHERE smp < 0) AS negative_prices, count(*) FILTER (WHERE smp = 0) AS zero_prices, count(*) FILTER (WHERE gap != INTERVAL \'1 hour\') AS non_hourly_gaps FROM (SELECT *, ts - lag(ts) OVER (ORDER BY ts) AS gap FROM {name})')[0]
checks['k_mismatches'] = query("SELECT count(*) AS n FROM smp_csv WHERE k != CASE WHEN date_part('hour', ts) BETWEEN 11 AND 15 THEN 3 ELSE 1 END")[0]['n']
checks['k_counts'] = query('SELECT k, count(*) AS rows FROM smp_csv GROUP BY k ORDER BY k')
for provider in ('wa', 'wb'):
    station = provider + '_station'
    checks[f'{provider}_joins'] = {}
    for split in ('train', 'test'):
        view = f'{split}_forecast_{provider}_parquet'
        checks[f'{provider}_joins'][split] = query(f'SELECT count(*) AS unknown_station_rows FROM {view} f ANTI JOIN (SELECT DISTINCT {station} FROM sites_csv) s USING ({station})')[0]
    view = f'test_forecast_{provider}_parquet'
    grid = f'(SELECT s.{station}, r.round, t.ts FROM (SELECT DISTINCT {station} FROM sites_csv) s CROSS JOIN (VALUES (10), (17)) r(round) CROSS JOIN smp_csv t)'
    con.execute(f'CREATE VIEW missing_{provider} AS SELECT g.* FROM {grid} g ANTI JOIN {view} f USING ({station}, round, ts)')
    checks[f'{provider}_missing_test'] = query(f'SELECT count(*) AS missing_station_round_hours, count(DISTINCT {station}) AS stations, count(DISTINCT ts) AS timestamps, min(ts) AS start, max(ts) AS end FROM missing_{provider}')[0]
    checks[f'{provider}_missing_test']['by_round'] = query(f'SELECT round, count(*) AS rows FROM missing_{provider} GROUP BY round ORDER BY round')
    checks[f'{provider}_missing_test']['affected_site_round_hours'] = query(f'SELECT count(*) AS n FROM missing_{provider} JOIN sites_csv USING ({station})')[0]['n']
    checks[f'{provider}_missing_test']['both_rounds_missing_station_hours'] = query(f'SELECT count(*) AS n FROM (SELECT {station}, ts FROM missing_{provider} GROUP BY {station}, ts HAVING count(*)=2)')[0]['n']
    checks[f'{provider}_extra_test_rows'] = query(f'SELECT count(*) AS n FROM {view} f ANTI JOIN {grid} g USING ({station}, round, ts)')[0]['n']
for split in ('train', 'test'):
    checks[f'{split}_wa_equal_temperatures'] = query(f'SELECT count(*) AS unequal_rows FROM {split}_forecast_wa_parquet WHERE wa_temp IS DISTINCT FROM wa_temp_max OR wa_temp IS DISTINCT FROM wa_temp_min')[0]
checks['generation'] = query('SELECT min(n) AS min_rows_per_site, median(n) AS median_rows_per_site, max(n) AS max_rows_per_site, min(first_ts) AS earliest_site_start, max(first_ts) AS latest_site_start FROM (SELECT site_id, count(*) AS n, min(ts) AS first_ts FROM train_generation_parquet GROUP BY site_id)')[0]
checks['generation']['unknown_site_rows'] = query('SELECT count(*) AS n FROM train_generation_parquet ANTI JOIN sites_csv USING (site_id)')[0]['n']
checks['generation']['outside_train_smp_rows'] = query('SELECT count(*) AS n FROM train_generation_parquet ANTI JOIN train_smp_csv USING (ts)')[0]['n']
checks['generation']['full_grid_missing_rows'] = 253 * files['train/smp.csv']['rows'] - files['train/generation.parquet']['rows']
checks['generation']['negative_energy_rows'] = query('SELECT count(*) AS n FROM train_generation_parquet WHERE energy < 0')[0]['n']
checks['generation']['complete_253_site_hours'] = query('SELECT count(*) AS n FROM (SELECT ts FROM train_generation_parquet GROUP BY ts HAVING count(DISTINCT site_id)=253)')[0]['n']
checks['generation']['complete_hours_range'] = query("SELECT min(ts) AS start, max(ts) AS end, count(*) FILTER (WHERE gap != INTERVAL '1 hour') AS non_hourly_gaps FROM (SELECT ts, ts-lag(ts) OVER (ORDER BY ts) AS gap FROM (SELECT ts FROM train_generation_parquet GROUP BY ts HAVING count(DISTINCT site_id)=253))")[0]
checks['observed_wa_duplicates'] = query('SELECT count(*) AS duplicated_key_groups, sum(n-1) AS excess_rows, max(n) AS largest_group FROM (SELECT wa_station, ts, count(*) AS n FROM train_observed_wa_parquet GROUP BY wa_station, ts HAVING count(*) > 1)')[0]
checks['observed_wa_duplicates']['exact_duplicate_excess_rows'] = query('SELECT (SELECT count(*) FROM train_observed_wa_parquet)-count(*) AS n FROM (SELECT DISTINCT * FROM train_observed_wa_parquet)')[0]['n']
checks['wb_off_hour'] = query("SELECT date_part('minute',ts) AS minute, date_part('second',ts) AS second, count(*) AS rows, count(DISTINCT ts) AS timestamps, min(ts) AS start, max(ts) AS end FROM train_forecast_wb_parquet WHERE date_part('minute',ts) != 0 OR date_part('second',ts) != 0 GROUP BY ALL")
for provider in ('wa', 'wb'):
    station = provider + '_station'
    for kind in ('forecast', 'observed'):
        view = f'train_{kind}_{provider}_parquet'
        rounds = 'CROSS JOIN (VALUES (10), (17)) r(round)' if kind == 'forecast' else ''
        round_select = ', r.round' if kind == 'forecast' else ''
        keys = f'{station}, ts' + (', round' if kind == 'forecast' else '')
        grid = f'(SELECT s.{station}, t.ts{round_select} FROM (SELECT DISTINCT {station} FROM sites_csv) s CROSS JOIN train_smp_csv t {rounds})'
        checks[f'train_{kind}_{provider}_coverage'] = {
            'missing_grid_rows': query(f'SELECT count(*) AS n FROM {grid} g ANTI JOIN {view} f USING ({keys})')[0]['n'],
            'outside_grid_rows': query(f'SELECT count(*) AS n FROM {view} f ANTI JOIN {grid} g USING ({keys})')[0]['n'],
        }
checks['wb_precip_missing_union'] = query('SELECT count(*) AS n FROM train_forecast_wb_parquet WHERE wb_precip_prob IS NULL OR wb_rain_prob IS NULL OR wb_snow_prob IS NULL OR wb_ice_prob IS NULL OR wb_total_liq IS NULL OR wb_rain IS NULL OR wb_snow IS NULL OR wb_ice IS NULL')[0]['n']
checks['observed_anomalies'] = {}
for col, predicate, provider in [('wb_cloud', '>100', 'wb'), ('wb_uv_idx', '>15', 'wb'), ('wb_vis', '>50', 'wb'), ('wb_wind_speed', '>200', 'wb'), ('wb_pressure', '<900', 'wb'), ('wa_wind_speed', '>50', 'wa')]:
    checks['observed_anomalies'][col] = query(f'SELECT count(*) FILTER (WHERE {col}{predicate}) AS rows, 100.0 * count(*) FILTER (WHERE {col}{predicate})/count(*) AS percent FROM train_observed_{provider}_parquet')[0]

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({'checked_on': '2026-10-27', 'duckdb_version': duckdb.__version__, 'source': str(DATA.relative_to(ROOT)), 'files': files, 'checks': checks}, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
print('Saved aggregate audit:', OUT.relative_to(ROOT))
