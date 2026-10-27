"""Build local, un-imputed forecast tables without fitting a model."""
import argparse
import json
from pathlib import Path

from pipeline_common import (check_finite, check_keys, check_sites, check_time,
                             connect, export, prepare_prices, read_view, reject)
from scoring import validate_composition
from clean_observed import run as clean_observed


def normalize_forecast(con, raw, name, provider):
    station = f'{provider}_station'
    check_keys(con, raw, [station, 'round', 'ts'])
    check_time(con, raw, hourly=False)
    reject(con, f'SELECT count(*) FROM {raw} WHERE round NOT IN (10, 17)',
           f'{raw}: unknown forecast round')
    reject(con, f'''SELECT count(*) FROM {raw} ANTI JOIN
        (SELECT DISTINCT {station} FROM sites) s USING ({station})''',
        f'{raw}: unknown station')
    conversions = {
        'wa_wind_speed': ('wa_wind_speed_ms', 1),
        'wa_snow': ('wa_snow_mm', 1),
        'wb_wind_speed': ('wb_wind_speed_ms', 1 / 3.6),
        'wb_wind_gust_spd': ('wb_wind_gust_spd_ms', 1 / 3.6),
        'wb_snow': ('wb_snow_mm', 10),
    }
    columns = [r[0] for r in con.execute(f'DESCRIBE {raw}').fetchall()
               if r[0] not in (station, 'round', 'ts')]
    expressions, output_columns = [], []
    for col in columns:
        reject(con, f'SELECT count(*) FROM {raw} WHERE isinf("{col}")',
               f'{raw}: infinite {col}')
        alias, factor = conversions.get(col, (col, 1))
        expressions.append(f'CASE WHEN isnan("{col}") THEN NULL ELSE "{col}"::DOUBLE * {factor} END AS "{alias}"')
        output_columns.append(alias)
    con.execute(f'''CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT {station}, round, ts, {', '.join(expressions)}
        FROM {raw} WHERE ts = date_trunc('hour', ts)''')
    return output_columns


def joined_query(split, round_number, weather_columns):
    base = ('SELECT site_id, ts FROM generation' if split == 'train' else
            'SELECT site_id, ts FROM sites CROSS JOIN test_prices')
    weather = ', '.join(f'{p}."{col}"' for p in ('wa', 'wb') for col in weather_columns[p])
    # At midnight the interval belongs to the preceding target day.
    target_day = "date_trunc('day', b.ts - INTERVAL '1 hour')"
    return f'''SELECT b.site_id, b.ts, {round_number} AS round,
        CAST({target_day} AS DATE) AS target_day,
        {target_day} - INTERVAL '1 day' + INTERVAL '{round_number} hours' AS issued_at,
        s.capacity_kw, s.lat, s.lon, s.wa_station, s.wb_station, s.wa_dist, s.wb_dist,
        wa.ts IS NULL AS wa_missing, wb.ts IS NULL AS wb_missing, {weather}
        FROM ({base}) b JOIN sites s USING (site_id)
        LEFT JOIN {split}_wa wa ON s.wa_station=wa.wa_station AND b.ts=wa.ts AND wa.round={round_number}
        LEFT JOIN {split}_wb wb ON s.wb_station=wb.wb_station AND b.ts=wb.ts AND wb.round={round_number}'''


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError('Output must be outside the original dataset directory')
    # A fresh directory prevents accidental overwrite or mixing of different runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'duckdb_version': __import__('duckdb').__version__, 'source': str(data_dir),
              'policy': {'negative_energy': 'preserve raw; null training target; ineligible',
                         'off_hour_forecast': 'quarantine without rounding',
                         'missing_weather': 'retain null; no fitted imputation',
                         'missing_generation': 'do not create rows or zero-fill',
                         'observed_weather': 'cleaned separately; never joined to forecast features',
                         'prices': 'separate evaluation tables, not model features'},
              'inputs': {}, 'outputs': {}}
    with connect() as con:
        read_view(con, 'sites', data_dir / 'sites.csv')
        check_sites(con)
        for col in ('lat', 'lon', 'wa_dist', 'wb_dist'):
            check_finite(con, 'sites', col)
        reject(con, 'SELECT count(*) FROM sites WHERE wa_station IS NULL OR wb_station IS NULL',
               'sites: missing weather station')
        con.execute('''CREATE TEMP VIEW composition AS SELECT site_id,
            CASE current_group WHEN 'A' THEN 'G1' WHEN 'B' THEN 'G2' END AS "group" FROM sites''')
        report['reference_capacity_kw'] = validate_composition(con)
        export(con, 'SELECT * FROM sites', output_dir / 'sites.parquet')
        export(con, 'SELECT * FROM composition', output_dir / 'reference_composition.parquet')
        for split in ('train', 'test'):
            read_view(con, 'raw_prices', data_dir / ('train/smp.csv' if split == 'train' else 'smp.csv'))
            # Materialize because raw_prices will be replaced for the next split.
            prepare_prices(con, name='checked_prices')
            con.execute(f'CREATE TEMP TABLE {split}_prices AS SELECT * FROM checked_prices')
            reject(con, f'''SELECT count(*) FROM (SELECT ts-lag(ts) OVER (ORDER BY ts) AS gap
                FROM {split}_prices) WHERE gap != INTERVAL '1 hour' ''', f'{split}: price-axis gap')
            export(con, f'SELECT * FROM {split}_prices ORDER BY ts', output_dir / f'{split}_prices.parquet')
        read_view(con, 'generation', data_dir / 'train/generation.parquet')
        check_keys(con, 'generation', ['site_id', 'ts'])
        check_time(con, 'generation')
        check_finite(con, 'generation', 'energy')
        reject(con, 'SELECT count(*) FROM generation ANTI JOIN sites USING (site_id)',
               'generation: unknown site')
        reject(con, 'SELECT count(*) FROM generation ANTI JOIN train_prices USING (ts)',
               'generation: outside training time axis')
        report['generation_rows'], report['negative_energy_rows'] = con.execute(
            'SELECT count(*), count(*) FILTER (WHERE energy<0) FROM generation').fetchone()
        export(con, '''SELECT site_id, ts, energy AS energy_raw_kwh,
            CASE WHEN energy >= 0 THEN energy END AS target_kwh,
            energy < 0 AS negative_energy, energy >= 0 AS training_eligible
            FROM generation''', output_dir / 'train_targets.parquet')
        report['complete_actual_hours'] = con.execute('''SELECT count(*) FROM
            (SELECT ts FROM generation GROUP BY ts HAVING count(*)=253)''').fetchone()[0]
        for split in ('train', 'test'):
            weather_columns = {}
            for provider in ('wa', 'wb'):
                raw, name = f'raw_{split}_{provider}', f'{split}_{provider}'
                read_view(con, raw, data_dir / split / f'forecast_{provider}.parquet')
                weather_columns[provider] = normalize_forecast(con, raw, name, provider)
                n, off_hour = con.execute(f'''SELECT count(*), count(*) FILTER
                    (WHERE ts != date_trunc('hour', ts)) FROM {raw}''').fetchone()
                report['inputs'][name] = {'rows': n, 'quarantined_off_hour_rows': off_hour}
                if off_hour:
                    export(con, f"SELECT * FROM {raw} WHERE ts != date_trunc('hour', ts)",
                           output_dir / f'{name}_off_hour.parquet')
                reject(con, f'SELECT count(*) FROM {name} ANTI JOIN {split}_prices USING (ts)',
                       f'{name}: forecast outside price time axis')
            for round_number in (10, 17):
                filename = f'{split}_round{round_number}.parquet'
                print(f'Writing {filename}', flush=True)
                export(con, joined_query(split, round_number, weather_columns), output_dir / filename)
                read_view(con, 'result', output_dir / filename)
                check_keys(con, 'result', ['site_id', 'ts', 'round'])
                rows, missing_wa, missing_wb = con.execute('''SELECT count(*),
                    count(*) FILTER (WHERE wa_missing), count(*) FILTER (WHERE wb_missing)
                    FROM result''').fetchone()
                expected = (report['generation_rows'] if split == 'train' else
                            253 * con.execute('SELECT count(*) FROM test_prices').fetchone()[0])
                if rows != expected:
                    raise ValueError(f'{filename}: join changed row count')
                columns = weather_columns['wa'] + weather_columns['wb']
                nulls = con.execute('SELECT ' + ', '.join(
                    f'count(*) FILTER (WHERE "{col}" IS NULL)' for col in columns) + ' FROM result').fetchone()
                report['outputs'][filename] = {'rows': rows, 'missing_wa_rows': missing_wa,
                                              'missing_wb_rows': missing_wb,
                                              'weather_nulls': dict(zip(columns, nulls))}
        report['observed_weather'] = clean_observed(data_dir, output_dir / 'observed_clean')
        report['status'] = 'complete'
        (output_dir / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory outside source data')
    args = parser.parse_args()
    report = run(args.data_dir, args.output_dir)
    print(json.dumps({k: report[k] for k in ('generation_rows', 'negative_energy_rows',
                                           'complete_actual_hours', 'status')}, indent=2))


if __name__ == '__main__':
    main()
