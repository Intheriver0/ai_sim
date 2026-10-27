"""Mask observed-weather anomalies listed in the participant README."""
import argparse
import json
from pathlib import Path

from pipeline_common import connect, export, read_view


# Thresholds apply to observations in their ORIGINAL units, not forecasts.
RULES = {
    'wa': {'wa_wind_speed': '> 50'},
    'wb': {'wb_cloud': '> 100', 'wb_uv_idx': '> 15', 'wb_vis': '> 50',
           'wb_wind_speed': '> 200', 'wb_pressure': '< 900'},
}


def clean(con, raw, name, provider):
    columns = [r[0] for r in con.execute(f'DESCRIBE {raw}').fetchall()]
    expressions, flags, counts = [], [], {}
    for col in columns:
        if col in (f'{provider}_station', 'ts'):
            expressions.append(f'"{col}"')
            continue
        condition = f'NOT isfinite("{col}")'
        if col in RULES[provider]:
            condition += f' OR "{col}" {RULES[provider][col]}'
        condition = f'coalesce(({condition}), false)'
        expressions.append(f'CASE WHEN {condition} THEN NULL ELSE "{col}" END AS "{col}"')
        flags.append(f'{condition} AS "{col}_invalid"')
        counts[col] = con.execute(
            f'SELECT count(*) FROM {raw} WHERE {condition}').fetchone()[0]
    # Preserve duplicates for audit, but prevent their use as teacher labels.
    station = f'{provider}_station'
    con.execute(f'''CREATE OR REPLACE TEMP VIEW {name} AS SELECT
        {', '.join(expressions + flags)},
        count(*) OVER (PARTITION BY {station}, ts) > 1 AS duplicate_key
        FROM {raw}''')
    return counts


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError('Output must be outside the original dataset directory')
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'source': str(data_dir), 'rules_original_units': RULES,
              'policy': 'mask invalid cells; preserve rows and units; exclude duplicate keys from teacher use',
              'outputs': {}}
    with connect() as con:
        for provider in RULES:
            read_view(con, 'raw', data_dir / 'train' / f'observed_{provider}.parquet')
            counts = clean(con, 'raw', 'cleaned', provider)
            filename = f'observed_{provider}.parquet'
            export(con, 'SELECT * FROM cleaned', output_dir / filename)
            read_view(con, 'saved', output_dir / filename)
            rows, duplicates = con.execute(
                'SELECT count(*), count(*) FILTER (WHERE duplicate_key) FROM saved').fetchone()
            if rows != con.execute('SELECT count(*) FROM raw').fetchone()[0]:
                raise ValueError('Observed cleaning changed row count')
            for col in RULES[provider]:
                if con.execute(f'SELECT count(*) FROM saved WHERE "{col}" {RULES[provider][col]}').fetchone()[0]:
                    raise ValueError(f'Remaining anomaly: {col}')
            report['outputs'][filename] = {'rows': rows, 'duplicate_key_rows': duplicates,
                                           'masked_cells': counts}
    report['status'] = 'complete'
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
