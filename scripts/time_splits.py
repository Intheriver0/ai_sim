"""Frozen chronological folds and guarded data views for later model runners."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import pytz

from pipeline_common import (check_finite, check_keys, check_sites, check_time,
                             connect, export, literal, prepare_prices, read_view, reject)


def stamp(value):
    return 'TIMESTAMPTZ ' + literal(value.isoformat())


def load_plan(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('status') == 'rejected':
        raise ValueError('Rejected split configuration: ' + config['reason'])
    if config['timezone'] != 'Asia/Seoul':
        raise ValueError('The settlement calendar must use Asia/Seoul')
    zone = pytz.timezone(config['timezone'])
    training_start = datetime.fromisoformat(config['training_start'])
    if training_start.tzinfo is None:
        raise ValueError('training_start requires timezone')
    lag = config['generation_availability_lag_hours']
    before, end_before = config['early_stop_start_days_before'], config['early_stop_end_days_before']
    if lag < 0 or not before > end_before >= 1 or config['minimum_training_rows_per_site'] < 1:
        raise ValueError('Invalid lag, early-stop window or minimum training rows')
    folds = []
    for item in config['folds']:
        start = zone.localize(datetime.strptime(item['start_day'], '%Y-%m-%d'))
        end = zone.localize(datetime.strptime(item['end_day_exclusive'], '%Y-%m-%d'))
        stop_start, stop_end = start - timedelta(days=before), start - timedelta(days=end_before)
        origin = start - timedelta(days=1) + timedelta(hours=17)
        fit_origin = stop_start - timedelta(days=1) + timedelta(hours=17)
        fold = {**item, 'start': start, 'end': end, 'early_stop_start': stop_start,
                'early_stop_end': stop_end, 'forecast_origin': origin,
                'fit_cutoff': fit_origin - timedelta(hours=lag),
                'refit_cutoff': origin - timedelta(hours=lag), 'training_start': training_start}
        if not training_start <= fold['fit_cutoff'] < stop_start < stop_end <= fold['refit_cutoff'] < start < end:
            raise ValueError(f"{item['id']}: invalid chronology or label availability")
        folds.append(fold)
    expected = ['model_selection', 'model_selection', 'blend_selection',
                'composition_selection', 'final_holdout']
    if [f['purpose'] for f in folds] != expected or len({f['id'] for f in folds}) != len(folds):
        raise ValueError('Expected two tuning folds, then blend, composition and final holdout')
    if any(a['end'] > b['start'] for a, b in zip(folds, folds[1:])):
        raise ValueError('Evaluation blocks overlap or are out of order')
    return config, folds


def interval(start, end):
    # Right-closed hourly energy: days [start, end) correspond to timestamps (start, end].
    return f'ts > {stamp(start)} AND ts <= {stamp(end)}'


def training_filter(fold, role):
    cutoff = fold['fit_cutoff'] if role == 'fit' else fold['refit_cutoff']
    return f'ts >= {stamp(fold["training_start"])} AND ts <= {stamp(cutoff)} AND training_eligible'


def attach_data(con, data_dir):
    data_dir = Path(data_dir)
    report = json.loads((data_dir / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete':
        raise ValueError('Preprocessing must have completed successfully')
    for table, filename in [('sites', 'sites.parquet'), ('targets', 'train_targets.parquet'),
                            ('raw_prices', 'train_prices.parquet'),
                            ('features10', 'train_round10.parquet'), ('features17', 'train_round17.parquet')]:
        read_view(con, table, data_dir / filename)
    prepare_prices(con)


def fold_views(con, fold, allow_final_holdout=False):
    """Create views after attach_data; final evaluation requires explicit opt-in.

    Fit transforms/model on fit_features, use stop_features for early stopping,
    then refit on refit_features with the chosen iteration count. Fold evaluation
    targets never participate in fitting. All feature views contain both rounds.
    """
    if fold['purpose'] == 'final_holdout' and not allow_final_holdout:
        raise ValueError('Final holdout is locked; freeze all choices before explicitly opening it')
    filters = {'fit': training_filter(fold, 'fit'),
               'stop': interval(fold['early_stop_start'], fold['early_stop_end']) + ' AND training_eligible',
               'refit': training_filter(fold, 'refit')}
    for role, where in filters.items():
        con.execute(f'''CREATE OR REPLACE TEMP VIEW {role}_targets AS
            SELECT site_id, ts, target_kwh FROM targets WHERE {where}''')
        con.execute(f'''CREATE OR REPLACE TEMP VIEW {role}_features AS
            SELECT f.*, t.target_kwh FROM
            (SELECT * FROM features10 UNION ALL SELECT * FROM features17) f
            JOIN {role}_targets t USING (site_id, ts)''')
    where = interval(fold['start'], fold['end'])
    con.execute(f'''CREATE OR REPLACE TEMP VIEW evaluation_features AS
        SELECT * FROM features10 WHERE {where} UNION ALL
        SELECT * FROM features17 WHERE {where}''')
    con.execute(f'''CREATE OR REPLACE TEMP VIEW evaluation_actual AS
        SELECT site_id, ts, energy_raw_kwh AS energy FROM targets WHERE {where}''')
    con.execute(f'CREATE OR REPLACE TEMP VIEW evaluation_prices AS SELECT * FROM prices WHERE {where}')


def validate_inputs(con):
    check_sites(con)
    check_keys(con, 'targets', ['site_id', 'ts'])
    check_time(con, 'targets')
    check_finite(con, 'targets', 'energy_raw_kwh')
    reject(con, '''SELECT count(*) FROM targets WHERE training_eligible IS NULL
        OR training_eligible IS DISTINCT FROM (energy_raw_kwh >= 0)
        OR target_kwh IS DISTINCT FROM CASE WHEN energy_raw_kwh >= 0 THEN energy_raw_kwh END''',
        'Inconsistent training target policy')
    reject(con, 'SELECT count(*) FROM targets ANTI JOIN sites USING(site_id)', 'Unknown target site')
    reject(con, 'SELECT count(*) FROM targets ANTI JOIN prices USING(ts)', 'Target outside price axis')
    for r in (10, 17):
        table = f'features{r}'
        check_keys(con, table, ['site_id', 'ts'])
        check_time(con, table)
        reject(con, f'''SELECT count(*) FROM {table} WHERE round IS DISTINCT FROM {r}
            OR target_day IS DISTINCT FROM CAST(ts-INTERVAL '1 hour' AS DATE)
            OR issued_at IS DISTINCT FROM (date_trunc('day',ts-INTERVAL '1 hour')
                -INTERVAL '1 day'+INTERVAL '{r} hours')''', f'{table}: invalid round or issue time')
        reject(con, f'''SELECT count(*) FROM (
            (SELECT site_id,ts FROM targets EXCEPT SELECT site_id,ts FROM {table}) UNION ALL
            (SELECT site_id,ts FROM {table} EXCEPT SELECT site_id,ts FROM targets))''',
            f'{table}: feature and target keys differ')


def audit_fold(con, fold, minimum_rows):
    site_counts = con.execute(f'''SELECT s.site_id,
        count(t.ts) FILTER(WHERE {training_filter(fold, 'fit')}) AS fit_rows,
        count(t.ts) FILTER(WHERE {training_filter(fold, 'refit')}) AS refit_rows
        FROM sites s LEFT JOIN targets t USING(site_id) GROUP BY s.site_id ORDER BY s.site_id''').fetchall()
    if min(r[1] for r in site_counts) < minimum_rows:
        raise ValueError(f'{fold["id"]}: insufficient per-site training history')
    hours = int((fold['end'] - fold['start']).total_seconds() / 3600)
    for start, end in [(fold['start'], fold['end']), (fold['early_stop_start'], fold['early_stop_end'])]:
        where = interval(start, end)
        expected_hours = int((end - start).total_seconds() / 3600)
        price_hours = con.execute(f'SELECT count(*) FROM prices WHERE {where}').fetchone()[0]
        if price_hours != expected_hours:
            raise ValueError(f'{fold["id"]}: incomplete price coverage')
        reject(con, f'''SELECT count(*) FROM (SELECT site_id,ts FROM sites CROSS JOIN prices WHERE {where}) e
            ANTI JOIN targets USING(site_id,ts)''', f'{fold["id"]}: incomplete actual coverage')
    return {**fold, 'hours': hours, 'evaluation_site_rows': hours * 253,
            'minimum_fit_rows': min(r[1] for r in site_counts),
            'minimum_refit_rows': min(r[2] for r in site_counts),
            'site_training_counts': [{'site_id': s, 'fit_rows': a, 'refit_rows': b} for s, a, b in site_counts],
            'holdout_locked': fold['purpose'] == 'final_holdout'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(data_dir, config_path, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    config, folds = load_plan(config_path)
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError('Split outputs must be outside preprocessing inputs')
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with connect() as con:
        attach_data(con, data_dir)
        validate_inputs(con)
        audited = [audit_fold(con, fold, config['minimum_training_rows_per_site']) for fold in folds]
        output_dir.mkdir(parents=True)
        # Calendar only, no held-out energy or scores exported.
        calendar = ' UNION ALL '.join(
            f'''SELECT {literal(f['id'])} AS fold_id, {literal(f['purpose'])} AS purpose, ts,
                CAST(ts-INTERVAL '1 hour' AS DATE) AS target_day
                FROM prices WHERE {interval(f['start'], f['end'])}''' for f in folds)
        export(con, calendar, output_dir / 'evaluation_calendar.parquet')
        manifest = {'status': 'complete', 'config': config, 'config_sha256': digest(config_path),
                    'input_sha256': {p.name: digest(p) for p in [data_dir / n for n in
                        ('report.json','sites.parquet','train_targets.parquet','train_prices.parquet',
                         'train_round10.parquet','train_round17.parquet')]},
                    'folds': audited, 'final_holdout_scores_computed': False}
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, default=str, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/time_splits_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_dir, args.config, args.output_dir)
    print(json.dumps([{k: f[k] for k in ('id','hours','minimum_fit_rows','minimum_refit_rows','holdout_locked')}
                      for f in result['folds']], indent=2))


if __name__ == '__main__':
    main()
