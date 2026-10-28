"""Frozen LightGBM baseline on seasonal v2 outer months (no tuning)."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from fixed_model import FEATURES, PARAMS, WEATHER, fit, predict
from gpu_training import check_cuda
from pipeline_common import check_keys, connect, export, read_view, reject
from scoring import score
from seasonal_splits import load_plan, window_views
from time_splits import attach_data, digest, validate_inputs


def evaluate(con, predictions):
    con.register('site_predictions', predictions)
    check_keys(con, 'site_predictions', ['site_id', 'ts'])
    reject(con, '''SELECT count(*) FROM (
        (SELECT site_id,ts FROM diagnostic_actual EXCEPT SELECT site_id,ts FROM site_predictions)
        UNION ALL (SELECT site_id,ts FROM site_predictions EXCEPT SELECT site_id,ts FROM diagnostic_actual))''',
        'Prediction/actual keys differ')
    diagnostic = con.execute('''SELECT avg(nmae),sum(n),count(*) FROM (
        SELECT site_id,avg(abs(forecast_kwh-energy)/capacity_kw) AS nmae,count(*) AS n
        FROM site_predictions JOIN diagnostic_actual USING(site_id,ts)
        JOIN sites USING(site_id) GROUP BY site_id)''').fetchall()[0]
    hours = con.execute('SELECT count(*) FROM settlement_prices').fetchall()[0][0]
    settlement = None
    if hours:
        # Use an independent connection: scoring's prices must not replace the
        # training price view used by the next seasonal window.
        with connect() as scorer:
            for table, query in {
                'sites': 'SELECT * FROM sites',
                'composition': 'SELECT * FROM composition',
                'prices': 'SELECT * FROM settlement_prices',
                'actual': 'SELECT * FROM settlement_actual',
                'forecast': '''SELECT "group",ts,sum(forecast_kwh) AS forecast_kwh
                    FROM site_predictions JOIN composition USING(site_id)
                    JOIN settlement_prices USING(ts) GROUP BY "group",ts''',
            }.items():
                scorer.register(table, con.execute(query).df())
            settlement = score(scorer)
    cold = con.execute('''SELECT count(*) FROM (SELECT DISTINCT site_id FROM diagnostic_actual
        EXCEPT SELECT DISTINCT site_id FROM training_targets)''').fetchall()[0][0]
    return {'site_equal_nmae': diagnostic[0], 'observed_site_hours': diagnostic[1],
            'observed_sites': diagnostic[2], 'cold_start_sites': cold,
            'complete_253_hours': hours, 'settlement': settlement}


def run(data_dir, config, output_dir, fold_ids=None, threads=2):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    if threads < 1:
        raise ValueError('threads must be positive')
    _, windows = load_plan(config)
    if fold_ids:
        if set(fold_ids) - {w['id'] for w in windows}:
            raise ValueError('Unknown outer fold')
        windows = [w for w in windows if w['id'] in fold_ids]
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError('Output must be outside preprocessing inputs')
    gpu = check_cuda(threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', '-C', str(root),
                          'rev-parse', 'HEAD'], capture_output=True, text=True)
    report = {'status': 'running', 'procedure': 'fixed 600 trees; no selection or early stopping; two-round mean; reference composition',
              'params': PARAMS, 'features': FEATURES, 'threads': threads, 'gpu': gpu,
              'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                           'numpy': np.__version__, 'pandas': pd.__version__},
              'git_head': git.stdout.strip(),
              'source_sha256': {p.name: digest(p) for p in (root/'scripts').glob('*.py')},
              'config_sha256': digest(config),
              'input_sha256': {p.name: digest(p) for p in data_dir.glob('*.parquet')},
              'windows': [{k: v for k, v in w.items() if k != 'inner'} for w in windows],
              'folds': []}

    def save():
        (output_dir/'report.json').write_text(json.dumps(report, default=str, indent=2)+'\n', encoding='utf-8')

    save()  # Freeze procedure before any outer score is exposed.
    with connect() as con:
        attach_data(con, data_dir)
        validate_inputs(con)
        read_view(con, 'composition', data_dir/'reference_composition.parquet')
        for window in windows:
            started = time.monotonic()
            destination = output_dir/window['id']
            destination.mkdir()
            window_views(con, window, allow_outer_evaluation=True)
            per_round, training_rows = [], {}
            for r in (10, 17):
                columns = ', '.join(['site_id', 'ts', 'capacity_kw', 'lat', 'lon'] + WEATHER)
                frame = con.execute(f'''SELECT {columns},target_kwh FROM training_features
                    WHERE round={r} ORDER BY ts,site_id''').df()
                print(f'{window["id"]} round={r}: fitting {len(frame):,} rows', flush=True)
                model, daytime = fit(frame, threads)
                training_rows[str(r)] = {'eligible': len(frame), 'daytime': daytime}
                del frame
                model_path = destination/f'round{r}.txt'
                model.save_model(str(model_path))
                frame = con.execute(f'''SELECT {columns} FROM diagnostic_features
                    WHERE round={r} ORDER BY ts,site_id''').df()
                prediction = predict(model, frame, threads)
                restored = lgb.Booster(model_file=str(model_path))
                np.testing.assert_allclose(prediction.forecast_kwh.iloc[:1000],
                    predict(restored, frame.iloc[:1000], threads).forecast_kwh, rtol=1e-12)
                per_round.append(prediction.set_index(['site_id', 'ts']))
                del frame, model, restored
            if not per_round[0].index.equals(per_round[1].index):
                raise ValueError('Round keys differ')
            predictions = ((per_round[0] + per_round[1])/2).reset_index()
            metrics = evaluate(con, predictions)
            export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', destination/'site_predictions.parquet')
            metrics.update(id=window['id'], training_rows=training_rows, seconds=time.monotonic()-started)
            report['folds'].append(metrics)
            save()
            print(json.dumps(metrics), flush=True)
    report['all_12_months'] = len(windows) == 12
    report['month_equal_nmae'] = float(np.mean([f['site_equal_nmae'] for f in report['folds']]))
    report['status'] = 'complete'
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fold', action='append', help='Default: all twelve months; partial runs are labeled')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.data_dir, args.config, args.output_dir, args.fold, args.threads)


if __name__ == '__main__':
    main()
