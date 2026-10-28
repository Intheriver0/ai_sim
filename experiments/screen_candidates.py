"""CUDA-only exploratory comparison; optionally wait for a main run to finish."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

import lightgbm as lgb
import numpy as np
import pandas as pd

from fixed_model import FEATURES, PARAMS, WEATHER, features
from gpu_training import check_cuda, require_cuda_params
from pipeline_common import connect, export, read_view
from seasonal_splits import load_plan, window_views
from time_splits import attach_data, digest, stamp, validate_inputs
from train_fixed import evaluate
from train_grouped import save_json
from train_tuned import blend, checked_result


CANDIDATES = [
    {'name': 'baseline', 'params': {}, 'reason': 'Fixed CUDA reference'},
    {'name': 'l1_loss', 'params': {'objective': 'regression_l1'}, 'reason': 'Absolute-error training'},
    {'name': 'simple_tree', 'params': {'num_leaves': 31, 'min_child_samples': 500},
     'reason': 'Lower tree complexity'},
    {'name': 'recent_2years', 'params': {}, 'years': 2, 'reason': 'Recent history only'},
    {'name': 'round17_only', 'params': {}, 'rounds': [17], 'reason': 'Latest forecast round only'},
    {'name': 'extra_trees', 'params': {'extra_trees': True}, 'reason': 'Random split thresholds'},
    {'name': 'no_direct_coordinates', 'params': {}, 'drop': ['lat', 'lon'],
     'reason': 'Remove direct coordinates, keep solar geometry'},
    {'name': 'tiny_tree', 'params': {'num_leaves': 7}, 'reason': 'Very low tree complexity'},
]


def wait_for_run(path):
    path = Path(path).resolve()
    manifest_path = path/'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Existing run manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('smoke') or len(manifest.get('folds', [])) != 12:
        raise ValueError('Expected a full twelve-month main run, not smoke or a partial-fold run')
    print(f'Waiting for all twelve months to complete: {path}', flush=True)
    while True:
        summary_path = path/'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('status') == 'complete' and summary.get('all_12_months') is True and not summary.get('smoke'):
                print('Main run complete. Starting the separate candidate experiment.', flush=True)
                return
            print(f'Main run status={summary.get("status")}; waiting (no GPU work yet).', flush=True)
        else:
            print('Main run has no completed outer-month summary yet; waiting.', flush=True)
        time.sleep(60)


def training_condition(window, candidate):
    if 'years' not in candidate:
        return 'TRUE'
    # Fixed lookback relative to this historical block, never relative to today.
    start = pd.Timestamp(window['training_cutoff']) - pd.DateOffset(years=candidate['years'])
    return f'ts >= {stamp(start.to_pydatetime())}'


def candidate_features(frame, candidate):
    full = features(frame)
    return full.drop(columns=candidate.get('drop', [])), full.elev.to_numpy() > 0


def predict_candidate(model, frame, candidate, threads):
    x, day = candidate_features(frame, candidate)
    values = np.where(day, np.maximum(model.predict(x, num_threads=threads), 0), 0)
    result = frame[['site_id', 'ts']].copy()
    result['forecast_kwh'] = values * frame.capacity_kw.to_numpy(dtype='float32')
    if not np.isfinite(result.forecast_kwh).all():
        raise ValueError('Non-finite candidate predictions')
    return result


def run_block(con, window, candidate, destination, threads, trees):
    effective = dict(window)
    if 'years' in candidate:
        recent = pd.Timestamp(window['training_cutoff'])-pd.DateOffset(years=candidate['years'])
        effective['training_start'] = max(window['training_start'], recent.to_pydatetime())
    window_views(con, effective)  # Outer evaluation stays locked in this screening experiment.
    previous = checked_result(destination)
    if previous is not None:
        read_view(con, 'candidate_predictions', destination/'predictions.parquet')
        print(f'Skip complete {window["id"]}/{candidate["name"]}', flush=True)
        return con.execute('SELECT * FROM candidate_predictions').df(), previous
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    params = dict(PARAMS, **candidate['params'])
    params['n_estimators'] = trees
    require_cuda_params(params)
    rounds, counts, artifacts = [], [], []
    columns = ', '.join(['site_id', 'ts', 'capacity_kw', 'lat', 'lon'] + WEATHER)
    condition = training_condition(window, candidate)
    for round_id in candidate.get('rounds', [10, 17]):
        print(f'{window["id"]}/{candidate["name"]}: CUDA round={round_id}', flush=True)
        frame = con.execute(f'SELECT {columns},target_kwh FROM training_features '
                            f'WHERE round={round_id} AND {condition} ORDER BY ts,site_id').df()
        x, day = candidate_features(frame, candidate)
        if not day.any():
            raise ValueError('No eligible daytime rows')
        y = (frame.target_kwh/x.cap).clip(0, 1.2).astype('float32')
        model = lgb.LGBMRegressor(**params, n_jobs=threads).fit(x.loc[day], y.loc[day]).booster_
        counts.append({'round': round_id, 'daytime_rows': int(day.sum())})
        del frame, x, y
        models = destination/'models'
        models.mkdir(exist_ok=True)
        model_path = models/f'round{round_id}.txt'
        model.save_model(str(model_path))
        frame = con.execute(f'SELECT {columns} FROM diagnostic_features '
                            f'WHERE round={round_id} ORDER BY ts,site_id').df()
        prediction = predict_candidate(model, frame, candidate, threads)
        sample = np.linspace(0, len(frame)-1, min(1000, len(frame)), dtype=int)
        np.testing.assert_allclose(prediction.forecast_kwh.iloc[sample],
            predict_candidate(lgb.Booster(model_file=str(model_path)), frame.iloc[sample], candidate, threads).forecast_kwh,
            rtol=1e-12, atol=1e-12)
        rounds.append(prediction)
        artifacts.append(model_path)
        del frame, model
    prediction = blend(rounds[0], rounds[-1], .5)
    metrics = evaluate(con, prediction)
    artifact = destination/'predictions.parquet'
    export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', artifact)
    artifacts.append(artifact)
    metrics.update(candidate=candidate, params=params, features=[f for f in FEATURES if f not in candidate.get('drop', [])],
        training_condition=condition, training_cutoff=window['training_cutoff'], training_rows=counts,
        seconds=time.monotonic()-started,
        artifacts_sha256={p.relative_to(destination).as_posix(): digest(p) for p in artifacts})
    save_json(destination/'metrics.json', metrics)
    return prediction, metrics


def error_relationship(con, baseline, candidate):
    con.register('reference_prediction', baseline)
    con.register('candidate_prediction', candidate)
    correlation, difference = con.execute('''SELECT
        corr(b.forecast_kwh-a.energy,c.forecast_kwh-a.energy),
        avg(abs(b.forecast_kwh-c.forecast_kwh))
        FROM diagnostic_actual a JOIN reference_prediction b USING(site_id,ts)
        JOIN candidate_prediction c USING(site_id,ts)''').fetchone()
    correlation = float(correlation) if correlation is not None and np.isfinite(correlation) else None
    return {'residual_correlation_with_baseline': correlation, 'mean_prediction_difference_kwh': difference}


def summarize(output, windows, smoke):
    rows = []
    for candidate in CANDIDATES:
        monthly = []
        for window in windows:
            path = output/window['id']/candidate['name']/'comparison.json'
            if path.exists():
                monthly.append(json.loads(path.read_text(encoding='utf-8')))
        errors = [m['site_equal_nmae'] for m in monthly]
        rows.append({'candidate': candidate['name'], 'completed_months': len(monthly),
            'month_equal_nmae': float(np.mean(errors)) if errors else None,
            'worst_month_nmae': max(errors) if errors else None,
            'complete_253_hours': sum(m['complete_253_hours'] for m in monthly),
            'W_available_hours_won': sum(m['settlement']['W_won'] for m in monthly if m['settlement'])
                if any(m['settlement'] for m in monthly) else None, 'monthly': monthly})
    complete = all(r['completed_months'] == len(windows) for r in rows)
    report = {'status': 'complete' if complete else 'partial', 'smoke': smoke,
        'interpretation': 'Exploratory internal screening, not independent selected-model performance. No outer scoring.',
        'all_12_months': complete and len(windows) == 12 and not smoke,
        'exploratory_best': min(rows, key=lambda r: r['month_equal_nmae'])['candidate'] if complete and not smoke else None,
        'candidates': rows}
    save_json(output/'summary.json', report)
    lines = ['# Candidate screening', '', report['interpretation'], '',
        'SMOKE ONLY: 12 trees and one internal block.' if smoke else '600 trees per model; K=1; reference G1/G2 fixed.', '',
        '| Candidate | Months | Mean NMAE | Worst NMAE | Complete hours | W (won) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        error = f"{r['month_equal_nmae']:.4%}" if r['month_equal_nmae'] is not None else 'N/A'
        worst = f"{r['worst_month_nmae']:.4%}" if r['worst_month_nmae'] is not None else 'N/A'
        cost = f"{r['W_available_hours_won']:,.0f}" if r['W_available_hours_won'] is not None else 'N/A'
        lines.append(f"| {r['candidate']} | {r['completed_months']} | {error} | {worst} | {r['complete_253_hours']} | {cost} |")
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def run(data_dir, output, config, threads=4, smoke=False, resume=False, after_run=None):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if ROOT/'results' not in output.parents or data_dir == output or data_dir in output.parents:
        raise ValueError('Output must be inside ai_sim/results and outside input data')
    if threads < 1:
        raise ValueError('threads must be positive')
    _, outers = load_plan(config)
    # The latest outer origin provides twelve historical seasonal blocks for screening.
    windows = [w for w in outers[-1]['inner'] if w['purpose'] == 'model_selection']
    if smoke:
        windows = windows[:1]
    if output.exists() and not resume:
        raise FileExistsError('Use a new output directory or --resume')
    if after_run:
        if Path(after_run).resolve() == output:
            raise ValueError('Cannot wait for this experiment itself')
        wait_for_run(after_run)
    gpu = check_cuda(threads)
    print('Hashing inputs for separate candidate screening...', flush=True)
    manifest = {'gpu': gpu, 'threads': threads, 'smoke': smoke, 'trees': 12 if smoke else 600,
        'candidates': CANDIDATES, 'base_params': PARAMS, 'features': FEATURES,
        'windows': [{k: str(v) for k, v in w.items()} for w in windows],
        'config_sha256': digest(config),
        'source_sha256': {p.name: digest(p) for p in [Path(__file__), *(ROOT/'scripts').glob('*.py')]},
        'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        'input_sha256': {n: digest(data_dir/n) for n in ['report.json', 'sites.parquet',
            'reference_composition.parquet', 'train_targets.parquet', 'train_prices.parquet',
            'train_round10.parquet', 'train_round17.parquet']}}
    if output.exists():
        if json.loads((output/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Configuration, source, environment or input changed; use a new output')
    else:
        output.mkdir(parents=True)
        save_json(output/'manifest.json', manifest)
    with connect() as con:
        attach_data(con, data_dir)
        validate_inputs(con)
        read_view(con, 'composition', data_dir/'reference_composition.parquet')
        for window in windows:
            for candidate in CANDIDATES:
                destination = output/window['id']/candidate['name']
                prediction, metrics = run_block(con, window, candidate, destination, threads, manifest['trees'])
                if candidate['name'] == 'baseline':
                    baseline = prediction
                comparison = dict(metrics, window=window['id'], **error_relationship(con, baseline, prediction))
                save_json(destination/'comparison.json', comparison)
                summarize(output, windows, smoke)
                print(f'{window["id"]}/{candidate["name"]}: NMAE={metrics["site_equal_nmae"]:.6f}', flush=True)
    return summarize(output, windows, smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--after-run', type=Path, help='Wait for this full main run before using the GPU')
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, args.config, args.threads, args.smoke, args.resume, args.after_run)
    print(json.dumps({'status': result['status'], 'smoke': result['smoke'],
                      'exploratory_best': result['exploratory_best']}, indent=2))


if __name__ == '__main__':
    main()
