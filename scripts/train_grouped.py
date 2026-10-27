"""Portable, resumable comparison of 1/2/3 predictive plant clusters."""
import argparse
import json
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from fixed_model import FEATURES, PARAMS, WEATHER, fit, predict
from pipeline_common import connect, export, read_view
from seasonal_splits import load_plan, window_views
from site_groups import group_sites, profiles
from time_splits import attach_data, digest, validate_inputs
from train_fixed import evaluate

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, default=str, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def train_candidate(con, window, k, destination, threads):
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    window_views(con, window, allow_outer_evaluation=True)
    with threadpool_limits(limits=threads):
        assignment, clustering = group_sites(profiles(con), k)
    save_json(destination/'groups.json', {'training_cutoff': window['training_cutoff'],
        'clustering': clustering, 'plants': json.loads(assignment.to_json(orient='records'))})
    con.register('assignment', assignment[['site_id', 'cluster']])
    models_dir = destination/'models'
    models_dir.mkdir(exist_ok=True)
    outputs, sizes = [], []
    columns = ', '.join(['site_id', 'ts', 'capacity_kw', 'lat', 'lon']+WEATHER)
    for cluster in range(k):
        rounds = []
        for r in (10, 17):
            frame = con.execute(f'''SELECT {columns},target_kwh FROM training_features
                JOIN assignment USING(site_id) WHERE cluster={cluster} AND round={r}
                ORDER BY ts,site_id''').df()
            print(f'{window["id"]} k={k} cluster={cluster} round={r}: {len(frame):,} training rows', flush=True)
            model, daytime = fit(frame, threads)
            sizes.append({'cluster': cluster, 'round': r, 'eligible_rows': len(frame), 'daytime_rows': daytime})
            del frame
            path = models_dir/f'cluster{cluster}_round{r}.txt'
            model.save_model(str(path))
            frame = con.execute(f'''SELECT {columns} FROM diagnostic_features
                JOIN assignment USING(site_id) WHERE cluster={cluster} AND round={r}
                ORDER BY ts,site_id''').df()
            prediction = (predict(model, frame, threads) if len(frame) else
                          frame[['site_id', 'ts']].assign(forecast_kwh=pd.Series(dtype=float)))
            if len(frame):
                sample = np.linspace(0, len(frame)-1, min(1000, len(frame)), dtype=int)
                np.testing.assert_allclose(prediction.forecast_kwh.iloc[sample],
                    predict(lgb.Booster(model_file=str(path)), frame.iloc[sample], threads).forecast_kwh,
                    rtol=1e-12, atol=1e-12)
            rounds.append(prediction.set_index(['site_id', 'ts']))
            del frame, model
        if not rounds[0].index.equals(rounds[1].index):
            raise ValueError('Round prediction keys differ')
        outputs.append(((rounds[0]+rounds[1])/2).reset_index())
    predictions = pd.concat(outputs, ignore_index=True).sort_values(['ts', 'site_id'])
    metrics = evaluate(con, predictions)
    export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', destination/'site_predictions.parquet')
    site_metrics = con.execute('''SELECT site_id,cluster,count(*) AS hours,
        avg(abs(forecast_kwh-energy)) AS mae_kwh,
        avg(abs(forecast_kwh-energy)/capacity_kw) AS nmae
        FROM site_predictions JOIN diagnostic_actual USING(site_id,ts)
        JOIN sites USING(site_id) JOIN assignment USING(site_id)
        GROUP BY site_id,cluster ORDER BY site_id''').df()
    save_json(destination/'site_metrics.json', json.loads(site_metrics.to_json(orient='records')))
    metrics.update(k=k, fold=window['id'], training_cutoff=window['training_cutoff'],
                   groups=assignment.groupby('cluster').size().to_dict(), training_rows=sizes,
                   silhouette=clustering['silhouette'], seconds=time.monotonic()-started)
    files = [destination/'groups.json', destination/'site_metrics.json', destination/'site_predictions.parquet']
    files += list(models_dir.glob('*.txt'))
    metrics['artifacts_sha256'] = {p.relative_to(destination).as_posix(): digest(p) for p in files}
    metrics['status'] = 'complete'
    save_json(destination/'metrics.json', metrics)  # Completion marker is written last.
    return metrics


def summarize(output, folds, ks):
    rows, monthly = [], []
    for k in ks:
        available = []
        for fold in folds:
            path = output/fold['id']/f'k{k}'/'metrics.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                available.append(result)
                monthly.append(result)
        rows.append({'k': k, 'completed_months': len(available),
                     'mean_monthly_nmae': float(np.mean([m['site_equal_nmae'] for m in available])) if available else None,
                     'complete_253_hours': sum(m['complete_253_hours'] for m in available),
                     'W_available_hours_won': sum(m['settlement']['W_won'] for m in available if m['settlement'])
                         if any(m['settlement'] for m in available) else None,
                     'seconds': sum(m['seconds'] for m in available)})
    complete = all(r['completed_months'] == len(folds) for r in rows)
    result = {'status': 'complete' if complete else 'partial', 'all_12_months': complete and len(folds)==12,
              'interpretation': 'Exploratory candidate comparison; best K is not an unbiased selected-model score. G1/G2 unchanged.',
              'exploratory_best_k': min(rows, key=lambda r:r['mean_monthly_nmae'])['k'] if complete and len(folds)==12 else None,
              'candidates': rows}
    save_json(output/'summary.json', result)
    lines = ['# Predictive plant grouping comparison', '', result['interpretation'], '',
             '| K | Completed months | Month-equal NMAE | Complete 253 hours | W on those hours (won) |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        error = f"{r['mean_monthly_nmae']:.4%}" if r['mean_monthly_nmae'] is not None else 'N/A'
        cost = f"{r['W_available_hours_won']:,.0f}" if r['W_available_hours_won'] is not None else 'N/A'
        lines.append(f"| {r['k']} | {r['completed_months']} | {error} | {r['complete_253_hours']} | {cost} |")
    lines += ['', '| Month | K | NMAE | Plant counts per cluster |', '|---|---:|---:|---|']
    for r in monthly:
        lines.append(f"| {r['fold']} | {r['k']} | {r['site_equal_nmae']:.4%} | {r['groups']} |")
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def run(data_dir, output, config, ks=(1,2,3), fold_ids=None, threads=4, resume=False):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if ROOT/'results' not in output.parents:
        raise ValueError('Output must be a run directory inside ai_sim/results')
    if data_dir == output or data_dir in output.parents or output in data_dir.parents:
        raise ValueError('Inputs and outputs must be separate')
    if threads < 1 or not ks or any(k not in (1,2,3) for k in ks) or len(set(ks)) != len(ks):
        raise ValueError('Use positive threads and unique cluster candidates from 1,2,3')
    _, folds = load_plan(config)
    if fold_ids:
        if set(fold_ids)-{w['id'] for w in folds}:
            raise ValueError('Unknown outer fold')
        folds = [w for w in folds if w['id'] in fold_ids]
    filenames = ['report.json','sites.parquet','reference_composition.parquet','train_targets.parquet',
                 'train_prices.parquet','train_round10.parquet','train_round17.parquet']
    manifest = {'ks': list(ks), 'folds': [w['id'] for w in folds], 'threads': threads,
                'model_params': PARAMS, 'features': FEATURES, 'profile_min_bin_rows': 30,
                'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                             'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'config_sha256': digest(config), 'input_sha256': {n:digest(data_dir/n) for n in filenames},
                'source_sha256': {p.name:digest(p) for p in (ROOT/'scripts').glob('*.py')}}
    if output.exists():
        if not resume:
            raise FileExistsError('Use a new output or --resume')
        if json.loads((output/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Resume configuration, source, versions or inputs differ; use a new run')
    else:
        output.mkdir(parents=True)
        save_json(output/'manifest.json', manifest)
    with connect() as con:
        attach_data(con, data_dir)
        validate_inputs(con)
        read_view(con, 'composition', data_dir/'reference_composition.parquet')
        for window in folds:
            for k in ks:
                destination = output/window['id']/f'k{k}'
                marker = destination/'metrics.json'
                if marker.exists():
                    previous = json.loads(marker.read_text(encoding='utf-8'))
                    for name, expected in previous['artifacts_sha256'].items():
                        if digest(destination/name) != expected:
                            raise ValueError(f'Completed artifact changed: {destination/name}')
                    print(f'Skip complete {window["id"]} k={k}', flush=True)
                else:
                    train_candidate(con, window, k, destination, threads)
                summarize(output, folds, ks)
    return summarize(output, folds, ks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/grouping_v1')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    parser.add_argument('--clusters', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--fold', action='append')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir, args.config, args.clusters,
                         args.fold, args.threads, args.resume), indent=2))


if __name__ == '__main__':
    main()
