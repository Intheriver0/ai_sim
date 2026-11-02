"""One entry point: preflight, seasonal comparison, final fit and validated CSVs."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import duckdb
from threadpoolctl import threadpool_limits

from fixed_model import PARAMS
from gpu_training import check_cuda
from paired_model import attach_pair, fit, predict, PAIRED_FEATURES
from pipeline_common import connect, read_view, check_keys, check_time, export, reject
from residual_groups import objective, optimize, residuals
from scoring import validate_composition, validate_forecast
from seasonal_splits import load_plan, month_shift, window_views
from site_groups import profiles, group_sites
from sparse_site_groups import discover, terrain_profile, error_profile, assign_sparse
from time_splits import attach_data, validate_inputs, digest, stamp, interval
from train_cycles import train_predict, make_panel, candidate_params, completed, finish
from train_fixed import evaluate
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]


def seasonal_windows(outer):
    """All optimization labels precede all selection predictions."""
    anchor = outer['start']
    available_end = outer['training_cutoff'].replace(hour=0, minute=0, second=0, microsecond=0)
    lag = anchor-timedelta(days=1)+timedelta(hours=17)-outer['training_cutoff']
    blocks = []
    for role, offsets in [('optimization', range(-24, -12)), ('selection', range(-12, 0))]:
        windows = []
        for offset in offsets:
            start = month_shift(anchor, offset)
            end = min(month_shift(start, 1), available_end)
            cutoff = start-timedelta(days=1)+timedelta(hours=17)-lag
            if not outer['training_start'] < cutoff < start < end <= outer['training_cutoff']:
                raise ValueError('Insufficient chronological history for seasonal selection')
            windows.append(dict(outer, id=f'{outer["id"]}_{role}_{start:%Y_%m}', purpose=role,
                                start=start, end=end, training_cutoff=cutoff))
        blocks.append(windows)
    # The final optimization labels must already exist at the first selection origin,
    # not merely before its target timestamp (the final seven hours matter).
    blocks[0][-1]['end'] = min(blocks[0][-1]['end'], blocks[1][0]['training_cutoff'])
    if blocks[0][-1]['end'] > blocks[1][0]['training_cutoff']:
        raise ValueError('Optimization and selection overlap')
    return blocks


def span(windows):
    return dict(windows[0], end=windows[-1]['end'])


def predict_windows(con, windows, assignment, params, args, family='paired', refresh=None):
    pieces, keys = [], []
    for window in windows:
        current = assignment
        if refresh is not None:
            base, terrain, optimization_predictions, history_start = refresh
            history = pd.concat([optimization_predictions]+pieces, ignore_index=True)
            errors = available_errors(con, history, history_start, window['training_cutoff'], args)
            current = assign_sparse(base, terrain, errors)[['site_id', 'cluster']]
        frame, key = train_predict(con, window, current, params, args.output_dir, args.threads, family)
        pieces.append(frame)
        keys.append(key)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(['site_id', 'ts']).any():
        raise ValueError('Overlapping out-of-sample predictions')
    return result, keys


def new_error_profile(con, prediction, window):
    con.register('refresh_predictions', prediction)
    return con.execute(f'''SELECT site_id,count(*) AS error_rows,
        avg((forecast_kwh-energy_raw_kwh)/capacity_kw) AS signed_nmae,
        avg((forecast_kwh>energy_raw_kwh)::INT) AS positive_fraction,
        avg((forecast_kwh<energy_raw_kwh)::INT) AS negative_fraction
        FROM refresh_predictions JOIN targets USING(site_id,ts) JOIN sites USING(site_id)
        WHERE {interval(window['start'],window['end'])} AND hour(ts) BETWEEN 9 AND 16
        GROUP BY site_id ORDER BY site_id''').df()


def available_errors(con, prediction, start, cutoff, args):
    # Prefer this model's genuinely past OOF errors. Use CUDA 1028 when its history
    # is sufficient and the new model has fewer than 30 daytime observations.
    old = error_profile(con, args.error_run, cutoff)
    new = new_error_profile(con, prediction, {'start': start, 'end': cutoff})
    reliable = new.loc[new.error_rows >= 30]
    return pd.concat([reliable, old.loc[~old.site_id.isin(reliable.site_id)]], ignore_index=True)


def select_candidates(con, outer, args):
    directory = args.output_dir/outer['id']/'search'
    previous = completed(directory)
    if previous:
        for key in previous['cache_keys']:
            if completed(args.output_dir/'cache'/key) is None:
                raise ValueError('Selected candidate cache is missing')
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    opt_windows, sel_windows = seasonal_windows(outer)
    opt_actual, opt_weights = make_panel(con, span(opt_windows), False)
    sel_actual, sel_weights = make_panel(con, span(sel_windows), False)
    sites = con.execute('SELECT site_id,capacity_kw FROM sites ORDER BY site_id').df()
    reference = con.execute('''SELECT site_id,CASE "group" WHEN 'G1' THEN 0 ELSE 1 END AS cluster
        FROM reference_composition ORDER BY site_id''').df()
    global_assignment = sites[['site_id']].assign(cluster=0)
    window_views(con, opt_windows[0])
    with threadpool_limits(limits=args.threads):
        profile = profiles(con)
        terrain = terrain_profile(args.terrain_dir, profile)
        past_errors = error_profile(con, args.error_run, opt_windows[0]['training_cutoff'])
        starts = {f'profiles{k}': group_sites(profile, k)[0] for k in args.clusters}
    candidates, all_keys = [], set()
    for mode in ['separate', 'paired', 'cycles']+list(starts):
        for trial in range(1 if mode in ('separate', 'paired') else args.trials):
            params = dict(PARAMS) if mode in ('separate', 'paired') else candidate_params(trial, args.trees)
            assignment = (assign_sparse(starts[mode], terrain, past_errors)[['site_id', 'cluster']]
                          if mode in starts else global_assignment.copy())
            composition, seen = reference.copy(), set()
            for cycle in range(1 if mode in ('separate', 'paired') else args.cycles+1):
                signature = (tuple(assignment.sort_values('site_id').cluster),
                             tuple(composition.sort_values('site_id').cluster))
                if signature in seen:
                    break
                seen.add(signature)
                refresh = None
                if mode in ('separate', 'paired'):
                    allocations = [('reference', reference)]
                    proposed = reference
                else:
                    opt_prediction, keys = predict_windows(con, opt_windows, assignment, params, args)
                    all_keys.update(keys)
                    proposed, _ = optimize(residuals(opt_prediction, opt_actual), opt_weights,
                        opt_actual.columns, sites, composition, args.search_steps, seed=42+trial*100+cycle)
                    allocations = [('reference', reference), ('optimized', proposed)]
                    if mode in starts:
                        refresh = assign_sparse(starts[mode], terrain,
                                                new_error_profile(con, opt_prediction, span(opt_windows)))
                        save_json(directory/f'{mode}_trial{trial}_cycle{cycle}_refresh.json',
                            {'latest_error_time': opt_windows[-1]['end'],
                             'assignment': json.loads(refresh.to_json(orient='records'))})
                family = 'separate' if mode == 'separate' else 'paired'
                refresh_policy = ((starts[mode], terrain, opt_prediction, opt_windows[0]['start'])
                                  if mode in starts else None)
                selection, keys = predict_windows(con, sel_windows, assignment, params, args, family, refresh_policy)
                all_keys.update(keys)
                errors = residuals(selection, sel_actual)
                refit_assignment = assignment
                if mode in starts:
                    history = pd.concat([opt_prediction, selection], ignore_index=True)
                    refit = assign_sparse(starts[mode], terrain,
                        available_errors(con, history, opt_windows[0]['start'], outer['training_cutoff'], args))
                    refit_assignment = refit[['site_id', 'cluster']]
                    save_json(directory/f'{mode}_trial{trial}_cycle{cycle}_refit.json',
                        {'cutoff': outer['training_cutoff'], 'assignment': json.loads(refit.to_json(orient='records'))})
                for label, allocation in allocations:
                    labels = allocation.set_index('site_id').loc[sel_actual.columns, 'cluster'].to_numpy()
                    candidates.append({'mode': mode, 'trial': trial, 'cycle': cycle, 'family': family,
                        'allocation': label, 'selection_W_proxy': objective(errors, sel_weights, labels),
                        'params': params, 'assignment': refit_assignment.to_dict('records'),
                        'sparse_refresh': 'past-only monthly OOF; same update at refit' if mode in starts else None,
                        'composition': allocation.to_dict('records')})
                save_json(directory/'progress.json', {'candidates': candidates})
                print(f'{outer["id"]}: {mode} trial={trial} cycle={cycle} selection complete', flush=True)
                assignment = refresh[['site_id', 'cluster']] if refresh is not None else proposed.copy()
                composition = proposed.copy()
    finalists = {mode: min([c for c in candidates if c['mode'] == mode], key=lambda c:c['selection_W_proxy'])
                 for mode in dict.fromkeys(c['mode'] for c in candidates)}
    best = min(finalists.values(), key=lambda c:c['selection_W_proxy'])
    result = {'status': 'complete', 'selected': best, 'finalists': finalists,
              'cache_keys': sorted(all_keys), 'selection_basis': 'annual observed-panel W proxy; not full settlement',
              'optimization_panel': {'sites': list(opt_actual.columns), 'hours': len(opt_actual)},
              'selection_panel': {'sites': list(sel_actual.columns), 'hours': len(sel_actual)},
              'windows': {'optimization': opt_windows, 'selection': sel_windows}}
    save_json(directory/'selected.json', result)  # Freeze BEFORE outer evaluation.
    return finish(directory, result, list(directory.glob('*.json')))


def set_composition(con, candidate):
    frame = pd.DataFrame(candidate['composition'])
    con.register('composition', frame[['site_id']].assign(group=frame.cluster.map({0: 'G1', 1: 'G2'})))
    validate_composition(con)


def compare_fold(con, outer, args):
    directory = args.output_dir/outer['id']/'comparison'
    search = select_candidates(con, outer, args)
    previous = completed(directory)
    if previous:
        for key in previous['cache_keys']:
            if completed(args.output_dir/'cache'/key) is None:
                raise ValueError('Comparison model cache missing')
        return previous
    directory.mkdir(exist_ok=True)
    results, keys = {}, []
    for mode, candidate in search['finalists'].items():
        set_composition(con, candidate)
        prediction, key = train_predict(con, outer, pd.DataFrame(candidate['assignment']), candidate['params'],
                                        args.output_dir, args.threads, candidate['family'])
        keys.append(key)
        window_views(con, outer, allow_outer_evaluation=True)
        metrics = evaluate(con, prediction)
        con.register('comparison_prediction', prediction)
        grouped = con.execute('''SELECT "group",ts,sum(forecast_kwh-energy) AS error,sum(capacity_kw) AS capacity
            FROM comparison_prediction JOIN diagnostic_actual USING(site_id,ts) JOIN sites USING(site_id)
            JOIN composition USING(site_id) JOIN settlement_prices USING(ts) GROUP BY "group",ts''').df()
        metrics['group_nmae_complete_hours'] = float((grouped.error.abs()/grouped.capacity).mean()) if len(grouped) else None
        results[mode] = metrics
    baseline = results['separate']
    for result in results.values():
        result['W_reduction_vs_separate'] = (baseline['settlement']['W_won']-result['settlement']['W_won']
            if baseline['settlement'] and result['settlement'] else None)
    save_json(directory/'metrics.json', results)
    return finish(directory, {'fold': outer['id'], 'selected_mode_before_outer': search['selected']['mode'],
                              'metrics': results, 'cache_keys': keys}, [directory/'metrics.json'])


def comparison_report(output, results, expected):
    modes = list(results[0]['metrics'])+['selected_procedure']
    summary = {}
    for mode in modes:
        metrics = [r['metrics'][r['selected_mode_before_outer'] if mode=='selected_procedure' else mode]
                   for r in results]
        settled = [m for m in metrics if m['settlement'] is not None]
        summary[mode] = {'months': len(metrics),
            'month_equal_site_nmae': float(np.mean([m['site_equal_nmae'] for m in metrics])),
            'complete_hours': sum(m['complete_253_hours'] for m in metrics),
            'W_available_hours': sum(m['settlement']['W_won'] for m in settled) if settled else None,
            'W_reduction_vs_separate': sum(m['W_reduction_vs_separate'] for m in settled) if settled else None}
    report = {'status': 'complete' if len(results)==expected else 'partial',
        'all_12_months': len(results)==expected==12,
        'interpretation': 'Diagnostic comparison. selected_procedure was chosen before each outer month; deployment uses only inner selection.',
        'summary': summary, 'folds': results}
    save_json(output/'comparison.json', report)
    lines = ['# Seasonal comparison', '', report['interpretation'], '',
             '| Mode | Months | Site NMAE | Complete hours | W | W reduction vs separate |',
             '|---|---:|---:|---:|---:|---:|']
    for mode, m in summary.items():
        lines.append(f"| {mode} | {m['months']} | {m['month_equal_site_nmae']:.4%} | "
                     f"{m['complete_hours']} | {m['W_available_hours']} | {m['W_reduction_vs_separate']} |")
    (output/'COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def deployment_window(con, config):
    first, last = con.execute('SELECT min(ts),max(ts) FROM test_prices').fetchall()[0]
    start = first-timedelta(hours=1)
    if start.day != 1 or start.hour != 0:
        raise ValueError('Submission calendar must start at month boundary')
    cutoff = start-timedelta(days=1)+timedelta(hours=17-config['generation_availability_lag_hours'])
    return {'id': 'deployment', 'purpose': 'deployment', 'start': start, 'end': last,
        'training_start': pd.Timestamp(config['training_start']).to_pydatetime(), 'training_cutoff': cutoff,
        'fit_cutoff': cutoff, 'early_stop_start': start-timedelta(days=7), 'early_stop_end': start-timedelta(days=1)}


def submit(con, outer, args):
    search = select_candidates(con, outer, args)
    directory = args.output_dir/'submission'
    previous = completed(directory)
    if previous:
        if completed(args.output_dir/'cache'/previous['cache_key']) is None:
            raise ValueError('Submission model missing')
        return previous
    directory.mkdir(exist_ok=True)
    selected = search['selected']
    set_composition(con, selected)
    prediction, key = train_predict(con, outer, pd.DataFrame(selected['assignment']), selected['params'],
                                    args.output_dir, args.threads, selected['family'])
    con.register('submission_predictions', prediction)
    check_keys(con, 'submission_predictions', ['site_id', 'ts'])
    reject(con, '''SELECT count(*) FROM ((SELECT site_id,ts FROM sites CROSS JOIN test_prices
        EXCEPT SELECT site_id,ts FROM submission_predictions) UNION ALL
        (SELECT site_id,ts FROM submission_predictions EXCEPT SELECT site_id,ts FROM sites CROSS JOIN test_prices))''',
        'Incomplete submission site-hour keys')
    reject(con, '''SELECT count(*) FROM submission_predictions WHERE forecast_kwh IS NULL
        OR NOT isfinite(forecast_kwh) OR forecast_kwh<0''', 'Invalid site predictions')
    con.execute('''CREATE OR REPLACE TEMP VIEW forecast AS SELECT "group",ts,sum(forecast_kwh) AS forecast_kwh
        FROM submission_predictions JOIN composition USING(site_id) GROUP BY "group",ts''')
    # Validate independently; keep training prices intact for resume and selection.
    with connect() as validator:
        for name, query in [('sites', 'SELECT * FROM sites'), ('composition', 'SELECT * FROM composition'),
                            ('prices', 'SELECT * FROM test_prices'), ('forecast', 'SELECT * FROM forecast')]:
            validator.register(name, con.execute(query).df())
        capacities = validate_composition(validator)
        validate_forecast(validator)
    con.execute('SELECT * FROM composition ORDER BY site_id').df().to_csv(directory/'composition.csv', index=False)
    con.execute('SELECT * FROM forecast ORDER BY "group",ts').df().to_csv(directory/'forecast.csv', index=False)
    export(con, 'SELECT * FROM submission_predictions ORDER BY ts,site_id', directory/'site_predictions.parquet')
    # Read exported CSVs back through the same validator, including timezone and exact grid checks.
    with connect() as validator:
        validator.register('sites', con.execute('SELECT * FROM sites').df())
        validator.register('prices', con.execute('SELECT * FROM test_prices').df())
        read_view(validator, 'composition', directory/'composition.csv')
        read_view(validator, 'forecast', directory/'forecast.csv')
        validate_composition(validator)
        validate_forecast(validator)
    return finish(directory, {'status': 'complete', 'cache_key': key, 'selected': selected,
        'training_cutoff': outer['training_cutoff'], 'capacity_kw': capacities,
        'note': 'Fixed composition and models for full test period; no test actuals or test price features.'},
        [directory/n for n in ('composition.csv', 'forecast.csv', 'site_predictions.parquet')])


def attach_all(con, data_dir):
    attach_data(con, data_dir)
    validate_inputs(con)
    attach_pair(con, name='paired_train')
    for r in (10, 17):
        read_view(con, f'test_features{r}', data_dir/f'test_round{r}.parquet')
        check_time(con, f'test_features{r}')
        reject(con, f'''SELECT count(*) FROM test_features{r}
            WHERE issued_at IS DISTINCT FROM (date_trunc('day',ts-INTERVAL '1 hour')
                -INTERVAL '1 day'+INTERVAL '{r} hours')
            OR target_day IS DISTINCT FROM CAST(ts-INTERVAL '1 hour' AS DATE)''',
            'Invalid test forecast issue time')
    read_view(con, 'test_prices', data_dir/'test_prices.parquet')
    attach_pair(con, 'test_features10', 'test_features17', 'paired_test')
    check_keys(con, 'test_prices', ['ts'])
    check_time(con, 'test_prices')
    reject(con, '''SELECT count(*) FROM ((SELECT site_id,ts FROM sites CROSS JOIN test_prices
        EXCEPT SELECT site_id,ts FROM paired_test) UNION ALL
        (SELECT site_id,ts FROM paired_test EXCEPT SELECT site_id,ts FROM sites CROSS JOIN test_prices))''',
        'Test forecast grid differs from prices')
    reject(con, 'SELECT count(*) FROM paired_train JOIN paired_test USING(site_id,ts)', 'Train/test overlap')
    con.execute('CREATE TEMP VIEW paired AS SELECT * FROM paired_train UNION ALL SELECT * FROM paired_test')
    read_view(con, 'reference_composition', data_dir/'reference_composition.parquet')


def preflight(con, folds, deploy, args):
    # Validate every chronological panel before spending days fitting models.
    for outer in folds+[deploy]:
        optimization, selection = seasonal_windows(outer)
        for windows in (optimization, selection):
            actual, _ = make_panel(con, span(windows), False)
            print(f'Preflight {outer["id"]}/{windows[0]["purpose"]}: {actual.shape}', flush=True)
        window_views(con, optimization[0])
        with threadpool_limits(limits=args.threads):
            profile = profiles(con)
            for k in args.clusters:
                group_sites(profile, k)
    sites = con.execute('SELECT * FROM sites ORDER BY site_id').df()
    terrain_profile(args.terrain_dir, sites)
    error_profile(con, args.error_run, deploy['training_cutoff'])
    gpu = check_cuda(args.threads)
    sample = con.execute('''SELECT p.*,target_kwh FROM paired_train p JOIN targets USING(site_id,ts)
        WHERE training_eligible AND hour(ts)=12 ORDER BY ts,site_id LIMIT 1000''').df()
    model = fit(sample, args.threads, dict(PARAMS, n_estimators=3))
    prediction = predict(model, sample, args.threads)
    if not np.isfinite(prediction.forecast_kwh).all():
        raise ValueError('Real-data paired CUDA smoke failed')
    save_json(args.output_dir/'preflight.json', {'status': 'complete', 'gpu': gpu, 'real_smoke_rows': len(sample)})


def run(args):
    if min(args.trials, args.trees, args.search_steps, args.threads) < 1 or args.cycles < 0:
        raise ValueError('Invalid training budgets')
    if not args.clusters or len(set(args.clusters)) != len(args.clusters) or any(k < 2 for k in args.clusters):
        raise ValueError('Use distinct predictive cluster counts >=2')
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.terrain_dir = args.terrain_dir.resolve()
    if ROOT/'results' not in args.output_dir.parents:
        raise ValueError('Output must be a dedicated directory under ai_sim/results')
    args.error_run, error_files = discover(args.results_root, args.error_run)
    source = json.loads((args.error_run/'manifest.json').read_text(encoding='utf-8'))
    for name in ('sites.parquet', 'train_targets.parquet'):
        if source['input_sha256'].get(name) != digest(args.data_dir/name):
            raise ValueError(f'CUDA 1028 inputs differ: {name}')
    config, folds = load_plan(args.config)
    manifest = {'options': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                           if k not in ('stage', 'resume')}, 'config': digest(args.config),
        'paired_features': PAIRED_FEATURES,
        'inputs': {p.name:digest(p) for p in args.data_dir.glob('*.parquet')},
        'sources': {p.name:digest(p) for p in (ROOT/'scripts').glob('*.py')}, 'error_files': error_files,
        'terrain': {n:digest(args.terrain_dir/n) for n in ('manifest.json','site_quality.json')},
        'versions': {'python':platform.python_version(), 'lightgbm':lgb.__version__, 'numpy':np.__version__,
                     'pandas':pd.__version__, 'sklearn':sklearn.__version__, 'duckdb':duckdb.__version__}}
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError('Use --resume with the same budgets and inputs')
        if json.loads((args.output_dir/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run configuration, code, versions or inputs changed')
    else:
        args.output_dir.mkdir(parents=True)
        save_json(args.output_dir/'manifest.json', manifest)
    with connect() as con:
        attach_all(con, args.data_dir)
        deploy = deployment_window(con, config)
        preflight(con, folds, deploy, args)
        if args.stage in ('all', 'compare'):
            results = []
            for outer in folds:
                results.append(compare_fold(con, outer, args))
                comparison_report(args.output_dir, results, len(folds))
        if args.stage in ('all', 'submit'):
            comparison = args.output_dir/'comparison.json'
            if not comparison.exists() or json.loads(comparison.read_text(encoding='utf-8'))['status'] != 'complete':
                raise ValueError('Complete comparison stage before submission')
            for outer in folds:
                compare_fold(con, outer, args)  # Verify completed artifacts, never select on these scores.
            return submit(con, deploy, args)
    return {'status': 'complete', 'stage': args.stage}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', 'preflight', 'compare', 'submit'], default='all')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--results-root', type=Path, default=ROOT/'results')
    p.add_argument('--error-run', type=Path)
    p.add_argument('--terrain-dir', type=Path, default=ROOT/'data/terrain_features_v2')
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--cycles', type=int, default=6)
    p.add_argument('--trees', type=int, default=1200)
    p.add_argument('--search-steps', type=int, default=20000)
    p.add_argument('--clusters', nargs='+', type=int, default=[2, 3])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), default=str, indent=2))
