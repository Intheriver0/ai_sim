"""Paired forecasts, residual grouping cycles, and separate predictive cluster models."""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fixed_model import PARAMS
import fixed_model
from gpu_training import check_cuda
from paired_model import attach_pair, fit, predict, PAIRED_FEATURES
from pipeline_common import connect, export, read_view
from residual_groups import panel, residuals, objective, optimize
from seasonal_splits import load_plan, window_views
from site_groups import profiles, group_sites
from sparse_site_groups import discover, error_profile, terrain_profile, assign_sparse
from time_splits import attach_data, validate_inputs, digest, stamp, interval
from train_fixed import evaluate
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]


def search_windows(outer):
    """Disjoint historical optimization/selection blocks; outer labels stay locked."""
    end = outer['training_cutoff'].replace(hour=0, minute=0, second=0, microsecond=0)
    windows = []
    for role, start, stop in [('optimization', end-timedelta(days=120), end-timedelta(days=45)),
                              ('selection', end-timedelta(days=45), end)]:
        # Preserve the configured generation availability lag relative to 17:00.
        lag = outer['start']-timedelta(days=1)+timedelta(hours=17)-outer['training_cutoff']
        cutoff = start-timedelta(days=1)+timedelta(hours=17)-lag
        if not outer['training_start'] < cutoff < start < stop <= outer['training_cutoff']:
            raise ValueError('Insufficient history for group-cycle windows')
        windows.append(dict(outer, id=f'{outer["id"]}_{role}', purpose=role,
                            start=start, end=stop, training_cutoff=cutoff))
    return windows


def completed(directory):
    path = directory/'done.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    for name, expected in result['artifacts'].items():
        if digest(directory/name) != expected:
            raise ValueError(f'Changed completed artifact: {directory/name}')
    return result


def finish(directory, result, paths):
    result['artifacts'] = {p.relative_to(directory).as_posix(): digest(p) for p in paths}
    save_json(directory/'done.json', result)
    return result


def train_predict(con, window, assignment, params, output, threads, family='paired'):
    if family not in ('paired', 'separate'):
        raise ValueError('Unknown model family')
    identity = {'window': {k: str(window[k]) for k in ('training_start', 'training_cutoff', 'start', 'end')},
                'assignment': assignment.sort_values('site_id').to_dict('records'), 'params': params, 'family': family}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = output/'cache'/key
    if completed(directory):
        read_view(con, 'cached_cycle', directory/'predictions.parquet')
        return con.execute('SELECT * FROM cached_cycle').df(), key
    directory.mkdir(parents=True, exist_ok=True)
    con.register('model_assignment', assignment)
    results, paths = [], []
    for cluster in sorted(assignment.cluster.unique()):
        print(f'{window["id"]} cluster={cluster} trees={params["n_estimators"]}', flush=True)
        train = con.execute(f'''SELECT p.*,t.target_kwh FROM paired p
            JOIN targets t USING(site_id,ts) JOIN model_assignment USING(site_id)
            WHERE cluster={int(cluster)} AND training_eligible
            AND {stamp(window['training_start'])} <= ts AND ts <= {stamp(window['training_cutoff'])}
            ORDER BY ts,site_id''').df()
        frame = con.execute(f'''SELECT p.* FROM paired p JOIN model_assignment USING(site_id)
            WHERE cluster={int(cluster)} AND {interval(window['start'], window['end'])}
            ORDER BY ts,site_id''').df()
        rounds = []
        for round_id in ([None] if family == 'paired' else [10, 17]):
            if round_id is None:
                model, infer, test_frame = fit(train, threads, params), predict, frame
            else:
                if params != PARAMS:
                    raise ValueError('Separate-round reference must use fixed baseline parameters')
                rename = {f'{c}_r{round_id}': c for c in fixed_model.WEATHER}
                model = fixed_model.fit(train.rename(columns=rename), threads)[0]
                infer, test_frame = fixed_model.predict, frame.rename(columns=rename)
            suffix = '' if round_id is None else f'_round{round_id}'
            path = directory/f'cluster{cluster}{suffix}.txt'
            model.save_model(str(path))
            paths.append(path)
            if len(test_frame):
                prediction = infer(model, test_frame, threads)
                sample = test_frame.iloc[::max(1, len(test_frame)//100)]
                np.testing.assert_allclose(infer(model, sample, threads).forecast_kwh,
                    infer(lgb.Booster(model_file=str(path)), sample, threads).forecast_kwh)
                rounds.append(prediction.set_index(['site_id', 'ts']))
            del model
        if rounds:
            if len(rounds) == 2 and not rounds[0].index.equals(rounds[1].index):
                raise ValueError('Round prediction keys differ')
            results.append((sum(rounds)/len(rounds)).reset_index())
        del train, frame
    if not results:
        raise ValueError('No prediction rows in window')
    prediction = pd.concat(results, ignore_index=True).sort_values(['ts', 'site_id'])
    con.register('cycle_predictions', prediction)
    path = directory/'predictions.parquet'
    export(con, 'SELECT * FROM cycle_predictions', path)
    finish(directory, {'identity': identity}, paths+[path])
    return prediction, key


def make_panel(con, window, complete_only):
    actual = con.execute(f'''SELECT site_id,ts,energy_raw_kwh AS energy FROM targets
        WHERE {interval(window['start'], window['end'])}''').df()
    prices = con.execute(f"SELECT * FROM prices WHERE {interval(window['start'], window['end'])} ORDER BY ts").df()
    if complete_only:
        # Reindex to all sites so absent plants cannot masquerade as complete coverage.
        grid = pd.MultiIndex.from_product([prices.ts, con.execute('SELECT site_id FROM sites').df().site_id],
                                         names=['ts', 'site_id'])
        actual = actual.set_index(['ts', 'site_id']).reindex(grid).reset_index()
    return panel(actual, prices, complete_only=complete_only)


def candidate_params(trial, trees):
    if trial == 0:
        return dict(PARAMS, n_estimators=trees)
    rng = np.random.default_rng(42+trial)
    return dict(PARAMS, n_estimators=trees, learning_rate=float(rng.choice([.02, .04, .06])),
                num_leaves=int(rng.choice([31, 63, 127])),
                min_child_samples=int(rng.choice([100, 200, 500])),
                reg_lambda=float(rng.choice([0., 1., 5.])), random_state=42+trial)


def run_fold(con, outer, args):
    directory = args.output_dir/outer['id']
    cached = completed(directory)
    if cached:
        if completed(args.output_dir/'cache'/cached['outer_cache']) is None:
            raise ValueError('Completed fold is missing its fitted models')
        return cached
    directory.mkdir(exist_ok=True)
    opt_window, select_window = search_windows(outer)
    opt_actual, opt_weights = make_panel(con, opt_window, args.complete_only)
    sel_actual, sel_weights = make_panel(con, select_window, args.complete_only)
    sites = con.execute('SELECT site_id,capacity_kw FROM sites ORDER BY site_id').df()
    reference = con.execute('''SELECT site_id,CASE "group" WHEN 'G1' THEN 0 ELSE 1 END AS cluster
        FROM reference_composition ORDER BY site_id''').df()
    global_assignment = sites[['site_id']].assign(cluster=0)
    # Profiles use only the oldest training window, before any optimization labels.
    window_views(con, opt_window)
    starts = {'global': global_assignment}
    if 'profiles' in args.modes:
        with threadpool_limits(limits=args.threads):
            profile = profiles(con)
            terrain = terrain_profile(args.terrain_dir, profile)
            errors = error_profile(con, args.error_run, opt_window['training_cutoff'])
            sparse_audit = {}
            for k in args.clusters:
                grouped = group_sites(profile, k)[0]
                grouped = assign_sparse(grouped, terrain, errors)
                starts[f'profiles{k}'] = grouped[['site_id', 'cluster']]
                sparse_audit[str(k)] = json.loads(grouped.to_json(orient='records'))
            save_json(directory/'sparse_assignments.json', {
                'error_run': str(args.error_run), 'error_cutoff': opt_window['training_cutoff'],
                'groups': sparse_audit})
    candidates = []
    for trial in range(args.trials):
        params = candidate_params(trial, args.trees)
        for mode, start in starts.items():
            if mode == 'global' and not {'global', 'cycles'} & set(args.modes):
                continue
            assignment, composition = start.copy(), reference.copy()
            rounds = args.cycles+1 if mode == 'global' and 'cycles' in args.modes else 1
            seen = set()
            for cycle in range(rounds):
                signature = tuple(assignment.sort_values('site_id').cluster)
                if signature in seen:
                    break
                seen.add(signature)
                prediction, opt_key = train_predict(con, opt_window, assignment, params, args.output_dir, args.threads)
                errors = residuals(prediction, opt_actual)
                proposed, search = optimize(errors, opt_weights, opt_actual.columns, sites, composition,
                                            args.search_steps, seed=42+trial*100+cycle)
                selection, sel_key = train_predict(con, select_window, assignment, params, args.output_dir, args.threads)
                sel_errors = residuals(selection, sel_actual)
                # Keep the unoptimized reference as a candidate, including when search overfits.
                for label, allocation in [('reference', reference), ('optimized', proposed)]:
                    labels = allocation.set_index('site_id').loc[sel_actual.columns, 'cluster'].to_numpy()
                    value = objective(sel_errors, sel_weights, labels)
                    candidates.append({'trial': trial, 'mode': mode, 'cycle': cycle, 'allocation': label,
                        'selection_W_proxy': value, 'params': params,
                        'assignment': assignment.to_dict('records'), 'composition': allocation.to_dict('records'),
                        'optimization_cache': opt_key, 'selection_cache': sel_key, 'search': search})
                save_json(directory/'progress.json', {'status': 'searching', 'candidates': candidates})
                print(f'{outer["id"]}: {mode} trial={trial} cycle={cycle} search={search}', flush=True)
                assignment, composition = proposed.copy(), proposed.copy()
    best = min(candidates, key=lambda c: c['selection_W_proxy'])
    # Freeze selection before reading any outer actuals or evaluating outer predictions.
    save_json(directory/'selected.json', best)
    composition = pd.DataFrame(best['composition'])
    composition['group'] = composition.cluster.map({0: 'G1', 1: 'G2'})
    con.register('composition', composition[['site_id', 'group']])
    prediction, outer_key = train_predict(con, outer, pd.DataFrame(best['assignment']), best['params'],
                                         args.output_dir, args.threads)
    window_views(con, outer, allow_outer_evaluation=True)
    metrics = evaluate(con, prediction)
    export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', directory/'site_predictions.parquet')
    export(con, 'SELECT * FROM composition ORDER BY site_id', directory/'composition.parquet')
    metrics.update(fold=outer['id'], status='complete', selected=best, outer_cache=outer_key,
                   selection_panel={'sites': list(sel_actual.columns), 'hours': len(sel_actual)},
                   optimization_panel={'sites': list(opt_actual.columns), 'hours': len(opt_actual)},
                   selection_score_basis='SMP-weighted simultaneous observed-panel residuals; not full settlement',
                   candidates=len(candidates))
    artifacts = [directory/n for n in
        ('selected.json', 'progress.json', 'site_predictions.parquet', 'composition.parquet')]
    if 'profiles' in args.modes:
        artifacts.append(directory/'sparse_assignments.json')
    return finish(directory, metrics, artifacts)


def run(args):
    if min(args.threads, args.trials, args.trees, args.search_steps) < 1 or args.cycles < 0:
        raise ValueError('Positive budgets required; cycles may be zero')
    if not args.clusters or any(k < 2 for k in args.clusters):
        raise ValueError('Predictive cluster counts must be at least two')
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if ROOT/'results' not in args.output_dir.parents:
        raise ValueError('Use a dedicated directory under ai_sim/results')
    _, folds = load_plan(args.config)
    if args.fold:
        if set(args.fold)-{w['id'] for w in folds}:
            raise ValueError('Unknown fold')
        folds = [w for w in folds if w['id'] in args.fold]
    external = {}
    if 'profiles' in args.modes:
        args.error_run, external['error_files'] = discover(args.results_root, args.error_run)
        args.terrain_dir = args.terrain_dir.resolve()
        source_manifest = json.loads((args.error_run/'manifest.json').read_text(encoding='utf-8'))
        for name in ('sites.parquet', 'train_targets.parquet'):
            if source_manifest['input_sha256'].get(name) != digest(args.data_dir/name):
                raise ValueError(f'CUDA 1028 input differs: {name}')
        external['terrain_files'] = {name: digest(args.terrain_dir/name)
                                     for name in ('manifest.json', 'site_quality.json')}
        print(f'Using final historical blend predictions: {args.error_run}', flush=True)
    manifest = {'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'},
        'sparse_group_sources': external,
        'features': PAIRED_FEATURES, 'config': digest(args.config),
        'inputs': {p.name: digest(p) for p in args.data_dir.glob('*.parquet')},
        'sources': {p.name: digest(p) for p in (ROOT/'scripts').glob('*.py')},
        'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__}}
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError('Use a new run directory or --resume')
        if json.loads((args.output_dir/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Resume inputs/code/configuration differ; use a new directory')
    else:
        args.output_dir.mkdir(parents=True)
        save_json(args.output_dir/'manifest.json', manifest)
    check_cuda(args.threads)
    with connect() as con:
        attach_data(con, args.data_dir)
        validate_inputs(con)
        attach_pair(con)
        read_view(con, 'reference_composition', args.data_dir/'reference_composition.parquet')
        results = []
        for outer in folds:
            results.append(run_fold(con, outer, args))
            summary = {'status': 'complete' if len(results) == len(folds) else 'partial',
                       'all_12_months': len(results) == 12,
                       'folds': [{'fold': r['fold'], 'site_equal_nmae': r['site_equal_nmae'],
                                  'settlement': r['settlement']} for r in results]}
            save_json(args.output_dir/'summary.json', summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--modes', nargs='+', choices=['global', 'cycles', 'profiles'], default=['global', 'cycles', 'profiles'])
    p.add_argument('--clusters', nargs='+', type=int, default=[2, 3])
    p.add_argument('--cycles', type=int, default=6)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--trees', type=int, default=1200)
    p.add_argument('--search-steps', type=int, default=20000)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--fold', action='append')
    p.add_argument('--complete-only', action='store_true')
    p.add_argument('--results-root', type=Path, default=ROOT/'results', help='Search local *1028* run directories')
    p.add_argument('--error-run', type=Path, help='Explicit completed CUDA 1028 run; otherwise auto-discover')
    p.add_argument('--terrain-dir', type=Path, default=ROOT/'data/terrain_features_v2')
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), indent=2))
