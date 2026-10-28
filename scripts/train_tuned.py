"""Past-only LightGBM tuning and scalar blending on seasonal v2 windows."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

from fixed_model import FEATURES, PARAMS, WEATHER, features, predict
from gpu_training import check_cuda, require_cuda_params
from pipeline_common import connect, export, read_view
from seasonal_splits import load_plan, window_views
from time_splits import attach_data, digest, validate_inputs
from train_fixed import evaluate
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = np.linspace(0, 1, 21)
COLUMNS = ', '.join(['site_id', 'ts', 'capacity_kw', 'lat', 'lon'] + WEATHER)


def aligned(frame):
    if frame[['site_id', 'ts']].isna().any().any() or frame.duplicated(['site_id', 'ts']).any():
        raise ValueError('Null or duplicate prediction key')
    values = frame.forecast_kwh.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid prediction value')
    return frame.set_index(['site_id', 'ts']).sort_index()[['forecast_kwh']]


def blend(fixed, tuned, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between zero and one')
    a, b = aligned(fixed), aligned(tuned)
    if not a.index.equals(b.index):
        raise ValueError('Prediction keys differ')
    return (alpha*a + (1-alpha)*b).reset_index()


def daytime_xy(frame):
    x = features(frame)
    day = x.elev > 0
    if not day.any():
        raise ValueError('No daytime training examples')
    return x.loc[day], (frame.loc[day, 'target_kwh']/x.loc[day, 'cap']).clip(0, 1.2).astype('float32')


def fit_round(con, round_id, params, threads, early_stop):
    require_cuda_params(params)
    def xy(view):
        return daytime_xy(con.execute(f'SELECT {COLUMNS},target_kwh FROM {view} '
                                      f'WHERE round={round_id} ORDER BY ts,site_id').df())
    iterations = params['n_estimators']
    if early_stop:
        x, y = xy('fit_features')
        stop_x, stop_y = xy('stop_features')
        initial = lgb.LGBMRegressor(**params, n_jobs=threads).fit(
            x, y, eval_set=[(stop_x, stop_y)], eval_metric='l1',
            callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)])
        iterations = initial.best_iteration_
        del x, y, stop_x, stop_y, initial
    x, y = xy('training_features')
    model = lgb.LGBMRegressor(**dict(params, n_estimators=iterations), n_jobs=threads).fit(x, y)
    return model.booster_, iterations, len(x)


def checked_result(destination):
    marker = destination/'metrics.json'
    if not marker.exists():
        return None
    result = json.loads(marker.read_text(encoding='utf-8'))
    for name, expected in result['artifacts_sha256'].items():
        if digest(destination/name) != expected:
            raise ValueError(f'Cached artifact changed: {destination/name}')
    return result


def run_model(con, window, params, early_stop, output, threads):
    require_cuda_params(params)
    # Exclude outer-specific names so identical historical blocks can be reused.
    identity = {'window': {k: str(v) for k, v in window.items() if k not in ('id', 'inner')},
                'params': params, 'early_stop': early_stop}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = output/'cache'/key
    window_views(con, window, allow_outer_evaluation=True)
    cached = checked_result(destination)
    if cached is not None:
        read_view(con, 'cached_predictions', destination/'predictions.parquet')
        return con.execute('SELECT * FROM cached_predictions').df(), cached
    destination.mkdir(parents=True, exist_ok=True)
    print(f'{window["id"]}: starting CUDA training (early_stop={early_stop})', flush=True)
    started = time.monotonic()
    rounds, training, artifacts = [], [], []
    for round_id in (10, 17):
        model, iterations, rows = fit_round(con, round_id, params, threads, early_stop)
        frame = con.execute(f'SELECT {COLUMNS} FROM diagnostic_features '
                            f'WHERE round={round_id} ORDER BY ts,site_id').df()
        prediction = predict(model, frame, threads)
        if window['purpose'] == 'outer_evaluation':
            model_dir = destination/'models'
            model_dir.mkdir(exist_ok=True)
            path = model_dir/f'round{round_id}.txt'
            model.save_model(str(path))
            sample = np.linspace(0, len(frame)-1, min(1000, len(frame)), dtype=int)
            np.testing.assert_allclose(prediction.forecast_kwh.iloc[sample],
                predict(lgb.Booster(model_file=str(path)), frame.iloc[sample], threads).forecast_kwh,
                rtol=1e-12, atol=1e-12)
            artifacts.append(path)
        rounds.append(prediction)
        training.append({'round': round_id, 'daytime_rows': rows, 'iterations': iterations})
        del model, frame
    prediction = blend(rounds[0], rounds[1], .5)
    metrics = evaluate(con, prediction)
    path = destination/'predictions.parquet'
    export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', path)
    artifacts.append(path)
    metrics.update(identity=identity, training=training, cache_key=key,
                   seconds=time.monotonic()-started,
                   artifacts_sha256={p.relative_to(destination).as_posix(): digest(p) for p in artifacts})
    save_json(destination/'metrics.json', metrics)
    print(f'{window["id"]}: NMAE={metrics["site_equal_nmae"]:.6f}, {metrics["seconds"]:.1f}s', flush=True)
    return prediction, metrics


def suggest(trial, max_trees):
    return dict(PARAMS, n_estimators=max_trees, metric='l1',
        learning_rate=trial.suggest_float('learning_rate', .015, .12, log=True),
        num_leaves=trial.suggest_int('num_leaves', 15, 127),
        min_child_samples=trial.suggest_int('min_child_samples', 50, 1000, log=True),
        subsample=trial.suggest_float('subsample', .6, 1.),
        colsample_bytree=trial.suggest_float('colsample_bytree', .6, 1.),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10., log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10., log=True))


def tune(con, windows, destination, output, threads, trials, timeout, max_trees, fixed_params):
    storage = optuna.storages.RDBStorage('sqlite:///'+(destination/'study.db').resolve().as_posix())
    try:
        study = optuna.create_study(study_name='lightgbm', direction='minimize',
                                   storage=storage, load_if_exists=True)
        return tune_study(study, con, windows, destination, output, threads, trials, timeout, max_trees, fixed_params)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def tune_study(study, con, windows, destination, output, threads, trials, timeout, max_trees, fixed_params):
    baseline = [run_model(con, w, fixed_params, False, output, threads)[1] for w in windows]
    fixed_score = float(np.mean([m['site_equal_nmae'] for m in baseline]))
    # Sequential single-process runner: a RUNNING trial means an interrupted process.
    # Retry its exact parameters; completed window artifacts are reused.
    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
        study.enqueue_trial(trial.params)
    # Persist compute time per completed block; process downtime is not trial time.
    used = sum(t.user_attrs.get('elapsed_seconds', 0.) for t in study.trials if t.state.is_finished())
    started = time.monotonic()
    while len(study.get_trials(states=(TrialState.COMPLETE,))) < trials and used + time.monotonic()-started < timeout:
        # Seed each trial independently; restart does not reset an in-memory sampler sequence.
        study.sampler = optuna.samplers.TPESampler(seed=42+len(study.trials))
        trial = study.ask()
        tick = time.monotonic()
        params = suggest(trial, max_trees)
        trial.set_user_attr('model_params', params)
        trial.set_user_attr('elapsed_seconds', 0.)
        scores = []
        for window in windows:
            scores.append(run_model(con, window, params, True, output, threads)[1]['site_equal_nmae'])
            trial.set_user_attr('elapsed_seconds', time.monotonic()-tick)
        trial.set_user_attr('window_scores', scores)
        trial.set_user_attr('elapsed_seconds', time.monotonic()-tick)
        study.tell(trial, float(np.mean(scores)))
    completed = study.get_trials(states=(TrialState.COMPLETE,))
    best = study.best_trial if completed else None
    use_tuned = best is not None and best.value < fixed_score
    result = {'objective': 'month_equal_site_nmae_proxy', 'fixed_score': fixed_score,
              'selected_score': best.value if use_tuned else fixed_score,
              'params': best.user_attrs['model_params'] if use_tuned else fixed_params,
              'early_stop': use_tuned, 'selected': 'tuned' if use_tuned else 'fixed',
              'completed_trials': len(completed), 'best_trial': best.number if best else None,
              'trial_seconds_used': used+time.monotonic()-started,
              'budget_note': 'soft timeout checked between trials; baseline excluded',
              'model_windows': [w['id'] for w in windows]}
    save_json(destination/'selection.json', result)
    save_json(destination/'trials.json', [{'number': t.number, 'state': t.state.name,
        'value': t.value, 'params': t.params, 'user_attrs': t.user_attrs} for t in study.trials])
    return result


def choose_alpha(curves):
    means = np.mean(np.asarray(curves), axis=0)
    # Prefer the fixed model on an exact tie, avoiding a gratuitous mixture.
    index = len(means)-1-int(np.argmin(means[::-1]))
    return float(ALPHAS[index]), means.tolist()


def select_blend(con, windows, selected, fixed_params, output, threads):
    curves = []
    for window in windows:
        fixed, _ = run_model(con, window, fixed_params, False, output, threads)
        tuned, _ = run_model(con, window, selected['params'], selected['early_stop'], output, threads)
        curves.append([evaluate(con, blend(fixed, tuned, float(a)))['site_equal_nmae'] for a in ALPHAS])
    alpha, curve = choose_alpha(curves)
    return {'alpha_fixed': alpha, 'alphas': ALPHAS.tolist(), 'month_equal_nmae': curve,
            'monthly_curves': curves, 'blend_windows': [w['id'] for w in windows],
            'objective': 'month_equal_site_nmae_proxy'}


def summarize(output, windows, smoke):
    folds = []
    for w in windows:
        path = output/w['id']/'result.json'
        if path.exists():
            folds.append(json.loads(path.read_text(encoding='utf-8')))
    candidates = {}
    for name in ('fixed', 'selected', 'blend'):
        metrics = [f['metrics'][name] for f in folds]
        if metrics:
            candidates[name] = {'month_equal_nmae': float(np.mean([m['site_equal_nmae'] for m in metrics])),
                'worst_month_nmae': max(m['site_equal_nmae'] for m in metrics),
                'complete_253_hours': sum(m['complete_253_hours'] for m in metrics),
                'W_available_hours_won': sum(m['settlement']['W_won'] for m in metrics if m['settlement'])
                    if any(m['settlement'] for m in metrics) else None}
    report = {'status': 'complete' if len(folds) == len(windows) else 'partial', 'smoke': smoke,
              'all_12_months': len(folds) == 12 and not smoke, 'candidates': candidates, 'folds': folds}
    save_json(output/'summary.json', report)
    lines = ['# Tuning and blending', '',
        'SMOKE ONLY: reduced trees and inner windows; not a performance comparison.' if smoke else
        'Past-only selection; reference G1/G2 fixed. NMAE is a proxy; W covers only complete actuals.', '',
        '| Model | Month-equal NMAE | Worst month NMAE | Complete hours | W (won) |',
        '|---|---:|---:|---:|---:|']
    for name, m in candidates.items():
        cost = f"{m['W_available_hours_won']:,.0f}" if m['W_available_hours_won'] is not None else 'N/A'
        lines.append(f"| {name} | {m['month_equal_nmae']:.4%} | {m['worst_month_nmae']:.4%} | {m['complete_253_hours']} | {cost} |")
    lines += ['', '| Outer month | Selected model | Fixed weight | Fixed NMAE | Selected NMAE | Blend NMAE |',
              '|---|---|---:|---:|---:|---:|']
    for f in folds:
        m = f['metrics']
        lines.append(f"| {f['fold']} | {f['selected']} | {f['alpha_fixed']:.2f} | {m['fixed']['site_equal_nmae']:.4%} | {m['selected']['site_equal_nmae']:.4%} | {m['blend']['site_equal_nmae']:.4%} |")
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def run(data_dir, output, config, fold_ids=None, threads=4, trials=20, timeout=14400,
        max_trees=1200, resume=False, smoke=False):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if ROOT/'results' not in output.parents or data_dir == output or data_dir in output.parents:
        raise ValueError('Output must be inside ai_sim/results and outside inputs')
    if threads < 1 or trials < 1 or timeout <= 0 or max_trees < 1:
        raise ValueError('Budgets and threads must be positive')
    _, windows = load_plan(config)
    if fold_ids:
        if set(fold_ids)-{w['id'] for w in windows}:
            raise ValueError('Unknown outer fold')
        windows = [w for w in windows if w['id'] in fold_ids]
    if smoke:
        windows, trials, max_trees = windows[:1], 1, 12
    fixed_params = dict(PARAMS, n_estimators=12) if smoke else dict(PARAMS)
    gpu = check_cuda(threads)
    print('Hashing inputs and freezing the CUDA run configuration...', flush=True)
    manifest = {'folds': [w['id'] for w in windows], 'threads': threads, 'trials': trials,
        'gpu': gpu,
        'timeout_seconds_per_outer': timeout, 'max_trees': max_trees, 'smoke': smoke,
        'fixed_params': fixed_params, 'features': FEATURES, 'alphas': ALPHAS.tolist(),
        'selection_objective': 'month_equal_site_nmae_proxy',
        'versions': {'python': platform.python_version(), 'lightgbm': lgb.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'optuna': optuna.__version__},
        'source_sha256': {p.name: digest(p) for p in (ROOT/'scripts').glob('*.py')},
        'config_sha256': digest(config),
        'input_sha256': {name: digest(data_dir/name) for name in ('report.json', 'sites.parquet',
            'reference_composition.parquet', 'train_targets.parquet', 'train_prices.parquet',
            'train_round10.parquet', 'train_round17.parquet')}}
    if output.exists():
        if not resume:
            raise FileExistsError('Use a new output directory or --resume')
        if json.loads((output/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Configuration, source, versions or inputs changed; use a new output')
    else:
        output.mkdir(parents=True)
        save_json(output/'manifest.json', manifest)
        git = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT),
                              'rev-parse', 'HEAD'], capture_output=True, text=True)
        save_json(output/'git.json', {'head': git.stdout.strip()})
    with connect() as con:
        print('Validating input data before training...', flush=True)
        attach_data(con, data_dir)
        validate_inputs(con)
        read_view(con, 'composition', data_dir/'reference_composition.parquet')
        for window in windows:
            destination = output/window['id']
            destination.mkdir(exist_ok=True)
            model_windows = [w for w in window['inner'] if w['purpose'] == 'model_selection']
            blend_windows = [w for w in window['inner'] if w['purpose'] == 'blend_selection']
            if smoke:
                model_windows, blend_windows = model_windows[:1], blend_windows[:1]
            selection_path = destination/'selection.json'
            selected = (json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists()
                else tune(con, model_windows, destination, output, threads, trials, timeout, max_trees, fixed_params))
            mixture = select_blend(con, blend_windows, selected, fixed_params, output, threads)
            save_json(destination/'blend.json', mixture)
            # Persist all choices before opening the outer evaluation window.
            fixed, fixed_metrics = run_model(con, window, fixed_params, False, output, threads)
            tuned, tuned_metrics = run_model(con, window, selected['params'], selected['early_stop'], output, threads)
            prediction = blend(fixed, tuned, mixture['alpha_fixed'])
            metrics = evaluate(con, prediction)
            export(con, 'SELECT * FROM site_predictions ORDER BY ts,site_id', destination/'blend_predictions.parquet')
            save_json(destination/'result.json', {'fold': window['id'], 'selected': selected['selected'],
                'alpha_fixed': mixture['alpha_fixed'], 'training_cutoff': window['training_cutoff'],
                'metrics': {'fixed': fixed_metrics, 'selected': tuned_metrics, 'blend': metrics}})
            summarize(output, windows, smoke)
    return summarize(output, windows, smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    parser.add_argument('--fold', action='append')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--trials', type=int, default=20, help='Completed trials per outer month')
    parser.add_argument('--timeout', type=float, default=14400, help='Cumulative soft trial budget per outer, seconds')
    parser.add_argument('--max-trees', type=int, default=1200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One outer/inner month, one trial, 12 trees; not a score')
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, args.config, args.fold, args.threads,
                 args.trials, args.timeout, args.max_trees, args.resume, args.smoke)
    print(json.dumps({k: v for k, v in result.items() if k != 'folds'}, indent=2))


if __name__ == '__main__':
    main()
