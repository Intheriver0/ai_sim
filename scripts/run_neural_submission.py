"""One neural model: past-only G1/G2 selection, twelve-month audit, final CSVs."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import connect, read_view, export, check_keys, check_time, reject
from parameter_search import spec
from residual_groups import residuals, objective, optimize
from run_literature import assignments
from run_parameter_methods import Experiment, key
from run_pipeline import (attach_all, deployment_window, seasonal_windows, span,
                          set_composition, write_submission)
from scoring import validate_composition
from seasonal_splits import load_plan, month_shift, window_views
from sparse_site_groups import terrain_profile
from time_splits import digest, interval
from train_cycles import completed, finish, make_panel
from train_fixed import evaluate
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = spec(dict(hidden=64, lr=.001, weight_decay=.0003))
# Only training/inference dependencies are required for importing old predictions.
TRAINING_SOURCES = ('run_parameter_methods.py', 'parameter_models.py', 'parameter_updates.py',
                    'literature_models.py', 'paired_model.py', 'fixed_model.py',
                    'solar_time.py', 'time_splits.py', 'pipeline_common.py',
                    'run_literature.py', 'sparse_site_groups.py')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['tabm', 'hyper_head', 'mlp'], default='tabm')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    p.add_argument('--regions-dir', type=Path, default=ROOT/'data/ridge_regions_v1')
    p.add_argument('--terrain-dir', type=Path, default=ROOT/'data/terrain_features_v2')
    p.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--prediction-run', type=Path, help='Optional completed fixed run with local prediction caches')
    p.add_argument('--max-epochs', type=int, default=45)
    p.add_argument('--search-steps', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--stage', choices=['all', 'preflight', 'compare', 'submit'], default='all')
    p.add_argument('--resume', action='store_true')
    return p


def manifest(args):
    versions = {n: importlib.metadata.version(n) for n in
                ('torch', 'tabm', 'numpy', 'pandas', 'duckdb', 'scikit-learn', 'lightgbm')}
    result = dict(options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                           if k not in ('stage', 'resume')},
                  versions=versions, config=digest(args.config),
                  regions=digest(args.regions_dir/'manifest.json'),
                  inputs={p.name: digest(p) for p in args.data_dir.glob('*.parquet')},
                  preprocessing=digest(args.data_dir/'report.json'),
                  sources={p.name: digest(p) for p in (ROOT/'scripts').glob('*.py')},
                  procedure='Fixed ridge_k4 neural model; annual past optimization then annual selection; '
                            'freeze G1/G2 before outer scores; final deployment refit')
    if args.model == 'hyper_head':
        result['terrain'] = {n: digest(args.terrain_dir/n) for n in ('manifest.json', 'site_quality.json')}
    if args.prediction_run:
        result['prediction_manifest'] = digest(args.prediction_run/'manifest.json')
    return result


def validate_prediction_run(args, current):
    if not args.prediction_run:
        return
    old = json.loads((args.prediction_run/'manifest.json').read_text(encoding='utf-8'))
    expected = dict(model=args.model, region='ridge_k4', max_epochs=args.max_epochs,
                    seed=args.seed, threads=args.threads, methods=['fixed'])
    for name, value in expected.items():
        if old['options'].get(name) != value:
            raise ValueError(f'Prediction run configuration differs: {name}')
    for name in ('config', 'regions', 'inputs'):
        if old.get(name) != current[name]:
            raise ValueError(f'Prediction run identity differs: {name}')
    for name in TRAINING_SOURCES:
        if old['sources'].get(name) != current['sources'][name]:
            raise ValueError(f'Prediction run training source differs: {name}')
    for name in ('torch', 'tabm', 'numpy', 'pandas', 'duckdb'):
        if old['versions'].get(name) != current['versions'][name]:
            raise ValueError(f'Prediction run library differs: {name}')
    if args.model == 'hyper_head' and old.get('terrain') != current['terrain']:
        raise ValueError('Prediction run lacks matching terrain provenance; omit --prediction-run to refit')
    report = json.loads((args.prediction_run/'comparison.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or report.get('folds') != 12 or report.get('inner_evaluation') is not False:
        raise ValueError('Prediction run must be a complete twelve-month standalone fixed run')


def validate_predictions(con, prediction, window):
    con.register('checked_prediction', prediction)
    check_keys(con, 'checked_prediction', ['site_id', 'ts'])
    check_time(con, 'checked_prediction')
    where = interval(window['start'], window['end'])
    reject(con, f'''SELECT count(*) FROM (
        (SELECT site_id,ts FROM paired WHERE {where} EXCEPT SELECT site_id,ts FROM checked_prediction)
        UNION ALL (SELECT site_id,ts FROM checked_prediction EXCEPT SELECT site_id,ts FROM paired WHERE {where}))''',
        'Prediction site/hour grid differs')
    reject(con, '''SELECT count(*) FROM checked_prediction WHERE forecast_kwh IS NULL
        OR NOT isfinite(forecast_kwh) OR forecast_kwh<0''', 'Invalid site prediction')
    con.fetchall()  # Release DuckDB's result before the next DataFrame registration.


class Predictions:
    """Share each monthly fit across all folds; crop labels at availability cutoffs."""
    def __init__(self, con, assignment, args):
        self.con, self.args = con, args
        self.experiment = Experiment(con, assignment, args.model, args.output_dir,
                                     args.threads, args.seed)

    def canonical(self, window):
        if window['purpose'] == 'deployment':
            return window
        # Only predictions extend to month end. Selection labels remain strictly past.
        return dict(window, id=f'oof_{window["start"]:%Y_%m}', purpose='prediction',
                    end=month_shift(window['start'], 1))

    def directory(self, window):
        return self.args.output_dir/'cache'/self.canonical(window)['id']/key(CANDIDATE)/f'budget{self.args.max_epochs}'

    def get(self, window):
        full = self.canonical(window)
        directory = self.directory(window)
        if not completed(directory) and self.args.prediction_run and window['purpose'] != 'deployment':
            source = self.args.prediction_run/'cache'/f'outer_{full["start"]:%Y_%m}'/key(CANDIDATE)/f'budget{self.args.max_epochs}'
            if completed(source):
                read_view(self.con, 'imported_prediction', source/'predictions.parquet')
                prediction = self.con.execute('SELECT * FROM imported_prediction ORDER BY ts,site_id').df()
                validate_predictions(self.con, prediction, full)
                directory.mkdir(parents=True, exist_ok=True)
                export(self.con, 'SELECT * FROM imported_prediction', directory/'predictions.parquet')
                finish(directory, dict(imported_from=str(source), source_sha256=digest(source/'predictions.parquet')),
                       [directory/'predictions.parquet'])
                print(f'Imported {full["id"]}', flush=True)
        prediction = self.experiment.predict_window(full, CANDIDATE, self.args.max_epochs)
        validate_predictions(self.con, prediction, full)
        return prediction.loc[(prediction.ts > window['start']) & (prediction.ts <= window['end'])].copy()

    def history(self, windows):
        return pd.concat([self.get(w) for w in windows], ignore_index=True)


def select_groups(con, outer, predictions, args):
    directory = args.output_dir/outer['id']/'group_selection'
    if completed(directory):
        return json.loads((directory/'selected.json').read_text(encoding='utf-8'))
    directory.mkdir(parents=True, exist_ok=True)
    optimization, selection = seasonal_windows(outer)
    opt_actual, opt_weights = make_panel(con, span(optimization), False)
    sel_actual, sel_weights = make_panel(con, span(selection), False)
    sites = con.execute('SELECT site_id,capacity_kw FROM sites ORDER BY site_id').df()
    reference = con.execute('''SELECT site_id,CASE "group" WHEN 'G1' THEN 0 ELSE 1 END AS cluster
        FROM reference_composition ORDER BY site_id''').df()
    errors = residuals(predictions.history(optimization), opt_actual)
    proposed, search = optimize(errors, opt_weights, opt_actual.columns, sites, reference,
                                steps=args.search_steps, seed=args.seed)
    # Never optimize again on selection labels. Keep the reference on ties.
    errors = residuals(predictions.history(selection), sel_actual)
    scores = {}
    for name, composition in [('reference', reference), ('optimized', proposed)]:
        labels = composition.set_index('site_id').loc[sel_actual.columns, 'cluster'].to_numpy()
        scores[name] = objective(errors, sel_weights, labels)
    chosen = 'optimized' if scores['optimized'] < scores['reference'] else 'reference'
    composition = proposed if chosen == 'optimized' else reference
    result = dict(selected=chosen, composition=composition.to_dict('records'), selection_W_proxy=scores,
                  search=search, windows=dict(optimization=optimization, selection=selection),
                  optimization_panel=dict(sites=list(opt_actual.columns), hours=len(opt_actual)),
                  selection_panel=dict(sites=list(sel_actual.columns), hours=len(sel_actual)),
                  note='Observed-panel W proxy, not full settlement. Sites absent from optimization stay fixed.')
    set_composition(con, result)
    save_json(directory/'selected.json', result)  # Freeze before any outer labels are read.
    finish(directory, dict(status='complete'), [directory/'selected.json'])
    print(f'{outer["id"]}: G1/G2={chosen}, selection W proxy={scores}', flush=True)
    return result


def compare_fold(con, outer, predictions, args):
    selected = select_groups(con, outer, predictions, args)
    directory = args.output_dir/outer['id']/'comparison'
    if completed(directory):
        return json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    directory.mkdir(parents=True, exist_ok=True)
    prediction = predictions.get(outer)
    window_views(con, outer, allow_outer_evaluation=True)
    con.register('composition', con.execute('SELECT * FROM reference_composition').df())
    reference = evaluate(con, prediction)
    set_composition(con, selected)
    grouped = evaluate(con, prediction)
    result = dict(fold=outer['id'], allocation=selected['selected'], reference=reference, selected=grouped)
    save_json(directory/'metrics.json', result)
    finish(directory, dict(status='complete'), [directory/'metrics.json'])
    return result


def report(output, results):
    summary = {}
    for name in ('reference', 'selected'):
        metrics = [r[name] for r in results]
        settled = [m['settlement'] for m in metrics if m['settlement'] is not None]
        summary[name] = dict(month_equal_nmae=float(np.mean([m['site_equal_nmae'] for m in metrics])),
                             complete_253_hours=sum(m['complete_253_hours'] for m in metrics),
                             W_won=sum(s['W_won'] for s in settled) if settled else None)
    result = dict(status='complete' if len(results) == 12 else 'partial', folds=results, summary=summary,
                  note='Retrospective diagnostic: these twelve months already informed model shortlist. '
                       'G1/G2 for each fold was selected using only earlier labels. No independent final score claim.')
    save_json(output/'comparison.json', result)
    return result


def submit(con, deploy, predictions, args):
    selected = select_groups(con, deploy, predictions, args)
    directory = args.output_dir/'submission'
    previous = completed(directory)
    if previous:
        if completed(predictions.directory(deploy)) is None:
            raise ValueError('Submission prediction cache is missing')
        for name, expected in previous['model_sha256'].items():
            if digest(args.output_dir/name) != expected:
                raise ValueError(f'Submission model changed: {name}')
        return previous
    prediction = predictions.get(deploy)
    set_composition(con, selected)
    directory.mkdir(parents=True, exist_ok=True)
    capacities = write_submission(con, prediction, directory)
    models = predictions.directory(deploy).parent/'models'
    checkpoints = list(models.glob('cluster*/state.pt'))
    if len(checkpoints) != predictions.experiment.assignment.cluster.nunique():
        raise ValueError('Missing final model checkpoints')
    paths = checkpoints + [p.with_suffix('.sha256') for p in checkpoints]
    result = dict(status='complete', model=args.model, allocation=selected['selected'], capacity_kw=capacities,
                  training_cutoff=deploy['training_cutoff'], site_rows=len(prediction),
                  composition_rows=253, forecast_rows=17520,
                  model_sha256={p.relative_to(args.output_dir).as_posix(): digest(p) for p in paths},
                  note='Local format checks passed. Upload composition.csv and forecast.csv together. '
                       'No test actuals are available; no final score is computed.')
    save_json(directory/'validation.json', result)
    return finish(directory, result, [directory/n for n in
                  ('composition.csv', 'forecast.csv', 'site_predictions.parquet', 'validation.json')])


def prepare(con, args, config):
    attach_all(con, args.data_dir)
    # The competition requires exactly one full hourly year, not just an internally consistent subset.
    first, last, count = con.execute('SELECT min(ts),max(ts),count(*) FROM test_prices').fetchone()
    if count != 8760 or last-first != pd.Timedelta(hours=8759):
        raise ValueError('Submission requires the complete 8760-hour test calendar')
    sites = con.execute('SELECT * FROM sites ORDER BY site_id').df()
    regions = json.loads((args.regions_dir/'manifest.json').read_text(encoding='utf-8'))
    if regions['sites_sha256'] != digest(args.data_dir/'sites.parquet'):
        raise ValueError('Region sites differ')
    available = assignments(sites, args.regions_dir)
    if 'ridge_k4' not in available:
        raise ValueError('The tested ridge_k4 partition is unavailable')
    if args.model == 'hyper_head':
        terrain = terrain_profile(args.terrain_dir, sites)
        terrain = terrain.rename(columns={c: 'terrain_'+c for c in terrain if c != 'site_id'})
        con.register('neural_terrain', terrain)
        con.execute('''CREATE OR REPLACE TEMP VIEW paired AS SELECT p.*,t.* EXCLUDE(site_id)
            FROM (SELECT * FROM paired_train UNION ALL SELECT * FROM paired_test) p
            JOIN neural_terrain t USING(site_id)''')
    con.register('composition', con.execute('SELECT * FROM reference_composition').df())
    validate_composition(con)
    return available['ridge_k4'], deployment_window(con, config)


def preflight(con, folds, deploy, args):
    panels = []
    for outer in folds+[deploy]:
        for windows in seasonal_windows(outer):
            actual, _ = make_panel(con, span(windows), False)
            panels.append(dict(fold=outer['id'], role=windows[0]['purpose'],
                               sites=len(actual.columns), hours=len(actual)))
    from parameter_models import Learner
    import torch
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch required for production neural submission')
    sample = con.execute('''SELECT p.*,target_kwh FROM paired p JOIN targets USING(site_id,ts)
        WHERE training_eligible AND hour(ts)=12 ORDER BY ts,site_id LIMIT 512''').df()
    probe = Learner(sample, args.model, dict(hidden=16, seed=args.seed), args.output_dir/'preflight_models')
    probe.advance(1)
    expected = probe.predict(sample).forecast_kwh.to_numpy()
    probe.load(args.output_dir/'preflight_models/state.pt')
    np.testing.assert_allclose(expected, probe.predict(sample).forecast_kwh, rtol=1e-5, atol=1e-4)
    save_json(args.output_dir/'preflight.json', dict(status='complete', gpu=torch.cuda.get_device_name(0), panels=panels))


def run(args):
    if min(args.max_epochs, args.search_steps, args.threads) < 1:
        raise ValueError('Training and search budgets must be positive')
    for name in ('output_dir', 'data_dir', 'regions_dir', 'terrain_dir', 'config', 'prediction_run'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if ROOT/'results' not in args.output_dir.parents:
        raise ValueError('Use a dedicated output directory under ai_sim/results')
    config, folds = load_plan(args.config)
    current = manifest(args)
    validate_prediction_run(args, current)
    if args.output_dir.exists():
        if not args.resume or json.loads((args.output_dir/'manifest.json').read_text(encoding='utf-8')) != current:
            raise ValueError('Resume requires identical configuration, inputs, code and versions')
    else:
        args.output_dir.mkdir(parents=True)
        save_json(args.output_dir/'manifest.json', current)
    # Exclusive creation avoids two trainers writing the same checkpoints. Removed even on errors.
    lock = args.output_dir/'run.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write('Active neural submission run; remove only after confirming its process stopped.\n')
    try:
        with connect() as con:
            assignment, deploy = prepare(con, args, config)
            preflight(con, folds, deploy, args)
            if args.stage == 'preflight':
                return dict(status='complete', stage='preflight')
            predictions = Predictions(con, assignment, args)
            if args.stage in ('all', 'compare'):
                results = []
                for outer in folds:
                    results.append(compare_fold(con, outer, predictions, args))
                    report(args.output_dir, results)
            if args.stage in ('all', 'submit'):
                results = []
                for outer in folds:
                    directory = args.output_dir/outer['id']
                    if not completed(directory/'comparison') or not completed(directory/'group_selection'):
                        raise ValueError('Complete --stage compare before submission')
                    results.append(json.loads((directory/'comparison/metrics.json').read_text(encoding='utf-8')))
                report(args.output_dir, results)
                return submit(con, deploy, predictions, args)
            return dict(status='complete', stage=args.stage)
    finally:
        lock.unlink()


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args()), default=str, indent=2))
