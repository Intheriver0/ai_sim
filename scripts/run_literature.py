"""Frozen methodology x geographic-region ablations, separate from automatic tuning."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from literature_models import PRESETS, check_backend, fit_save, restore, predict
from paired_model import attach_pair, PAIRED_FEATURES
from pipeline_common import connect, read_view, export
from seasonal_splits import load_plan, window_views
from time_splits import attach_data, validate_inputs, digest, stamp, interval
from train_cycles import completed, finish
from train_fixed import evaluate
from train_grouped import save_json

ROOT = Path(__file__).resolve().parents[1]


def assignments(sites, directory):
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['artifacts'].items():
        if digest(directory/name) != sha:
            raise ValueError(f'Changed region artifact: {name}')
    result = {'global': sites[['site_id']].assign(cluster=0)}
    for name, info in manifest['candidates'].items():
        if not info['feasible']:
            continue
        frame = pd.read_csv(directory/f'{name}.csv', dtype={'site_id':str})[['site_id','cluster']]
        if (frame.site_id.duplicated().any() or set(frame.site_id) != set(sites.site_id)
                or frame.cluster.isna().any() or frame.cluster.nunique() != int(name[-1])
                or not np.equal(frame.cluster, frame.cluster.astype(int)).all()):
            raise ValueError(f'Invalid region assignment: {name}')
        if name.startswith('ridge'):
            pairs = pd.read_csv(directory/'ridge_pairs.csv', dtype={'site_a':str, 'site_b':str})
            pairs = pairs[pairs.barrier.eq(True)]
            labels = frame.set_index('site_id').cluster
            if (pairs.site_a.map(labels) == pairs.site_b.map(labels)).any():
                raise ValueError('Region assignment violates a ridge constraint')
        result[name] = frame
    if not any(n.startswith('ridge') for n in result):
        raise ValueError('No feasible ridge partition; inspect terrain thresholds before training')
    return result


def daily_penalty(con, predictions, window):
    con.register('penalty_predictions', predictions)
    rows = con.execute('''WITH g AS (
        SELECT ts,"group",sum(forecast_kwh-energy) AS residual
        FROM penalty_predictions JOIN settlement_actual USING(site_id,ts)
        JOIN composition USING(site_id) GROUP BY ts,"group")
        SELECT CAST(ts-INTERVAL '1 hour' AS DATE) AS day,
        sum(.1*smp*k*abs(residual)) AS W FROM g JOIN settlement_prices USING(ts) GROUP BY day''').df()
    days = pd.date_range(window['start'], window['end'], inclusive='left', freq='D').date
    # Explicit calendar preserves gaps: an unscored day has zero penalty contribution.
    return rows.set_index('day').reindex(pd.to_datetime(days)).fillna(0).rename_axis('day').reset_index()


def reduction(reference, candidate):
    """Positive means lower penalty. Synchronized 7-day circular blocks within each month."""
    if not reference.day.equals(candidate.day):
        raise ValueError('Daily comparison keys differ')
    delta = reference.W.to_numpy()-candidate.W.to_numpy()
    rng = np.random.default_rng(42)
    sampled = np.zeros(2000)
    months = pd.to_datetime(reference.day).dt.to_period('M')
    for month in months.unique():
        values = delta[months.eq(month).to_numpy()]
        starts = rng.integers(0, len(values), size=(2000, int(np.ceil(len(values)/7))))
        indices = ((starts[:,:,None]+np.arange(7)) % len(values)).reshape(2000,-1)[:,:len(values)]
        sampled += values[indices].sum(axis=1)
    base = float(reference.W.sum())
    return {'W_reduction_won': float(delta.sum()),
            'W_reduction_percent': float(100*delta.sum()/base) if base else None,
            'W_reduction_95pct_block_interval_won': np.quantile(sampled, [.025,.975]).tolist(),
            'bootstrap': '2000 paired circular 7-day blocks, stratified by month; exploratory, not multiplicity adjusted'}


def train_region(con, window, assignment, name, destination, threads):
    cached = completed(destination)
    if cached:
        read_view(con, 'restored_predictions', destination/'predictions.parquet')
        return con.execute('SELECT * FROM restored_predictions ORDER BY ts,site_id').df()
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    con.register('region_assignment', assignment)
    results = []
    for cluster in sorted(assignment.cluster.unique()):
        train = con.execute(f'''SELECT p.*,t.target_kwh FROM paired p
            JOIN targets t USING(site_id,ts) JOIN region_assignment USING(site_id)
            WHERE cluster={int(cluster)} AND training_eligible
            AND {stamp(window['training_start'])} <= ts AND ts <= {stamp(window['training_cutoff'])}
            ORDER BY ts,site_id''').df()
        frame = con.execute(f'''SELECT p.* FROM paired p JOIN region_assignment USING(site_id)
            WHERE cluster={int(cluster)} AND {interval(window['start'],window['end'])}
            ORDER BY ts,site_id''').df()
        print(f'{window["id"]} {destination.parent.name}/{name} cluster={cluster} rows={len(train):,}', flush=True)
        model_dir = destination/'models'/f'cluster{cluster}'
        model = fit_save(train, name, model_dir, threads)
        del train
        prediction = predict(model, frame, name, threads)
        sample = frame.iloc[::max(1, len(frame)//128)]
        expected = predict(model, sample, name, threads).forecast_kwh.to_numpy()
        del model
        loaded = restore(name, model_dir)
        np.testing.assert_allclose(expected, predict(loaded, sample, name, threads).forecast_kwh,
                                   rtol=1e-5, atol=1e-4)
        del loaded, frame
        results.append(prediction)
    predictions = pd.concat(results, ignore_index=True).sort_values(['ts','site_id'])
    con.register('region_predictions', predictions)
    export(con, 'SELECT * FROM region_predictions', destination/'predictions.parquet')
    finish(destination, {'status':'complete','seconds':time.monotonic()-started},
           [destination/'predictions.parquet']+[p for p in (destination/'models').rglob('*') if p.is_file()])
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/preprocessed_v1')
    p.add_argument('--regions-dir', type=Path, default=ROOT/'data/ridge_regions_v1')
    p.add_argument('--config', type=Path, default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=list(PRESETS), default=['lgbm_l2'])
    p.add_argument('--fold', action='append')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or ROOT/'results' not in args.output_dir.resolve().parents:
        raise ValueError('Use positive threads and a dedicated output directory under ai_sim/results')
    names = list(dict.fromkeys(['lgbm_l2']+args.models))
    _, windows = load_plan(args.config)
    if args.fold:
        if set(args.fold)-{w['id'] for w in windows}:
            raise ValueError('Unknown fold')
        windows = [w for w in windows if w['id'] in args.fold]
    versions = {}
    for package in ('numpy','pandas','duckdb','lightgbm','catboost','torch','tabm','scipy','scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {'procedure':'fixed factorial ablations; no tuning, early stopping, selection or submission',
        'presets':{n:PRESETS[n] for n in names}, 'paired_features':PAIRED_FEATURES,
        'folds':[w['id'] for w in windows], 'config':digest(args.config), 'threads':args.threads,
        'inputs':{f.name:digest(f) for f in args.data_dir.glob('*.parquet')},
        'region_manifest':digest(args.regions_dir/'manifest.json'), 'versions':versions,
        'sources':{f.name:digest(f) for f in (ROOT/'scripts').glob('*.py')}}
    if args.output_dir.exists():
        if not args.resume or json.loads((args.output_dir/'manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Resume requires unchanged inputs, code, versions and experiment settings')
    else:
        args.output_dir.mkdir(parents=True)
        save_json(args.output_dir/'manifest.json', manifest)
    with connect() as con:
        attach_data(con, args.data_dir)
        validate_inputs(con)
        attach_pair(con)
        read_view(con, 'composition', args.data_dir/'reference_composition.parquet')
        sites = con.execute('SELECT * FROM sites ORDER BY site_id').df()
        region_manifest = json.loads((args.regions_dir/'manifest.json').read_text(encoding='utf-8'))
        if region_manifest['sites_sha256'] != digest(args.data_dir/'sites.parquet'):
            raise ValueError('Regions were built for different sites')
        regions = assignments(sites, args.regions_dir)
        gpu = {n:check_backend(n,args.threads) for n in names}
        save_json(args.output_dir/'preflight.json', {'gpu':gpu,'regions':list(regions),'status':'complete'})
        if args.preflight_only:
            return
        scores, daily = [], {}
        save_json(args.output_dir/'comparison.json', {'status':'running'})
        for w in windows:
            window_views(con, w, allow_outer_evaluation=True)
            for region, assignment in regions.items():
                for name in names:
                    key = f'{region}/{name}'
                    dest = args.output_dir/w['id']/region/name
                    predictions = train_region(con,w,assignment,name,dest,args.threads)
                    metrics = evaluate(con,predictions)
                    scores.append(dict(fold=w['id'],region=region,model=name,**metrics))
                    penalty = daily_penalty(con,predictions,w)
                    if metrics['settlement'] is not None:
                        np.testing.assert_allclose(penalty.W.sum(),metrics['settlement']['W_won'],rtol=1e-10)
                    daily.setdefault(key,[]).append(penalty)
                    save_json(dest/'metrics.json',metrics)
                    save_json(args.output_dir/'fold_metrics.json',scores)
        daily = {key:pd.concat(parts,ignore_index=True) for key,parts in daily.items()}
        comparisons = []
        for region in regions:
            for name in names:
                candidate = [s for s in scores if s['region']==region and s['model']==name]
                refs = [('global','lgbm_l2'),('global',name)]
                if region.startswith('ridge'):
                    refs.append((region.replace('ridge','distance'),name))
                for ref_region, ref_model in dict.fromkeys(refs):
                    if (ref_region,ref_model)==(region,name):
                        continue
                    reference = [s for s in scores if s['region']==ref_region and s['model']==ref_model]
                    deltas = np.array([a['site_equal_nmae']-b['site_equal_nmae'] for a,b in zip(reference,candidate)])
                    result = reduction(daily[f'{ref_region}/{ref_model}'],daily[f'{region}/{name}'])
                    result.update(candidate=f'{region}/{name}',reference=f'{ref_region}/{ref_model}',
                        month_equal_nmae_reduction_pp=float(deltas.mean()*100),
                        nmae_months_improved=int((deltas>0).sum()), months=len(windows))
                    if not sum(s['complete_253_hours'] for s in candidate):
                        result.update(W_reduction_won=None,W_reduction_percent=None,
                                      W_reduction_95pct_block_interval_won=None)
                    comparisons.append(result)
        save_json(args.output_dir/'comparison.json', {'status':'complete' if len(windows)==12 else 'partial',
            'all_12_months':len(windows)==12, 'comparisons':comparisons,
            'note':'Positive reductions are improvements; fixed reference G1/G2; no outer-score winner selection.'})
        print(f'Finished {len(windows)} months. See {args.output_dir}/comparison.json',flush=True)


if __name__ == '__main__':
    main()
