"""Nested past-only comparison of search methods and parameter-generating models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from paired_model import attach_pair
from pipeline_common import connect, read_view, export
from seasonal_splits import load_plan, window_views
from time_splits import attach_data, validate_inputs, digest, stamp, interval
from train_cycles import completed, finish
from train_fixed import evaluate
from train_grouped import save_json
from run_literature import assignments, daily_penalty, reduction
from parameter_search import METHODS, run_search, spec, SPACE

ROOT = Path(__file__).resolve().parents[1]


def key(candidate):
    return hashlib.sha256(json.dumps(candidate,sort_keys=True).encode()).hexdigest()


def inner_windows(outer):
    windows = [w for w in outer['inner'] if w['purpose']=='model_selection']
    if len(windows)!=12 or any(not w['training_cutoff']<w['start']<w['end']<=outer['training_cutoff'] for w in windows):
        raise ValueError('Expected twelve strictly past model-selection windows')
    return windows


class Experiment:
    def __init__(self,con,assignment,model,root,threads,seed,device='cuda'):
        self.con,self.assignment,self.model,self.root = con,assignment,model,root
        self.threads,self.seed,self.device = threads,seed,device

    def predict_window(self,window,candidate,budget):
        from parameter_models import Learner
        import torch
        torch.set_num_threads(self.threads)
        con = self.con
        directory = self.root/'cache'/window['id']/key(candidate)
        result_dir = directory/f'budget{budget}'
        if completed(result_dir):
            read_view(con,'cached_parameter_predictions',result_dir/'predictions.parquet')
            return con.execute('SELECT * FROM cached_parameter_predictions ORDER BY ts,site_id').df()
        con.register('parameter_assignment',self.assignment)
        results = []
        started = time.monotonic()
        for cluster in sorted(self.assignment.cluster.unique()):
            train = con.execute(f'''SELECT p.*,target_kwh FROM paired p
                JOIN targets USING(site_id,ts) JOIN parameter_assignment USING(site_id)
                WHERE cluster={int(cluster)} AND training_eligible
                AND {stamp(window['training_start'])}<=ts AND ts<={stamp(window['training_cutoff'])}
                ORDER BY ts,site_id''').df()
            frame = con.execute(f'''SELECT p.* FROM paired p JOIN parameter_assignment USING(site_id)
                WHERE cluster={int(cluster)} AND {interval(window['start'],window['end'])}
                ORDER BY ts,site_id''').df()
            state_dir = directory/'models'/f'cluster{cluster}'
            params = dict(candidate['params'],seed=self.seed)
            learner = Learner(train,self.model,params,state_dir,self.device)
            if learner.epoch==0 and 'parent' in candidate and window['purpose']!='outer_evaluation':
                parent = self.root/'cache'/window['id']/key(candidate['parent'])/'models'/f'cluster{cluster}'/'state.pt'
                if parent.exists():
                    learner.load(parent,clone=True)
                    if learner.epoch!=candidate['parent_budget']:
                        raise ValueError('PBT donor checkpoint is at the wrong epoch')
                # On outer refit there is no donor history: replay the selected schedule.
            schedule = candidate['schedule'] or [dict(end=budget,lr=params['lr'],weight_decay=params['weight_decay'])]
            for segment in schedule:
                stop = min(budget,segment['end'])
                if stop>learner.epoch:
                    learner.advance(stop,segment['lr'],segment['weight_decay'])
            if learner.epoch!=budget:
                raise ValueError('Schedule does not cover requested training budget')
            results.append(learner.predict(frame))
            sample = frame.iloc[::max(1,len(frame)//128)]
            expected = learner.predict(sample).forecast_kwh.to_numpy()
            learner.load(state_dir/'state.pt')
            np.testing.assert_allclose(expected,learner.predict(sample).forecast_kwh,rtol=1e-5,atol=1e-4)
            del train,frame,learner
        predictions = pd.concat(results,ignore_index=True).sort_values(['ts','site_id'])
        result_dir.mkdir(parents=True,exist_ok=True)
        con.register('parameter_predictions',predictions)
        export(con,'SELECT * FROM parameter_predictions',result_dir/'predictions.parquet')
        # Epoch checkpoints advance later. A budget completion hashes its immutable predictions;
        # current mutable training state is checked independently by Learner's data identity.
        finish(result_dir,{'seconds':time.monotonic()-started,'budget':budget},[result_dir/'predictions.parquet'])
        return predictions

    def score(self,windows,candidate,budget,objective):
        metrics = []
        for window in windows:
            prediction = self.predict_window(window,candidate,budget)
            window_views(self.con,window)
            metrics.append(evaluate(self.con,prediction))
        if objective=='nmae':
            return float(np.mean([m['site_equal_nmae'] for m in metrics]))
        complete = sum(m['complete_253_hours'] for m in metrics)
        if not complete:
            raise ValueError('No complete 253-site hours in past selection; cannot select on W')
        return sum(m['settlement']['W_won'] for m in metrics if m['settlement'] is not None)/complete


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/preprocessed_v1')
    p.add_argument('--regions-dir',type=Path,default=ROOT/'data/ridge_regions_v1')
    p.add_argument('--terrain-dir',type=Path,default=ROOT/'data/terrain_features_v2')
    p.add_argument('--region',default='ridge_k4',choices=['global','distance_k2','distance_k3','distance_k4','ridge_k2','ridge_k3','ridge_k4'])
    p.add_argument('--model',choices=['mlp','tabm','hyper_head','maml','mlp_sgd',
                                    'mlp_hypergradient','mlp_learned_optimizer'],default='mlp')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['fixed','random','tpe','asha','pbt'])
    p.add_argument('--objective',choices=['W','nmae'],default='nmae')
    p.add_argument('--config',type=Path,default=ROOT/'configs/seasonal_splits_v2.json')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--trials',type=int,default=24)
    p.add_argument('--min-epochs',type=int,default=5)
    p.add_argument('--max-epochs',type=int,default=45)
    p.add_argument('--population',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--fold',action='append')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    return p


def run(args):
    if not 1<=args.min_epochs<=args.max_epochs or args.trials<1 or args.threads<1:
        raise ValueError('Invalid training budgets')
    if args.population<2 and 'pbt' in args.methods:
        raise ValueError('PBT requires population >=2')
    if args.model=='mlp_hypergradient' and 'pbt' in args.methods:
        raise ValueError('Use --methods fixed random tpe gp for hypergradient; PBT would overwrite its learned rates')
    if 'dehb' in args.methods:
        raise ValueError('DEHB is deferred: needs a separate environment; see parameter_methods_pending.md')
    args.output_dir = args.output_dir.resolve()
    if ROOT/'results' not in args.output_dir.parents:
        raise ValueError('Use a dedicated output directory under ai_sim/results')
    _,windows = load_plan(args.config)
    if args.fold:
        if set(args.fold)-{w['id'] for w in windows}:
            raise ValueError('Unknown fold')
        windows = [w for w in windows if w['id'] in args.fold]
    methods = list(dict.fromkeys(['fixed']+args.methods))
    versions = {}
    for package in ('torch','tabm','optuna','smac','ConfigSpace','dehb','numpy','pandas','duckdb'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = dict(options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','preflight_only')},
        versions=versions,space=SPACE,config=digest(args.config),regions=digest(args.regions_dir/'manifest.json'),
        inputs={p.name:digest(p) for p in args.data_dir.glob('*.parquet')},
        sources={p.name:digest(p) for p in (ROOT/'scripts').glob('*.py')},
        procedure='Past-only nested search; reference composition fixed; selected schedule refit before outer evaluation')
    if args.model=='hyper_head':
        manifest['terrain'] = {n:digest(args.terrain_dir/n) for n in ('manifest.json','site_quality.json')}
    # Normalize tuple-valued search definitions exactly as stored JSON.
    manifest = json.loads(json.dumps(manifest))
    if args.output_dir.exists():
        if not args.resume or json.loads((args.output_dir/'manifest.json').read_text())!=manifest:
            raise ValueError('Resume requires identical inputs, versions, source and configuration')
    else:
        args.output_dir.mkdir(parents=True)
        save_json(args.output_dir/'manifest.json',manifest)
    with connect() as con:
        attach_data(con,args.data_dir)
        validate_inputs(con)
        attach_pair(con,name='paired_base')
        sites = con.execute('SELECT * FROM sites ORDER BY site_id').df()
        region_meta = json.loads((args.regions_dir/'manifest.json').read_text())
        if region_meta['sites_sha256']!=digest(args.data_dir/'sites.parquet'):
            raise ValueError('Region sites differ')
        available = assignments(sites,args.regions_dir)
        if args.region not in available:
            raise ValueError('Requested region partition is infeasible/unavailable')
        if args.model=='hyper_head':
            from sparse_site_groups import terrain_profile
            terrain = terrain_profile(args.terrain_dir,sites)
            terrain = terrain.rename(columns={c:'terrain_'+c for c in terrain if c!='site_id'})
            con.register('static_terrain',terrain)
            con.execute('CREATE TEMP VIEW paired AS SELECT p.*,t.* EXCLUDE(site_id) FROM paired_base p JOIN static_terrain t USING(site_id)')
        else:
            con.execute('CREATE TEMP VIEW paired AS SELECT * FROM paired_base')
        read_view(con,'composition',args.data_dir/'reference_composition.parquet')
        for window in windows:
            history = inner_windows(window)
            if args.objective=='W':
                hours = 0
                for w in history:
                    window_views(con,w)
                    hours += con.execute('SELECT count(*) FROM settlement_prices').fetchone()[0]
                if not hours:
                    raise ValueError(f'{window["id"]}: no complete inner hours for W; explicitly use --objective nmae for proxy selection')
        from parameter_models import Learner
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch required for production. Unit tests use explicit CPU fixtures only.')
        if any(m.startswith('smac') for m in methods):
            from smac import Scenario
        if 'dehb' in methods:
            from dehb import DEHB
        sample = con.execute('''SELECT p.*,target_kwh FROM paired p JOIN targets USING(site_id,ts)
            WHERE training_eligible AND hour(ts)=12 ORDER BY ts,site_id LIMIT 512''').df()
        smoke = Learner(sample,args.model,dict(hidden=16,task_steps=2),args.output_dir/'preflight_models')
        smoke.advance(1)
        smoke.predict(sample)
        del smoke
        save_json(args.output_dir/'preflight.json',dict(status='complete',gpu=torch.cuda.get_device_name(0),versions=versions))
        if args.preflight_only:
            return
        experiment = Experiment(con,available[args.region],args.model,args.output_dir,args.threads,args.seed)
        report,daily = [],{}
        references = (['mlp','mlp_sgd'] if args.model=='mlp_hypergradient' else
                      ['mlp'] if args.model=='mlp_learned_optimizer' else [])
        for outer in windows:
            history = inner_windows(outer)
            objective = lambda candidate,budget:experiment.score(history,candidate,budget,args.objective)
            for reference in references:
                reference_name = 'reference_'+reference
                control = Experiment(con,available[args.region],reference,args.output_dir/reference_name,args.threads,args.seed)
                candidate = spec(dict(hidden=64,lr=.001,weight_decay=.0003))
                prediction = control.predict_window(outer,candidate,args.max_epochs)
                window_views(con,outer,allow_outer_evaluation=True)
                metrics = evaluate(con,prediction)
                daily.setdefault(reference_name,[]).append(daily_penalty(con,prediction,outer))
                report.append(dict(fold=outer['id'],method=reference_name,metrics=metrics))
            for method in methods:
                destination = args.output_dir/outer['id']/method
                selected = run_search(method,objective,destination,args.trials,args.min_epochs,args.max_epochs,args.seed,args.population)
                candidate = selected['spec']
                # No parent checkpoints from inner windows can be loaded for this new outer training origin.
                prediction = experiment.predict_window(outer,candidate,args.max_epochs)
                window_views(con,outer,allow_outer_evaluation=True)
                metrics = evaluate(con,prediction)
                penalty = daily_penalty(con,prediction,outer)
                if metrics['settlement'] is not None:
                    np.testing.assert_allclose(penalty.W.sum(),metrics['settlement']['W_won'],rtol=1e-10)
                daily.setdefault(method,[]).append(penalty)
                report.append(dict(fold=outer['id'],method=method,selected=selected,metrics=metrics))
                save_json(destination/'outer_metrics.json',report[-1])
                save_json(args.output_dir/'fold_metrics.json',report)
        comparisons = []
        for reference in ['fixed']+['reference_'+name for name in references]:
            baseline = pd.concat(daily[reference],ignore_index=True)
            base_metrics = [r['metrics'] for r in report if r['method']==reference]
            for method in methods:
                current = [r['metrics'] for r in report if r['method']==method]
                change = reduction(baseline,pd.concat(daily[method],ignore_index=True))
                change.update(method=method,reference=reference,nmae_reduction_pp=float(np.mean([a['site_equal_nmae']-b['site_equal_nmae'] for a,b in zip(base_metrics,current)])*100))
                if not sum(m['complete_253_hours'] for m in current):
                    change.update(W_reduction_won=None,W_reduction_percent=None,W_reduction_95pct_block_interval_won=None)
                comparisons.append(change)
        save_json(args.output_dir/'comparison.json',dict(status='complete' if len(windows)==12 else 'partial',
            folds=len(windows),comparisons=comparisons,
            note='Cached objective work may be shared: observed wall time is NOT a fair optimizer speed benchmark. No outer-score winner selection.'))


if __name__=='__main__':
    run(parser().parse_args())
