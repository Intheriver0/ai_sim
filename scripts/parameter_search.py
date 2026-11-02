"""Search adapters share an objective; full-budget finalists are evaluated uniformly."""
import json
import time

import numpy as np

from train_grouped import save_json
from train_cycles import completed, finish

METHODS = ('fixed','random','tpe','gp','asha','hyperband','tpe_hyperband','smac','smac_mf','dehb','pbt')
SPACE = {'lr':('float',1e-4,.01,True), 'weight_decay':('float',1e-6,.01,True),
         'hidden':('categorical',[16,32,64,128,256])}


def rungs(min_budget,max_budget):
    if not 1<=min_budget<=max_budget:
        raise ValueError('Require 1 <= min budget <= max budget')
    result = [min_budget]
    while result[-1]<max_budget:
        result.append(min(max_budget,result[-1]*3))
    return result


def sample(trial):
    return dict(lr=trial.suggest_float('lr',1e-4,.01,log=True),
                weight_decay=trial.suggest_float('weight_decay',1e-6,.01,log=True),
                hidden=trial.suggest_categorical('hidden',SPACE['hidden'][1]))


def config_space(seed):
    from ConfigSpace import ConfigurationSpace, Float, Categorical
    space = ConfigurationSpace(seed=seed)
    space.add([Float('lr',(1e-4,.01),default=.001,log=True),
               Float('weight_decay',(1e-6,.01),default=.0003,log=True),
               Categorical('hidden',SPACE['hidden'][1],default=64)])
    return space


def spec(params):
    return {'params':{k:v.item() if isinstance(v,np.generic) else v for k,v in params.items()},'schedule':[]}


def run_search(method, objective, directory, trials, min_budget, max_budget, seed=42, population=4):
    """objective(spec, budget) persists every score/checkpoint; no outer labels here.

    Optimizer proposals are replayed deterministically on resume, reusing completed
    objective evaluations. Native scheduler state is kept in this fresh replay only.
    """
    if method not in METHODS or trials<1:
        raise ValueError('Unknown method or invalid trial budget')
    budgets = rungs(min_budget,max_budget)
    directory.mkdir(parents=True,exist_ok=True)
    if completed(directory):
        return json.loads((directory/'selected.json').read_text())
    trace = []

    def call(candidate,budget):
        budget = int(round(budget))
        if not min_budget<=budget<=max_budget:
            raise ValueError('Scheduler returned out-of-range budget')
        start = time.monotonic()
        score = float(objective(candidate,budget))
        if not np.isfinite(score):
            raise ValueError('Non-finite search score')
        trace.append(dict(spec=candidate,budget=budget,score=score,seconds=time.monotonic()-start))
        save_json(directory/'trace.json',trace)
        return score

    baseline = spec(dict(lr=.001,weight_decay=.0003,hidden=64))
    baseline_score = call(baseline,max_budget)
    if method in ('random','tpe','gp','asha','hyperband','tpe_hyperband'):
        import optuna
        sampler = (optuna.samplers.TPESampler(seed=seed) if method in ('tpe','tpe_hyperband')
                   else optuna.samplers.RandomSampler(seed=seed))
        if method=='gp':
            sampler = optuna.samplers.GPSampler(seed=seed,n_startup_trials=10,deterministic_objective=False)
        if method=='asha':
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=min_budget,reduction_factor=3)
        elif method in ('hyperband','tpe_hyperband'):
            pruner = optuna.pruners.HyperbandPruner(min_resource=min_budget,max_resource=max_budget,reduction_factor=3)
        else:
            pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(study_name='parameter_methods',direction='minimize',sampler=sampler,pruner=pruner)
        def evaluate(trial):
            candidate = spec(sample(trial))
            for budget in (budgets if method in ('asha','hyperband','tpe_hyperband') else [max_budget]):
                value = call(candidate,budget)
                trial.report(value,budget)
                if budget<max_budget and trial.should_prune():
                    raise optuna.TrialPruned()
            return value
        study.optimize(evaluate,n_trials=trials)
    elif method in ('smac','smac_mf'):
        from smac import Scenario, HyperparameterOptimizationFacade, MultiFidelityFacade
        space = config_space(seed)
        failures = []
        def smac_call(config,budget):
            try:
                return call(spec(dict(config)),budget)
            except Exception as exc:
                failures.append(str(exc))
                raise
        scenario_args = dict(configspace=space,deterministic=True,n_trials=trials,seed=seed,
                             output_directory=directory/'smac_replay')
        if method=='smac_mf':
            scenario_args.update(min_budget=min_budget,max_budget=max_budget)
            def evaluate(config,seed=0,budget=max_budget):
                return smac_call(config,budget)
            facade = MultiFidelityFacade
        else:
            def evaluate(config,seed=0):
                return smac_call(config,max_budget)
            facade = HyperparameterOptimizationFacade
        facade(Scenario(**scenario_args),evaluate,overwrite=True).optimize()
        if failures:
            raise RuntimeError('SMAC objective failed; refusing to mark search complete: '+failures[0])
    elif method=='dehb':
        from dehb import DEHB
        space = config_space(seed)
        def evaluate(config,fidelity=None,**kwargs):
            start = time.monotonic()
            value = call(spec(dict(config)),fidelity)
            return dict(fitness=value,cost=time.monotonic()-start)
        optimizer = DEHB(f=evaluate,cs=space,dimensions=len(space),min_fidelity=min_budget,
                         max_fidelity=max_budget,n_workers=1,seed=seed,output_path=str(directory/'dehb_replay'))
        optimizer.run(fevals=trials)
    elif method=='pbt':
        if population<2:
            raise ValueError('PBT requires at least two members')
        rng = np.random.default_rng(seed)
        members = [spec(dict(lr=float(10**rng.uniform(-4,-2)),weight_decay=float(10**rng.uniform(-6,-2)),hidden=64))
                   for _ in range(population)]
        # Synchronous PBT: all regions and historical windows are one candidate bundle.
        # A new schedule's prefix identifies the donor's complete optimizer checkpoint.
        for stage,budget in enumerate(budgets):
            scores = [call(member,budget) for member in members]
            if budget==max_budget:
                break
            order = np.argsort(scores,kind='stable')
            replacement = set(order[-max(1,population//4):])
            donors = order[:max(1,population//4)]
            next_members = []
            for i,member in enumerate(members):
                source = members[int(rng.choice(donors))] if i in replacement else member
                history = list(source['schedule'])
                current = history[-1] if history else source['params']
                # Store the completed segment so refit can replay the chosen schedule.
                if not history:
                    history.append(dict(end=budget,lr=current['lr'],weight_decay=current['weight_decay']))
                else:
                    history[-1] = dict(history[-1],end=budget)
                lr,wd = current['lr'],current['weight_decay']
                if i in replacement:
                    lr = float(np.clip(lr*rng.choice([.8,1.2]),1e-4,.01))
                    wd = float(np.clip(wd*rng.choice([.8,1.2]),1e-6,.01))
                history.append(dict(end=budgets[stage+1],lr=lr,weight_decay=wd))
                next_members.append(dict(params=source['params'],schedule=history,
                                         parent=source,parent_budget=budget))
            members = next_members
    # Only completed maximum-budget candidates compete; never compare cheap proxy
    # scores to full training scores. Baseline always remains eligible.
    full = [row for row in trace if row['budget']==max_budget]
    best = min(full,key=lambda row:row['score'])
    result = dict(method=method,spec=best['spec'],score=best['score'],baseline_score=baseline_score,
                  max_budget=max_budget,search_evaluations=len(trace),full_budget_evaluations=len(full),
                  note='Best full-budget inner score; includes fixed fallback; no outer selection')
    save_json(directory/'selected.json',result)
    finish(directory,{'status':'complete'},[directory/'selected.json',directory/'trace.json'])
    return result
