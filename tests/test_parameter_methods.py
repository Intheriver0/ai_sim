from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_model import SOURCE_WEATHER
from parameter_search import run_search, rungs, spec
from run_parameter_methods import inner_windows, Experiment
from seasonal_splits import load_plan
from pipeline_common import connect


def frame():
    times = pd.date_range('2024-05-01T12:00:00+09:00',periods=16,freq='D')
    f = pd.DataFrame([(site,ts) for ts in times for site in ['a','b']],columns=['site_id','ts'])
    f = f.assign(lat=37.,lon=127.,capacity_kw=100.,target_kwh=np.tile([30.,60.],16))
    for r in (10,17):
        for c in SOURCE_WEATHER:
            f[f'{c}_r{r}'] = float(r)
        for prefix in ('wa','wb'):
            f[f'{prefix}_missing_r{r}'] = False
    return f


class SearchTests(unittest.TestCase):
    def test_rungs_and_chronology(self):
        self.assertEqual(rungs(5,45),[5,15,45])
        self.assertEqual(rungs(5,40),[5,15,40])
        with self.assertRaises(ValueError):
            rungs(6,5)
        _,windows = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
        for outer in windows:
            self.assertEqual(len(inner_windows(outer)),12)
        bad = dict(windows[0],training_cutoff=windows[0]['inner'][0]['start'])
        with self.assertRaises(ValueError):
            inner_windows(bad)

    def test_schedulers_full_budget_selection_and_resume(self):
        for method in ['fixed','random','tpe','asha','hyperband','tpe_hyperband','pbt']:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                calls=[]
                def objective(candidate,budget):
                    calls.append((candidate,budget))
                    return -100. if budget<3 else float(candidate['params']['hidden'])
                result = run_search(method,objective,Path(temp),4,1,3,population=4)
                self.assertGreater(result['score'],0.)
                self.assertTrue(any(b==3 for c,b in calls))
                count = len(calls)
                again = run_search(method,objective,Path(temp),4,1,3,population=4)
                self.assertEqual(result,again)
                self.assertEqual(len(calls),count)
                if method=='pbt':
                    offspring=[c for c,b in calls if 'parent' in c]
                    self.assertEqual(len(offspring),4)
                    self.assertTrue(all(c['parent_budget']==1 and c['schedule'][-1]['end']==3 for c in offspring))
                (Path(temp)/'selected.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'Changed completed artifact'):
                    run_search(method,objective,Path(temp),4,1,3,population=4)

    @unittest.skipUnless(importlib.util.find_spec('smac'),'SMAC optional dependency absent')
    def test_smac_real_adapter(self):
        for method in ['smac','smac_mf']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as temp:
                result = run_search(method,lambda c,b:float(c['params']['hidden']),Path(temp),4,1,3)
                self.assertTrue(np.isfinite(result['score']))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch optional dependency absent')
class NeuralTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_all_models_actual_training_and_reload(self):
        from parameter_models import Learner
        models=['mlp','hyper_head','maml','mlp_sgd','mlp_hypergradient','mlp_learned_optimizer']
        if importlib.util.find_spec('tabm'):
            models.append('tabm')
        for name in models:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                train=frame()
                params=dict(hidden=16,batch_size=16,task_steps=2,meta_epochs=1,unroll_steps=2)
                learner=Learner(train,name,params,Path(temp),device='cpu')
                learner.advance(1)
                expected=learner.predict(train)
                restored=Learner(train,name,params,Path(temp),device='cpu')
                np.testing.assert_allclose(expected.forecast_kwh,restored.predict(train).forecast_kwh)
                # Evaluation labels must have zero influence, including MAML adaptation.
                np.testing.assert_allclose(expected.forecast_kwh,restored.predict(train.assign(target_kwh=99999.)).forecast_kwh)

    def test_resume_equivalence_and_pbt_optimizer_clone(self):
        from parameter_models import Learner
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            train=frame()
            params=dict(hidden=16,batch_size=16)
            direct=Learner(train,'mlp',params,root/'direct','cpu')
            direct.advance(2)
            split=Learner(train,'mlp',params,root/'split','cpu')
            split.advance(1)
            split=Learner(train,'mlp',params,root/'split','cpu')
            split.advance(2)
            np.testing.assert_allclose(direct.predict(train).forecast_kwh,split.predict(train).forecast_kwh,rtol=0,atol=0)
            clone=Learner(train,'mlp',params,root/'clone','cpu')
            clone.load(root/'split/state.pt',clone=True)
            self.assertEqual(clone.epoch,2)
            self.assertTrue(clone.optimizer.state_dict()['state'])
            split.advance(3,lr=.002)
            clone.advance(3,lr=.002)
            np.testing.assert_allclose(split.predict(train).forecast_kwh,clone.predict(train).forecast_kwh,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'identity'):
                Learner(train.assign(target_kwh=20.),'mlp',params,root/'split','cpu')
            (root/'split/state.pt').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):
                Learner(train,'mlp',params,root/'split','cpu')

    def test_member_loss_not_loss_of_average(self):
        from parameter_models import member_loss
        import torch
        prediction=torch.tensor([[[-1.],[1.]]],requires_grad=True)
        loss=member_loss(prediction,torch.tensor([0.]))
        self.assertEqual(float(loss.detach()),1.)
        loss.backward()
        self.assertGreater(float(prediction.grad.abs().sum()),0.)

    def test_hypernetwork_conditions_and_maml_gradients(self):
        from parameter_models import HyperHead,network,DEFAULT,adapt,member_loss
        from torch.func import functional_call
        import torch
        torch.manual_seed(42)
        model=HyperHead(4,8,3)
        context=torch.tensor([[0.,0.,0.],[1.,2.,3.]])
        weights=model.generator(context)
        self.assertFalse(torch.equal(weights[0],weights[1]))
        model(torch.ones(2,4),context).square().mean().backward()
        self.assertGreater(float(model.generator[0].weight.grad.abs().sum()),0.)
        model=network('maml',4,dict(DEFAULT,hidden=8))
        x=torch.randn(8,4)
        y=torch.randn(8)
        fast=adapt(model,x[:4],y[:4],DEFAULT,higher_order=True)
        member_loss(functional_call(model,fast,(x[4:],)),y[4:]).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_experiment_cutoff_clone_and_refit(self):
        from parameter_models import Learner
        with tempfile.TemporaryDirectory() as temp,connect() as con:
            f=frame()
            con.register('paired',f.drop(columns='target_kwh'))
            con.register('targets',f[['site_id','ts','target_kwh']].assign(training_eligible=True))
            assignment=pd.DataFrame({'site_id':['a','b'],'cluster':[0,1]})
            w=dict(id='inner',purpose='model_selection',training_start=f.ts.min(),
                training_cutoff=f.ts.unique()[9],start=f.ts.unique()[10].normalize(),end=f.ts.unique()[12].normalize())
            exp=Experiment(con,assignment,'mlp',Path(temp),1,42,'cpu')
            parent=spec(dict(hidden=16,lr=.001,weight_decay=.0003,batch_size=16))
            first=exp.predict_window(w,parent,1)
            child=dict(params=parent['params'],schedule=[dict(end=1,lr=.001,weight_decay=.0003),dict(end=2,lr=.002,weight_decay=.0003)],parent=parent,parent_budget=1)
            second=exp.predict_window(w,child,2)
            self.assertEqual(len(first),len(second))
            outer=dict(w,id='outer',purpose='outer_evaluation')
            replay=exp.predict_window(outer,child,2)
            np.testing.assert_allclose(second.forecast_kwh,replay.forecast_kwh,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
