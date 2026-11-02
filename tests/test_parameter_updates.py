from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.func import functional_call

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from parameter_updates import virtual_step, mse, CoordinateLSTM, meta_loss
from parameter_models import Learner
from parameter_search import run_search
from test_parameter_methods import frame


class UpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_hypergradient_matches_finite_difference(self):
        model=torch.nn.Linear(2,1).double()
        x=torch.randn(10,2,dtype=torch.double)
        y=torch.randn(10,dtype=torch.double)
        rates=torch.tensor([-5.,-4.],dtype=torch.double,requires_grad=True)
        def objective(r):
            fast=virtual_step(model,x[:6],y[:6],r)
            return mse(functional_call(model,fast,(x[6:],)),y[6:])
        gradient,=torch.autograd.grad(objective(rates),rates)
        numerical=[]
        for i in range(2):
            delta=torch.zeros_like(rates)
            delta[i]=1e-5
            numerical.append(float(((objective(rates+delta)-objective(rates-delta))/2e-5).detach()))
        np.testing.assert_allclose(gradient.detach(),numerical,rtol=1e-5,atol=1e-9)

    def test_lstm_meta_gradient_and_recurrent_state(self):
        updater=CoordinateLSTM()
        model=torch.nn.Linear(2,1)
        x=torch.randn(12,2)
        y=torch.randn(12)
        loss=meta_loss(model,updater,x[:8],y[:8],x[8:],y[8:],3,.001,.0003)
        loss.backward()
        for parameter in updater.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(updater.first.weight_ih.grad.abs().sum()),0.)
        g=torch.tensor([0.,1e-8,-.5,1.])
        first,state=updater(g)
        second,_=updater(g,state)
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first,second))

    def test_adaptive_resume_and_clone(self):
        for name in ['mlp_sgd','mlp_hypergradient','mlp_learned_optimizer']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                f=frame()
                params=dict(hidden=16,batch_size=8,task_steps=2,meta_epochs=1,unroll_steps=2)
                direct=Learner(f,name,params,root/'direct','cpu')
                direct.advance(2)
                split=Learner(f,name,params,root/'split','cpu')
                split.advance(1)
                split=Learner(f,name,params,root/'split','cpu')
                split.advance(2)
                np.testing.assert_array_equal(direct.predict(f).forecast_kwh,split.predict(f).forecast_kwh)
                clone=Learner(f,name,params,root/'clone','cpu')
                clone.load(root/'split/state.pt',clone=True)
                split.advance(3)
                clone.advance(3)
                np.testing.assert_array_equal(split.predict(f).forecast_kwh,clone.predict(f).forecast_kwh)
                if name=='mlp_hypergradient':
                    self.assertLess(f.ts.iloc[direct.support].max(),f.ts.iloc[direct.validation].min())
                    self.assertFalse(np.allclose(direct.controller.log_rates.detach().exp(),[.001,.0003]))

    def test_gp_reaches_model_based_proposals(self):
        with tempfile.TemporaryDirectory() as temp:
            result=run_search('gp',lambda c,b:(np.log10(c['params']['lr'])+3)**2+c['params']['hidden']/1000,
                              Path(temp),11,1,1)
            self.assertEqual(result['search_evaluations'],12)
            self.assertTrue(np.isfinite(result['score']))


if __name__=='__main__':
    unittest.main()
