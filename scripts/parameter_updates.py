"""Past-only, project adaptations of differentiable hyperparameter/optimizer learning."""
import math

import torch
from torch import nn
from torch.func import functional_call


def mse(output, y):
    return (output.reshape(-1)-y).square().mean()


def virtual_step(model, x, y, log_rates):
    """One differentiable SGD step: log learning rate and log L2 coefficient."""
    weights = dict(model.named_parameters())
    loss = mse(model(x),y)
    gradients = torch.autograd.grad(loss,tuple(weights.values()))
    lr,wd = log_rates.exp()
    return {k:w-lr*(g.detach()+wd*w) for (k,w),g in zip(weights.items(),gradients)}


class Hypergradient:
    """One-step truncated bilevel SGD, NOT full reversible-training differentiation."""
    def __init__(self, model, params, device):
        self.model = model
        self.log_rates = nn.Parameter(torch.tensor([math.log(params['lr']),math.log(params['weight_decay'])],device=device))
        self.optimizer = torch.optim.Adam([self.log_rates],lr=params['hyper_lr'])

    def step(self, x, y, validation_x, validation_y):
        self.optimizer.zero_grad(set_to_none=True)
        fast = virtual_step(self.model,x,y,self.log_rates)
        loss = mse(functional_call(self.model,fast,(validation_x,)),validation_y)
        gradient, = torch.autograd.grad(loss,self.log_rates)
        if not torch.isfinite(gradient).all():
            raise ValueError('Non-finite hypergradient')
        self.log_rates.grad = gradient
        self.optimizer.step()
        with torch.no_grad():
            self.log_rates[0].clamp_(math.log(1e-4),math.log(.01))
            self.log_rates[1].clamp_(math.log(1e-6),math.log(.01))
        # Recompute at the updated hyperparameters; only support labels update weights.
        fast = virtual_step(self.model,x,y,self.log_rates)
        with torch.no_grad():
            for name,weight in self.model.named_parameters():
                weight.copy_(fast[name])

    def state_dict(self):
        return dict(log_rates=self.log_rates.detach(),optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        with torch.no_grad():
            self.log_rates.copy_(state['log_rates'])
        self.optimizer.load_state_dict(state['optimizer'])


class CoordinateLSTM(nn.Module):
    """Shared per-coordinate two-layer LSTM; paper-style log/sign preprocessing."""
    def __init__(self, hidden=20):
        super().__init__()
        self.first = nn.LSTMCell(2,hidden)
        self.second = nn.LSTMCell(hidden,hidden)
        self.output = nn.Linear(hidden,1)

    def forward(self, gradient, state=None):
        g = gradient.reshape(-1)
        threshold = math.exp(-10)
        large = g.abs()>=threshold
        inputs = torch.stack([torch.where(large,g.abs().clamp_min(threshold).log()/10,-torch.ones_like(g)),
                              torch.where(large,g.sign(),math.exp(10)*g)],1)
        first = self.first(inputs,None if state is None else state[:2])
        second = self.second(first[0],None if state is None else state[2:])
        # Bounded output is a project stability choice, not an original-paper setting.
        return .01*self.output(second[0]).tanh().reshape(gradient.shape), (*first,*second)


def learned_step(weights, gradients, updater, states, lr, wd):
    updated,next_states = {},{}
    for (name,weight),gradient in zip(weights.items(),gradients):
        delta,state = updater(gradient.detach(),states.get(name))
        updated[name] = weight*(1-lr*wd)+(lr/.001)*delta
        next_states[name] = state
    return updated,next_states


def meta_loss(model, updater, support_x, support_y, query_x, query_y, steps, lr, wd):
    """First-order gradient inputs; retain recurrence/update paths across the unroll."""
    weights = {k:w.detach().requires_grad_() for k,w in model.named_parameters()}
    states,losses = {},[]
    for _ in range(steps):
        loss = mse(functional_call(model,weights,(support_x,)),support_y)
        gradients = torch.autograd.grad(loss,tuple(weights.values()),retain_graph=True)
        weights,states = learned_step(weights,gradients,updater,states,lr,wd)
        losses.append(mse(functional_call(model,weights,(query_x,)),query_y))
    return torch.stack(losses).mean()
