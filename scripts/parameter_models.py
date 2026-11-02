"""Resumable neural models for parameter-method experiments (production requires CUDA)."""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from literature_models import matrix, fit_scaler, transform


DEFAULT = dict(hidden=64, lr=.001, weight_decay=.0003, batch_size=1024,
               seed=42, inner_lr=.01, inner_steps=1, task_steps=100,
               hyper_lr=.01, meta_epochs=3, unroll_steps=5)
MODELS = ('mlp', 'tabm', 'hyper_head', 'maml', 'mlp_sgd',
          'mlp_hypergradient', 'mlp_learned_optimizer')


class HyperHead(nn.Module):
    """Project adaptation: static metadata generates only a small prediction head."""
    def __init__(self, width, hidden, context_width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, hidden), nn.ReLU())
        self.generator = nn.Sequential(nn.Linear(context_width, 16), nn.ReLU(), nn.Linear(16, hidden+1))

    def forward(self, x, context):
        head = self.generator(context)
        return (self.encoder(x)*head[:,:-1]).sum(1,keepdim=True)/self.encoder[0].out_features**.5+head[:,-1:]


def network(name, width, params, context_width=3):
    if name == 'hyper_head':
        return HyperHead(width, params['hidden'], context_width)
    if name == 'tabm':
        from tabm import TabM
        return TabM.make(n_num_features=width, d_out=1, k=32, n_blocks=3,
                         d_block=params['hidden'], dropout=.1)
    return nn.Sequential(nn.Linear(width, params['hidden']),nn.ReLU(),nn.Linear(params['hidden'],1))


def member_loss(output, target):
    # TabM: average member losses, never loss of the mean prediction.
    return (output-target.reshape((-1,)+(1,)*(output.ndim-1))).square().mean()


def adapt(model, x, y, params, higher_order=False):
    weights = dict(model.named_parameters())
    for _ in range(params['inner_steps']):
        loss = member_loss(functional_call(model,weights,(x,)),y)
        gradients = torch.autograd.grad(loss,tuple(weights.values()),create_graph=higher_order)
        weights = {k:w-params['inner_lr']*g for (k,w),g in zip(weights.items(),gradients)}
    return weights


def metadata(frame):
    # Static known metadata only. Terrain columns are attached by the runner.
    columns = [c for c in frame if c.startswith('terrain_')]
    return np.column_stack([frame.lat,frame.lon,np.log(frame.capacity_kw),frame[columns]]).astype('float32')


class Learner:
    def __init__(self, frame, name, params, directory, device='cuda'):
        if name not in MODELS:
            raise ValueError('Unknown neural method')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch required; CPU is only for explicit unit tests')
        self.name,self.params,self.directory,self.device = name,dict(DEFAULT,**params),directory,device
        self.directory.mkdir(parents=True,exist_ok=True)
        x = matrix(frame,'mlp10')
        day = x.elev.to_numpy()>0
        if not day.any():
            raise ValueError('No daytime training observations')
        self.columns = list(x)
        raw = x.to_numpy(dtype='float32')[day]
        self.scaler = fit_scaler(raw)
        context = metadata(frame)[day]
        self.context_scaler = fit_scaler(context)
        self.x = torch.from_numpy(transform(raw,self.scaler))
        self.context = torch.from_numpy(transform(context,self.context_scaler))
        self.y = torch.from_numpy((frame.target_kwh.to_numpy()[day]/x.cap.to_numpy()[day]).clip(0,1.2).astype('float32'))
        self.sites = frame.site_id.to_numpy()[day]
        self.tasks = {s:np.flatnonzero(self.sites==s) for s in np.unique(self.sites)}
        if name=='mlp_hypergradient':
            dates = frame.ts.to_numpy()[day]
            unique = np.unique(dates)
            if len(unique)<2:
                raise ValueError('Hypergradient needs at least two training timestamps')
            cutoff = unique[max(1,int(len(unique)*.75))]
            self.support = np.flatnonzero(dates<cutoff)
            self.validation = np.flatnonzero(dates>=cutoff)
        # SQL caller sorts by timestamp. Never construct adaptation episodes by random time split.
        if not frame.ts.is_monotonic_increasing:
            raise ValueError('Training frame must be chronologically sorted')
        identity = hashlib.sha256()
        for a in (raw,self.y.numpy(),context):
            identity.update(np.ascontiguousarray(a).tobytes())
        identity.update(frame.loc[day,['site_id','ts']].to_json(date_format='iso').encode())
        self.identity = identity.hexdigest()
        torch.manual_seed(self.params['seed'])
        self.net = network(name,self.x.shape[1],self.params,self.context.shape[1]).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(),lr=self.params['lr'],weight_decay=self.params['weight_decay'])
        self.initialize_updates()
        self.epoch = 0
        self.schedule = []
        if (directory/'state.pt').exists():
            self.load(directory/'state.pt')

    def initialize_updates(self):
        if self.name=='mlp_hypergradient':
            from parameter_updates import Hypergradient
            self.controller = Hypergradient(self.net,self.params,self.device)
        elif self.name=='mlp_sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=self.params['lr'],weight_decay=self.params['weight_decay'])
        elif self.name=='mlp_learned_optimizer':
            from parameter_updates import CoordinateLSTM
            self.updater = CoordinateLSTM().to(self.device)
            self.update_states = {}
            self.meta_ready = False

    def update_state(self):
        if self.name=='mlp_hypergradient':
            return self.controller.state_dict()
        if self.name=='mlp_learned_optimizer':
            return dict(updater=self.updater.state_dict(),states=self.update_states,meta_ready=self.meta_ready)
        return None

    def train_updater(self):
        from parameter_updates import meta_loss
        tasks = [ids for ids in self.tasks.values() if len(ids)>=4]
        if not tasks:
            raise ValueError('Learned optimizer needs sites with at least four training daylight observations')
        # Meta-training is repeated from its seed after a pre-checkpoint interruption.
        torch.manual_seed(self.params['seed']+100000)
        rng = np.random.default_rng(self.params['seed']+100000)
        optimizer = torch.optim.Adam(self.updater.parameters(),lr=.001)
        for _ in range(self.params['meta_epochs']*self.params['task_steps']):
            ids = tasks[int(rng.integers(len(tasks)))]
            split = max(1,int(len(ids)*.75))
            support = rng.choice(ids[:split],min(64,split),replace=False)
            query = rng.choice(ids[split:],min(64,len(ids)-split),replace=False)
            task = network('mlp',self.x.shape[1],self.params).to(self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = meta_loss(task,self.updater,self.x[support].to(self.device),self.y[support].to(self.device),
                self.x[query].to(self.device),self.y[query].to(self.device),self.params['unroll_steps'],
                self.params['lr'],self.params['weight_decay'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite learned optimizer meta loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.updater.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        self.meta_ready = True
        self.save()

    def load(self,path,clone=False):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text():
            raise ValueError('Checkpoint checksum differs')
        saved = torch.load(path,map_location='cpu',weights_only=False)  # Own hashed local checkpoints only.
        if saved['identity']!=self.identity or saved['name']!=self.name or saved['columns']!=self.columns:
            raise ValueError('Checkpoint data/model identity differs')
        if not clone and saved['initial_params']!=self.params:
            raise ValueError('Checkpoint initial parameters differ')
        self.params = saved['initial_params']
        self.net = network(self.name,self.x.shape[1],self.params,self.context.shape[1]).to(self.device)
        self.net.load_state_dict(saved['model'])
        self.optimizer = torch.optim.AdamW(self.net.parameters(),lr=self.params['lr'])
        self.initialize_updates()
        self.optimizer.load_state_dict(saved['optimizer'])
        if self.name=='mlp_hypergradient':
            self.controller.load_state_dict(saved['update_state'])
        elif self.name=='mlp_learned_optimizer':
            update = saved['update_state']
            self.updater.load_state_dict(update['updater'])
            self.update_states = {k:tuple(t.to(self.device) for t in v) for k,v in update['states'].items()}
            self.meta_ready = update['meta_ready']
        self.epoch,self.schedule = saved['epoch'],saved['schedule']

    def save(self):
        temporary = self.directory/'state.tmp'
        torch.save(dict(identity=self.identity,name=self.name,columns=self.columns,
            initial_params=self.params,model=self.net.state_dict(),optimizer=self.optimizer.state_dict(),
            epoch=self.epoch,schedule=self.schedule,scaler=self.scaler,context_scaler=self.context_scaler,
            update_state=self.update_state()),temporary)
        temporary.replace(self.directory/'state.pt')
        checkpoint = self.directory/'state.pt'
        (self.directory/'state.sha256').write_text(hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def advance(self, budget, lr=None, weight_decay=None):
        if budget<self.epoch:
            raise ValueError('Cannot rewind a checkpoint')
        lr = self.params['lr'] if lr is None else float(lr)
        wd = self.params['weight_decay'] if weight_decay is None else float(weight_decay)
        if self.name=='mlp_hypergradient' and (lr!=self.params['lr'] or wd!=self.params['weight_decay']):
            raise ValueError('Hypergradient cannot be combined with an external PBT schedule')
        if self.name=='mlp_learned_optimizer' and not self.meta_ready:
            self.train_updater()
        for group in self.optimizer.param_groups:
            group.update(lr=lr,weight_decay=wd)
        for epoch in range(self.epoch,budget):
            # Per-epoch seeds make uninterrupted and resumed runs equivalent.
            torch.manual_seed(self.params['seed']+epoch)
            rng = np.random.default_rng(self.params['seed']+epoch)
            self.net.train()
            if self.name=='maml':
                tasks = [ids for ids in self.tasks.values() if len(ids)>=4]
                if not tasks:
                    raise ValueError('MAML needs sites with at least four training daylight observations')
                batches = [tasks[int(rng.integers(len(tasks)))] for _ in range(self.params['task_steps'])]
            else:
                order = rng.permutation(self.support if self.name=='mlp_hypergradient' else len(self.x))
                batches = [order[i:i+self.params['batch_size']] for i in range(0,len(order),self.params['batch_size'])]
            for ids in batches:
                if self.name=='mlp_hypergradient':
                    query = rng.choice(self.validation,min(self.params['batch_size'],len(self.validation)),replace=False)
                    self.controller.step(self.x[ids].to(self.device),self.y[ids].to(self.device),
                                         self.x[query].to(self.device),self.y[query].to(self.device))
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                if self.name=='maml':
                    split = max(1,int(len(ids)*.75))
                    support = rng.choice(ids[:split],min(64,split),replace=False)
                    query = rng.choice(ids[split:],min(64,len(ids)-split),replace=False)
                    fast = adapt(self.net,self.x[support].to(self.device),self.y[support].to(self.device),self.params,True)
                    loss = member_loss(functional_call(self.net,fast,(self.x[query].to(self.device),)),self.y[query].to(self.device))
                else:
                    inp = self.x[ids].to(self.device)
                    output = self.net(inp,self.context[ids].to(self.device)) if self.name=='hyper_head' else self.net(inp)
                    loss = member_loss(output,self.y[ids].to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                if self.name=='mlp_learned_optimizer':
                    from parameter_updates import learned_step
                    weights = dict(self.net.named_parameters())
                    gradients = torch.autograd.grad(loss,tuple(weights.values()))
                    with torch.no_grad():
                        fast,self.update_states = learned_step(weights,gradients,self.updater,self.update_states,lr,wd)
                        for name,weight in weights.items():
                            weight.copy_(fast[name])
                    continue
                loss.backward()
                self.optimizer.step()
            self.epoch = epoch+1
            if self.name=='mlp_hypergradient':
                effective_lr,effective_wd = self.controller.log_rates.detach().exp().cpu().tolist()
            else:
                effective_lr,effective_wd = lr,wd
            self.schedule.append(dict(epoch=self.epoch,lr=effective_lr,weight_decay=effective_wd))
            self.save()
            print(f'{self.name}: epoch {self.epoch}/{budget}',flush=True)

    def predict(self,frame):
        x = matrix(frame,'mlp10')
        if list(x)!=self.columns:
            raise ValueError('Inference feature schema differs')
        data = torch.from_numpy(transform(x.to_numpy(dtype='float32'),self.scaler))
        context = torch.from_numpy(transform(metadata(frame),self.context_scaler))
        values = np.zeros(len(frame))
        self.net.eval()
        # MAML adapts solely on the last observed training rows, never evaluation labels.
        groups = frame.groupby('site_id',sort=False).indices if self.name=='maml' else {'all':np.arange(len(frame))}
        for site,ids in groups.items():
            fast = None
            if self.name=='maml' and site in self.tasks:
                support = self.tasks[site][-256:]
                fast = adapt(self.net,self.x[support].to(self.device),self.y[support].to(self.device),self.params)
            with torch.no_grad():
                for start in range(0,len(ids),self.params['batch_size']):
                    batch = ids[start:start+self.params['batch_size']]
                    inp = data[batch].to(self.device)
                    if self.name=='hyper_head':
                        out = self.net(inp,context[batch].to(self.device))
                    elif fast is not None:
                        out = functional_call(self.net,fast,(inp,))
                    else:
                        out = self.net(inp)
                    values[batch] = out.reshape(len(batch),-1).mean(1).cpu().numpy()
        values = np.where(x.elev>0,np.maximum(values,0),0)*x.cap
        if not np.isfinite(values).all():
            raise ValueError('Non-finite forecast')
        return frame[['site_id','ts']].assign(forecast_kwh=values)
