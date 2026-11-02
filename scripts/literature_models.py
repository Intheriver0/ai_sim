"""Fixed, explicitly attributed experiments; no search or outer-label early stopping."""
import warnings

import numpy as np

from fixed_model import PARAMS
from paired_model import features, WEATHER

# See literature_experiments.md for source values versus project adaptations.
PRESETS = {
    'lgbm_l2': dict(PARAMS),
    'lgbm_l1': dict(PARAMS, objective='regression_l1'),
    'lgbm_revision': dict(PARAMS, objective='regression_l1'),
    'catboost_mae': dict(iterations=1200, depth=8, learning_rate=.03,
                         l2_leaf_reg=3, loss_function='MAE', random_seed=42,
                         task_type='GPU', devices='0', allow_writing_files=False),
    'mlp10': dict(hidden=10, epochs=50, batch_size=1024, lr=.001, weight_decay=0., seed=42),
    'tabm': dict(k=32, n_blocks=3, d_block=512, dropout=.1, epochs=50,
                 batch_size=1024, lr=.002, weight_decay=.0003, seed=42),
}


def matrix(frame, name):
    x = features(frame)
    if name == 'lgbm_revision':
        for c in WEATHER:
            if c.endswith('wind_dir'):
                for r in (10, 17):
                    angle = np.deg2rad(x.pop(f'{c}_r{r}'))
                    x[f'{c}_sin_r{r}'] = np.sin(angle)
                    x[f'{c}_cos_r{r}'] = np.cos(angle)
                for axis in ('sin', 'cos'):
                    x[f'{c}_{axis}_revision'] = x[f'{c}_{axis}_r17']-x[f'{c}_{axis}_r10']
            else:
                x[f'{c}_revision'] = x[f'{c}_r17']-x[f'{c}_r10']
    return x.replace([np.inf, -np.inf], np.nan)


def fit_scaler(x):
    """Fit on training daylight rows only; preserve even all-missing columns."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(x, axis=0)
    median = np.nan_to_num(median)
    filled = np.where(np.isfinite(x), x, median)
    return dict(median=median, mean=filled.mean(axis=0), scale=np.maximum(filled.std(axis=0), 1e-6))


def transform(x, scaler):
    missing = ~np.isfinite(x)
    filled = np.where(missing, scaler['median'], x)
    return np.column_stack([(filled-scaler['mean'])/scaler['scale'], missing]).astype('float32')


def make_neural(name, width):
    import torch
    p = PRESETS[name]
    if name == 'mlp10':
        return torch.nn.Sequential(torch.nn.Linear(width, p['hidden']), torch.nn.ReLU(),
                                   torch.nn.Linear(p['hidden'], 1))
    from tabm import TabM
    return TabM.make(n_num_features=width, d_out=1, k=p['k'], n_blocks=p['n_blocks'],
                     d_block=p['d_block'], dropout=p['dropout'])


def check_backend(name, threads):
    if name.startswith('lgbm'):
        from gpu_training import check_cuda
        return check_cuda(threads)
    if name == 'catboost_mae':
        from catboost import CatBoostRegressor
        # Actual tiny GPU fit catches unsupported binaries/drivers, not just imports.
        CatBoostRegressor(**dict(PRESETS[name], iterations=2, depth=2), verbose=False).fit(
            np.arange(128, dtype=float).reshape(64, 2), np.arange(64, dtype=float)/64)
        return {'backend': 'CatBoost GPU'}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required; no CPU fallback')
    model = make_neural(name, 4).cuda()
    model(torch.zeros(8, 4, device='cuda')).square().mean().backward()
    return {'backend': name, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}


def fit_save(frame, name, directory, threads):
    x = matrix(frame, name)
    day = x.elev > 0
    if not day.any():
        raise ValueError('No daytime training rows in region')
    y = (frame.loc[day, 'target_kwh']/x.loc[day, 'cap']).clip(0, 1.2).to_numpy(dtype='float32')
    x = x.loc[day]
    p = PRESETS[name]
    directory.mkdir(parents=True, exist_ok=True)
    if name.startswith('lgbm'):
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**p, n_jobs=threads).fit(x, y).booster_
        model.save_model(str(directory/'model.txt'))
    elif name == 'catboost_mae':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**p, thread_count=threads, verbose=100).fit(x, y)
        model.save_model(str(directory/'model.cbm'))
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch required')
        torch.set_num_threads(threads)
        torch.manual_seed(p['seed'])
        torch.cuda.manual_seed_all(p['seed'])
        raw = x.to_numpy(dtype='float32')
        scaler = fit_scaler(raw)
        np.savez(directory/'scaler.npz', **scaler)
        data = torch.from_numpy(transform(raw, scaler))
        target = torch.from_numpy(y)
        model = make_neural(name, data.shape[1]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        rng = np.random.default_rng(p['seed'])
        model.train()
        for epoch in range(p['epochs']):
            total = 0.
            order = rng.permutation(len(data))
            for start in range(0, len(data), p['batch_size']):
                ids = order[start:start+p['batch_size']]
                output = model(data[ids].cuda())
                truth = target[ids].cuda().reshape((-1,)+(1,)*(output.ndim-1))
                # Each TabM member learns separately; average only at inference.
                loss = (output-truth).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite neural training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())*len(ids)
            print(f'{name} epoch={epoch+1}/{p["epochs"]} train_mse={total/len(data):.7f}', flush=True)
        torch.save(model.state_dict(), directory/'model.pt')
        model = (model, scaler)
    return model


def restore(name, directory):
    if name.startswith('lgbm'):
        import lightgbm as lgb
        return lgb.Booster(model_file=str(directory/'model.txt'))
    if name == 'catboost_mae':
        from catboost import CatBoostRegressor
        return CatBoostRegressor().load_model(str(directory/'model.cbm'))
    import torch
    with np.load(directory/'scaler.npz') as saved:
        scaler = dict(saved)
    model = make_neural(name, 2*len(scaler['median'])).cuda()
    model.load_state_dict(torch.load(directory/'model.pt', map_location='cuda', weights_only=True))
    return model, scaler


def predict(model, frame, name, threads):
    x = matrix(frame, name)
    if name.startswith('lgbm'):
        values = model.predict(x, num_threads=threads)
    elif name == 'catboost_mae':
        values = model.predict(x, thread_count=threads)
    else:
        import torch
        net, scaler = model
        data = transform(x.to_numpy(dtype='float32'), scaler)
        net.eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(data), PRESETS[name]['batch_size']):
                output = net(torch.from_numpy(data[start:start+PRESETS[name]['batch_size']]).cuda())
                parts.append(output.reshape(len(output), -1).mean(dim=1).cpu().numpy())
        values = np.concatenate(parts)
    values = np.where(x.elev > 0, np.maximum(values, 0), 0)*x.cap
    if not np.isfinite(values).all():
        raise ValueError('Non-finite prediction')
    return frame[['site_id', 'ts']].assign(forecast_kwh=values)
