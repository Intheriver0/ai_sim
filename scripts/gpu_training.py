"""Mandatory CUDA training checks; never fall back to CPU or OpenCL."""
import json

import lightgbm as lgb
import numpy as np


def require_cuda_params(params):
    if params.get('device_type') != 'cuda' or params.get('device', 'cuda') != 'cuda':
        raise ValueError('GPU training is mandatory: device_type must be cuda')


def check_cuda(threads=1):
    print('CUDA preflight: fitting a small model on the GPU...', flush=True)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 4)).astype('float32')
    y = (x[:, 0] + x[:, 1]**2).astype('float32')
    try:
        model = lgb.train({'device_type': 'cuda', 'objective': 'regression',
            'num_threads': threads, 'num_leaves': 7, 'min_data_in_leaf': 5,
            'verbosity': 1, 'seed': 42}, lgb.Dataset(x, label=y), num_boost_round=3)
        require_cuda_params(model.params)
        if model.num_trees() < 1 or not np.isfinite(model.predict(x, num_threads=threads)).all():
            raise RuntimeError('CUDA probe produced no trees or invalid predictions')
    except Exception as exc:
        raise RuntimeError('CUDA training is required and the GPU probe failed. No CPU fallback. '
                           'See gpu_training.md for the GB10 CUDA build and verification steps. '
                           f'Original error: {exc}') from exc
    print('CUDA preflight PASSED. Training device=cuda; CPU fallback disabled.', flush=True)
    return {'device_type': 'cuda', 'probe': 'passed', 'lightgbm': lgb.__version__}


if __name__ == '__main__':
    print(json.dumps(check_cuda(), indent=2))
