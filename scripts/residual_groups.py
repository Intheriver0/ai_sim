"""Capacity-constrained G1/G2 search on simultaneous held-out residuals."""
import numpy as np
import pandas as pd


def panel(actual, prices, min_coverage=.95, complete_only=False):
    matrix = actual.pivot(index='ts', columns='site_id', values='energy')
    matrix = matrix.reindex(pd.Index(prices.ts, name='ts'))
    if complete_only:
        selected = matrix.columns
    else:
        selected = matrix.columns[matrix.notna().mean() >= min_coverage]
    matrix = matrix[selected].dropna()
    if len(selected) < 2 or matrix.empty:
        raise ValueError('No usable simultaneous residual panel; adjust history/coverage explicitly')
    weights = prices.set_index('ts').loc[matrix.index].eval('0.1 * smp * k').to_numpy()
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid settlement weights')
    return matrix, weights


def residuals(predictions, actual):
    values = predictions.pivot(index='ts', columns='site_id', values='forecast_kwh')
    values = values.reindex(index=actual.index, columns=actual.columns)
    result = (values-actual).to_numpy()
    if not np.isfinite(result).all():
        raise ValueError('Predictions missing from residual panel')
    return result


def objective(errors, weights, labels):
    first = errors[:, labels == 0].sum(axis=1)
    second = errors[:, labels == 1].sum(axis=1)
    return float(weights @ (np.abs(first)+np.abs(second)))


def optimize(errors, weights, panel_sites, sites, initial, steps=20000, seed=42,
             minimum_capacity=20000):
    """Greedy randomized single moves and pair swaps; missing sites stay fixed."""
    ids = list(sites.site_id)
    capacities = sites.capacity_kw.to_numpy(dtype=float)
    labels = initial.set_index('site_id').loc[ids, 'cluster'].to_numpy(dtype=int).copy()
    if set(labels) != {0, 1}:
        raise ValueError('Expected two nonempty settlement groups')
    cap = np.bincount(labels, weights=capacities, minlength=2)
    if min(cap) < minimum_capacity:
        raise ValueError('Initial groups violate minimum capacity')
    indices = np.array([ids.index(s) for s in panel_sites])
    active = labels[indices].copy()
    first = errors[:, active == 0].sum(axis=1)
    total = errors.sum(axis=1)
    best = objective(errors, weights, active)
    initial_score = best
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(steps):
        a = int(rng.integers(len(indices)))
        chosen = [a]
        if rng.random() < .8:
            other = np.flatnonzero(active != active[a])
            if not len(other):
                continue
            chosen.append(int(rng.choice(other)))
        delta_cap = sum((1 if active[i] else -1)*capacities[indices[i]] for i in chosen)
        new_cap = cap + np.array([delta_cap, -delta_cap])
        if min(new_cap) < minimum_capacity-1e-6:
            continue
        candidate = first + sum((1 if active[i] else -1)*errors[:, i] for i in chosen)
        value = float(weights @ (np.abs(candidate)+np.abs(total-candidate)))
        if value < best-1e-9:
            for i in chosen:
                active[i] = 1-active[i]
                labels[indices[i]] = active[i]
            first, cap, best = candidate, new_cap, value
            accepted += 1
    return pd.DataFrame({'site_id': ids, 'cluster': labels}), {
        'before': initial_score, 'after': best, 'accepted_moves': accepted,
        'capacity_kw': cap.tolist(), 'panel_sites': len(indices), 'panel_hours': len(errors)}
