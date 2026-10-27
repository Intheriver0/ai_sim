"""Past-only plant profiles and predictive clusters, unrelated to submission G1/G2."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


META = ['lat', 'lon', 'log_capacity']
PATTERN = ['morning', 'noon', 'afternoon']


def profiles(con):
    # training_targets is the guarded seasonal view; never read all targets here.
    aggregates = []
    for name, left, right in [('morning', 9, 11), ('noon', 12, 14), ('afternoon', 15, 17)]:
        condition = f'hour(t.ts) BETWEEN {left} AND {right}'
        aggregates += [f'avg(least(t.target_kwh/s.capacity_kw,1.2)) FILTER(WHERE {condition}) AS {name}',
                       f'count(t.ts) FILTER(WHERE {condition}) AS {name}_rows']
    return con.execute(f'''SELECT s.site_id,s.lat,s.lon,ln(s.capacity_kw) AS log_capacity,
        count(t.ts) AS history_rows, {', '.join(aggregates)}
        FROM sites s LEFT JOIN training_targets t USING(site_id)
        GROUP BY s.site_id,s.lat,s.lon,s.capacity_kw ORDER BY s.site_id''').df()


def group_sites(profile, k, min_bin_rows=30):
    if k < 1 or k > len(profile):
        raise ValueError('Invalid number of clusters')
    result = profile.copy()
    reliable = (profile[[f'{p}_rows' for p in PATTERN]] >= min_bin_rows).all(axis=1)
    result['profile_reliable'] = reliable
    if k == 1:
        result['cluster'] = 0
        return result, {'k': 1, 'silhouette': None, 'method': 'global baseline'}
    if reliable.sum() <= k:
        raise ValueError('Too few reliable plant profiles')
    columns = META + PATTERN
    scaler = StandardScaler().fit(profile.loc[reliable, columns])
    x = scaler.transform(profile.loc[reliable, columns])
    model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(x)
    if len(set(model.labels_)) != k:
        raise ValueError('Profiles do not support distinct requested clusters')
    result['cluster'] = -1
    result.loc[reliable, 'cluster'] = model.labels_
    # Cold/sparse sites use only known metadata; no fabricated output history.
    if (~reliable).any():
        meta = (profile.loc[~reliable, META].to_numpy()-scaler.mean_[:3])/scaler.scale_[:3]
        distances = ((meta[:, None, :] - model.cluster_centers_[None, :, :3])**2).sum(axis=2)
        result.loc[~reliable, 'cluster'] = distances.argmin(axis=1)
    return result, {'k': k, 'method': 'standardized metadata and past daytime capacity factors; KMeans',
                    'silhouette': float(silhouette_score(x, model.labels_)),
                    'features': columns, 'scale_mean': scaler.mean_.tolist(),
                    'scale_std': scaler.scale_.tolist(), 'centers': model.cluster_centers_.tolist(),
                    'min_bin_rows': min_bin_rows, 'seed': 42, 'n_init': 20}
