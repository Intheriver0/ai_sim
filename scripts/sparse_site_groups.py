"""Local final-run discovery and past-only geography/terrain/error fallback."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import read_view, check_keys, check_time, reject
from time_splits import digest, stamp

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ['disclosed_coordinate_dsm_m', 'max_horizon_deg',
           'dominant_blocker_within_2km_directions']
ERRORS = ['signed_nmae', 'positive_fraction', 'negative_fraction']


def discover(results_root, explicit=None):
    candidates = [Path(explicit)] if explicit else sorted(Path(results_root).glob('*1028*'))
    valid, rejected = [], []
    for path in candidates:
        if not path.is_dir():
            continue
        try:
            summary = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            folds = summary['folds']
            if (summary.get('status') != 'complete' or summary.get('all_12_months') is not True
                    or summary.get('smoke') is not False or manifest.get('smoke') is not False
                    or len(folds) != 12 or len({f['fold'] for f in folds}) != 12):
                raise ValueError('requires complete, non-smoke twelve-month final results')
            files = [path/'summary.json', path/'manifest.json']
            for fold in folds:
                name = fold['fold']
                if not name.startswith('outer_') or Path(name).name != name:
                    raise ValueError('Invalid outer fold name')
                files += [path/name/'result.json', path/name/'blend_predictions.parquet']
            if not all(p.is_file() for p in files):
                raise ValueError('missing final result.json or blend_predictions.parquet')
            valid.append((path.resolve(), {str(p.relative_to(path)): digest(p) for p in files}))
        except (OSError, ValueError, KeyError) as exc:
            rejected.append(f'{path}: {exc}')
    if len(valid) != 1:
        raise ValueError(f'Expected one completed CUDA 1028 run, found {len(valid)}. '
                         'Use --error-run to specify its directory. '+ '; '.join(rejected))
    return valid[0]


def error_profile(con, run, cutoff):
    """Only final outer blend predictions, never arbitrary inner trial caches."""
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    pieces = []
    for fold in summary['folds']:
        directory = run/fold['fold']
        result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        origin = pd.Timestamp(result['training_cutoff'])
        if origin.tzinfo is None:
            raise ValueError('Historical model cutoff requires timezone')
        if origin >= cutoff:
            continue
        read_view(con, 'historical_predictions', directory/'blend_predictions.parquet')
        check_keys(con, 'historical_predictions', ['site_id', 'ts'])
        check_time(con, 'historical_predictions')
        reject(con, '''SELECT count(*) FROM historical_predictions ANTI JOIN sites USING(site_id)''',
               'Unknown site in historical predictions')
        # Validate prediction chronology, including rows later filtered out.
        invalid = con.execute(f'''SELECT count(*) FROM historical_predictions
            WHERE ts <= {stamp(origin)} OR NOT isfinite(forecast_kwh) OR forecast_kwh IS NULL''').fetchall()[0][0]
        if invalid:
            raise ValueError('Invalid final held-out predictions')
        pieces.append(con.execute(f'''SELECT p.site_id,p.ts,
            (p.forecast_kwh-t.energy_raw_kwh)/s.capacity_kw AS error
            FROM historical_predictions p JOIN targets t USING(site_id,ts) JOIN sites s USING(site_id)
            WHERE p.ts <= {stamp(cutoff)} AND hour(p.ts) BETWEEN 9 AND 16''').df())
    if not pieces:
        return pd.DataFrame(columns=['site_id', 'error_rows']+ERRORS)
    rows = pd.concat(pieces, ignore_index=True)
    if rows.duplicated(['site_id', 'ts']).any():
        raise ValueError('Overlapping final outer prediction keys')
    rows['positive'] = rows.error > 0
    rows['negative'] = rows.error < 0
    return rows.groupby('site_id', as_index=False).agg(
        error_rows=('error', 'size'), signed_nmae=('error', 'mean'),
        positive_fraction=('positive', 'mean'), negative_fraction=('negative', 'mean'))


def terrain_profile(directory, sites):
    directory = Path(directory)
    if json.loads((directory/'manifest.json').read_text(encoding='utf-8')).get('status') != 'complete':
        raise ValueError('Terrain preprocessing is incomplete')
    frame = pd.read_json(directory/'site_quality.json')
    if frame.site_id.duplicated().any() or set(frame.site_id) != set(sites.site_id):
        raise ValueError('Terrain site keys differ')
    aligned = frame.set_index('site_id').loc[sites.site_id]
    if not np.allclose(aligned[['lat', 'lon']], sites[['lat', 'lon']], atol=1e-8, rtol=0):
        raise ValueError('Terrain coordinates differ from training sites')
    if not aligned.origin_valid.all() or not np.isfinite(aligned[TERRAIN]).all().all():
        raise ValueError('Missing terrain; provide completed terrain_features_v2')
    return frame[['site_id']+TERRAIN]


def assign_sparse(assignment, terrain, errors, min_error_rows=30):
    """Transfer cluster from the closest reliable peer; record every fallback."""
    result = assignment.merge(terrain, on='site_id', validate='one_to_one').merge(
        errors, on='site_id', how='left', validate='one_to_one')
    reliable = result.profile_reliable.to_numpy(dtype=bool)
    peers = np.flatnonzero(reliable)
    if not len(peers):
        raise ValueError('No reliable pattern peers')
    coords = np.radians(result[['lat', 'lon']].to_numpy(dtype=float))
    t = result[TERRAIN].to_numpy(dtype=float)
    scale = np.std(t[peers], axis=0)
    scale[scale < 1e-8] = 1
    t = t/scale
    e = result[ERRORS].to_numpy(dtype=float)
    known = (result.error_rows.fillna(0).to_numpy() >= min_error_rows) & np.isfinite(e).all(axis=1)
    error_peers = peers[known[peers]]
    escale = np.std(e[error_peers], axis=0) if len(error_peers) else np.ones(len(ERRORS))
    escale[escale < 1e-8] = 1
    result['assignment_basis'] = 'generation_pattern'
    result['peer_site_id'] = None
    result['peer_distance_km'] = np.nan
    for i in np.flatnonzero(~reliable):
        use_errors = known[i] and len(error_peers) > 0
        pool = error_peers if use_errors else peers
        delta = coords[pool]-coords[i]
        h = np.sin(delta[:, 0]/2)**2 + np.cos(coords[i, 0])*np.cos(coords[pool, 0])*np.sin(delta[:, 1]/2)**2
        km = 6371*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        geography = (km/100)**2
        topography = ((t[pool]-t[i])**2).mean(axis=1)
        distance = (geography+topography)/2
        if use_errors:
            distance = (distance+(((e[pool]-e[i])/escale)**2).mean(axis=1))/2
        peer = pool[np.argmin(distance)]
        result.loc[i, 'cluster'] = result.loc[peer, 'cluster']
        result.loc[i, 'assignment_basis'] = 'geography_terrain_error' if use_errors else 'geography_terrain_no_past_error'
        result.loc[i, 'peer_site_id'] = result.loc[peer, 'site_id']
        result.loc[i, 'peer_distance_km'] = float(km[np.argmin(distance)])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find completed local CUDA 1028 final outputs; no Git or GPU required')
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--error-run', type=Path)
    args = parser.parse_args()
    path, files = discover(args.results_root, args.error_run)
    print(json.dumps({'run': str(path), 'files_sha256': files}, indent=2))
