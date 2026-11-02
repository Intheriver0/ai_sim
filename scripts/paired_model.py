"""Joint 10/17 forecast input; one row and one target per site/hour."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fixed_model import WEATHER as BASELINE_WEATHER, FEATURES, features as single_features, PARAMS
from gpu_training import require_cuda_params
from pipeline_common import check_keys, connect, export, read_view, reject

WEATHER = [
    'wa_cloud', 'wa_temp', 'wa_humidity',
    'wa_ground_press', 'wa_wind_speed_ms', 'wa_wind_dir',
    'wb_temp', 'wb_feel_temp', 'wb_wet_bulb_temp', 'wb_dew_point',
    'wb_wind_speed_ms', 'wb_wind_dir', 'wb_rel_hum',
    'wb_vis', 'wb_ceiling', 'wb_uv_idx', 'wb_precip_prob', 'wb_rain_prob',
    'wb_snow_prob', 'wb_ice_prob', 'wb_cloud',
]
# The frozen separate-round benchmark still needs its precipitation columns.
# They remain in the shared table, but never enter the paired model matrix.
SOURCE_WEATHER = list(dict.fromkeys(WEATHER + BASELINE_WEATHER))
PAIRED_WEATHER = [f'{c}_r{r}' for r in (10, 17) for c in WEATHER]
MISSING = [f'{p}_missing_r{r}' for r in (10, 17) for p in ('wa', 'wb')]
PAIRED_FEATURES = PAIRED_WEATHER + MISSING + [c for c in FEATURES if c not in BASELINE_WEATHER]


def attach_pair(con, left='features10', right='features17', name='paired'):
    for table in (left, right):
        check_keys(con, table, ['site_id', 'ts'])
    reject(con, f'''SELECT count(*) FROM (
        (SELECT site_id,ts FROM {left} EXCEPT SELECT site_id,ts FROM {right}) UNION ALL
        (SELECT site_id,ts FROM {right} EXCEPT SELECT site_id,ts FROM {left}))''',
        'Paired forecast keys differ')
    reject(con, f'''SELECT count(*) FROM {left} a JOIN {right} b USING(site_id,ts)
        WHERE a.capacity_kw IS DISTINCT FROM b.capacity_kw
        OR a.lat IS DISTINCT FROM b.lat OR a.lon IS DISTINCT FROM b.lon
        OR a.round IS DISTINCT FROM 10 OR b.round IS DISTINCT FROM 17''', 'Inconsistent pair metadata or rounds')
    weather = ', '.join(f'{alias}."{c}" AS "{c}_r{r}"'
                        for alias, r in [('a', 10), ('b', 17)]
                        for c in SOURCE_WEATHER + ['wa_missing', 'wb_missing'])
    con.execute(f'''CREATE OR REPLACE TEMP VIEW {name} AS
        SELECT a.site_id,a.ts,a.capacity_kw,a.lat,a.lon,{weather}
        FROM {left} a JOIN {right} b USING(site_id,ts)''')


def features(frame):
    # Reuse the established solar calculation without averaging the forecasts.
    base = frame.rename(columns={f'{c}_r10': c for c in BASELINE_WEATHER})
    solar = single_features(base).drop(columns=BASELINE_WEATHER)
    x = pd.concat([frame[PAIRED_WEATHER + MISSING].astype('float32'), solar], axis=1)
    return x[PAIRED_FEATURES]


def fit(frame, threads, params=None):
    params = PARAMS if params is None else params
    require_cuda_params(params)
    x = features(frame)
    day = x.elev > 0
    if not day.any():
        raise ValueError('No daytime training examples in group')
    y = (frame.loc[day, 'target_kwh']/x.loc[day, 'cap']).clip(0, 1.2).astype('float32')
    model = lgb.LGBMRegressor(**params, n_jobs=threads).fit(x.loc[day], y)
    return model.booster_


def predict(model, frame, threads):
    x = features(frame)
    values = np.where(x.elev > 0, np.maximum(model.predict(x, num_threads=threads), 0), 0)*x.cap
    if not np.isfinite(values).all():
        raise ValueError('Non-finite prediction')
    return frame[['site_id', 'ts']].assign(forecast_kwh=values)


def materialize(data_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    with connect() as con:
        for split in ('train', 'test'):
            for r in (10, 17):
                read_view(con, f'features{r}', data_dir/f'{split}_round{r}.parquet')
            attach_pair(con)
            export(con, 'SELECT * FROM paired ORDER BY ts,site_id', output/f'{split}_paired.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    materialize(args.data_dir, args.output_dir)
