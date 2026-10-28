"""Fixed participant-baseline model, independent of split and scoring code."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from solar_time import solar_clock
from gpu_training import require_cuda_params


PARAMS = dict(n_estimators=600, learning_rate=0.06, num_leaves=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, objective='regression', random_state=42,
              device_type='cuda', verbosity=-1)
WEATHER = ['wa_cloud', 'wa_temp', 'wa_humidity', 'wa_rain', 'wa_wind_speed_ms',
           'wa_ground_press', 'wb_cloud', 'wb_temp', 'wb_rel_hum', 'wb_uv_idx',
           'wb_precip_prob', 'wb_total_liq', 'wb_vis', 'wb_wind_speed_ms', 'wb_dew_point']
FEATURES = WEATHER + ['elev', 'azim', 'cosz', 'doy_sin', 'doy_cos', 'hour', 'cap', 'lat', 'lon']


def features(frame):
    """Participant baseline solar approximation; KST and right-closed midpoint."""
    ts = pd.DatetimeIndex(frame.ts).tz_convert('Asia/Seoul')
    doy = ts.day_of_year.to_numpy()
    minutes, decl = solar_clock(ts)
    ha = np.radians((minutes + 4*(frame.lon.to_numpy()-135)) / 4 - 180)
    la = np.radians(frame.lat.to_numpy())
    cosz = np.clip(np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha), -1, 1)
    x = frame[WEATHER + ['lat', 'lon']].astype('float32').copy()
    x['elev'] = (90 - np.degrees(np.arccos(cosz))).astype('float32')
    x['azim'] = (np.degrees(np.arctan2(-np.sin(ha), np.tan(decl)*np.cos(la)
                                     - np.sin(la)*np.cos(ha))) % 360).astype('float32')
    x['cosz'] = cosz.astype('float32')
    x['doy_sin'] = np.sin(2*np.pi*doy/365.25).astype('float32')
    x['doy_cos'] = np.cos(2*np.pi*doy/365.25).astype('float32')
    x['hour'] = ts.hour.to_numpy().astype('float32')
    x['cap'] = frame.capacity_kw.astype('float32')
    return x[FEATURES]


def fit(frame, threads):
    require_cuda_params(PARAMS)
    x = features(frame)
    day = x.elev > 0
    if not day.any():
        raise ValueError('No daytime training examples')
    y = (frame.loc[day, 'target_kwh'] / x.loc[day, 'cap']).clip(0, 1.2).astype('float32')
    model = lgb.LGBMRegressor(**PARAMS, n_jobs=threads).fit(x.loc[day], y)
    return model.booster_, int(day.sum())


def predict(model, frame, threads):
    x = features(frame)
    values = model.predict(x, num_threads=threads)
    values = np.where(x.elev.to_numpy() > 0, np.maximum(values, 0), 0) * x.cap.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError('Non-finite prediction')
    result = frame[['site_id', 'ts']].copy()
    result['forecast_kwh'] = values
    return result
