from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from literature_models import PRESETS, matrix, fit_scaler, transform, predict
from ridge_regions import broad_barrier, partition
from run_literature import daily_penalty, reduction, train_region
from pipeline_common import connect
from paired_model import SOURCE_WEATHER, WEATHER, PAIRED_FEATURES


class LiteratureTests(unittest.TestCase):
    def test_complete_weather_contract_across_fixed_models(self):
        # Independent inventory from the supplied forecast schema, after unit renaming.
        raw = set('''wa_cloud wa_temp wa_temp_max wa_temp_min wa_humidity
            wa_ground_press wa_wind_speed_ms wa_wind_dir wa_rain wa_snow_mm
            wb_temp wb_feel_temp wb_wet_bulb_temp wb_dew_point wb_wind_speed_ms
            wb_wind_gust_spd_ms wb_wind_dir wb_rel_hum wb_vis wb_ceiling wb_uv_idx
            wb_precip_prob wb_rain_prob wb_snow_prob wb_ice_prob wb_total_liq
            wb_rain wb_snow_mm wb_ice wb_cloud'''.split())
        excluded = set('''wa_temp_max wa_temp_min wa_rain wa_snow_mm
            wb_wind_gust_spd_ms wb_total_liq wb_rain wb_snow_mm wb_ice'''.split())
        self.assertEqual(set(WEATHER), raw-excluded)
        frame = self.frame()
        for index, column in enumerate(WEATHER):
            for r in (10, 17):
                frame[f'{column}_r{r}'] = index+r
        for model in PRESETS:
            with self.subTest(model=model):
                x = matrix(frame, model)
                self.assertEqual(len(x.columns), 82 if model == 'lgbm_revision' else 55)
                for column in WEATHER:
                    for r in (10, 17):
                        if model == 'lgbm_revision' and column.endswith('wind_dir'):
                            angle = np.deg2rad(frame[f'{column}_r{r}'])
                            np.testing.assert_allclose(x[f'{column}_sin_r{r}'], np.sin(angle), atol=1e-7)
                            np.testing.assert_allclose(x[f'{column}_cos_r{r}'], np.cos(angle), atol=1e-7)
                        else:
                            np.testing.assert_allclose(x[f'{column}_r{r}'], frame[f'{column}_r{r}'])
                changed = frame.assign(**{f'{c}_r{r}': 99999. for c in excluded for r in (10,17)})
                pd.testing.assert_frame_equal(x, matrix(changed, model))
        x = matrix(frame, 'mlp10')
        self.assertEqual(list(x), PAIRED_FEATURES)
        self.assertEqual(transform(x.to_numpy(), fit_scaler(x.to_numpy())).shape, (2, 110))

    def frame(self):
        frame = pd.DataFrame({'site_id':['a','b'], 'ts':pd.to_datetime(['2024-06-01T12:00:00+09:00']*2),
                              'lat':37.,'lon':127.,'capacity_kw':100.})
        for r in (10,17):
            for c in SOURCE_WEATHER:
                frame[f'{c}_r{r}'] = float(r)
            for prefix in ('wa','wb'):
                frame[f'{prefix}_missing_r{r}'] = False
        return frame

    def test_revision_circular_wind_and_excluded_amounts(self):
        frame = self.frame()
        frame['wa_wind_dir_r10'], frame['wa_wind_dir_r17'] = 359., 1.
        x = matrix(frame, 'lgbm_revision')
        self.assertAlmostEqual(x.wa_wind_dir_sin_revision.iloc[0], 2*np.sin(np.deg2rad(1)), places=5)
        self.assertAlmostEqual(x.wa_wind_dir_cos_revision.iloc[0], 0., places=5)
        self.assertEqual(x.wa_temp_revision.iloc[0], 7.)
        self.assertNotIn('wa_wind_dir_r10', x)
        changed = frame.assign(wa_rain_r10=99999., wb_total_liq_r17=99999., target_kwh=999.)
        pd.testing.assert_frame_equal(x,matrix(changed,'lgbm_revision'))

    def test_scaler_uses_training_only_and_preserves_missing(self):
        train = np.array([[1.,np.nan],[3.,np.nan]],dtype='float32')
        scaler = fit_scaler(train)
        original = {k:v.copy() for k,v in scaler.items()}
        transformed = transform(np.array([[100.,np.nan]],dtype='float32'),scaler)
        np.testing.assert_allclose(transformed,[[98.,0.,0.,1.]])
        for key in scaler:
            np.testing.assert_equal(scaler[key],original[key])

    def test_broad_vs_small_ridge_and_unknown(self):
        self.assertTrue(broad_barrier(np.array([0,600,600,600,600,0]),1000))
        self.assertFalse(broad_barrier(np.array([0,600,0]),1000))
        self.assertFalse(broad_barrier(np.array([0,200,200,200,200,0]),1000))
        self.assertIsNone(broad_barrier(np.array([0,np.nan,0]),1000))

    def test_hard_constraints_infeasible_k_is_not_relaxed(self):
        xy = np.array([[0,0],[1000,0],[1000,1000],[0,1000]])
        blocked = ~np.eye(4,dtype=bool)
        for k in (2,3):
            labels,info = partition(xy,blocked,k)
            self.assertIsNone(labels)
            self.assertEqual(info['reason'],'infeasible')
        labels,info = partition(xy,blocked,4)
        self.assertEqual(len(set(labels)),4)
        self.assertTrue(info['feasible'])

    def test_penalty_alignment_and_right_closed_calendar(self):
        with connect() as con:
            ts = pd.to_datetime(['2024-01-02T00:00:00+09:00'])
            con.register('settlement_actual',pd.DataFrame({'site_id':['a'],'ts':ts,'energy':[10.]}))
            con.register('settlement_prices',pd.DataFrame({'ts':ts,'smp':[100.],'k':[1.]}))
            con.register('composition',pd.DataFrame({'site_id':['a'],'group':['G1']}))
            pred = pd.DataFrame({'site_id':['a'],'ts':ts,'forecast_kwh':[12.]})
            w = {'start':pd.Timestamp('2024-01-01',tz='Asia/Seoul'),
                 'end':pd.Timestamp('2024-01-03',tz='Asia/Seoul')}
            daily = daily_penalty(con,pred,w)
            self.assertEqual(daily.W.tolist(),[20.,0.])
            better = daily.assign(W=daily.W/2)
            result = reduction(daily,better)
            self.assertEqual(result['W_reduction_won'],10.)
            self.assertEqual(result['W_reduction_percent'],50.)
            self.assertEqual(reduction(daily,daily)['W_reduction_95pct_block_interval_won'],[0.,0.])
            with self.assertRaisesRegex(ValueError,'keys differ'):
                reduction(daily,better.iloc[::-1])

    def test_training_cutoff_and_region_isolation_and_resume(self):
        with connect() as con, tempfile.TemporaryDirectory() as temp:
            frame = self.frame()
            future = frame.assign(ts=frame.ts+pd.Timedelta(days=2))
            train = frame.assign(ts=frame.ts-pd.Timedelta(days=2))
            paired = pd.concat([train,frame,future],ignore_index=True)
            con.register('paired',paired)
            con.register('targets',paired[['site_id','ts']].assign(target_kwh=50.,training_eligible=True))
            w = {'id':'test','training_start':train.ts.iloc[0]-pd.Timedelta(days=1),
                 'training_cutoff':frame.ts.iloc[0]-pd.Timedelta(days=1),
                 'start':frame.ts.iloc[0].normalize(),'end':frame.ts.iloc[0].normalize()+pd.Timedelta(days=1)}
            seen = []
            class Constant:
                def predict(self,x,num_threads):
                    return np.ones(len(x))*.5
            def fit(training,name,directory,threads):
                seen.append(training.copy())
                directory.mkdir(parents=True)
                (directory/'model.txt').write_text('mock')
                return Constant()
            dest = Path(temp)/'run'
            assignment = frame[['site_id']].assign(cluster=[0,1])
            with patch('run_literature.fit_save',side_effect=fit), patch('run_literature.restore',return_value=Constant()):
                result = train_region(con,w,assignment,'lgbm_l2',dest,1)
                self.assertEqual(len(result),2)
                self.assertEqual(len(seen),2)
                self.assertTrue(all(len(t)==1 and t.ts.max()<=w['training_cutoff'] for t in seen))
                resumed = train_region(con,w,assignment,'lgbm_l2',dest,1)
                self.assertEqual(len(seen),2)
                np.testing.assert_allclose(result.forecast_kwh,resumed.forecast_kwh)
            (dest/'models/cluster0/model.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed completed artifact'):
                train_region(con,w,assignment,'lgbm_l2',dest,1)


if __name__ == '__main__':
    unittest.main()
