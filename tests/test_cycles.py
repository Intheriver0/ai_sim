from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_model import attach_pair, features, fit, predict, WEATHER, SOURCE_WEATHER, PAIRED_FEATURES
from pipeline_common import connect
from residual_groups import optimize, objective, panel, residuals
from seasonal_splits import load_plan
from train_cycles import search_windows, train_predict, completed, run_fold, parser
from test_pipeline import create_sites


class PairTests(unittest.TestCase):
    def source(self, r):
        data = pd.DataFrame({'site_id': ['A', 'B'],
            'ts': pd.to_datetime(['2024-05-01T12:00:00+09:00']*2),
            'round': r, 'capacity_kw': 200., 'lat': 37., 'lon': 127.,
            'wa_missing': False, 'wb_missing': r == 17})
        for column in SOURCE_WEATHER:
            data[column] = float(r)
        return data

    def test_pair_values_nulls_and_feature_contract(self):
        with connect() as con:
            a, b = self.source(10), self.source(17)
            b.loc[0, WEATHER[0]] = np.nan
            con.register('features10', a)
            con.register('round17_source', b)
            con.execute('CREATE TEMP TABLE features17 AS SELECT * FROM round17_source ORDER BY site_id DESC')
            attach_pair(con)
            result = con.execute('SELECT * FROM paired ORDER BY site_id').df()
            self.assertEqual(len(result), 2)
            self.assertEqual(result.wa_cloud_r10.tolist(), [10, 10])
            self.assertTrue(pd.isna(result.wa_cloud_r17.iloc[0]))
            x = features(result)
            self.assertEqual(list(x), PAIRED_FEATURES)
            pd.testing.assert_frame_equal(x, features(result.assign(target_kwh=999, smp=999)))
            self.assertEqual(x.wb_missing_r17.tolist(), [1, 1])

    def test_selected_inputs_reach_fit_and_predict(self):
        with connect() as con:
            con.register('features10', self.source(10))
            con.register('features17', self.source(17))
            attach_pair(con)
            frame = con.execute('SELECT * FROM paired ORDER BY site_id').df().assign(target_kwh=100.)
        self.assertEqual(len(WEATHER), 21)
        self.assertEqual(len(PAIRED_FEATURES), 55)
        excluded = ['wa_temp_max', 'wa_temp_min', 'wa_rain', 'wa_snow_mm',
                    'wb_wind_gust_spd_ms', 'wb_total_liq', 'wb_rain', 'wb_snow_mm', 'wb_ice']
        changed = frame.assign(**{f'{c}_r{r}': 99999. for c in excluded for r in (10,17)})
        pd.testing.assert_frame_equal(features(frame), features(changed))
        with patch('paired_model.lgb.LGBMRegressor') as regressor:
            fit(frame, 1)
            x = regressor.return_value.fit.call_args.args[0]
            self.assertEqual(list(x), PAIRED_FEATURES)
            self.assertFalse({f'{c}_r{r}' for c in excluded for r in (10,17)} & set(x.columns))
            for r in (10,17):
                for c in WEATHER:
                    self.assertTrue((x[f'{c}_r{r}'] == r).all())
        class Model:
            def predict(self, x, num_threads):
                if list(x) != PAIRED_FEATURES:
                    raise AssertionError('Prediction feature schema differs')
                return np.ones(len(x))*.5
        self.assertEqual(len(predict(Model(), frame, 1)), len(frame))

    def test_missing_and_duplicate_keys_rejected(self):
        with connect() as con:
            con.register('features10', self.source(10))
            con.register('features17', self.source(17).iloc[:1])
            with self.assertRaisesRegex(ValueError, 'keys differ'):
                attach_pair(con)
            con.fetchall()
            con.unregister('features17')
            con.register('features17', pd.concat([self.source(17)]*2))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                attach_pair(con)


class GroupTests(unittest.TestCase):
    def test_full_cycle_selection_refit_and_resume(self):
        outer = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')[1][0]
        opt, selection = search_windows(outer)
        times = pd.DataFrame({'ts': [opt['training_cutoff']-timedelta(hours=5),
            opt['start']+timedelta(hours=12), selection['start']+timedelta(hours=12),
            outer['start']+timedelta(hours=12)]})

        class Model:
            def save_model(self, path):
                Path(path).write_text('fake', encoding='utf-8')

        def prediction(model, frame, threads):
            # Opposite residual signs across the original groups make swaps useful.
            sign = np.where(frame.site_id.str[2:].astype(int) <= 126, 1., -1.)
            return frame[['site_id', 'ts']].assign(forecast_kwh=100+sign)

        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.register('times', times)
            con.execute('CREATE TABLE prices AS SELECT ts,100.0 AS smp,3 AS k FROM times')
            con.execute('''CREATE TABLE targets AS SELECT site_id,ts,100.0 AS target_kwh,
                100.0 AS energy_raw_kwh,true AS training_eligible FROM sites CROSS JOIN times''')
            con.execute('CREATE TABLE paired AS SELECT site_id,ts FROM targets')
            con.execute('CREATE VIEW features10 AS SELECT *,10 AS round FROM paired')
            con.execute('CREATE VIEW features17 AS SELECT *,17 AS round FROM paired')
            con.execute('''CREATE TABLE reference_composition AS SELECT site_id,
                CASE current_group WHEN 'A' THEN 'G1' ELSE 'G2' END AS "group" FROM sites''')
            args = parser().parse_args(['--output-dir', tmp, '--modes', 'cycles',
                '--trials', '1', '--cycles', '1', '--search-steps', '500'])
            with patch('train_cycles.fit', return_value=Model()) as fitter, \
                 patch('train_cycles.predict', side_effect=prediction), \
                 patch('train_cycles.lgb.Booster', return_value=Model()):
                result = run_fold(con, outer, args)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['candidates'], 4)
                self.assertIsNotNone(result['settlement'])
                self.assertEqual(result['selected']['allocation'], 'optimized')
                calls = fitter.call_count
                run_fold(con, outer, args)
                self.assertEqual(fitter.call_count, calls)

    def test_error_cancellation_capacity_and_missing_site_preservation(self):
        sites = pd.DataFrame({'site_id': list('ABCDE'), 'capacity_kw': [10000]*4+[1000]})
        initial = sites[['site_id']].assign(cluster=[0, 0, 1, 1, 0])
        errors = np.array([[4., 4., -4., -4.], [2., 2., -2., -2.]])
        result, report = optimize(errors, np.ones(2), list('ABCD'), sites, initial, steps=500)
        self.assertLess(report['after'], report['before'])
        self.assertEqual(report['after'], 0)
        self.assertGreaterEqual(min(report['capacity_kw']), 20000)
        self.assertEqual(result.iloc[-1].cluster, 0)
        self.assertEqual(objective(errors, np.ones(2), result.cluster.to_numpy()[:4]), report['after'])

    def test_panel_never_zero_fills_and_missing_prediction_rejected(self):
        ts = pd.date_range('2024-01-01', periods=3, freq='h', tz='Asia/Seoul')
        actual = pd.DataFrame({'ts': list(ts)*2, 'site_id': ['A']*3+['B']*3,
                               'energy': [1, 2, 3, 1, np.nan, 3]})
        prices = pd.DataFrame({'ts': ts, 'smp': 100, 'k': 1})
        matrix, weights = panel(actual, prices, min_coverage=.5)
        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_array_equal(weights, [10, 10])
        prediction = actual.rename(columns={'energy': 'forecast_kwh'}).iloc[:3]
        with self.assertRaisesRegex(ValueError, 'missing'):
            residuals(prediction, matrix)

    def test_chronology(self):
        _, outers = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
        for outer in outers:
            opt, select = search_windows(outer)
            self.assertLess(opt['training_cutoff'], opt['start'])
            self.assertEqual(opt['end'], select['start'])
            self.assertLess(select['training_cutoff'], select['start'])
            self.assertLessEqual(select['end'], outer['training_cutoff'])
            self.assertLess(outer['training_cutoff'], outer['start'])

    def test_training_filters_cache_and_artifact_tampering(self):
        # Exercise real DuckDB queries and cache IO; replace GPU fitting only.
        ts = pd.to_datetime(['2024-01-01T12:00:00+09:00', '2024-01-03T12:00:00+09:00'])
        frame = pd.DataFrame({'site_id': ['A', 'A'], 'ts': ts})
        window = {'id': 'test', 'training_start': ts[0]-timedelta(days=1),
                  'training_cutoff': ts[0], 'start': ts[1]-timedelta(hours=1), 'end': ts[1]}
        assignment = pd.DataFrame({'site_id': ['A'], 'cluster': [0]})

        class Model:
            def save_model(self, path):
                Path(path).write_text('fake model', encoding='utf-8')

        def fit_mock(train, threads, params):
            self.assertEqual(train.ts.tolist(), [ts[0]])
            self.assertEqual(train.target_kwh.tolist(), [1.])
            return Model()

        def predict_mock(model, frame, threads):
            self.assertEqual(frame.ts.tolist(), [ts[1]])
            return frame[['site_id', 'ts']].assign(forecast_kwh=1.)

        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            con.register('paired', frame)
            con.register('targets', frame.assign(target_kwh=[1., 999.], training_eligible=True))
            with patch('train_cycles.fit', side_effect=fit_mock) as fitter, \
                 patch('train_cycles.predict', side_effect=predict_mock), \
                 patch('train_cycles.lgb.Booster', return_value=Model()):
                output = Path(tmp)
                first, key = train_predict(con, window, assignment, {'n_estimators': 1}, output, 1)
                second, _ = train_predict(con, window, assignment, {'n_estimators': 1}, output, 1)
                self.assertEqual(fitter.call_count, 1)
                pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
                path = output/'cache'/key
                (path/'cluster0.txt').write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Changed completed artifact'):
                    completed(path)


if __name__ == '__main__':
    unittest.main()
