from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fixed_model import PARAMS
from seasonal_splits import load_plan
from train_tuned import ALPHAS, blend, checked_result, choose_alpha, fit_round, tune
from train_grouped import save_json
from time_splits import digest


class TuningTests(unittest.TestCase):
    def predictions(self):
        return pd.DataFrame({'site_id': ['PV001', 'PV002'],
            'ts': pd.to_datetime(['2025-01-01T12:00:00+09:00']*2), 'forecast_kwh': [10., 30.]})

    def test_blend_matches_keys_not_row_order_and_endpoints(self):
        fixed = self.predictions()
        tuned = fixed.assign(forecast_kwh=[30., 10.]).iloc[::-1]
        np.testing.assert_allclose(blend(fixed, tuned, .25).forecast_kwh, [25., 15.])
        np.testing.assert_allclose(blend(fixed, tuned, 1).forecast_kwh, [10., 30.])
        np.testing.assert_allclose(blend(fixed, tuned, 0).forecast_kwh, [30., 10.])

    def test_blend_rejects_missing_duplicate_or_invalid_values(self):
        fixed = self.predictions()
        for invalid in (fixed.iloc[:1], pd.concat([fixed, fixed]),
                        fixed.assign(forecast_kwh=np.nan), fixed.assign(forecast_kwh=-1)):
            with self.assertRaises(ValueError):
                blend(fixed, invalid, .5)
        with self.assertRaises(ValueError):
            blend(fixed, fixed, 1.1)

    def test_alpha_uses_equal_month_weights_and_tie_prefers_fixed(self):
        curves = [(ALPHAS-.25)**2, (ALPHAS-.75)**2]
        alpha, _ = choose_alpha(curves)
        self.assertAlmostEqual(alpha, .5)
        self.assertEqual(choose_alpha([np.ones(21), np.ones(21)])[0], 1.)

    def test_early_stopping_never_reads_diagnostic_labels(self):
        class Connection:
            def __init__(self):
                self.queries = []
            def execute(self, query):
                self.queries.append(query)
                return self
            def df(self):
                return self.queries[-1]
        class Model:
            def __init__(self, **params):
                self.params = params
                self.best_iteration_ = 7
                self.booster_ = self
            def fit(self, x, y, **kwargs):
                calls.append((self.params, x, kwargs))
                return self
        calls = []
        con = Connection()
        with patch('train_tuned.daytime_xy', side_effect=lambda f: (f, [1.])), \
                patch('train_tuned.lgb.LGBMRegressor', Model):
            _, iterations, _ = fit_round(con, 17, PARAMS, 1, True)
        self.assertEqual(iterations, 7)
        self.assertIn('FROM fit_features', calls[0][1])
        self.assertIn('FROM stop_features', calls[0][2]['eval_set'][0][0])
        self.assertIn('FROM training_features', calls[1][1])
        self.assertEqual(calls[1][0]['n_estimators'], 7)
        self.assertTrue(all('round=17' in q and 'diagnostic' not in q for q in con.queries))

    def test_tuning_fallback_and_resume_total_budget(self):
        windows = [{'id': 'month_a'}, {'id': 'month_b'}]
        calls = []
        def fake_run(con, window, params, early_stop, output, threads):
            calls.append(early_stop)
            return None, {'site_equal_nmae': .2 if early_stop else .1}
        with tempfile.TemporaryDirectory() as temp, patch('train_tuned.run_model', side_effect=fake_run):
            path = Path(temp)
            result = tune(None, windows, path, path, 1, 2, 60, 12, PARAMS)
            self.assertEqual(result['selected'], 'fixed')
            self.assertEqual(result['completed_trials'], 2)
            count = sum(calls)
            again = tune(None, windows, path, path, 1, 2, 60, 12, PARAMS)
            self.assertEqual(sum(calls), count)
            self.assertEqual(again['completed_trials'], 2)

    def test_interrupted_trial_retries_exact_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            storage = optuna.storages.RDBStorage('sqlite:///'+(path/'study.db').as_posix())
            study = optuna.create_study(study_name='lightgbm', storage=storage)
            trial = study.ask()
            trial.suggest_float('learning_rate', .015, .12, log=True)
            saved_rate = trial.params['learning_rate']
            storage.remove_session()
            storage.engine.dispose()
            def fake_run(con, window, params, early_stop, output, threads):
                if early_stop:
                    self.assertEqual(params['learning_rate'], saved_rate)
                return None, {'site_equal_nmae': .1 if early_stop else .2}
            with patch('train_tuned.run_model', side_effect=fake_run):
                result = tune(None, [{'id': 'month'}], path, path, 1, 1, 60, 12, PARAMS)
            self.assertEqual(result['selected'], 'tuned')

    def test_timeout_without_completed_trial_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as temp, patch('train_tuned.run_model',
                return_value=(None, {'site_equal_nmae': .1})):
            path = Path(temp)
            result = tune(None, [{'id': 'month'}], path, path, 1, 5, 1e-12, 12, PARAMS)
            self.assertEqual(result['completed_trials'], 0)
            self.assertEqual(result['selected'], 'fixed')

    def test_cache_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            artifact = path/'predictions.parquet'
            artifact.write_bytes(b'original')
            save_json(path/'metrics.json', {'artifacts_sha256': {artifact.name: digest(artifact)}})
            self.assertIsNotNone(checked_result(path))
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                checked_result(path)

    def test_all_selection_windows_precede_outer_cutoff(self):
        _, windows = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
        for outer in windows:
            model = [w for w in outer['inner'] if w['purpose'] == 'model_selection']
            mixed = [w for w in outer['inner'] if w['purpose'] == 'blend_selection']
            self.assertEqual(len(model), 12)
            self.assertEqual(len(mixed), 12)
            for a, b in zip(model, mixed):
                self.assertLessEqual(a['end'], b['start'])
                self.assertLessEqual(b['end'], outer['training_cutoff'])
                self.assertLess(a['early_stop_end'], a['start'])
                self.assertLess(b['training_cutoff'], b['start'])


if __name__ == '__main__':
    unittest.main()
