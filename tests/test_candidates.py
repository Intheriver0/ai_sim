import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from screen_candidates import (CANDIDATES, candidate_features, predict_candidate, run_block,
                               summarize, training_condition, wait_for_run)
from fixed_model import PARAMS, WEATHER, features, predict
from seasonal_splits import load_plan
from pipeline_common import connect
from train_grouped import save_json
from test_pipeline import create_sites


class CandidateTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({'site_id': ['PV001']*24,
            'ts': pd.date_range('2024-06-01', periods=24, freq='h', tz='Asia/Seoul'),
            'lat': 37., 'lon': 127., 'capacity_kw': 123.456789, 'target_kwh': 50.})
        for col in WEATHER:
            frame[col] = 1.
        return frame

    def test_baseline_feature_and_prediction_parity(self):
        frame = self.frame()
        candidate = CANDIDATES[0]
        pd.testing.assert_frame_equal(candidate_features(frame, candidate)[0], features(frame))
        model = Mock()
        model.predict.side_effect = lambda x, **kwargs: np.full(len(x), .4)
        pd.testing.assert_frame_equal(predict_candidate(model, frame, candidate, 1), predict(model, frame, 1))

    def test_remove_direct_coordinates_keeps_solar_features(self):
        candidate = next(c for c in CANDIDATES if c['name'] == 'no_direct_coordinates')
        original = features(self.frame())
        x, day = candidate_features(self.frame(), candidate)
        self.assertNotIn('lat', x)
        self.assertNotIn('lon', x)
        pd.testing.assert_frame_equal(x, original.drop(columns=['lat', 'lon']))
        np.testing.assert_array_equal(day, original.elev > 0)

    def test_recent_history_uses_historical_origin(self):
        _, outers = load_plan(ROOT/'configs/seasonal_splits_v2.json')
        window = outers[-1]['inner'][0]
        candidate = next(c for c in CANDIDATES if c['name'] == 'recent_2years')
        expected = pd.Timestamp(window['training_cutoff'])-pd.DateOffset(years=2)
        with connect() as con:
            frame = pd.DataFrame({'ts': [expected-pd.Timedelta(hours=1), expected, expected+pd.Timedelta(hours=1)]})
            con.register('history', frame)
            self.assertEqual(con.execute(f'SELECT count(*) FROM history WHERE {training_condition(window,candidate)}').fetchone()[0], 2)
        self.assertEqual(training_condition(window, CANDIDATES[0]), 'TRUE')

    def test_outer_windows_stay_locked(self):
        _, outers = load_plan(ROOT/'configs/seasonal_splits_v2.json')
        with self.assertRaisesRegex(ValueError, 'Outer evaluation is locked'):
            run_block(None, outers[-1], CANDIDATES[0], None, 1, 12)

    def test_round17_pipeline_and_resume_with_mock_gpu(self):
        _, outers = load_plan(ROOT/'configs/seasonal_splits_v2.json')
        window = outers[-1]['inner'][0]
        frame = pd.concat([self.frame(), self.frame()], ignore_index=True)
        frame['ts'] = pd.date_range(window['start']-pd.Timedelta(days=2), periods=24, freq='h').append(
            pd.date_range(window['start']+pd.Timedelta(hours=1), periods=24, freq='h'))
        calls = []
        class Model:
            def __init__(self, **params):
                self.params = params
                self.booster_ = self
            def fit(self, x, y):
                calls.append((self.params, x.copy()))
                return self
            def predict(self, x, **kwargs):
                return np.full(len(x), .4)
            def save_model(self, path):
                Path(path).write_text('mock GPU model', encoding='utf-8')
        candidate = next(c for c in CANDIDATES if c['name'] == 'round17_only')
        with tempfile.TemporaryDirectory() as temp, connect() as con:
            create_sites(con)
            targets = frame[['site_id', 'ts', 'target_kwh']].assign(training_eligible=True, energy_raw_kwh=50.)
            con.register('targets', targets)
            f = frame.drop(columns='target_kwh')
            con.register('features10', f.assign(round=10, wa_temp=10.))
            con.register('features17', f.assign(round=17, wa_temp=17.))
            con.register('prices', pd.DataFrame({'ts': frame.ts, 'smp': 100., 'k': 1}))
            with patch('screen_candidates.lgb.LGBMRegressor', Model), \
                    patch('screen_candidates.lgb.Booster', side_effect=lambda **kwargs: Model()):
                prediction, result = run_block(con, window, candidate, Path(temp), 1, 12)
                restored, _ = run_block(con, window, candidate, Path(temp), 1, 12)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0]['device_type'], 'cuda')
            self.assertTrue((calls[0][1].wa_temp == 17).all())
            self.assertEqual(result['observed_site_hours'], 24)
            self.assertIsNone(result['settlement'])
            pd.testing.assert_frame_equal(prediction, restored)

    def test_candidate_count_and_cuda_contract(self):
        self.assertEqual(len(CANDIDATES), 8)
        self.assertEqual(len({c['name'] for c in CANDIDATES}), 8)
        self.assertEqual(sum(len(c.get('rounds', [10, 17])) for c in CANDIDATES)*12, 180)
        for candidate in CANDIDATES:
            self.assertEqual(dict(PARAMS, **candidate['params'])['device_type'], 'cuda')

    def test_queue_waits_until_full_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_json(root/'manifest.json', {'smoke': False, 'folds': list(range(12))})
            save_json(root/'summary.json', {'status': 'partial', 'all_12_months': False})
            def complete(_):
                save_json(root/'summary.json', {'status': 'complete', 'all_12_months': True, 'smoke': False})
            with patch('screen_candidates.time.sleep', side_effect=complete) as sleep:
                wait_for_run(root)
            sleep.assert_called_once_with(60)

    def test_queue_rejects_wrong_directory_and_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'manifest not found'):
                wait_for_run(root)
            save_json(root/'manifest.json', {'smoke': True, 'folds': list(range(12))})
            with self.assertRaisesRegex(ValueError, 'full twelve-month'):
                wait_for_run(root)

    def test_summary_never_ranks_partial_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            windows = [{'id': 'month'}]
            self.assertIsNone(summarize(root, windows, False)['exploratory_best'])
            for candidate in CANDIDATES:
                path = root/'month'/candidate['name']
                path.mkdir(parents=True)
                save_json(path/'comparison.json', {'site_equal_nmae': .1, 'complete_253_hours': 0, 'settlement': None})
            result = summarize(root, windows, True)
            self.assertEqual(result['status'], 'complete')
            self.assertIsNone(result['exploratory_best'])
            self.assertFalse(result['all_12_months'])
            self.assertIsNone(result['candidates'][0]['W_available_hours_won'])


if __name__ == '__main__':
    unittest.main()
