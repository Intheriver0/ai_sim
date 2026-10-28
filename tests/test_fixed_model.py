import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fixed_model import FEATURES, WEATHER, features, fit, predict
from pipeline_common import connect
from train_fixed import evaluate
from test_pipeline import create_sites


class FixedModelTests(unittest.TestCase):
    def frame(self):
        ts = pd.date_range('2024-06-01', periods=24*30, freq='h', tz='Asia/Seoul')
        frame = pd.DataFrame({'ts': ts, 'site_id': 'PV001', 'lat': 37., 'lon': 127.,
                              'capacity_kw': 200., 'target_kwh': 100.})
        for col in WEATHER:
            frame[col] = 1.
        frame.loc[0, 'wa_cloud'] = np.nan
        return frame

    def test_feature_contract(self):
        frame = self.frame()
        x = features(frame)
        self.assertEqual(list(x), FEATURES)
        changed = frame.assign(target_kwh=9999, energy=9999, smp=9999)
        pd.testing.assert_frame_equal(x, features(changed))

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS') == '1', 'Set RUN_CUDA_TESTS=1 on the CUDA training host')
    def test_cuda_saved_predictions(self):
        frame = self.frame()
        x = features(frame)
        model, daytime = fit(frame, 1)
        self.assertGreater(daytime, 0)
        self.assertLess(daytime, len(frame))
        p = predict(model, frame, 1)
        self.assertTrue((p.loc[x.elev <= 0, 'forecast_kwh'] == 0).all())
        self.assertTrue(np.isfinite(p.forecast_kwh).all())
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp)/'model.txt')
            model.save_model(path)
            pd.testing.assert_frame_equal(p, predict(lgb.Booster(model_file=path), frame, 1))

    def test_reference_solar_parity(self):
        path = Path(__file__).resolve().parents[2]/'OIBC_2026_DATA/handover 2/01_participant_dataset/baseline.py'
        if not path.exists():
            self.skipTest('Local participant baseline not distributed with repository')
        spec = importlib.util.spec_from_file_location('participant_baseline', path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        ts = pd.date_range('2024-01-01', '2025-01-01', freq='h', tz='Asia/Seoul')
        frame = self.frame().iloc[:1].loc[np.zeros(len(ts), dtype=int)].reset_index(drop=True)
        frame['ts'] = ts
        frame['lat'] = np.linspace(33., 39., len(ts))
        frame['lon'] = np.linspace(124., 132., len(ts))
        original = frame.copy(deep=True)
        x = features(frame)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(x, features(frame.assign(ts=frame.ts.dt.tz_convert('UTC'))))
        for col, expected in zip(('elev', 'azim', 'cosz'), baseline.solar(
                pd.DatetimeIndex(frame.ts), frame.lat.to_numpy(), frame.lon.to_numpy())):
            np.testing.assert_array_equal(x[col], expected)

    def test_partial_settlement_and_hand_score(self):
        with connect() as c:
            create_sites(c)
            c.execute('''CREATE TABLE composition AS SELECT site_id,
                CASE current_group WHEN 'A' THEN 'G1' ELSE 'G2' END AS "group" FROM sites''')
            c.execute("CREATE TABLE diagnostic_actual AS SELECT site_id, TIMESTAMPTZ '2025-01-01 11:00:00+09' AS ts, 1.0 AS energy FROM sites")
            c.execute('CREATE TABLE training_targets AS SELECT * FROM diagnostic_actual')
            c.execute('CREATE TABLE settlement_actual AS SELECT * FROM diagnostic_actual')
            c.execute('CREATE TABLE settlement_prices AS SELECT DISTINCT ts,100.0 AS smp,3 AS k FROM diagnostic_actual')
            prediction = c.execute('SELECT site_id,ts,2.0 AS forecast_kwh FROM diagnostic_actual').df()
            result = evaluate(c, prediction)
            self.assertAlmostEqual(result['site_equal_nmae'], .005)
            self.assertAlmostEqual(result['settlement']['W_won'], 253*30)
            c.execute('DELETE FROM settlement_prices')
            self.assertIsNone(evaluate(c, prediction)['settlement'])
            with self.assertRaisesRegex(ValueError, 'keys differ'):
                evaluate(c, prediction.iloc[1:])


if __name__ == '__main__':
    unittest.main()
