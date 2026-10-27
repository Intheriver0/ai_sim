from pathlib import Path
import unittest

import test_time_splits
from seasonal_splits import load_plan, window_views

CONFIG = Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json'


class SeasonalSplitTests(unittest.TestCase):
    setUp = test_time_splits.TimeSplitTests.setUp
    tearDown = test_time_splits.TimeSplitTests.tearDown

    def test_all_months_and_all_inner_roles(self):
        cfg, outers = load_plan(CONFIG)
        self.assertEqual(len(outers),12)
        self.assertEqual({o['start'].month for o in outers},set(range(1,13)))
        for o in outers:
            self.assertEqual(len(o['inner']),36)
            for role in cfg['inner_role_start_days']:
                windows = [i for i in o['inner'] if i['purpose']==role]
                self.assertEqual({i['start'].month for i in windows},set(range(1,13)))
            for i in o['inner']:
                self.assertLessEqual(i['end'],o['training_cutoff'])
                self.assertLess(i['training_cutoff'],i['start'])
            ordered = sorted(o['inner'],key=lambda x:x['start'])
            self.assertTrue(all(a['end']<=b['start'] for a,b in zip(ordered,ordered[1:])))

    def test_partial_actuals_never_become_full_settlement(self):
        old = self.folds[0]
        w = {**old, 'training_cutoff':old['refit_cutoff'],'purpose':'model_selection'}
        self.c.execute("DELETE FROM targets WHERE site_id='PV001' AND ts=TIMESTAMPTZ '2025-01-15 01:00+09'")
        window_views(self.c,w)
        self.assertEqual(self.c.execute('SELECT count(*) FROM diagnostic_actual').fetchone()[0],408*253-1)
        self.assertEqual(self.c.execute('SELECT count(*) FROM settlement_prices').fetchone()[0],407)
        self.assertEqual(self.c.execute('SELECT count(*) FROM settlement_actual').fetchone()[0],407*253)
        self.assertEqual(self.c.execute('SELECT count(*) FROM training_targets JOIN diagnostic_actual USING(site_id,ts)').fetchone()[0],0)

    def test_outer_lock_and_early_stopping(self):
        old = self.folds[0]
        w = {**old,'training_cutoff':old['refit_cutoff'],'purpose':'outer_evaluation'}
        with self.assertRaisesRegex(ValueError,'locked'):
            window_views(self.c,w)
        window_views(self.c,w,allow_outer_evaluation=True)
        self.assertLess(self.c.execute('SELECT max(ts) FROM fit_features').fetchone()[0],
                        self.c.execute('SELECT min(ts) FROM stop_features').fetchone()[0])
        self.assertNotIn('target_kwh',[r[0] for r in self.c.execute('DESCRIBE diagnostic_features').fetchall()])


if __name__ == '__main__':
    unittest.main()
