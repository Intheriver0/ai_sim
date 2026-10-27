from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_common import connect
from seasonal_splits import load_plan, window_views
from site_groups import META, PATTERN, group_sites, profiles
import test_time_splits


class SiteGroupTests(unittest.TestCase):
    def profile(self):
        frame = pd.DataFrame({'site_id': [f'PV{i:03}' for i in range(1,254)],
                              'history_rows': 1000})
        rng = np.random.default_rng(42)
        for name in META + PATTERN:
            frame[name] = np.repeat([0., 10., 20.], [84,84,85])+rng.normal(0,.1,253)
        for name in PATTERN:
            frame[f'{name}_rows'] = 100
        frame.loc[252, PATTERN] = np.nan
        frame.loc[252, [f'{p}_rows' for p in PATTERN]] = 0
        return frame

    def test_all_sites_deterministic_and_cold_assignment(self):
        for k in (1,2,3):
            profile = self.profile()
            first, info = group_sites(profile,k)
            second, _ = group_sites(profile,k)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(first),253)
            self.assertEqual(first.site_id.nunique(),253)
            self.assertEqual(set(first.cluster),set(range(k)))
            self.assertFalse(first.profile_reliable.iloc[-1])
            self.assertTrue(first.cluster.between(0,k-1).all())

    def test_degenerate_profiles_rejected(self):
        frame = self.profile()
        frame.loc[:,META+PATTERN] = 1.
        with self.assertRaisesRegex(ValueError,'distinct'):
            group_sites(frame,3)

    def test_future_actual_does_not_change_profile(self):
        fixture = test_time_splits.TimeSplitTests()
        fixture.setUp()
        try:
            c = fixture.c
            _, windows = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
            w = next(w for w in windows if w['id']=='outer_2025_01')
            window_views(c,w,allow_outer_evaluation=True)
            before = profiles(c)
            c.execute("UPDATE targets SET target_kwh=8,energy_raw_kwh=8 WHERE ts>TIMESTAMPTZ '2025-01-01 00:00+09'")
            pd.testing.assert_frame_equal(before,profiles(c))
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
