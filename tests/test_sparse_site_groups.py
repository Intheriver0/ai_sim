import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_common import connect, export
from sparse_site_groups import discover, error_profile, assign_sparse, TERRAIN, ERRORS


class SparseGroupsTests(unittest.TestCase):
    def test_discovery_requires_final_complete_run_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'tunning_cuda_1028'
            run.mkdir()
            folds = [{'fold': f'outer_2024_{m:02}'} for m in range(1, 13)]
            summary = {'status': 'partial', 'all_12_months': False, 'smoke': False, 'folds': folds}
            (run/'summary.json').write_text(json.dumps(summary))
            (run/'manifest.json').write_text(json.dumps({'smoke': False}))
            with self.assertRaisesRegex(ValueError, 'found 0'):
                discover(root)
            summary.update(status='complete', all_12_months=True)
            (run/'summary.json').write_text(json.dumps(summary))
            for f in folds:
                folder = run/f['fold']
                folder.mkdir()
                (folder/'result.json').write_text('{}')
                (folder/'blend_predictions.parquet').write_text('fixture')
            found, hashes = discover(root)
            self.assertEqual(found, run.resolve())
            self.assertEqual(len(hashes), 26)
            import shutil
            shutil.copytree(run, root/'other_cuda_1028')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                discover(root)
            self.assertEqual(discover(root, run)[0], run.resolve())

    def test_historical_errors_exclude_future_and_preserve_sign(self):
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            run = Path(tmp)
            folder = run/'outer_2024_06'
            folder.mkdir()
            (run/'summary.json').write_text(json.dumps({'folds': [{'fold': folder.name}]}))
            (folder/'result.json').write_text(json.dumps({'training_cutoff': '2024-05-31T17:00:00+09:00'}))
            ts = pd.to_datetime(['2024-06-01T12:00:00+09:00', '2024-06-02T12:00:00+09:00',
                                 '2024-06-03T12:00:00+09:00'])
            frame = pd.DataFrame({'site_id': 'A', 'ts': ts, 'forecast_kwh': [12., 8., 9999.]})
            con.register('pred', frame)
            export(con, 'SELECT * FROM pred', folder/'blend_predictions.parquet')
            con.register('targets', frame[['site_id', 'ts']].assign(energy_raw_kwh=10.))
            con.register('sites', pd.DataFrame({'site_id': ['A'], 'capacity_kw': [100.]}))
            result = error_profile(con, run, ts[1]).iloc[0]
            self.assertEqual(result.error_rows, 2)
            self.assertAlmostEqual(result.signed_nmae, 0.)
            self.assertEqual(result.positive_fraction, .5)
            self.assertEqual(result.negative_fraction, .5)
            self.assertTrue(error_profile(con, run, pd.Timestamp('2024-01-01', tz='Asia/Seoul')).empty)

    def fixture(self):
        assignment = pd.DataFrame({'site_id': ['A', 'B', 'C'], 'lat': [37., 37., 37.],
            'lon': [127., 127.1, 127.01], 'cluster': [0, 1, 0], 'profile_reliable': [True, True, False]})
        terrain = pd.DataFrame({'site_id': ['A', 'B', 'C']})
        for col in TERRAIN:
            terrain[col] = [0., 10., 0.]
        errors = pd.DataFrame({'site_id': ['A', 'B', 'C'], 'error_rows': [100, 100, 100],
            'signed_nmae': [.1, -.1, -.1], 'positive_fraction': [1., 0., 0.],
            'negative_fraction': [0., 1., 1.]})
        return assignment, terrain, errors

    def test_error_bias_can_change_sparse_assignment(self):
        a, t, e = self.fixture()
        # Same terrain lets the independent error block select the similar-bias peer.
        t[TERRAIN] = 0.
        result = assign_sparse(a, t, e)
        self.assertEqual(result.iloc[2].cluster, 1)
        self.assertEqual(result.iloc[2].assignment_basis, 'geography_terrain_error')
        self.assertEqual(result.iloc[2].peer_site_id, 'B')
        self.assertEqual(result.cluster.iloc[:2].tolist(), [0, 1])

    def test_insufficient_error_history_uses_geography_and_terrain(self):
        a, t, e = self.fixture()
        e.loc[2, 'error_rows'] = 2
        result = assign_sparse(a, t, e)
        self.assertEqual(result.iloc[2].cluster, 0)
        self.assertEqual(result.iloc[2].assignment_basis, 'geography_terrain_no_past_error')
        t.loc[2, TERRAIN] = 10.
        self.assertEqual(assign_sparse(a, t, e).iloc[2].cluster, 1)


if __name__ == '__main__':
    unittest.main()
