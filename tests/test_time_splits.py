from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pipeline_common import connect, prepare_prices
from time_splits import audit_fold, fold_views, load_plan, validate_inputs
from test_pipeline import create_sites

CONFIG = Path(__file__).resolve().parents[1] / 'configs/time_splits_v1.json'


class TimeSplitTests(unittest.TestCase):
    def setUp(self):
        self.c = connect()
        create_sites(self.c)
        # The old boundaries remain regression fixtures, not an approved plan.
        self.temp = tempfile.TemporaryDirectory()
        historical = json.loads(CONFIG.read_text(encoding='utf-8'))
        historical.pop('status', None)
        path = Path(self.temp.name)/'historical_fixture.json'
        path.write_text(json.dumps(historical),encoding='utf-8')
        self.config, self.folds = load_plan(path)
        self.c.execute('''CREATE TABLE raw_prices AS SELECT ts, 100.0 AS smp FROM
            generate_series(TIMESTAMPTZ '2024-12-25 01:00+09',
                            TIMESTAMPTZ '2025-02-01 00:00+09', INTERVAL '1 hour') t(ts)''')
        prepare_prices(self.c)
        self.c.execute('''CREATE TABLE targets AS SELECT site_id, ts,
            2.0 AS energy_raw_kwh, 2.0 AS target_kwh, true AS training_eligible
            FROM sites CROSS JOIN prices''')
        for r in (10, 17):
            self.c.execute(f'''CREATE TABLE features{r} AS SELECT site_id, ts, {r} AS round,
                CAST(ts-INTERVAL '1 hour' AS DATE) AS target_day,
                date_trunc('day',ts-INTERVAL '1 hour')-INTERVAL '1 day'+INTERVAL '{r} hours' AS issued_at
                FROM targets''')

    def tearDown(self):
        self.c.close()
        self.temp.cleanup()

    def test_old_default_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Rejected'):
            load_plan(CONFIG)

    def test_default_boundaries(self):
        self.assertEqual([int((f['end']-f['start']).total_seconds()/3600) for f in self.folds],
                         [408, 672, 744, 720, 744])
        first = self.folds[0]
        self.assertEqual(first['refit_cutoff'].isoformat(), '2025-01-14T17:00:00+09:00')
        self.assertEqual(first['fit_cutoff'].isoformat(), '2025-01-07T17:00:00+09:00')
        self.assertLess(first['early_stop_end'], first['refit_cutoff'])

    def test_views_are_chronological_and_keep_both_rounds(self):
        validate_inputs(self.c)
        first = self.folds[0]
        audited = audit_fold(self.c, first, 300)
        self.assertEqual(audited['minimum_fit_rows'], 329)
        fold_views(self.c, first)
        self.assertLess(self.c.execute('SELECT max(ts) FROM fit_targets').fetchone()[0],
                        self.c.execute('SELECT min(ts) FROM stop_targets').fetchone()[0])
        self.assertEqual(self.c.execute('SELECT max(ts) FROM refit_targets').fetchone()[0], first['refit_cutoff'])
        self.assertEqual(self.c.execute('SELECT min(ts),max(ts) FROM evaluation_prices').fetchone(),
                         (first['start']+timedelta(hours=1), first['end']))
        self.assertEqual(self.c.execute('SELECT count(*) FROM evaluation_features').fetchone()[0], 408*253*2)
        self.assertNotIn('target_kwh', [r[0] for r in self.c.execute('DESCRIBE evaluation_features').fetchall()])
        self.assertEqual(self.c.execute('''SELECT count(*) FROM refit_targets
            JOIN evaluation_actual USING(site_id,ts)''').fetchone()[0], 0)

    def test_holdout_requires_explicit_open(self):
        with self.assertRaisesRegex(ValueError, 'locked'):
            fold_views(self.c, self.folds[-1])
        fold_views(self.c, self.folds[-1], allow_final_holdout=True)

    def test_missing_actual_and_short_history_rejected(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            audit_fold(self.c, self.folds[0], 400)
        self.c.execute("DELETE FROM targets WHERE site_id='PV001' AND ts=TIMESTAMPTZ '2025-01-15 01:00+09'")
        with self.assertRaisesRegex(ValueError, 'incomplete actual'):
            audit_fold(self.c, self.folds[0], 300)

    def test_future_issue_time_rejected(self):
        self.c.execute("UPDATE features17 SET issued_at=issued_at+INTERVAL '1 day' WHERE site_id='PV001'")
        with self.assertRaisesRegex(ValueError, 'issue time'):
            validate_inputs(self.c)

    def test_invalid_plans_rejected(self):
        variants = []
        cfg = deepcopy(self.config)
        cfg['folds'][1]['start_day'] = '2025-01-31'
        variants.append(cfg)
        cfg = deepcopy(self.config)
        cfg['folds'][-1]['purpose'] = 'model_selection'
        variants.append(cfg)
        cfg = deepcopy(self.config)
        cfg['generation_availability_lag_hours'] = 48
        variants.append(cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plan.json'
            for cfg in variants:
                path.write_text(json.dumps(cfg), encoding='utf-8')
                with self.subTest(config=cfg), self.assertRaises(ValueError):
                    load_plan(path)


if __name__ == '__main__':
    unittest.main()
