from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_common import connect
from seasonal_splits import load_plan
from run_pipeline import seasonal_windows, span, select_candidates, submit, parser, available_errors
from train_cycles import train_predict
from fixed_model import PARAMS, WEATHER
from sparse_site_groups import TERRAIN, ERRORS
from test_pipeline import create_sites


class IntegratedPipelineTests(unittest.TestCase):
    def test_refresh_uses_only_available_new_errors_and_old_fallback(self):
        times = pd.date_range('2024-01-01 12:00', periods=40, freq='D', tz='Asia/Seoul')
        frame = pd.DataFrame({'site_id':'A','ts':times,'forecast_kwh':[101.]*30+[9999.]*10})
        old = pd.DataFrame({'site_id':['A','B'],'error_rows':[100,100],
            'signed_nmae':[-.2,-.3],'positive_fraction':[0.,0.],'negative_fraction':[1.,1.]})
        with connect() as con:
            con.register('targets',frame[['site_id','ts']].assign(energy_raw_kwh=100.))
            con.register('sites',pd.DataFrame({'site_id':['A'],'capacity_kw':[100.]}))
            args = parser().parse_args(['--output-dir','unused'])
            with patch('run_pipeline.error_profile',return_value=old) as historical:
                result = available_errors(con,frame,times[0]-pd.Timedelta(hours=1),times[29],args).set_index('site_id')
                self.assertEqual(result.loc['A','error_rows'],30)
                self.assertAlmostEqual(result.loc['A','signed_nmae'],.01)
                self.assertEqual(result.loc['B','signed_nmae'],-.3)
                self.assertEqual(historical.call_args.args[2],times[29])

    def test_separate_round_baseline_averages_key_aligned_predictions(self):
        times = pd.to_datetime(['2024-01-01T12:00:00+09:00','2024-01-03T12:00:00+09:00'])
        frame = pd.DataFrame({'site_id':['A','A'],'ts':times})
        for r in (10,17):
            for col in WEATHER:
                frame[f'{col}_r{r}'] = float(r)
        window = {'id':'fixture','training_start':times[0]-pd.Timedelta(days=1),
                  'training_cutoff':times[0], 'start':times[1]-pd.Timedelta(hours=1),'end':times[1]}
        class Model:
            def save_model(self,path):
                Path(path).write_text('fixture')
        def predict_round(model, data, threads):
            return data[['site_id','ts']].assign(forecast_kwh=data.wa_cloud)
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            con.register('paired',frame)
            con.register('targets',frame[['site_id','ts']].assign(target_kwh=1.,training_eligible=True))
            with patch('train_cycles.fixed_model.fit',return_value=(Model(),1)) as fitter, \
                 patch('train_cycles.fixed_model.predict',side_effect=predict_round), \
                 patch('train_cycles.lgb.Booster',return_value=Model()):
                result,key = train_predict(con,window,pd.DataFrame({'site_id':['A'],'cluster':[0]}),
                                           dict(PARAMS),Path(tmp),1,'separate')
                self.assertEqual(result.forecast_kwh.tolist(),[13.5])
                self.assertEqual(fitter.call_count,2)
                self.assertTrue((Path(tmp)/'cache'/key/'cluster0_round17.txt').exists())

    def test_seasonal_origins_and_last_seven_hours(self):
        _, folds = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')
        for outer in folds:
            opt, selection = seasonal_windows(outer)
            self.assertEqual(len(opt), 12)
            self.assertEqual(len(selection), 12)
            self.assertLessEqual(opt[-1]['end'], selection[0]['training_cutoff'])
            self.assertLessEqual(selection[-1]['end'], outer['training_cutoff'])
            for w in opt+selection:
                self.assertLess(w['training_cutoff'], w['start'])

    def test_seasonal_search_never_uses_outer_actuals(self):
        outer = load_plan(Path(__file__).resolve().parents[1]/'configs/seasonal_splits_v2.json')[1][0]
        months = pd.date_range('2021-02-01', '2024-06-01', freq='MS', tz='Asia/Seoul')
        times = pd.DataFrame({'ts': [m+pd.Timedelta(days=d,hours=h) for m in months
                                    for d in (1,2,3) for h in range(10,18)]})
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.register('times', times)
            con.execute('CREATE TABLE prices AS SELECT ts,100.0 AS smp,1 AS k FROM times')
            con.execute('''CREATE TABLE targets AS SELECT site_id,ts,
                80.0+(right(site_id,3)::INT%3)*20 AS target_kwh,
                80.0+(right(site_id,3)::INT%3)*20 AS energy_raw_kwh,true AS training_eligible
                FROM sites CROSS JOIN times''')
            con.execute('CREATE VIEW features10 AS SELECT site_id,ts,10 AS round FROM targets')
            con.execute('CREATE VIEW features17 AS SELECT site_id,ts,17 AS round FROM targets')
            con.execute('''CREATE TABLE reference_composition AS SELECT site_id,
                CASE current_group WHEN 'A' THEN 'G1' ELSE 'G2' END AS "group" FROM sites''')
            ids = con.execute('SELECT site_id FROM sites ORDER BY site_id').df()
            terrain = ids.copy()
            for col in TERRAIN:
                terrain[col] = 0.
            args = parser().parse_args(['--output-dir', tmp, '--trials', '1', '--cycles', '1',
                                       '--search-steps', '5', '--clusters', '2'])
            args.error_run = Path(tmp)
            calls = []

            def predictions(c, windows, assignment, params, options, family='paired', refresh=None):
                calls.append((windows[0]['purpose'], family))
                self.assertLessEqual(windows[-1]['end'], outer['training_cutoff'])
                start, end = span(windows)['start'], span(windows)['end']
                # A fixed predictor independent of labels, with paired better than separate.
                return c.execute('''SELECT site_id,ts,? AS forecast_kwh FROM targets
                    WHERE ts>? AND ts<=? ORDER BY ts,site_id''',
                    [100. if family=='paired' else 120., start, end]).df(), []

            with patch('run_pipeline.predict_windows', side_effect=predictions), \
                 patch('run_pipeline.terrain_profile', return_value=terrain), \
                 patch('run_pipeline.error_profile', return_value=pd.DataFrame(columns=['site_id','error_rows']+ERRORS)):
                first = select_candidates(con, outer, args)
                self.assertEqual(set(first['finalists']), {'separate','paired','cycles','profiles2'})
                self.assertEqual(first['selected']['mode'], 'paired')
                n = len(calls)
                select_candidates(con, outer, args)
                self.assertEqual(len(calls), n)
                con.execute("UPDATE targets SET target_kwh=999999,energy_raw_kwh=999999 WHERE ts>='2024-06-01'::TIMESTAMPTZ")
                args.output_dir = Path(tmp)/'other'
                second = select_candidates(con, outer, args)
                self.assertEqual(first['selected'], second['selected'])
                self.assertTrue(list((Path(tmp)/outer['id']/'search').glob('*_refresh.json')))

    def test_submission_roundtrip_exact_grid_and_resume_tamper(self):
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.execute('''CREATE TABLE test_prices AS SELECT * FROM (VALUES
                (TIMESTAMPTZ '2025-06-01 01:00:00+09',100.0,1),
                (TIMESTAMPTZ '2025-06-01 02:00:00+09',100.0,1)) p(ts,smp,k)''')
            assignment = con.execute('''SELECT site_id,CASE current_group WHEN 'A' THEN 0 ELSE 1 END AS cluster
                FROM sites ORDER BY site_id''').df()
            candidate = {'family':'paired', 'mode':'paired', 'params':{},
                         'assignment':assignment.to_dict('records'), 'composition':assignment.to_dict('records')}
            prediction = con.execute('SELECT site_id,ts,1.0 AS forecast_kwh FROM sites CROSS JOIN test_prices').df()
            args = parser().parse_args(['--output-dir', tmp])
            outer = {'training_cutoff': pd.Timestamp('2025-05-31T17:00:00+09:00')}
            with patch('run_pipeline.select_candidates', return_value={'selected':candidate}), \
                 patch('run_pipeline.train_predict', return_value=(prediction,'fake')):
                result = submit(con, outer, args)
                self.assertEqual(result['status'], 'complete')
                frame = pd.read_csv(Path(tmp)/'submission/forecast.csv')
                self.assertEqual(len(frame), 4)
                self.assertEqual(frame.forecast_kwh.sum(), 506)
                self.assertTrue(frame.ts.str.endswith('+09:00').all())
                composition = pd.read_csv(Path(tmp)/'submission/composition.csv')
                self.assertEqual(len(composition), 253)
                (Path(tmp)/'submission/forecast.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError,'Changed completed artifact'):
                    submit(con, outer, args)

    def test_submission_rejects_missing_site_hour(self):
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.execute("CREATE TABLE test_prices AS SELECT TIMESTAMPTZ '2025-06-01 01:00:00+09' AS ts,100 AS smp,1 AS k")
            a = con.execute("SELECT site_id,CASE current_group WHEN 'A' THEN 0 ELSE 1 END AS cluster FROM sites").df()
            candidate = {'family':'paired','params':{},'assignment':a.to_dict('records'),'composition':a.to_dict('records')}
            prediction = con.execute('SELECT site_id,ts,1.0 AS forecast_kwh FROM sites CROSS JOIN test_prices LIMIT 252').df()
            args = parser().parse_args(['--output-dir',tmp])
            with patch('run_pipeline.select_candidates',return_value={'selected':candidate}), \
                 patch('run_pipeline.train_predict',return_value=(prediction,'fake')):
                with self.assertRaisesRegex(ValueError,'Incomplete submission'):
                    submit(con, {}, args)


if __name__ == '__main__':
    unittest.main()
