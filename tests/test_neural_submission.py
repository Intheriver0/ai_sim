from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline_common import connect, export
from run_neural_submission import (CANDIDATE, TRAINING_SOURCES, Predictions, parser,
    validate_prediction_run, select_groups, compare_fold, submit, prepare, run)
from run_parameter_methods import key
from run_pipeline import seasonal_windows, write_submission
from seasonal_splits import load_plan
from train_cycles import finish
from test_pipeline import create_sites
from test_parameter_methods import frame

ROOT = Path(__file__).resolve().parents[1]


class NeuralSubmissionTests(unittest.TestCase):
    def test_three_models_real_fit_prediction_cache_and_cutoff(self):
        import torch
        torch.set_num_threads(1)
        data = frame()
        window = dict(id='outer_2024_05', purpose='outer_evaluation',
            training_start=data.ts.min()-pd.Timedelta(days=2),
            training_cutoff=data.ts.iloc[15], start=pd.Timestamp('2024-05-01T00:00:00+09:00'),
            end=pd.Timestamp('2024-05-31T17:00:00+09:00'))
        # Forecast origin precedes the test month; training fixtures belong to April.
        train = data.copy()
        train.ts = train.ts-pd.Timedelta(days=30)
        window.update(training_start=train.ts.min()-pd.Timedelta(days=2),
                      training_cutoff=pd.Timestamp('2024-04-30T17:00:00+09:00'))
        for model in ('tabm', 'hyper_head', 'mlp'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp, connect() as con:
                con.register('paired', pd.concat([train, data], ignore_index=True).drop(columns='target_kwh'))
                targets = train[['site_id', 'ts', 'target_kwh']].assign(training_eligible=True)
                con.register('targets', targets)
                args = parser().parse_args(['--output-dir', tmp, '--model', model, '--max-epochs', '1'])
                assignment = pd.DataFrame(dict(site_id=['a', 'b'], cluster=[0, 0]))
                predictions = Predictions(con, assignment, args)
                predictions.experiment.device = 'cpu'
                first = predictions.get(window)
                self.assertEqual(len(first), len(data))
                # Future labels cannot affect either fit or cached forecasts.
                con.register('targets', pd.concat([targets, data[['site_id', 'ts']].assign(
                    target_kwh=999999., training_eligible=True)], ignore_index=True))
                again = predictions.get(dict(window, id='different_fold', end=data.ts.iloc[15]))
                pd.testing.assert_frame_equal(again.reset_index(drop=True),
                    first.loc[first.ts <= data.ts.iloc[15]].reset_index(drop=True))
                self.assertEqual(predictions.directory(window), predictions.directory(dict(window, id='other')))
                self.assertTrue(np.isfinite(first.forecast_kwh).all())
                (predictions.directory(window)/'predictions.parquet').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Changed completed artifact'):
                    predictions.get(window)

    def test_prediction_import_and_missing_history_refit(self):
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            root = Path(tmp)
            args = parser().parse_args(['--output-dir', str(root/'new'), '--prediction-run', str(root/'old')])
            assignment = pd.DataFrame(dict(site_id=['a'], cluster=[0]))
            window = dict(id='outer_2024_06', purpose='outer_evaluation',
                start=pd.Timestamp('2024-06-01T00:00:00+09:00'), end=pd.Timestamp('2024-07-01T00:00:00+09:00'),
                training_cutoff=pd.Timestamp('2024-05-31T17:00:00+09:00'))
            prediction = pd.DataFrame(dict(site_id=['a'], ts=[window['start']+pd.Timedelta(hours=1)], forecast_kwh=[1.]))
            con.register('paired', prediction)
            source = root/'old/cache/outer_2024_06'/key(CANDIDATE)/'budget45'
            source.mkdir(parents=True)
            export(con, 'SELECT * FROM paired', source/'predictions.parquet')
            finish(source, {}, [source/'predictions.parquet'])
            provider = Predictions(con, assignment, args)
            actual = provider.get(window)
            self.assertEqual(prediction.ts.tolist(), actual.ts.tolist())
            np.testing.assert_array_equal(prediction.forecast_kwh, actual.forecast_kwh)
            self.assertTrue((provider.directory(window)/'done.json').exists())
            # A month absent in the old run is trained, never filled with another month's predictions.
            prior = dict(window, start=window['start']-pd.DateOffset(months=1), end=window['start'])
            prior_prediction = prediction.assign(ts=[prior['start']+pd.Timedelta(hours=1)])
            con.register('paired', prior_prediction)
            with patch.object(provider.experiment, 'predict_window', return_value=prior_prediction) as fit:
                provider.get(prior)
                self.assertEqual(fit.call_args.args[0]['id'], 'oof_2024_05')

    def test_import_rejects_wrong_model_data_code_and_missing_hyper_terrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(['--output-dir', 'unused', '--prediction-run', tmp])
            current = dict(config='config', regions='regions', inputs={'targets': 'hash'},
                sources={n: 'source' for n in TRAINING_SOURCES},
                versions={n: 'version' for n in ('torch', 'tabm', 'numpy', 'pandas', 'duckdb')})
            old = dict(current, options=dict(model='tabm', region='ridge_k4', max_epochs=45,
                                            seed=42, threads=4, methods=['fixed']))
            path = Path(tmp)/'manifest.json'
            path.write_text(json.dumps(old))
            (Path(tmp)/'comparison.json').write_text(json.dumps(dict(status='complete', folds=12, inner_evaluation=False)))
            validate_prediction_run(args, current)
            for change in ('model', 'inputs', 'sources'):
                bad = deepcopy(old)
                if change == 'model':
                    bad['options']['model'] = 'mlp'
                elif change == 'inputs':
                    bad['inputs']['targets'] = 'other'
                else:
                    bad['sources']['parameter_models.py'] = 'other'
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    validate_prediction_run(args, current)
            args.model = 'hyper_head'
            old['options']['model'] = 'hyper_head'
            current['terrain'] = {'site_quality.json': 'hash'}
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, 'terrain provenance'):
                validate_prediction_run(args, current)

    def test_groups_use_only_past_labels_and_reference_wins_on_tie(self):
        _, folds = load_plan(ROOT/'configs/seasonal_splits_v2.json')
        outer = folds[0]
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.execute('''CREATE TABLE reference_composition AS SELECT site_id,
                CASE current_group WHEN 'A' THEN 'G1' ELSE 'G2' END AS "group" FROM sites''')
            dates = pd.date_range('2022-06-15T12:00:00+09:00', periods=26, freq='30D')
            con.register('dates', pd.DataFrame(dict(ts=dates)))
            con.execute('CREATE TABLE prices AS SELECT ts,100. AS smp,1 AS k FROM dates')
            con.execute('''CREATE TABLE targets AS SELECT site_id,ts,100.0::DOUBLE AS energy_raw_kwh,
                100.0 AS target_kwh,TRUE AS training_eligible FROM sites CROSS JOIN prices''')
            for r in (10, 17):
                con.execute(f'CREATE VIEW features{r} AS SELECT site_id,ts FROM targets')
            prediction = con.execute('''SELECT site_id,ts,
                100.+CASE current_group WHEN 'A' THEN 10. ELSE -10. END AS forecast_kwh
                FROM targets JOIN sites USING(site_id)''').df()
            class History:
                def history(self, windows):
                    self_outer = outer
                    for w in windows:
                        self_test.assertLess(w['training_cutoff'], w['start'])
                        self_test.assertLessEqual(w['end'], self_outer['training_cutoff'])
                    return prediction.loc[(prediction.ts > windows[0]['start']) & (prediction.ts <= windows[-1]['end'])]
                def get(self, window):
                    return prediction.loc[(prediction.ts > window['start']) & (prediction.ts <= window['end'])]
            self_test = self
            args = parser().parse_args(['--output-dir', str(Path(tmp)/'first'), '--search-steps', '20'])
            first = select_groups(con, outer, History(), args)
            self.assertEqual(first['selected'], 'optimized')
            self.assertLess(first['search']['after'], first['search']['before'])
            # Real settlement evaluation leaves site NMAE unchanged but improves group W.
            result = compare_fold(con, outer, History(), args)
            self.assertAlmostEqual(result['reference']['site_equal_nmae'], result['selected']['site_equal_nmae'])
            self.assertLess(result['selected']['settlement']['W_won'], result['reference']['settlement']['W_won'])
            opt, sel = seasonal_windows(outer)
            self.assertLessEqual(opt[-1]['end'], sel[0]['training_cutoff'])
            con.execute("UPDATE targets SET energy_raw_kwh=999999 WHERE ts>='2024-06-01'::TIMESTAMPTZ")
            args.output_dir = Path(tmp)/'second'
            second = select_groups(con, outer, History(), args)
            self.assertEqual(first, second)
            # A frozen selection is protected against manual edits on resume.
            (args.output_dir/outer['id']/'group_selection/selected.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed completed artifact'):
                select_groups(con, outer, History(), args)
            prediction.forecast_kwh = 100.
            args.output_dir = Path(tmp)/'tie'
            self.assertEqual(select_groups(con, outer, History(), args)['selected'], 'reference')

    def test_full_year_submission_roundtrip_checkpoint_integrity_and_bad_forecast(self):
        with tempfile.TemporaryDirectory() as tmp, connect() as con:
            create_sites(con)
            con.execute('''CREATE TABLE test_prices AS SELECT ts,100. AS smp,1 AS k
                FROM generate_series(TIMESTAMPTZ '2025-06-01 01:00:00+09',
                TIMESTAMPTZ '2026-06-01 00:00:00+09',INTERVAL '1 hour') t(ts)''')
            assignment = con.execute("SELECT site_id,CASE current_group WHEN 'A' THEN 0 ELSE 1 END AS cluster FROM sites").df()
            prediction = con.execute('SELECT site_id,ts,1. AS forecast_kwh FROM sites CROSS JOIN test_prices').df()
            args = parser().parse_args(['--output-dir', tmp])
            deploy = dict(id='deployment', purpose='deployment', training_cutoff=pd.Timestamp('2025-05-31T17:00:00+09:00'))
            provider = Predictions(con, assignment, args)
            directory = provider.directory(deploy)
            directory.mkdir(parents=True)
            con.register('fixture_prediction', prediction)
            export(con, 'SELECT * FROM fixture_prediction', directory/'predictions.parquet')
            finish(directory, {}, [directory/'predictions.parquet'])
            for cluster in (0, 1):
                state = directory.parent/'models'/f'cluster{cluster}'
                state.mkdir(parents=True)
                (state/'state.pt').write_bytes(b'fixture')
                (state/'state.sha256').write_text('fixture')
            with patch('run_neural_submission.select_groups', return_value=dict(
                selected='reference', composition=assignment.to_dict('records'))), \
                 patch.object(provider, 'get', return_value=prediction):
                result = submit(con, deploy, provider, args)
                self.assertEqual(result['status'], 'complete')
                forecast = pd.read_csv(Path(tmp)/'submission/forecast.csv')
                self.assertEqual(list(forecast), ['group', 'ts', 'forecast_kwh'])
                self.assertEqual(len(forecast), 17520)
                self.assertTrue(forecast.ts.str.endswith('+09:00').all())
                self.assertEqual(forecast.forecast_kwh.sum(), 253*8760)
                self.assertEqual(len(pd.read_csv(Path(tmp)/'submission/composition.csv')), 253)
                self.assertEqual(submit(con, deploy, provider, args)['status'], 'complete')
                (state/'state.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'Submission model changed'):
                    submit(con, deploy, provider, args)
            output = Path(tmp)/'invalid'
            output.mkdir()
            with self.assertRaisesRegex(ValueError, 'Incomplete submission'):
                write_submission(con, prediction.iloc[:-1], output)
            con.fetchall()
            bad = prediction.copy()
            bad.loc[0, 'forecast_kwh'] = np.inf
            with self.assertRaisesRegex(ValueError, 'Invalid site predictions'):
                write_submission(con, bad, output)

    def test_rejects_short_test_calendar_before_training(self):
        with connect() as con:
            con.execute("CREATE TABLE test_prices AS SELECT TIMESTAMPTZ '2025-06-01 01:00:00+09' AS ts")
            with patch('run_neural_submission.attach_all'):
                with self.assertRaisesRegex(ValueError, '8760-hour'):
                    prepare(con, parser().parse_args(['--output-dir', 'unused']), {})

    def test_stage_gate_and_lock_cleanup(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            output = Path(tmp)/'run'
            args = parser().parse_args(['--output-dir', str(output), '--stage', 'preflight'])
            assignment = pd.DataFrame(dict(site_id=['a'], cluster=[0]))
            with patch('run_neural_submission.manifest', return_value={'fixture': True}), \
                 patch('run_neural_submission.prepare', return_value=(assignment, {})), \
                 patch('run_neural_submission.preflight') as probe, \
                 patch('run_neural_submission.submit') as final:
                self.assertEqual(run(args)['stage'], 'preflight')
                self.assertFalse((output/'run.lock').exists())
                args.stage, args.resume = 'submit', True
                with self.assertRaisesRegex(ValueError, 'Complete --stage compare'):
                    run(args)
                final.assert_not_called()
                self.assertFalse((output/'run.lock').exists())
                (output/'run.lock').write_text('other process')
                with self.assertRaises(FileExistsError):
                    run(args)
                self.assertEqual((output/'run.lock').read_text(), 'other process')
                (output/'run.lock').unlink()
                probe.side_effect = RuntimeError('CUDA unavailable')
                with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                    run(args)
                self.assertFalse((output/'run.lock').exists())


if __name__ == '__main__':
    unittest.main()
