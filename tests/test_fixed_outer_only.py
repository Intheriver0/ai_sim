from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_parameter_methods as runner
from parameter_models import MODELS
from parameter_search import spec
from seasonal_splits import load_plan


class FixedOuterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = runner.parser().parse_args(['--output-dir', str(self.root/'results/fixed')])
        self.config, self.windows = load_plan(self.args.config)
        self.args.regions_dir = self.root/'regions'
        self.args.regions_dir.mkdir()
        (self.args.regions_dir/'manifest.json').write_text(json.dumps({'sites_sha256':'hash'}))
        self.args.data_dir = self.root/'data'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        def mock(name, **kwargs):
            return self.stack.enter_context(patch.object(runner, name, **kwargs))
        mock('ROOT', new=self.root)
        mock('digest', return_value='hash')
        mock('load_plan', return_value=(self.config, self.windows))
        con = MagicMock()
        sample = pd.DataFrame({'site_id':['a'], 'cluster':[0]})
        con.execute.return_value.df.return_value = sample
        mock('connect').return_value.__enter__.return_value = con
        mock('attach_data')
        mock('validate_inputs')
        mock('attach_pair')
        mock('read_view')
        mock('assignments', return_value={'ridge_k4':sample})
        self.views = mock('window_views')
        mock('evaluate', return_value={'site_equal_nmae':.1,'complete_253_hours':0,'settlement':None})
        mock('daily_penalty', side_effect=lambda con,p,w:pd.DataFrame({'day':[pd.Timestamp(w['start']).normalize().tz_localize(None)], 'W':[0.]}))
        self.stack.enter_context(patch('sparse_site_groups.terrain_profile', return_value=sample))
        self.stack.enter_context(patch('torch.cuda.is_available', return_value=True))
        self.stack.enter_context(patch('torch.cuda.get_device_name', return_value='mock GPU'))
        self.stack.enter_context(patch('parameter_models.Learner'))
        self.predictions = self.stack.enter_context(patch.object(runner.Experiment, 'predict_window', return_value=sample))
        self.score = self.stack.enter_context(patch.object(runner.Experiment, 'score', side_effect=AssertionError('Unexpected inner fit')))
        self.real_inner = runner.inner_windows
        self.inner = mock('inner_windows', side_effect=AssertionError('Unexpected inner windows'))
        self.search = mock('run_search', side_effect=AssertionError('Unexpected search'))

    def test_every_fixed_model_skips_inner_but_keeps_outer_and_controls(self):
        for model in MODELS:
            with self.subTest(model=model):
                self.args.model = model
                self.args.objective = 'W'  # Inner W coverage must not block fixed-only runs.
                self.args.output_dir = self.root/'results'/model
                self.predictions.reset_mock()
                self.views.reset_mock()
                runner.run(self.args)
                references = 2 if model == 'mlp_hypergradient' else 1 if model == 'mlp_learned_optimizer' else 0
                self.assertEqual(self.predictions.call_count, 12*(1+references))
                for call in self.predictions.call_args_list:
                    outer, candidate, budget = call.args
                    self.assertEqual(outer['purpose'], 'outer_evaluation')
                    self.assertEqual(candidate, spec(dict(hidden=64,lr=.001,weight_decay=.0003)))
                    self.assertEqual(budget, 45)
                self.assertEqual({c.args[0]['id'] for c in self.predictions.call_args_list}, {w['id'] for w in self.windows})
                self.assertTrue(all(c.kwargs['allow_outer_evaluation'] for c in self.views.call_args_list))
                report = json.loads((self.args.output_dir/'comparison.json').read_text())
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(report['folds'], 12)
                self.assertFalse(report['inner_evaluation'])
                selected = json.loads((self.args.output_dir/self.windows[0]['id']/'fixed/selected.json').read_text())
                self.assertIsNone(selected['score'])
                self.assertIsNone(selected['baseline_score'])
                self.assertEqual(selected['search_evaluations'], 0)
        self.inner.assert_not_called()
        self.score.assert_not_called()
        self.search.assert_not_called()

    def test_fixed_preflight_resume_and_selection_integrity(self):
        self.args.preflight_only = True
        runner.run(self.args)
        self.predictions.assert_not_called()
        self.args.preflight_only = False
        self.args.resume = True
        runner.run(self.args)
        before = (self.args.output_dir/'fold_metrics.json').read_text()
        runner.run(self.args)
        self.assertEqual(before, (self.args.output_dir/'fold_metrics.json').read_text())
        self.args.max_epochs = 44
        with self.assertRaisesRegex(ValueError, 'Resume requires identical'):
            runner.run(self.args)
        self.args.max_epochs = 45
        (self.args.output_dir/self.windows[0]['id']/'fixed/selected.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Changed completed artifact'):
            runner.run(self.args)

    def test_mixed_search_still_scores_fixed_and_candidates_on_twelve_inner_windows(self):
        self.args.methods = ['fixed', 'tpe']
        # Restore the real chronological selector, wrapped to inspect every invocation.
        self.inner.side_effect = self.real_inner
        self.score.side_effect = None
        self.score.return_value = .2
        def search(method, objective, directory, trials, minimum, maximum, seed, population):
            directory.mkdir(parents=True,exist_ok=True)
            self.assertEqual((trials,minimum,maximum,seed,population),(24,5,45,42,4))
            candidate = spec(dict(hidden=64,lr=.001,weight_decay=.0003))
            return dict(method=method,spec=candidate,score=objective(candidate,maximum))
        self.search.side_effect = search
        runner.run(self.args)
        self.assertEqual(self.search.call_count, 24)
        self.assertEqual([c.args[0] for c in self.search.call_args_list], ['fixed','tpe']*12)
        self.assertEqual(self.score.call_count, 24)
        for call in self.score.call_args_list:
            self.assertEqual(len(call.args[0]),12)
        report = json.loads((self.args.output_dir/'comparison.json').read_text())
        self.assertTrue(report['inner_evaluation'])
        self.assertEqual(self.predictions.call_count,24)


if __name__ == '__main__':
    unittest.main()
