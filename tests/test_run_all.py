import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_all


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'scripts').mkdir()
        self.patch = patch.object(run_all, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.args = run_all.parser().parse_args([])
        self.calls = []

    def execute(self, command, emit):
        self.calls.append(command)
        emit('model output')
        output = Path(command[command.index('--output-dir')+1])
        output.mkdir(parents=True, exist_ok=True)
        run_all.write_json(output/'manifest.json', {})
        preflight = '--preflight-only' in command or ('--stage' in command and
                    command[command.index('--stage')+1] == 'preflight')
        run_all.write_json(output/('preflight.json' if preflight else 'comparison.json'),
                           {'status': 'complete', 'comparisons': [{'value': 1}]})
        return 0

    def run_suite(self, execute=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_all.run(self.args, execute or self.execute)

    def report(self):
        return json.loads((self.args.output_dir/'results.json').read_text(encoding='utf-8'))

    def test_coverage_and_child_arguments(self):
        from parameter_search import METHODS
        from parameter_models import MODELS
        from run_parameter_methods import parser as parameter_parser
        from run_pipeline import parser as pipeline_parser
        self.assertEqual(set(run_all.METHODS), set(METHODS)-{'dehb'})
        self.assertEqual(set(run_all.MODELS), set(MODELS))
        plans = run_all.jobs(self.args)
        self.assertEqual(len(plans), 9)
        for job in plans:
            command = job['command'][3:]
            if job['name'].startswith('parameter_'):
                parsed = parameter_parser().parse_args(command)
                expected = set(METHODS)-{'dehb'}
                if parsed.model == 'mlp_hypergradient':
                    expected -= {'pbt'}
                self.assertEqual(set(parsed.methods), expected)
            elif job['name'] == 'pipeline':
                pipeline_parser().parse_args(command+['--stage', 'compare'])

    def test_combined_logs_reports_and_resume(self):
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(len(self.report()['jobs']), 9)
        self.assertEqual(self.report()['status'], 'complete')
        self.assertTrue(all(j['artifacts']['comparison.json']['comparisons'] for j in self.report()['jobs']))
        log = (self.args.output_dir/'run.log').read_text(encoding='utf-8')
        self.assertEqual(log.count('model output'), 9)
        self.args.resume = True
        self.calls.clear()
        self.assertEqual(self.run_suite(), 0)
        self.assertTrue(all('--resume' in c for c in self.calls))
        self.args.threads += 1
        with self.assertRaises(ValueError):
            self.run_suite()

    def test_failure_continues_and_exit_zero_without_result_fails(self):
        def execute(command, emit):
            if not self.calls:
                self.calls.append(command)
                return 2
            return self.execute(command, emit)
        self.assertEqual(self.run_suite(execute), 1)
        self.assertEqual(len(self.calls), 9)
        self.assertEqual(self.report()['jobs'][0]['status'], 'failed')
        self.args.resume = True
        self.assertEqual(self.run_suite(lambda command, emit: 0), 1)

    def test_preflight_then_full_and_interrupt(self):
        self.args.preflight_only = True
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(self.report()['status'], 'preflight_complete')
        self.args.resume = True
        self.args.preflight_only = False
        self.assertEqual(self.run_suite(), 0)
        def interrupt(command, emit):
            raise KeyboardInterrupt
        self.assertEqual(self.run_suite(interrupt), 130)
        self.assertEqual(self.report()['status'], 'interrupted')
        self.assertEqual(self.report()['jobs'][1]['status'], 'pending')

    def test_dry_run_writes_nothing(self):
        self.args.dry_run = True
        self.assertEqual(self.run_suite(), 0)
        self.assertFalse(self.args.output_dir.exists())

    def test_real_child_stdout_stderr_and_failure(self):
        lines = []
        code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
        self.assertEqual(run_all.launch([sys.executable, '-u', '-c', code], lines.append), 3)
        self.assertEqual(set(lines), {'out', 'err'})


if __name__ == '__main__':
    unittest.main()
