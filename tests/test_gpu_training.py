from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fixed_model import PARAMS, fit
from gpu_training import check_cuda, require_cuda_params
from train_tuned import fit_round, run_model, suggest


class GPUTrainingTests(unittest.TestCase):
    def test_fixed_and_every_trial_require_cuda(self):
        require_cuda_params(PARAMS)
        trial = Mock()
        trial.suggest_float.return_value = .1
        trial.suggest_int.return_value = 31
        require_cuda_params(suggest(trial, 100))
        self.assertNotIn('force_col_wise', PARAMS)
        self.assertNotIn('deterministic', PARAMS)

    def test_cpu_and_opencl_are_rejected_before_reading_data(self):
        for device in ('cpu', 'gpu', None):
            params = dict(PARAMS, device_type=device)
            with self.assertRaisesRegex(ValueError, 'mandatory'):
                fit_round(None, 10, params, 1, True)
            with self.assertRaisesRegex(ValueError, 'mandatory'):
                run_model(None, None, params, True, None, 1)
            with patch.dict(PARAMS, device_type=device), self.assertRaisesRegex(ValueError, 'mandatory'):
                fit(None, 1)

    def test_probe_failure_does_not_retry_on_cpu(self):
        with patch('gpu_training.lgb.train', side_effect=RuntimeError('CUDA unavailable')) as trainer:
            with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                check_cuda()
            self.assertEqual(trainer.call_count, 1)
            self.assertEqual(trainer.call_args.args[0]['device_type'], 'cuda')

    def test_probe_checks_trained_backend(self):
        model = Mock(params={'device_type': 'cpu'})
        with patch('gpu_training.lgb.train', return_value=model):
            with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                check_cuda()

    def test_probe_success_is_reported(self):
        model = Mock(params={'device_type': 'cuda'})
        model.num_trees.return_value = 3
        model.predict.return_value = np.zeros(256)
        with patch('gpu_training.lgb.train', return_value=model):
            self.assertEqual(check_cuda()['probe'], 'passed')


if __name__ == '__main__':
    unittest.main()
