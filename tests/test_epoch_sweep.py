import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from boundary_unlearning import diagnostics, phases, training
import test_experiment_phases as fixtures


class EpochSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.fixture = fixtures.ExperimentPhaseTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.models, self.rows, config, destination = self.fixture.bundle()
        config.update(seed=2022, batch_size=8, num_workers=0, evaluate_train_splits=True,
                      unlearn_learning_rate=.001, momentum=.9, max_batches_per_epoch=None,
                      fgsm_bound=.1, attack_normalization='matched', attack_mean=[.5]*3, attack_std=[.5]*3,
                      run_tag_settings={'unlearn_epochs': 10}, unlearn_epochs=10)
        self.manifest = phases.save_phase1_bundle(self.models, self.rows, config, destination, run_tag='fixture')
        self.manifest_bytes = self.manifest.read_bytes()

    def run_sweep(self, **options):
        return diagnostics.run_phase1_epoch_sweep(
            self.manifest, self.fixture.train, self.fixture.test, self.root / 'result',
            device='cpu', **options)

    def test_fresh_seeded_students_and_phase2_bundles_without_baseline_training(self):
        torch_state = torch.get_rng_state().clone()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with patch.object(training, 'train_from_scratch', side_effect=AssertionError('No baseline training')):
            result = self.run_sweep(epochs=[1, 2])
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(result['baseline_rows'], self.rows[:2])
        self.assertEqual(self.manifest.read_bytes(), self.manifest_bytes)
        for row in result['rows']:
            self.assertEqual(row['unlearning_update_steps'], row['unlearn_epochs'])
        for budget, manifest in result['bundles'].items():
            restored = phases.load_phase1_bundle(manifest, device='cpu')
            config = restored['config']
            self.assertEqual(config['unlearn_epochs'], budget)
            self.assertEqual(config['run_tag_settings']['unlearn_epochs'], budget)
            self.assertEqual(config['bn_stats_mode'], 'update')
            self.assertEqual(config['unlearning_stats']['boundary_shrink']['update_steps'], budget)
            self.assertEqual([row['model'] for row in restored['rows']], list(phases.PHASE1_MODELS))
            for name in ('original', 'retrain'):
                for key, value in self.models[name].state_dict().items():
                    torch.testing.assert_close(value, restored['models'][name].state_dict()[key], rtol=0, atol=0)
            counter = restored['models']['boundary_shrink'].features[0][1].num_batches_tracked.item()
            self.assertEqual(counter, budget)  # Each budget starts at Original, not the previous student.
        for extension in ('png', 'pdf'):
            self.assertTrue((result['output_dir'] / f'epochs.{extension}').is_file())
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        self.assertEqual(random.getstate(), python_state)

    def test_same_budget_matches_single_seeded_author_run(self):
        result = self.run_sweep(epochs=[2], plots=False)
        restored = phases.load_phase1_bundle(result['bundles'][2], device='cpu')
        config = restored['config']
        from boundary_unlearning import data, unlearning
        from torch.utils.data import Subset
        forget = Subset(self.fixture.train, [4])
        with diagnostics._seeded(config['seed'], 'cpu'):
            loader = data.make_loader(forget, batch_size=8, seed=2022, num_workers=0, pin_memory=False, shuffle=True)
            expected, _ = unlearning.run_boundary_shrink(
                self.models['original'], loader, self.root / 'single.pt', device='cpu', epochs=2,
                learning_rate=.001, momentum=.9, bound=.1, normalization_mode='matched', bn_stats_mode='update')
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(value, restored['models']['boundary_shrink'].state_dict()[key], rtol=0, atol=0)

    def test_invalid_budgets_and_mismatched_dataset_fail_before_training(self):
        for budgets in ([], [0], [True], [1, 1], [1.5]):
            with self.assertRaisesRegex(ValueError, 'unique positive integers'):
                self.run_sweep(epochs=budgets)
        self.fixture.train.data[0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'pixels/labels/order'):
            self.run_sweep(epochs=[1])
        self.assertFalse((self.root / 'result' / 'epoch_sweep').exists())

    def test_entry_cell_runs_independently_and_keeps_main_results(self):
        script = Path(__file__).resolve().parents[1] / 'experiments/run-epoch-sweep.py'
        main_rows, main_config, main_models, main_stats = copy.deepcopy(self.rows), {}, {}, {}
        namespace = dict(EPOCH_SWEEP_SOURCE=self.manifest, UNLEARN_EPOCHS_TO_TEST=[1],
                         train_dataset=self.fixture.train, evaluation_dataset=self.fixture.test,
                         DEVICE='cpu', OUTPUT_ROOT=self.root,
                         rows=main_rows, config=main_config, unlearned_models=main_models, unlearn_stats=main_stats)
        with patch('IPython.display.display'), redirect_stdout(io.StringIO()), patch.object(training, 'train_from_scratch', side_effect=AssertionError('No training baselines')):
            exec(compile(script.read_text(), str(script), 'exec'), namespace)
        for key, expected in (('rows', main_rows), ('config', main_config), ('unlearned_models', main_models), ('unlearn_stats', main_stats)):
            self.assertIs(namespace[key], expected)
        self.assertEqual(namespace['sweep_frame'].shape[0], 2)
        with zipfile.ZipFile(namespace['sweep_archive']) as zipped:
            files = zipped.namelist()
            self.assertTrue(any('epoch_sweep/' in path and path.endswith('epochs.png') for path in files))
            self.assertTrue(any('epochs1_' in path and path.endswith(phases.MANIFEST_NAME) for path in files))
        chosen = phases.load_phase1_bundle(namespace['sweep_budget_archives'][1], device='cpu')
        self.assertEqual(chosen['config']['unlearn_epochs'], 1)


if __name__ == '__main__':
    unittest.main()
