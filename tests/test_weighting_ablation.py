import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from boundary_unlearning import diagnostics, training, unlearning
from test_bn_ablation import TinyModel


class WeightingAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(8)
        self.dataset = TensorDataset(torch.rand(6, 3, 4, 4, generator=generator), torch.full((6,), 4))
        self.original, self.retrain = TinyModel().eval(), TinyModel().eval()

    def loader(self, dataset=None):
        return DataLoader(self.dataset if dataset is None else dataset, batch_size=2, shuffle=True,
                          generator=torch.Generator().manual_seed(123))

    def evaluate(self, model):
        accuracy = training.accuracy(model, self.loader(), device='cpu')
        return dict(test_all_acc=accuracy, test_retain_acc=accuracy, test_forget_acc=accuracy,
                    train_retain_acc=accuracy, train_forget_acc=accuracy)

    def run_pair(self, **options):
        return diagnostics.run_weighting_ablation(
            self.original, self.retrain, self.loader, self.evaluate, self.root / 'result',
            device='cpu', source_run_tag='fixture100', epochs=1, learning_rate=.01,
            bound=.01, epsilon_step=.01, **options)

    def test_pair_preserves_targets_baselines_bn_and_exports_comparison(self):
        initial = [copy.deepcopy(model.state_dict()) for model in (self.original, self.retrain)]
        with patch.object(training, 'train_from_scratch', side_effect=AssertionError('No baseline training')):
            result = self.run_pair(plots=True)
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(len(result['models']), 2)
        self.assertTrue(all(result['pair_validation'].values()))
        self.assertIsNone(result['audit'])
        for name, stats in result['stats'].items():
            self.assertEqual(stats['update_steps'], 3)
            self.assertEqual(stats['bn_stats_mode'], 'freeze')
            self.assertEqual(stats['bn_running_mean_max_change'], 0)
            self.assertEqual(stats['bn_running_var_max_change'], 0)
            self.assertEqual(stats['bn_batches_max_change'], 0)
            self.assertEqual(stats['mean_retain_kl'], 0)
            self.assertFalse(torch.equal(result['models'][name].classifier[0].weight,
                                         self.original.classifier[0].weight))
        for model, state in zip((self.original, self.retrain), initial):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        config = json.loads(result['config_path'].read_text())
        self.assertEqual(config['settings']['weighting_modes'], ['distance', 'uniform'])
        self.assertEqual(config['settings']['kl_lambda'], 0)
        self.assertEqual(config['settings']['modes'], ['freeze'])
        rows = result['rows']
        delta = result['comparison'][0]['test_retain_delta_pp']
        self.assertAlmostEqual(delta, 100 * (rows[3]['test_retain_acc'] - rows[2]['test_retain_acc']))
        for name in ('metrics.csv', 'results.html', 'pair_validation.json', 'weighting_comparison.csv',
                     'applied_weights.png', 'applied_weights.pdf', 'losses.png'):
            self.assertTrue((result['output_dir'] / name).is_file(), name)
        uniform = result['applied_weights'][f'{unlearning.DISTANCE_METHOD}_uniform_bn_freeze']
        self.assertEqual(uniform, dict(min=1., max=1., mean=1., std=0.))

    def test_uniform_control_is_exactly_identical_if_distance_weights_are_one(self):
        with patch.object(unlearning, 'weight_assign_from_epsilon', side_effect=lambda eps, bound: torch.ones_like(eps)):
            result = self.run_pair(plots=False)
        distance, uniform = result['models'].values()
        for key, value in distance.state_dict().items():
            torch.testing.assert_close(value, uniform.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(result['comparison'][0]['mean_forget_ce_delta'], 0.)

    def test_uniform_ce_is_unweighted_and_distance_default_is_unchanged(self):
        def targets(images, labels, teacher, **options):
            return torch.zeros_like(labels), torch.linspace(0, .01, len(labels))

        common = dict(device='cpu', epochs=1, learning_rate=.01, bn_stats_mode='freeze',
                      bound=.01, epsilon_step=.01)
        with patch.object(unlearning, 'find_boundary_epsilon', side_effect=targets):
            default, _ = unlearning.run_boundary_distance(self.original, self.loader(), self.root / 'default.pt', **common)
            explicit, _ = unlearning.run_boundary_distance(self.original, self.loader(), self.root / 'explicit.pt',
                                                          weighting_mode='distance', **common)
            common['learning_rate'] = 0
            _, stats = unlearning.run_boundary_distance(self.original, self.loader(), self.root / 'uniform.pt',
                                                       weighting_mode='uniform', diagnostics_dir=self.root / 'uniform', **common)
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            expected = torch.nn.functional.cross_entropy(self.original(self.dataset.tensors[0]), torch.zeros(6, dtype=torch.long))
        self.assertAlmostEqual(stats['mean_forget_ce'], expected.item(), places=5)
        with np.load(stats['boundary_samples_path']) as arrays:
            np.testing.assert_array_equal(arrays['weight'], np.ones(6))
        with self.assertRaisesRegex(ValueError, 'weighting_mode'):
            unlearning.run_boundary_distance(self.original, self.loader(), self.root / 'invalid.pt',
                                             weighting_mode='invalid', **common)
        self.assertFalse((self.root / 'invalid.pt').exists())

    def test_mismatched_loaders_reject_comparison(self):
        calls = 0

        def mismatched_factory():
            nonlocal calls
            calls += 1
            return self.loader(TensorDataset(self.dataset.tensors[0], torch.full((6,), calls)))

        with self.assertRaisesRegex(RuntimeError, 'Invalid weighting pair'):
            diagnostics.run_weighting_ablation(
                self.original, self.retrain, mismatched_factory, self.evaluate, self.root,
                device='cpu', source_run_tag='invalid_pair', epochs=1, bound=.01, epsilon_step=.01, plots=False)

    def test_entry_runs_separately_preserves_existing_results_and_exports_zip(self):
        script = Path(__file__).resolve().parents[1] / 'experiments/run-weighting-ablation.py'
        main_rows, main_models, main_stats, bn_result = [], {}, {}, {'existing': True}
        namespace = dict(
            original_model=self.original, retrain_model=self.retrain,
            make_loader=lambda dataset, shuffle=False: self.loader(dataset), train_forget_set=self.dataset,
            DEVICE='cpu', UNLEARN_EPOCHS=1, UNLEARN_LR=1e-5, MOMENTUM=.9,
            PIN_MEMORY=False, MAX_BATCHES_PER_EPOCH=1, FGSM_BOUND=.01, WEIGHT_EPSILON_STEP=.01,
            CIFAR_MEAN=(.5,) * 3, CIFAR_STD=(.5,) * 3, REPO_ATTACK_MEAN=(.5,) * 3, REPO_ATTACK_STD=(.5,) * 3,
            run_tag='fixture100', OUTPUT_ROOT=self.root, SEED=2022, DATASET_NAME='CIFAR-100',
            rows=main_rows, unlearned_models=main_models, unlearn_stats=main_stats, bn_ablation=bn_result,
        )
        for key in ('test_loader', 'test_retain_loader', 'test_forget_loader',
                    'train_retain_eval_loader', 'train_forget_eval_loader'):
            namespace[key] = self.loader()
        with patch('IPython.display.display'), patch.object(training, 'train_from_scratch', side_effect=AssertionError('No baseline training')):
            exec(compile(script.read_text(), str(script), 'exec'), namespace)
        for key, expected in (('rows', main_rows), ('unlearned_models', main_models),
                              ('unlearn_stats', main_stats), ('bn_ablation', bn_result)):
            self.assertIs(namespace[key], expected)
        with zipfile.ZipFile(namespace['weight_archive']) as zipped:
            names = zipped.namelist()
            self.assertTrue(any('weighting_ablation/' in name and name.endswith('pair_validation.json') for name in names))
            self.assertTrue(any(name.endswith('_uniform_bn_freeze.pt') for name in names))
            self.assertTrue(any(name.endswith('weighting_comparison.csv') for name in names))


if __name__ == '__main__':
    unittest.main()
