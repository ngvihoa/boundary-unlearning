import copy
import csv
import json
import random
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from boundary_unlearning import diagnostics, training, unlearning


class TinyModel(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(4, classes))

    def forward(self, images):
        return self.classifier(self.features(images))


class BNAblationTests(unittest.TestCase):
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
        generator = torch.Generator().manual_seed(5)
        self.forget = TensorDataset(torch.rand(6, 3, 4, 4, generator=generator) * .5 + .4,
                                    torch.full((6,), 4))
        self.retain = TensorDataset(torch.rand(4, 3, 4, 4, generator=generator) - .5,
                                    torch.zeros(4, dtype=torch.long))

    def loader(self, dataset):
        return DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(123))

    def evaluate(self, model):
        return training.evaluate_model_splits(
            model, device='cpu', test_loader=self.loader(self.forget),
            test_retain_loader=self.loader(self.retain), test_forget_loader=self.loader(self.forget),
            train_retain_eval_loader=self.loader(self.retain), train_forget_eval_loader=self.loader(self.forget))

    def test_shrink_freezes_buffers_but_updates_weights_and_default_matches_update(self):
        for classes in (10, 100):
            original = TinyModel(classes).eval()
            state = copy.deepcopy(original.state_dict())
            common = dict(device='cpu', epochs=1, learning_rate=.01, bound=.01, normalization_mode='matched')
            default, _ = unlearning.run_boundary_shrink(original, self.loader(self.forget), self.root / 'default.pt', **common)
            update, _ = unlearning.run_boundary_shrink(original, self.loader(self.forget), self.root / 'update.pt', bn_stats_mode='update', **common)
            freeze, stats = unlearning.run_boundary_shrink(original, self.loader(self.forget), self.root / 'freeze.pt', bn_stats_mode='freeze', **common)
            for name, value in default.state_dict().items():
                torch.testing.assert_close(value, update.state_dict()[name], rtol=0, atol=0)
            for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
                torch.testing.assert_close(freeze.state_dict()[f'features.1.{suffix}'], state[f'features.1.{suffix}'], rtol=0, atol=0)
            self.assertEqual(update.features[1].num_batches_tracked.item(), 3)
            self.assertFalse(torch.equal(freeze.classifier[0].weight, original.classifier[0].weight))
            self.assertFalse(torch.equal(freeze.features[1].weight, original.features[1].weight))
            self.assertTrue(freeze.features[1].weight.requires_grad)
            self.assertEqual(stats['bn_stats_mode'], 'freeze')
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_distance_kl_freeze_and_loss_sample_logs(self):
        original = TinyModel().eval()
        state = copy.deepcopy(original.state_dict())
        samples = []
        for mode in ('update', 'freeze'):
            model, stats = unlearning.run_boundary_distance(
                original, self.loader(self.forget), self.root / f'{mode}.pt', device='cpu', epochs=1,
                method=unlearning.ORIGINAL_KL_METHOD, retain_loader=self.loader(self.retain),
                bn_stats_mode=mode, diagnostics_dir=self.root / mode, bound=.01, epsilon_step=.01)
            self.assertEqual(model.features[1].num_batches_tracked.item(), 0 if mode == 'freeze' else 3)
            if mode == 'freeze':
                torch.testing.assert_close(model.features[1].running_mean, state['features.1.running_mean'])
            with Path(stats['losses_path']).open() as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 3)
            for row in history:
                self.assertAlmostEqual(float(row['forget_ce']) + float(row['scaled_retain_kl']), float(row['total_loss']), places=5)
                self.assertAlmostEqual(.1 * float(row['retain_kl']), float(row['scaled_retain_kl']), places=6)
            with np.load(stats['boundary_samples_path']) as arrays:
                self.assertEqual(len(arrays['epsilon']), 6)
                samples.append({key: arrays[key].copy() for key in arrays.files})
        for key in samples[0]:
            np.testing.assert_array_equal(samples[0][key], samples[1][key])

    def test_paired_ablation_preserves_models_rng_and_exports_provenance(self):
        original, retrain = TinyModel(), TinyModel()
        original.features[1].eval()
        for parameter in original.parameters():
            parameter.grad = torch.ones_like(parameter)
        states = [copy.deepcopy(model.state_dict()) for model in (original, retrain)]
        modes = [[module.training for module in model.modules()] for model in (original, retrain)]
        torch_state = torch.get_rng_state().clone()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with patch.object(training, 'train_from_scratch', side_effect=AssertionError('Must not retrain baselines')):
            result = diagnostics.run_batchnorm_ablation(
                original, retrain, lambda: self.loader(self.forget), self.evaluate, self.root / 'result',
                device='cpu', source_run_tag='fixture100', epochs=1, bound=.01, epsilon_step=.01,
                max_batches_per_epoch=1, plots=True)
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(result['stats']['boundary_shrink_bn_update']['bn_batches_max_change'], 1)
        self.assertEqual(result['stats']['boundary_shrink_bn_freeze']['bn_running_mean_max_change'], 0)
        self.assertEqual(result['stats']['boundary_shrink_bn_freeze']['bn_running_var_max_change'], 0)
        self.assertEqual(result['audit']['samples'], 2)
        config = json.loads(result['config_path'].read_text())
        self.assertEqual(config['settings']['original_state_sha256'], diagnostics._fingerprint(original))
        self.assertEqual(config['settings']['epochs'], 1)
        for model, state, expected_modes in zip((original, retrain), states, modes):
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            self.assertEqual([module.training for module in model.modules()], expected_modes)
        for parameter in original.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        self.assertEqual(random.getstate(), python_state)
        for name in ('metrics.csv', 'config.json', 'results.html', 'losses.png', 'boundary_search_audit.pdf'):
            self.assertTrue((result['output_dir'] / name).is_file())

    def test_invalid_bn_mode_and_ablation_methods(self):
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, 'bn_stats_mode'):
            unlearning.run_boundary_shrink(model, self.loader(self.forget), self.root / 'unused.pt', device='cpu', bn_stats_mode='invalid')
        with self.assertRaisesRegex(ValueError, 'methods'):
            diagnostics.run_batchnorm_ablation(model, model, lambda: self.loader(self.forget), self.evaluate,
                                               self.root, device='cpu', source_run_tag='fixture',
                                               methods=(unlearning.BOUNDARY_KL_METHOD,))
        self.assertFalse((self.root / 'unused.pt').exists())

    def test_kaggle_entry_cell_runs_independently_and_preserves_main_results(self):
        repo = Path(__file__).resolve().parents[1]
        notebook = json.loads((repo / 'experiments/boundary-unlearning-cifar-100-phase2.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c.get('source', [])).startswith('# BN ABLATION:'))
        source = source.replace('RUN_BN_ABLATION = False', 'RUN_BN_ABLATION = True')
        main_rows = [{'model': 'original', 'test_all_acc': .5}]
        main_models, main_stats = {'old': object()}, {'old': {'seconds': 3}}
        checkpoint_dir = self.root / 'result' / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        namespace = dict(
            CODE_ROOT=repo, original_model=TinyModel(), retrain_model=TinyModel(),
            make_loader=lambda dataset, shuffle=False: self.loader(dataset),
            train_forget_set=self.forget, train_retain_set=self.retain,
            DEVICE=torch.device('cpu'), UNLEARN_EPOCHS=1, UNLEARN_LR=1e-5, MOMENTUM=.9,
            PIN_MEMORY=False, MAX_BATCHES_PER_EPOCH=1, FGSM_BOUND=.01, ATTACK_NORMALIZATION='matched',
            CIFAR_MEAN=(.5,) * 3, CIFAR_STD=(.5,) * 3,
            REPO_ATTACK_MEAN=(.5,) * 3, REPO_ATTACK_STD=(.5,) * 3,
            run_tag='fixture100', OUTPUT_ROOT=self.root, SEED=2022, CHECKPOINT_ROOT=checkpoint_dir,
            DATASET_NAME='CIFAR-100', rows=main_rows, unlearned_models=main_models, unlearn_stats=main_stats,
        )
        for name, dataset in [('test_loader', self.forget), ('test_retain_loader', self.retain),
                              ('test_forget_loader', self.forget), ('train_retain_eval_loader', self.retain),
                              ('train_forget_eval_loader', self.forget)]:
            namespace[name] = self.loader(dataset)
        with patch('IPython.display.display'), patch.object(training, 'train_from_scratch', side_effect=AssertionError('No baseline training')):
            exec(compile(source, '<BN notebook cell>', 'exec'), namespace)
        self.assertIs(namespace['rows'], main_rows)
        self.assertIs(namespace['unlearned_models'], main_models)
        self.assertIs(namespace['unlearn_stats'], main_stats)
        self.assertEqual(list(main_models), ['old'])
        with zipfile.ZipFile(namespace['bn_archive']) as zipped:
            names = zipped.namelist()
            self.assertTrue(any('bn_ablation/' in name and name.endswith('metrics.csv') for name in names))
            self.assertTrue(any(name.endswith('boundary_shrink_bn_freeze.pt') for name in names))
            self.assertTrue(any(name.endswith('boundary_search_audit.npz') for name in names))


if __name__ == '__main__':
    unittest.main()
