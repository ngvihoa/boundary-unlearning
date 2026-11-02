import copy
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset
from threadpoolctl import threadpool_limits

from boundary_unlearning import visualization as vis
from boundary_unlearning.artifacts import export_results
from boundary_unlearning.models import AllCNN
from boundary_unlearning.unlearning import SUPPORTED_METHODS


class TinyModel(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 6, 1), nn.BatchNorm2d(6), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(6, classes))

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class VisualizationTests(unittest.TestCase):
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
        self.addCleanup(plt.close, 'all')
        self.root = Path(self.temp.name) / 'result' / 'figures' / 'fixture'
        generator = torch.Generator().manual_seed(5)
        self.dataset = TensorDataset(torch.rand(30, 3, 8, 8, generator=generator) * 2 - 1,
                                     torch.tensor([0, 4, 9] * 10))

    def test_sampling_is_balanced_bounded_reproducible_without_global_rng(self):
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        selected = vis.select_sample_indices(self.dataset, max_samples=11, seed=123)
        np.testing.assert_array_equal(selected, vis.select_sample_indices(self.dataset, max_samples=11, seed=123))
        self.assertFalse(np.array_equal(selected, vis.select_sample_indices(self.dataset, max_samples=11, seed=124)))
        labels = vis.dataset_labels(self.dataset)[selected]
        counts = np.unique(labels, return_counts=True)[1]
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(len(selected), 11)
        capped = vis.select_sample_indices(self.dataset, max_samples=100, per_class=2)
        self.assertEqual(len(capped), 6)
        nested = Subset(Subset(self.dataset, [9, 8, 7, 6]), [3, 0])
        np.testing.assert_array_equal(vis.dataset_labels(nested), [0, 0])
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        with self.assertRaises(ValueError):
            vis.select_sample_indices(self.dataset, max_samples=0)

    def test_gradcam_preserves_allcnn_state_gradients_modes_and_hooks(self):
        for classes in (10, 100):
            allcnn = AllCNN(num_classes=classes)
            for model in (allcnn, nn.Sequential(copy.deepcopy(allcnn.features), copy.deepcopy(allcnn.classifier[0]))):
                model.train()
                list(model.modules())[1].eval()  # Mixed train/eval modes must survive.
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                state = copy.deepcopy(model.state_dict())
                modes = [module.training for module in model.modules()]
                hooks_before = sum(len(module._forward_hooks) for module in model.modules())
                cam, probabilities, raw_max = vis.gradcam_map(model, torch.zeros(3, 32, 32), 4, device='cpu')
                self.assertEqual(cam.shape, (32, 32))
                self.assertEqual(len(probabilities), classes)
                self.assertTrue(np.isfinite(cam).all())
                self.assertGreaterEqual(raw_max, 0)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
                self.assertEqual([module.training for module in model.modules()], modes)
                for parameter in model.parameters():
                    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
                self.assertEqual(sum(len(module._forward_hooks) for module in model.modules()), hooks_before)
                with self.assertRaises(IndexError):
                    vis.gradcam_map(model, torch.zeros(3, 32, 32), classes, device='cpu')
                self.assertEqual(sum(len(module._forward_hooks) for module in model.modules()), hooks_before)

    def test_gradcam_pages_contain_same_samples_and_target_probabilities(self):
        models = {name: TinyModel() for name in ('original', 'retrain', *SUPPORTED_METHODS)}
        figures = vis.plot_gradcam_comparison(
            models, self.dataset, [1, 4], device='cpu', class_names=[f'c{i}' for i in range(10)],
            forget_class=4, output_dir=self.root, models_per_page=4)
        self.assertEqual(len(figures), 2)
        self.assertTrue(all('P(c4)' in ax.get_xlabel() for ax in figures[0].axes if ax.get_xlabel().startswith('Pred:')))
        for page in (1, 2):
            metadata = json.loads((self.root / f'gradcam_{page}.json').read_text())
            self.assertEqual(metadata['indices'], [1, 4])
            self.assertEqual(metadata['target_class'], 4)
            self.assertTrue((self.root / f'gradcam_{page}.pdf').is_file())

    def test_entropy_matches_analytic_uniform_entropy_and_exact_sample_limits(self):
        model = TinyModel()
        with torch.no_grad():
            model.classifier[0].weight.zero_()
            model.classifier[0].bias.zero_()
        model.train()
        state = copy.deepcopy(model.state_dict())
        indices = vis.select_sample_indices(self.dataset, max_samples=7)
        vis.plot_entropy_comparison({'original': model}, {'test': self.dataset}, {'test': indices},
                                    device='cpu', output_dir=self.root, batch_size=4)
        arrays = np.load(self.root / 'entropy.npz')
        values = arrays['original_test_log_entropy']
        self.assertEqual(len(values), 7)  # Not rounded up to the next batch.
        np.testing.assert_allclose(values, np.log(np.log(10)), atol=1e-6)
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name])

    def test_tsne_handles_pruned_model_100_classes_and_small_samples(self):
        model = TinyModel(100)
        pruned = nn.Sequential(copy.deepcopy(model.features), copy.deepcopy(model.classifier[0]))
        indices = vis.select_sample_indices(self.dataset, max_samples=12)
        with threadpool_limits(limits=1):
            figure = vis.plot_tsne_comparison(
                {'original': model, 'boundary_expanding': pruned}, self.dataset, indices,
                device='cpu', class_names=[f'class{i}' for i in range(100)], forget_class=4,
                output_dir=self.root, dataset_name='CIFAR-100', max_iter=250)
        metadata = json.loads((self.root / 'tsne.json').read_text())
        self.assertEqual(metadata['perplexity'], 11)
        self.assertEqual(metadata['retain_display'], 'one_gray_group')
        arrays = np.load(self.root / 'tsne.npz')
        np.testing.assert_array_equal(arrays['indices'], indices)
        self.assertEqual(arrays['boundary_expanding_embedding'].shape, (12, 2))
        self.assertEqual(figure.axes[0].get_legend_handles_labels()[1], ['Retain classes', 'Forget: class4'])
        with self.assertRaises(ValueError):
            vis.plot_tsne_comparison({'original': model}, self.dataset, [0, 1], device='cpu',
                                     class_names=['unused'] * 100, forget_class=4, output_dir=self.root)

    def test_metric_units_cost_totals_and_recursive_zip_export(self):
        rows = [
            {'model': 'original', 'test_retain_acc': .9, 'test_forget_acc': .8},
            {'model': 'retrain', 'test_retain_acc': .85, 'test_forget_acc': 0, 'training_seconds': 100},
            {'model': SUPPORTED_METHODS[-1], 'test_retain_acc': .8, 'test_forget_acc': .01,
             'unlearning_seconds': 2, 'total_unlearning_seconds': 9,
             'unlearning_update_steps': 10, 'student_update_steps': 20, 'total_update_steps': 30},
        ]
        figure = vis.plot_metric_comparison(rows, output_dir=self.root)
        np.testing.assert_allclose([bar.get_width() for bar in figure.axes[0].containers[0]], [5, 0, -5])
        figure = vis.plot_cost_comparison(rows, output_dir=self.root)
        self.assertEqual(figure.axes[0].containers[0][0].get_width(), 2)
        self.assertEqual(figure.axes[0].containers[1][0].get_width(), 9)
        self.assertEqual(figure.axes[1].containers[1][0].get_width(), 20)
        archive = export_results(self.root.parents[1])
        import zipfile
        with zipfile.ZipFile(archive) as zipped:
            self.assertIn('result/figures/fixture/accuracy.png', zipped.namelist())
            self.assertIn('result/figures/fixture/cost.pdf', zipped.namelist())
            self.assertIn('result/figures/fixture/cost.json', zipped.namelist())

    def test_summary_notebook_cells_run_without_model_objects_or_setup(self):
        rows = [{'model': 'retrain', 'test_retain_acc': .8, 'test_forget_acc': 0},
                {'model': 'boundary_shrink', 'test_retain_acc': .7, 'test_forget_acc': .1,
                 'unlearning_seconds': 1, 'total_unlearning_seconds': 1,
                 'unlearning_update_steps': 2, 'student_update_steps': 2, 'total_update_steps': 2}]
        repo = Path(__file__).resolve().parents[1]
        for path in (repo / 'experiments').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            source = next(''.join(cell['source']) for cell in notebook['cells']
                          if ''.join(cell['source']).startswith('# VISUALIZATION: accuracy and cost.'))
            # Deliberately no datasets, model objects, or variables from visualization setup.
            namespace = {'rows': rows, 'OUTPUT_ROOT': self.root,
                         'run_tag': path.stem, 'NUM_CLASSES': 100 if '100' in path.name else 10}
            from unittest.mock import patch
            with patch.object(plt, 'show'):
                exec(compile(source, str(path), 'exec'), namespace)
            self.assertTrue((namespace['FIGURE_ROOT'] / 'accuracy.png').is_file())
            self.assertTrue((namespace['FIGURE_ROOT'] / 'cost.json').is_file())


if __name__ == '__main__':
    unittest.main()
