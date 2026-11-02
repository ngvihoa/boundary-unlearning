import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from boundary_unlearning.attacks import find_boundary_epsilon
from boundary_unlearning.enhanced import retain_distillation_loss, weight_assign_from_epsilon
from boundary_unlearning.unlearning import (
    BOUNDARY_KL_METHOD, DISTANCE_METHOD, ORIGINAL_KL_METHOD,
    run_boundary_distance, validate_methods,
)
from boundary_unlearning.reporting import results_table, unlearning_details_table


class PixelTeacher(nn.Module):
    def forward(self, inputs):
        # Input normalization: mean=.25, std=.5; boundary at pixel value .5.
        margin = inputs[:, 0, 0, 0] * .5 + .25 - .5
        return torch.stack([margin, -margin], dim=1)


class TinyModel(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4),
                                      nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(4, classes)

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class DistanceUnlearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_first_cross_already_wrong_and_no_cross(self):
        pixels = torch.tensor([.6, .4, .95]).view(-1, 1, 1, 1).expand(-1, 3, 2, 2)
        inputs = (pixels - .25) / .5
        targets, epsilons = find_boundary_epsilon(
            inputs, torch.zeros(3, dtype=torch.long), PixelTeacher(),
            bound=.2, epsilon_step=.03, cifar_mean=(.25,) * 3, cifar_std=(.5,) * 3,
        )
        torch.testing.assert_close(epsilons, torch.tensor([.12, 0, .2]))
        self.assertEqual(targets.tolist(), [1, 1, 0])
        self.assertIsNone(inputs.grad)
        # An all-wrong batch takes the early exit without a candidate forward.
        targets, epsilons = find_boundary_epsilon(
            inputs[1:2], torch.tensor([0]), PixelTeacher(),
            cifar_mean=(.25,) * 3, cifar_std=(.5,) * 3,
        )
        self.assertEqual(targets.item(), 1)
        self.assertEqual(epsilons.item(), 0)

    def test_final_non_multiple_grid_point_and_invalid_bounds(self):
        inputs = torch.full((1, 3, 2, 2), (.6 - .25) / .5)
        targets, epsilons = find_boundary_epsilon(
            inputs, torch.tensor([0]), PixelTeacher(), bound=.105, epsilon_step=.03,
            cifar_mean=(.25,) * 3, cifar_std=(.5,) * 3,
        )
        self.assertEqual(targets.item(), 1)
        self.assertAlmostEqual(epsilons.item(), .105, places=6)
        for settings in ({"bound": 0}, {"epsilon_step": -1}, {"bound": float('nan')}):
            with self.assertRaises(ValueError):
                find_boundary_epsilon(inputs, torch.tensor([0]), PixelTeacher(), **settings)

    def test_distance_weights_are_normalized_detached_and_can_exceed_one(self):
        epsilons = torch.tensor([0., .05, .1], requires_grad=True)
        weights = weight_assign_from_epsilon(epsilons, .1)
        torch.testing.assert_close(weights, torch.tensor([1., .5, .1]) / (1.6 / 3))
        self.assertAlmostEqual(weights.mean().item(), 1, places=6)
        self.assertGreater(weights[0].item(), 1)
        self.assertFalse(weights.requires_grad)

    def test_kl_direction_temperature_gradients_and_batchnorm(self):
        torch.manual_seed(2)
        student, teacher = TinyModel(), TinyModel().eval()
        images = torch.randn(3, 3, 4, 4)
        student.train()
        student(images)  # Forget forward updates BN once.
        bn = student.features[1]
        mean_before = bn.running_mean.clone()
        count_before = bn.num_batches_tracked.clone()
        reference = copy.deepcopy(student).eval()
        temperature = 2.0
        with torch.no_grad():
            logits = reference(images) / temperature
            probabilities = (teacher(images) / temperature).softmax(1)
            expected = (probabilities * (probabilities.log() - logits.log_softmax(1))).sum() / len(images) * temperature ** 2
        loss = retain_distillation_loss(student, teacher, images, temperature=temperature)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bn.training)
        torch.testing.assert_close(bn.running_mean, mean_before)
        torch.testing.assert_close(bn.num_batches_tracked, count_before)
        self.assertIsNotNone(student.classifier.weight.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def loaders(self):
        images = torch.randn(6, 3, 4, 4).clamp(-1, 1)
        forget = DataLoader(TensorDataset(images, torch.full((6,), 4)), batch_size=2)
        retain = DataLoader(TensorDataset(images[:2], torch.zeros(2, dtype=torch.long)), batch_size=2)
        return forget, retain

    def test_three_variants_on_10_and_100_classes_costs_and_checkpoints(self):
        for classes in (10, 100):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(3)
                original = TinyModel(classes).eval()
                original_state = copy.deepcopy(original.state_dict())
                forget, retain = self.loaders()
                common = dict(device='cpu', epochs=1, bound=.01, epsilon_step=.01,
                              learning_rate=.01, retain_loader=retain)
                weighted, weighted_stats = run_boundary_distance(
                    original, forget, Path(folder) / 'weighted.pt', **common)
                distilled, original_stats = run_boundary_distance(
                    original, forget, Path(folder) / 'original_kl.pt', method=ORIGINAL_KL_METHOD, **common)
                boundary_teacher = copy.deepcopy(weighted).eval()
                boundary, boundary_stats = run_boundary_distance(
                    original, forget, Path(folder) / 'boundary_kl.pt', method=BOUNDARY_KL_METHOD,
                    weight_assign_model=weighted, boundary_model=boundary_teacher,
                    dependency_stats={DISTANCE_METHOD: weighted_stats,
                                      'boundary_shrink': {'seconds': 2., 'update_steps': 7}}, **common)
                self.assertEqual(weighted_stats['update_steps'], 3)
                self.assertEqual(original_stats['student_update_steps'], 3)
                self.assertEqual(boundary_stats['student_update_steps'], 6)
                self.assertEqual(boundary_stats['total_update_steps'], 13)
                self.assertAlmostEqual(boundary_stats['total_seconds'],
                                       boundary_stats['seconds'] + weighted_stats['seconds'] + 2.)
                # BN runs only for forget data, even while cycling a single retain batch.
                self.assertEqual(distilled.features[1].num_batches_tracked.item(), 3)
                self.assertEqual(boundary.features[1].num_batches_tracked.item(), 6)
                self.assertEqual(boundary(torch.randn(2, 3, 4, 4)).shape, (2, classes))
                for name, tensor in original.state_dict().items():
                    torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in original.parameters()))
                state = torch.load(Path(folder) / 'boundary_kl.pt', weights_only=True)
                torch.testing.assert_close(state['classifier.weight'], boundary.classifier.weight)
                json.dumps(boundary_stats)  # Exportable without tensor serialization hacks.

    def test_initialization_and_batch_limit(self):
        forget, retain = self.loaders()
        original, weighted, boundary = TinyModel(), TinyModel(), TinyModel()
        with tempfile.TemporaryDirectory() as folder:
            for method in (DISTANCE_METHOD, ORIGINAL_KL_METHOD, BOUNDARY_KL_METHOD):
                model, stats = run_boundary_distance(
                    original, forget, Path(folder) / f'{method}.pt', device='cpu', method=method,
                    epochs=0, retain_loader=retain, weight_assign_model=weighted, boundary_model=boundary)
                source = weighted if method == BOUNDARY_KL_METHOD else original
                torch.testing.assert_close(model.classifier.weight, source.classifier.weight)
                self.assertEqual(stats['update_steps'], 0)
            _, stats = run_boundary_distance(original, forget, Path(folder) / 'limited.pt',
                                            device='cpu', epochs=2, max_batches_per_epoch=1,
                                            bound=.01, epsilon_step=.01)
            self.assertEqual(stats['update_steps'], 2)
            self.assertEqual(stats['sample_seen'], 4)

    def test_dependency_validation(self):
        with self.assertRaisesRegex(ValueError, 'needs'):
            validate_methods([BOUNDARY_KL_METHOD])
        with self.assertRaisesRegex(ValueError, 'plain'):
            validate_methods(['boundary_shrink', DISTANCE_METHOD, BOUNDARY_KL_METHOD], extra_exp='curv')
        forget, _ = self.loaders()
        with self.assertRaisesRegex(ValueError, 'retain_loader'):
            run_boundary_distance(TinyModel(), forget, 'unused.pt', device='cpu', method=ORIGINAL_KL_METHOD)

    def test_tables_show_variant_names_and_distinguish_total_cost(self):
        rows = [{'model': BOUNDARY_KL_METHOD, 'test_retain_acc': .7,
                 'unlearning_seconds': 3., 'total_unlearning_seconds': 9.,
                 'unlearning_update_steps': 10, 'student_update_steps': 20,
                 'total_update_steps': 30, 'initializer': DISTANCE_METHOD,
                 'retain_teacher': 'boundary_shrink'}]
        self.assertIn('Distance + KL (Shrink, warm start)', results_table(rows).to_html())
        html = unlearning_details_table(rows).to_html()
        self.assertIn('With prerequisites (s)', html)
        self.assertIn('9.00', html)
        self.assertIn('Shrink + distance weights', html)


if __name__ == '__main__':
    unittest.main()
