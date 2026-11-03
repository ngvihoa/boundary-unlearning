import copy
from contextlib import redirect_stdout
import io
import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from boundary_unlearning import phases, training
from boundary_unlearning.artifacts import export_results
from boundary_unlearning.cifar100 import CIFAR100Dataset
from boundary_unlearning.models import AllCNN


REPO = Path(__file__).resolve().parents[1]


class ExperimentPhaseTests(unittest.TestCase):
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
        self.root = Path(self.temp.name)
        self.classes = [f'class{i}' for i in range(100)]
        self.classes[4] = 'beaver'
        pixels = np.random.default_rng(2022).integers(0, 256, (100, 3072), dtype=np.uint8)
        self.train = CIFAR100Dataset(pixels, np.arange(100), self.classes)
        self.test = CIFAR100Dataset(pixels[::-1].copy(), np.arange(100), self.classes)

    def bundle(self):
        original = AllCNN(num_classes=100).eval()
        expanded = copy.deepcopy(original)
        model_set = {'original': original, 'retrain': copy.deepcopy(original),
                     'boundary_shrink': copy.deepcopy(original),
                     'boundary_expanding': nn.Sequential(expanded.features, expanded.classifier[0]).eval()}
        rows = [dict(model=name, test_all_acc=.5, test_retain_acc=.5, test_forget_acc=0.) for name in phases.PHASE1_MODELS]
        config = dict(experiment_phase=1, dataset='CIFAR-100', num_classes=100, label_type='fine',
                      methods=['boundary_shrink', 'boundary_expanding'], extra_exp=None, bn_stats_mode='update',
                      class_names=self.classes, forget_class=4, input_mean=[.5]*3, input_std=[.5]*3,
                      data_identity=phases.dataset_identity(self.train, self.test),
                      model_architecture=dict(n_channels=3, num_classes=100, filters_percentage=.5, batch_norm=True))
        destination = self.root / 'result' / 'phase1' / 'fixture'
        return model_set, rows, config, destination

    def test_roundtrip_exact_evaluated_states_zip_folder_and_pruned_layout(self):
        models, rows, config, destination = self.bundle()
        state = torch.get_rng_state().clone()
        manifest = phases.save_phase1_bundle(models, rows, config, destination, run_tag='fixture')
        torch.testing.assert_close(torch.get_rng_state(), state)
        archive = export_results(self.root / 'result')
        for source in (manifest, self.root / 'result', archive):
            restored = phases.load_phase1_bundle(source, device='cpu', copy_to=self.root / 'phase2' / 'result' / 'phase1_source')
            self.assertEqual(restored['rows'], rows)
            images = torch.zeros(2, 3, 32, 32)
            for name, original in models.items():
                actual = restored['models'][name]
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(original(images), actual(images), rtol=0, atol=0)
                self.assertFalse(actual.training)
            self.assertIsInstance(restored['models']['boundary_expanding'], nn.Sequential)
            self.assertTrue((restored['output_dir'] / phases.MANIFEST_NAME).is_file())
        torch.testing.assert_close(torch.get_rng_state(), state)

    def test_corrupted_checkpoint_and_mismatched_dataset_are_rejected(self):
        models, rows, config, destination = self.bundle()
        phases.save_phase1_bundle(models, rows, config, destination, run_tag='fixture')
        phases.validate_phase1_dataset(config, self.train, self.test)
        self.train.data[0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'pixels/labels/order'):
            phases.validate_phase1_dataset(config, self.train, self.test)
        self.train.data[0, 0] ^= 1
        self.test.classes = self.test.classes[::-1]
        with self.assertRaisesRegex(ValueError, 'class names/order'):
            phases.validate_phase1_dataset(config, self.train, self.test)
        (destination / 'checkpoints/original_evaluated.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            phases.load_phase1_bundle(destination, device='cpu')

    def test_missing_ambiguous_or_non_paper_source_fails_without_training(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Attach the Phase 1'):
            phases.find_phase1_source(self.root)
        models, rows, config, destination = self.bundle()
        config['bn_stats_mode'] = 'freeze'
        with self.assertRaisesRegex(ValueError, 'paper-method bundle'):
            phases.save_phase1_bundle(models, rows, config, destination, run_tag='fixture')
        config['bn_stats_mode'] = 'update'
        manifest = phases.save_phase1_bundle(models, rows, config, destination, run_tag='fixture')
        self.assertEqual(phases.find_phase1_source(self.root), manifest)
        export_results(self.root / 'result')
        # A saved Kaggle Output contains the extracted result and its ZIP.
        self.assertEqual(phases.find_phase1_source(self.root), manifest)
        phases.save_phase1_bundle(models, rows, config, self.root / 'other_run', run_tag='fixture2')
        with self.assertRaisesRegex(FileNotFoundError, 'found 2'):
            phases.find_phase1_source(self.root)

    def test_zip_reader_ignores_unlisted_files_instead_of_extracting_them(self):
        models, rows, config, destination = self.bundle()
        phases.save_phase1_bundle(models, rows, config, destination, run_tag='fixture')
        archive = export_results(self.root / 'result')
        with zipfile.ZipFile(archive, 'a') as zipped:
            zipped.writestr('../unlisted.txt', 'do not extract')
        phases.load_phase1_bundle(archive, device='cpu', copy_to=self.root / 'copy')
        self.assertFalse((self.root / 'unlisted.txt').exists())

    def test_notebooks_smoke_across_fresh_sessions_without_baseline_training_in_phase2(self):
        input_root = self.root / 'input'
        dataset_root = input_root / 'cifar100'
        dataset_root.mkdir(parents=True)
        for name, dataset in (('train', self.train), ('test', self.test)):
            with (dataset_root / name).open('wb') as stream:
                pickle.dump(dict(data=dataset.data, fine_labels=dataset.targets.tolist()), stream)
        with (dataset_root / 'meta').open('wb') as stream:
            pickle.dump(dict(fine_label_names=self.classes), stream)

        def execute(filename, output, source=None):
            notebook = json.loads((REPO / 'experiments' / filename).read_text())
            namespace = {}
            for index, item in enumerate(notebook['cells']):
                if item['cell_type'] != 'code':
                    continue
                code = ''.join(item['source'])
                # Execute every visualization cell, including restored/pruned models.
                # Bound inference/TSNE work without mocking any plotting function.
                code = code.replace('VIS_ENTROPY_SAMPLES = 2000', 'VIS_ENTROPY_SAMPLES = 12')
                code = code.replace('VIS_TSNE_SAMPLES = 1500', 'VIS_TSNE_SAMPLES = 100')
                code = code.replace('VIS_TSNE_ITERATIONS = 1000', 'VIS_TSNE_ITERATIONS = 250')
                code = code.replace('GITHUB_REPO = "https://github.com/ngvihoa/boundary-unlearning.git"', 'GITHUB_REPO = ""')
                code = code.replace('Path("/kaggle/input")', f'Path({str(input_root)!r})')
                code = code.replace('"/kaggle/input/datasets/alincijov/cifar-100"', repr(str(dataset_root)))
                code = code.replace('Path("/kaggle/working")', f'Path({str(output)!r})')
                if source is None:
                    code = code.replace('RUN_PROFILE = "paper"', 'RUN_PROFILE = "smoke"')
                    code = code.replace('EXPECTED_TRAIN_IMAGES = 50_000', 'EXPECTED_TRAIN_IMAGES = 100')
                    code = code.replace('EXPECTED_TEST_IMAGES = 10_000', 'EXPECTED_TEST_IMAGES = 100')
                    code = code.replace('BATCH_SIZE = 64', 'BATCH_SIZE = 8').replace('NUM_WORKERS = 2', 'NUM_WORKERS = 0')
                    code = code.replace('MAX_BATCHES_PER_EPOCH = 3', 'MAX_BATCHES_PER_EPOCH = 1')
                    code = code.replace('EVALUATE_TRAIN_SPLITS = False', 'EVALUATE_TRAIN_SPLITS = True')
                else:
                    code = code.replace(
                        'PHASE1_SOURCE = Path(CODE_ROOT) / "results" / "boundary-unlearning-cifar-100-result"',
                        f'PHASE1_SOURCE = Path({str(source)!r})',
                    )
                plotted_models = namespace.get('visualization_models', {}) if code.startswith('# VISUALIZATION:') else {}
                before = {
                    name: {
                        'state': copy.deepcopy(model.state_dict()),
                        'modes': [module.training for module in model.modules()],
                        'grads': [None if param.grad is None else param.grad.clone() for param in model.parameters()],
                        'hooks': sum(len(module._forward_hooks) for module in model.modules()),
                    }
                    for name, model in plotted_models.items()
                }
                exec(compile(code, f'{filename}:cell{index}', 'exec'), namespace)
                for name, model in plotted_models.items():
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[name]['state'][key], rtol=0, atol=0)
                    self.assertEqual([module.training for module in model.modules()], before[name]['modes'])
                    self.assertEqual(sum(len(module._forward_hooks) for module in model.modules()), before[name]['hooks'])
                    for param, grad in zip(model.parameters(), before[name]['grads']):
                        if grad is None:
                            self.assertIsNone(param.grad)
                        else:
                            torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
            return namespace

        with patch('IPython.display.display'), patch.object(plt, 'show'), redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            first = execute('boundary-unlearning-cifar-100-experiment.ipynb', self.root / 'phase1')
            self.assertEqual([row['model'] for row in first['rows']], list(phases.PHASE1_MODELS))
            self.assertEqual(first['config']['unlearning_stats']['boundary_shrink']['bn_stats_mode'], 'update')
            # Deliberately change the legacy best checkpoint: Phase 2 must ignore it.
            different = copy.deepcopy(first['original_model'])
            with torch.no_grad():
                different.classifier[0].bias.add_(100)
            training.save_state_dict(different, first['original_checkpoint'])
            archive = export_results(first['RESULT_ROOT'])
            attachment = input_root / 'phase1-output'
            attachment.mkdir()
            attached = attachment / 'result.zip'
            shutil.copy2(archive, attached)
            self.assertEqual(phases.find_phase1_source(input_root), attached)
            with patch.object(training, 'train_from_scratch', side_effect=AssertionError('Phase 2 must not train baselines')):
                second = execute('boundary-unlearning-cifar-100-phase2.ipynb', self.root / 'phase2', attached)
            self.assertEqual(len(second['rows']), 7)
            self.assertEqual(second['config']['experiment_phase'], 2)
            for name, model in first['models_to_evaluate'].items():
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, second['models_to_evaluate'][name].state_dict()[key], rtol=0, atol=0)
            for before, after in zip(first['rows'], second['rows'][:4]):
                for key in ('test_all_acc', 'test_retain_acc', 'test_forget_acc', 'train_retain_acc', 'train_forget_acc'):
                    self.assertEqual(before[key], after[key])
            with self.assertRaisesRegex(ValueError, 'baselines are restored and fixed'):
                second['run_method']('boundary_shrink')
            with zipfile.ZipFile(second['archive_path']) as zipped:
                self.assertTrue(any(name.endswith(phases.MANIFEST_NAME) for name in zipped.namelist()))
                self.assertTrue(any('phase2_' in name and name.endswith('_metrics.csv') for name in zipped.namelist()))
            for session, expected_count in ((first, 4), (second, 7)):
                folder = session['FIGURE_ROOT']
                expected_models = list(session['models_to_evaluate'])
                for name in ('accuracy', 'cost', 'entropy', 'tsne'):
                    metadata = json.loads((folder / f'{name}.json').read_text())
                    if name in ('entropy', 'tsne'):
                        self.assertEqual(metadata['models'], expected_models)
                    for extension in ('png', 'pdf', 'json'):
                        self.assertTrue((folder / f'{name}.{extension}').is_file())
                cam_models = []
                for page in range(1, (expected_count + 3) // 4 + 1):
                    for extension in ('png', 'pdf', 'json', 'npz'):
                        self.assertTrue((folder / f'gradcam_{page}.{extension}').is_file())
                    metadata = json.loads((folder / f'gradcam_{page}.json').read_text())
                    cam_models.extend(metadata['models'])
                    self.assertEqual(metadata['target_class'], 4)
                    with np.load(folder / f'gradcam_{page}.npz') as arrays:
                        np.testing.assert_array_equal(arrays['indices'], session['gradcam_indices'])
                        for name in metadata['models']:
                            self.assertEqual(arrays[f'{name}_sample_0_cam'].shape, (32, 32))
                            self.assertEqual(arrays[f'{name}_sample_0_probabilities'].shape, (100,))
                self.assertEqual(cam_models, expected_models)
                with np.load(folder / 'tsne.npz') as arrays:
                    np.testing.assert_array_equal(arrays['indices'], session['tsne_indices'])
                    for name in expected_models:
                        self.assertEqual(arrays[f'{name}_embedding'].shape, (100, 2))
                        self.assertTrue(np.isfinite(arrays[f'{name}_embedding']).all())
                with np.load(folder / 'entropy.npz') as arrays:
                    for split, indices in session['entropy_indices'].items():
                        np.testing.assert_array_equal(arrays[f'{split}_indices'], indices)
                        for name in expected_models:
                            values = arrays[f'{name}_{split}_log_entropy']
                            self.assertEqual(len(values), len(indices))
                            self.assertTrue(np.isfinite(values).all())
                with zipfile.ZipFile(session['archive_path']) as zipped:
                    for path in folder.iterdir():
                        self.assertIn(str(path.relative_to(session['OUTPUT_ROOT'])), zipped.namelist())
            # Export still works after models have been released; it does not retrain/resnapshot them.
            del first['original_model'], first['retrain_model'], first['models_to_evaluate'], first['unlearned_models']
            notebook = json.loads((REPO / 'experiments/boundary-unlearning-cifar-100-experiment.ipynb').read_text())
            export_cell = ''.join(notebook['cells'][-1]['source'])
            with patch.object(training, 'train_from_scratch', side_effect=AssertionError('No training during export')):
                exec(compile(export_cell, '<independent Phase 1 export>', 'exec'), first)


if __name__ == '__main__':
    unittest.main()
