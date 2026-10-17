import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from boundary_unlearning.cifar100 import CIFAR100Dataset, load_cifar100, resolve_data_root
from boundary_unlearning.reporting import results_table


class CIFAR100Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.classes = [f"class_{i}" for i in range(100)]
        self.pixels = np.random.default_rng(2022).integers(0, 256, (3, 3072), dtype=np.uint8)

    def write_pickles(self, root, *, byte_keys=False):
        root.mkdir(parents=True, exist_ok=True)
        meta = {"fine_label_names": self.classes}
        split = {"data": self.pixels, "fine_labels": [0, 4, 99], "coarse_labels": [0, 1, 19]}
        for name, record in [("meta", meta), ("train", split), ("test", split)]:
            if byte_keys:
                record = {key.encode(): value for key, value in record.items()}
                if name == "meta":
                    record[b"fine_label_names"] = [value.encode() for value in self.classes]
            with (root / name).open("wb") as stream:
                pickle.dump(record, stream, protocol=2)

    def test_fine_labels_and_rgb_layout(self):
        self.write_pickles(self.root)
        train, test = load_cifar100(self.root)
        self.assertEqual(train.targets.tolist(), [0, 4, 99])
        self.assertEqual(test.classes, self.classes)
        expected = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])(self.pixels[1].reshape(3, 32, 32).transpose(1, 2, 0))
        image, label = train[1]
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertEqual(label, 4)

    def test_nested_folder_and_byte_keys(self):
        nested = self.root / "cifar-100-python"
        self.write_pickles(nested, byte_keys=True)
        self.assertEqual(resolve_data_root(self.root), nested)
        train, _ = load_cifar100(self.root)
        self.assertEqual(train.classes, self.classes)
        self.assertEqual(train.targets.tolist(), [0, 4, 99])

    def test_missing_or_ambiguous_dataset(self):
        with self.assertRaises(FileNotFoundError):
            resolve_data_root(self.root)
        self.write_pickles(self.root / "one")
        self.write_pickles(self.root / "two")
        with self.assertRaises(FileNotFoundError):
            resolve_data_root(self.root)

    def test_missing_fine_labels_does_not_use_coarse(self):
        self.write_pickles(self.root)
        with (self.root / "train").open("wb") as stream:
            pickle.dump({"data": self.pixels, "coarse_labels": [0, 1, 19]}, stream)
        with self.assertRaisesRegex(ValueError, "fine_labels"):
            load_cifar100(self.root)

    def test_invalid_shapes_labels_and_class_names(self):
        cases = [
            (self.pixels[:, :100], [0, 4, 99], self.classes),
            (self.pixels.astype(np.float32), [0, 4, 99], self.classes),
            (self.pixels, [0, 4], self.classes),
            (self.pixels, [0, 4, 100], self.classes),
            (self.pixels, [0.0, 4.0, 99.0], self.classes),
            (self.pixels, [0, 4, 99], self.classes[:20]),
            (self.pixels, [0, 4, 99], ["duplicate"] * 100),
        ]
        for images, labels, classes in cases:
            with self.subTest(labels=labels, shape=images.shape, classes=len(classes)):
                with self.assertRaises(ValueError):
                    CIFAR100Dataset(images, labels, classes)

    def test_caption_is_dataset_specific_with_cifar10_default(self):
        rows = [{"model": "original", "test_all_acc": 0.1234}]
        self.assertIn("Official CIFAR-100 test", results_table(rows, dataset_name="CIFAR-100").to_html())
        self.assertIn("Official CIFAR-10 test", results_table(rows).to_html())


if __name__ == "__main__":
    unittest.main()
