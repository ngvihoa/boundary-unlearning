"""Read the train/test/meta pickle files in alincijov/cifar-100."""

import pickle
from pathlib import Path

import numpy as np
import torch

from .constants import CIFAR_MEAN, CIFAR_STD


def resolve_data_root(root):
    """Accept either the dataset mount or a parent of cifar-100-python/."""
    root = Path(root)
    required = ("train", "test", "meta")
    if all((root / name).is_file() for name in required):
        return root
    matches = [path.parent for path in root.rglob("meta")
               if all((path.parent / name).is_file() for name in required)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one folder containing train, test and meta under {root}; "
            f"found {len(matches)}. Attach alincijov/cifar-100 and check DATA_ROOT."
        )
    return matches[0]


def _read_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def _field(record, name):
    if name in record:
        return record[name]
    if name.encode() in record:
        return record[name.encode()]
    raise ValueError(f"CIFAR-100 pickle is missing {name!r}")


class CIFAR100Dataset(torch.utils.data.Dataset):
    """CIFAR-100 fine labels and planar RGB pixels, normalized on access."""

    def __init__(self, images, labels, classes, *, mean=CIFAR_MEAN, std=CIFAR_STD):
        self.data = np.asarray(images)
        raw_labels = np.asarray(labels)
        self.classes = list(classes)
        if self.data.ndim != 2 or self.data.shape[1] != 3072 or self.data.dtype != np.uint8:
            raise ValueError("Expected CIFAR-100 uint8 image data with shape (N, 3072)")
        if raw_labels.ndim != 1 or len(raw_labels) != len(self.data) or len(raw_labels) == 0:
            raise ValueError("Expected one fine label per image in a non-empty split")
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError("CIFAR-100 fine labels must be integers")
        self.targets = raw_labels.astype(np.int64, copy=False)
        if len(self.classes) != 100 or len(set(self.classes)) != 100:
            raise ValueError("Expected 100 distinct fine_label_names in meta")
        if self.targets.min() < 0 or self.targets.max() >= 100:
            raise ValueError("CIFAR-100 fine labels must be between 0 and 99")
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Normalization needs finite means and positive finite standard deviations")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Official pickle rows store R, G and B planes consecutively.
        image = torch.from_numpy(self.data[index].reshape(3, 32, 32)).float() / 255.0
        return (image - self.mean) / self.std, int(self.targets[index])


def load_cifar100(root, *, mean=CIFAR_MEAN, std=CIFAR_STD):
    """Load both labeled splits, using fine_labels rather than coarse_labels."""
    root = resolve_data_root(root)
    meta = _read_pickle(root / "meta")
    classes = [name.decode("utf-8") if isinstance(name, bytes) else name
               for name in _field(meta, "fine_label_names")]
    splits = []
    for name in ("train", "test"):
        record = _read_pickle(root / name)
        splits.append(CIFAR100Dataset(
            _field(record, "data"), _field(record, "fine_labels"), classes,
            mean=mean, std=std,
        ))
    return tuple(splits)
