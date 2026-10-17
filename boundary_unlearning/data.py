"""Kaggle image records, labels, datasets and deterministic loaders."""

import csv
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import CLASS_TO_INDEX


def load_concatenated_npy_images(path):
    path = Path(path)
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version: {version}")
        header_size = stream.tell()
        stream.seek(0)
        first_header = stream.read(header_size)

    if dtype != np.dtype(np.uint8) or fortran_order:
        raise ValueError(f"Expected C-order uint8 images, got dtype={dtype}, fortran_order={fortran_order}")
    if shape not in {(32, 32, 3), (3, 32, 32), (3072,)}:
        raise ValueError(f"Unsupported image record shape: {shape}")

    image_size = int(np.prod(shape)) * dtype.itemsize
    record_size = header_size + image_size
    file_size = path.stat().st_size
    if file_size % record_size != 0:
        raise ValueError(f"Unexpected file size: {file_size}")
    num_images = file_size // record_size
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    for i in sorted({0, min(1, num_images - 1), num_images - 1}):
        offset = i * record_size
        if bytes(raw[offset:offset + header_size]) != first_header:
            raise ValueError(f"Inconsistent NPY header at record {i}")

    # Preserve the image layout declared in each record (HWC for this dataset).
    image_strides = np.empty(shape, dtype=dtype).strides
    images = np.ndarray(
        shape=(num_images, *shape), dtype=dtype, buffer=raw,
        offset=header_size, strides=(record_size, *image_strides),
    )
    return images, raw


def read_csv_rows(path):
    path = Path(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {path}"
            )

        columns = {
            name.strip(): name
            for name in reader.fieldnames
            if name is not None
        }

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    return rows, columns


def required_column(columns, name, path):
    if name not in columns:
        raise ValueError(
            f"Required column '{name}' not found in {path}. "
            f"Available columns: {list(columns.keys())}"
        )

    return columns[name]


def encode_cifar_labels(raw_labels, num_classes=10):
    labels_as_text = [
        str(value).strip().lower()
        for value in raw_labels
    ]

    numeric_values = []
    all_numeric = True

    for value in labels_as_text:
        try:
            numeric_values.append(int(value))
        except (ValueError, TypeError):
            all_numeric = False
            break

    if all_numeric:
        unique = set(numeric_values)

        # Labels from 0 to 9
        if unique.issubset(set(range(num_classes))):
            return np.asarray(numeric_values, dtype=np.int64)

        # Labels from 1 to 10
        if unique.issubset(set(range(1, num_classes + 1))):
            return np.asarray(
                numeric_values, dtype=np.int64
            ) - 1

        raise ValueError(
            f"Invalid numeric CIFAR-10 labels: {sorted(unique)}"
        )

    unknown = sorted(
        set(labels_as_text) - set(CLASS_TO_INDEX.keys())
    )

    if unknown:
        raise ValueError(
            f"Unknown CIFAR-10 class labels: {unknown}"
        )

    return np.asarray(
        [CLASS_TO_INDEX[value] for value in labels_as_text],
        dtype=np.int64
    )


def parse_numeric_id(value):
    text = str(value).strip()
    match = re.search(r"\d+", text)

    return int(match.group()) if match else None


def validate_image_array(array, name):
    if array.ndim == 2 and array.shape[1] == 3072:
        return
    if array.ndim == 4 and array.shape[1:] in {(32, 32, 3), (3, 32, 32)}:
        return
    raise ValueError(f"{name} has unsupported shape {array.shape}. Expected (N, 3072), (N, 32, 32, 3), or (N, 3, 32, 32).")


class CIFAR10NpyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.images) != len(self.labels):
            raise ValueError("Images and labels have different lengths.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index])
        if image.ndim == 1:
            if image.size != 3072:
                raise ValueError(f"Invalid flattened image shape: {image.shape}")
            image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        elif image.shape == (32, 32, 3):
            pass
        elif image.shape == (3, 32, 32):
            image = image.transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        source_is_integer = np.issubdtype(image.dtype, np.integer)
        image = image.astype(np.float32)
        min_value = float(image.min())
        max_value = float(image.max())
        if source_is_integer or (min_value >= 0.0 and max_value > 1.0):
            image /= 255.0
        elif min_value < 0.0 or max_value > 1.0:
            raise ValueError(f"Unexpected image range: [{min_value}, {max_value}]")

        image = np.clip(image, 0.0, 1.0)
        image = np.ascontiguousarray(((image - 0.5) / 0.5).transpose(2, 0, 1), dtype=np.float32)
        image_tensor = torch.from_numpy(image)
        if self.labels is None:
            return image_tensor
        return image_tensor, int(self.labels[index])


def make_loader(dataset, shuffle=False, *, batch_size=64, seed=2022, num_workers=2, pin_memory=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, generator=generator, persistent_workers=False)
