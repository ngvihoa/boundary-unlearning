"""Export the evaluated CIFAR-100 baselines and restore them for Phase 2."""

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import torch
from torch import nn

from .models import AllCNN
from .reporting import results_table


PHASE1_MODELS = ("original", "retrain", "boundary_shrink", "boundary_expanding")
MANIFEST_NAME = "phase1_manifest.json"
BUNDLE_FORMAT = "boundary_unlearning.cifar100.phase1.v1"


def _sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def dataset_identity(train_dataset, test_dataset):
    """Fingerprint ordered raw pixels and fine labels without copying full splits."""
    identity = {}
    for split, dataset in (("train", train_dataset), ("test", test_dataset)):
        identity[f"{split}_count"] = len(dataset)
        for name, values in (("images", dataset.data), ("labels", dataset.targets)):
            array = np.ascontiguousarray(values)
            digest = hashlib.sha256(f"{array.dtype}:{array.shape}".encode())
            view = memoryview(array).cast("B")
            for offset in range(0, len(view), 4 * 1024 * 1024):
                digest.update(view[offset:offset + 4 * 1024 * 1024])
            identity[f"{split}_{name}_sha256"] = digest.hexdigest()
    return identity


def validate_phase1_dataset(config, train_dataset, test_dataset):
    """Require the same ordered data, class mapping and normalization in Phase 2."""
    if list(train_dataset.classes) != config["class_names"] or list(test_dataset.classes) != config["class_names"]:
        raise ValueError("Phase 2 class names/order differ from Phase 1")
    for dataset in (train_dataset, test_dataset):
        for key, value in (("input_mean", dataset.mean), ("input_std", dataset.std)):
            if not np.array_equal(value.cpu().numpy().reshape(-1), np.asarray(config[key], dtype=np.float32)):
                raise ValueError(f"Phase 2 {key} differs from Phase 1")
    if dataset_identity(train_dataset, test_dataset) != config["data_identity"]:
        raise ValueError("Phase 2 pixels/labels/order differ from Phase 1")


def _model_for(name, architecture):
    # Constructor initialization must not change the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]):
        base = AllCNN(**architecture)
    return nn.Sequential(base.features, base.classifier[0]) if name == "boundary_expanding" else base


def _validate_config(config):
    if (config.get("experiment_phase") != 1 or config.get("dataset") != "CIFAR-100"
            or config.get("num_classes") != 100 or config.get("label_type") != "fine"
            or config.get("methods") != list(PHASE1_MODELS[2:])
            or config.get("extra_exp") is not None or config.get("bn_stats_mode") != "update"):
        raise ValueError("Expected a Phase 1 CIFAR-100 paper-method bundle (plain Shrink, BN update)")
    if len(config.get("class_names", [])) != 100 or not 0 <= config.get("forget_class", -1) < 100:
        raise ValueError("Invalid Phase 1 class mapping")
    architecture = config.get("model_architecture")
    if architecture != dict(n_channels=3, num_classes=100, filters_percentage=.5, batch_norm=True):
        raise ValueError("Unsupported Phase 1 AllCNN architecture")
    if not config.get("data_identity"):
        raise ValueError("Missing Phase 1 dataset identity")
    return architecture


def save_phase1_bundle(models, rows, config, destination, *, run_tag):
    """Save the exact in-memory states evaluated in rows, separately from best checkpoints.

    Call immediately after evaluation, before doing any further training. The
    final notebook export cell only archives this existing bundle.
    """
    architecture = _validate_config(config)
    if set(models) != set(PHASE1_MODELS) or [row["model"] for row in rows] != list(PHASE1_MODELS):
        raise ValueError("Phase 1 requires exactly Original, Retrain, Shrink and Expanding")
    if not run_tag or Path(run_tag).name != run_tag:
        raise ValueError("run_tag must be a filename component")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"format": BUNDLE_FORMAT, "run_tag": run_tag,
                "checkpoint_selection": "exact_evaluated_in_memory_state",
                "files": {}, "models": {}}

    def write(relative, contents):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return {"path": relative, "sha256": _sha256(contents)}

    manifest["files"]["config"] = write("config.json", json.dumps(config, indent=2).encode())
    manifest["files"]["rows"] = write("rows.json", json.dumps(rows, indent=2).encode())
    table = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    manifest["files"]["metrics"] = write("metrics.csv", table.getvalue().encode())
    manifest["files"]["results"] = write("results.html", results_table(rows, dataset_name="CIFAR-100").to_html().encode())
    for name in PHASE1_MODELS:
        state = {key: value.detach().cpu() for key, value in models[name].state_dict().items()}
        _model_for(name, architecture).load_state_dict(state, strict=True)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        manifest["models"][name] = write(f"checkpoints/{name}_evaluated.pt", buffer.getvalue())
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def find_phase1_source(input_root):
    """Auto-select one attached bundle; ambiguous inputs require an explicit path."""
    root = Path(input_root)
    # Saved Kaggle Output includes both result/ and result.zip. Count identical
    # manifests once and prefer the extracted folder when both are attached.
    sources = {_sha256(path.read_bytes()): path for path in root.rglob(MANIFEST_NAME)}
    multiple_bundle_archives = []
    for archive in root.rglob("*.zip"):
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zipped:
                manifests = [name for name in zipped.namelist() if PurePosixPath(name).name == MANIFEST_NAME]
                if len(manifests) == 1:
                    sources.setdefault(_sha256(zipped.read(manifests[0])), archive)
                elif manifests:
                    multiple_bundle_archives.append(archive)
    if len(sources) != 1 or multiple_bundle_archives:
        raise FileNotFoundError(
            f"Expected one Phase 1 bundle under {root}; found {len(sources)}. "
            "Attach the Phase 1 result.zip (or extracted result folder) and set PHASE1_SOURCE explicitly if needed. "
            f"Archives containing multiple runs: {multiple_bundle_archives}"
        )
    return next(iter(sources.values()))


def _relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError("Invalid bundle-relative path")
    return path


def load_phase1_bundle(source, *, device, copy_to=None):
    """Load a ZIP, extracted result folder or manifest without training any model.

    Read only manifest-listed files, verify hashes, and reconstruct the pruned
    Expanding layout correctly. ZIP entries are never extracted as filesystem paths.
    copy_to optionally keeps a verified bundle in the Phase 2 result export.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Phase 1 source not found: {source}")
    if source.is_dir():
        matches = list(source.rglob(MANIFEST_NAME))
        if len(matches) != 1:
            raise ValueError("Select a folder containing exactly one Phase 1 manifest")
        source = matches[0]
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zipped:
            matches = [name for name in zipped.namelist() if PurePosixPath(name).name == MANIFEST_NAME]
            if len(matches) != 1:
                raise ValueError("Select a ZIP containing exactly one Phase 1 manifest")
            manifest_member = _relative_path(matches[0])
            manifest_bytes = zipped.read(str(manifest_member))

        def read(relative):
            with zipfile.ZipFile(source) as zipped:
                return zipped.read(str(manifest_member.parent / _relative_path(relative)))
    else:
        if source.name != MANIFEST_NAME:
            raise ValueError(f"Expected {MANIFEST_NAME}, a result folder, or a ZIP")
        manifest_bytes = source.read_bytes()

        def read(relative):
            path = (source.parent / str(_relative_path(relative))).resolve()
            if not path.is_relative_to(source.parent.resolve()):
                raise ValueError("Bundle file resolves outside its folder")
            return path.read_bytes()

    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("checkpoint_selection") != "exact_evaluated_in_memory_state":
        raise ValueError("Unsupported Phase 1 bundle format/model selection")
    run_tag = manifest["run_tag"]
    if not run_tag or Path(run_tag).name != run_tag:
        raise ValueError("Invalid Phase 1 run tag")
    if set(manifest["models"]) != set(PHASE1_MODELS):
        raise ValueError("Phase 1 bundle must contain exactly four evaluated models")
    verified = {}

    def checked(entry):
        contents = read(entry["path"])
        if _sha256(contents) != entry["sha256"]:
            raise ValueError(f"Phase 1 SHA-256 mismatch: {entry['path']}")
        verified[entry["path"]] = contents
        return contents

    contents = {name: checked(entry) for name, entry in manifest["files"].items()}
    config, rows = json.loads(contents["config"]), json.loads(contents["rows"])
    architecture = _validate_config(config)
    if [row["model"] for row in rows] != list(PHASE1_MODELS):
        raise ValueError("Phase 1 rows must describe the four baseline models")
    models = {}
    for name in PHASE1_MODELS:
        model = _model_for(name, architecture)
        state = torch.load(io.BytesIO(checked(manifest["models"][name])), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        models[name] = model.to(device).eval()
    output_dir = None
    if copy_to is not None:
        output_dir = Path(copy_to) / run_tag
        for relative, contents in verified.items():
            path = output_dir / str(_relative_path(relative))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        (output_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
    return {"models": models, "rows": rows, "config": config, "manifest": manifest,
            "manifest_sha256": _sha256(manifest_bytes), "source": str(source), "output_dir": output_dir}
