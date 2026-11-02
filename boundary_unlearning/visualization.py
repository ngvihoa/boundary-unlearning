"""Independent visualization cells for CIFAR experiments; never update models.

Figures are accompanied by metadata and numeric arrays so they can be replotted.
Sampling uses a local RNG, and every model sees exactly the same selected images.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from .constants import CIFAR_MEAN, CIFAR_STD
from .reporting import MODEL_NAMES


@contextmanager
def _evaluation(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _label(name):
    return textwrap.fill(MODEL_NAMES.get(name, name), width=23)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(fig, output_dir, name, metadata, arrays=None):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    (directory / f"{name}.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    if arrays:
        np.savez_compressed(directory / f"{name}.npz", **arrays)
    print(f"Saved: {directory / name} (.png, .pdf, .json" + (", .npz)" if arrays else ")"))
    return fig


def dataset_labels(dataset):
    """Read labels without decoding all images, including nested Subsets."""
    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    for attribute in ("targets", "labels"):
        labels = getattr(dataset, attribute, None)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            return np.asarray(labels, dtype=np.int64)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1].detach().cpu().numpy().astype(np.int64)
    raise ValueError("A labeled dataset with targets, labels or TensorDataset tensors is required")


def select_sample_indices(dataset, *, max_samples=1500, seed=2022, per_class=None):
    """Random, class-balanced selection capped before any model forward."""
    if max_samples < 1 or (per_class is not None and per_class < 1):
        raise ValueError("Sample limits must be positive")
    labels = dataset_labels(dataset)
    if not len(labels):
        raise ValueError("Cannot visualize an empty dataset")
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))
    groups = [rng.permutation(np.flatnonzero(labels == label)) for label in classes]
    selected = []
    rounds = max(len(group) for group in groups)
    if per_class is not None:
        rounds = min(rounds, per_class)
    for offset in range(rounds):
        for group in groups:
            if offset < len(group):
                selected.append(int(group[offset]))
                if len(selected) == max_samples:
                    return np.asarray(selected, dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)


def _sample_tensors(dataset, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError("Expected nonempty, valid dataset-relative sample indices")
    records = [dataset[int(index)] for index in indices]
    return torch.stack([record[0] for record in records]), np.asarray([int(record[1]) for record in records])


def _features_module(model):
    if hasattr(model, "features"):
        return model.features
    if isinstance(model, nn.Sequential):
        return model[0]
    raise ValueError("Feature extraction supports AllCNN and pruned Boundary Expanding")


def _outputs(model, images, *, device, batch_size=128, features=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks = []
    with _evaluation(model), torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)
            output = _features_module(model)(batch) if features else model(batch)
            chunks.append(output.flatten(1).detach().cpu().numpy())
    return np.concatenate(chunks)


def gradcam_map(model, image, target_class, *, device):
    """Class-specific CAM, with no parameter gradients or persistent hooks."""
    layer = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
    if layer is None:
        raise ValueError("Grad-CAM needs a Conv2d layer")
    activations = []
    hook = layer.register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        with _evaluation(model), torch.enable_grad():
            inputs = image.detach().unsqueeze(0).to(device).requires_grad_(True)
            logits = model(inputs)
            gradient = torch.autograd.grad(logits[0, target_class], activations[0])[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[0]).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=inputs.shape[2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            probabilities = logits.softmax(1)[0].detach().cpu().numpy()
    finally:
        hook.remove()
    raw_max = float(cam.max())
    span = float(cam.max() - cam.min())
    cam = (cam - cam.min()) / span if span > 0 else np.zeros_like(cam)
    return cam, probabilities, raw_max


def plot_gradcam_comparison(models, dataset, indices, *, device, class_names,
                            forget_class, output_dir, dataset_name="CIFAR-10",
                            mean=CIFAR_MEAN, std=CIFAR_STD, models_per_page=4):
    """Page model columns to keep captions legible, using the same samples."""
    if not models or models_per_page < 1:
        raise ValueError("Choose models and a positive models_per_page")
    images, labels = _sample_tensors(dataset, indices)
    groups = [list(models.items())[start:start + models_per_page]
              for start in range(0, len(models), models_per_page)]
    mean, std = np.asarray(mean), np.asarray(std)
    figures = []
    for page, group in enumerate(groups, 1):
        fig, axes = plt.subplots(len(images), len(group) + 1, squeeze=False,
                                 figsize=(2.7 * (len(group) + 1), 3.15 * len(images)), layout="constrained")
        arrays = {"indices": np.asarray(indices), "labels": labels}
        for row, image in enumerate(images):
            rgb = np.clip(image.cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_xlabel(f"Sample #{indices[row]} · {class_names[labels[row]]}", fontsize=9)
            if row == 0:
                axes[row, 0].set_title("Input image", fontsize=11)
            for col, (name, model) in enumerate(group, 1):
                cam, probabilities, raw_max = gradcam_map(model, image, forget_class, device=device)
                overlay = .55 * rgb + .45 * plt.get_cmap("magma")(cam)[..., :3]
                axes[row, col].imshow(overlay)
                prediction = int(probabilities.argmax())
                axes[row, col].set_xlabel(
                    f"Pred: {class_names[prediction]} ({probabilities[prediction]:.1%})\n"
                    f"P({class_names[forget_class]}): {probabilities[forget_class]:.1%}", fontsize=9,
                )
                if row == 0:
                    axes[row, col].set_title(_label(name), fontsize=11)
                arrays[f"{name}_sample_{row}_cam"] = cam
                arrays[f"{name}_sample_{row}_probabilities"] = probabilities
                arrays[f"{name}_sample_{row}_raw_max"] = np.asarray(raw_max)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle(f"{dataset_name} · Grad-CAM for {class_names[forget_class]} · page {page}/{len(groups)}")
        fig.supxlabel("Each map is normalized separately; color intensity is not an absolute cross-model comparison.", fontsize=9)
        figures.append(_save(fig, output_dir, f"gradcam_{page}", {
            "dataset": dataset_name, "split": "caller_selected", "indices": indices,
            "models": [name for name, _ in group], "class_names": class_names,
            "target_class": forget_class, "mean": mean, "std": std,
            "normalization": "per_map_min_max", "target_layer": "last_Conv2d",
        }, arrays))
    return figures


def _panel_grid(count):
    if count < 1:
        raise ValueError("Choose at least one model")
    columns = min(3, count)
    fig, axes = plt.subplots(math.ceil(count / columns), columns, squeeze=False,
                             figsize=(5 * columns, 4.2 * math.ceil(count / columns)), layout="constrained")
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def plot_entropy_comparison(models, datasets, sample_indices, *, device,
                            output_dir, dataset_name="CIFAR-10", batch_size=128):
    """Log Shannon entropy ECDFs; the same explicit indices for every model.

    `datasets` and `sample_indices` share split keys (train_retain, train_forget,
    test). These are descriptive diagnostics, not a membership inference test.
    """
    if not datasets:
        raise ValueError("Choose at least one labeled split")
    samples = {split: _sample_tensors(dataset, sample_indices[split])
               for split, dataset in datasets.items()}
    fig, axes = _panel_grid(len(models))
    arrays = {f"{split}_indices": np.asarray(sample_indices[split]) for split in datasets}
    for split, (_, labels) in samples.items():
        arrays[f"{split}_labels"] = labels
    all_values = []
    for ax, (name, model) in zip(axes, models.items()):
        for split_index, (split, (images, _)) in enumerate(samples.items()):
            logits = _outputs(model, images, device=device, batch_size=batch_size)
            log_probs = torch.from_numpy(logits).double().log_softmax(1)
            entropy = -(log_probs.exp() * log_probs).sum(1).numpy()
            values = np.log(np.clip(entropy, 1e-16, None))
            ordered = np.sort(values)
            ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
                    label=f"{split} (n={len(ordered)})", linewidth=1.7,
                    linestyle=("-", "--", ":")[split_index % 3])
            arrays[f"{name}_{split}_log_entropy"] = values
            all_values.extend((float(values.min()), float(values.max())))
        ax.set_title(_label(name))
        ax.set_xlabel("ln(Shannon entropy in nats)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.04)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * .03, .1)
    for ax in axes:
        ax.set_xlim(low - padding, high + padding)
    fig.suptitle(f"{dataset_name} · Output entropy distributions")
    fig.supxlabel("Descriptive comparison on fixed samples; distribution overlap alone does not establish unlearning.", fontsize=9)
    return _save(fig, output_dir, "entropy", {
        "dataset": dataset_name, "models": list(models), "sample_indices": sample_indices,
        "quantity": "ln(Shannon entropy in nats)", "display": "ECDF", "batch_size": batch_size,
    }, arrays)


def plot_tsne_comparison(models, dataset, indices, *, device, class_names,
                         forget_class, output_dir, dataset_name="CIFAR-10",
                         seed=2022, perplexity=30, max_iter=1000, batch_size=128):
    """Separate feature embeddings on identical samples, with a bounded workload."""
    from sklearn.manifold import TSNE

    images, labels = _sample_tensors(dataset, indices)
    if len(images) < 3:
        raise ValueError("t-SNE needs at least three selected samples")
    effective_perplexity = min(float(perplexity), len(images) - 1)
    if not math.isfinite(effective_perplexity) or effective_perplexity <= 0:
        raise ValueError("perplexity must be finite and positive")
    fig, axes = _panel_grid(len(models))
    arrays = {"indices": np.asarray(indices), "labels": labels}
    for ax, (name, model) in zip(axes, models.items()):
        features = _outputs(model, images, device=device, batch_size=batch_size, features=True)
        embedding = TSNE(n_components=2, perplexity=effective_perplexity, random_state=seed,
                         max_iter=max_iter, init="pca", learning_rate="auto").fit_transform(features)
        arrays[f"{name}_features"] = features
        arrays[f"{name}_embedding"] = embedding
        retain = labels != forget_class
        if len(class_names) > 10:
            ax.scatter(*embedding[retain].T, s=9, color="#87929e", alpha=.4, label="Retain classes")
        else:
            for label in np.unique(labels[retain]):
                mask = labels == label
                ax.scatter(*embedding[mask].T, s=10, alpha=.55,
                           color=plt.get_cmap("tab10")(int(label)), label=class_names[label])
        ax.scatter(*embedding[~retain].T, s=30, facecolors="none", edgecolors="#d62728",
                   linewidths=1.2, label=f"Forget: {class_names[forget_class]}", zorder=5)
        ax.set_title(_label(name))
        ax.set_xticks([])
        ax.set_yticks([])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside right upper", fontsize=8)
    fig.suptitle(f"{dataset_name} · Penultimate features · {len(images)} fixed samples")
    fig.supxlabel("Embeddings are fitted separately; axes are not aligned across models.", fontsize=9)
    return _save(fig, output_dir, "tsne", {
        "dataset": dataset_name, "models": list(models), "indices": indices,
        "class_names": class_names, "forget_class": forget_class,
        "seed": seed, "perplexity": effective_perplexity, "max_iter": max_iter,
        "init": "pca", "learning_rate": "auto", "feature_layer": "features_output",
        "retain_display": "one_gray_group" if len(class_names) > 10 else "per_class",
        "batch_size": batch_size,
    }, arrays)


def _number(value):
    return float(value) if value is not None and value != "" else float("nan")


def plot_metric_comparison(rows, *, output_dir, dataset_name="CIFAR-10"):
    """Accuracy and signed differences from Retrain, in percentage points."""
    reference = next((row for row in rows if row["model"] == "retrain"), None)
    if reference is None:
        raise ValueError("Accuracy comparison needs a retrain reference row")
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(rows) * .6)), layout="constrained")
    positions = np.arange(len(rows))
    names = [_label(row["model"]) for row in rows]
    for column, (metric, title) in enumerate((("test_retain_acc", "Test retain"), ("test_forget_acc", "Test forget"))):
        values = np.asarray([_number(row.get(metric)) * 100 for row in rows])
        delta = values - _number(reference.get(metric)) * 100
        axes[column].barh(positions, delta, color=np.where(delta >= 0, "#3568a8", "#d78b44"))
        axes[column].axvline(0, color="#444444", linewidth=.8)
        axes[column].set_title(f"{title} − Retrain", pad=36)
        axes[column].set_xlabel("Difference (percentage points)")
        axes[column].bar_label(axes[column].containers[0], fmt="%+.2f", padding=4, fontsize=8)
        axes[column].margins(x=.25)
    width = .35
    for offset, (metric, label) in enumerate((("test_retain_acc", "Retain"), ("test_forget_acc", "Forget"))):
        values = [_number(row.get(metric)) * 100 for row in rows]
        axes[2].barh(positions + (offset - .5) * width, values, height=width, label=label)
    axes[2].set_title("Test accuracy", pad=36)
    axes[2].set_xlabel("Accuracy (%)")
    axes[2].set_xlim(0, 105)
    axes[2].legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, fontsize=8)
    for i, ax in enumerate(axes):
        ax.set_yticks(positions, names if i == 0 else [])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(f"{dataset_name} · Accuracy against Retrain")
    return _save(fig, output_dir, "accuracy", {"dataset": dataset_name, "reference": "retrain", "rows": rows})


def plot_cost_comparison(rows, *, output_dir, dataset_name="CIFAR-10"):
    """Stage, student and prerequisite costs; missing values stay missing."""
    methods = [row for row in rows if row["model"] not in {"original", "retrain"}]
    if not methods:
        raise ValueError("Cost comparison needs unlearning rows")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(methods) * .8)), layout="constrained")
    groups = [(("unlearning_seconds", "Stage"), ("total_unlearning_seconds", "With prerequisites")),
              (("unlearning_update_steps", "Stage"), ("student_update_steps", "Student total"),
               ("total_update_steps", "With prerequisites"))]
    positions = np.arange(len(methods))
    for ax, fields in zip(axes, groups):
        for offset, (key, label) in enumerate(fields):
            values = [_number(row.get(key)) for row in methods]
            bars = ax.barh(positions + (offset - (len(fields) - 1) / 2) * .23,
                           values, height=.21, label=label)
            ax.bar_label(bars, fmt="%.1f" if "seconds" in key else "%.0f", padding=3, fontsize=8)
        ax.set_yticks(positions, [_label(row["model"]) for row in methods] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.margins(x=.15)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=len(fields), fontsize=8)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_title("Unlearning time", pad=36)
    axes[0].set_xlabel("Seconds (Original training excluded)")
    axes[1].set_title("Optimization budget", pad=36)
    axes[1].set_xlabel("Update steps")
    retrain = next((row.get("training_seconds") for row in rows if row["model"] == "retrain"), None)
    fig.suptitle(f"{dataset_name} · Unlearning cost")
    fig.supxlabel("Missing measurements have no bar. " +
                  (f"Retrain training: {float(retrain):.1f} s." if retrain not in (None, "") else "Retrain time unavailable."), fontsize=9)
    return _save(fig, output_dir, "cost", {"dataset": dataset_name, "rows": rows,
                                           "original_training_included": False})
