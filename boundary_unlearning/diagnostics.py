"""Paired BN-statistics ablations using existing Original/Retrain models."""

from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .attacks import find_boundary_epsilon
from .constants import CIFAR_MEAN, CIFAR_STD, REPO_ATTACK_MEAN, REPO_ATTACK_STD
from .enhanced import weight_assign_from_epsilon
from .reporting import results_table
from .unlearning import DISTANCE_METHOD, ORIGINAL_KL_METHOD, run_boundary_distance, run_boundary_shrink


@contextmanager
def _seeded(seed, device):
    """Give each candidate the same RNG stream and restore the caller's stream."""
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _evaluate(evaluate_model, model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        return evaluate_model(model)
    finally:
        for module, training in modes:
            module.training = training


def _fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bn_change(original, student):
    initial = original.state_dict()
    final = student.state_dict()
    means, variances, counts = [], [], []
    for name, module in original.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm) or not module.track_running_stats:
            continue
        prefix = f"{name}." if name else ""
        means.append(float((final[prefix + "running_mean"] - initial[prefix + "running_mean"]).abs().max()))
        variances.append(float((final[prefix + "running_var"] - initial[prefix + "running_var"]).abs().max()))
        counts.append(int(final[prefix + "num_batches_tracked"] - initial[prefix + "num_batches_tracked"]))
    return {"bn_running_mean_max_change": max(means, default=0),
            "bn_running_var_max_change": max(variances, default=0),
            "bn_batches_min_change": min(counts, default=0),
            "bn_batches_max_change": max(counts, default=0)}


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_search(original, loader, directory, *, device, bound, epsilon_step,
                  max_batches_per_epoch, cifar_mean, cifar_std, pin_memory):
    """One teacher-only pass; separate from the fixed-bound Shrink objective."""
    teacher = copy.deepcopy(original).to(device).eval().requires_grad_(False)
    chunks = {key: [] for key in ("epsilon", "weight", "label", "pseudo_label", "batch_index")}
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch:
            break
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        targets, epsilons = find_boundary_epsilon(
            images, labels, teacher, bound=bound, epsilon_step=epsilon_step,
            cifar_mean=cifar_mean, cifar_std=cifar_std,
        )
        weights = weight_assign_from_epsilon(epsilons, bound)
        for key, value in (("epsilon", epsilons), ("weight", weights), ("label", labels),
                           ("pseudo_label", targets), ("batch_index", torch.full_like(labels, batch_index))):
            chunks[key].append(value.detach().cpu().numpy())
    if not chunks["epsilon"]:
        raise ValueError("Boundary audit needs at least one forget batch")
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    np.savez_compressed(directory / "boundary_search_audit.npz", **arrays)
    epsilon, weight = arrays["epsilon"], arrays["weight"]
    summary = {"samples": len(epsilon), "epsilon_mean": float(epsilon.mean()),
               "epsilon_min": float(epsilon.min()), "epsilon_max": float(epsilon.max()),
               "epsilon_std": float(epsilon.std()), "weight_mean": float(weight.mean()),
               "weight_min": float(weight.min()), "weight_max": float(weight.max()),
               "weight_std": float(weight.std()),
               "already_wrong_fraction": float(np.mean(epsilon == 0)),
               "first_step_fraction": float(np.mean(np.isclose(epsilon, min(epsilon_step, bound)))),
               "cross_fraction": float(np.mean(arrays["pseudo_label"] != arrays["label"])),
               "normalization": "matched", "bound": bound, "epsilon_step": epsilon_step,
               "scope": "one seed-shuffled teacher-only pass; not Shrink training pseudo-labels"}
    (directory / "boundary_search_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _plot_logs(directory, stats, audit):
    import matplotlib.pyplot as plt

    if audit is not None:
        arrays = np.load(directory / "boundary_search_audit.npz")
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
        for ax, key, title in zip(axes, ("epsilon", "weight"), ("Boundary epsilon", "Batch-normalized sample weight")):
            ax.hist(arrays[key], bins=30, edgecolor="white")
            ax.set_title(title)
            ax.set_xlabel(key)
            ax.set_ylabel("Samples")
        figure.suptitle("Original teacher audit · one forget pass")
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"boundary_search_audit.{extension}", bbox_inches="tight", dpi=180)
        plt.close(figure)
        arrays.close()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, values in stats.items():
        with Path(values["losses_path"]).open() as stream:
            history = list(csv.DictReader(stream))
        steps = [int(row["step"]) for row in history]
        axes[0].plot(steps, [float(row["forget_ce"]) for row in history], label=name)
        axes[1].plot(steps, [float(row["scaled_retain_kl"]) for row in history], label=name)
    axes[0].set_title("Forget CE (weighted for distance variants)")
    axes[1].set_title("lambda × T² × retain KL (zero for plain Shrink)")
    for ax in axes:
        ax.set_xlabel("Update step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"losses.{extension}", bbox_inches="tight", dpi=180)
    plt.close(figure)


def run_batchnorm_ablation(original, retrain, forget_loader_factory, evaluate_model, output_root, *,
                           device, source_run_tag, seed=2022, methods=("boundary_shrink",),
                           retain_loader_factory=None, epochs=10, learning_rate=1e-5,
                           momentum=0.9, pin_memory=False, max_batches_per_epoch=None,
                           bound=0.1, normalization_mode="matched", epsilon_step=1 / 255,
                           kl_lambda=0.1, temperature=2.0, cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD,
                           attack_mean=REPO_ATTACK_MEAN, attack_std=REPO_ATTACK_STD,
                           dataset_name="CIFAR-100", metadata=None, baseline_training_seconds=None,
                           audit_search=True, plots=True):
    """Run update/freeze pairs from the exact same provided Original model.

    Factories must return a NEW seed-controlled loader on every call. Defaults
    run plain Shrink only. Distance and Original-KL can be enabled explicitly;
    the warm-start variant is excluded to isolate BN from initializer changes.
    Existing Original/Retrain and main-flow result dictionaries are untouched.
    """
    allowed = {"boundary_shrink", DISTANCE_METHOD, ORIGINAL_KL_METHOD}
    if not methods or len(set(methods)) != len(methods) or set(methods) - allowed:
        raise ValueError("Choose unique plain Shrink, distance, or Original-KL methods")
    if epochs < 1 or (max_batches_per_epoch is not None and max_batches_per_epoch < 1):
        raise ValueError("Ablation needs positive epochs and batch limits")
    if not callable(forget_loader_factory):
        raise ValueError("Provide a factory for nonempty fresh forget loaders")
    with _seeded(seed, device):
        if len(forget_loader_factory()) == 0:
            raise ValueError("Provide a factory for nonempty fresh forget loaders")
    if not np.isfinite(bound) or bound <= 0 or not np.isfinite(epsilon_step) or epsilon_step <= 0:
        raise ValueError("bound and epsilon_step must be finite and positive")
    if ORIGINAL_KL_METHOD in methods and retain_loader_factory is None:
        raise ValueError("Original-KL needs a fresh retain_loader_factory")
    if Path(source_run_tag).name != source_run_tag:
        raise ValueError("source_run_tag must be a filename component")
    settings = {
        "implementation": "bn_statistics_ablation_v1", "source_run_tag": source_run_tag,
        "dataset": dataset_name, "seed": seed, "methods": list(methods), "modes": ["update", "freeze"],
        "epochs": epochs, "learning_rate": learning_rate, "momentum": momentum,
        "bound": bound, "normalization_mode": normalization_mode, "epsilon_step": epsilon_step,
        "kl_lambda": kl_lambda, "temperature": temperature,
        "cifar_mean": cifar_mean, "cifar_std": cifar_std, "attack_mean": attack_mean, "attack_std": attack_std,
        "max_batches_per_epoch": max_batches_per_epoch, "audit_search": audit_search,
        "original_state_sha256": _fingerprint(original), "retrain_state_sha256": _fingerprint(retrain),
        "model_source": "provided_in_memory_models; no checkpoint reloaded", "metadata": metadata or {},
        "freeze_scope": "BN.eval: use and preserve stored running statistics; affine BN parameters remain trainable",
    }
    suffix = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    output_dir = Path(output_root) / "bn_ablation" / f"{source_run_tag}_{suffix}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows, models, stats = [], {}, {}
    for name, model in (("original", original), ("retrain", retrain)):
        with _seeded(seed, device):
            metrics = _evaluate(evaluate_model, model)
        rows.append({"model": name, **metrics, "training_seconds": (baseline_training_seconds or {}).get(name),
                     "unlearning_seconds": None, "fgsm_cross_rate": None})
    common = dict(device=device, epochs=epochs, learning_rate=learning_rate, momentum=momentum,
                  pin_memory=pin_memory, max_batches_per_epoch=max_batches_per_epoch,
                  bound=bound, cifar_mean=cifar_mean, cifar_std=cifar_std)
    for method in methods:
        for mode in ("update", "freeze"):
            name = f"{method}_bn_{mode}"
            with _seeded(seed, device):
                loader = forget_loader_factory()
                kwargs = dict(common, bn_stats_mode=mode, diagnostics_dir=output_dir / name)
                if method == "boundary_shrink":
                    model, values = run_boundary_shrink(
                        original, loader, checkpoint_dir / f"{name}.pt", **kwargs,
                        normalization_mode=normalization_mode, attack_mean=attack_mean, attack_std=attack_std,
                    )
                else:
                    model, values = run_boundary_distance(
                        original, loader, checkpoint_dir / f"{name}.pt", **kwargs, method=method,
                        retain_loader=retain_loader_factory() if method == ORIGINAL_KL_METHOD else None,
                        epsilon_step=epsilon_step, kl_lambda=kl_lambda, temperature=temperature,
                    )
                metrics = _evaluate(evaluate_model, model)
            values.update(_bn_change(original, model))
            rows.append({"model": name, **metrics, "training_seconds": None,
                         "unlearning_seconds": values["seconds"], "fgsm_cross_rate": values.get("cross_rate"),
                         "unlearning_update_steps": values["update_steps"], "bn_stats_mode": mode,
                         **{key: value for key, value in values.items() if key.startswith(("bn_", "mean_"))}})
            models[name], stats[name] = model, values
    audit = None
    if audit_search:
        with _seeded(seed, device):
            audit = _audit_search(original, forget_loader_factory(), output_dir, device=device,
                                  bound=bound, epsilon_step=epsilon_step,
                                  max_batches_per_epoch=max_batches_per_epoch,
                                  cifar_mean=cifar_mean, cifar_std=cifar_std, pin_memory=pin_memory)
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "config.json"
    _write_csv(metrics_path, rows)
    config_path.write_text(json.dumps({"settings": settings, "stats": stats, "boundary_search_audit": audit}, indent=2), encoding="utf-8")
    results_table(rows, dataset_name=dataset_name).to_html(output_dir / "results.html")
    if plots:
        _plot_logs(output_dir, stats, audit)
    print(f"BN ablation saved: {output_dir}")
    return {"models": models, "rows": rows, "stats": stats, "audit": audit,
            "output_dir": output_dir, "config_path": config_path, "metrics_path": metrics_path}
