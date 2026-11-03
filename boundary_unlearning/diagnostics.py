"""Paired BN-statistics and sample-weighting ablations using existing baselines."""

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
from .unlearning import DISTANCE_METHOD, ORIGINAL_KL_METHOD, run_boundary_distance, run_boundary_shrink, run_boundary_expanding


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
                           audit_search=True, plots=True, bn_modes=("update", "freeze"),
                           weighting_modes=("distance",), experiment_name="bn_ablation"):
    """Run update/freeze pairs from the exact same provided Original model.

    Factories must return a NEW seed-controlled loader on every call. Defaults
    run plain Shrink only. Distance and Original-KL can be enabled explicitly;
    the warm-start variant is excluded to isolate BN from initializer changes.
    Existing Original/Retrain and main-flow result dictionaries are untouched.
    """
    allowed = {"boundary_shrink", DISTANCE_METHOD, ORIGINAL_KL_METHOD}
    if not bn_modes or len(set(bn_modes)) != len(bn_modes) or set(bn_modes) - {"update", "freeze"}:
        raise ValueError("Choose unique update/freeze BN modes")
    if not weighting_modes or len(set(weighting_modes)) != len(weighting_modes) or set(weighting_modes) - {"distance", "uniform"}:
        raise ValueError("Choose unique distance/uniform weighting modes")
    if "uniform" in weighting_modes and "boundary_shrink" in methods:
        raise ValueError("Uniform weighting control requires distance pseudo-label methods")
    if experiment_name not in {"bn_ablation", "weighting_ablation"}:
        raise ValueError("Unsupported experiment_name")
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
        "dataset": dataset_name, "seed": seed, "methods": list(methods), "modes": list(bn_modes),
        "epochs": epochs, "learning_rate": learning_rate, "momentum": momentum,
        "bound": bound, "normalization_mode": normalization_mode, "epsilon_step": epsilon_step,
        "kl_lambda": kl_lambda, "temperature": temperature,
        "cifar_mean": cifar_mean, "cifar_std": cifar_std, "attack_mean": attack_mean, "attack_std": attack_std,
        "max_batches_per_epoch": max_batches_per_epoch, "audit_search": audit_search,
        "original_state_sha256": _fingerprint(original), "retrain_state_sha256": _fingerprint(retrain),
        "model_source": "provided_in_memory_models; no checkpoint reloaded", "metadata": metadata or {},
        "freeze_scope": "BN.eval: use and preserve stored running statistics; affine BN parameters remain trainable",
    }
    if experiment_name != "bn_ablation" or tuple(weighting_modes) != ("distance",):
        settings.update(implementation="distance_weighting_ablation_v1", weighting_modes=list(weighting_modes),
                        experiment_name=experiment_name)
    suffix = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    output_dir = Path(output_root) / experiment_name / f"{source_run_tag}_{suffix}"
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
        for weighting_mode in weighting_modes:
            for mode in bn_modes:
                weight_suffix = "_uniform" if weighting_mode == "uniform" else ""
                name = f"{method}{weight_suffix}_bn_{mode}"
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
                            weighting_mode=weighting_mode,
                        )
                    metrics = _evaluate(evaluate_model, model)
                values.update(_bn_change(original, model))
                rows.append({"model": name, **metrics, "training_seconds": None,
                             "unlearning_seconds": values["seconds"], "fgsm_cross_rate": values.get("cross_rate"),
                             "unlearning_update_steps": values["update_steps"], "bn_stats_mode": mode,
                             "weighting_mode": values.get("weighting_mode"),
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
    print(f"{experiment_name} saved: {output_dir}")
    return {"models": models, "rows": rows, "stats": stats, "audit": audit,
            "output_dir": output_dir, "config_path": config_path, "metrics_path": metrics_path}


def run_weighting_ablation(original, retrain, forget_loader_factory, evaluate_model, output_root, **settings):
    """Isolate sample weighting: same distance targets, frozen BN, no retain KL.

    Both students start independently from Original with the same seed/loaders.
    Validate the logged labels, pseudo-labels and epsilons before comparing rows.
    """
    result = run_batchnorm_ablation(
        original, retrain, forget_loader_factory, evaluate_model, output_root,
        methods=(DISTANCE_METHOD,), bn_modes=("freeze",), weighting_modes=("distance", "uniform"),
        experiment_name="weighting_ablation", audit_search=False, kl_lambda=0.0, **settings,
    )
    names = (f"{DISTANCE_METHOD}_bn_freeze", f"{DISTANCE_METHOD}_uniform_bn_freeze")
    samples = []
    for name in names:
        with np.load(result["stats"][name]["boundary_samples_path"]) as arrays:
            samples.append({key: arrays[key].copy() for key in arrays.files})
    paired = all(np.array_equal(samples[0][key], samples[1][key])
                 for key in ("label", "pseudo_label", "epsilon", "epoch"))
    uniform = bool(np.all(samples[1]["weight"] == 1))
    validation = {"labels_targets_epsilons_epochs_identical": paired, "uniform_weights_all_one": uniform}
    (result["output_dir"] / "pair_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not paired or not uniform:
        raise RuntimeError("Invalid weighting pair: check fresh, seed-controlled forget loaders and uniform weights")
    by_name = {row["model"]: row for row in result["rows"]}
    distance, control = (by_name[name] for name in names)
    comparison = {"model": names[1], "reference": names[0]}
    for metric in ("test_retain_acc", "test_forget_acc", "train_retain_acc", "train_forget_acc"):
        comparison[metric.replace("_acc", "_delta_pp")] = (
            100 * (control[metric] - distance[metric]) if control.get(metric) is not None and distance.get(metric) is not None else None)
    comparison["mean_forget_ce_delta"] = control["mean_forget_ce"] - distance["mean_forget_ce"]
    _write_csv(result["output_dir"] / "weighting_comparison.csv", [comparison])
    weight_summary = {}
    for name, arrays in zip(names, samples):
        weights = arrays["weight"]
        weight_summary[name] = {"min": float(weights.min()), "max": float(weights.max()),
                                "mean": float(weights.mean()), "std": float(weights.std())}
    config = json.loads(result["config_path"].read_text())
    config.update(pair_validation=validation, weighting_comparison=comparison, applied_weights=weight_summary)
    result["config_path"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    if settings.get("plots", True):
        import matplotlib.pyplot as plt
        figure, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        low = min(float(arrays["weight"].min()) for arrays in samples)
        high = max(float(arrays["weight"].max()) for arrays in samples)
        padding = max((high - low) * .05, .001)
        bins = np.linspace(low - padding, high + padding, 31)
        for arrays, label in zip(samples, ("Distance weights", "Uniform weights = 1")):
            ax.hist(arrays["weight"], bins=bins, histtype="step", linewidth=2, label=label)
        ax.set_xlabel("Applied sample weight")
        ax.set_ylabel("Sample presentations across all unlearning epochs")
        ax.set_title("Weighting control · identical distance pseudo-labels · BN freeze")
        ax.legend()
        for extension in ("png", "pdf"):
            figure.savefig(result["output_dir"] / f"applied_weights.{extension}", bbox_inches="tight", dpi=180)
        plt.close(figure)
    result.update(comparison=[comparison], pair_validation=validation, applied_weights=weight_summary)
    return result


def run_phase1_epoch_sweep(source, train_dataset, test_dataset, output_root, *, device,
                           epochs=(1, 3, 5, 10, 20), plots=True):
    """Compare paper-method budgets using the same evaluated Phase 1 initializer.

    Restore evaluated snapshots (never the legacy best checkpoints). Every
    method/budget starts independently with a fresh seeded loader and optimizer.
    Export a complete Phase 1 bundle per budget so Phase 2 can select one later.
    """
    from functools import partial
    import platform
    import subprocess
    from torch.utils.data import Subset
    from . import data, phases, training, unlearning as algorithms

    budgets = list(epochs)
    if (not budgets or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
                           for value in budgets) or len(set(budgets)) != len(budgets)):
        raise ValueError("Epoch budgets must be unique positive integers")
    budgets = [int(value) for value in budgets]
    loaded = phases.load_phase1_bundle(source, device="cpu")
    config = loaded["config"]
    phases.validate_phase1_dataset(config, train_dataset, test_dataset)
    settings = {"implementation": "phase1_epoch_sweep_v1", "epochs": budgets,
                "source_run": loaded["manifest"]["run_tag"],
                "source_manifest_sha256": loaded["manifest_sha256"],
                "source_checkpoint_hashes": {name: entry["sha256"] for name, entry in loaded["manifest"]["models"].items()},
                "seed": config["seed"], "bn_stats_mode": "update",
                "unlearning_code_sha256": hashlib.sha256(Path(algorithms.__file__).read_bytes()).hexdigest(),
                "runner_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "initializer": "same evaluated Original for every method/budget; fresh optimizer and loader"}
    code_root = Path(__file__).resolve().parents[1]
    current_commit = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip() if (code_root / ".git").exists() else None
    suffix = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(output_root) / "epoch_sweep" / f"{settings['source_run']}_{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    original, retrain = (loaded["models"][name].to(device) for name in ("original", "retrain"))
    baseline_rows = copy.deepcopy(loaded["rows"][:2])
    forget_class = config["forget_class"]
    forget = Subset(train_dataset, np.flatnonzero(train_dataset.targets == forget_class).tolist())
    retain = Subset(train_dataset, np.flatnonzero(train_dataset.targets != forget_class).tolist())
    test_forget = Subset(test_dataset, np.flatnonzero(test_dataset.targets == forget_class).tolist())
    test_retain = Subset(test_dataset, np.flatnonzero(test_dataset.targets != forget_class).tolist())
    loader = partial(data.make_loader, batch_size=config["batch_size"], seed=config["seed"],
                     num_workers=config["num_workers"], pin_memory=torch.device(device).type == "cuda")
    evaluate = partial(training.evaluate_model_splits, device=device,
                       test_loader=loader(test_dataset), test_retain_loader=loader(test_retain),
                       test_forget_loader=loader(test_forget), train_retain_eval_loader=loader(retain),
                       train_forget_eval_loader=loader(forget), include_train=config["evaluate_train_splits"])
    common = dict(device=device, learning_rate=config["unlearn_learning_rate"], momentum=config["momentum"],
                  pin_memory=torch.device(device).type == "cuda", max_batches_per_epoch=config["max_batches_per_epoch"])
    rows, bundles = [], {}
    for budget in budgets:
        run_tag = f"cifar100_phase1_epochs{budget}_{suffix}"
        destination = directory / "runs" / run_tag
        current_rows, students, stats = copy.deepcopy(baseline_rows), {}, {}
        for method in phases.PHASE1_MODELS[2:]:
            with _seeded(config["seed"], device):
                checkpoint = destination / "checkpoints" / f"{method}.pt"
                if method == "boundary_shrink":
                    model, values = run_boundary_shrink(
                        original, loader(forget, shuffle=True), checkpoint, **common, epochs=budget,
                        bound=config["fgsm_bound"], normalization_mode=config["attack_normalization"],
                        cifar_mean=config["input_mean"], cifar_std=config["input_std"],
                        attack_mean=config["attack_mean"], attack_std=config["attack_std"],
                        extra_exp=None, bn_stats_mode="update", diagnostics_dir=destination / method,
                    )
                else:
                    model, values = run_boundary_expanding(
                        original, loader(forget, shuffle=True), checkpoint, **common, epochs=budget,
                        num_classes=config["num_classes"],
                    )
                metrics = _evaluate(evaluate, model)
            row = {"model": method, **metrics, "training_seconds": None,
                   "unlearning_seconds": values["seconds"], "fgsm_cross_rate": values.get("cross_rate"),
                   "unlearning_update_steps": values["update_steps"], "student_update_steps": values["update_steps"],
                   "total_update_steps": values["update_steps"], "total_unlearning_seconds": values["seconds"],
                   "initializer": "original", "unlearn_epochs": budget}
            current_rows.append(row)
            rows.append(row)
            students[method], stats[method] = model, values
        candidate_config = copy.deepcopy(config)
        candidate_config.update(unlearn_epochs=budget, unlearning_stats=stats, epoch_sweep=settings,
                                source_code_commit=config.get("code_commit"), code_commit=current_commit,
                                device=str(device),
                                environment={**config.get("environment", {}), "python": platform.python_version(),
                                             "torch": str(torch.__version__), "numpy": np.__version__},
                                baseline_checkpoint_policy="evaluated snapshots; paths relative to this bundle",
                                baseline_checkpoints={name: f"checkpoints/{name}_evaluated.pt" for name in ("original", "retrain")})
        candidate_config["run_tag_settings"] = {**config["run_tag_settings"], "unlearn_epochs": budget,
                                                "epoch_sweep": settings}
        bundles[budget] = phases.save_phase1_bundle(
            {"original": original, "retrain": retrain, **students}, current_rows,
            candidate_config, destination, run_tag=run_tag,
        )
    _write_csv(directory / "metrics.csv", [*baseline_rows, *rows])
    (directory / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if plots:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        metrics = ("test_retain_acc", "test_forget_acc", "train_retain_acc", "train_forget_acc")
        for ax, metric in zip(axes.flat, metrics):
            for method in phases.PHASE1_MODELS[2:]:
                selected = sorted((row for row in rows if row["model"] == method), key=lambda row: row["unlearn_epochs"])
                ax.plot([row["unlearn_epochs"] for row in selected],
                        [100 * row[metric] if row[metric] is not None else np.nan for row in selected],
                        marker="o", label=method.replace("boundary_", "").title())
            for row, style in zip(baseline_rows, (":", "--")):
                if row.get(metric) is not None:
                    ax.axhline(100 * row[metric], linestyle=style, color="gray", label=row["model"].title())
            ax.set_title(metric.replace("_acc", "").replace("_", " ").title())
            ax.set_xlabel("Unlearning epochs")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(sorted(budgets))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        figure.suptitle("Phase 1 · Epoch comparison · Same evaluated Original · BN update")
        for extension in ("png", "pdf"):
            figure.savefig(directory / f"epochs.{extension}", bbox_inches="tight", dpi=180)
        plt.close(figure)
    return {"rows": rows, "baseline_rows": baseline_rows, "bundles": bundles,
            "output_dir": directory, "settings": settings}
