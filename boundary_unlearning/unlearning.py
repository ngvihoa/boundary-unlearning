"""Boundary Shrink, Boundary Expanding and distance-weighted extensions."""

import copy
import csv
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attacks import fgsm_cifar, find_boundary_epsilon
from .constants import CIFAR_MEAN, CIFAR_STD, REPO_ATTACK_MEAN, REPO_ATTACK_STD
from .enhanced import (
    curvature, retain_distillation_loss, weight_assign, weight_assign_from_epsilon,
)
from .models import init_linear_like_repo
from .training import save_state_dict, sync_if_cuda


DISTANCE_METHOD = "boundary_shrink_weight_assign"
ORIGINAL_KL_METHOD = "boundary_shrink_weight_assign_distill"
BOUNDARY_KL_METHOD = "boundary_shrink_weight_assign_boundary_distill"
DISTANCE_METHODS = (DISTANCE_METHOD, ORIGINAL_KL_METHOD, BOUNDARY_KL_METHOD)
SUPPORTED_METHODS = ("boundary_shrink", "boundary_expanding", *DISTANCE_METHODS)


def _validate_bn_mode(mode):
    if mode not in {"update", "freeze"}:
        raise ValueError("bn_stats_mode must be 'update' or 'freeze'")


def _student_train_mode(model, bn_stats_mode):
    # Only freeze running statistics; BN affine parameters still receive gradients.
    model.train()
    if bn_stats_mode == "freeze":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def _save_diagnostics(directory, history, samples=None):
    if directory is None:
        return {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    loss_path = directory / "losses.csv"
    fields = ("epoch", "step", "batch_size", "forget_ce", "retain_kl",
              "scaled_retain_kl", "total_loss")
    with loss_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    result = {"losses_path": str(loss_path)}
    if samples and samples["epsilon"]:
        sample_path = directory / "boundary_samples.npz"
        np.savez_compressed(sample_path, **{key: np.concatenate(value) for key, value in samples.items()})
        result["boundary_samples_path"] = str(sample_path)
    if history:
        total = sum(row["batch_size"] for row in history)
        for key in fields[3:]:
            result[f"mean_{key}"] = sum(row[key] * row["batch_size"] for row in history) / total
    return result


def validate_methods(methods, *, extra_exp=None):
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Choose at least one method, without duplicates")
    if any(method not in SUPPORTED_METHODS for method in methods):
        raise ValueError("Unsupported unlearning method")
    if BOUNDARY_KL_METHOD in methods:
        if "boundary_shrink" not in methods or DISTANCE_METHOD not in methods:
            raise ValueError("Boundary KL needs boundary_shrink and distance weighting")
        if extra_exp is not None:
            raise ValueError("Boundary KL uses the plain Boundary Shrink teacher: set EXTRA_EXP=None")


def run_boundary_distance(original, forget_loader, checkpoint_path, *, device,
                          method=DISTANCE_METHOD, retain_loader=None,
                          weight_assign_model=None, boundary_model=None,
                          dependency_stats=None, epochs=10, learning_rate=1e-5,
                          momentum=0.9, pin_memory=False,
                          max_batches_per_epoch=None, bound=0.1,
                          epsilon_step=1 / 255, kl_lambda=0.1, temperature=2.0,
                          cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD,
                          bn_stats_mode="update", diagnostics_dir=None,
                          weighting_mode="distance"):
    """Run one of the three distance-weighted variants in no-refactor-new.

    The Original-KL student starts from Original. Boundary-KL starts from the
    distance-weighted student and uses plain Boundary Shrink as retain teacher.
    All pseudo-labels come from Original. Existing model objects are never changed.
    Timing excludes checkpoint writing; totals include both prerequisite stages
    for Boundary-KL, but exclude Original training for every method.
    bn_stats_mode='freeze' uses stored BN statistics on forget forwards while
    keeping affine parameters trainable. Optional diagnostics log each update.
    weighting_mode='uniform' preserves boundary search/targets but uses weights=1.
    """
    _validate_bn_mode(bn_stats_mode)
    if weighting_mode not in {"distance", "uniform"}:
        raise ValueError("weighting_mode must be 'distance' or 'uniform'")
    if method not in DISTANCE_METHODS:
        raise ValueError(f"Unsupported distance method: {method}")
    if not math.isfinite(bound) or bound <= 0 or not math.isfinite(epsilon_step) or epsilon_step <= 0:
        raise ValueError("bound and epsilon_step must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(kl_lambda) or kl_lambda < 0:
        raise ValueError("temperature must be positive and kl_lambda nonnegative, both finite")
    distill = method != DISTANCE_METHOD
    boundary_distill = method == BOUNDARY_KL_METHOD
    if distill and (retain_loader is None or len(retain_loader) == 0):
        raise ValueError("Distillation needs a nonempty retain_loader")
    if boundary_distill and (weight_assign_model is None or boundary_model is None):
        raise ValueError("Boundary KL needs weight_assign_model and boundary_model")

    sync_if_cuda(device)
    start = time.perf_counter()
    pseudo_teacher = copy.deepcopy(original).to(device).eval()
    pseudo_teacher.requires_grad_(False)
    student = copy.deepcopy(weight_assign_model if boundary_distill else original).to(device)
    retain_teacher = pseudo_teacher
    if boundary_distill:
        retain_teacher = copy.deepcopy(boundary_model).to(device).eval()
        retain_teacher.requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=momentum)
    retain_iterator = iter(retain_loader) if distill else None
    seen = crossed = steps = 0
    weight_sum = epsilon_sum = 0.0
    history = []
    samples = {key: [] for key in ("epsilon", "weight", "label", "pseudo_label", "epoch")}

    for epoch in range(epochs):
        for batch_index, (images, labels) in enumerate(forget_loader):
            if train_limit_reached(batch_index, max_batches_per_epoch):
                break
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            targets, epsilons = find_boundary_epsilon(
                images, labels, pseudo_teacher, bound=bound,
                epsilon_step=epsilon_step, cifar_mean=cifar_mean, cifar_std=cifar_std,
            )
            weights = (weight_assign_from_epsilon(epsilons, bound) if weighting_mode == "distance"
                       else torch.ones_like(epsilons))
            _student_train_mode(student, bn_stats_mode)
            optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            loss = (F.cross_entropy(logits, targets, reduction="none") * weights).mean()
            forget_ce = loss
            retain_kl = logits.new_zeros(())
            if distill:
                try:
                    retain_images, _ = next(retain_iterator)
                except StopIteration:
                    retain_iterator = iter(retain_loader)
                    retain_images, _ = next(retain_iterator)
                retain_images = retain_images.to(device, non_blocking=pin_memory)
                retain_kl = retain_distillation_loss(
                    student, retain_teacher, retain_images, temperature=temperature,
                )
                loss = loss + kl_lambda * retain_kl
            loss.backward()
            optimizer.step()
            steps += 1
            seen += labels.numel()
            crossed += targets.ne(labels).sum().item()
            weight_sum += weights.sum().item()
            epsilon_sum += epsilons.sum().item()
            if diagnostics_dir is not None:
                history.append({"epoch": epoch + 1, "step": steps, "batch_size": labels.numel(),
                                "forget_ce": forget_ce.detach().item(), "retain_kl": retain_kl.detach().item(),
                                "scaled_retain_kl": kl_lambda * retain_kl.detach().item(),
                                "total_loss": loss.detach().item()})
                for key, value in (("epsilon", epsilons), ("weight", weights), ("label", labels),
                                   ("pseudo_label", targets), ("epoch", torch.full_like(labels, epoch + 1))):
                    samples[key].append(value.detach().cpu().numpy())
        print(f"[{method}] Epoch {epoch + 1}/{epochs}")

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    save_state_dict(student, checkpoint_path)
    dependencies = [DISTANCE_METHOD, "boundary_shrink"] if boundary_distill else []
    dependency_stats = dependency_stats or {}
    costs_known = all(name in dependency_stats for name in dependencies)
    stage_stats = [dependency_stats[name] for name in dependencies] if costs_known else []
    total_seconds = elapsed + sum(s["seconds"] for s in stage_stats) if costs_known else None
    total_steps = steps + sum(s["update_steps"] for s in stage_stats) if costs_known else None
    initializer_stats = dependency_stats.get(DISTANCE_METHOD) if boundary_distill else None
    student_steps = (steps + initializer_stats["update_steps"] if initializer_stats else
                     None if boundary_distill else steps)
    student_epochs = (epochs + initializer_stats["student_epochs"] if initializer_stats else
                      None if boundary_distill else epochs)
    stats = {
        "seconds": elapsed, "total_seconds": total_seconds,
        "update_steps": steps, "student_update_steps": student_steps,
        "total_update_steps": total_steps, "sample_seen": seen,
        "cross_rate": crossed / seen if seen else float("nan"),
        "mean_sample_weight": weight_sum / seen if seen else float("nan"),
        "mean_boundary_epsilon": epsilon_sum / seen if seen else float("nan"),
        "initializer": DISTANCE_METHOD if boundary_distill else "original",
        "pseudo_teacher": "original",
        "retain_teacher": "boundary_shrink" if boundary_distill else "original" if distill else None,
        "dependencies": dependencies,
        "stage_epochs": epochs,
        "student_epochs": student_epochs,
        "epsilon_step": epsilon_step, "bound": bound,
        "normalization_mode": "matched",
        "retain_kl_lambda": kl_lambda if distill else 0.0,
        "retain_kl_temperature": temperature if distill else None,
        "bn_stats_mode": bn_stats_mode,
        "weighting_mode": weighting_mode,
    }
    stats.update(_save_diagnostics(diagnostics_dir, history, samples))
    print(f"[{method}] Stage: {elapsed:.2f}s | Updates: {steps} | Checkpoint: {checkpoint_path}")
    return student, stats


def train_limit_reached(batch_index, max_batches_per_epoch):
    return max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch


def run_boundary_shrink(original, forget_loader, checkpoint_path, *, device,
                        epochs=10, learning_rate=1e-5, momentum=0.9,
                        pin_memory=False, max_batches_per_epoch=None,
                        bound=0.1, normalization_mode="repo_exact", extra_exp=None,
                        curvature_lambda=0.7, reweight_bias=-0.5, reweight_slope=5.0,
                        cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD,
                        attack_mean=REPO_ATTACK_MEAN, attack_std=REPO_ATTACK_STD,
                        bn_stats_mode="update", diagnostics_dir=None):
    """Plain/author-enhanced Shrink; optionally use frozen BN running statistics."""
    _validate_bn_mode(bn_stats_mode)
    sync_if_cuda(device)
    start = time.perf_counter()
    teacher = copy.deepcopy(original).to(device)
    teacher.eval()

    unlearned = copy.deepcopy(original).to(device)

    optimizer = torch.optim.SGD(unlearned.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    crossed = 0
    seen = 0
    update_steps = 0
    history = []

    for epoch in range(epochs):
        for batch_index, (x, y) in enumerate(forget_loader):
            if train_limit_reached(batch_index, max_batches_per_epoch):
                break
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            x_adv = fgsm_cifar(
                x, y, teacher, bound=bound, normalization_mode=normalization_mode,
                cifar_mean=cifar_mean, cifar_std=cifar_std,
                attack_mean=attack_mean, attack_std=attack_std,
            )

            with torch.no_grad():
                adv_logits = teacher(x_adv)
                pseudo_label = adv_logits.argmax(dim=1)

            crossed += (pseudo_label != y).sum().item()
            seen += y.numel()

            _student_train_mode(unlearned, bn_stats_mode)
            optimizer.zero_grad(set_to_none=True)
            logits = unlearned(x)
            ori_loss = criterion(logits, pseudo_label)
            if extra_exp == "curv":
                ori_curv = curvature(original, x, y, h=0.9)[1]
                cur_curv = curvature(unlearned, x, y, h=0.9)[1]
                loss = ori_loss + curvature_lambda * torch.norm(ori_curv - cur_curv, p=2)
            elif extra_exp == "weight_assign":
                weight = weight_assign(adv_logits, pseudo_label, bias=reweight_bias, slope=reweight_slope)
                loss = (F.cross_entropy(logits, pseudo_label, reduction="none") * weight).mean()
            else:
                loss = ori_loss
            loss.backward()
            optimizer.step()
            update_steps += 1
            if diagnostics_dir is not None:
                history.append({"epoch": epoch + 1, "step": update_steps, "batch_size": y.numel(),
                                "forget_ce": ori_loss.detach().item(), "retain_kl": 0.0,
                                "scaled_retain_kl": 0.0, "total_loss": loss.detach().item()})

        print(f"[boundary_shrink] Epoch {epoch + 1}/{epochs}")

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    save_state_dict(unlearned, checkpoint_path)
    stats = {
        "seconds": elapsed,
        "cross_rate": crossed / seen if seen else float("nan"),
        "update_steps": update_steps,
        "sample_seen": seen,
        "bn_stats_mode": bn_stats_mode,
    }
    stats.update(_save_diagnostics(diagnostics_dir, history))
    print(f"[boundary_shrink] Training time: {elapsed:.2f}s")
    print(f"[boundary_shrink] Boundary crossing rate: {stats['cross_rate']:.4f}")
    print(f"[boundary_shrink] Update steps: {update_steps}")
    print(f"[boundary_shrink] Samples seen: {seen}")
    print(f"[boundary_shrink] Checkpoint saved: {checkpoint_path}")
    return unlearned, stats


def run_boundary_expanding(original, forget_loader, checkpoint_path, *, device,
                           epochs=10, learning_rate=1e-5, momentum=0.9,
                           num_classes=10, pin_memory=False,
                           max_batches_per_epoch=None):
    sync_if_cuda(device)
    start = time.perf_counter()

    # --------------------------------------------------------
    # 1. COPY FEATURE EXTRACTOR
    # --------------------------------------------------------
    features = copy.deepcopy(original.features).to(device)
    features.eval()

    original_classifier = original.classifier[0]
    feature_width = original_classifier.in_features

    # --------------------------------------------------------
    # 2. EXPAND CLASSIFIER: NUM_CLASSES -> NUM_CLASSES + 1
    # --------------------------------------------------------
    widened_classifier = nn.Linear(
        feature_width,
        num_classes + 1
    ).to(device)

    init_linear_like_repo(widened_classifier)

    with torch.no_grad():
        widened_classifier.weight[:num_classes].copy_(
            original_classifier.weight
        )
        widened_classifier.bias[:num_classes].copy_(
            original_classifier.bias
        )

    widened_model = nn.Sequential(
        features,
        widened_classifier
    ).to(device)

    # --------------------------------------------------------
    # 3. TRAIN ON FORGET SET WITH SHADOW LABEL
    # --------------------------------------------------------
    optimizer = torch.optim.SGD(
        widened_model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    criterion = nn.CrossEntropyLoss()
    update_steps = 0

    for epoch in range(epochs):

        for batch_index, (x, y) in enumerate(forget_loader):
            if train_limit_reached(batch_index, max_batches_per_epoch):
                break

            x = x.to(device, non_blocking=pin_memory)

            shadow_label = torch.full(
                (x.shape[0],),
                num_classes,
                dtype=torch.long,
                device=device
            )


            logits = widened_model(x)
            widened_model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, shadow_label)

            loss.backward()
            optimizer.step()

            update_steps += 1


        print(f"[boundary_expanding] Epoch {epoch + 1}/{epochs} ")

    # --------------------------------------------------------
    # 4. PRUNE SHADOW CLASS
    # --------------------------------------------------------
    pruned_classifier = nn.Linear(
        feature_width,
        num_classes
    ).to(device)

    with torch.no_grad():
        pruned_classifier.weight.copy_(
            widened_classifier.weight[:num_classes]
        )
        pruned_classifier.bias.copy_(
            widened_classifier.bias[:num_classes]
        )

    pruned_model = nn.Sequential(
        features,
        pruned_classifier
    ).to(device)

    # --------------------------------------------------------
    # 5. SAVE CHECKPOINTS
    # --------------------------------------------------------

    widened_checkpoint = checkpoint_path.with_name(
        checkpoint_path.stem + "_widened.pt"
    )

    save_state_dict(widened_model, widened_checkpoint)
    save_state_dict(pruned_model, checkpoint_path)

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    stats = {
        "seconds": elapsed,
        "update_steps": update_steps,
    }

    print(f"[boundary_expanding] Training time: {elapsed:.2f}s")
    print(f"[boundary_expanding] Update steps: {update_steps}")
    print(f"[boundary_expanding] Widened checkpoint saved: {widened_checkpoint}")
    print(f"[boundary_expanding] Pruned checkpoint saved: {checkpoint_path}")

    return pruned_model, stats
