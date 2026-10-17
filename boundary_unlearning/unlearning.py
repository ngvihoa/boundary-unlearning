"""Boundary Shrink and Boundary Expanding with explicit runtime settings."""

import copy
import time

import torch
import torch.nn.functional as F
from torch import nn

from .attacks import fgsm_cifar
from .constants import CIFAR_MEAN, CIFAR_STD, REPO_ATTACK_MEAN, REPO_ATTACK_STD
from .enhanced import curvature, weight_assign
from .models import init_linear_like_repo
from .training import save_state_dict, sync_if_cuda


def train_limit_reached(batch_index, max_batches_per_epoch):
    return max_batches_per_epoch is not None and batch_index >= max_batches_per_epoch


def run_boundary_shrink(original, forget_loader, checkpoint_path, *, device,
                        epochs=10, learning_rate=1e-5, momentum=0.9,
                        pin_memory=False, max_batches_per_epoch=None,
                        bound=0.1, normalization_mode="repo_exact", extra_exp=None,
                        curvature_lambda=0.7, reweight_bias=-0.5, reweight_slope=5.0,
                        cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD,
                        attack_mean=REPO_ATTACK_MEAN, attack_std=REPO_ATTACK_STD):
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

            unlearned.train()
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

        print(f"[boundary_shrink] Epoch {epoch + 1}/{epochs}")

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    save_state_dict(unlearned, checkpoint_path)
    stats = {
        "seconds": elapsed,
        "cross_rate": crossed / seen if seen else float("nan"),
        "update_steps": update_steps,
        "sample_seen": seen,
    }
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
