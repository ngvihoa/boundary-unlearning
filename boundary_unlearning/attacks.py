"""Author FGSM and the distance search from the extended notebook."""

import math

import torch
import torch.nn.functional as F

from .constants import CIFAR_MEAN, CIFAR_STD, REPO_ATTACK_MEAN, REPO_ATTACK_STD


def fgsm_cifar(x_norm, y, model, bound=0.1, normalization_mode="repo_exact", *,
               cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD,
               attack_mean=REPO_ATTACK_MEAN, attack_std=REPO_ATTACK_STD):
    if normalization_mode == "matched":
        mean_values, std_values = cifar_mean, cifar_std
    elif normalization_mode == "repo_exact":
        mean_values, std_values = attack_mean, attack_std
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
    mean = torch.tensor(mean_values, dtype=x_norm.dtype, device=x_norm.device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, dtype=x_norm.dtype, device=x_norm.device).view(1, 3, 1, 1)
    x_natural_raw = x_norm.detach() * std + mean
    x_leaf = x_norm.detach().clone().requires_grad_(True)
    model.eval()
    logits = model(x_leaf)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_leaf, only_inputs=True)[0]

    x_adv_raw = x_leaf.detach() * std + mean + bound * grad.sign()
    delta = (x_adv_raw - x_natural_raw).clamp(-bound, bound)
    x_adv_raw = (x_natural_raw + delta).clamp(0.0, 1.0)
    x_adv_raw = torch.round(x_adv_raw * 255.0) / 255.0
    x_adv = (x_adv_raw - mean) / std
    return x_adv.detach()


def find_boundary_epsilon(images, labels, teacher, *, bound=0.1,
                          epsilon_step=1 / 255, cifar_mean=CIFAR_MEAN,
                          cifar_std=CIFAR_STD):
    """First wrong prediction along one fixed FGSM direction, in pixel units.

    As in no-refactor-new.ipynb, this search uses the *input* normalization.
    Already-wrong samples have epsilon zero; samples that never cross use the
    bound and the final prediction. This is a directional estimate of distance.
    """
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    if not math.isfinite(epsilon_step) or epsilon_step <= 0:
        raise ValueError("epsilon_step must be finite and positive")
    mean = images.new_tensor(cifar_mean).view(1, 3, 1, 1)
    std = images.new_tensor(cifar_std).view(1, 3, 1, 1)
    teacher.eval()
    with torch.enable_grad():
        leaf = images.detach().clone().requires_grad_(True)
        logits = teacher(leaf)
        predictions = logits.argmax(dim=1)
        gradient = torch.autograd.grad(F.cross_entropy(logits, labels), leaf)[0]
    direction = gradient.sign().detach()
    pixels = (images.detach() * std + mean).clamp(0, 1)
    crossed = predictions.ne(labels)
    epsilons = images.new_zeros(labels.shape)
    pseudo_labels = predictions.detach().clone()

    with torch.no_grad():
        for step in range(1, math.ceil(bound / epsilon_step) + 1):
            if crossed.all():
                break
            epsilon = min(step * epsilon_step, bound)
            candidate = (pixels + epsilon * direction).clamp(0, 1)
            candidate = torch.round(candidate * 255) / 255
            predictions = teacher((candidate - mean) / std).argmax(dim=1)
            changed = predictions.ne(labels)
            first_cross = ~crossed & changed
            epsilons[first_cross] = epsilon
            pseudo_labels[first_cross] = predictions[first_cross]
            crossed |= changed

    epsilons[~crossed] = bound
    pseudo_labels[~crossed] = predictions[~crossed]
    return pseudo_labels.detach(), epsilons.detach()
