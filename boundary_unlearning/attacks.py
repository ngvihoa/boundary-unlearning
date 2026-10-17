"""FGSM with the author's normalization and projection order."""

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
