"""Author losses plus distance weighting and retain distillation."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _find_z(model, inputs, targets, h):
    inputs.requires_grad_()
    # Preserve the author's forward calls, including BatchNorm updates.
    outputs = model(inputs)
    loss_z = nn.CrossEntropyLoss()(model(inputs), targets)
    loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.0
    z = h * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
    inputs.grad.zero_()
    model.zero_grad()
    return z, norm_grad


def curvature(model, inputs, targets, h=3.0, lambda_=4):
    z, norm_grad = _find_z(model, inputs, targets, h)
    inputs.requires_grad_()
    outputs_pos = model(inputs + z)
    outputs_orig = model(inputs)
    loss_pos = nn.CrossEntropyLoss()(outputs_pos, targets)
    loss_orig = nn.CrossEntropyLoss()(outputs_orig, targets)
    grad_diff = torch.autograd.grad(loss_pos - loss_orig, inputs, create_graph=True)[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    model.zero_grad()
    return torch.sum(lambda_ * reg) / float(inputs.size(0)), reg


def PM(logit, target):
    # Allocate on the logits' device instead of hard-coding CUDA.
    eye = torch.eye(logit.shape[1], device=logit.device)
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest=True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1, 1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:, 0]).detach()
    return probs_2nd - probs_GT


def weight_assign(logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    return reweight * 3


def weight_assign_from_epsilon(epsilons, max_epsilon):
    """Distance weights from the new notebook, normalized to batch mean one."""
    if not math.isfinite(max_epsilon) or max_epsilon <= 0:
        raise ValueError("max_epsilon must be finite and positive")
    raw = (1 - epsilons / max_epsilon).clamp(min=0.1, max=1.0)
    return (raw / raw.mean().clamp(min=1e-8)).detach()


def retain_distillation_loss(student, teacher, images, *, temperature=2.0):
    """T² KL(teacher || student); freeze BN only during the retain forward.

    Dropout retains the student's training mode, matching the source notebook.
    Restore each BN mode afterwards so the next forget forward can update it.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    with torch.no_grad():
        teacher_logits = teacher(images)
    bn_modes = [(m, m.training) for m in student.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in bn_modes:
            module.eval()
        student_logits = student(images)
    finally:
        for module, training in bn_modes:
            module.train(training)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature ** 2
