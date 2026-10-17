"""Optional curvature and reweighting losses from the author."""

import torch
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
