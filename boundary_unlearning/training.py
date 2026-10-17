"""Training, checkpoint selection and split accuracy evaluation."""

import time
from pathlib import Path

import torch
from torch import nn

from .models import AllCNN


def sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def accuracy(model, loader, *, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else float("nan")


def save_state_dict(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state_dict_into(model, path, *, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def train_from_scratch(train_loader, label, checkpoint_path, evaluation_loader, *,
                       device, num_classes=10, epochs=30, learning_rate=0.1,
                       momentum=0.9, max_batches_per_epoch=None):
    sync_if_cuda(device)
    start = time.perf_counter()
    model = AllCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / total if total else float("nan")
        epoch_acc = correct / total if total else float("nan")
        test_acc = accuracy(model, evaluation_loader, device=device)
        if test_acc >= best_acc:
            best_acc = test_acc
            save_state_dict(model, checkpoint_path)
        print(f"[{label}] Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Test: {test_acc:.4f}")

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    # As in the author code, return the final model while saving the best checkpoint.
    print(f"[{label}] Training time: {elapsed:.2f}s")
    print(f"[{label}] Best checkpoint saved: {checkpoint_path} | Test: {best_acc:.4f}")
    return model, elapsed


def create_or_load_model(label, loader, checkpoint_path, evaluation_loader, *,
                         device, reuse_checkpoints=False, num_classes=10,
                         epochs=30, learning_rate=0.1, momentum=0.9,
                         max_batches_per_epoch=None):
    checkpoint_path = Path(checkpoint_path)
    if reuse_checkpoints and checkpoint_path.exists():
        model = AllCNN(num_classes=num_classes).to(device)
        load_state_dict_into(model, checkpoint_path, device=device)
        print(f"[{label}] Loaded checkpoint: {checkpoint_path}")
        return model, None
    result = train_from_scratch(
        loader, label, checkpoint_path, evaluation_loader, device=device,
        num_classes=num_classes, epochs=epochs, learning_rate=learning_rate,
        momentum=momentum, max_batches_per_epoch=max_batches_per_epoch,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(f"train_from_scratch must return (model, elapsed), got {type(result)} with {len(result) if isinstance(result, tuple) else 'non-tuple'} result.")
    return result


def evaluate_model_splits(model, *, device, test_loader, test_retain_loader,
                          test_forget_loader, train_retain_eval_loader=None,
                          train_forget_eval_loader=None, include_train=True):
    result = {
        "test_all_acc": accuracy(model, test_loader, device=device),
        "test_retain_acc": accuracy(model, test_retain_loader, device=device),
        "test_forget_acc": accuracy(model, test_forget_loader, device=device),
    }

    if include_train:
        result["train_retain_acc"] = accuracy(model, train_retain_eval_loader, device=device)
        result["train_forget_acc"] = accuracy(model, train_forget_eval_loader, device=device)
    else:
        result["train_retain_acc"] = None
        result["train_forget_acc"] = None

    return result
