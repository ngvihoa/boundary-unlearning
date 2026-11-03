"""Execute this file inside an existing Kaggle notebook namespace (no training of baselines)."""

import importlib
import os
from pathlib import Path
from functools import partial

import pandas as pd
from IPython.display import Image, FileLink, display
from boundary_unlearning import attacks, enhanced, unlearning, reporting, diagnostics, training
from boundary_unlearning.artifacts import export_results

for bn_module in (attacks, enhanced, unlearning, reporting, diagnostics):
    importlib.reload(bn_module)

bn_required = (
    "original_model", "retrain_model", "make_loader", "train_forget_set", "train_retain_set",
    "test_loader", "test_retain_loader", "test_forget_loader", "train_retain_eval_loader",
    "train_forget_eval_loader", "DEVICE", "UNLEARN_EPOCHS", "UNLEARN_LR", "MOMENTUM",
    "PIN_MEMORY", "MAX_BATCHES_PER_EPOCH", "FGSM_BOUND", "ATTACK_NORMALIZATION", "CIFAR_MEAN",
    "CIFAR_STD", "REPO_ATTACK_MEAN", "REPO_ATTACK_STD", "run_tag", "OUTPUT_ROOT", "SEED",
)
bn_missing = [name for name in bn_required if name not in globals()]
if bn_missing:
    raise RuntimeError(f"Use the existing trained notebook session. Missing objects: {bn_missing}")

bn_evaluate = partial(
    training.evaluate_model_splits, device=DEVICE,
    test_loader=test_loader, test_retain_loader=test_retain_loader, test_forget_loader=test_forget_loader,
    train_retain_eval_loader=train_retain_eval_loader, train_forget_eval_loader=train_forget_eval_loader,
    include_train=globals().get("EVALUATE_TRAIN_SPLITS", True),
)
bn_ablation = diagnostics.run_batchnorm_ablation(
    original_model, retrain_model,
    lambda: make_loader(train_forget_set, shuffle=True), bn_evaluate,
    Path(OUTPUT_ROOT) / "result", device=DEVICE, source_run_tag=run_tag, seed=SEED,
    methods=globals().get("BN_ABLATION_METHODS", ["boundary_shrink"]),
    retain_loader_factory=lambda: make_loader(train_retain_set, shuffle=True),
    epochs=UNLEARN_EPOCHS, learning_rate=UNLEARN_LR, momentum=MOMENTUM,
    pin_memory=PIN_MEMORY, max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
    bound=FGSM_BOUND, normalization_mode=ATTACK_NORMALIZATION,
    epsilon_step=globals().get("WEIGHT_EPSILON_STEP", 1 / 255),
    kl_lambda=globals().get("RETAIN_KL_LAMBDA", .1), temperature=globals().get("RETAIN_KL_TEMPERATURE", 2.),
    cifar_mean=CIFAR_MEAN, cifar_std=CIFAR_STD, attack_mean=REPO_ATTACK_MEAN, attack_std=REPO_ATTACK_STD,
    dataset_name=globals().get("DATASET_NAME", "CIFAR-10"),
    baseline_training_seconds={"original": globals().get("original_train_seconds"),
                               "retrain": globals().get("retrain_train_seconds")},
    metadata={"forget_class": globals().get("FORGET_CLASS"),
              "class_names": list(globals().get("CIFAR_CLASSES", [])),
              "batch_size": globals().get("BATCH_SIZE"), "profile": globals().get("RUN_PROFILE"),
              "data_root": str(globals().get("DATA_ROOT", "")),
              "include_train": globals().get("EVALUATE_TRAIN_SPLITS", True)},
)
display(reporting.results_table(bn_ablation["rows"], dataset_name=globals().get("DATASET_NAME", "CIFAR-10")))
bn_details = pd.DataFrame([
    {"model": name, **{key: value for key, value in stats.items()
                       if key.startswith(("bn_", "mean_")) or key == "update_steps"}}
    for name, stats in bn_ablation["stats"].items()
]).set_index("model")
display(bn_details)
print("Original teacher boundary-search audit (one pass):", bn_ablation["audit"])
for bn_image_name in ("boundary_search_audit.png", "losses.png"):
    display(Image(filename=str(bn_ablation["output_dir"] / bn_image_name)))

# Export directly: main-flow rows/config and the visualization cells need not be rerun.
bn_archive = export_results(Path(OUTPUT_ROOT) / "result", checkpoint_dir=globals().get("CHECKPOINT_ROOT"))
print("BN ablation folder:", bn_ablation["output_dir"])
print("Archive:", bn_archive)
display(FileLink(os.path.relpath(bn_archive), result_html_prefix="Download: "))
