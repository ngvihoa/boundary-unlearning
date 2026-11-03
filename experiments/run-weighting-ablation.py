"""Compare distance vs uniform weights on frozen BN, inside the existing notebook session."""

import importlib
import os
from pathlib import Path
from functools import partial

import pandas as pd
from IPython.display import Image, FileLink, display
from boundary_unlearning import attacks, enhanced, unlearning, reporting, diagnostics, training
from boundary_unlearning.artifacts import export_results

for weight_module in (attacks, enhanced, unlearning, reporting, diagnostics):
    importlib.reload(weight_module)

weight_required = (
    "original_model", "retrain_model", "make_loader", "train_forget_set",
    "test_loader", "test_retain_loader", "test_forget_loader", "train_retain_eval_loader",
    "train_forget_eval_loader", "DEVICE", "UNLEARN_EPOCHS", "UNLEARN_LR", "MOMENTUM",
    "PIN_MEMORY", "MAX_BATCHES_PER_EPOCH", "FGSM_BOUND", "CIFAR_MEAN",
    "CIFAR_STD", "REPO_ATTACK_MEAN", "REPO_ATTACK_STD", "run_tag", "OUTPUT_ROOT", "SEED",
)
weight_missing = [name for name in weight_required if name not in globals()]
if weight_missing:
    raise RuntimeError(f"Use the existing trained notebook session. Missing objects: {weight_missing}")

weight_evaluate = partial(
    training.evaluate_model_splits, device=DEVICE,
    test_loader=test_loader, test_retain_loader=test_retain_loader, test_forget_loader=test_forget_loader,
    train_retain_eval_loader=train_retain_eval_loader, train_forget_eval_loader=train_forget_eval_loader,
    include_train=globals().get("EVALUATE_TRAIN_SPLITS", True),
)
weighting_ablation = diagnostics.run_weighting_ablation(
    original_model, retrain_model,
    lambda: make_loader(train_forget_set, shuffle=True), weight_evaluate,
    Path(OUTPUT_ROOT) / "result", device=DEVICE, source_run_tag=run_tag, seed=SEED,
    epochs=UNLEARN_EPOCHS, learning_rate=UNLEARN_LR, momentum=MOMENTUM,
    pin_memory=PIN_MEMORY, max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
    bound=FGSM_BOUND,
    epsilon_step=globals().get("WEIGHT_EPSILON_STEP", 1 / 255),
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
display(reporting.results_table(weighting_ablation["rows"], dataset_name=globals().get("DATASET_NAME", "CIFAR-10")))
weight_details = pd.DataFrame([
    {"model": name, **{key: value for key, value in stats.items()
                       if key.startswith(("bn_", "mean_")) or key in {"update_steps", "weighting_mode"}}}
    for name, stats in weighting_ablation["stats"].items()
]).set_index("model")
display(weight_details)
display(pd.DataFrame(weighting_ablation["comparison"]).set_index("model"))
print("Pair validation:", weighting_ablation["pair_validation"])
print("Applied weights:", weighting_ablation["applied_weights"])
for weight_image_name in ("applied_weights.png", "losses.png"):
    display(Image(filename=str(weighting_ablation["output_dir"] / weight_image_name)))

# Export directly: main-flow rows/config and the visualization cells need not be rerun.
weight_archive = export_results(Path(OUTPUT_ROOT) / "result", checkpoint_dir=globals().get("CHECKPOINT_ROOT"))
print("Weighting ablation folder:", weighting_ablation["output_dir"])
print("Archive:", weight_archive)
display(FileLink(os.path.relpath(weight_archive), result_html_prefix="Download: "))
