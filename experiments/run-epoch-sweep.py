"""Run the Phase 1 epoch comparison in an existing dataset session."""

import importlib
import os
from pathlib import Path
import shutil

import pandas as pd
from IPython.display import display, Image, FileLink
from boundary_unlearning import unlearning, phases, diagnostics
from boundary_unlearning.artifacts import export_results
from boundary_unlearning.reporting import MODEL_NAMES, results_table

for sweep_module in (unlearning, phases, diagnostics):
    importlib.reload(sweep_module)

sweep_source = globals().get("EPOCH_SWEEP_SOURCE") or globals().get("phase1_manifest")
if sweep_source is None:
    raise RuntimeError("Run Phase 1 evaluation first, or set EPOCH_SWEEP_SOURCE to the completed Phase 1 bundle/ZIP.")
sweep_missing = [name for name in ("train_dataset", "evaluation_dataset", "OUTPUT_ROOT", "DEVICE") if name not in globals()]
if sweep_missing:
    raise RuntimeError(f"Run the Phase 1 dataset setup cells first. Missing: {sweep_missing}")
epoch_sweep = diagnostics.run_phase1_epoch_sweep(
    sweep_source, train_dataset, evaluation_dataset, Path(OUTPUT_ROOT) / "result", device=DEVICE,
    epochs=globals().get("UNLEARN_EPOCHS_TO_TEST", [1, 3, 5, 10, 20]),
)
display(results_table(epoch_sweep["baseline_rows"], dataset_name="CIFAR-100"))
sweep_frame = pd.DataFrame(epoch_sweep["rows"])[[
    "model", "unlearn_epochs", "unlearning_update_steps", "train_retain_acc", "train_forget_acc",
    "test_retain_acc", "test_forget_acc", "unlearning_seconds",
]]
sweep_frame["model"] = sweep_frame["model"].map(MODEL_NAMES)
sweep_accuracy_columns = [name for name in sweep_frame if name.endswith("_acc")]
sweep_frame[sweep_accuracy_columns] *= 100
sweep_frame = sweep_frame.rename(columns={
    "model": "Method", "unlearn_epochs": "Epochs", "unlearning_update_steps": "Updates",
    "train_retain_acc": "Train retain (%)", "train_forget_acc": "Train forget (%)",
    "test_retain_acc": "Test retain (%)", "test_forget_acc": "Test forget (%)",
    "unlearning_seconds": "Unlearning (s)",
}).set_index(["Method", "Epochs"])
display(sweep_frame.style.format({name: "{:.2f}" for name in sweep_frame if name != "Updates"}, na_rep="—"))
display(Image(filename=str(epoch_sweep["output_dir"] / "epochs.png")))
print("Comparison folder:", epoch_sweep["output_dir"])
sweep_budget_archives = {}
for sweep_epochs, sweep_manifest in epoch_sweep["bundles"].items():
    print(f"Phase 2 source for {sweep_epochs} epochs: {sweep_manifest}")
    # One manifest per budget ZIP; keep these archives outside result/.
    sweep_budget_archives[sweep_epochs] = Path(shutil.make_archive(
        str(Path(OUTPUT_ROOT) / f"{sweep_manifest.parent.name}_bundle"), "zip",
        root_dir=sweep_manifest.parent.parent, base_dir=sweep_manifest.parent.name,
    ))
    display(FileLink(os.path.relpath(sweep_budget_archives[sweep_epochs]),
                     result_html_prefix=f"Download {sweep_epochs}-epoch bundle for Phase 2: "))
sweep_archive = export_results(Path(OUTPUT_ROOT) / "result")
print("Archive:", sweep_archive)
display(FileLink(os.path.relpath(sweep_archive), result_html_prefix="Download: "))
