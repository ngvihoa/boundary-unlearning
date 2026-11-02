"""Notebook tables built from the experiment's raw evaluation rows."""

import pandas as pd


MODEL_NAMES = {
    "original": "Original",
    "retrain": "Retrain",
    "boundary_shrink": "Boundary Shrink",
    "boundary_expanding": "Boundary Expanding",
    "boundary_shrink_weight_assign": "Shrink + distance weights",
    "boundary_shrink_weight_assign_distill": "Distance + KL (Original)",
    "boundary_shrink_weight_assign_boundary_distill": "Distance + KL (Shrink, warm start)",
}


def results_table(rows, *, dataset_name="CIFAR-10"):
    """Format a comparison table without modifying the values exported to CSV."""
    columns = {
        "train_retain_acc": ("Train", "Retain (%)"),
        "train_forget_acc": ("Train", "Forget (%)"),
        "test_all_acc": ("Test", "All (%)"),
        "test_retain_acc": ("Test", "Retain (%)"),
        "test_forget_acc": ("Test", "Forget (%)"),
        "training_seconds": ("Time", "Training (s)"),
        "unlearning_seconds": ("Time", "Unlearning (s)"),
        "fgsm_cross_rate": ("Attack", "FGSM cross (%)"),
    }
    frame = pd.DataFrame(rows).set_index("model").reindex(columns=columns)
    percent_columns = [name for name in columns if name.endswith("_acc") or name == "fgsm_cross_rate"]
    # Coerce all-missing smoke-profile columns to numeric before formatting.
    frame = frame.astype(float)
    frame[percent_columns] = frame[percent_columns] * 100
    frame = frame.rename(index=MODEL_NAMES).rename_axis("Model")
    frame.columns = pd.MultiIndex.from_tuples(columns.values())

    formats = {
        (group, label): "{:.1f}" if group == "Time" else "{:.2f}"
        for group, label in frame.columns
    }
    return (
        frame.style
        .format(formats, na_rep="—")
        .format_index(escape="html", axis=0)
        .set_caption(f"Model comparison · Official {dataset_name} test")
        .set_properties(**{"text-align": "right", "font-variant-numeric": "tabular-nums"})
        .set_table_styles([
            {"selector": "caption", "props": [("text-align", "left"), ("font-weight", "bold")]},
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "th.row_heading", "props": [("text-align", "left"), ("white-space", "nowrap")]},
            {"selector": "th, td", "props": [("padding", "6px 10px")]},
        ])
    )


def unlearning_details_table(rows):
    """Separate stage costs from prerequisites and student optimization budget."""
    columns = {
        "unlearning_seconds": "Stage (s)",
        "total_unlearning_seconds": "With prerequisites (s)",
        "unlearning_update_steps": "Stage updates",
        "student_update_steps": "Student updates",
        "total_update_steps": "With prerequisites updates",
        "mean_boundary_epsilon": "Mean epsilon",
        "mean_sample_weight": "Mean weight",
        "initializer": "Student starts from",
        "retain_teacher": "Retain teacher",
    }
    frame = pd.DataFrame([row for row in rows if row["model"] not in {"original", "retrain"}])
    frame = frame.set_index("model").reindex(columns=columns)
    for column in ("initializer", "retain_teacher"):
        frame[column] = frame[column].map(lambda value: MODEL_NAMES.get(value, value))
    frame = frame.rename(index=MODEL_NAMES, columns=columns).rename_axis("Method")
    return (frame.style
            .format({"Stage (s)": "{:.2f}", "With prerequisites (s)": "{:.2f}",
                     "Stage updates": "{:.0f}", "Student updates": "{:.0f}",
                     "With prerequisites updates": "{:.0f}",
                     "Mean epsilon": "{:.5f}", "Mean weight": "{:.3f}"}, na_rep="—")
            .format_index(escape="html", axis=0)
            .set_caption("Unlearning cost · excludes Original training; includes both prerequisite models")
            .set_properties(**{"text-align": "right"}))
