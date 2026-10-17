"""Notebook tables built from the experiment's raw evaluation rows."""

import pandas as pd


def results_table(rows):
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
    model_names = {
        "original": "Original",
        "retrain": "Retrain",
        "boundary_shrink": "Boundary Shrink",
        "boundary_expanding": "Boundary Expanding",
    }
    frame = pd.DataFrame(rows).set_index("model").reindex(columns=columns)
    percent_columns = [name for name in columns if name.endswith("_acc") or name == "fgsm_cross_rate"]
    # Coerce all-missing smoke-profile columns to numeric before formatting.
    frame = frame.astype(float)
    frame[percent_columns] = frame[percent_columns] * 100
    frame = frame.rename(index=model_names).rename_axis("Model")
    frame.columns = pd.MultiIndex.from_tuples(columns.values())

    formats = {
        (group, label): "{:.1f}" if group == "Time" else "{:.2f}"
        for group, label in frame.columns
    }
    return (
        frame.style
        .format(formats, na_rep="—")
        .format_index(escape="html", axis=0)
        .set_caption("Model comparison · Official CIFAR-10 test")
        .set_properties(**{"text-align": "right", "font-variant-numeric": "tabular-nums"})
        .set_table_styles([
            {"selector": "caption", "props": [("text-align", "left"), ("font-weight", "bold")]},
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
            {"selector": "th.row_heading", "props": [("text-align", "left"), ("white-space", "nowrap")]},
            {"selector": "th, td", "props": [("padding", "6px 10px")]},
        ])
    )
