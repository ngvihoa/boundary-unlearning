"""Collect experiment outputs and archive them for download."""

import shutil
from pathlib import Path


def export_results(result_dir, *, checkpoint_dir=None, files=()):
    """Collect existing outputs, including older paths, and return the ZIP path."""
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        destination = result_dir / "checkpoints"
        if checkpoint_dir != destination:
            shutil.copytree(checkpoint_dir, destination, dirs_exist_ok=True)

    for file in files:
        file = Path(file).resolve()
        if not file.is_file():
            raise FileNotFoundError(f"Result file not found: {file}")
        destination = result_dir / file.name
        if file != destination:
            shutil.copy2(file, destination)

    # Keep the archive outside result/ so a repeated export cannot include itself.
    return Path(shutil.make_archive(
        str(result_dir), "zip", root_dir=result_dir.parent, base_dir=result_dir.name,
    ))
