from pathlib import Path
import os

PLOTS_ROOT = Path("plots")

def ensure_dir(path: Path) -> Path:
    os.makedirs(path, exist_ok=True)
    return path

def plot_path(subdir: str, filename: str) -> Path:
    """
    Build a path like plots/subdir/filename.png and ensure the folder exists.
    """
    folder = ensure_dir(PLOTS_ROOT / subdir)
    return folder / filename
