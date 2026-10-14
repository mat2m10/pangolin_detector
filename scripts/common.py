"""Shared helpers for the train / evaluate / predict scripts."""

import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "pangolin" / "weights" / "best.pt"


def resolve_data_yaml(data_yaml: Path, require_splits=("train", "val")) -> Path:
    """Return a dataset YAML whose `path` is absolute.

    Ultralytics resolves a relative `path` against the current directory (or
    its own datasets dir), which breaks when the scripts are run from
    elsewhere. Relative paths in our config are relative to the repo root, so
    we rewrite them into a temporary copy. Also fail early, with a clear
    message, if the split folders have not been built yet.
    """
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if not cfg:
        raise SystemExit(f"Dataset config is empty: {data_yaml}")

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    cfg["path"] = str(root)

    missing = [
        f"{split}: {root / cfg[split]}"
        for split in require_splits
        if not (root / cfg.get(split, "")).is_dir()
    ]
    if missing:
        raise SystemExit(
            "Dataset split folders not found:\n  "
            + "\n  ".join(missing)
            + "\nBuild them first with notebooks/data_exploration.ipynb."
        )

    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="pangolin_data_", delete=False, encoding="utf-8"
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return Path(tmp.name)
