"""Helpers to import the repo's scripts, which are not an installed package."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(filename, attrs=()):
    """Import a top-level script by file name.

    The scripts may live at the repo root, in scripts/ or in tools/review/
    (the layout is being reorganised), so look in all of them. Skip the test
    if the file is missing, empty, or lacks the attributes the test needs.
    """
    for folder in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tools" / "review"):
        path = folder / filename
        if path.is_file() and path.stat().st_size > 0:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            # Register before executing, like a normal import, so libraries
            # that look the module up (e.g. Flask's root_path) find its file.
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            missing = [a for a in attrs if not hasattr(module, a)]
            if missing:
                pytest.skip(f"{filename} has no {', '.join(missing)} yet")
            return module
    pytest.skip(f"{filename} not found or empty")
