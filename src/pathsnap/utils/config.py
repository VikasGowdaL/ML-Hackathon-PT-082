"""Configuration and path management for PathSnap.

Reads configs/paths.yaml and supports environment variable overrides
(e.g., RAW_DATA_DIR). Returns cross-platform pathlib.Path objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def get_repo_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    # This file is at src/pathsnap/utils/config.py -> parent.parent.parent.parent is repo root
    return Path(__file__).resolve().parent.parent.parent.parent


def load_paths_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load paths configuration from YAML file."""
    repo_root = get_repo_root()
    if config_path is None:
        cfg_file = repo_root / "configs" / "paths.yaml"
    else:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = repo_root / cfg_file

    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    return data


def get_path(key: str, default: str | None = None) -> Path:
    """Retrieve a path by key, prioritizing environment variables over configs/paths.yaml.

    Priority:
      1. Environment variable: KEY.upper() (e.g. RAW_DATA_DIR for raw_data_dir)
      2. Key in configs/paths.yaml
      3. Provided default value (or sensible fallback)

    Always returns an absolute pathlib.Path.
    """
    repo_root = get_repo_root()
    env_var_name = key.upper()
    env_val = os.environ.get(env_var_name)

    if env_val:
        p = Path(env_val)
        return p if p.is_absolute() else (repo_root / p).resolve()

    cfg = load_paths_config()
    val = cfg.get(key, default)
    if val is None:
        raise KeyError(f"Path key '{key}' not found in configs/paths.yaml or environment.")

    p = Path(val)
    return p if p.is_absolute() else (repo_root / p).resolve()


def get_raw_data_dir() -> Path:
    """Convenience accessor for the raw data directory."""
    return get_path("raw_data_dir", default="data/raw")
