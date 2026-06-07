"""
src/utils/config.py

Config loader — reads configs/config.yaml and returns a dot-accessible dict.

WHY THIS PATTERN:
  Every module does `from src.utils.config import cfg` and accesses
  cfg.data.ticker, cfg.training.wfv_start_year etc.
  No module ever imports another module's constants.
  Changing a parameter = editing config.yaml only.

LESSON FOR ANY PROJECT:
  Use this exact pattern for any project. The only thing that changes
  is the structure of your config.yaml.
"""

import yaml
from pathlib import Path


class DotDict(dict):
    """
    A dict subclass that allows attribute-style access.
    cfg['data']['ticker']  →  cfg.data.ticker

    This makes config access clean and IDE-friendly
    (you get autocomplete hints).
    """
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str = None) -> DotDict:
    """
    Load config.yaml and return as a DotDict.

    Args:
        path: Path to config file. Defaults to configs/config.yaml
              relative to the project root.

    Returns:
        DotDict with dot-accessible nested config values.

    Usage:
        from src.utils.config import load_config
        cfg = load_config()
        print(cfg.data.ticker)       # "SPY"
        print(cfg.training.models.random_forest.n_estimators)  # 300
    """
    if path is None:
        # Walk up from this file to find project root (contains configs/)
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                path = str(candidate)
                break
        if path is None:
            raise FileNotFoundError("Could not find configs/config.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return DotDict(raw)


# Module-level singleton — import this directly in other modules
# Usage: from src.utils.config import cfg
cfg = load_config()
