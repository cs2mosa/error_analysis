"""ea — the error-analysis package (python -m ea <cmd>)."""

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(None)
def load_cfg(path: str | None = None) -> dict:
    """Load config.yaml (GC-6: the single source of [PROPOSED] values)."""
    p = Path(path) if path else Path(__file__).parent / "config.yaml"
    return yaml.safe_load(open(p, encoding="utf-8"))


def model_ref(cfg) -> str:
    """Resolve the model to load: explicit path > HF_HOME snapshot > name."""
    import os
    if cfg["model"].get("path") and Path(cfg["model"]["path"]).exists():
        return cfg["model"]["path"]
    name = cfg["model"]["name"]
    local = Path(os.environ.get("HF_HOME", "")) / "hub" / f"models--{name.replace('/', '--')}"
    if local.exists():
        snaps = sorted((local / "snapshots").glob("*"))
        if snaps:
            return str(snaps[-1])
    return name
