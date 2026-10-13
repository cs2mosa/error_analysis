"""ea — the error-analysis package (python -m ea <cmd>)."""

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(None)
def load_cfg(path: str | None = None) -> dict:
    """Load config.yaml (GC-6: the single source of [PROPOSED] values)."""
    p = Path(path) if path else Path(__file__).parent / "config.yaml"
    return yaml.safe_load(open(p, encoding="utf-8"))


def model_ref(cfg, override=None) -> str:
    """Resolve the model to load (BUG-12 fix: the single implementation).

    Precedence: explicit override > MODEL_PATH env > cfg.model.path (if it
    exists) > $HF_HOME snapshot of cfg.model.name > the repo name itself.
    cli.py and selfcheck.py both import this; the duplicate _model_ref in
    selfcheck.py was removed."""
    import os
    if override:
        return override
    env_p = os.environ.get("MODEL_PATH")
    if env_p:
        ep = Path(env_p)
        if ep.exists() or (not env_p.startswith("/") and not env_p.startswith(".")
                           and ":" not in env_p):
            return env_p
    if cfg and cfg.get("model", {}).get("path"):
        p = Path(cfg["model"]["path"])
        if p.exists():
            return str(p)
    name = (cfg.get("model", {}).get("name") if cfg else None) or "Qwen/Qwen3-VL-4B-Instruct"
    hf_home = Path(os.environ.get("HF_HOME", ""))
    local = hf_home / "hub" / f"models--{name.replace('/', '--')}"
    if local.exists():
        snaps = sorted((local / "snapshots").glob("*"))
        if snaps:
            return str(snaps[-1])
    return name
