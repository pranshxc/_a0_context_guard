"""
Central config reader for _a0_context_guard.
Reads from default_config.yaml, then allows per-value env-var override
with prefix CTX_GUARD_.  E.g.  CTX_GUARD_MAX_RESULT_CHARS=5000

Imported as: from plugins._a0_context_guard.helpers.config import get
"""
import os
import yaml
from functools import lru_cache

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "default_config.yaml")


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    try:
        with open(os.path.abspath(_CONFIG_PATH), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get(key: str, default=None):
    """Return config value, env-var CTX_GUARD_<KEY> overrides yaml."""
    env_val = os.environ.get(f"CTX_GUARD_{key.upper()}")
    if env_val is not None:
        yaml_val = _load_yaml().get(key, default)
        try:
            if isinstance(yaml_val, bool):
                return env_val.lower() in ("1", "true", "yes")
            if isinstance(yaml_val, int):
                return int(env_val)
            if isinstance(yaml_val, float):
                return float(env_val)
        except (ValueError, TypeError):
            pass
        return env_val
    return _load_yaml().get(key, default)


def is_enabled() -> bool:
    return bool(get("enabled", True))
