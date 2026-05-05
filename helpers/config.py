"""
Central config reader for _a0_context_guard.
Reads from default_config.yaml, then allows per-value override via env vars
with the prefix CTX_GUARD_.  E.g. CTX_GUARD_MAX_RESULT_CHARS=5000.
"""
import os
import yaml
from functools import lru_cache

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "default_config.yaml")


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get(key: str, default=None):
    """Return config value, with env-var override (CTX_GUARD_<KEY_UPPER>)."""
    env_key = f"CTX_GUARD_{key.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        # try to coerce to the same type as the yaml default
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
