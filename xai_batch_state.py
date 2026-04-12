"""Shared helpers for xAI batch state file naming and persistence."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


def resolve_default_xai_state_file(input_dir: str) -> str:
    """Match the default state-file naming used by the tagger."""
    base_dir = os.path.abspath(input_dir)
    return resolve_xai_state_file_from_train_dir(base_dir)


def resolve_xai_state_file_from_train_dir(train_data_dir: str) -> str:
    """Build the default xAI batch state path from a dataset directory."""
    base_dir = os.path.abspath(train_data_dir)
    parent_dir = os.path.dirname(base_dir)
    dataset_name = os.path.basename(base_dir)
    key = hashlib.md5(base_dir.encode("utf-8")).hexdigest()[:10]
    return os.path.join(parent_dir, f".xai_batch_state_{dataset_name}_{key}.json")


def load_xai_state(path: str, *, logger=None) -> dict:
    """Load persisted xAI batch state, returning {} on missing/invalid file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        if logger is not None:
            logger.warning(f"could not read xai batch state from {path}: {e}")
        return {}


def save_xai_state(path: str, state: dict) -> None:
    """Persist xAI batch state as pretty JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def hash_jsonable(value: Any) -> str:
    """Stable SHA256 for nested JSON-serializable values."""
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def hydrate_prompt_settings_from_xai_state(args, state: dict[str, Any]) -> None:
    """Reuse saved prompt profile/vars for xAI status+collect flows when omitted."""
    if not state or getattr(args, "xai_batch_action", None) not in ("status", "collect"):
        return

    saved_profile = state.get("prompt_profile")
    current_profile = getattr(args, "prompt_profile", None)
    if saved_profile and current_profile in (None, "", "default"):
        args.prompt_profile = saved_profile

    saved_vars = state.get("prompt_vars")
    if isinstance(saved_vars, dict) and not getattr(args, "prompt_var", None):
        args.prompt_var = [f"{key}={value}" for key, value in sorted(saved_vars.items())]
        setattr(args, "_prompt_vars_cache", dict(saved_vars))
