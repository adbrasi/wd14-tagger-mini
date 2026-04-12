"""Helpers for prompt-profile discovery and interactive variable collection."""

from __future__ import annotations

import json
import os
from typing import Callable

PROMPT_PROFILE_METADATA_FILE = "profile.json"


def list_prompt_profiles(prompts_dir: str, mode: str) -> list[str]:
    """List available prompt profiles for a mode (image/video)."""
    mode_dir = os.path.join(prompts_dir, mode)
    if not os.path.isdir(mode_dir):
        return ["default"]

    profiles = []
    for entry in os.listdir(mode_dir):
        profile_dir = os.path.join(mode_dir, entry)
        if os.path.isdir(profile_dir) and os.path.exists(
            os.path.join(profile_dir, "system_prompt.md")
        ):
            profiles.append(entry)

    if not profiles:
        return ["default"]

    profiles.sort()
    if "default" in profiles:
        profiles.remove("default")
        profiles.insert(0, "default")
    return profiles


def load_prompt_profile_metadata(prompts_dir: str, mode: str, profile: str) -> dict:
    """Load optional profile metadata from prompts/<mode>/<profile>/profile.json."""
    meta_path = os.path.join(prompts_dir, mode, profile, PROMPT_PROFILE_METADATA_FILE)
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else {}


def build_prompt_profile_options(prompts_dir: str, mode: str, profiles: list[str]) -> list[str]:
    """Build human-friendly prompt profile menu labels."""
    options = []
    for profile in profiles:
        meta = load_prompt_profile_metadata(prompts_dir, mode, profile)
        label = meta.get("display_name") or profile
        description = meta.get("description")
        if description:
            label = f"{label} — {description}"
        options.append(label)
    return options


def collect_prompt_profile_values(
    prompts_dir: str,
    mode: str,
    profile: str,
    *,
    ask_input: Callable[[str, str], str],
    warn: Callable[[str], None],
) -> dict[str, str]:
    """Ask the user for any variables declared by the selected prompt profile."""
    meta = load_prompt_profile_metadata(prompts_dir, mode, profile)
    raw_variables = meta.get("variables")
    if not isinstance(raw_variables, list):
        return {}

    values: dict[str, str] = {}
    for entry in raw_variables:
        if not isinstance(entry, dict):
            continue

        name = str(entry.get("name", "")).strip()
        if not name:
            continue

        prompt_text = str(entry.get("prompt") or entry.get("label") or name).strip()
        default_value = str(entry.get("default", ""))
        required = bool(entry.get("required", True))

        while True:
            value = ask_input(prompt_text, default_value)
            if value or not required:
                values[name] = value
                break
            warn(f"'{prompt_text}' is required for this preset.")

    return values


def prompt_profile_has_declared_variables(prompts_dir: str, mode: str, profile: str) -> bool:
    """Return True when a prompt profile declares interactive variables."""
    meta = load_prompt_profile_metadata(prompts_dir, mode, profile)
    return isinstance(meta.get("variables"), list) and bool(meta.get("variables"))


def describe_prompt_vars(values: dict[str, str]) -> str:
    """Build a compact, human-readable summary for prompt preset variables."""
    if not values:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
