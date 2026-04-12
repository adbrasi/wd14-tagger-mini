"""Helpers for the interactive tagging wizard."""

from __future__ import annotations

from typing import Callable


def choose_processing_mode(
    *,
    image_count: int,
    video_count: int,
    ask_choice: Callable[[str, list[str], int], int],
) -> bool:
    """Return True for video mode, False for image mode."""
    mode = ask_choice(
        "What are you processing?",
        [
            f"Videos (extract frames and tag) — {video_count:,} found",
            f"Images (tag directly) — {image_count:,} found",
        ],
        default=1 if video_count > 0 else 2,
    )
    return mode == 1


def choose_pipeline(
    *,
    ask_choice: Callable[[str, list[str], int], int],
    ask_input: Callable[[str, str], str],
) -> tuple[int, str]:
    """Return (choice_id, raw tagger string) for the selected pipeline."""
    options = [
        "Auto captions: PixAI tags + LLM captions (recommended)",
        "Auto captions: WD14 + PixAI tags + LLM captions (more complete, slower)",
        "Tags only: PixAI (fastest local tagging)",
        "Tags only: WD14",
        "Captions only: LLM (uses image, and existing .txt if available)",
        "Tags only with manual prefix/trigger word",
        "Custom raw tagger list",
    ]
    choice = ask_choice("What kind of output do you want?", options, default=1)
    tagger_map = {
        1: "pixai,grok",
        2: "wd14,pixai,grok",
        3: "pixai",
        4: "wd14",
        5: "grok",
        6: "pixai,wd14",
    }
    if choice == 7:
        return choice, ask_input("Enter taggers (comma-separated)", "pixai,grok")
    return choice, tagger_map[choice]


def configure_caption_backend(
    *,
    has_llm: bool,
    is_video: bool,
    input_dir: str,
    resolve_default_xai_state_file: Callable[[str], str],
    ask_choice: Callable[[str, list[str], int], int],
    ask_yes_no: Callable[[str, bool], bool],
    ask_input: Callable[[str, str], str],
    ask_int: Callable[[str, int, int], int],
) -> dict:
    """Collect caption-backend settings for the wizard."""
    config = {
        "caption_provider": "openrouter",
        "xai_batch_action": "submit",
        "xai_batch_state_file": "",
        "xai_batch_submit_chunk": "1000",
        "xai_batch_page_size": "100",
        "xai_batch_no_image": False,
        "monitor_xai": False,
        "monitor_poll_seconds": "20",
    }
    if not has_llm:
        return config

    default_provider = 2 if is_video else 1
    provider_choice = ask_choice(
        "How should captions be generated?",
        [
            "OpenRouter API (real-time requests, best for smaller/test runs)",
            "xAI Batch API (background jobs, cheaper for larger datasets)",
        ],
        default=default_provider,
    )
    if provider_choice != 2:
        return config

    config["caption_provider"] = "xai-batch"
    action_choice = ask_choice(
        "What do you want to do with the xAI batch?",
        [
            "Submit new requests",
            "Only check batch status",
            "Collect completed results and write .txt files",
        ],
        default=1,
    )
    config["xai_batch_action"] = {1: "submit", 2: "status", 3: "collect"}[action_choice]
    config["xai_batch_state_file"] = ask_input(
        "Batch state file (.json)",
        resolve_default_xai_state_file(input_dir),
    )

    if config["xai_batch_action"] == "submit":
        send_images = ask_yes_no(
            "Include images in each request? (better captions, larger payload)",
            True,
        )
        config["xai_batch_no_image"] = not send_images
        config["xai_batch_submit_chunk"] = ask_input(
            "Requests per submit call",
            "500" if send_images else "5000",
        )
        config["monitor_xai"] = ask_yes_no("Monitor batch progress after submit?", True)
    elif config["xai_batch_action"] == "status":
        config["monitor_xai"] = ask_yes_no("Keep polling batch status until you stop it?", True)
    elif config["xai_batch_action"] == "collect":
        config["xai_batch_page_size"] = ask_input("Results page size for collect", "100")

    if config["monitor_xai"]:
        config["monitor_poll_seconds"] = str(
            ask_int("Monitor poll interval (seconds)", 20, 3)
        )

    return config


def ask_literal_prefix(
    *,
    has_llm: bool,
    prompt_vars: dict[str, str],
    pipeline_choice: int,
    ask_input: Callable[[str, str], str],
    print_info: Callable[[str], None],
    print_warning: Callable[[str], None],
) -> str:
    """Ask for an optional literal .txt prefix, distinct from prompt preset vars."""
    prepend_prompt = "Optional literal text to prepend to every saved .txt"
    if has_llm and prompt_vars:
        print_info("This preset already injects trigger text inside the caption prompt.")
        print_info("Use the prefix below only if you also want extra text written verbatim at the start of the saved .txt file.")
    elif pipeline_choice == 6:
        print_info("This legacy mode is intended for a manual trigger prefix on local tags.")

    while True:
        value = ask_input(f"{prepend_prompt} (leave empty to skip)", "")
        if value or pipeline_choice != 6:
            if value and any(value.strip().lower() == str(v).strip().lower() for v in prompt_vars.values()):
                print_warning("The literal .txt prefix matches a preset variable and may duplicate the trigger.")
            return value
        print_warning("This legacy mode needs a manual prefix/trigger word.")


def build_tagging_summary_rows(
    *,
    input_dir: str,
    is_video: bool,
    pro_mode: bool,
    pipeline_label: str,
    prepend_text: str,
    has_local_taggers: bool,
    batch_size: str,
    has_llm: bool,
    caption_provider: str,
    llm_model_display: str,
    needs_prompt_profile: bool,
    prompt_profile: str,
    prompt_vars: dict[str, str],
    llm_concurrency: str,
    xai_batch_action: str,
    xai_batch_state_file: str,
    recursive: bool,
    force: bool,
) -> list[tuple[str, str]]:
    """Build CONFIGURATION summary rows for the wizard."""
    caption_backend_label = {
        "openrouter": "OpenRouter API",
        "xai-batch": "xAI Batch API",
    }.get(caption_provider, caption_provider)

    rows = [
        ("Input", input_dir),
        ("Mode", f"{'video' if is_video else 'images'}{' (PRO)' if pro_mode else ''}"),
        ("Pipeline", pipeline_label),
    ]
    if prepend_text:
        rows.append(("Literal .txt prefix", prepend_text))
    if has_local_taggers:
        rows.append(("Batch size", batch_size))
    if has_llm:
        rows.append(("Caption backend", caption_backend_label))
        rows.append(("Model", llm_model_display))
        if needs_prompt_profile:
            rows.append(("Caption preset", prompt_profile))
            for key, value in sorted(prompt_vars.items()):
                rows.append((f"Preset var: {key}", value or "(empty)"))
        if caption_provider == "openrouter":
            rows.append(("Concurrency", llm_concurrency))
        else:
            rows.append(("Batch action", xai_batch_action))
            rows.append(("State file", xai_batch_state_file))
    rows.extend([
        ("Recursive", str(recursive)),
        ("Force", str(force)),
    ])
    return rows
