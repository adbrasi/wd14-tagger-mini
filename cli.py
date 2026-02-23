#!/usr/bin/env python3
"""Interactive CLI for video/image tagging pipeline.

Handles venv setup, dependency installation, and provides a menu-driven
interface for processing video datasets for LoRA training.
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")
TAGGER_SCRIPT = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")


def print_banner():
    print("\n" + "=" * 60)
    print("  VIDEO TAGGER FOR LORA TRAINING")
    print("  wd14 / camie / pixai / grok pipeline")
    print("=" * 60 + "\n")


def ensure_venv():
    """Create and activate venv if needed."""
    python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.exists(python):
        return python

    print("[*] Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    print("[+] venv created at", VENV_DIR)
    return python


def install_deps(python: str):
    """Install/update requirements."""
    pip = os.path.join(VENV_DIR, "bin", "pip")
    print("[*] Installing dependencies...")
    result = subprocess.run(
        [pip, "install", "-q", "-r", REQUIREMENTS],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[!] pip install failed:")
        print(result.stderr)
        sys.exit(1)
    print("[+] Dependencies installed.\n")


def ask_input(prompt: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "
    value = input(display).strip()
    return value if value else default


def ask_choice(prompt: str, options: list, default: int = 1) -> int:
    """Display numbered options and get user choice."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " *" if i == default else ""
        print(f"  {i}) {opt}{marker}")
    while True:
        raw = input(f"\nChoice [{default}]: ").strip()
        if not raw:
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "sim", "s")


def check_env_key(name: str) -> str:
    """Check for env var and return its value."""
    return os.environ.get(name, "")


def main():
    print_banner()

    # Setup
    python = ensure_venv()
    install_deps(python)

    # Input directory
    input_dir = ask_input("Input directory (path to videos/images)")
    if not input_dir or not os.path.exists(input_dir):
        print(f"[!] Directory not found: {input_dir}")
        sys.exit(1)

    # Mode selection
    mode = ask_choice("What are you processing?", [
        "Videos (extract frames and tag)",
        "Images (tag directly)",
    ], default=1)
    is_video = mode == 1

    # Tagger selection
    print("\nAvailable taggers:")
    tagger_options = [
        "pixai + grok (recommended for video LoRA)",
        "wd14 + pixai + grok (full pipeline)",
        "pixai only (fast, tags only)",
        "wd14 only",
        "grok only (needs API key)",
        "Custom (enter manually)",
    ]
    tagger_choice = ask_choice("Select tagger combination:", tagger_options, default=1)

    tagger_map = {
        1: "pixai,grok",
        2: "wd14,pixai,grok",
        3: "pixai",
        4: "wd14",
        5: "grok",
    }
    if tagger_choice == 6:
        taggers = ask_input("Enter taggers (comma-separated)", "pixai,grok")
    else:
        taggers = tagger_map[tagger_choice]

    has_grok = "grok" in taggers

    # Pro mode
    pro_mode = False
    if is_video:
        pro_mode = ask_yes_no("Enable PRO mode? (2 frames per video, better quality)", default=False)

    # Grok API key
    api_key = ""
    if has_grok:
        api_key = check_env_key("OPENROUTER_API_KEY")
        if api_key:
            print(f"[+] Found OPENROUTER_API_KEY in environment ({api_key[:8]}...)")
        else:
            api_key = ask_input("Enter OpenRouter API key (sk-or-...)")
            if not api_key:
                print("[!] No API key provided. Grok tagger will fail.")
                if not ask_yes_no("Continue anyway?", default=False):
                    sys.exit(1)

    # HF token (for pixai gated repo)
    hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
    if "pixai" in taggers and not hf_token:
        print("\n[!] PixAI model is gated. You may need a HuggingFace token.")
        hf_token = ask_input("Enter HF token (or press Enter to skip)", "")

    # Batch size
    batch_size = ask_input("Batch size for taggers", "4")

    # Grok concurrency
    grok_concurrency = "8"
    if has_grok:
        grok_concurrency = ask_input("Grok API concurrency (parallel requests)", "8")

    # Recursive
    recursive = ask_yes_no("Search subdirectories recursively?", default=True)

    # Force reprocess
    force = ask_yes_no("Force reprocess already-processed files?", default=False)

    # Build command
    cmd = [python, TAGGER_SCRIPT, input_dir]
    cmd.extend(["--taggers", taggers])
    cmd.extend(["--batch_size", batch_size])

    if is_video:
        cmd.append("--video")
    if pro_mode:
        cmd.append("--pro")
    if recursive:
        cmd.append("--recursive")
    if force:
        cmd.append("--force")

    cmd.append("--remove_underscore")

    if has_grok and api_key:
        cmd.extend(["--grok_api_key", api_key])
        cmd.extend(["--grok_concurrency", grok_concurrency])

    if hf_token:
        cmd.extend(["--hf_token", hf_token])

    # Thresholds
    cmd.extend(["--thresh", "0.30"])

    # Summary
    print("\n" + "-" * 60)
    print("CONFIGURATION SUMMARY")
    print("-" * 60)
    print(f"  Input:        {input_dir}")
    print(f"  Mode:         {'video' if is_video else 'images'}{' (PRO)' if pro_mode else ''}")
    print(f"  Taggers:      {taggers}")
    print(f"  Batch size:   {batch_size}")
    if has_grok:
        print(f"  Grok model:   x-ai/grok-4.1-fast")
        print(f"  Concurrency:  {grok_concurrency}")
    print(f"  Recursive:    {recursive}")
    print(f"  Force:        {force}")
    print("-" * 60)

    if not ask_yes_no("\nStart processing?", default=True):
        print("Aborted.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  STARTING PIPELINE")
    print("=" * 60 + "\n")

    # Run
    env = os.environ.copy()
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
    if hf_token:
        env["HF_TOKEN"] = hf_token

    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("  DONE! Check your input directory for .txt files.")
            print("=" * 60 + "\n")
        else:
            print(f"\n[!] Process exited with code {result.returncode}")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
