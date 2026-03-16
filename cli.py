#!/usr/bin/env python3
"""Interactive CLI for data_araknideo.

Handles venv setup, dependency installation, and provides a menu-driven
interface for dataset preprocessing and tagging.
"""
import os
import re
import subprocess
import sys
import time
import json
import hashlib
import zipfile

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")
TAGGER_SCRIPT = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
XAI_API_BASE_URL = "https://api.x.ai"
XAI_BATCH_DEFAULT_MODEL = "grok-4-1-fast-non-reasoning"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}

# HuggingFace URL / ID detection
_HF_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
    r"(?:/tree/[^/]+/(.+))?"
)
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def print_banner():
    print("\n" + "=" * 60)
    print("  DATA ARAKNIDEO")
    print("  dataset preprocessing & tagging pipeline")
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
    """Ask a yes/no question. Accepts y/yes or n/no (case-insensitive)."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return default


def check_env_key(name: str) -> str:
    """Check for env var and return its value."""
    return os.environ.get(name, "")


def resolve_default_xai_state_file(input_dir: str) -> str:
    """Match the default state-file naming used by tag_images_by_wd14_tagger.py."""
    base_dir = os.path.abspath(input_dir)
    parent_dir = os.path.dirname(base_dir)
    dataset_name = os.path.basename(base_dir)
    key = hashlib.md5(base_dir.encode("utf-8")).hexdigest()[:10]
    return os.path.join(parent_dir, f".xai_batch_state_{dataset_name}_{key}.json")


def fetch_xai_batch_status(
    batch_id: str,
    api_key: str,
    base_url: str = XAI_API_BASE_URL,
    max_retries: int = 5,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url}/v1/batches/{batch_id}"

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=120)

            # Retry transient throttling/server errors.
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= max_retries:
                    resp.raise_for_status()
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else None
            # Non-retriable client errors (except 429).
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            time.sleep(wait)

    return {}


def monitor_xai_batch(state_file: str, api_key: str, base_url: str, poll_seconds: int = 20):
    """Poll xAI batch progress and print periodic progress with ETA."""
    if not os.path.exists(state_file):
        print(f"[!] State file not found: {state_file}")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    batch_id = state.get("batch_id")
    if not batch_id:
        print(f"[!] batch_id not found in state file: {state_file}")
        return

    print(f"\n[*] Monitoring xAI batch: {batch_id}")
    print(f"[*] Poll interval: {poll_seconds}s")
    print("[*] Press Ctrl+C to stop monitoring (batch keeps running in xAI).\n")

    first_ts = None
    first_done = None
    consecutive_errors = 0

    while True:
        try:
            data = fetch_xai_batch_status(batch_id, api_key, base_url)
            consecutive_errors = 0
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else None
            timestamp = time.strftime("%H:%M:%S")
            if status in (401, 403, 404):
                print(f"[{timestamp}] [!] Monitor stopped: HTTP {status} (non-retriable).")
                raise
            consecutive_errors += 1
            print(f"[{timestamp}] [!] transient monitor error ({consecutive_errors}): {e}")
            time.sleep(poll_seconds)
            continue

        counters = data.get("state", {})
        total = int(counters.get("num_requests", 0) or 0)
        pending = int(counters.get("num_pending", 0) or 0)
        success = int(counters.get("num_success", 0) or 0)
        errors = int(counters.get("num_error", 0) or 0)
        cancelled = int(counters.get("num_cancelled", 0) or 0)
        done = success + errors + cancelled
        pct = (done / total * 100.0) if total else 0.0

        now = time.time()
        eta_text = "estimating..."
        if first_ts is None:
            first_ts = now
            first_done = done
        else:
            elapsed = max(now - first_ts, 1e-6)
            rate = max((done - (first_done or 0)) / elapsed, 0.0)
            remaining = max(total - done, 0)
            if rate > 0:
                eta_sec = int(remaining / rate)
                eta_text = f"{eta_sec // 60}m {eta_sec % 60}s"

        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[{timestamp}] total={total} done={done} ({pct:.2f}%) "
            f"pending={pending} success={success} error={errors} "
            f"cancelled={cancelled} eta={eta_text}"
        )

        if pending <= 0 and total > 0:
            print("\n[+] Batch completed (no pending requests).")
            return

        time.sleep(poll_seconds)


# -------------------------
# HuggingFace download
# -------------------------

def detect_hf_reference(path: str):
    """Returns (repo_id, subfolder) if HF reference, else None."""
    path = path.strip()
    m = _HF_URL_RE.match(path)
    if m:
        subfolder = m.group(2).rstrip("/") if m.group(2) else None
        return m.group(1), subfolder
    if _HF_ID_RE.match(path) and not os.path.exists(path):
        return path, None
    return None


def download_hf_dataset(repo_id: str, subfolder, local_dir: str, token, python_path: str) -> str:
    """Download HF dataset using the venv's huggingface_hub. Uses xet when available."""
    pip = os.path.join(VENV_DIR, "bin", "pip")

    print("[*] Installing hf_xet for maximum download speed...")
    r = subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)
    if r.returncode == 0:
        print("[+] hf_xet ready — xet protocol enabled")
    else:
        print("[!] hf_xet unavailable, using standard HTTPS transfer")

    os.makedirs(local_dir, exist_ok=True)

    # Run snapshot_download inside the venv python (where huggingface_hub is installed)
    script = (
        "import sys, os\n"
        "from huggingface_hub import snapshot_download\n"
        "repo_id, local_dir = sys.argv[1], sys.argv[2]\n"
        "subfolder = sys.argv[3] if len(sys.argv) > 3 else None\n"
        "token = os.environ.get('HF_TOKEN') or None\n"
        "kwargs = dict(repo_id=repo_id, repo_type='dataset', local_dir=local_dir)\n"
        "if token: kwargs['token'] = token\n"
        "if subfolder:\n"
        "    kwargs['allow_patterns'] = [subfolder + '/*', subfolder + '/**/*']\n"
        "snapshot_download(**kwargs)\n"
        "print('__done__')\n"
    )

    cmd_args = [python_path, "-c", script, repo_id, local_dir]
    if subfolder:
        cmd_args.append(subfolder)

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token

    print(f"[*] Downloading {repo_id}{f'/{subfolder}' if subfolder else ''} → {local_dir}")
    result = subprocess.run(cmd_args, env=env)
    if result.returncode != 0:
        print("[!] Download failed. Check HF token and dataset ID.")
        sys.exit(1)

    result_path = os.path.join(local_dir, subfolder) if subfolder else local_dir
    if not os.path.isdir(result_path):
        result_path = local_dir
    print(f"[+] Download complete: {result_path}")
    return result_path


# -------------------------
# Helpers
# -------------------------

def count_images_quick(path: str, recursive: bool = True) -> int:
    """Quick image count without loading any ML deps."""
    count = 0
    try:
        if recursive:
            for root, _, files in os.walk(path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                        count += 1
        else:
            for f in os.listdir(path):
                if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    count += 1
    except OSError:
        pass
    return count


def list_zip_archives(path: str, recursive: bool = True) -> list[str]:
    """List .zip files under a directory."""
    zips: list[str] = []
    try:
        if recursive:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(".zip"):
                        zips.append(os.path.join(root, f))
        else:
            for f in os.listdir(path):
                full = os.path.join(path, f)
                if os.path.isfile(full) and f.lower().endswith(".zip"):
                    zips.append(full)
    except OSError:
        pass
    return sorted(zips)


def count_pending_zips(path: str, recursive: bool = True) -> int:
    """Count zip archives that haven't been extracted yet (no valid marker file)."""
    pending = 0
    for zpath in list_zip_archives(path, recursive=recursive):
        marker = zpath + ".extracted.ok"
        try:
            if not os.path.exists(marker) or os.path.getmtime(marker) < os.path.getmtime(zpath):
                pending += 1
        except OSError:
            pending += 1
    return pending


def extract_zip_archives(path: str, recursive: bool = True):
    """Extract zip archives in place, with marker files for idempotency."""
    zip_files = list_zip_archives(path, recursive=recursive)
    if not zip_files:
        return {"total": 0, "extracted": 0, "skipped": 0, "failed": 0}

    extracted = 0
    skipped = 0
    failed = 0

    print(f"[*] Found {len(zip_files):,} zip archives. Extracting in place...")
    for idx, zpath in enumerate(zip_files, 1):
        marker = zpath + ".extracted.ok"
        try:
            z_mtime = os.path.getmtime(zpath)
            if os.path.exists(marker) and os.path.getmtime(marker) >= z_mtime:
                skipped += 1
                continue

            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(os.path.dirname(zpath))

            with open(marker, "w", encoding="utf-8") as f:
                f.write(json.dumps({"zip": zpath, "extracted_at": time.time()}, ensure_ascii=False))

            extracted += 1
            if idx % 50 == 0 or idx == len(zip_files):
                print(f"    progress: {idx}/{len(zip_files)} zips")
        except Exception as e:
            failed += 1
            print(f"[!] Failed to extract {zpath}: {e}")

    return {
        "total": len(zip_files),
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
    }


def _collect_precheck(xai_batch_state_file: str, xai_api_key: str) -> bool:
    """Show batch status before collect and ask user if partial is ok.
    Returns False if user aborts.
    """
    print("\n[!] IMPORTANT: collect writes .txt next to images.")
    print("    Prefer same machine/path as submit. If using another machine, keep the")
    print("    same dataset folder structure under the chosen input directory.\n")

    if not os.path.exists(xai_batch_state_file):
        print(f"[!] State file not found: {xai_batch_state_file}")
        return True  # let the subprocess handle the error

    key = xai_api_key or check_env_key("XAI_API_KEY")
    if not key:
        return True  # no key to check status, proceed anyway

    try:
        with open(xai_batch_state_file, "r", encoding="utf-8") as f:
            st = json.load(f)
        bid = st.get("batch_id")
        if not bid:
            return True

        print("[*] Fetching batch status before collecting...")
        data = fetch_xai_batch_status(bid, key)
        c = data.get("state", {})
        total = int(c.get("num_requests", 0) or 0)
        pending = int(c.get("num_pending", 0) or 0)
        success = int(c.get("num_success", 0) or 0)
        errors = int(c.get("num_error", 0) or 0)
        done = success + errors
        pct = (done / total * 100) if total else 0

        print(f"\n  Batch ID:  {bid}")
        print(f"  Total:     {total:,}")
        print(f"  Done:      {done:,} ({pct:.1f}%)")
        print(f"  Success:   {success:,}")
        print(f"  Errors:    {errors:,}")
        print(f"  Pending:   {pending:,}")

        if pending > 0:
            print(f"\n  [!] {pending:,} requests still pending on xAI.")
            if not ask_yes_no(
                f"  Collect {done:,} completed results now (partial — missing {pending:,})?",
                default=True,
            ):
                print("Aborted. Run collect again when batch is fully complete.")
                return False
        else:
            print("\n  [+] Batch complete — all results available.\n")

    except Exception as e:
        print(f"[!] Could not fetch batch status: {e}")

    return True


# -------------------------
# Main
# -------------------------

def main():
    print_banner()

    # Setup
    python = ensure_venv()
    install_deps(python)

    # Input directory — supports HuggingFace URLs / dataset IDs
    raw_input = ask_input("Input directory (or HuggingFace ID/URL, e.g. user/dataset)")

    hf_ref = detect_hf_reference(raw_input)
    if hf_ref:
        repo_id, subfolder = hf_ref
        print(f"[*] HuggingFace dataset detected: {repo_id}" + (f"  subfolder: {subfolder}" if subfolder else ""))
        default_dl = os.path.join(
            os.path.expanduser("~"), "datasets", repo_id.replace("/", "_")
        )
        if subfolder:
            default_dl = os.path.join(default_dl, subfolder.replace("/", "_"))
        local_dir = ask_input("Download to", default_dl)
        dl_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
        if not dl_token:
            dl_token = ask_input("HF token (Enter to skip for public datasets)", "")
        input_dir = download_hf_dataset(repo_id, subfolder, local_dir, dl_token or None, python)
    else:
        input_dir = raw_input

    if not input_dir or not os.path.isdir(input_dir):
        print(f"[!] Directory not found: {input_dir}")
        sys.exit(1)

    zip_count = len(list_zip_archives(input_dir, recursive=True))
    if zip_count > 0:
        pending_zips = count_pending_zips(input_dir, recursive=True)
        if pending_zips == 0:
            print(f"[+] Found {zip_count:,} zip files — all already extracted, skipping.")
        else:
            print(f"[+] Found {zip_count:,} zip files ({pending_zips:,} pending extraction)")
            if ask_yes_no("Extract pending zip files now?", default=True):
                report = extract_zip_archives(input_dir, recursive=True)
                print(
                    f"[+] Zip extraction finished: extracted={report['extracted']}, "
                    f"skipped={report['skipped']}, failed={report['failed']}"
                )
                if report["failed"] > 0:
                    print("[!] Some zips failed to extract. Fix them before processing.")

    img_count = count_images_quick(input_dir, recursive=True)
    if img_count > 0:
        print(f"[+] Found ~{img_count:,} images in {input_dir}")
    else:
        print(f"[!] No images found in {input_dir} (subdirs will be scanned during processing)")

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
    has_local_taggers = any(t in taggers for t in ("wd14", "camie", "pixai"))
    grok_provider = "openrouter"
    xai_api_key = ""
    xai_batch_action = "submit"
    xai_batch_state_file = ""
    xai_batch_submit_chunk = "1000"
    xai_batch_page_size = "100"
    xai_batch_no_image = False
    monitor_xai = False
    monitor_poll_seconds = "20"
    grok_load_existing = False

    # Pro mode
    pro_mode = False
    if is_video:
        pro_mode = ask_yes_no("Enable PRO mode? (2 frames per video, better quality)", default=False)

    # Grok provider and mode (images only for xAI batch)
    if has_grok and not is_video:
        provider_choice = ask_choice(
            "Grok backend:",
            [
                "OpenRouter (real-time requests)",
                "xAI Batch API (background jobs, lower cost for large datasets)",
            ],
            default=1,
        )
        if provider_choice == 2:
            grok_provider = "xai-batch"
            action_choice = ask_choice(
                "xAI batch action:",
                [
                    "Submit requests to batch",
                    "Check batch status only",
                    "Collect completed results and write .txt",
                ],
                default=1,
            )
            action_map = {1: "submit", 2: "status", 3: "collect"}
            xai_batch_action = action_map[action_choice]

            xai_batch_state_file = ask_input(
                "Batch state file (.json)",
                resolve_default_xai_state_file(input_dir),
            )

            if xai_batch_action == "submit":
                send_images = ask_yes_no(
                    "Include images in each request? (better captions, larger payload)",
                    default=True,
                )
                xai_batch_no_image = not send_images
                chunk_default = "500" if send_images else "5000"
                xai_batch_submit_chunk = ask_input(
                    "Requests per submit call (encoding batch size)",
                    chunk_default,
                )
                monitor_xai = ask_yes_no("Monitor batch progress after submit?", default=True)
            elif xai_batch_action == "status":
                monitor_xai = ask_yes_no("Keep monitoring status continuously?", default=True)
            elif xai_batch_action == "collect":
                xai_batch_page_size = ask_input("Results page size for collect", "100")

            if monitor_xai:
                monitor_poll_seconds = ask_input("Monitor poll interval (seconds)", "20")

    # Load existing .txt as grok context
    # Only relevant for submit/openrouter — skip for status/collect (no output is written)
    is_collect_or_status = grok_provider == "xai-batch" and xai_batch_action in ("status", "collect")
    if has_grok and not is_video and not is_collect_or_status:
        if not has_local_taggers:
            grok_load_existing = ask_yes_no(
                "Load existing .txt tags as context for grok? (recommended if you already have booru tags)",
                default=True,
            )
        else:
            grok_load_existing = ask_yes_no(
                "Also load existing .txt tags as additional context for grok?",
                default=False,
            )

    # Grok API key
    api_key = ""
    if has_grok:
        if grok_provider == "xai-batch":
            xai_api_key = check_env_key("XAI_API_KEY")
            if xai_api_key:
                print(f"[+] Found XAI_API_KEY in environment ({xai_api_key[:8]}...)")
            else:
                xai_api_key = ask_input("Enter xAI API key")
                if not xai_api_key:
                    print("[!] No xAI API key provided. xAI batch mode will fail.")
                    if not ask_yes_no("Continue anyway?", default=False):
                        sys.exit(1)
        else:
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

    # Batch size — only relevant for local booru taggers (wd14/camie/pixai)
    batch_size = "4"
    if has_local_taggers:
        batch_size = ask_input("Batch size for local taggers", "4")

    # Grok concurrency
    grok_concurrency = "8"
    if has_grok and grok_provider == "openrouter":
        grok_concurrency = ask_input("Grok API concurrency (parallel requests)", "8")

    # Recursive
    recursive = ask_yes_no("Search subdirectories recursively?", default=True)

    # Force reprocess
    force = ask_yes_no("Force reprocess already-processed files?", default=False)

    # Collect pre-check: show batch status, ask about partial collect, warn about paths
    if has_grok and grok_provider == "xai-batch" and xai_batch_action == "collect":
        if not _collect_precheck(xai_batch_state_file, xai_api_key):
            sys.exit(0)

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

    if has_grok:
        cmd.extend(["--grok_provider", grok_provider])
        if grok_provider == "xai-batch":
            # status/collect should not trigger local booru taggers accidentally
            if xai_batch_action in ("status", "collect"):
                taggers = "grok"
                cmd = [python, TAGGER_SCRIPT, input_dir, "--taggers", taggers, "--batch_size", batch_size]
                cmd.extend(["--grok_provider", grok_provider])
                if is_video:
                    cmd.append("--video")
                if pro_mode:
                    cmd.append("--pro")
                if recursive:
                    cmd.append("--recursive")
                if force:
                    cmd.append("--force")
                cmd.append("--remove_underscore")

            if xai_api_key:
                cmd.extend(["--xai_api_key", xai_api_key])
            cmd.extend(["--xai_batch_action", xai_batch_action])
            cmd.extend(["--xai_batch_state_file", xai_batch_state_file])
            cmd.extend(["--xai_api_base_url", XAI_API_BASE_URL])
            cmd.extend(["--xai_batch_model", XAI_BATCH_DEFAULT_MODEL])
            if xai_batch_action == "submit":
                cmd.extend(["--xai_batch_submit_chunk", xai_batch_submit_chunk])
            if xai_batch_action == "collect":
                cmd.extend(["--xai_batch_page_size", xai_batch_page_size])
            if xai_batch_no_image:
                cmd.append("--xai_batch_no_image")
        else:
            if api_key:
                cmd.extend(["--grok_api_key", api_key])
            cmd.extend(["--grok_concurrency", grok_concurrency])

    if grok_load_existing:
        cmd.append("--grok_context_from_existing")

    if hf_token:
        cmd.extend(["--hf_token", hf_token])

    # Thresholds
    cmd.extend(["--thresh", "0.30"])

    # Summary
    print("\n" + "-" * 60)
    print("CONFIGURATION SUMMARY")
    print("-" * 60)
    print(f"  Input:        {input_dir}")
    if img_count:
        print(f"  Images:       ~{img_count:,}")
    print(f"  Mode:         {'video' if is_video else 'images'}{' (PRO)' if pro_mode else ''}")
    print(f"  Taggers:      {taggers}")
    if has_local_taggers:
        print(f"  Batch size:   {batch_size}")
    if has_grok:
        print(f"  Grok provider:{grok_provider}")
        grok_model_display = XAI_BATCH_DEFAULT_MODEL if grok_provider == "xai-batch" else "x-ai/grok-4.1-fast"
        print(f"  Grok model:   {grok_model_display}")
        if grok_provider == "openrouter":
            print(f"  Concurrency:  {grok_concurrency}")
        else:
            print(f"  Batch action: {xai_batch_action}")
            print(f"  State file:   {xai_batch_state_file}")
            if xai_batch_action == "submit":
                print(f"  Submit chunk: {xai_batch_submit_chunk}")
                print(f"  Send images:  {not xai_batch_no_image}")
            if xai_batch_action == "collect":
                print(f"  Page size:    {xai_batch_page_size}")
            if monitor_xai:
                print(f"  Monitor poll: {monitor_poll_seconds}s")
    if has_grok and not is_video and not is_collect_or_status:
        print(f"  Load existing tags: {grok_load_existing}")
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
    if xai_api_key:
        env["XAI_API_KEY"] = xai_api_key
    if hf_token:
        env["HF_TOKEN"] = hf_token

    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            print("\n" + "=" * 60)
            if xai_batch_action == "collect":
                print("  COLLECT DONE! .txt files written next to your images.")
            else:
                print("  DONE! Check your input directory for .txt files.")
            print("=" * 60 + "\n")
            if has_grok and grok_provider == "xai-batch" and monitor_xai and xai_batch_action in ("submit", "status"):
                try:
                    monitor_xai_batch(
                        state_file=xai_batch_state_file,
                        api_key=xai_api_key or env.get("XAI_API_KEY", ""),
                        base_url=XAI_API_BASE_URL,
                        poll_seconds=max(3, int(monitor_poll_seconds)),
                    )
                except KeyboardInterrupt:
                    print("\n[!] Monitoring stopped by user. Batch keeps running on xAI.")
                except Exception as e:
                    print(f"\n[!] Monitor error: {e}")
        else:
            print(f"\n[!] Process exited with code {result.returncode}")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
