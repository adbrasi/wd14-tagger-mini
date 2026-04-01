#!/usr/bin/env python3
"""Interactive CLI for data_araknideo.

Full pipeline wizard: download → validate → preprocess → tag → upload.
Uses Rich for styled terminal UI throughout.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import pathlib
import zipfile

import requests

from ui import (
    ask_choice,
    ask_input,
    ask_int,
    ask_yes_no,
    console,
    make_progress,
    print_banner,
    print_error,
    print_info,
    print_section,
    print_success,
    print_summary_table,
    print_warning,
)

# -------------------------
# Constants
# -------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv")
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")
TAGGER_SCRIPT = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
XAI_API_BASE_URL = "https://api.x.ai"
XAI_BATCH_DEFAULT_MODEL = "grok-4-1-fast-reasoning"
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")

from constants import IMAGE_EXTS, VIDEO_EXTS

# HuggingFace URL / ID detection
_HF_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"
    r"(?:/tree/[^/]+/(.+))?"
)
_HF_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


# -------------------------
# Security helpers
# -------------------------

def _validate_zip_members(zf: zipfile.ZipFile, dest: str) -> None:
    """Raise ValueError if any zip member would escape *dest* (Zip Slip)."""
    dest_path = pathlib.Path(dest).resolve()
    for member in zf.namelist():
        target = (dest_path / member).resolve()
        if target != dest_path and not str(target).startswith(str(dest_path) + os.sep):
            raise ValueError(f"Unsafe zip member path (path traversal): {member}")


# -------------------------
# Environment & Setup
# -------------------------

def check_env_key(name: str) -> str:
    """Check for env var and return its value."""
    return os.environ.get(name, "")


def ensure_venv() -> str:
    """Create and activate venv if needed. Returns python path."""
    python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.exists(python):
        return python

    print_info("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    print_success(f"venv created at {VENV_DIR}")
    return python


def install_deps(python: str):
    """Install/update requirements."""
    pip = os.path.join(VENV_DIR, "bin", "pip")
    print_info("Installing dependencies...")
    result = subprocess.run(
        [pip, "install", "-q", "-r", REQUIREMENTS],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print_error("pip install failed:")
        console.print(result.stderr)
        sys.exit(1)
    print_success("Dependencies installed")


# -------------------------
# xAI Batch Monitoring
# -------------------------

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
        print_error(f"State file not found: {state_file}")
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    batch_id = state.get("batch_id")
    if not batch_id:
        print_error(f"batch_id not found in state file: {state_file}")
        return

    print_info(f"Monitoring xAI batch: {batch_id}")
    print_info(f"Poll interval: {poll_seconds}s — Press Ctrl+C to stop")

    first_ts = None
    first_done = None
    zero_total_streak = 0

    while True:
        try:
            data = fetch_xai_batch_status(batch_id, api_key, base_url)
        except requests.exceptions.RequestException as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else None
            if status in (401, 403, 404):
                print_error(f"Monitor stopped: HTTP {status}")
                raise
            print_warning(f"Transient error: {e}")
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
        console.print(
            f"[dim]{timestamp}[/] total={total} done=[bold]{done}[/] ({pct:.1f}%) "
            f"pending={pending} success=[green]{success}[/] error=[red]{errors}[/] "
            f"eta={eta_text}"
        )

        if total == 0:
            zero_total_streak += 1
            if zero_total_streak >= 5:
                print_warning("Batch reports 0 requests for 5 consecutive polls — check batch ID")
                return
        else:
            zero_total_streak = 0

        if pending <= 0 and total > 0:
            print_success("Batch completed (no pending requests)")
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
    """Download HF dataset using the venv's huggingface_hub."""
    pip = os.path.join(VENV_DIR, "bin", "pip")

    print_info("Installing hf_xet for maximum download speed...")
    r = subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)
    if r.returncode == 0:
        print_success("hf_xet ready — xet protocol enabled")
    else:
        print_warning("hf_xet unavailable, using standard HTTPS transfer")

    os.makedirs(local_dir, exist_ok=True)

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
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    if token:
        env["HF_TOKEN"] = token

    print_info(f"Downloading {repo_id}{f'/{subfolder}' if subfolder else ''} → {local_dir}")
    result = subprocess.run(cmd_args, env=env)
    if result.returncode != 0:
        print_error("Download failed. Check HF token and dataset ID.")
        sys.exit(1)

    result_path = os.path.join(local_dir, subfolder) if subfolder else local_dir
    if not os.path.isdir(result_path):
        result_path = local_dir
    print_success(f"Download complete: {result_path}")
    return result_path


# -------------------------
# Prompt profiles
# -------------------------

def list_prompt_profiles(mode: str) -> list:
    """List available prompt profiles for a mode (image/video).

    Scans prompts/<mode>/ for subdirectories containing system_prompt.md.
    Returns sorted list of profile names, with 'default' first if it exists.
    """
    mode_dir = os.path.join(PROMPTS_DIR, mode)
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


# -------------------------
# File helpers
# -------------------------

def count_media_quick(path: str, recursive: bool = True) -> dict:
    """Quick media count without loading any ML deps."""
    images = 0
    videos = 0
    try:
        walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
        for root, _, files in walker:
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    images += 1
                elif ext in VIDEO_EXTS:
                    videos += 1
    except OSError:
        pass
    return {"images": images, "videos": videos}


def list_zip_archives(path: str, recursive: bool = True) -> list:
    """List .zip files under a directory."""
    zips = []
    try:
        walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
        for root, _, files in walker:
            for f in files:
                if f.lower().endswith(".zip"):
                    zips.append(os.path.join(root, f))
    except OSError:
        pass
    return sorted(zips)


def count_pending_zips(path: str, recursive: bool = True) -> int:
    """Count zip archives that haven't been extracted yet."""
    pending = 0
    for zpath in list_zip_archives(path, recursive=recursive):
        marker = zpath + ".extracted.ok"
        try:
            if not os.path.exists(marker) or os.path.getmtime(marker) < os.path.getmtime(zpath):
                pending += 1
        except OSError:
            pending += 1
    return pending


def extract_zip_archives(path: str, recursive: bool = True) -> dict:
    """Extract zip archives in place, with marker files for idempotency."""
    zip_files = list_zip_archives(path, recursive=recursive)
    if not zip_files:
        return {"total": 0, "extracted": 0, "skipped": 0, "failed": 0}

    extracted = 0
    skipped = 0
    failed = 0

    with make_progress() as progress:
        task = progress.add_task("Extracting zips", total=len(zip_files))
        for zpath in zip_files:
            marker = zpath + ".extracted.ok"
            try:
                z_mtime = os.path.getmtime(zpath)
                if os.path.exists(marker) and os.path.getmtime(marker) >= z_mtime:
                    skipped += 1
                    progress.advance(task)
                    continue

                with zipfile.ZipFile(zpath, "r") as zf:
                    _validate_zip_members(zf, os.path.dirname(zpath))
                    zf.extractall(os.path.dirname(zpath))

                with open(marker, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"zip": zpath, "extracted_at": time.time()}, ensure_ascii=False))

                extracted += 1
            except Exception as e:
                failed += 1
                print_warning(f"Failed to extract {os.path.basename(zpath)}: {e}")
            progress.advance(task)

    return {"total": len(zip_files), "extracted": extracted, "skipped": skipped, "failed": failed}


def _collect_precheck(xai_batch_state_file: str, xai_api_key: str) -> bool:
    """Show batch status before collect and ask user if partial is ok."""
    print_warning("IMPORTANT: collect writes .txt next to images.")
    print_info("Prefer same machine/path as submit.")

    if not os.path.exists(xai_batch_state_file):
        print_error(f"State file not found: {xai_batch_state_file}")
        return True

    key = xai_api_key or check_env_key("XAI_API_KEY")
    if not key:
        return True

    try:
        with open(xai_batch_state_file, "r", encoding="utf-8") as f:
            st = json.load(f)
        bid = st.get("batch_id")
        if not bid:
            return True

        print_info("Fetching batch status before collecting...")
        data = fetch_xai_batch_status(bid, key)
        c = data.get("state", {})
        total = int(c.get("num_requests", 0) or 0)
        pending = int(c.get("num_pending", 0) or 0)
        success = int(c.get("num_success", 0) or 0)
        errors = int(c.get("num_error", 0) or 0)
        done = success + errors
        pct = (done / total * 100) if total else 0

        print_summary_table("Batch Status", [
            ("Batch ID", bid),
            ("Total", f"{total:,}"),
            ("Done", f"{done:,} ({pct:.1f}%)"),
            ("Success", f"{success:,}"),
            ("Errors", f"{errors:,}"),
            ("Pending", f"{pending:,}"),
        ])

        if pending > 0:
            print_warning(f"{pending:,} requests still pending on xAI.")
            if not ask_yes_no(
                f"Collect {done:,} completed results now (partial — missing {pending:,})?",
                default=True,
            ):
                print_info("Aborted. Run collect again when batch is fully complete.")
                return False
        else:
            print_success("Batch complete — all results available")

    except Exception as e:
        print_warning(f"Could not fetch batch status: {e}")

    return True


# -------------------------
# Input source resolution
# -------------------------

def _detect_source_type(raw: str) -> str:
    """Detect what kind of input source a string is."""
    raw = raw.strip()
    if os.path.isdir(raw):
        return "local_dir"
    if raw.startswith("https://mega.nz/"):
        return "mega"
    if "huggingface.co" in raw and "/resolve/" in raw:
        return "hf_file"  # Direct file URL (e.g. .zip)
    if "huggingface.co" in raw:
        return "hf_dataset"
    if _HF_ID_RE.match(raw) and not os.path.exists(raw):
        return "hf_dataset"
    return "unknown"


def _parse_hf_file_url(url: str):
    """Parse HF direct file URL into (repo_id, filename) or None.

    Handles: https://huggingface.co/user/repo/resolve/main/path/file.zip?download=true
    """
    m = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+?)(?:\?.*)?$",
        url.strip(),
    )
    return (m.group(1), m.group(2)) if m else None


def _download_direct_url(url: str, dest_dir: str, python: str = "") -> str:
    """Download a direct file URL. Uses HF xet for HF URLs, aria2c for others."""
    os.makedirs(dest_dir, exist_ok=True)
    from urllib.parse import unquote, urlparse
    parsed_url = urlparse(url.split("?")[0])
    filename = unquote(os.path.basename(parsed_url.path)) or "download"
    dest_file = os.path.join(dest_dir, filename)

    # Try HF native download (uses xet protocol — fastest for HF files)
    hf_parsed = _parse_hf_file_url(url)
    if hf_parsed and python:
        repo_id, filepath = hf_parsed
        print_info(f"Downloading {filename} via HuggingFace xet protocol...")

        pip = os.path.join(VENV_DIR, "bin", "pip")
        subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)

        dl_script = (
            "from huggingface_hub import hf_hub_download\n"
            "import sys, os\n"
            "token = os.environ.get('HF_TOKEN') or None\n"
            "path = hf_hub_download(\n"
            "    repo_id=sys.argv[1], filename=sys.argv[2],\n"
            "    repo_type='dataset', local_dir=sys.argv[3],\n"
            "    token=token,\n"
            ")\n"
            "print(path)\n"
        )
        env = os.environ.copy()
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        r = subprocess.run(
            [python, "-c", dl_script, repo_id, filepath, dest_dir],
            env=env,
        )
        if r.returncode == 0:
            # hf_hub_download puts file in local_dir preserving repo structure
            # Find the actual downloaded file
            for root, _, files in os.walk(dest_dir):
                for f in files:
                    if f == filename:
                        downloaded = os.path.join(root, f)
                        if os.path.abspath(downloaded) != os.path.abspath(dest_file):
                            import shutil as _shutil
                            _shutil.move(downloaded, dest_file)
                        print_success(f"Downloaded via xet: {filename}")
                        return dest_file
            # File may have been saved with a different name
            for root, _, files in os.walk(dest_dir):
                for f in files:
                    full = os.path.join(root, f)
                    if os.path.getsize(full) > 0:
                        print_success(f"Downloaded via xet: {f}")
                        return full
        print_warning("HF xet download failed, falling back to wget...")

    # Fallback: wget with HF token auth header (via temp file to avoid ps leaks)
    import tempfile as _tmpmod
    print_info(f"Downloading {filename} via wget...")
    wget_cmd = ["wget", "-c", "-O", dest_file]
    hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
    _header_file = None
    try:
        if hf_token:
            _header_file = _tmpmod.NamedTemporaryFile(
                mode="w", prefix=".wget_hdr_", suffix=".txt", delete=False
            )
            _header_file.write(f"header = Authorization: Bearer {hf_token}\n")
            _header_file.close()
            wget_cmd.extend(["--config", _header_file.name])
        wget_cmd.append(url)
        result = subprocess.run(wget_cmd)
    finally:
        if _header_file is not None:
            try:
                os.unlink(_header_file.name)
            except OSError:
                pass

    if result.returncode != 0 or not os.path.exists(dest_file):
        print_error(f"Download failed: {url}")
        return ""

    print_success(f"Downloaded: {dest_file}")
    return dest_file


def _process_single_source(
    raw: str, target_dir: str, python: str, source_num: int
) -> bool:
    """Process a single data source and move files into target_dir.

    Returns True if files were added successfully.
    """
    from mega_download import (
        check_mega_installed,
        flatten_directory,
        install_mega,
        mega_download,
    )

    source_type = _detect_source_type(raw)

    if source_type == "local_dir":
        print_info(f"Source {source_num}: local directory → {raw}")
        # If source IS the target, skip flatten (files already there)
        if os.path.abspath(raw) == os.path.abspath(target_dir):
            print_info("Source is target directory — no flatten needed")
            return True
        # Prevent flatten if source is parent of target (would cause duplication)
        if os.path.abspath(target_dir).startswith(os.path.abspath(raw) + os.sep):
            print_warning("Source is parent of target — copying instead of moving to avoid duplication")
            import shutil as _shutil
            for root, _, files in os.walk(raw):
                if os.path.abspath(root).startswith(os.path.abspath(target_dir)):
                    continue
                for f in files:
                    src = os.path.join(root, f)
                    _shutil.copy2(src, target_dir)
            return True
        flatten_directory(raw, target_dir)
        return True

    elif source_type == "mega":
        print_info(f"Source {source_num}: MEGA link")

        if not check_mega_installed():
            print_info("MEGAcmd not found — installing automatically...")
            if not install_mega():
                print_error("Could not install MEGAcmd")
                return False

        import tempfile as _tmp
        tmp_dir = _tmp.mkdtemp(prefix=f"araknideo_mega_{source_num}_")
        if not mega_download(raw, tmp_dir):
            print_error("MEGA download failed")
            return False

        # Auto-extract any zip files downloaded from MEGA
        import zipfile as _zf
        for dirpath, _, filenames in os.walk(tmp_dir):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                if fname.lower().endswith(".zip") and _zf.is_zipfile(fpath):
                    print_info(f"Extracting zip: {fname}")
                    try:
                        with _zf.ZipFile(fpath, "r") as z:
                            _validate_zip_members(z, tmp_dir)
                            z.extractall(tmp_dir)
                        os.remove(fpath)
                        print_success(f"Extracted: {fname}")
                    except Exception as e:
                        print_warning(f"Failed to extract {fname}: {e}")

        stats = flatten_directory(tmp_dir, target_dir)
        print_success(
            f"MEGA: {stats['moved']:,} files "
            f"({stats['conflicts']:,} conflicts resolved, "
            f"{stats['pairs']:,} txt pairs)"
        )
        return True

    elif source_type == "hf_file":
        print_info(f"Source {source_num}: HuggingFace direct file URL")
        import tempfile as _tmp
        tmp_dir = _tmp.mkdtemp(prefix=f"araknideo_hf_{source_num}_")
        dest_file = _download_direct_url(raw, tmp_dir, python=python)
        if not dest_file:
            return False

        # If it's a zip, extract it
        if dest_file.lower().endswith(".zip"):
            print_info("Extracting zip...")
            import zipfile as zf
            try:
                with zf.ZipFile(dest_file, "r") as z:
                    _validate_zip_members(z, tmp_dir)
                    z.extractall(tmp_dir)
                os.remove(dest_file)
                print_success("Zip extracted")
            except Exception as e:
                print_error(f"Zip extraction failed: {e}")
                return False

        stats = flatten_directory(tmp_dir, target_dir)
        print_success(
            f"HF: {stats['moved']:,} files "
            f"({stats['conflicts']:,} conflicts resolved, "
            f"{stats['pairs']:,} txt pairs)"
        )
        return True

    elif source_type == "hf_dataset":
        print_info(f"Source {source_num}: HuggingFace dataset")
        hf_ref = detect_hf_reference(raw)
        if not hf_ref:
            print_error(f"Not a valid HF reference: {raw}")
            return False

        repo_id, subfolder = hf_ref
        import tempfile as _tmp
        tmp_dir = _tmp.mkdtemp(prefix=f"araknideo_hfds_{source_num}_")
        dl_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
        downloaded = download_hf_dataset(repo_id, subfolder, tmp_dir, dl_token or None, python)
        stats = flatten_directory(downloaded, target_dir)
        print_success(
            f"HF: {stats['moved']:,} files "
            f"({stats['conflicts']:,} conflicts resolved, "
            f"{stats['pairs']:,} txt pairs)"
        )
        return True

    else:
        print_error(f"Could not detect source type for: {raw}")
        return False


def resolve_input_source(python: str, default_target: str = "") -> str:
    """Ask user for data sources (supports multiple) and merge into one directory.

    Accepts any mix of: local dirs, MEGA links, HF URLs, HF dataset IDs.
    All files are flattened into a single target directory with conflict resolution.
    """
    if not default_target:
        default_target = os.path.join(os.path.expanduser("~"), "datasets", "araknideo_dataset")
    target_dir = default_target
    os.makedirs(target_dir, exist_ok=True)
    print_info(f"Dataset directory: {target_dir}")

    # If target dir already has files, data source is optional
    existing_files = any(
        f for _, _, files in os.walk(target_dir) for f in files
    ) if os.path.isdir(target_dir) else False

    source_num = 0
    while True:
        is_first = source_num == 0
        if is_first:
            if existing_files:
                print_info(f"Files already found in {target_dir}")
                raw = ask_input(
                    "Data source (Enter to skip)\n"
                    "  Accepts: local path, MEGA link/folder, HuggingFace URL, HF dataset ID"
                )
            else:
                raw = ask_input(
                    "Data source\n"
                    "  Accepts: local path, MEGA link/folder, HuggingFace URL, HF dataset ID"
                )
        else:
            raw = ask_input("Another data source (or Enter to continue)")

        if not raw:
            if is_first and not existing_files:
                print_error("At least one data source is required")
                continue
            break

        source_num += 1
        print_section(f"PROCESSING SOURCE {source_num}")
        ok = _process_single_source(raw, target_dir, python, source_num)
        if not ok:
            print_warning(f"Source {source_num} failed — skipping")

        # Count what we have so far
        from dataset_validate import scan_pairs
        pairs = scan_pairs(target_dir, recursive=True)
        print_info(
            f"Dataset so far: {pairs['total_media']:,} media files, "
            f"{len(pairs['media_with_txt']):,} with captions"
        )

        if not ask_yes_no("Add another data source?", default=False):
            break

    if not os.listdir(target_dir):
        print_error(f"No files in {target_dir}")
        sys.exit(1)

    return target_dir


# -------------------------
# Dataset validation
# -------------------------

def run_validation(input_dir: str):
    """Validate media/txt pairs and offer to fix issues."""
    from dataset_validate import delete_files, scan_pairs

    print_section("DATASET VALIDATION")
    pairs = scan_pairs(input_dir, recursive=True)

    print_summary_table("Dataset Contents", [
        ("Media files", f"{pairs['total_media']:,}"),
        ("Text files", f"{pairs['total_txt']:,}"),
        ("Paired", f"{len(pairs['media_with_txt']):,}"),
        ("Media without caption", f"{len(pairs['media_without_txt']):,}"),
        ("Orphan .txt files", f"{len(pairs['txt_without_media']):,}"),
    ])

    if pairs["media_without_txt"]:
        print_warning(f"{len(pairs['media_without_txt']):,} media files WITHOUT captions")
        action = ask_choice("What to do with uncaptioned media?", [
            f"Delete {len(pairs['media_without_txt']):,} uncaptioned files",
            "Keep them (will be captioned during tagging)",
            "Show file list first",
        ], default=2)

        if action == 1:
            deleted = delete_files(pairs["media_without_txt"])
            print_success(f"Deleted {deleted:,} uncaptioned media files")
        elif action == 3:
            for p in pairs["media_without_txt"][:20]:
                print_info(os.path.basename(p))
            if len(pairs["media_without_txt"]) > 20:
                print_info(f"... and {len(pairs['media_without_txt']) - 20:,} more")
            if ask_yes_no(f"Delete all {len(pairs['media_without_txt']):,}?", default=False):
                deleted = delete_files(pairs["media_without_txt"])
                print_success(f"Deleted {deleted:,} files")

    if pairs["txt_without_media"]:
        print_warning(f"{len(pairs['txt_without_media']):,} orphan .txt files (no matching media)")
        if ask_yes_no(f"Delete {len(pairs['txt_without_media']):,} orphan .txt files?", default=False):
            deleted = delete_files(pairs["txt_without_media"])
            print_success(f"Deleted {deleted:,} orphan .txt files")


# -------------------------
# Video preprocessing
# -------------------------

def run_preprocessing(input_dir: str):
    """Run video preprocessing: normalize formats, fix audio, trim frames."""
    import shutil

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print_info("ffmpeg not found — installing automatically...")
        result = subprocess.run(
            ["sudo", "apt", "install", "-y", "ffmpeg"],
            capture_output=False,
        )
        if result.returncode != 0 or shutil.which("ffmpeg") is None:
            print_error("ffmpeg installation failed. Install manually: sudo apt install ffmpeg")
            return
        print_success("ffmpeg installed")

    # Import CONVERTIBLE_EXTS from the normalize module (single source of truth)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "video_normalize",
        os.path.join(SCRIPT_DIR, "video_normalize.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    convertible_exts = _mod.CONVERTIBLE_EXTS

    # Scan all media files: mp4 + convertible formats (gif/webp/avi/mov/etc)
    scan_exts = {".mp4"} | convertible_exts
    all_media = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in scan_exts:
                all_media.append(os.path.join(root, f))

    if not all_media:
        print_info("No videos found — skipping preprocessing")
        return

    mp4_count = sum(1 for p in all_media if os.path.splitext(p)[1].lower() == ".mp4")
    non_mp4_count = len(all_media) - mp4_count

    print_section("VIDEO PREPROCESSING")
    print_info(f"Found {len(all_media):,} media files ({mp4_count:,} mp4, {non_mp4_count:,} other formats)")

    workers = max(4, min(os.cpu_count() or 4, 64))

    # ── Step 1: Normalize formats + fix mono audio + fps ──
    do_normalize = False
    if non_mp4_count > 0:
        print_info(f"{non_mp4_count:,} files can be converted to mp4 (gif/webp/avi/etc)")
        do_normalize = ask_yes_no("Convert all formats to mp4?", default=True)
    do_stereo = ask_yes_no("Fix mono audio → stereo?", default=True)
    do_fps = ask_yes_no("Normalize framerate (fixed fps for all videos)?", default=False)
    target_fps = None
    if do_fps:
        target_fps = float(ask_int("Target fps", default=25, minimum=1))

    if do_normalize or do_stereo or do_fps:
        # _mod already loaded above for CONVERTIBLE_EXTS
        print_section("NORMALIZING VIDEOS")
        progress = make_progress()
        progress.start()
        convert_task = None
        stereo_task = None
        fps_task = None

        def _on_norm_progress(*args):
            nonlocal convert_task, stereo_task, fps_task
            phase = args[0]
            if phase == "convert":
                _, done, total = args
                if convert_task is None and total > 0:
                    convert_task = progress.add_task("Converting formats → mp4", total=total)
                if convert_task is not None:
                    progress.update(convert_task, completed=done)
            elif phase == "stereo":
                _, done, total = args
                if stereo_task is None and total > 0:
                    stereo_task = progress.add_task("Fixing mono → stereo", total=total)
                if stereo_task is not None:
                    progress.update(stereo_task, completed=done)
            elif phase == "fps":
                _, done, total = args
                if fps_task is None and total > 0:
                    fps_task = progress.add_task(f"Normalizing fps → {target_fps}", total=total)
                if fps_task is not None:
                    progress.update(fps_task, completed=done)

        try:
            norm_result = _mod.normalize_videos(
                all_media, fix_stereo=do_stereo, target_fps=target_fps,
                max_workers=workers, on_progress=_on_norm_progress,
            )
        except KeyboardInterrupt:
            progress.stop()
            print_warning("Normalization interrupted")
            return

        progress.stop()

        if norm_result["converted"] > 0:
            print_success(f"{norm_result['converted']:,} files converted to mp4")
        if norm_result["convert_failed"] > 0:
            print_warning(f"{norm_result['convert_failed']:,} conversions failed")
        if do_stereo:
            print_success(
                f"Stereo: {norm_result['stereo_converted']:,} fixed, "
                f"{norm_result['stereo_skipped']:,} already stereo, "
                f"{norm_result['stereo_no_audio']:,} no audio"
            )
            if norm_result["stereo_failed"] > 0:
                print_warning(f"{norm_result['stereo_failed']:,} stereo conversions failed")
        if do_fps:
            print_success(
                f"FPS: {norm_result['fps_converted']:,} re-encoded to {target_fps} fps, "
                f"{norm_result['fps_skipped']:,} already correct"
            )
            if norm_result["fps_failed"] > 0:
                print_warning(f"{norm_result['fps_failed']:,} fps conversions failed")

        if norm_result["details"]:
            for d in norm_result["details"][:5]:
                print_info(f"  {os.path.basename(d['path'])}: {d.get('detail', '')}")
            if len(norm_result["details"]) > 5:
                print_info(f"  ... and {len(norm_result['details']) - 5:,} more errors")

    # ── Step 2: Trim frames ──
    # Re-scan for mp4 files after normalization
    videos = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() == ".mp4":
                videos.append(os.path.join(root, f))

    if not videos:
        print_info("No mp4 videos found after normalization")
        return

    from video_preprocess import preprocess_videos, snap_frames

    print_info(f"\n{len(videos):,} mp4 videos ready for trimming")
    print_info("Method: stream copy (lossless, preserves audio)")

    do_cut = ask_yes_no("Trim videos to max frame count?", default=True)
    if not do_cut:
        print_info("Trimming skipped — preprocessing done")
        return

    raw_frames = ask_int("Max frames", default=49, minimum=1)
    max_frames = snap_frames(raw_frames)
    if max_frames != raw_frames:
        print_info(f"Snapped {raw_frames} → {max_frames} (F % 8 == 1 rule)")

    # Optional sample limit for testing
    test_limit = ask_int("How many videos to process? (0 = all)", default=0, minimum=0)
    if test_limit > 0 and test_limit < len(videos):
        import random
        random.shuffle(videos)
        videos = videos[:test_limit]
        print_info(f"Testing with {test_limit} sample videos")

    print_summary_table("Trim Config", [
        ("Videos", f"{len(videos):,}"),
        ("Frame limit", str(max_frames)),
        ("Method", "stream copy (lossless)"),
        ("Workers", str(workers)),
    ])

    if not ask_yes_no("Trim videos? (MODIFIES ORIGINALS)", default=False):
        print_info("Trimming skipped")
        return

    print_section("TRIMMING VIDEOS")
    progress = make_progress()
    progress.start()
    scan_task = progress.add_task("Scanning videos (ffprobe)", total=len(videos))
    trim_task = None

    def _on_progress(*args):
        nonlocal trim_task
        phase = args[0]
        if phase == "scan":
            _, done, total = args
            progress.update(scan_task, completed=done, total=total)
        elif phase == "trim":
            _, success, failed, total = args
            if trim_task is None and total > 0:
                trim_task = progress.add_task("Trimming videos (stream copy)", total=total)
            if trim_task is not None:
                progress.update(trim_task, completed=success + failed)

    try:
        result = preprocess_videos(
            videos, max_frames=max_frames, max_workers=workers,
            on_progress=_on_progress,
        )
    except KeyboardInterrupt:
        progress.stop()
        print_warning("Trimming interrupted by user")
        return

    progress.update(scan_task, completed=len(videos))
    if trim_task is not None:
        progress.update(trim_task, completed=result["trimmed"] + result["failed"])
    progress.stop()

    print_success(
        f"Done: {result['trimmed']:,} trimmed, "
        f"{result['failed']:,} failed, "
        f"{result['skipped']:,} already short enough"
    )
    if result["probe_failed"] > 0:
        print_warning(f"{result['probe_failed']:,} videos could not be probed (corrupt or unreadable)")

    if result["failed"] > 0:
        print_warning(f"{result['failed']:,} videos failed to trim")
        errors = [d for d in result.get("details", []) if not d.get("ok")]
        for e in errors[:5]:
            print_info(f"  {os.path.basename(e['path'])}: {e.get('detail', 'unknown error')}")
        if len(errors) > 5:
            print_info(f"  ... and {len(errors) - 5:,} more")


# -------------------------
# HuggingFace upload
# -------------------------

def run_hf_upload(input_dir: str, python: str, project_name: str = ""):
    """Upload dataset to HuggingFace using upload_large_folder.

    Uses HuggingFace's native upload_large_folder which provides:
    - Multi-threaded upload (16 workers)
    - Automatic resume (locally cached state)
    - Resilient retry on any error
    - xet high-performance protocol
    - No manual chunking needed
    """
    hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        hf_token = ask_input("HuggingFace token")
        if not hf_token:
            print_error("No HF token provided")
            return

    # Try to get HF username for default repo name
    default_repo = ""
    try:
        r = subprocess.run(
            [python, "-c",
             "from huggingface_hub import HfApi; import os; "
             "api = HfApi(token=os.environ.get('HF_TOKEN','')); "
             "print(api.whoami()['name'])"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "HF_TOKEN": hf_token},
        )
        if r.returncode == 0 and r.stdout.strip():
            repo_suffix = project_name or os.path.basename(os.path.abspath(input_dir))
            default_repo = f"{r.stdout.strip()}/{repo_suffix}"
    except Exception:
        pass

    repo_name = ask_input("Repository name (user/dataset)", default=default_repo)
    if not repo_name or "/" not in repo_name:
        print_error("Repository name must be in format user/dataset")
        return

    private = ask_yes_no("Private repository?", default=True)

    # Ask upload format
    upload_format = ask_choice("Upload format", [
        "Zip chunks (images+txt packed into zip shards — recommended)",
        "WebDataset TAR shards (video+txt pairs, streaming format)",
        "Individual files (upload_large_folder — resumable, simple)",
    ])

    upload_dir = input_dir

    if upload_format == 1:
        # Zip chunks
        print_section("PACKING INTO ZIP FILES")
        zip_output = os.path.join(os.path.dirname(os.path.abspath(input_dir)),
                                  os.path.basename(input_dir) + "_zips")
        os.makedirs(zip_output, exist_ok=True)
        chunk_gb = 5.0
        max_chunk_bytes = int(chunk_gb * (1024**3))

        # Collect all media+txt files
        all_files = []
        for dirpath, _, filenames in os.walk(input_dir):
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS or ext == ".txt":
                    all_files.append(fpath)

        if not all_files:
            print_error(f"No media files found in {input_dir}")
            return

        dataset_name = os.path.basename(os.path.abspath(input_dir))
        print_info(f"Packing {len(all_files):,} files into ~{chunk_gb}GB zips → {zip_output}")

        part_idx = 1
        current_bytes = 0
        current_zip = None
        parts_created = 0
        files_packed = 0

        try:
            for fpath in all_files:
                fsize = os.path.getsize(fpath)
                rel_path = os.path.relpath(fpath, input_dir)

                if current_zip is None or (current_bytes > 0 and current_bytes + fsize > max_chunk_bytes):
                    if current_zip is not None:
                        current_zip.close()
                        parts_created += 1
                    zip_path = os.path.join(zip_output, f"{dataset_name}-part{part_idx:03d}.zip")
                    current_zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED)
                    part_idx += 1
                    current_bytes = 0

                current_zip.write(fpath, rel_path)
                current_bytes += fsize
                files_packed += 1

            if current_zip is not None:
                current_zip.close()
                parts_created += 1
        except KeyboardInterrupt:
            if current_zip is not None:
                current_zip.close()
            print_warning("\nZip building interrupted")
            return

        print_success(f"{parts_created} zip files created, {files_packed:,} files packed")
        upload_dir = zip_output

    elif upload_format == 2:
        # WebDataset TAR
        print_section("BUILDING WEBDATASET TAR SHARDS")
        tar_output = os.path.join(os.path.dirname(os.path.abspath(input_dir)),
                                  os.path.basename(input_dir) + "_webdataset")
        print_info(f"Packing video+txt pairs into TAR shards → {tar_output}")

        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "build_webdataset_tars",
            os.path.join(SCRIPT_DIR, "build_webdataset_tars.py"),
        )
        if _spec is None or _spec.loader is None:
            print_error("build_webdataset_tars.py not found")
            return
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        from pathlib import Path
        try:
            stats = _mod.build_tars(
                root=Path(input_dir),
                output_dir=Path(tar_output),
                shard_size_gb=1.0,
                split="train",
            )
        except KeyboardInterrupt:
            print_warning("\nTAR building interrupted")
            return

        print_success(
            f"{stats['shards_created']} shards created, "
            f"{stats['total_pairs']} samples, "
            f"{stats['total_bytes'] / (1024**3):.1f} GB"
        )
        if stats["missing_txt"]:
            print_warning(f"{stats['missing_txt']} files without .txt caption")
        upload_dir = tar_output

    print_section("UPLOADING TO HUGGINGFACE")
    print_info(f"Uploading {upload_dir} → {repo_name} ({'private' if private else 'public'})")
    print_info("Using upload_large_folder (multi-threaded, resumable, auto-retry)")

    # Install hf_xet for max speed
    pip = os.path.join(VENV_DIR, "bin", "pip")
    subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)

    num_workers = max(4, min(64, (os.cpu_count() or 4) * 2))

    upload_script = (
        "from huggingface_hub import HfApi\n"
        "import sys, os\n"
        "api = HfApi(token=os.environ['HF_TOKEN'])\n"
        "api.create_repo(sys.argv[1], repo_type='dataset', "
        "private=sys.argv[3]=='true', exist_ok=True)\n"
        "api.upload_large_folder(\n"
        "    repo_id=sys.argv[1],\n"
        "    repo_type='dataset',\n"
        "    folder_path=sys.argv[2],\n"
        "    num_workers=int(sys.argv[4]),\n"
        ")\n"
        "print('__done__')\n"
    )

    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    env["HF_XET_HIGH_PERFORMANCE"] = "1"

    proc = subprocess.Popen(
        [python, "-c", upload_script, repo_name, upload_dir,
         "true" if private else "false", str(num_workers)],
        env=env,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        print_warning("\nInterrupted — killing upload process...")
        proc.kill()
        proc.wait()
        print_info("Re-run to resume automatically (upload_large_folder is resumable)")
        return

    if proc.returncode == 0:
        print_success(f"Uploaded to https://huggingface.co/datasets/{repo_name}")
    else:
        print_error("Upload failed — check logs above")
        print_info("Re-run the same command to resume (upload_large_folder is resumable)")


# -------------------------
# Tagging pipeline
# -------------------------

def run_tagging(input_dir: str, python: str, media_counts: dict):
    """Run the tagging pipeline with full configuration wizard."""
    img_count = media_counts["images"]
    vid_count = media_counts["videos"]

    # Mode selection
    mode = ask_choice("What are you processing?", [
        f"Videos (extract frames and tag) — {vid_count:,} found",
        f"Images (tag directly) — {img_count:,} found",
    ], default=1 if vid_count > 0 else 2)
    is_video = mode == 1

    # Tagger selection
    tagger_options = [
        "pixai + grok (recommended for video LoRA)",
        "wd14 + pixai + grok (full pipeline)",
        "pixai only (fast, tags only)",
        "wd14 only",
        "grok only (needs API key)",
        "Custom (enter manually)",
    ]
    tagger_choice = ask_choice("Select tagger combination:", tagger_options, default=1)
    tagger_map = {1: "pixai,grok", 2: "wd14,pixai,grok", 3: "pixai", 4: "wd14", 5: "grok"}
    if tagger_choice == 6:
        taggers = ask_input("Enter taggers (comma-separated)", "pixai,grok")
    else:
        taggers = tagger_map[tagger_choice]

    has_grok = "grok" in taggers
    has_local_taggers = any(t in taggers for t in ("wd14", "camie", "pixai"))

    # Grok provider — xAI Batch as default for video
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

    # Prompt profile selection
    prompt_profile = "default"
    if has_grok:
        mode_name = "video" if is_video else "image"
        profiles = list_prompt_profiles(mode_name)
        if len(profiles) > 1:
            profile_choice = ask_choice(
                f"Prompt profile ({mode_name}):",
                profiles,
                default=1,
            )
            prompt_profile = profiles[profile_choice - 1]
        else:
            prompt_profile = profiles[0]
            print_info(f"Using prompt profile: {prompt_profile}")

    # Grok provider selection
    if has_grok:
        default_provider = 2 if is_video else 1  # xAI Batch default for video
        provider_choice = ask_choice(
            "Grok backend:",
            [
                "OpenRouter (real-time, concurrent requests)",
                "xAI Batch API (background jobs, 50% cheaper, no rate limits)",
            ],
            default=default_provider,
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
                    "Requests per submit call",
                    chunk_default,
                )
                monitor_xai = ask_yes_no("Monitor batch progress after submit?", default=True)
            elif xai_batch_action == "status":
                monitor_xai = ask_yes_no("Keep monitoring status continuously?", default=True)
            elif xai_batch_action == "collect":
                xai_batch_page_size = ask_input("Results page size for collect", "100")

            if monitor_xai:
                monitor_poll_seconds = str(ask_int("Monitor poll interval (seconds)", default=20, minimum=3))

    # Load existing .txt as grok context
    is_collect_or_status = grok_provider == "xai-batch" and xai_batch_action in ("status", "collect")
    if has_grok and not is_video and not is_collect_or_status:
        if not has_local_taggers:
            grok_load_existing = ask_yes_no(
                "Load existing .txt tags as context for grok?",
                default=True,
            )
        else:
            grok_load_existing = ask_yes_no(
                "Also load existing .txt tags as additional context for grok?",
                default=True,
            )

    # API keys
    api_key = ""
    if has_grok:
        if grok_provider == "xai-batch":
            xai_api_key = check_env_key("XAI_API_KEY")
            if xai_api_key:
                print_success(f"Found XAI_API_KEY in environment ({xai_api_key[:4]}****)")
            else:
                xai_api_key = ask_input("Enter xAI API key")
                if not xai_api_key:
                    print_error("No xAI API key provided. xAI batch mode will fail.")
                    if not ask_yes_no("Continue anyway?", default=False):
                        sys.exit(1)
        else:
            api_key = check_env_key("OPENROUTER_API_KEY")
            if api_key:
                print_success(f"Found OPENROUTER_API_KEY in environment ({api_key[:4]}****)")
            else:
                api_key = ask_input("Enter OpenRouter API key (sk-or-...)")
                if not api_key:
                    print_error("No API key provided. Grok tagger will fail.")
                    if not ask_yes_no("Continue anyway?", default=False):
                        sys.exit(1)

    # HF token (for pixai gated repo)
    hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
    if "pixai" in taggers and not hf_token:
        print_warning("PixAI model is gated. You may need a HuggingFace token.")
        hf_token = ask_input("Enter HF token (or press Enter to skip)", "")

    # Batch size — auto VRAM detection (skip for collect/status — no local inference)
    batch_size = "4"
    is_collect_or_status = (grok_provider == "xai-batch" and has_grok
                            and xai_batch_action in ("collect", "status"))
    if has_local_taggers and not is_collect_or_status:
        batch_size = ask_input("Batch size for local taggers (or 'auto' for VRAM-based)", "auto")
        if batch_size.lower() == "auto":
            cmd_args = [
                python, "-c",
                "from tag_images_by_wd14_tagger import recommend_batch_by_vram; "
                "r = recommend_batch_by_vram(); print(r if r else 4)",
            ]
            try:
                r = subprocess.run(cmd_args, capture_output=True, text=True, timeout=10)
                batch_size = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "4"
                print_success(f"Auto batch size from VRAM: {batch_size}")
            except Exception:
                batch_size = "4"
                print_warning("Could not detect VRAM, using batch_size=4")

    # Grok concurrency
    grok_concurrency = "32"
    if has_grok and grok_provider == "openrouter":
        grok_concurrency = ask_input("Grok API concurrency (parallel requests)", "32")

    # Recursive
    recursive = ask_yes_no("Search subdirectories recursively?", default=True)

    # Force reprocess
    force = ask_yes_no("Force reprocess already-processed files?", default=False)

    # Collect pre-check
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
        cmd.extend(["--prompt_profile", prompt_profile])
        if grok_provider == "xai-batch":
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
            cmd.extend(["--grok_concurrency", grok_concurrency])

    if grok_load_existing:
        cmd.append("--grok_context_from_existing")

    cmd.extend(["--thresh", "0.30"])

    # Summary
    grok_model_display = XAI_BATCH_DEFAULT_MODEL if grok_provider == "xai-batch" else "google/gemini-3-flash-preview"
    summary_rows = [
        ("Input", input_dir),
        ("Mode", f"{'video' if is_video else 'images'}{' (PRO)' if pro_mode else ''}"),
        ("Taggers", taggers),
    ]
    if has_local_taggers:
        summary_rows.append(("Batch size", batch_size))
    if has_grok:
        summary_rows.append(("Grok provider", grok_provider))
        summary_rows.append(("Grok model", grok_model_display))
        summary_rows.append(("Prompt profile", prompt_profile))
        if grok_provider == "openrouter":
            summary_rows.append(("Concurrency", grok_concurrency))
        else:
            summary_rows.append(("Batch action", xai_batch_action))
            summary_rows.append(("State file", xai_batch_state_file))
    summary_rows.extend([
        ("Recursive", str(recursive)),
        ("Force", str(force)),
    ])
    print_summary_table("CONFIGURATION", summary_rows)

    if not ask_yes_no("Start processing?", default=True):
        print_info("Aborted")
        return False

    # --- Dry-run: test with a single sample first ---
    # Skip test run for collect/status actions (nothing to test locally)
    skip_test = (grok_provider == "xai-batch" and has_grok
                 and xai_batch_action in ("collect", "status"))
    if skip_test:
        print_info("Skipping test run (collect/status — no local test needed)")

    if not skip_test:
        print_section("TEST RUN (1 sample)")
        print_info("Running pipeline on a single file to verify everything works...")

        # Find one sample file to test with
        import tempfile
        sample_file = None
        test_exts = VIDEO_EXTS if is_video else IMAGE_EXTS
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in test_exts:
                    sample_file = os.path.join(root, f)
                    break
            if sample_file:
                break

        if not sample_file:
            print_warning("No sample file found — skipping test run")
        else:
            print_info(f"Test file: {os.path.basename(sample_file)}")

            # Create temp dir with just that one file (symlinked)
            with tempfile.TemporaryDirectory(prefix="araknideo_test_") as test_dir:
                test_link = os.path.join(test_dir, os.path.basename(sample_file))
                os.symlink(os.path.abspath(sample_file), test_link)

                # Also copy the .txt if it exists (for context)
                txt_src = os.path.splitext(sample_file)[0] + ".txt"
                if os.path.exists(txt_src):
                    txt_link = os.path.join(test_dir, os.path.basename(txt_src))
                    os.symlink(os.path.abspath(txt_src), txt_link)

                # Build test command — same as full but on the test dir, no recursive, force
                test_cmd = list(cmd)
                test_cmd[2] = test_dir
                if "--recursive" in test_cmd:
                    test_cmd.remove("--recursive")
                if "--force" not in test_cmd:
                    test_cmd.append("--force")
                # xAI batch: use a separate state file for the test
                if grok_provider == "xai-batch" and has_grok:
                    test_state = os.path.join(test_dir, ".xai_test_state.json")
                    for i, arg in enumerate(test_cmd):
                        if arg == "--xai_batch_state_file":
                            test_cmd[i + 1] = test_state

                test_env = os.environ.copy()
                if api_key:
                    test_env["OPENROUTER_API_KEY"] = api_key
                if xai_api_key:
                    test_env["XAI_API_KEY"] = xai_api_key
                if hf_token:
                    test_env["HF_TOKEN"] = hf_token

                test_proc = subprocess.Popen(test_cmd, env=test_env)
                try:
                    test_proc.wait()
                except KeyboardInterrupt:
                    print_warning("\nInterrupted — killing test process...")
                    test_proc.kill()
                    test_proc.wait()
                    return False
                test_result = test_proc

                # For xAI batch: wait for the single request then collect
                if grok_provider == "xai-batch" and has_grok and test_result.returncode == 0:
                    test_state = None
                    for i, arg in enumerate(test_cmd):
                        if arg == "--xai_batch_state_file" and i + 1 < len(test_cmd):
                            test_state = test_cmd[i + 1]
                            break
                    if test_state and os.path.exists(test_state):
                        print_info("Waiting for xAI batch to process 1 test request...")
                        with open(test_state, "r", encoding="utf-8") as sf:
                            st = json.load(sf)
                        bid = st.get("batch_id", "")
                        xkey = xai_api_key or check_env_key("XAI_API_KEY")
                        if bid and xkey:
                            last_total = 0
                            for _ in range(60):  # max 5 min
                                time.sleep(5)
                                try:
                                    data = fetch_xai_batch_status(bid, xkey)
                                    last_total = int(data.get("state", {}).get("num_requests", 0) or 0)
                                    pending = int(data.get("state", {}).get("num_pending", 0) or 0)
                                    if pending <= 0 and last_total > 0:
                                        break
                                except Exception:
                                    pass
                            if last_total == 0:
                                print_warning("xAI batch returned 0 requests — test submission may have failed")
                            else:
                                collect_cmd = list(test_cmd)
                                for i, arg in enumerate(collect_cmd):
                                    if arg == "--xai_batch_action":
                                        collect_cmd[i + 1] = "collect"
                                subprocess.run(collect_cmd, env=test_env)

                if test_result.returncode != 0:
                    print_error("Test run FAILED")
                    if not ask_yes_no("Continue with full pipeline anyway?", default=False):
                        print_info("Aborted")
                        return False
                else:
                    # Show the generated caption
                    test_txt = os.path.splitext(test_link)[0] + ".txt"
                    if os.path.exists(test_txt):
                        with open(test_txt, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                        print_success("Test caption generated:")
                        console.print(f"\n[dim]{'─' * 50}[/]")
                        console.print(f"[italic]{caption[:500]}[/]")
                        if len(caption) > 500:
                            console.print(f"[dim]... ({len(caption)} chars total)[/]")
                        console.print(f"[dim]{'─' * 50}[/]\n")
                    else:
                        print_success("Test run completed (no caption file — may be tags-only mode)")

                    if not ask_yes_no("Looks good? Proceed with full dataset?", default=True):
                        print_info("Aborted")
                        return False

    # Run full pipeline
    print_section("STARTING PIPELINE")

    env = os.environ.copy()
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
    if xai_api_key:
        env["XAI_API_KEY"] = xai_api_key
    if hf_token:
        env["HF_TOKEN"] = hf_token

    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print_warning("\nInterrupted — killing tagger process...")
        proc.kill()
        proc.wait()
        return False

    if proc.returncode == 0:
        if xai_batch_action == "collect":
            print_success("COLLECT DONE! .txt files written next to your images.")
        else:
            print_success("DONE! Check your input directory for .txt files.")

        if has_grok and grok_provider == "xai-batch" and monitor_xai and xai_batch_action in ("submit", "status"):
            try:
                monitor_xai_batch(
                    state_file=xai_batch_state_file,
                    api_key=xai_api_key or env.get("XAI_API_KEY", ""),
                    base_url=XAI_API_BASE_URL,
                    poll_seconds=max(3, int(monitor_poll_seconds)),
                )
            except KeyboardInterrupt:
                print_warning("Monitoring stopped by user. Batch keeps running on xAI.")
            except Exception as e:
                print_error(f"Monitor error: {e}")
        return True
    else:
        print_error(f"Process exited with code {proc.returncode}")
        return False


# -------------------------
# Frame Pair Pipeline
# -------------------------

def run_frame_pair(input_dir: str, python: str, project_name: str = ""):
    """Run the frame-pair captioning pipeline (A→B scene transitions)."""
    from frame_pair_pipeline import (
        organize_pairs,
        run_caption_b,
        run_describe_a,
        run_similarity,
        run_upload,
        run_wd_tagging,
    )

    print_section("FRAME PAIR CAPTION PIPELINE")

    # xAI API key
    xai_api_key = check_env_key("XAI_API_KEY")
    if xai_api_key:
        print_success(f"Found XAI_API_KEY in environment ({xai_api_key[:4]}****)")
    else:
        xai_api_key = ask_input("Enter xAI API key")
        if not xai_api_key:
            print_error("No xAI API key provided. Pipeline requires xAI for captioning.")
            return

    # xAI model
    xai_model = ask_input("xAI model", XAI_BATCH_DEFAULT_MODEL)

    # GPU device
    device = ask_input("Device for similarity computation", "cuda")

    # Force re-tag existing images?
    force_retag = ask_yes_no("Force re-tag images (overwrite existing .txt tags)?", default=False)

    # Output directory
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(input_dir)),
        os.path.basename(os.path.abspath(input_dir)) + "_frame_pairs",
    )
    output_dir = ask_input("Output directory for organized pairs", default_output)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Organize
    try:
        print_section("STEP 1: ORGANIZE PAIRS")
        result = organize_pairs(input_dir, output_dir)
        pairs = result["pairs"]
        counts = result["counts"]
    except KeyboardInterrupt:
        print_warning("\nInterrupted during organization.")
        return

    print_summary_table("Pair Organization", [
        ("Dataset 1 (A+B+C)", str(counts.get("dataset_1", 0))),
        ("Dataset 2 (A+B+base)", str(counts.get("dataset_2", 0))),
        ("Dataset 3 (A+B only)", str(counts.get("dataset_3", 0))),
        ("Total pairs", str(len(pairs))),
    ])

    if not pairs:
        print_error("No valid pairs found. Check that your files follow the naming convention: *_A.ext, *_B.ext")
        return

    if not ask_yes_no(f"Found {len(pairs)} pairs. Proceed with pipeline?", default=True):
        print_info("Aborted")
        return

    # Step 2: WD/PixAI tagging (PixAI first, fallback to WD14)
    try:
        print_section("STEP 2: WD/PIXAI TAGGING")
        print_info(f"Tagging {len(pairs) * 2} images (PixAI → WD14 fallback)...")
        if force_retag:
            print_info("Force mode: will overwrite existing tags")
        tagging_ok = run_wd_tagging(pairs, python, force=force_retag)
        if not tagging_ok:
            print_error("Tagging failed — cannot proceed without WD tags.")
            if not ask_yes_no("Continue anyway (captions will lack tag context)?", default=False):
                return
    except KeyboardInterrupt:
        print_warning("\nInterrupted during tagging. Progress saved — you can re-run to resume.")
        return

    # Step 3: Describe A images
    try:
        if not ask_yes_no("Proceed to describe A images via Grok xAI Batch?", default=True):
            print_info("Skipped step 3")
        else:
            run_describe_a(pairs, xai_api_key, xai_model, output_dir, python)
    except KeyboardInterrupt:
        print_warning("\nInterrupted during A descriptions. State saved — re-run to resume batch.")
        return

    # Step 4: Similarity
    try:
        if not ask_yes_no("Proceed to compute similarity (CLIP + SSCD + SSIM)?", default=True):
            print_info("Skipped step 4 — using 50% default similarity for all pairs")
            similarities = [50.0] * len(pairs)
        else:
            similarities = run_similarity(pairs, device)
    except KeyboardInterrupt:
        print_warning("\nInterrupted during similarity computation.")
        return

    # Step 5: Caption B images
    try:
        if not ask_yes_no("Proceed to caption B images via Grok xAI Batch?", default=True):
            print_info("Skipped step 5")
            return
        run_caption_b(pairs, similarities, xai_api_key, xai_model, output_dir, python)
    except KeyboardInterrupt:
        print_warning("\nInterrupted during B captioning. State saved — re-run to resume batch.")
        return

    print_section("PIPELINE COMPLETE")
    print_success(f"Processed {len(pairs)} frame pairs")
    print_success(f"Output directory: {output_dir}")

    # Offer HuggingFace upload
    if ask_yes_no("Upload B images + captions to HuggingFace?", default=False):
        hf_token = check_env_key("HF_TOKEN") or check_env_key("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            hf_token = ask_input("HuggingFace token")
            if not hf_token:
                print_error("No HF token provided")
                return

        # Try to get username for default repo name
        default_repo = ""
        try:
            import subprocess as _sp
            r = _sp.run(
                [python, "-c",
                 "from huggingface_hub import HfApi; import os; "
                 "api = HfApi(token=os.environ.get('HF_TOKEN','')); "
                 "print(api.whoami()['name'])"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "HF_TOKEN": hf_token},
            )
            if r.returncode == 0 and r.stdout.strip():
                repo_suffix = project_name or os.path.basename(os.path.abspath(output_dir))
                default_repo = f"{r.stdout.strip()}/{repo_suffix}"
        except Exception:
            pass

        hf_repo = ask_input("Repository name (user/dataset)", default=default_repo)
        if not hf_repo or "/" not in hf_repo:
            print_error("Repository name must be in format user/dataset")
            return

        run_upload(output_dir, hf_token, hf_repo, python)


# -------------------------
# Main
# -------------------------

def main():
    print_banner()

    # Setup
    python = ensure_venv()
    install_deps(python)

    # Project name — detect existing projects or create new
    datasets_root = os.path.join(os.path.expanduser("~"), "datasets")
    existing_projects = []
    if os.path.isdir(datasets_root):
        existing_projects = sorted(
            d for d in os.listdir(datasets_root)
            if os.path.isdir(os.path.join(datasets_root, d))
            and not d.startswith(".")
        )

    if existing_projects:
        options = existing_projects + ["+ New project"]
        choice = ask_choice("Select project", options)
        if choice == len(options):
            project_name = ask_input("New project name")
        else:
            project_name = existing_projects[choice - 1]
    else:
        project_name = ask_input("Project name")

    if not project_name:
        print_error("Project name is required")
        sys.exit(1)

    # Resolve input source (local / HuggingFace / MEGA)
    default_target = os.path.join(os.path.expanduser("~"), "datasets", project_name)
    input_dir = resolve_input_source(python, default_target=default_target)

    # Handle zip archives
    zip_count = len(list_zip_archives(input_dir, recursive=True))
    if zip_count > 0:
        pending_zips = count_pending_zips(input_dir, recursive=True)
        if pending_zips == 0:
            print_success(f"Found {zip_count:,} zip files — all already extracted")
        else:
            print_info(f"Found {zip_count:,} zip files ({pending_zips:,} pending extraction)")
            if ask_yes_no("Extract pending zip files now?", default=True):
                report = extract_zip_archives(input_dir, recursive=True)
                print_success(
                    f"Zip extraction: extracted={report['extracted']}, "
                    f"skipped={report['skipped']}, failed={report['failed']}"
                )

    # Count media
    media_counts = count_media_quick(input_dir, recursive=True)
    if media_counts["images"] > 0 or media_counts["videos"] > 0:
        print_info(f"Found ~{media_counts['images']:,} images, ~{media_counts['videos']:,} videos in {input_dir}")
    else:
        print_warning(f"No media found in {input_dir} (subdirs will be scanned during processing)")

    # What to do?
    workflow = ask_choice("What do you want to do?", [
        "Pre-process dataset (trim to max frames)",
        "Tag dataset (wd14 / pixai / grok pipeline)",
        "Full pipeline (preprocess → tag → upload)",
        "Upload dataset to HuggingFace",
        "Frame Pair Caption (A→B scene-transition pipeline)",
    ], default=3)

    if workflow in (1, 3):
        run_preprocessing(input_dir)

    if workflow in (2, 3):
        tag_ok = run_tagging(input_dir, python, media_counts)
        if not tag_ok and workflow == 3:
            print_info("Pipeline stopped — tagging was aborted")
            return
        # After tagging, offer upload if not already in full pipeline
        if workflow == 2 and tag_ok:
            if ask_yes_no("Upload dataset to HuggingFace?", default=False):
                run_hf_upload(input_dir, python, project_name)

    if workflow in (3, 4):
        run_hf_upload(input_dir, python, project_name)

    if workflow == 5:
        run_frame_pair(input_dir, python, project_name)

    print_section("ALL DONE")
    print_success(f"Dataset ready at: {input_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
        sys.exit(130)
