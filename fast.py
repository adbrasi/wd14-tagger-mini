"""Fast preset-driven captioning pipeline for Anima LoRA datasets.

Run: `python fast.py`

Linear flow (no branching wizard):
    project name → input source → preset → trigger → caption → upload to HF.

Presets (see prompts/image/):
    1. anima-style    → trigger_style    (e.g. greg_rutkowski)
    2. anima-character→ trigger_character (e.g. naruto_uzumaki)
    3. anima-concept  → no trigger
    4. anima-outfit   → trigger_outfit   (e.g. red_kimono_v2)

Provider selection:
    Submits a single-image probe to the xAI Batch API and waits up to
    PROBE_TIMEOUT_SECS for completion. If the probe finishes in time, the
    main run uses xAI Batch (50% discount). Otherwise it falls back to
    xAI Sync (real-time, no discount) so the user is not blocked.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Venv bootstrap: re-exec under .venv if deps are missing.
# Must run BEFORE any third-party imports.
# ---------------------------------------------------------------------------

def _bootstrap_venv() -> None:
    try:
        import rich  # noqa: F401
        import requests  # noqa: F401
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    venv_root = here / ".venv"
    venv_python = venv_root / "bin" / "python"
    if not venv_python.exists():
        print(f"[fast] Creating virtual env at {venv_root}...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv_root)], check=True)
    pip = venv_python.parent / "pip"
    req = here / "requirements.txt"
    print("[fast] Installing dependencies (one-time)...", flush=True)
    subprocess.run([str(pip), "install", "-q", "-r", str(req)], check=True)
    print("[fast] Re-executing under venv.", flush=True)
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_venv()


import requests

# Importing readline makes input() honor arrow keys / backspace / history.
# Without it, terminals in raw mode print ^[[A etc. when arrows are pressed.
try:
    import readline  # noqa: F401
except ImportError:
    pass

import signal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
TAGGER_SCRIPT = SCRIPT_DIR / "tag_images_by_wd14_tagger.py"
PROMPTS_DIR = SCRIPT_DIR / "prompts" / "image"
WORKDIR_ROOT = SCRIPT_DIR / "_fast_workdir"
ENV_FILE = SCRIPT_DIR / ".env"

XAI_API_BASE_URL = "https://api.x.ai"
XAI_BATCH_MODEL = "grok-4-1-fast-reasoning"
PROBE_TIMEOUT_SECS = 60
PROBE_POLL_SECS = 4
BATCH_POLL_SECS = 15
BATCH_MAX_WAIT_SECS = int(os.environ.get("FAST_XAI_BATCH_MAX_WAIT_SECS", str(6 * 60 * 60)))

PRESETS = [
    {
        "id": "anima-style",
        "label": "Anima Style (artist/style LoRA)",
        "trigger_var": "trigger_style",
        "trigger_prompt": "Trigger tag of the style (e.g. greg_rutkowski)",
    },
    {
        "id": "anima-character",
        "label": "Anima Character (character LoRA)",
        "trigger_var": "trigger_character",
        "trigger_prompt": "Trigger name of the character (e.g. naruto_uzumaki)",
    },
    {
        "id": "anima-concept",
        "label": "Anima Concept (no trigger, descriptive)",
        "trigger_var": None,
        "trigger_prompt": None,
    },
    {
        "id": "anima-outfit",
        "label": "Anima Outfit (outfit LoRA)",
        "trigger_var": "trigger_outfit",
        "trigger_prompt": "Trigger name of the outfit (e.g. red_kimono_v2)",
    },
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

console = Console()


@dataclass
class RunConfig:
    project_name: str
    input_dir: Path
    preset_id: str
    trigger_var: Optional[str]
    trigger_value: Optional[str]
    xai_api_key: str
    hf_token: str


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]data_araknideo · fast mode[/bold cyan]\n"
        "[dim]preset → caption → huggingface in one shot[/dim]",
        border_style="cyan",
    ))


def step(title: str) -> None:
    console.print(f"\n[bold yellow]▸ {title}[/bold yellow]")


def info(msg: str) -> None:
    console.print(f"  [cyan]·[/cyan] {msg}")


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]![/yellow] {msg}")


def err(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")


# ---------------------------------------------------------------------------
# Env / API keys
# ---------------------------------------------------------------------------

def load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env (project-local) into os.environ if absent."""
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def save_to_dotenv(key: str, value: str) -> None:
    """Append KEY=VALUE to .env, preserving existing entries."""
    lines: list[str] = []
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and stripped.partition("=")[0].strip() == key:
                continue
            lines.append(raw_line)
    lines.append(f'{key}="{value}"')
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def require_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        err("HF_TOKEN missing. Export it before running:")
        console.print("    [dim]export HF_TOKEN=hf_...[/dim]")
        sys.exit(1)
    return token


def require_xai_key() -> str:
    key = os.environ.get("XAI_API_KEY")
    if key:
        return key
    warn("XAI_API_KEY not set.")
    key = Prompt.ask("Paste your xAI API key", password=True).strip()
    if not key:
        err("Empty key. Aborting.")
        sys.exit(1)
    save_to_dotenv("XAI_API_KEY", key)
    ok(f"Saved to {ENV_FILE.name} for next runs")
    return key


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def ask_project_name() -> str:
    while True:
        name = Prompt.ask("[bold]Project name[/bold]").strip()
        if not name:
            err("Project name is required.")
            continue
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        if sanitized != name:
            warn(f"Sanitized to: {sanitized}")
        return sanitized


def ask_input_source(project_name: str) -> Path:
    """Resolve the dataset to a local directory under WORKDIR_ROOT/<project>/."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]1[/bold]", "Local folder (path)")
    table.add_row("[bold]2[/bold]", "MEGA shared link")
    table.add_row("[bold]3[/bold]", "HuggingFace dataset (user/repo or URL)")
    console.print(table)
    choice = Prompt.ask("Input source", choices=["1", "2", "3"], default="1")

    project_workdir = WORKDIR_ROOT / project_name
    project_workdir.mkdir(parents=True, exist_ok=True)

    if choice == "1":
        while True:
            raw = Prompt.ask("Path to local folder").strip().rstrip("/")
            if not raw:
                err("Path is required.")
                continue
            path = Path(os.path.expanduser(raw)).resolve()
            if path.is_dir():
                return path
            err(f"Not a directory: {path}")
            warn("Tip: leading '/' makes the path absolute; otherwise it's relative to CWD.")

    if choice == "2":
        link = Prompt.ask("MEGA link").strip()
        return _download_mega(link, project_workdir / "raw")

    link = Prompt.ask("HuggingFace dataset (user/repo or URL)").strip()
    return _download_hf(link, project_workdir / "raw")


def _safe_zip_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip after checking for path traversal (zip slip)."""
    dest_resolved = dest.resolve()
    for member in zf.namelist():
        # zipfile normalizes "/" but we still need to reject absolute paths and "..".
        target = (dest_resolved / member).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError:
            raise RuntimeError(f"Refusing zip with unsafe member path: {member!r}")
    zf.extractall(dest_resolved)


def _download_mega(link: str, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPT_DIR))
    from mega_download import check_mega_installed, install_mega, mega_download, merge_directory

    if not check_mega_installed():
        info("Installing MEGAcmd...")
        if not install_mega():
            err("Could not install MEGAcmd")
            sys.exit(1)
    tmp = Path(tempfile.mkdtemp(prefix="fast_mega_"))
    try:
        if not mega_download(link, str(tmp)):
            err("MEGA download failed")
            sys.exit(1)
        # Auto-extract zips defensively against zip-slip.
        for fpath in tmp.rglob("*.zip"):
            if zipfile.is_zipfile(fpath):
                with zipfile.ZipFile(fpath, "r") as z:
                    _safe_zip_extract(z, tmp)
                fpath.unlink()
        stats = merge_directory(str(tmp), str(target))
        ok(f"MEGA: {stats.get('moved', 0)} files moved into {target}")
        return target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _download_hf(ref: str, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    repo_id = _parse_hf_ref(ref)
    info(f"Downloading {repo_id} → {target}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
    )
    return target


def _parse_hf_ref(ref: str) -> str:
    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        # https://huggingface.co/datasets/user/repo
        parts = ref.split("/datasets/", 1)
        if len(parts) != 2:
            err(f"Cannot parse HF URL: {ref}")
            sys.exit(1)
        tail = parts[1].split("?")[0].rstrip("/")
        segments = tail.split("/")
        return "/".join(segments[:2])
    if ref.count("/") == 1:
        return ref
    err(f"Invalid HF dataset reference: {ref}")
    sys.exit(1)


def ask_preset() -> dict:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for i, p in enumerate(PRESETS, start=1):
        table.add_row(f"[bold]{i}[/bold]", p["label"])
    console.print(table)
    choice = Prompt.ask(
        "Preset",
        choices=[str(i) for i in range(1, len(PRESETS) + 1)],
        default="2",
    )
    return PRESETS[int(choice) - 1]


def ask_trigger(preset: dict) -> Optional[str]:
    if preset["trigger_var"] is None:
        return None
    while True:
        value = Prompt.ask(preset["trigger_prompt"]).strip()
        if not value:
            err("Trigger is required for this preset.")
            continue
        # The tagger parses --prompt_var KEY=VALUE with split("=", 1), so a value
        # containing "=" would corrupt the key. Whitespace also breaks shell-style
        # tooling that may consume these args.
        if "=" in value or any(c.isspace() for c in value):
            err("Trigger cannot contain '=' or whitespace. Use underscores instead.")
            continue
        return value


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def scan_images(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (with_caption, without_caption) lists, recursive."""
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    with_caption: list[Path] = []
    without_caption: list[Path] = []
    for img in images:
        if img.with_suffix(".txt").exists():
            with_caption.append(img)
        else:
            without_caption.append(img)
    return with_caption, without_caption


def warn_videos_present(root: Path) -> bool:
    has_videos = any(p.is_file() and p.suffix.lower() in VIDEO_EXTS for p in root.rglob("*"))
    if has_videos:
        warn("Videos found in dataset — fast.py is image-only. Videos will be ignored.")
    return has_videos


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_tagger(args: list[str], env: Optional[dict] = None) -> int:
    """Run tag_images_by_wd14_tagger.py and propagate Ctrl+C to its process group.

    subprocess.run forwards SIGINT to the immediate child only; the tagger spawns
    32 worker threads via ThreadPoolExecutor and can take a while to drain.
    Putting the child in its own session (start_new_session=True) lets us SIGTERM
    the entire group on Ctrl+C, then escalate to SIGKILL if it doesn't exit.
    """
    cmd = [sys.executable, str(TAGGER_SCRIPT), *args]
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    proc = subprocess.Popen(cmd, env=process_env, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        warn("Ctrl+C — stopping tagger and its workers...")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        raise


def run_pixai_for_uncaptioned(input_dir: Path, no_caption: list[Path]) -> None:
    """Run pixai locally on images that lack a .txt sibling."""
    if not no_caption:
        info("All images already have captions — skipping pixai pre-pass.")
        return
    info(f"Running pixai on {len(no_caption)} images (no existing .txt)...")
    rc = run_tagger([
        str(input_dir),
        "--taggers", "pixai",
        "--recursive",
        "--remove_underscore",
        "--thresh", "0.30",
    ])
    if rc != 0:
        warn(f"pixai exited with code {rc} — continuing anyway")


# ---------------------------------------------------------------------------
# Probe: 1-image xAI batch with 60s deadline
# ---------------------------------------------------------------------------

def probe_xai_batch(
    config: RunConfig,
    sample_image: Path,
) -> str:
    """Submit a 1-image xAI batch and wait up to PROBE_TIMEOUT_SECS.

    Returns "xai-batch" if the probe completes in time, else "xai-sync".
    """
    info(f"Probe: 1-image xAI Batch with {PROBE_TIMEOUT_SECS}s deadline...")
    probe_dir = Path(tempfile.mkdtemp(prefix=f"fast_probe_{config.project_name}_"))
    state_file = probe_dir.parent / f".probe_state_{probe_dir.name}.json"
    try:
        # Symlink the sample image (and its sibling .txt if any) into probe_dir.
        link = probe_dir / sample_image.name
        try:
            link.symlink_to(sample_image)
        except OSError:
            shutil.copy2(sample_image, link)
        sibling_txt = sample_image.with_suffix(".txt")
        if sibling_txt.exists():
            try:
                (probe_dir / sibling_txt.name).symlink_to(sibling_txt)
            except OSError:
                shutil.copy2(sibling_txt, probe_dir / sibling_txt.name)

        # Submit a single-image batch.
        submit_args = [
            str(probe_dir),
            "--taggers", "grok",
            "--recursive",
            "--force",
            "--remove_underscore",
            "--grok_provider", "xai-batch",
            "--xai_batch_action", "submit",
            "--xai_batch_state_file", str(state_file),
            "--xai_batch_model", XAI_BATCH_MODEL,
            "--xai_batch_submit_chunk", "1",
            "--prompt_profile", config.preset_id,
            "--thresh", "0.30",
        ]
        if config.trigger_var and config.trigger_value:
            submit_args += ["--prompt_var", f"{config.trigger_var}={config.trigger_value}"]
        if sample_image.with_suffix(".txt").exists():
            submit_args.append("--grok_context_from_existing")

        rc = run_tagger(submit_args)
        if rc != 0 or not state_file.exists():
            warn("Probe submit failed — defaulting to xai-sync.")
            return "xai-sync"

        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            batch_id = state.get("batch_id")
        except Exception:
            batch_id = None
        if not batch_id:
            warn("Probe state has no batch_id — defaulting to xai-sync.")
            return "xai-sync"

        poll_start = time.time()
        deadline = poll_start + PROBE_TIMEOUT_SECS
        headers = {"Authorization": f"Bearer {config.xai_api_key}"}
        url = f"{XAI_API_BASE_URL}/v1/batches/{batch_id}"
        last_summary = "unknown"
        while time.time() < deadline:
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    # xAI batch progress lives in `state.num_*`. The top-level
                    # `status` field can stay "in_progress" even while requests
                    # have completed — what really matters is pending == 0.
                    counters = data.get("state") or {}
                    num_requests = int(counters.get("num_requests", 0))
                    num_pending = int(counters.get("num_pending", 0))
                    num_success = int(counters.get("num_success", 0))
                    num_error = int(counters.get("num_error", 0))
                    last_summary = (
                        f"req={num_requests} pending={num_pending} "
                        f"ok={num_success} err={num_error}"
                    )
                    # Single-image probe: any terminal result (success or error)
                    # tells us the queue is responsive.
                    if num_requests > 0 and num_pending == 0 and (num_success + num_error) >= num_requests:
                        elapsed = time.time() - poll_start
                        if num_success > 0:
                            ok(f"Batch fast (probe done in {elapsed:.1f}s) — using xAI Batch (50% discount).")
                            return "xai-batch"
                        warn(f"Probe ended with all errors ({last_summary}) — defaulting to xai-sync.")
                        return "xai-sync"
                    top_status = (data.get("status") or "").lower()
                    if top_status in ("failed", "cancelled", "expired"):
                        warn(f"Probe ended with status={top_status} — defaulting to xai-sync.")
                        return "xai-sync"
            except requests.RequestException as e:
                warn(f"Probe poll error: {e}")
            time.sleep(PROBE_POLL_SECS)

        warn(
            f"Batch slow ({last_summary} after {PROBE_TIMEOUT_SECS}s) — "
            "falling back to xAI Sync. Losing 50% batch discount."
        )
        # Cancel the orphan probe batch so it does not consume credits in the background.
        try:
            requests.post(
                f"{XAI_API_BASE_URL}/v1/batches/{batch_id}/cancel",
                headers=headers,
                timeout=10,
            )
        except requests.RequestException:
            pass
        return "xai-sync"
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
        try:
            state_file.unlink()
        except OSError:
            pass


def _read_batch_id(state_file: Path) -> str:
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as e:
        err(f"Cannot read xAI batch state file {state_file}: {e}")
        sys.exit(1)
    batch_id = state.get("batch_id")
    if not batch_id:
        err(f"xAI batch state file has no batch_id: {state_file}")
        sys.exit(1)
    return batch_id


def _get_batch_counters(api_key: str, batch_id: str) -> tuple[int, int, int, int]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{XAI_API_BASE_URL}/v1/batches/{batch_id}", headers=headers, timeout=30)
    resp.raise_for_status()
    counters = (resp.json().get("state") or {})
    total = int(counters.get("num_requests", 0) or 0)
    pending = int(counters.get("num_pending", 0) or 0)
    success = int(counters.get("num_success", 0) or 0)
    error_count = int(counters.get("num_error", 0) or 0)
    return total, pending, success, error_count


def wait_for_batch_complete(config: RunConfig, state_file: Path) -> None:
    batch_id = _read_batch_id(state_file)
    start = time.time()
    last_line = ""
    while True:
        try:
            total, pending, success, error_count = _get_batch_counters(config.xai_api_key, batch_id)
        except requests.RequestException as e:
            warn(f"Batch status poll failed: {e}")
            time.sleep(BATCH_POLL_SECS)
            continue

        elapsed = time.time() - start
        line = f"Batch status: total={total} pending={pending} success={success} error={error_count} elapsed={elapsed:.0f}s"
        if line != last_line:
            info(line)
            last_line = line

        if total > 0 and pending == 0 and (success + error_count) >= total:
            if error_count:
                warn(f"xAI Batch completed with {error_count}/{total} errors; failed items will use xAI Sync fallback.")
            return

        if elapsed > BATCH_MAX_WAIT_SECS:
            err(
                f"xAI Batch still pending after {BATCH_MAX_WAIT_SECS}s. "
                "Aborting before upload; run collect later when it finishes."
            )
            sys.exit(1)

        time.sleep(BATCH_POLL_SECS)


def incomplete_batch_image_paths(state_file: Path, input_dir: Path) -> list[Path]:
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as e:
        err(f"Cannot verify xAI batch state file {state_file}: {e}")
        sys.exit(1)

    request_map = state.get("request_map") or {}
    if not request_map:
        err("xAI batch state has no tracked requests after collect. Aborting before upload.")
        sys.exit(1)

    missing: list[Path] = []
    for meta in request_map.values():
        if not isinstance(meta, dict) or meta.get("state") == "succeeded":
            continue
        rel = meta.get("image_path_rel")
        raw_path = input_dir / rel if rel else Path(str(meta.get("image_path", "")))
        path = raw_path.resolve()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            missing.append(path)
    return sorted(set(missing))


def run_sync_fallback_for_images(
    config: RunConfig,
    images: list[Path],
    has_existing_captions: bool,
) -> None:
    if not images:
        return

    info(f"xAI Sync fallback for {len(images)} failed/missing batch captions...")
    fallback_dir = Path(tempfile.mkdtemp(prefix=f"fast_xai_sync_fallback_{config.project_name}_"))
    before_text: dict[Path, Optional[str]] = {}
    try:
        for image in images:
            try:
                rel = image.resolve().relative_to(config.input_dir.resolve())
            except ValueError:
                rel = Path(image.name)
            target = fallback_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(image)
            except OSError:
                shutil.copy2(image, target)

            sibling = image.with_suffix(".txt")
            before_text[image] = sibling.read_text(encoding="utf-8", errors="replace") if sibling.exists() else None
            if sibling.exists():
                txt_target = target.with_suffix(".txt")
                try:
                    txt_target.symlink_to(sibling)
                except OSError:
                    shutil.copy2(sibling, txt_target)

        args = [
            str(fallback_dir),
            "--taggers", "grok",
            "--recursive",
            "--force",
            "--remove_underscore",
            "--grok_provider", "xai-sync",
            "--prompt_profile", config.preset_id,
            "--grok_concurrency", str(min(16, max(1, len(images)))),
            "--thresh", "0.30",
        ]
        if has_existing_captions:
            args.append("--grok_context_from_existing")
        if config.trigger_var and config.trigger_value:
            args += ["--prompt_var", f"{config.trigger_var}={config.trigger_value}"]

        rc = run_tagger(args)
        if rc != 0:
            err(f"xAI Sync fallback failed (exit {rc}). Aborting before upload.")
            sys.exit(rc)

        still_missing: list[Path] = []
        for image in images:
            final_txt = image.with_suffix(".txt")
            try:
                rel = image.resolve().relative_to(config.input_dir.resolve())
            except ValueError:
                rel = Path(image.name)
            fallback_txt = (fallback_dir / rel).with_suffix(".txt")
            if fallback_txt.exists() and fallback_txt.stat().st_size > 0:
                new_text = fallback_txt.read_text(encoding="utf-8", errors="replace")
                if before_text.get(image) is not None and new_text == before_text[image]:
                    still_missing.append(image)
                    continue
                final_txt.parent.mkdir(parents=True, exist_ok=True)
                try:
                    same_file = final_txt.exists() and fallback_txt.samefile(final_txt)
                except OSError:
                    same_file = False
                if not same_file:
                    shutil.copy2(fallback_txt, final_txt)
            else:
                still_missing.append(image)

        if still_missing:
            err(f"{len(still_missing)} fallback captions are still missing. Aborting before upload.")
            for image in still_missing[:10]:
                console.print(f"    [dim]{image}[/dim]")
            sys.exit(1)
        ok(f"xAI Sync fallback wrote {len(images)} captions.")
    finally:
        shutil.rmtree(fallback_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Preview gate: caption ONE image, show it, ask Y/N before the full run.
# ---------------------------------------------------------------------------

def preview_and_confirm(config: RunConfig, sample_image: Path) -> None:
    """Caption a single image via xai-sync, show the result, abort if user says N.

    Always uses xai-sync regardless of the chosen provider — the preview's job is
    to validate the prompt/trigger config quickly, not to benchmark the full run.
    """
    info(f"Captioning preview image: {sample_image.name}")
    preview_dir = Path(tempfile.mkdtemp(prefix=f"fast_preview_{config.project_name}_"))
    try:
        link = preview_dir / sample_image.name
        try:
            link.symlink_to(sample_image)
        except OSError:
            shutil.copy2(sample_image, link)
        sibling = sample_image.with_suffix(".txt")
        has_sibling_txt = sibling.exists()
        if has_sibling_txt:
            try:
                (preview_dir / sibling.name).symlink_to(sibling)
            except OSError:
                shutil.copy2(sibling, preview_dir / sibling.name)

        args = [
            str(preview_dir),
            "--taggers", "grok",
            "--recursive",
            "--force",
            "--remove_underscore",
            "--grok_provider", "xai-sync",
            "--prompt_profile", config.preset_id,
            "--grok_concurrency", "1",
            "--thresh", "0.30",
        ]
        if has_sibling_txt:
            args.append("--grok_context_from_existing")
        if config.trigger_var and config.trigger_value:
            args += ["--prompt_var", f"{config.trigger_var}={config.trigger_value}"]

        rc = run_tagger(args)
        if rc != 0:
            err(f"Preview run failed (exit {rc}). Aborting.")
            sys.exit(rc)

        result_txt = preview_dir / sample_image.with_suffix(".txt").name
        if not result_txt.exists() or result_txt.stat().st_size == 0:
            err("Preview produced no .txt output — Grok call probably failed. Aborting.")
            sys.exit(1)
        caption = result_txt.read_text(encoding="utf-8", errors="replace").strip()

        console.print()
        console.print(Panel.fit(
            f"[bold]{sample_image.name}[/bold]\n\n{caption}",
            title="[bold cyan]preview caption[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        ))
        console.print()
        if not Confirm.ask("Looks good? Proceed with the full dataset?", default=True):
            warn("Aborted by user. Re-run after adjusting prompts/trigger.")
            sys.exit(0)
    finally:
        shutil.rmtree(preview_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main captioning
# ---------------------------------------------------------------------------

def caption_main(config: RunConfig, provider: str, has_existing_captions: bool) -> None:
    """Run the final captioning pass over the whole dataset."""
    state_file = SCRIPT_DIR / f"_fast_state_{config.project_name}.json"

    base_args = [
        str(config.input_dir),
        "--taggers", "grok",
        "--recursive",
        "--force",
        "--remove_underscore",
        "--grok_provider", provider,
        "--prompt_profile", config.preset_id,
        "--thresh", "0.30",
    ]
    # Only pass --grok_context_from_existing when at least one image has a
    # sibling .txt to use as context. Passing it on a dataset without prior
    # captions (or after the pixai pre-pass overwrote them) makes the tagger
    # try to read non-existent files.
    if has_existing_captions:
        base_args.append("--grok_context_from_existing")
    if config.trigger_var and config.trigger_value:
        base_args += ["--prompt_var", f"{config.trigger_var}={config.trigger_value}"]

    if provider == "xai-batch":
        # Submit, wait until the remote batch is complete, then collect all results.
        submit_args = base_args + [
            "--xai_batch_action", "submit",
            "--xai_batch_state_file", str(state_file),
            "--xai_batch_model", XAI_BATCH_MODEL,
        ]
        info("Submitting xAI Batch...")
        rc = run_tagger(submit_args)
        if rc != 0:
            err(f"Batch submit failed (exit {rc})")
            sys.exit(rc)
        info("Waiting for xAI Batch to finish before collect/upload...")
        wait_for_batch_complete(config, state_file)
        collect_args = base_args + [
            "--xai_batch_action", "collect",
            "--xai_batch_state_file", str(state_file),
            "--xai_batch_model", XAI_BATCH_MODEL,
        ]
        info("Collecting xAI Batch results...")
        rc = run_tagger(collect_args)
        if rc != 0:
            err(f"Batch collect failed (exit {rc})")
            sys.exit(rc)
        missing = incomplete_batch_image_paths(state_file, config.input_dir)
        run_sync_fallback_for_images(config, missing, has_existing_captions)
    else:
        info("Running xAI Sync (real-time, no discount)...")
        # 64 concurrent requests is well within xAI's grok-4-fast tier limit
        # and gives ~4x the throughput of the previous default (16).
        rc = run_tagger(base_args + ["--grok_concurrency", "64"])
        if rc != 0:
            err(f"Sync run failed (exit {rc})")
            sys.exit(rc)


# ---------------------------------------------------------------------------
# Sample preview
# ---------------------------------------------------------------------------

def show_samples(input_dir: Path, n: int = 3) -> None:
    pairs = []
    for img in input_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            txt = img.with_suffix(".txt")
            if txt.exists() and txt.stat().st_size > 0:
                pairs.append((img, txt))
    if not pairs:
        warn("No (image, .txt) pairs to preview.")
        return
    sample = random.sample(pairs, min(n, len(pairs)))
    console.print()
    for img, txt in sample:
        caption = txt.read_text(encoding="utf-8", errors="replace").strip()
        console.print(Panel.fit(
            f"[bold]{img.name}[/bold]\n\n{caption}",
            border_style="green",
            padding=(0, 1),
        ))


# ---------------------------------------------------------------------------
# HF upload (zip chunks → public dataset)
# ---------------------------------------------------------------------------

def hf_username(token: str) -> str:
    from huggingface_hub import HfApi
    return HfApi(token=token).whoami()["name"]


def build_zip_chunks(input_dir: Path, output_dir: Path, project_name: str, chunk_gb: float = 5.0) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(chunk_gb * (1024 ** 3))
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and (p.suffix.lower() in IMAGE_EXTS or p.suffix.lower() == ".txt")
    )
    if not files:
        return 0
    part = 1
    cur_bytes = 0
    cur_zip: Optional[zipfile.ZipFile] = None
    parts_written = 0
    try:
        for f in files:
            size = f.stat().st_size
            if cur_zip is None or (cur_bytes > 0 and cur_bytes + size > max_bytes):
                if cur_zip is not None:
                    cur_zip.close()
                    parts_written += 1
                zpath = output_dir / f"{project_name}-part{part:03d}.zip"
                cur_zip = zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED)
                part += 1
                cur_bytes = 0
            rel = f.relative_to(input_dir)
            cur_zip.write(f, str(rel))
            cur_bytes += size
        if cur_zip is not None:
            cur_zip.close()
            parts_written += 1
    except Exception:
        if cur_zip is not None:
            cur_zip.close()
        raise
    return parts_written


def upload_to_hf(config: RunConfig) -> None:
    from huggingface_hub import HfApi

    user = hf_username(config.hf_token)
    repo_id = f"{user}/{config.project_name}"
    info(f"Building zip chunks for {repo_id}...")

    zips_dir = WORKDIR_ROOT / config.project_name / "_zips"
    if zips_dir.exists():
        shutil.rmtree(zips_dir)
    parts = build_zip_chunks(config.input_dir, zips_dir, config.project_name)
    if parts == 0:
        warn("No files to upload.")
        return
    ok(f"Built {parts} zip chunks → {zips_dir}")

    api = HfApi(token=config.hf_token)
    info(f"Creating public dataset repo: {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

    info("Uploading (resumable, multi-threaded)...")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(zips_dir),
        allow_patterns=["*.zip"],
    )
    ok(f"Done: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    banner()
    load_dotenv()

    step("Checking credentials")
    hf_token = require_hf_token()
    ok("HF_TOKEN present")

    step("Project")
    project_name = ask_project_name()

    step("Input source")
    input_dir = ask_input_source(project_name)
    ok(f"Input: {input_dir}")

    has_videos = warn_videos_present(input_dir)
    with_caption, without_caption = scan_images(input_dir)
    total = len(with_caption) + len(without_caption)
    if total == 0:
        if has_videos:
            err("Dataset contains only videos — fast.py is image-only. Use the legacy cli.py for video pipelines.")
        else:
            err("No images found in input directory.")
        sys.exit(1)
    info(f"Found {total} images ({len(with_caption)} with .txt, {len(without_caption)} without)")

    step("Preset")
    preset = ask_preset()
    trigger_value = ask_trigger(preset)
    if trigger_value:
        ok(f"{preset['trigger_var']} = {trigger_value}")
    else:
        ok("No trigger (concept preset)")

    step("xAI key")
    xai_key = require_xai_key()
    ok("XAI_API_KEY present")

    config = RunConfig(
        project_name=project_name,
        input_dir=input_dir,
        preset_id=preset["id"],
        trigger_var=preset["trigger_var"],
        trigger_value=trigger_value,
        xai_api_key=xai_key,
        hf_token=hf_token,
    )

    if without_caption:
        step("Local pixai pre-pass")
        run_pixai_for_uncaptioned(input_dir, without_caption)

    # Re-scan after pixai pre-pass: at this point most/all images should have .txt.
    post_pixai_with_caption, _ = scan_images(input_dir)
    has_existing_captions = len(post_pixai_with_caption) > 0

    step("Probe xAI Batch (60s)")
    sample = random.choice(with_caption + without_caption)
    provider = probe_xai_batch(config, sample)

    step("Preview (1 image, awaits your approval)")
    preview_sample = random.choice(post_pixai_with_caption or [sample])
    preview_and_confirm(config, preview_sample)

    step(f"Captioning ({provider})")
    caption_main(config, provider, has_existing_captions)

    step("Preview")
    show_samples(input_dir)

    step("Uploading to HuggingFace")
    upload_to_hf(config)

    console.print(Panel.fit(
        "[bold green]Done.[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(130)
