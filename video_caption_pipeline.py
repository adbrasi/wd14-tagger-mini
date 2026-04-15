"""Video Caption Pipeline: Gemini (video understanding) + PixAI (keyframe tags) + Grok (synthesis).

Three-phase pipeline:
1. Gemini Flash Lite watches each video and writes a chronological description
2. PixAI tags 5 keyframes per video (0%, 25%, 50%, 75%, 100%)
3. Grok synthesizes everything into a final caption (text-only, never receives images)

Fallback: if Gemini refuses a video, Grok receives only tags + PixAI tags.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constants import VIDEO_EXTS
from wd14_utils import extract_frames, glob_videos_pathlib, setup_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAGGER_SCRIPT = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "video-caption")

# Keyframe positions as percentages
KEYFRAME_POSITIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
NUM_KEYFRAMES = len(KEYFRAME_POSITIONS)


# -------------------------
# Prompt loading
# -------------------------

def load_prompt(profile: str, filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    path = os.path.join(PROMPTS_DIR, profile, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# -------------------------
# Phase 1: Gemini video descriptions
# -------------------------

def run_gemini_phase(
    video_paths: List[str],
    gemini_api_key: str,
    profile: str = "default",
    model: str = "gemini-3.1-flash-lite-preview",
    poll_interval: int = 20,
    on_upload_progress=None,
    on_poll_progress=None,
) -> Dict[str, Optional[str]]:
    """Run Gemini batch to get video descriptions.

    Returns mapping of video_path -> description text (None for failures/refusals).
    """
    from gemini_batch import run_gemini_video_descriptions

    system_prompt = load_prompt(profile, "gemini_system_prompt.md")
    user_prompt = "Describe this video following your system instructions."

    return run_gemini_video_descriptions(
        video_paths=video_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=gemini_api_key,
        poll_interval=poll_interval,
        on_upload_progress=on_upload_progress,
        on_poll_progress=on_poll_progress,
    )


# -------------------------
# Phase 2: PixAI keyframe tagging
# -------------------------

def _compute_frame_numbers(video_path: str) -> List[int]:
    """Compute absolute frame numbers for 5 keyframe positions."""
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python-headless is required")
        return [0, 1, 2, 3, 4]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0, 1, 2, 3, 4]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        return [0, 1, 2, 3, 4]

    last = max(total_frames - 1, 0)
    return [min(int(pos * last), last) for pos in KEYFRAME_POSITIONS]


def extract_keyframes(
    video_paths: List[str],
    temp_dir: str,
    max_workers: int = 64,
) -> Dict[str, List[str]]:
    """Extract 5 keyframes per video, save as JPEG.

    Returns mapping of video_path -> [frame_0.jpg, frame_1.jpg, ..., frame_4.jpg]
    """
    from PIL import Image

    result: Dict[str, List[str]] = {}

    def _extract_single(vpath: str) -> Tuple[str, List[str]]:
        frame_numbers = _compute_frame_numbers(vpath)
        frames = extract_frames(vpath, frame_numbers)

        stem = Path(vpath).stem
        uid = hashlib.md5(vpath.encode()).hexdigest()[:10]
        saved = []

        for i, frame in enumerate(frames):
            if frame is None:
                saved.append("")  # placeholder for missing frame
                continue
            out_path = os.path.join(temp_dir, f"{stem}_{uid}_kf{i}.jpg")
            frame.save(out_path, "JPEG", quality=95)
            saved.append(out_path)

        return vpath, saved

    workers = min(max_workers, len(video_paths), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_single, vp): vp for vp in video_paths}
        for future in as_completed(futures):
            try:
                vpath, frame_paths = future.result()
                result[vpath] = frame_paths
            except Exception as e:
                vpath = futures[future]
                logger.warning(f"keyframe extraction failed for {vpath}: {e}")
                result[vpath] = [""] * NUM_KEYFRAMES

    return result


def _resolve_batch_size(batch_size: str, python: str) -> str:
    """Resolve 'auto' batch size to a concrete integer string."""
    if batch_size.lower() != "auto":
        return batch_size
    try:
        r = subprocess.run(
            [python, "-c",
             "from tag_images_by_wd14_tagger import recommend_batch_by_vram; "
             "r = recommend_batch_by_vram(); print(r if r else 4)"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "4"


def run_pixai_on_frames(
    keyframes: Dict[str, List[str]],
    python: str,
    batch_size: str = "auto",
) -> Dict[str, Dict[str, str]]:
    """Run PixAI tagger on all extracted keyframes.

    Returns mapping of video_path -> {"frame_1": "tag1, tag2, ...", ..., "frame_5": "..."}
    """
    # Resolve batch_size before passing to subprocess (argparse expects int)
    batch_size = _resolve_batch_size(batch_size, python)

    # Collect all frame paths that exist
    all_frames = []
    frame_to_video: Dict[str, Tuple[str, int]] = {}  # frame_path -> (video_path, frame_index)

    for vpath, frame_paths in keyframes.items():
        for i, fpath in enumerate(frame_paths):
            if fpath and os.path.exists(fpath):
                all_frames.append(fpath)
                frame_to_video[fpath] = (vpath, i)

    if not all_frames:
        logger.warning("no keyframes to tag")
        return {}

    # Create a temp directory with symlinks to all frames
    with tempfile.TemporaryDirectory(prefix="pixai_kf_") as tag_dir:
        for fpath in all_frames:
            link = os.path.join(tag_dir, os.path.basename(fpath))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(fpath), link)

        # Run PixAI tagger
        cmd = [
            python, TAGGER_SCRIPT, tag_dir,
            "--taggers", "pixai",
            "--batch_size", batch_size,
            "--force",
            "--remove_underscore",
            "--caption_separator", ", ",
        ]

        env = os.environ.copy()
        logger.info(f"running PixAI on {len(all_frames)} keyframes...")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"PixAI tagger failed: {proc.stderr[-500:]}")

        # Read generated .txt files
        results: Dict[str, Dict[str, str]] = {}
        for fpath in all_frames:
            txt_path = os.path.join(tag_dir, os.path.splitext(os.path.basename(fpath))[0] + ".txt")
            tags = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    tags = f.read().strip()

            vpath, idx = frame_to_video[fpath]
            if vpath not in results:
                results[vpath] = {}
            results[vpath][f"frame_{idx + 1}"] = tags

    # Fill missing frames
    for vpath in keyframes:
        if vpath not in results:
            results[vpath] = {}
        for i in range(NUM_KEYFRAMES):
            key = f"frame_{i + 1}"
            if key not in results[vpath]:
                results[vpath][key] = "(no tags)"

    return results


# -------------------------
# Phase 3: Grok synthesis
# -------------------------

def _load_existing_tags(video_paths: List[str]) -> Dict[str, str]:
    """Load existing .txt files next to videos as original tags."""
    tags: Dict[str, str] = {}
    for vpath in video_paths:
        txt_path = os.path.splitext(vpath)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                tags[vpath] = f.read().strip()
        else:
            tags[vpath] = "(no tags)"
    return tags


def build_grok_requests(
    video_paths: List[str],
    gemini_results: Dict[str, Optional[str]],
    pixai_results: Dict[str, Dict[str, str]],
    original_tags: Dict[str, str],
    grok_user_prompt_template: str,
) -> Dict[str, str]:
    """Build the text-only user prompt for each video.

    Returns mapping of video_path -> rendered user prompt.
    """
    prompts: Dict[str, str] = {}

    for vpath in video_paths:
        gemini_desc = gemini_results.get(vpath)
        pixai = pixai_results.get(vpath, {})
        tags = original_tags.get(vpath, "(no tags)")

        prompt = grok_user_prompt_template
        prompt = prompt.replace("{original_tags}", tags)
        prompt = prompt.replace("{source_description}", gemini_desc or "(not available)")
        for i in range(NUM_KEYFRAMES):
            key = f"frame_{i + 1}"
            prompt = prompt.replace(f"{{pixai_frame_{i + 1}}}", pixai.get(key, "(no tags)"))

        prompts[vpath] = prompt

    return prompts


def run_grok_phase(
    video_paths: List[str],
    grok_prompts: Dict[str, str],
    grok_system_prompt: str,
    xai_api_key: str,
    xai_model: str = "grok-4-1-fast-reasoning",
    input_dir: str = ".",
    python: str = "python",
    prepend_text: str = "",
) -> Dict[str, Optional[str]]:
    """Submit Grok synthesis via xAI Batch API (direct calls, no subprocess).

    Sends text-only requests (no images) with the pre-built prompts directly
    to the xAI batch API, bypassing the tagger entirely.

    Returns mapping of video_path -> final caption.
    """
    import requests as req
    import time

    if not xai_api_key:
        raise ValueError("xAI API key is required for Grok phase")

    base_url = "https://api.x.ai"
    headers = {
        "Authorization": f"Bearer {xai_api_key}",
        "Content-Type": "application/json",
    }

    def _api(method, path, payload=None, params=None):
        for attempt in range(6):
            try:
                r = req.request(
                    method, f"{base_url}{path}",
                    headers=headers, json=payload, params=params, timeout=120,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"xAI {r.status_code} on {path}, retry in {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except req.exceptions.RequestException as e:
                if attempt >= 5:
                    raise
                logger.warning(f"xAI request error: {e}, retrying...")
                time.sleep(min(2 ** attempt, 30))
        return {}

    # Step 1: Create batch
    logger.info("creating xAI batch...")
    batch_data = _api("POST", "/v1/batches", {"name": f"vcap_grok_{int(time.time())}"})
    batch_id = batch_data.get("batch_id")
    if not batch_id:
        logger.error(f"failed to create xAI batch: {batch_data}")
        return {vp: None for vp in video_paths}

    logger.info(f"xAI batch created: {batch_id}")

    # Step 2: Submit requests (text-only, no images)
    ordered_paths: List[str] = []
    req_ids: Dict[str, str] = {}  # req_id -> video_path

    batch_requests = []
    for vpath, user_prompt in grok_prompts.items():
        ordered_paths.append(vpath)
        req_entry = {
            "custom_id": hashlib.sha1(vpath.encode()).hexdigest()[:16],
            "params": {
                "model": xai_model,
                "messages": [
                    {"role": "system", "content": grok_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        }
        batch_requests.append(req_entry)

    # Submit in chunks of 500
    chunk_size = 500
    submitted = 0
    for i in range(0, len(batch_requests), chunk_size):
        chunk = batch_requests[i : i + chunk_size]
        try:
            _api("POST", f"/v1/batches/{batch_id}/requests", {"batch_requests": chunk})
            submitted += len(chunk)
            logger.info(f"submitted {submitted}/{len(batch_requests)} requests")
        except Exception as e:
            logger.error(f"failed to submit chunk at offset {i}: {e}")

    if submitted == 0:
        logger.error("no requests were submitted")
        return {vp: None for vp in video_paths}

    # Step 3: Poll until done
    logger.info(f"polling xAI batch {batch_id}...")
    start = time.time()
    timeout = 7200
    tail_start = None

    while (time.time() - start) < timeout:
        try:
            status = _api("GET", f"/v1/batches/{batch_id}")
            counters = status.get("state", {})
            total = int(counters.get("num_requests", 0) or 0)
            pending = int(counters.get("num_pending", 0) or 0)
            success = int(counters.get("num_success", 0) or 0)
            errors = int(counters.get("num_error", 0) or 0)
            done = success + errors

            pct = (done / total * 100) if total else 0
            logger.info(f"xAI batch: {done}/{total} ({pct:.1f}%) pending={pending}")

            if pending <= 0 and total > 0:
                logger.info("xAI batch complete")
                break

            # Tail timeout: 2 min for stragglers at 99%+
            if total > 0 and pct >= 99.0 and pending > 0:
                if tail_start is None:
                    tail_start = time.time()
                elif time.time() - tail_start >= 120:
                    logger.warning(f"tail timeout: {pending} straggler(s), moving on")
                    break
            else:
                tail_start = None

        except Exception as e:
            logger.warning(f"poll error: {e}")

        time.sleep(20)

    # Step 4: Collect results
    logger.info("collecting xAI batch results...")
    results: Dict[str, Optional[str]] = {}
    custom_id_to_path = {
        hashlib.sha1(vp.encode()).hexdigest()[:16]: vp for vp in ordered_paths
    }

    page_token = None
    collected = 0

    while True:
        params = {"limit": "100"}
        if page_token:
            params["after"] = page_token

        try:
            page = _api("GET", f"/v1/batches/{batch_id}/requests", params=params)
        except Exception as e:
            logger.error(f"collect error: {e}")
            break

        items = page.get("data", [])
        if not items:
            break

        for item in items:
            custom_id = item.get("custom_id", "")
            vpath = custom_id_to_path.get(custom_id)
            if vpath is None:
                continue

            response = item.get("response", {})
            status_code = response.get("status_code", 0)

            if status_code != 200:
                logger.warning(f"grok error for {vpath}: status {status_code}")
                results[vpath] = None
                continue

            # Extract caption from response body
            body = response.get("body", {})
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                caption = _extract_caption(content)
                results[vpath] = caption
                collected += 1
            else:
                results[vpath] = None

        # Pagination
        if page.get("has_more"):
            page_token = items[-1].get("id")
        else:
            break

    logger.info(f"collected {collected}/{len(ordered_paths)} captions from xAI batch")

    # Mark missing
    for vp in video_paths:
        if vp not in results:
            results[vp] = None

    return results


def _extract_caption(text: str) -> Optional[str]:
    """Extract caption from Grok's JSON response."""
    # Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "caption" in data:
            return data["caption"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON in text
    import re
    match = re.search(r'\{[^{}]*"caption"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', text, re.DOTALL)
    if match:
        return match.group(1).replace('\\"', '"').replace('\\n', '\n')

    # Fallback: use raw text if it looks like a caption
    if text and len(text) > 50:
        return text

    return None


# -------------------------
# Output writing
# -------------------------

def write_captions(
    results: Dict[str, Optional[str]],
    prepend_text: str = "",
):
    """Write final captions as .txt files next to original videos."""
    written = 0
    for vpath, caption in results.items():
        if caption is None:
            logger.warning(f"no caption for {vpath}, skipping")
            continue

        txt_path = os.path.splitext(vpath)[0] + ".txt"
        final = f"{prepend_text}{caption}" if prepend_text else caption

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final)
        written += 1

    logger.info(f"wrote {written} caption files")
    return written


# -------------------------
# Full pipeline
# -------------------------

def run_pipeline(
    input_dir: str,
    gemini_api_key: str,
    xai_api_key: str,
    python: str = "python",
    profile: str = "default",
    gemini_model: str = "gemini-3.1-flash-lite-preview",
    xai_model: str = "grok-4-1-fast-reasoning",
    batch_size: str = "auto",
    recursive: bool = True,
    prepend_text: str = "",
    on_phase_progress=None,
) -> Dict[str, Any]:
    """Run the full 3-phase video caption pipeline.

    Args:
        input_dir: directory containing video files
        gemini_api_key: Google AI API key
        xai_api_key: xAI API key
        python: python executable path
        profile: prompt profile name
        gemini_model: Gemini model ID
        xai_model: xAI model ID
        batch_size: PixAI batch size
        recursive: search subdirectories
        prepend_text: text to prepend to every caption
        on_phase_progress: callback(phase_name, detail)

    Returns stats dict.
    """
    setup_logging()

    # Discover videos
    video_paths_list = glob_videos_pathlib(Path(input_dir), recursive)
    video_paths = [str(p) for p in video_paths_list]

    if not video_paths:
        logger.warning("no videos found")
        return {"total": 0, "captioned": 0, "failed": 0}

    logger.info(f"found {len(video_paths)} videos")

    # Load existing .txt as original tags
    original_tags = _load_existing_tags(video_paths)

    stats = {"total": len(video_paths), "captioned": 0, "failed": 0}

    # ── Phase 1: Gemini ──
    if on_phase_progress:
        on_phase_progress("gemini", f"processing {len(video_paths)} videos...")

    def _on_upload(done, total):
        if on_phase_progress:
            on_phase_progress("gemini_upload", f"uploading {done}/{total} videos...")

    def _on_poll(state_name, batch_job):
        if on_phase_progress:
            on_phase_progress("gemini_poll", f"batch state: {state_name}")

    gemini_results = run_gemini_phase(
        video_paths,
        gemini_api_key=gemini_api_key,
        profile=profile,
        model=gemini_model,
        on_upload_progress=_on_upload,
        on_poll_progress=_on_poll,
    )

    gemini_ok = sum(1 for v in gemini_results.values() if v is not None)
    gemini_fail = sum(1 for v in gemini_results.values() if v is None)
    logger.info(f"gemini: {gemini_ok} succeeded, {gemini_fail} refused/failed")

    if on_phase_progress:
        on_phase_progress("gemini_done", f"{gemini_ok} ok, {gemini_fail} refused")

    # ── Phase 2: PixAI keyframes ──
    if on_phase_progress:
        on_phase_progress("pixai", "extracting keyframes and tagging...")

    with tempfile.TemporaryDirectory(prefix="vcap_keyframes_") as kf_dir:
        keyframes = extract_keyframes(video_paths, kf_dir)
        pixai_results = run_pixai_on_frames(keyframes, python, batch_size)

    if on_phase_progress:
        on_phase_progress("pixai_done", f"tagged {len(pixai_results)} videos")

    # ── Phase 3: Grok synthesis ──
    if on_phase_progress:
        on_phase_progress("grok", "synthesizing captions...")

    grok_system_prompt = load_prompt(profile, "grok_system_prompt.md")
    grok_user_template = load_prompt(profile, "grok_user_prompt.md")

    grok_prompts = build_grok_requests(
        video_paths, gemini_results, pixai_results, original_tags,
        grok_user_template,
    )

    grok_results = run_grok_phase(
        video_paths,
        grok_prompts,
        grok_system_prompt,
        xai_api_key=xai_api_key,
        xai_model=xai_model,
        input_dir=input_dir,
        python=python,
        prepend_text=prepend_text,
    )

    # ── Write output ──
    written = write_captions(grok_results, prepend_text)
    stats["captioned"] = written
    stats["failed"] = stats["total"] - written

    if on_phase_progress:
        on_phase_progress("done", f"{written}/{stats['total']} captions written")

    return stats


# -------------------------
# CLI entrypoint
# -------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Caption Pipeline: Gemini + PixAI + Grok",
    )
    parser.add_argument("input_dir", help="directory containing video files")
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Google AI API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--xai_api_key", default=os.environ.get("XAI_API_KEY", ""),
                        help="xAI API key (or set XAI_API_KEY env var)")
    parser.add_argument("--gemini_model", default="gemini-3.1-flash-lite-preview",
                        help="Gemini model ID")
    parser.add_argument("--xai_model", default="grok-4-1-fast-reasoning",
                        help="Grok model ID for xAI batch")
    parser.add_argument("--profile", default="default",
                        help="prompt profile name (folder under prompts/video-caption/)")
    parser.add_argument("--batch_size", default="auto",
                        help="PixAI batch size (int or 'auto')")
    parser.add_argument("--prepend_text", default="",
                        help="text to prepend to every .txt output")
    parser.add_argument("--no_recursive", action="store_true",
                        help="do not search subdirectories")
    parser.add_argument("--python", default=None,
                        help="python executable (default: auto-detect venv)")

    args = parser.parse_args()

    if not args.gemini_api_key:
        parser.error("Gemini API key required: --gemini_api_key or GEMINI_API_KEY env var")
    if not args.xai_api_key:
        parser.error("xAI API key required: --xai_api_key or XAI_API_KEY env var")

    # Auto-detect venv python if not specified
    python = args.python
    if python is None:
        venv_python = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")
        python = venv_python if os.path.exists(venv_python) else "python"

    def _progress(phase, detail):
        print(f"[{phase}] {detail}")

    stats = run_pipeline(
        input_dir=args.input_dir,
        gemini_api_key=args.gemini_api_key,
        xai_api_key=args.xai_api_key,
        python=python,
        profile=args.profile,
        gemini_model=args.gemini_model,
        xai_model=args.xai_model,
        batch_size=args.batch_size,
        recursive=not args.no_recursive,
        prepend_text=args.prepend_text,
        on_phase_progress=_progress,
    )

    print(f"\nDone: {stats['captioned']}/{stats['total']} captioned, {stats['failed']} failed")


if __name__ == "__main__":
    main()
