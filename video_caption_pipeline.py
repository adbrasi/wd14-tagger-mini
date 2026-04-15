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


def run_pixai_on_frames(
    keyframes: Dict[str, List[str]],
    python: str,
    batch_size: str = "auto",
) -> Dict[str, Dict[str, str]]:
    """Run PixAI tagger on all extracted keyframes.

    Returns mapping of video_path -> {"frame_1": "tag1, tag2, ...", ..., "frame_5": "..."}
    """
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
        prompt = prompt.replace("{gemini_description}", gemini_desc or "(not available)")
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
    """Submit Grok synthesis via xAI Batch API.

    Creates temporary files with the pre-built prompts, then calls the
    tagger script in xAI batch mode (submit → monitor → collect).

    Returns mapping of video_path -> final caption.
    """
    # Write temporary prompt files for each video
    with tempfile.TemporaryDirectory(prefix="grok_synthesis_") as work_dir:
        # Write system prompt
        sys_prompt_path = os.path.join(work_dir, "system_prompt.md")
        with open(sys_prompt_path, "w", encoding="utf-8") as f:
            f.write(grok_system_prompt)

        # Write user prompt template (with {tags} placeholder for the tagger)
        # The tagger expects {tags} in the user prompt, but we've already built
        # the full prompts. We'll use a pass-through: the "tags" will be the
        # full rendered prompt, and the user prompt template is just "{tags}".
        user_prompt_path = os.path.join(work_dir, "user_prompt.md")
        with open(user_prompt_path, "w", encoding="utf-8") as f:
            f.write("{tags}")

        # Write per-video .txt files containing the rendered grok prompts
        # These will be loaded as "existing tags" by the tagger
        prompt_dir = os.path.join(work_dir, "prompts")
        os.makedirs(prompt_dir)

        video_to_prompt_file: Dict[str, str] = {}
        for vpath, prompt_text in grok_prompts.items():
            stem = Path(vpath).stem
            uid = hashlib.md5(vpath.encode()).hexdigest()[:8]
            prompt_file = os.path.join(prompt_dir, f"{stem}_{uid}.txt")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            # Create a dummy video symlink so the tagger finds it
            video_link = os.path.join(prompt_dir, f"{stem}_{uid}.mp4")
            os.symlink(os.path.abspath(vpath), video_link)
            video_to_prompt_file[vpath] = video_link

        # Build xAI batch state file path
        state_file = os.path.join(
            os.path.dirname(os.path.abspath(input_dir)),
            f".xai_batch_state_vcap_{hashlib.md5(input_dir.encode()).hexdigest()[:10]}.json",
        )

        # Run tagger in xAI batch mode: submit
        cmd_submit = [
            python, TAGGER_SCRIPT, prompt_dir,
            "--taggers", "grok",
            "--grok_provider", "xai-batch",
            "--xai_batch_action", "submit",
            "--xai_batch_state_file", state_file,
            "--xai_batch_model", xai_model,
            "--grok_system_prompt_file", sys_prompt_path,
            "--grok_prompt_file", user_prompt_path,
            "--video",
            "--force",
            "--remove_underscore",
            "--append_tags",
        ]

        env = os.environ.copy()
        if xai_api_key:
            env["XAI_API_KEY"] = xai_api_key

        logger.info("submitting grok synthesis via xAI batch...")
        proc = subprocess.run(cmd_submit, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"grok submit failed: {proc.stderr[-500:]}")
            return {vp: None for vp in video_paths}

        # Poll status until done (reuse monitor from CLI)
        logger.info("waiting for xAI batch to complete...")
        _wait_for_xai_batch(state_file, xai_api_key)

        # Collect results
        cmd_collect = [
            python, TAGGER_SCRIPT, prompt_dir,
            "--taggers", "grok",
            "--grok_provider", "xai-batch",
            "--xai_batch_action", "collect",
            "--xai_batch_state_file", state_file,
            "--xai_batch_model", xai_model,
            "--video",
            "--force",
            "--remove_underscore",
        ]

        logger.info("collecting grok results...")
        proc = subprocess.run(cmd_collect, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"grok collect failed: {proc.stderr[-500:]}")

        # Read the generated .txt files and map back to original video paths
        results: Dict[str, Optional[str]] = {}
        for vpath, link_path in video_to_prompt_file.items():
            txt_path = os.path.splitext(link_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                # Try to parse JSON caption
                caption = _extract_caption(content)
                results[vpath] = caption
            else:
                results[vpath] = None

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


def _wait_for_xai_batch(state_file: str, api_key: str, timeout: int = 7200):
    """Poll xAI batch status until done."""
    import time

    if not os.path.exists(state_file):
        return

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    batch_id = state.get("batch_id")
    if not batch_id:
        return

    import requests as req

    base_url = "https://api.x.ai"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    start = time.time()
    while (time.time() - start) < timeout:
        try:
            r = req.get(f"{base_url}/v1/batch-jobs/{batch_id}", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            counters = data.get("state", {})
            total = int(counters.get("num_requests", 0) or 0)
            pending = int(counters.get("num_pending", 0) or 0)

            if pending <= 0 and total > 0:
                logger.info(f"xAI batch complete: {total} requests done")
                return

            logger.debug(f"xAI batch: {total - pending}/{total} done, {pending} pending")
        except Exception as e:
            logger.warning(f"xAI status check error: {e}")

        time.sleep(20)

    logger.warning(f"xAI batch timed out after {timeout}s")


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

    gemini_results = run_gemini_phase(
        video_paths,
        gemini_api_key=gemini_api_key,
        profile=profile,
        model=gemini_model,
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
