"""
Standalone video caption pipeline via xAI Batch API.

Designed for huge video datasets (30k+ files): extracts the middle frame from
each video, looks up the matching `.txt` tag file (same name, same folder),
and submits everything to xAI's batch endpoint with a single-frame video
caption system prompt. Resumable across machines.

Usage:
    export XAI_API_KEY=xai_xxx
    python video_caption_batch.py /path/to/videos --action submit
    python video_caption_batch.py /path/to/videos --action status
    python video_caption_batch.py /path/to/videos --action collect

    # one-shot end-to-end (submit, poll, collect):
    python video_caption_batch.py /path/to/videos --action run

The script persists progress to a state file next to the dataset directory so
that interrupting submit (Ctrl+C, instance shutdown, etc.) and re-running it
later only re-submits videos that were not already accepted by the API. The
collect phase is fully decoupled — you can shut the instance down after
submit, bring it back up much later (with the same dataset on disk), and run
`--action collect` to download results.

Output: for each successfully captioned video, writes `<video>.caption.json`
next to the original video file. Set `--jsonl-output PATH` to also write a
single JSON-Lines aggregate.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"})

XAI_API_BASE_URL = "https://api.x.ai"
XAI_BATCH_DEFAULT_MODEL = "grok-4-1-fast-reasoning"

# xAI add-requests endpoint hard limit is 25 MB per call. We stay below it
# to avoid 413s, and stay even lower when the payload contains images.
XAI_BATCH_MAX_ADD_PAYLOAD_BYTES = 25 * 1024 * 1024
XAI_BATCH_PAYLOAD_SAFETY_MARGIN_BYTES = 2 * 1024 * 1024
XAI_BATCH_IMAGE_SAFE_PAYLOAD_BYTES = 4 * 1024 * 1024
XAI_BATCH_ADAPTIVE_PAYLOAD_FLOOR_BYTES = 2 * 1024 * 1024

# xAI rolling rate limit on add-requests: 1000 calls / 30 s (per docs).
# We keep a 10 % safety headroom on top of that.
XAI_BATCH_MAX_ADD_CALLS_PER_30S = 1000
XAI_BATCH_ADD_WINDOW_SECONDS = 30.0
XAI_BATCH_POST_WORKERS = 8

DEFAULT_IMAGE_MAX_SIDE = 768
DEFAULT_IMAGE_QUALITY = 85
DEFAULT_PROFILE = "nsfw_caption_video"
DEFAULT_OUTPUT_SUFFIX = ".caption.json"
DEFAULT_STATE_VERSION = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "video")

logger = logging.getLogger("video_caption_batch")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    if logging.root.handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_videos(input_dir: Path, recursive: bool) -> List[Path]:
    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.lower() in VIDEO_EXTS else []

    iterator: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.iterdir()
    out: List[Path] = []
    for p in iterator:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    out.sort()
    return out


def read_tags_for_video(video_path: Path) -> str:
    txt_path = video_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("could not read tags for %s: %s", video_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompts(profile: str) -> Tuple[str, str]:
    profile_dir = os.path.join(PROMPTS_DIR, profile)
    sys_path = os.path.join(profile_dir, "system_prompt.md")
    user_path = os.path.join(profile_dir, "user_prompt.md")
    if not os.path.exists(sys_path):
        raise FileNotFoundError(f"system prompt not found: {sys_path}")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"user prompt not found: {user_path}")
    with open(sys_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open(user_path, "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()
    return system_prompt, user_prompt_template


def render_user_prompt(template: str, tags: str) -> str:
    rendered = template.replace("{tags}", tags or "(no prior tags)")
    return rendered


# ---------------------------------------------------------------------------
# Frame extraction (in memory, no temp files)
# ---------------------------------------------------------------------------

def extract_middle_frame_jpeg(
    video_path: str,
    max_side: int,
    quality: int,
) -> Optional[bytes]:
    """Open the video, seek to the middle frame, return a JPEG byte string."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python-headless is required (already in requirements.txt)"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            target = 0
        else:
            target = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if (not ok or frame is None) and target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

    img = Image.fromarray(rgb)
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(30, min(int(quality), 95)))
    return buf.getvalue()


def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def resolve_default_state_file(input_dir: Path) -> Path:
    base = input_dir.resolve()
    parent = base.parent
    name = base.name
    key = hashlib.md5(str(base).encode("utf-8")).hexdigest()[:10]
    primary = parent / f".video_caption_batch_state_{name}_{key}.json"

    def _has_data(p: Path) -> bool:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("request_map"))
        except Exception:
            return False

    if primary.exists() and _has_data(primary):
        return primary
    # Cross-machine resilience: when the dataset is moved to a different
    # absolute path the md5 hash changes. Reuse a sibling state file that
    # already has data, preferring the most recent.
    candidates = sorted(
        parent.glob(f".video_caption_batch_state_{name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        if cand == primary:
            continue
        if _has_data(cand):
            logger.info("using existing state file from a different path: %s", cand)
            return cand
    return primary


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("could not read state file %s: %s", path, exc)
        return {}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def request_id_for(video_abs: str, video_rel: Optional[str]) -> str:
    seed = video_rel if video_rel and not video_rel.startswith("..") else video_abs
    return "req_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()


def json_byte_size(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


# ---------------------------------------------------------------------------
# xAI API helpers
# ---------------------------------------------------------------------------

def xai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def xai_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict] = None,
    params: Optional[Dict] = None,
    max_retries: int = 5,
    timeout: int = 120,
) -> Dict:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=payload,
                params=params,
                timeout=timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "xAI %s on %s (attempt %d), retrying in %ds...",
                    resp.status_code, url, attempt + 1, wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                preview = (resp.text or "").strip().replace("\n", " ")
                if len(preview) > 600:
                    preview = preview[:600] + "..."
                logger.error("xAI %s on %s: %s", resp.status_code, url, preview)
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            logger.warning("xAI request error on %s (%d): %s; retrying in %ds...",
                           url, attempt + 1, e, wait)
            time.sleep(wait)
    return {}


def parse_grok_json(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def extract_caption_from_result(result_obj: Dict) -> Tuple[Optional[str], Optional[Dict]]:
    """Return (caption_text, full_parsed_json) from an xAI batch result item.

    The xAI batch API wraps the chat completion response under varying keys
    depending on version, so we walk the dict looking for the standard
    OpenAI-style `choices` array.
    """

    def _find_choices(obj: Any, depth: int = 0) -> List[Dict]:
        found: List[Dict] = []
        if depth > 5 or not isinstance(obj, dict):
            return found
        if "choices" in obj:
            found.append(obj)
        for v in obj.values():
            if isinstance(v, dict):
                found.extend(_find_choices(v, depth + 1))
        return found

    for candidate in _find_choices(result_obj):
        choices = candidate.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        message = first.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parsed = parse_grok_json(content)
            if parsed and "caption" in parsed:
                return parsed["caption"], parsed
            if parsed:
                return None, parsed
            return content.strip(), None
    return None, None


def strip_images_from_request(batch_item: Dict) -> bool:
    """Drop image parts from one batch_request, leaving only text content.

    Used as a last-resort retry after a 413/422; preserves the request id and
    submits a tags-only fallback rather than dropping the video entirely.
    """
    try:
        user_msg = batch_item["batch_request"]["chat_get_completion"]["messages"][1]
        content = user_msg.get("content")
        if not isinstance(content, list):
            return False
        text_chunks = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        user_msg["content"] = "\n".join(c for c in text_chunks if c.strip()) or "(no prompt)"
        batch_item["_images_stripped"] = True
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize_state(
    state: Dict[str, Any],
    *,
    state_file: Path,
    input_dir: Path,
    profile: str,
    model: str,
    system_prompt: str,
    user_template: str,
) -> Dict[str, Any]:
    if state:
        return state
    return {
        "version": DEFAULT_STATE_VERSION,
        "provider": "xai-batch",
        "input_dir": str(input_dir.resolve()),
        "state_file": str(state_file),
        "profile": profile,
        "model": model,
        "batch_id": None,
        "batch_name": None,
        "system_prompt_sha256": hash_text(system_prompt),
        "user_prompt_template_sha256": hash_text(user_template),
        "request_map": {},
        "created_at": time.time(),
    }


def warn_if_prompts_changed(state: Dict[str, Any], system_prompt: str, user_template: str) -> None:
    sys_h = hash_text(system_prompt)
    user_h = hash_text(user_template)
    saved_sys = state.get("system_prompt_sha256")
    saved_user = state.get("user_prompt_template_sha256")
    if saved_sys and saved_sys != sys_h:
        logger.warning(
            "system prompt changed since this batch was created (saved=%s current=%s); "
            "cache hits will be lost.",
            saved_sys[:12], sys_h[:12],
        )
    elif not saved_sys:
        state["system_prompt_sha256"] = sys_h
    if saved_user and saved_user != user_h:
        logger.warning(
            "user prompt template changed since this batch was created (saved=%s current=%s).",
            saved_user[:12], user_h[:12],
        )
    elif not saved_user:
        state["user_prompt_template_sha256"] = user_h


def create_xai_batch(
    api_base_url: str,
    headers: Dict[str, str],
    name: str,
) -> Dict[str, Any]:
    return xai_request("POST", f"{api_base_url}/v1/batches", headers, payload={"name": name})


def post_batch_requests(
    session: requests.Session,
    api_base_url: str,
    batch_id: str,
    payload: Dict,
    timeout: int = 120,
    max_retries: int = 5,
) -> Dict:
    url = f"{api_base_url}/v1/batches/{batch_id}/requests"
    for attempt in range(max_retries + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt, 30)
                logger.warning("xAI %s on add-requests (attempt %d), retry in %ds",
                               resp.status_code, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                preview = (resp.text or "").strip().replace("\n", " ")
                if len(preview) > 600:
                    preview = preview[:600] + "..."
                logger.error("xAI %s on add-requests: %s", resp.status_code, preview)
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            logger.warning("xAI add-requests error (%d): %s; retry in %ds",
                           attempt + 1, e, wait)
            time.sleep(wait)
    return {}


def submit_phase(
    *,
    videos: List[Path],
    input_dir: Path,
    state: Dict[str, Any],
    state_file: Path,
    api_key: str,
    api_base_url: str,
    model: str,
    system_prompt: str,
    user_template: str,
    include_images: bool,
    image_max_side: int,
    image_quality: int,
    submit_chunk: int,
    frame_workers: int,
    force: bool,
    no_progress: bool,
) -> None:
    headers = xai_headers(api_key)
    request_map: Dict[str, Dict[str, Any]] = state.setdefault("request_map", {})

    if force and request_map:
        logger.info("force mode: clearing %d prior request entries", len(request_map))
        request_map.clear()
        state["request_map_force_reset_at"] = time.time()
        save_state(state_file, state)

    if not state.get("batch_id"):
        name = f"video_caption_{int(time.time())}"
        created = create_xai_batch(api_base_url, headers, name)
        bid = created.get("batch_id")
        if not bid:
            raise RuntimeError(f"failed to create xAI batch: {created}")
        state["batch_id"] = bid
        state["batch_name"] = created.get("name", name)
        state["batch_created_at"] = created.get("created_at") or time.time()
        save_state(state_file, state)
        logger.info("created xAI batch: %s", bid)

    batch_id = state["batch_id"]
    # Sanity-check the batch is still writable; if not, rotate to a new one.
    try:
        meta = xai_request("GET", f"{api_base_url}/v1/batches/{batch_id}", headers)
        state["last_status"] = meta
        save_state(state_file, state)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (403, 404):
            logger.warning("existing batch_id=%s not visible (HTTP %s); creating fresh batch.",
                           batch_id, status)
            request_map.clear()
            name = f"video_caption_{int(time.time())}"
            created = create_xai_batch(api_base_url, headers, name)
            bid = created.get("batch_id")
            if not bid:
                raise RuntimeError(f"failed to create xAI batch: {created}")
            state["batch_id"] = bid
            state["batch_name"] = created.get("name", name)
            state["batch_reset_reason"] = f"preflight_http_{status}"
            state["batch_reset_at"] = time.time()
            save_state(state_file, state)
            batch_id = bid
        else:
            raise

    # --- Pass 1: scan, dedupe against prior submissions ---
    input_abs = input_dir.resolve()
    to_process: List[Tuple[str, str, Optional[str], str, str]] = []
    use_pbar = not no_progress
    scan_pbar = tqdm(total=len(videos), desc="scan", unit="vid", smoothing=0.0) if use_pbar else None

    for video in videos:
        v_abs = str(video.resolve())
        try:
            v_rel = str(video.resolve().relative_to(input_abs))
        except ValueError:
            v_rel = None
        req_id = request_id_for(v_abs, v_rel)
        if scan_pbar is not None:
            scan_pbar.update(1)
        prior = request_map.get(req_id, {})
        if prior.get("state") in ("submitted", "succeeded"):
            continue
        tags = read_tags_for_video(video)
        to_process.append((v_abs, str(video), v_rel, req_id, tags))
    if scan_pbar is not None:
        scan_pbar.close()

    logger.info("submit: %d to send, %d already submitted",
                len(to_process), len(videos) - len(to_process))
    if not to_process:
        logger.info("nothing to submit; rerun with --action status or --action collect")
        return

    # --- Pass 2: encode frames in parallel + size-aware POSTs in parallel ---
    max_payload_bytes = XAI_BATCH_MAX_ADD_PAYLOAD_BYTES - XAI_BATCH_PAYLOAD_SAFETY_MARGIN_BYTES
    if include_images:
        max_payload_bytes = min(max_payload_bytes, XAI_BATCH_IMAGE_SAFE_PAYLOAD_BYTES)
    submit_limits = {"target_payload_bytes": max_payload_bytes}
    overhead_bytes = json_byte_size({"batch_requests": []})

    rate_window = XAI_BATCH_ADD_WINDOW_SECONDS
    max_calls_per_window = max(1, int(XAI_BATCH_MAX_ADD_CALLS_PER_30S * 0.9))
    add_call_timestamps: Deque[float] = collections.deque()

    state_lock = threading.Lock()
    rate_lock = threading.Lock()
    fatal_lock = threading.Lock()
    fatal_error: Dict[str, Optional[BaseException]] = {"err": None}

    counters = {"submitted": 0, "skipped": 0, "failed": 0, "frame_failed": 0}

    submit_pbar = tqdm(total=len(to_process), desc="submit", unit="vid", smoothing=0.0) if use_pbar else None

    session = requests.Session()
    session.headers.update(headers)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=XAI_BATCH_POST_WORKERS + 2,
        pool_maxsize=XAI_BATCH_POST_WORKERS + 2,
        max_retries=0,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    def _encode(item: Tuple[str, str, Optional[str], str, str]) -> Optional[Tuple[str, str, Optional[str], List[Dict], str]]:
        v_abs, v_path, v_rel, req_id, tags = item
        user_prompt = render_user_prompt(user_template, tags)
        if not include_images:
            return req_id, v_abs, v_rel, [{"type": "text", "text": user_prompt}], tags
        try:
            jpeg = extract_middle_frame_jpeg(v_path, image_max_side, image_quality)
        except Exception as exc:
            logger.warning("frame extraction failed for %s: %s", v_path, exc)
            jpeg = None
        if not jpeg:
            with state_lock:
                counters["frame_failed"] += 1
                request_map[req_id] = {
                    "video_path": v_abs,
                    "video_path_rel": v_rel,
                    "state": "failed_frame",
                    "error_message": "could not extract middle frame",
                    "updated_at": time.time(),
                }
            if submit_pbar is not None:
                submit_pbar.update(1)
                submit_pbar.set_postfix(**counters)
            return None
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": jpeg_to_data_url(jpeg)}},
        ]
        return req_id, v_abs, v_rel, content, tags

    def _drain_rate_window() -> float:
        wait = 0.0
        with rate_lock:
            now = time.monotonic()
            while add_call_timestamps and now - add_call_timestamps[0] >= rate_window:
                add_call_timestamps.popleft()
            if len(add_call_timestamps) >= max_calls_per_window:
                wait = rate_window - (now - add_call_timestamps[0]) + 0.05
        if wait > 0:
            logger.info("rate-limit pause %.2fs", wait)
            time.sleep(wait)
        with rate_lock:
            now = time.monotonic()
            while add_call_timestamps and now - add_call_timestamps[0] >= rate_window:
                add_call_timestamps.popleft()
            add_call_timestamps.append(now)
        return wait

    def _post_sub_batch(sub_batch: List[Dict]) -> None:
        _drain_rate_window()
        payload = {"batch_requests": sub_batch}
        size = json_byte_size(payload)
        if size > submit_limits["target_payload_bytes"]:
            fake = requests.Response()
            fake.status_code = 413
            raise requests.exceptions.HTTPError(
                f"local payload too large: {size}b > {submit_limits['target_payload_bytes']}b",
                response=fake,
            )
        post_batch_requests(session, api_base_url, batch_id, payload)

    def _mark_submitted(sub_batch: List[Dict]) -> None:
        with state_lock:
            for item in sub_batch:
                req_id = item["batch_request_id"]
                entry = request_map.setdefault(req_id, {})
                entry["state"] = "submitted"
                entry["updated_at"] = time.time()
                if item.get("_images_stripped"):
                    entry["image_payload"] = "stripped_due_to_413"
            counters["submitted"] += len(sub_batch)
            state["last_submit_at"] = time.time()
            save_state(state_file, state)
        if submit_pbar is not None:
            submit_pbar.update(len(sub_batch))
            submit_pbar.set_postfix(**counters)

    def _mark_payload_too_large(item: Dict) -> None:
        with state_lock:
            req_id = item["batch_request_id"]
            entry = request_map.setdefault(req_id, {})
            entry["state"] = "failed_payload_too_large"
            entry["error_message"] = "xai_413_payload_too_large"
            entry["updated_at"] = time.time()
            counters["skipped"] += 1
            save_state(state_file, state)
        if submit_pbar is not None:
            submit_pbar.update(1)
            submit_pbar.set_postfix(**counters)

    def _mark_failed(item: Dict, status: Optional[int], message: str) -> None:
        with state_lock:
            req_id = item["batch_request_id"]
            entry = request_map.setdefault(req_id, {})
            entry["state"] = "failed_submit"
            entry["error_message"] = (message or "")[:1000]
            entry["error_status"] = status
            entry["updated_at"] = time.time()
            counters["failed"] += 1
            save_state(state_file, state)
        if submit_pbar is not None:
            submit_pbar.update(1)
            submit_pbar.set_postfix(**counters)

    def _flush_sub_batch(sub_batch: List[Dict]) -> None:
        if not sub_batch or fatal_error["err"] is not None:
            return
        status: Optional[int] = None
        err_text = ""
        try:
            _post_sub_batch(sub_batch)
            _mark_submitted(sub_batch)
            return
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            err_text = ((e.response.text or "") if e.response is not None else str(e)).strip()

        if status == 401:
            with fatal_lock:
                fatal_error["err"] = RuntimeError(
                    "xAI 401 Unauthorized — verify XAI_API_KEY"
                )
            return

        if status in (400, 403, 404, 413, 422) and len(sub_batch) > 1:
            if status == 413:
                cur = json_byte_size({"batch_requests": sub_batch})
                lowered = max(int(cur * 0.75), XAI_BATCH_ADAPTIVE_PAYLOAD_FLOOR_BYTES)
                with state_lock:
                    if lowered < submit_limits["target_payload_bytes"]:
                        submit_limits["target_payload_bytes"] = lowered
                        logger.warning("adapting payload target after 413: %d bytes", lowered)
            mid = len(sub_batch) // 2
            logger.warning("HTTP %s on sub-batch of %d; splitting %d+%d",
                           status, len(sub_batch), mid, len(sub_batch) - mid)
            _flush_sub_batch(sub_batch[:mid])
            _flush_sub_batch(sub_batch[mid:])
            return

        single = sub_batch[0]
        # Last-resort: drop image and retry once for single-request failures.
        if include_images and status in (400, 403, 413, 422) and strip_images_from_request(single):
            logger.warning("HTTP %s on single request; retrying once with images stripped", status)
            try:
                _post_sub_batch([single])
                _mark_submitted([single])
                return
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                err_text = ((e.response.text or "") if e.response is not None else str(e)).strip()

        if status == 401:
            with fatal_lock:
                fatal_error["err"] = RuntimeError(
                    "xAI 401 Unauthorized — verify XAI_API_KEY"
                )
            return

        if status == 413:
            _mark_payload_too_large(single)
            return

        body = err_text[:600] if err_text else "<empty>"
        logger.error("HTTP %s on single request; marking failed: %s", status, body)
        _mark_failed(single, status, body)

    post_pool = ThreadPoolExecutor(max_workers=XAI_BATCH_POST_WORKERS)
    post_futures: List = []

    def _drain_done(futs: List) -> None:
        still = []
        for f in futs:
            if f.done():
                exc = f.exception()
                if exc is not None:
                    logger.error("post worker raised: %s", exc)
            else:
                still.append(f)
        futs.clear()
        futs.extend(still)

    try:
        for chunk_start in range(0, len(to_process), submit_chunk):
            if fatal_error["err"] is not None:
                break
            chunk = to_process[chunk_start : chunk_start + submit_chunk]

            encoded: List[Tuple[str, str, Optional[str], List[Dict], str]] = []
            if frame_workers > 1 and len(chunk) > 1:
                with ThreadPoolExecutor(max_workers=frame_workers) as ex:
                    futures = [ex.submit(_encode, it) for it in chunk]
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                        except Exception as exc:
                            logger.error("encode worker raised: %s", exc)
                            continue
                        if res is not None:
                            encoded.append(res)
            else:
                for it in chunk:
                    try:
                        res = _encode(it)
                    except Exception as exc:
                        logger.error("encode raised: %s", exc)
                        continue
                    if res is not None:
                        encoded.append(res)

            sub_batch: List[Dict] = []
            sub_batch_bytes = 0

            for req_id, v_abs, v_rel, content, tags in encoded:
                if fatal_error["err"] is not None:
                    break
                request_body = {
                    "chat_get_completion": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content if include_images else content[0]["text"]},
                        ],
                        "response_format": {"type": "json_object"},
                    }
                }
                wrapper = {
                    "batch_request_id": req_id,
                    "batch_request": request_body,
                }
                wrapper_bytes = json_byte_size(wrapper)
                next_bytes = overhead_bytes + sub_batch_bytes + wrapper_bytes
                if sub_batch:
                    next_bytes += 1
                if sub_batch and next_bytes > submit_limits["target_payload_bytes"]:
                    snap = list(sub_batch)
                    post_futures.append(post_pool.submit(_flush_sub_batch, snap))
                    _drain_done(post_futures)
                    sub_batch = []
                    sub_batch_bytes = 0
                sub_batch.append(wrapper)
                sub_batch_bytes += wrapper_bytes
                with state_lock:
                    request_map[req_id] = {
                        "video_path": v_abs,
                        "video_path_rel": v_rel,
                        "state": "queued_for_submission",
                        "updated_at": time.time(),
                    }

            if sub_batch:
                post_futures.append(post_pool.submit(_flush_sub_batch, list(sub_batch)))
            _drain_done(post_futures)

        for fut in post_futures:
            fut.result()
        post_futures.clear()

        if fatal_error["err"] is not None:
            raise fatal_error["err"]

    except KeyboardInterrupt:
        save_state(state_file, state)
        logger.warning("submit interrupted by user (Ctrl+C). state saved: %s", state_file)
        raise
    finally:
        post_pool.shutdown(wait=True)
        session.close()
        if submit_pbar is not None:
            submit_pbar.close()

    logger.info(
        "submit done: batch_id=%s submitted=%d skipped=%d failed=%d frame_failed=%d state=%s",
        batch_id, counters["submitted"], counters["skipped"], counters["failed"],
        counters["frame_failed"], state_file,
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status_phase(
    *,
    state: Dict[str, Any],
    state_file: Path,
    api_key: str,
    api_base_url: str,
) -> Dict[str, Any]:
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError(f"no batch_id in state file ({state_file}) — run submit first")
    headers = xai_headers(api_key)
    meta = xai_request("GET", f"{api_base_url}/v1/batches/{batch_id}", headers)
    state["last_status"] = meta
    save_state(state_file, state)
    counters = meta.get("state", {}) or meta
    total = int(counters.get("num_requests", 0) or 0)
    pending = int(counters.get("num_pending", 0) or 0)
    success = int(counters.get("num_success", 0) or 0)
    error = int(counters.get("num_error", 0) or 0)
    done = success + error
    pct = (done / total * 100.0) if total else 0.0
    logger.info(
        "batch %s: total=%d done=%d (%.1f%%) pending=%d success=%d error=%d",
        batch_id, total, done, pct, pending, success, error,
    )
    return meta


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def resolve_local_video_path(meta: Dict[str, Any], input_dir: Path) -> Optional[Path]:
    # Prefer rel+input_dir so we always write captions next to the videos
    # under the *current* --input-dir, even when the state was created on
    # another machine (different absolute path).
    rel = meta.get("video_path_rel")
    if rel:
        p = (input_dir / rel).resolve()
        if p.exists():
            return p
    abs_path = meta.get("video_path")
    if abs_path:
        p = Path(abs_path)
        if p.exists():
            return p
    return None


def write_caption_output(
    video: Path,
    caption: str,
    parsed: Optional[Dict[str, Any]],
    tags: str,
    output_suffix: str,
    overwrite: bool,
) -> Optional[Path]:
    # Use `<video>.caption.json` (e.g. clip.mp4.caption.json) so we never
    # collide with existing tag files like `clip.txt`.
    out_path = Path(str(video) + output_suffix)
    if out_path.exists() and not overwrite:
        return None
    payload = {
        "video": video.name,
        "caption": caption,
    }
    if tags:
        payload["tags"] = tags
    if parsed:
        # Surface any extra fields the model returned alongside `caption`
        # (e.g. structured metadata) without losing the canonical caption text.
        for key, value in parsed.items():
            if key == "caption":
                continue
            payload[f"raw_{key}"] = value
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def collect_phase(
    *,
    input_dir: Path,
    state: Dict[str, Any],
    state_file: Path,
    api_key: str,
    api_base_url: str,
    page_size: int,
    output_suffix: str,
    overwrite: bool,
    jsonl_output: Optional[Path],
    no_progress: bool,
) -> Dict[str, int]:
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError(f"no batch_id in state file ({state_file}) — run submit first")
    headers = xai_headers(api_key)
    meta = xai_request("GET", f"{api_base_url}/v1/batches/{batch_id}", headers)
    counters = meta.get("state", {}) or meta
    total = int(counters.get("num_requests", 0) or 0)
    pending = int(counters.get("num_pending", 0) or 0)
    success_remote = int(counters.get("num_success", 0) or 0)
    error_remote = int(counters.get("num_error", 0) or 0)
    done_remote = success_remote + error_remote
    logger.info(
        "collect start: batch %s total=%d done=%d pending=%d success=%d error=%d",
        batch_id, total, done_remote, pending, success_remote, error_remote,
    )
    if pending > 0:
        logger.warning("%d still pending — collecting %d done; rerun later for the rest",
                       pending, done_remote)

    request_map: Dict[str, Dict[str, Any]] = state.get("request_map", {})
    pagination_token: Optional[str] = None
    use_pbar = not no_progress
    pbar = tqdm(total=done_remote or None, desc="collect", unit="res", smoothing=0.0) if use_pbar else None

    counts = {"written": 0, "errors": 0, "missing_video": 0, "skipped_existing": 0, "no_caption": 0}

    jsonl_fp = jsonl_output.open("a", encoding="utf-8") if jsonl_output else None
    jsonl_lock = threading.Lock()

    try:
        while True:
            params = {"limit": page_size}
            if pagination_token:
                params["pagination_token"] = pagination_token
            page = xai_request(
                "GET",
                f"{api_base_url}/v1/batches/{batch_id}/results",
                headers,
                params=params,
                timeout=180,
            )
            results = page.get("results", []) or []
            if not results:
                break

            for item in results:
                req_id = item.get("batch_request_id")
                if not req_id:
                    continue
                meta_entry = request_map.get(req_id, {})
                video_path = resolve_local_video_path(meta_entry, input_dir)
                if not video_path:
                    counts["missing_video"] += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue

                caption_text, parsed = extract_caption_from_result(item)
                if caption_text is None and parsed is None:
                    request_map[req_id] = {
                        **meta_entry,
                        "state": "failed",
                        "error_message": item.get("error_message") or item.get("error") or "no caption",
                        "updated_at": time.time(),
                    }
                    counts["errors"] += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(**counts)
                    continue

                final_caption = caption_text or (parsed.get("caption") if parsed else None)
                if not final_caption:
                    counts["no_caption"] += 1
                    request_map[req_id] = {
                        **meta_entry,
                        "state": "failed",
                        "error_message": "model returned no caption field",
                        "updated_at": time.time(),
                    }
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(**counts)
                    continue

                tags = meta_entry.get("tags", "") or read_tags_for_video(video_path)
                out_path = write_caption_output(
                    video_path, final_caption, parsed, tags, output_suffix, overwrite,
                )
                if out_path is None:
                    counts["skipped_existing"] += 1
                else:
                    counts["written"] += 1
                    if jsonl_fp is not None:
                        record = {
                            "video": str(video_path),
                            "video_rel": meta_entry.get("video_path_rel"),
                            "request_id": req_id,
                            "caption": final_caption,
                            "tags": tags,
                        }
                        with jsonl_lock:
                            jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                request_map[req_id] = {
                    **meta_entry,
                    "state": "succeeded",
                    "output_path": str(out_path) if out_path else meta_entry.get("output_path"),
                    "updated_at": time.time(),
                }
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(**counts)

            pagination_token = page.get("pagination_token")
            state["last_collect_at"] = time.time()
            save_state(state_file, state)
            if not pagination_token:
                break
    finally:
        if pbar is not None:
            pbar.close()
        if jsonl_fp is not None:
            jsonl_fp.close()

    save_state(state_file, state)
    logger.info(
        "collect done: written=%d skipped_existing=%d errors=%d no_caption=%d missing_video=%d pending_remote=%d",
        counts["written"], counts["skipped_existing"], counts["errors"],
        counts["no_caption"], counts["missing_video"], pending,
    )
    if pending > 0:
        logger.warning("%d still pending on the server — rerun --action collect later", pending)
    return counts


# ---------------------------------------------------------------------------
# Run (submit -> poll -> collect) one-shot
# ---------------------------------------------------------------------------

def run_phase(
    *,
    args: argparse.Namespace,
    videos: List[Path],
    input_dir: Path,
    state: Dict[str, Any],
    state_file: Path,
    api_key: str,
    api_base_url: str,
    system_prompt: str,
    user_template: str,
) -> None:
    submit_phase(
        videos=videos,
        input_dir=input_dir,
        state=state,
        state_file=state_file,
        api_key=api_key,
        api_base_url=api_base_url,
        model=args.xai_model,
        system_prompt=system_prompt,
        user_template=user_template,
        include_images=not args.no_image,
        image_max_side=args.image_max_side,
        image_quality=args.image_quality,
        submit_chunk=args.submit_chunk,
        frame_workers=args.frame_workers,
        force=args.force,
        no_progress=args.no_progress,
    )

    poll_interval = max(args.poll_interval, 5)
    tail_started: Optional[float] = None
    deadline = time.time() + args.poll_timeout if args.poll_timeout > 0 else None

    while True:
        meta = status_phase(
            state=state,
            state_file=state_file,
            api_key=api_key,
            api_base_url=api_base_url,
        )
        counters = meta.get("state", {}) or meta
        total = int(counters.get("num_requests", 0) or 0)
        pending = int(counters.get("num_pending", 0) or 0)
        success = int(counters.get("num_success", 0) or 0)
        error = int(counters.get("num_error", 0) or 0)
        done = success + error
        pct = (done / total * 100.0) if total else 0.0

        if pending <= 0 and total > 0:
            break
        if total > 0 and pct >= 99.0 and pending > 0:
            if tail_started is None:
                tail_started = time.time()
            elif time.time() - tail_started > args.tail_timeout:
                logger.warning("tail timeout: %d straggler(s) remaining; moving on", pending)
                break
        else:
            tail_started = None
        if deadline is not None and time.time() > deadline:
            logger.warning("poll timeout exceeded; moving on with %d pending", pending)
            break
        time.sleep(poll_interval)

    collect_phase(
        input_dir=input_dir,
        state=state,
        state_file=state_file,
        api_key=api_key,
        api_base_url=api_base_url,
        page_size=args.page_size,
        output_suffix=args.output_suffix,
        overwrite=args.overwrite,
        jsonl_output=Path(args.jsonl_output).expanduser().resolve() if args.jsonl_output else None,
        no_progress=args.no_progress,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone single-frame video caption pipeline via xAI Batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", help="folder with videos (and matching .txt tag files)")
    p.add_argument("--action", choices=["submit", "status", "collect", "run"], default="run",
                   help="phase to run: submit/status/collect, or `run` (submit -> poll -> collect)")
    p.add_argument("--profile", default=DEFAULT_PROFILE,
                   help="prompt profile under prompts/video/<profile>/")
    p.add_argument("--xai-api-key", default=os.environ.get("XAI_API_KEY", ""),
                   help="xAI API key (defaults to XAI_API_KEY env var)")
    p.add_argument("--xai-model", default=XAI_BATCH_DEFAULT_MODEL,
                   help="xAI model id for the batch")
    p.add_argument("--api-base-url", default=XAI_API_BASE_URL,
                   help="xAI API base URL")
    p.add_argument("--state-file", default=None,
                   help="override the auto-derived state file path")
    p.add_argument("--no-recursive", action="store_true",
                   help="do not recurse into subdirectories")
    p.add_argument("--no-image", action="store_true",
                   help="send tags-only requests (no image at all)")
    p.add_argument("--image-max-side", type=int, default=DEFAULT_IMAGE_MAX_SIDE,
                   help="max side of the JPEG sent to the model (in px)")
    p.add_argument("--image-quality", type=int, default=DEFAULT_IMAGE_QUALITY,
                   help="JPEG quality 30..95")
    p.add_argument("--submit-chunk", type=int, default=1000,
                   help="max requests per chunk before size-aware flushing")
    p.add_argument("--frame-workers", type=int, default=max(8, (os.cpu_count() or 4) * 2),
                   help="parallel workers for middle-frame extraction")
    p.add_argument("--page-size", type=int, default=100,
                   help="page size when paging xAI batch results")
    p.add_argument("--output-suffix", default=DEFAULT_OUTPUT_SUFFIX,
                   help="output filename suffix appended to the video name")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing caption files on collect")
    p.add_argument("--jsonl-output", default=None,
                   help="optional path for an aggregated JSONL of all captions")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="seconds between status polls in `run` mode")
    p.add_argument("--poll-timeout", type=int, default=0,
                   help="hard timeout in seconds for `run` polling (0 = unbounded)")
    p.add_argument("--tail-timeout", type=int, default=300,
                   help="extra seconds to wait once batch is >=99%% done in `run` mode")
    p.add_argument("--force", action="store_true",
                   help="clear submitted-state from the state file before submit")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        logger.error("input dir does not exist: %s", input_dir)
        return 2
    if input_dir.is_file():
        logger.error("input must be a directory, not a file: %s", input_dir)
        return 2

    state_file = Path(args.state_file).expanduser().resolve() if args.state_file \
        else resolve_default_state_file(input_dir)
    logger.info("state file: %s", state_file)

    if not args.xai_api_key:
        logger.error("XAI_API_KEY not set; pass --xai-api-key or export XAI_API_KEY")
        return 2

    system_prompt, user_template = load_prompts(args.profile)

    state = load_state(state_file)
    state = initialize_state(
        state,
        state_file=state_file,
        input_dir=input_dir,
        profile=args.profile,
        model=args.xai_model,
        system_prompt=system_prompt,
        user_template=user_template,
    )
    warn_if_prompts_changed(state, system_prompt, user_template)
    save_state(state_file, state)

    needs_videos = args.action in ("submit", "run")
    videos: List[Path] = []
    if needs_videos:
        videos = discover_videos(input_dir, recursive=not args.no_recursive)
        logger.info("discovered %d videos under %s", len(videos), input_dir)
        if not videos:
            logger.warning("no videos found; nothing to submit")
            return 0

    try:
        if args.action == "submit":
            submit_phase(
                videos=videos,
                input_dir=input_dir,
                state=state,
                state_file=state_file,
                api_key=args.xai_api_key,
                api_base_url=args.api_base_url,
                model=args.xai_model,
                system_prompt=system_prompt,
                user_template=user_template,
                include_images=not args.no_image,
                image_max_side=args.image_max_side,
                image_quality=args.image_quality,
                submit_chunk=args.submit_chunk,
                frame_workers=args.frame_workers,
                force=args.force,
                no_progress=args.no_progress,
            )
        elif args.action == "status":
            status_phase(
                state=state,
                state_file=state_file,
                api_key=args.xai_api_key,
                api_base_url=args.api_base_url,
            )
        elif args.action == "collect":
            collect_phase(
                input_dir=input_dir,
                state=state,
                state_file=state_file,
                api_key=args.xai_api_key,
                api_base_url=args.api_base_url,
                page_size=args.page_size,
                output_suffix=args.output_suffix,
                overwrite=args.overwrite,
                jsonl_output=Path(args.jsonl_output).expanduser().resolve() if args.jsonl_output else None,
                no_progress=args.no_progress,
            )
        elif args.action == "run":
            run_phase(
                args=args,
                videos=videos,
                input_dir=input_dir,
                state=state,
                state_file=state_file,
                api_key=args.xai_api_key,
                api_base_url=args.api_base_url,
                system_prompt=system_prompt,
                user_template=user_template,
            )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        body = (e.response.text or "")[:300] if e.response is not None else ""
        if status == 401 or (status == 400 and "API key" in body):
            logger.error(
                "xAI authentication failed (HTTP %s). Verify XAI_API_KEY (or --xai-api-key). "
                "Server said: %s",
                status, body.strip().replace("\n", " "),
            )
            return 2
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
