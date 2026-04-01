"""Frame-pair captioning pipeline.

Orchestrates the full pipeline for creating scene-transition captions from
paired images (A → B).  Steps:

1. Organize pairs from source directory into datasets
2. Run WD/PixAI tagging on all images
3. Describe A images via xAI Batch API
4. Compute similarity between A and B images
5. Caption B images via xAI Batch API (using A context + similarity)
6. Upload results to HuggingFace
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from constants import IMAGE_EXTS
from ui import (
    ask_yes_no,
    console,
    make_progress,
    print_error,
    print_info,
    print_section,
    print_success,
    print_summary_table,
    print_warning,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")
XAI_API_BASE_URL = "https://api.x.ai"

# Suffixes that identify each role in a pair group
_SUFFIXES = ("_A", "_B", "_C", "_image_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_suffix(stem: str) -> Optional[Tuple[str, str]]:
    """Return (base_name, suffix) if *stem* ends with a known suffix, else None."""
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], suf
    return None


def _xai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _xai_request(
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
                    "xAI API %s on %s (attempt %d), retrying in %ds...",
                    resp.status_code, url, attempt + 1, wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                body_preview = (resp.text or "").strip().replace("\n", " ")
                if len(body_preview) > 800:
                    body_preview = body_preview[:800] + "..."
                logger.error(
                    "xAI API %s on %s: %s", resp.status_code, url,
                    body_preview or "<empty body>",
                )
            resp.raise_for_status()
            if not resp.text:
                return {}
            return resp.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                status = e.response.status_code if e.response is not None else None
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            logger.warning(
                "xAI API request error on %s (attempt %d): %s; retrying in %ds...",
                url, attempt + 1, e, wait,
            )
            time.sleep(wait)
    return {}


def _image_to_base64(image_path: str, max_size: int = 768, quality: int = 85) -> str:
    """Convert image file to base64 data URI."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(30, min(quality, 95)))
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _xai_extract_text_from_result(result_obj: Dict) -> Optional[str]:
    """Extract text content from an xAI batch result item."""

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
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                chunks = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                        chunks.append(part.get("text", ""))
                text = "\n".join(chunks).strip()
                if text:
                    return text
    return None


def _resolve_state_file(output_dir: str, step_name: str) -> str:
    """Build a state file path for a pipeline step."""
    base_dir = os.path.abspath(output_dir)
    key = hashlib.md5(base_dir.encode("utf-8")).hexdigest()[:10]
    return os.path.join(base_dir, f".xai_fp_{step_name}_{key}.json")


def _load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("could not read state from %s: %s", path, e)
        return {}


def _save_state(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Step 1: Organize pairs
# ---------------------------------------------------------------------------


def organize_pairs(
    source_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Scan source_dir for frame-pair groups and copy into dataset subdirs.

    Returns a dict with:
        counts: {dataset_1: int, dataset_2: int, dataset_3: int}
        pairs: list of (path_a, path_b) tuples across all datasets
    """
    source = Path(source_dir)
    out = Path(output_dir)

    # Scan all image files and group by relative path + base name
    groups: Dict[str, Dict[str, str]] = {}  # group_key -> {suffix -> full_path}
    all_files = []
    for root, _, files in os.walk(source):
        rel_root = os.path.relpath(root, source)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = os.path.splitext(f)[0]
            parsed = _strip_suffix(stem)
            if parsed is None:
                continue
            base_name, suffix = parsed
            full_path = os.path.join(root, f)
            group_key = os.path.join(rel_root, base_name) if rel_root != "." else base_name
            groups.setdefault(group_key, {})[suffix] = full_path
            all_files.append(full_path)

    # Classify groups into datasets
    ds1_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B + C
    ds2_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B + image_base
    ds3_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B only

    for group_key, files_map in sorted(groups.items()):
        has_a = "_A" in files_map
        has_b = "_B" in files_map
        has_c = "_C" in files_map
        has_ib = "_image_base" in files_map

        if not (has_a and has_b):
            continue  # Must have at least A and B

        # Use a flat safe name for destination files (replace path separators)
        safe_name = group_key.replace(os.sep, "__")
        if has_c:
            ds1_groups.append((safe_name, files_map))
        elif has_ib:
            ds2_groups.append((safe_name, files_map))
        else:
            ds3_groups.append((safe_name, files_map))

    # Create output directories and copy files
    datasets = {
        "dataset_1": ds1_groups,
        "dataset_2": ds2_groups,
        "dataset_3": ds3_groups,
    }

    pairs: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    total_files = sum(len(g) for gs in datasets.values() for _, g in gs)

    with make_progress() as progress:
        task = progress.add_task("Organizing pairs", total=total_files)

        for ds_name, ds_groups in datasets.items():
            ds_dir = out / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)
            counts[ds_name] = len(ds_groups)

            for base_name, files_map in ds_groups:
                for suffix, src_path in files_map.items():
                    ext = os.path.splitext(src_path)[1]
                    dest = ds_dir / f"{base_name}{suffix}{ext}"
                    shutil.copy2(src_path, dest)
                    progress.advance(task)

                # Record pair (A, B)
                ext_a = os.path.splitext(files_map["_A"])[1]
                ext_b = os.path.splitext(files_map["_B"])[1]
                path_a = str(ds_dir / f"{base_name}_A{ext_a}")
                path_b = str(ds_dir / f"{base_name}_B{ext_b}")
                pairs.append((path_a, path_b))

    return {
        "counts": counts,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Step 2: WD / PixAI tagging
# ---------------------------------------------------------------------------


def _run_tagger_subprocess(
    python: str,
    tagger_script: str,
    directory: str,
    tagger_name: str,
    batch_size: int,
) -> int:
    """Run a tagger subprocess on a directory. Returns exit code."""
    import subprocess

    cmd = [
        python, tagger_script, directory,
        "--taggers", tagger_name,
        "--batch_size", str(batch_size),
        "--remove_underscore",
        "--thresh", "0.30",
    ]
    # Pass HF token explicitly for gated models (PixAI)
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print_warning("Interrupted — killing tagger process...")
        proc.kill()
        proc.wait()
        raise

    return proc.returncode


def run_wd_tagging(
    pairs: List[Tuple[str, str]],
    python: str,
    batch_size: int = 4,
) -> bool:
    """Run PixAI tagger (fallback to WD14) on all unique A and B images.

    Tries PixAI first. If it fails (e.g. gated model), falls back to WD14.
    Returns True if tagging succeeded, False otherwise.
    """
    # Collect unique directories
    dirs: set = set()
    for path_a, path_b in pairs:
        dirs.add(os.path.dirname(path_a))
        dirs.add(os.path.dirname(path_b))

    tagger_script = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
    sorted_dirs = sorted(dirs)
    total_dirs = len(sorted_dirs)
    any_success = False

    for idx, d in enumerate(sorted_dirs, 1):
        print_info(f"[{idx}/{total_dirs}] Tagging images in: {d}")

        # Try PixAI first
        print_info("Trying PixAI tagger...")
        rc = _run_tagger_subprocess(python, tagger_script, d, "pixai", batch_size)

        if rc != 0:
            # Fallback to WD14
            print_warning(f"PixAI failed (code {rc}), falling back to WD14...")
            rc = _run_tagger_subprocess(python, tagger_script, d, "wd14", batch_size)

            if rc != 0:
                print_error(f"WD14 also failed (code {rc}) for {d}")
            else:
                print_success(f"WD14 tagging complete for {d}")
                any_success = True
        else:
            print_success(f"PixAI tagging complete for {d}")
            any_success = True

    if not any_success:
        print_error("All taggers failed for all directories. Cannot proceed without tags.")

    return any_success


# ---------------------------------------------------------------------------
# Step 3: Describe A images via xAI Batch
# ---------------------------------------------------------------------------

_DESCRIBE_SYSTEM_PROMPT = (
    "You are an image describer. Describe this image in complete detail (~150+ words). "
    "Describe everything you see: characters, their appearance (skin tone, hair, eyes, "
    "body type, expression, clothing), pose, action, background, lighting, colors, mood. "
    'Output valid JSON: {"description": "..."}'
)
_DESCRIBE_USER_PROMPT = (
    "Describe this image in complete detail. "
    "Be thorough — describe every visual element you can identify."
)


def _xai_batch_submit(
    items: List[Dict[str, Any]],
    api_key: str,
    model: str,
    state_file: str,
    step_name: str,
) -> str:
    """Submit items to xAI Batch API and return batch_id.

    Each item in items should have:
        custom_id: str
        system_prompt: str
        user_content: str | list  (text or multimodal content)
        reasoning_effort: str (default "low")
    """
    headers = _xai_headers(api_key)

    # Load or create state
    state = _load_state(state_file)
    if not state:
        state = {
            "version": 1,
            "step": step_name,
            "batch_id": None,
            "request_map": {},
            "created_at": time.time(),
        }

    # Create batch if needed
    if not state.get("batch_id"):
        batch_name = f"fp_{step_name}_{int(time.time())}"
        created = _xai_request(
            "POST",
            f"{XAI_API_BASE_URL}/v1/batches",
            headers,
            payload={"name": batch_name},
        )
        batch_id = created.get("batch_id")
        if not batch_id:
            raise RuntimeError(f"Failed to create xAI batch: {created}")
        state["batch_id"] = batch_id
        state["batch_name"] = created.get("name", batch_name)
        _save_state(state_file, state)
        logger.info("Created xAI batch: %s", batch_id)

    batch_id = state["batch_id"]
    request_map = state.setdefault("request_map", {})

    # Build and submit requests
    batch_requests: List[Dict] = []

    with make_progress() as progress:
        task = progress.add_task(f"Submitting {step_name}", total=len(items))

        for item in items:
            req_id = item["custom_id"]
            if request_map.get(req_id, {}).get("state") in ("submitted", "succeeded"):
                progress.advance(task)
                continue

            request_body = {
                "chat_get_completion": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_content"]},
                    ],
                    "response_format": {"type": "json_object"},
                    "reasoning": {"effort": item.get("reasoning_effort", "low")},
                }
            }

            batch_requests.append({
                "batch_request_id": req_id,
                "batch_request": request_body,
            })

            request_map[req_id] = {
                "state": "queued",
                "meta": item.get("meta", {}),
            }

            # Flush in chunks of 50 to avoid oversized payloads
            if len(batch_requests) >= 50:
                _xai_request(
                    "POST",
                    f"{XAI_API_BASE_URL}/v1/batches/{batch_id}/requests",
                    headers,
                    payload={"batch_requests": batch_requests},
                    timeout=60,
                )
                for br in batch_requests:
                    request_map[br["batch_request_id"]]["state"] = "submitted"
                _save_state(state_file, state)
                progress.advance(task, len(batch_requests))
                batch_requests = []

        # Flush remaining
        if batch_requests:
            _xai_request(
                "POST",
                f"{XAI_API_BASE_URL}/v1/batches/{batch_id}/requests",
                headers,
                payload={"batch_requests": batch_requests},
                timeout=60,
            )
            for br in batch_requests:
                request_map[br["batch_request_id"]]["state"] = "submitted"
            _save_state(state_file, state)
            progress.advance(task, len(batch_requests))

    state["submitted_at"] = time.time()
    _save_state(state_file, state)
    return batch_id


def _xai_batch_poll(
    batch_id: str,
    api_key: str,
    poll_seconds: int = 20,
) -> Dict:
    """Poll xAI batch until completion. Returns final status dict."""
    headers = _xai_headers(api_key)
    first_ts = None
    first_done = 0

    while True:
        data = _xai_request(
            "GET",
            f"{XAI_API_BASE_URL}/v1/batches/{batch_id}",
            headers,
        )
        counters = data.get("state", {})
        total = int(counters.get("num_requests", 0) or 0)
        pending = int(counters.get("num_pending", 0) or 0)
        success = int(counters.get("num_success", 0) or 0)
        errors = int(counters.get("num_error", 0) or 0)
        done = success + errors
        pct = (done / total * 100.0) if total else 0.0

        now = time.time()
        eta_text = "estimating..."
        if first_ts is None:
            first_ts = now
            first_done = done
        else:
            elapsed = max(now - first_ts, 1e-6)
            rate = max((done - first_done) / elapsed, 0.0)
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

        if pending <= 0 and total > 0:
            print_success("Batch completed")
            return data

        if total == 0:
            # Batch may still be initializing
            pass

        time.sleep(poll_seconds)


def _xai_batch_collect(
    batch_id: str,
    api_key: str,
    state_file: str,
) -> Dict[str, str]:
    """Collect results from xAI batch. Returns {custom_id: text_content}."""
    headers = _xai_headers(api_key)
    state = _load_state(state_file)
    request_map = state.get("request_map", {})

    results: Dict[str, str] = {}
    pagination_token = None
    success_count = 0
    error_count = 0

    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if pagination_token:
            params["pagination_token"] = pagination_token

        page = _xai_request(
            "GET",
            f"{XAI_API_BASE_URL}/v1/batches/{batch_id}/results",
            headers,
            params=params,
            timeout=180,
        )
        page_results = page.get("results", [])
        if not page_results:
            break

        for item in page_results:
            req_id = item.get("batch_request_id")
            if not req_id:
                continue

            text = _xai_extract_text_from_result(item)
            if text:
                results[req_id] = text
                if req_id in request_map:
                    request_map[req_id]["state"] = "succeeded"
                success_count += 1
            else:
                if req_id in request_map:
                    request_map[req_id]["state"] = "failed"
                error_count += 1

        pagination_token = page.get("pagination_token")
        _save_state(state_file, state)

        if not pagination_token:
            break

    logger.info(
        "Batch collect: success=%d errors=%d batch_id=%s",
        success_count, error_count, batch_id,
    )
    return results


def run_describe_a(
    pairs: List[Tuple[str, str]],
    xai_api_key: str,
    model: str,
    output_dir: str,
) -> None:
    """Submit xAI Batch job to describe all A images in detail."""
    print_section("STEP 3: DESCRIBE A IMAGES")

    # Collect unique A images
    unique_a: Dict[str, str] = {}  # path -> custom_id
    for path_a, _ in pairs:
        if path_a not in unique_a:
            req_id = "desc_" + hashlib.sha1(path_a.encode("utf-8")).hexdigest()[:16]
            unique_a[path_a] = req_id

    print_info(f"Describing {len(unique_a)} unique A images via xAI Batch...")

    state_file = _resolve_state_file(output_dir, "describe_a")

    # Build batch items
    items: List[Dict[str, Any]] = []
    for path_a, req_id in unique_a.items():
        try:
            b64_uri = _image_to_base64(path_a)
        except Exception as e:
            logger.warning("Failed to encode image %s: %s", path_a, e)
            continue

        user_content = [
            {"type": "text", "text": _DESCRIBE_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": b64_uri}},
        ]

        items.append({
            "custom_id": req_id,
            "system_prompt": _DESCRIBE_SYSTEM_PROMPT,
            "user_content": user_content,
            "reasoning_effort": "low",
            "meta": {"image_path": path_a},
        })

    # Submit
    batch_id = _xai_batch_submit(items, xai_api_key, model, state_file, "describe_a")
    print_success(f"Submitted {len(items)} description requests (batch: {batch_id})")

    # Poll
    print_info("Waiting for descriptions to complete...")
    _xai_batch_poll(batch_id, xai_api_key)

    # Collect
    print_info("Collecting description results...")
    results = _xai_batch_collect(batch_id, xai_api_key, state_file)

    # Write description files
    written = 0
    for path_a, req_id in unique_a.items():
        text = results.get(req_id)
        if not text:
            continue

        # Try to parse JSON description
        description = text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "description" in parsed:
                description = parsed["description"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Save as *_A_description.txt
        stem = os.path.splitext(path_a)[0]
        desc_file = stem + "_description.txt"
        with open(desc_file, "w", encoding="utf-8") as f:
            f.write(description)
        written += 1

    print_success(f"Wrote {written} description files")


# ---------------------------------------------------------------------------
# Step 4: Similarity computation
# ---------------------------------------------------------------------------


def run_similarity(
    pairs: List[Tuple[str, str]],
    device: str = "cuda",
) -> List[float]:
    """Compute combined similarity for all A-B pairs."""
    print_section("STEP 4: COMPUTE SIMILARITY")

    from frame_pair_similarity import compute_combined_similarity

    paths_a = [a for a, _ in pairs]
    paths_b = [b for _, b in pairs]

    print_info(f"Computing similarity for {len(pairs)} pairs (CLIP + SSCD + SSIM)...")
    similarities = compute_combined_similarity(paths_a, paths_b, device=device)

    # Save individual similarity JSONs
    written = 0
    for i, (path_a, path_b) in enumerate(pairs):
        sim_data = {"combined": round(similarities[i], 2)}

        stem_b = os.path.splitext(path_b)[0]
        # Also derive the base name for the similarity file
        stem_a = os.path.splitext(path_a)[0]
        base_parsed = _strip_suffix(os.path.basename(stem_a))
        if base_parsed:
            base_name = base_parsed[0]
            sim_file = os.path.join(os.path.dirname(path_a), f"{base_name}_similarity.json")
        else:
            sim_file = stem_b + "_similarity.json"

        with open(sim_file, "w", encoding="utf-8") as f:
            json.dump(sim_data, f, ensure_ascii=False, indent=2)
        written += 1

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    print_success(f"Similarity computed: avg={avg_sim:.1f}%, wrote {written} JSON files")

    return similarities


# ---------------------------------------------------------------------------
# Step 5: Caption B images via xAI Batch
# ---------------------------------------------------------------------------


def run_caption_b(
    pairs: List[Tuple[str, str]],
    similarities: List[float],
    xai_api_key: str,
    model: str,
    output_dir: str,
) -> None:
    """Submit xAI Batch job to caption all B images."""
    print_section("STEP 5: CAPTION B IMAGES")

    # Load prompts
    system_prompt_path = os.path.join(PROMPTS_DIR, "image", "frame-pair", "system_prompt.md")
    user_prompt_path = os.path.join(PROMPTS_DIR, "image", "frame-pair", "user_prompt.md")

    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    if not os.path.exists(user_prompt_path):
        raise FileNotFoundError(f"User prompt not found: {user_prompt_path}")

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()

    state_file = _resolve_state_file(output_dir, "caption_b")

    print_info(f"Captioning {len(pairs)} B images via xAI Batch...")

    items: List[Dict[str, Any]] = []
    skipped = 0

    for i, (path_a, path_b) in enumerate(pairs):
        req_id = "cap_" + hashlib.sha1(path_b.encode("utf-8")).hexdigest()[:16]

        # Read WD tags for A
        tags_a_file = os.path.splitext(path_a)[0] + ".txt"
        wd_tags_a = "(no tags)"
        if os.path.exists(tags_a_file):
            with open(tags_a_file, "r", encoding="utf-8") as f:
                wd_tags_a = f.read().strip() or "(no tags)"

        # Read description for A
        desc_a_file = os.path.splitext(path_a)[0] + "_description.txt"
        description_a = "(no description)"
        if os.path.exists(desc_a_file):
            with open(desc_a_file, "r", encoding="utf-8") as f:
                description_a = f.read().strip() or "(no description)"

        # Read WD tags for B
        tags_b_file = os.path.splitext(path_b)[0] + ".txt"
        wd_tags_b = "(no tags)"
        if os.path.exists(tags_b_file):
            with open(tags_b_file, "r", encoding="utf-8") as f:
                wd_tags_b = f.read().strip() or "(no tags)"

        # Similarity
        similarity_percent = round(similarities[i], 1)

        # Build user prompt from template
        user_prompt = user_prompt_template
        user_prompt = user_prompt.replace("{wd_tags_a}", wd_tags_a)
        user_prompt = user_prompt.replace("{description_a}", description_a)
        user_prompt = user_prompt.replace("{similarity_percent}", str(similarity_percent))
        user_prompt = user_prompt.replace("{wd_tags_b}", wd_tags_b)

        # Encode B image
        try:
            b64_uri = _image_to_base64(path_b)
        except Exception as e:
            logger.warning("Failed to encode image %s: %s", path_b, e)
            skipped += 1
            continue

        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": b64_uri}},
        ]

        items.append({
            "custom_id": req_id,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "reasoning_effort": "low",
            "meta": {"image_path_b": path_b},
        })

    if skipped:
        print_warning(f"Skipped {skipped} images due to encoding errors")

    # Submit
    batch_id = _xai_batch_submit(items, xai_api_key, model, state_file, "caption_b")
    print_success(f"Submitted {len(items)} caption requests (batch: {batch_id})")

    # Poll
    print_info("Waiting for captions to complete...")
    _xai_batch_poll(batch_id, xai_api_key)

    # Collect
    print_info("Collecting caption results...")
    results = _xai_batch_collect(batch_id, xai_api_key, state_file)

    # Write caption files next to B images
    written = 0
    for i, (path_a, path_b) in enumerate(pairs):
        req_id = "cap_" + hashlib.sha1(path_b.encode("utf-8")).hexdigest()[:16]
        text = results.get(req_id)
        if not text:
            continue

        # Try to parse JSON caption
        caption = text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "caption" in parsed:
                caption = parsed["caption"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Save caption next to B image (use _caption.txt to avoid overwriting WD tags)
        txt_file = os.path.splitext(path_b)[0] + "_caption.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(caption)
        written += 1

    print_success(f"Wrote {written} caption files for B images")


# ---------------------------------------------------------------------------
# Step 6: Upload to HuggingFace
# ---------------------------------------------------------------------------


def run_upload(
    output_dir: str,
    hf_token: str,
    hf_repo: str,
    python: str,
) -> None:
    """Upload B images + their .txt caption files to HuggingFace."""
    print_section("STEP 6: UPLOAD TO HUGGINGFACE")

    import subprocess

    # Collect B images and their .txt files
    upload_files: List[str] = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem.endswith("_B"):
                full_path = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    upload_files.append(full_path)
                    # Include its _caption.txt file
                    caption_path = os.path.splitext(full_path)[0] + "_caption.txt"
                    if os.path.exists(caption_path):
                        upload_files.append(caption_path)

    if not upload_files:
        print_warning("No B images found to upload")
        return

    print_info(f"Uploading {len(upload_files)} files to {hf_repo}")

    # Create a temporary directory with just the B files for upload
    import tempfile
    with tempfile.TemporaryDirectory(prefix="araknideo_fp_upload_") as tmp_dir:
        for fpath in upload_files:
            rel = os.path.relpath(fpath, output_dir)
            dest = os.path.join(tmp_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(fpath, dest)

        venv_dir = os.path.join(SCRIPT_DIR, ".venv")
        pip = os.path.join(venv_dir, "bin", "pip")
        subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)

        num_workers = max(4, min(64, (os.cpu_count() or 4) * 2))

        upload_script = (
            "from huggingface_hub import HfApi\n"
            "import sys, os\n"
            "api = HfApi(token=os.environ['HF_TOKEN'])\n"
            "api.create_repo(sys.argv[1], repo_type='dataset', "
            "private=True, exist_ok=True)\n"
            "api.upload_large_folder(\n"
            "    repo_id=sys.argv[1],\n"
            "    repo_type='dataset',\n"
            "    folder_path=sys.argv[2],\n"
            "    num_workers=int(sys.argv[3]),\n"
            ")\n"
            "print('__done__')\n"
        )

        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token
        env["HF_XET_HIGH_PERFORMANCE"] = "1"

        proc = subprocess.Popen(
            [python, "-c", upload_script, hf_repo, tmp_dir, str(num_workers)],
            env=env,
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            print_warning("Interrupted — killing upload process...")
            proc.kill()
            proc.wait()
            print_info("Re-run to resume automatically")
            return

        if proc.returncode == 0:
            print_success(f"Uploaded to https://huggingface.co/datasets/{hf_repo}")
        else:
            print_error("Upload failed — check logs above")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_frame_pair_pipeline(
    source_dir: str,
    output_dir: str,
    xai_api_key: str,
    model: str = "grok-4-1-fast-reasoning",
    device: str = "cuda",
    python: str = "",
    hf_token: str = "",
    hf_repo: str = "",
) -> None:
    """Run the full frame-pair captioning pipeline."""

    if not python:
        python = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")

    # Step 1: Organize pairs
    print_section("STEP 1: ORGANIZE PAIRS")
    result = organize_pairs(source_dir, output_dir)
    pairs = result["pairs"]
    counts = result["counts"]

    print_summary_table("Pair Organization", [
        ("Dataset 1 (A+B+C)", str(counts.get("dataset_1", 0))),
        ("Dataset 2 (A+B+base)", str(counts.get("dataset_2", 0))),
        ("Dataset 3 (A+B only)", str(counts.get("dataset_3", 0))),
        ("Total pairs", str(len(pairs))),
    ])

    if not pairs:
        print_error("No valid pairs found. Aborting.")
        return

    # Step 2: WD/PixAI tagging
    print_section("STEP 2: WD/PIXAI TAGGING")
    print_info(f"Running PixAI tagger on {len(pairs) * 2} images...")
    run_wd_tagging(pairs, python)
    print_success("Tagging complete")

    # Step 3: Describe A images
    run_describe_a(pairs, xai_api_key, model, output_dir)

    # Step 4: Similarity
    similarities = run_similarity(pairs, device)

    # Step 5: Caption B images
    run_caption_b(pairs, similarities, xai_api_key, model, output_dir)

    # Step 6: Upload (if credentials provided)
    if hf_token and hf_repo:
        run_upload(output_dir, hf_token, hf_repo, python)
    else:
        print_info("Skipping HuggingFace upload (no token/repo provided)")

    print_section("PIPELINE COMPLETE")
    print_success(f"Processed {len(pairs)} frame pairs")
    print_success(f"Output directory: {output_dir}")
