"""Gemini Batch API integration for video understanding.

Handles:
- Video upload via File API
- Context caching for system prompts (90% token discount)
- Batch job submission and collection
- Cleanup of uploaded files and caches
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
# Minimum tokens for explicit caching (Flash models)
MIN_CACHE_TOKENS = 1024


def _get_client(api_key: Optional[str] = None):
    """Create a google-genai client."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "Gemini API key is required. "
            "Set GEMINI_API_KEY env var or pass api_key."
        )
    return genai.Client(api_key=key)


# -------------------------
# File API: upload / cleanup
# -------------------------

def upload_videos(
    video_paths: List[str],
    api_key: Optional[str] = None,
    on_progress=None,
) -> Dict[str, Any]:
    """Upload videos via Gemini File API.

    Returns:
        mapping of video_path -> file object (with .name and .uri)
    """
    client = _get_client(api_key)
    uploaded: Dict[str, Any] = {}
    failed: List[str] = []

    for i, vpath in enumerate(video_paths):
        try:
            file_obj = client.files.upload(file=vpath)
            uploaded[vpath] = file_obj
            logger.debug(f"uploaded {vpath} -> {file_obj.name}")
        except Exception as e:
            logger.warning(f"failed to upload {vpath}: {e}")
            failed.append(vpath)

        if on_progress:
            on_progress(i + 1, len(video_paths))

    if failed:
        logger.warning(f"{len(failed)} videos failed to upload")

    # Wait for processing to complete
    _wait_for_processing(client, uploaded)

    return uploaded


def _wait_for_processing(
    client,
    uploaded: Dict[str, Any],
    timeout: int = 600,
    poll_interval: int = 5,
):
    """Wait until all uploaded files finish processing."""
    pending = {vpath: fobj for vpath, fobj in uploaded.items()}
    start = time.time()

    while pending and (time.time() - start) < timeout:
        still_pending = {}
        for vpath, fobj in pending.items():
            try:
                refreshed = client.files.get(name=fobj.name)
                state = getattr(refreshed, "state", None)
                if state and str(state) == "PROCESSING":
                    still_pending[vpath] = fobj
                else:
                    # Update with refreshed object
                    uploaded[vpath] = refreshed
            except Exception as e:
                logger.warning(f"error checking file state for {vpath}: {e}")
                still_pending[vpath] = fobj

        pending = still_pending
        if pending:
            logger.debug(f"waiting for {len(pending)} files to finish processing...")
            time.sleep(poll_interval)

    if pending:
        logger.warning(f"{len(pending)} files still processing after {timeout}s timeout")
        for vpath in pending:
            del uploaded[vpath]


def cleanup_files(
    uploaded: Dict[str, Any],
    api_key: Optional[str] = None,
):
    """Delete uploaded files from Gemini File API."""
    if not uploaded:
        return

    client = _get_client(api_key)
    for vpath, fobj in uploaded.items():
        try:
            client.files.delete(name=fobj.name)
            logger.debug(f"deleted {fobj.name}")
        except Exception as e:
            logger.debug(f"failed to delete {fobj.name}: {e}")


# -------------------------
# Context caching
# -------------------------

def create_system_cache(
    system_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    ttl: str = "3600s",
) -> Optional[Any]:
    """Create a context cache for the system prompt.

    Returns the cache object, or None if caching is not possible
    (e.g., prompt too short).
    """
    from google.genai import types

    client = _get_client(api_key)

    try:
        cache = client.caches.create(
            model=f"models/{model}",
            config=types.CreateCachedContentConfig(
                display_name="video_caption_system_prompt",
                system_instruction=system_prompt,
                ttl=ttl,
            ),
        )
        logger.info(f"created system prompt cache: {cache.name}")
        return cache
    except Exception as e:
        logger.warning(f"context caching failed (will proceed without): {e}")
        return None


def delete_cache(cache, api_key: Optional[str] = None):
    """Delete a context cache."""
    if cache is None:
        return
    try:
        client = _get_client(api_key)
        client.caches.delete(cache.name)
        logger.debug(f"deleted cache: {cache.name}")
    except Exception as e:
        logger.debug(f"failed to delete cache: {e}")


# -------------------------
# Batch submission
# -------------------------

def submit_batch(
    video_file_map: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    cache=None,
) -> Tuple[Any, Dict[str, str]]:
    """Submit a batch job for video understanding.

    Args:
        video_file_map: mapping of video_path -> uploaded file object
        system_prompt: system prompt text (ignored if cache is provided)
        user_prompt: user prompt template (no placeholders needed)
        model: Gemini model ID
        api_key: API key
        cache: optional context cache object

    Returns:
        (batch_job, key_to_video_path mapping)
    """
    from google.genai import types

    client = _get_client(api_key)

    # Build inline requests
    inline_requests = []
    key_map: Dict[str, str] = {}  # request key -> video_path

    for vpath, fobj in video_file_map.items():
        key = os.path.basename(vpath)
        # Ensure unique keys
        if key in key_map:
            import hashlib
            uid = hashlib.md5(vpath.encode()).hexdigest()[:8]
            key = f"{uid}_{key}"
        key_map[key] = vpath

        request = {
            "key": key,
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"file_data": {"file_uri": fobj.uri, "mime_type": "video/mp4"}},
                            {"text": user_prompt},
                        ],
                    }
                ],
            },
        }

        # Add system instruction or cached content
        if cache is not None:
            request["request"]["cached_content"] = cache.name
        else:
            request["request"]["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        inline_requests.append(request)

    logger.info(f"submitting gemini batch with {len(inline_requests)} requests...")

    batch_job = client.batches.create(
        model=f"models/{model}",
        src=inline_requests,
        config={"display_name": f"video_caption_{len(inline_requests)}"},
    )

    logger.info(f"batch submitted: {batch_job.name}")
    return batch_job, key_map


# -------------------------
# Polling / status
# -------------------------

def poll_batch(
    batch_job,
    api_key: Optional[str] = None,
    poll_interval: int = 20,
    timeout: int = 7200,
    on_progress=None,
) -> Any:
    """Poll batch job until completion.

    Returns the final batch_job object.
    """
    client = _get_client(api_key)
    start = time.time()

    while (time.time() - start) < timeout:
        try:
            batch_job = client.batches.get(name=batch_job.name)
        except Exception as e:
            logger.warning(f"poll error: {e}")
            time.sleep(poll_interval)
            continue

        state_name = batch_job.state.name if batch_job.state else "UNKNOWN"

        if on_progress:
            on_progress(state_name, batch_job)

        if state_name in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                          "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            return batch_job

        time.sleep(poll_interval)

    logger.warning(f"batch poll timed out after {timeout}s")
    return batch_job


# -------------------------
# Result collection
# -------------------------

def collect_results(
    batch_job,
    key_map: Dict[str, str],
) -> Dict[str, Optional[str]]:
    """Extract text results from a completed batch job.

    Returns:
        mapping of video_path -> description text (or None if failed/refused)
    """
    results: Dict[str, Optional[str]] = {}

    state_name = batch_job.state.name if batch_job.state else "UNKNOWN"
    if state_name != "JOB_STATE_SUCCEEDED":
        logger.error(f"batch job did not succeed: {state_name}")
        # Mark all as failed
        for vpath in key_map.values():
            results[vpath] = None
        return results

    # Inline responses
    dest = getattr(batch_job, "dest", None)
    if dest is None:
        logger.error("batch job has no dest attribute")
        for vpath in key_map.values():
            results[vpath] = None
        return results

    responses = getattr(dest, "inlined_responses", None) or []

    for resp in responses:
        key = getattr(resp, "key", None)
        vpath = key_map.get(key)
        if vpath is None:
            logger.warning(f"unknown response key: {key}")
            continue

        try:
            response = getattr(resp, "response", None)
            if response is None:
                results[vpath] = None
                continue

            # Check for errors/refusals
            error = getattr(resp, "error", None)
            if error:
                logger.warning(f"gemini refused/errored for {vpath}: {error}")
                results[vpath] = None
                continue

            # Extract text from candidates
            text = getattr(response, "text", None)
            if text:
                results[vpath] = text.strip()
            else:
                # Try extracting from candidates manually
                candidates = getattr(response, "candidates", [])
                if candidates:
                    parts = getattr(candidates[0].content, "parts", [])
                    text_parts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                    if text_parts:
                        results[vpath] = "\n".join(text_parts).strip()
                    else:
                        results[vpath] = None
                else:
                    results[vpath] = None
        except Exception as e:
            logger.warning(f"error extracting response for {vpath}: {e}")
            results[vpath] = None

    # Mark any videos without responses
    for vpath in key_map.values():
        if vpath not in results:
            results[vpath] = None

    succeeded = sum(1 for v in results.values() if v is not None)
    failed = sum(1 for v in results.values() if v is None)
    logger.info(f"gemini results: {succeeded} succeeded, {failed} failed/refused")

    return results


# -------------------------
# High-level orchestration
# -------------------------

def run_gemini_video_descriptions(
    video_paths: List[str],
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    poll_interval: int = 20,
    on_upload_progress=None,
    on_poll_progress=None,
) -> Dict[str, Optional[str]]:
    """End-to-end: upload videos → cache prompt → batch submit → poll → collect → cleanup.

    Returns mapping of video_path -> description text (None for failures).
    """
    if not video_paths:
        return {}

    # Step 1: Upload videos
    logger.info(f"uploading {len(video_paths)} videos to Gemini File API...")
    uploaded = upload_videos(video_paths, api_key=api_key, on_progress=on_upload_progress)

    if not uploaded:
        logger.error("no videos were successfully uploaded")
        return {vp: None for vp in video_paths}

    try:
        # Step 2: Create context cache for system prompt
        cache = create_system_cache(system_prompt, model=model, api_key=api_key)

        try:
            # Step 3: Submit batch
            batch_job, key_map = submit_batch(
                uploaded, system_prompt, user_prompt,
                model=model, api_key=api_key, cache=cache,
            )

            # Step 4: Poll until done
            batch_job = poll_batch(
                batch_job, api_key=api_key,
                poll_interval=poll_interval,
                on_progress=on_poll_progress,
            )

            # Step 5: Collect results
            results = collect_results(batch_job, key_map)

        finally:
            # Step 6: Delete cache
            delete_cache(cache, api_key=api_key)
    finally:
        # Step 7: Cleanup uploaded files
        cleanup_files(uploaded, api_key=api_key)

    # Include videos that failed to upload
    for vp in video_paths:
        if vp not in results:
            results[vp] = None

    return results
