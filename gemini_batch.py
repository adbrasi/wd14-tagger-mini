"""Gemini Batch API integration for video understanding.

Handles:
- Video upload via File API
- Batch job submission and collection (inline requests)
- Cleanup of uploaded files

Note: Context caching is NOT supported with the Batch API.
The system prompt is sent inline per request.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


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
                # Use .name for stable enum comparison (avoids "FileState.ACTIVE" vs "ACTIVE")
                state_name = state.name if hasattr(state, "name") else str(state or "")

                if state_name == "PROCESSING":
                    still_pending[vpath] = refreshed  # keep refreshed object for next poll
                elif state_name == "FAILED":
                    logger.warning(f"file processing failed for {vpath}, skipping")
                    del uploaded[vpath]
                else:
                    # ACTIVE or other terminal state
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
# Batch submission
# -------------------------

def submit_batch(
    video_file_map: Dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """Submit a batch job for video understanding.

    Args:
        video_file_map: mapping of video_path -> uploaded file object
        system_prompt: system prompt text
        user_prompt: user prompt text
        model: Gemini model ID
        api_key: API key

    Returns:
        (batch_job, ordered list of video paths matching response order)
    """
    client = _get_client(api_key)

    # Build inline requests — order is preserved in responses
    inline_requests = []
    ordered_paths: List[str] = []

    for vpath, fobj in video_file_map.items():
        ordered_paths.append(vpath)

        request = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"file_data": {"file_uri": fobj.uri, "mime_type": "video/mp4"}},
                        {"text": user_prompt},
                    ],
                }
            ],
        }

        inline_requests.append(request)

    logger.info(f"submitting gemini batch with {len(inline_requests)} requests...")

    batch_job = client.batches.create(
        model=f"models/{model}",
        src=inline_requests,
        config={"display_name": f"video_caption_{len(inline_requests)}"},
    )

    logger.info(f"batch submitted: {batch_job.name}")
    return batch_job, ordered_paths


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

    logger.warning(
        f"batch poll timed out after {timeout}s — batch may still be running on Gemini servers. "
        f"Uploaded files will NOT be cleaned up to avoid breaking the in-flight batch."
    )
    return batch_job


# -------------------------
# Result collection
# -------------------------

def collect_results(
    batch_job,
    ordered_paths: List[str],
) -> Dict[str, Optional[str]]:
    """Extract text results from a completed batch job.

    Uses positional mapping — inline responses preserve insertion order.

    Returns:
        mapping of video_path -> description text (or None if failed/refused)
    """
    results: Dict[str, Optional[str]] = {}

    state_name = batch_job.state.name if batch_job.state else "UNKNOWN"
    if state_name != "JOB_STATE_SUCCEEDED":
        logger.error(f"batch job did not succeed: {state_name}")
        for vpath in ordered_paths:
            results[vpath] = None
        return results

    # Inline responses
    dest = getattr(batch_job, "dest", None)
    if dest is None:
        logger.error("batch job has no dest attribute")
        for vpath in ordered_paths:
            results[vpath] = None
        return results

    responses = getattr(dest, "inlined_responses", None) or []

    for i, resp in enumerate(responses):
        if i >= len(ordered_paths):
            logger.warning(f"more responses ({i + 1}) than requests ({len(ordered_paths)})")
            break

        vpath = ordered_paths[i]

        try:
            response = getattr(resp, "response", None)
            if response is None:
                # Check for error
                error = getattr(resp, "error", None)
                if error:
                    logger.warning(f"gemini refused/errored for {vpath}: {error}")
                results[vpath] = None
                continue

            # Extract text from response
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

    # Mark any videos without responses (fewer responses than requests)
    for vpath in ordered_paths:
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
    """End-to-end: upload videos → batch submit → poll → collect → cleanup.

    Returns mapping of video_path -> description text (None for failures).
    """
    if not video_paths:
        return {}

    results: Dict[str, Optional[str]] = {}

    # Step 1: Upload videos
    logger.info(f"uploading {len(video_paths)} videos to Gemini File API...")
    uploaded = upload_videos(video_paths, api_key=api_key, on_progress=on_upload_progress)

    if not uploaded:
        logger.error("no videos were successfully uploaded")
        return {vp: None for vp in video_paths}

    try:
        # Step 2: Submit batch (system prompt inline per request)
        batch_job, ordered_paths = submit_batch(
            uploaded, system_prompt, user_prompt,
            model=model, api_key=api_key,
        )

        # Step 3: Poll until done
        batch_job = poll_batch(
            batch_job, api_key=api_key,
            poll_interval=poll_interval,
            on_progress=on_poll_progress,
        )

        # Step 4: Collect results
        results = collect_results(batch_job, ordered_paths)

    finally:
        # Step 5: Cleanup uploaded files
        cleanup_files(uploaded, api_key=api_key)

    # Include videos that failed to upload
    for vp in video_paths:
        if vp not in results:
            results[vp] = None

    return results
