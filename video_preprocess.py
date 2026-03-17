"""Video preprocessing: trim videos to a maximum frame count.

Uses stream copy (-c copy) for lossless, codec-agnostic trimming.
No re-encoding, no filters, no pixel format conversion.
Works with any video codec/container. Preserves audio when present.

Frame count target follows F % 8 == 1 (1, 9, 17, 25, 33, 41, 49, ...).
Cutting is time-based: cut_time = max_frames / fps.
"""
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# NOTE: These functions run in subprocess workers — they must NOT import Rich
# (Rich Console is not fork-safe). Logging is returned as results instead.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

_DEVNULL = subprocess.DEVNULL


def snap_frames(n: int) -> int:
    """Snap frame count to nearest value satisfying F % 8 == 1.

    Valid values: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ...
    """
    if n <= 1:
        return 1
    lower = ((n - 1) // 8) * 8 + 1
    upper = lower + 8
    # On tie, prefer the larger value (keep more frames)
    return lower if abs(n - lower) < abs(n - upper) else upper


def get_video_info(video_path: str) -> Optional[dict]:
    """Get video fps, frame count, and duration via ffprobe.

    Returns dict with keys: fps (float), frames (int), duration (float).
    Returns None on any error (corrupt file, missing codec, etc.).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, stdin=_DEVNULL,
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None

        data = json.loads(r.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        # Prefer r_frame_rate (container clock used by -t in stream copy mode),
        # fall back to avg_frame_rate for files missing r_frame_rate
        fps = _parse_fps(stream.get("r_frame_rate", ""))
        if fps <= 0:
            fps = _parse_fps(stream.get("avg_frame_rate", ""))

        duration = float(fmt.get("duration", 0))

        # nb_frames can be "N/A" or missing for some containers
        nb = stream.get("nb_frames", "N/A")
        if nb and nb != "N/A":
            frames = int(nb)
        elif duration > 0 and fps > 0:
            frames = int(duration * fps)
        else:
            frames = 0

        return {"fps": fps, "frames": frames, "duration": duration}
    except Exception:
        return None


def _parse_fps(fps_str: str) -> float:
    """Parse ffprobe fps string like '30000/1001' or '24/1'."""
    if not fps_str or "/" not in fps_str:
        return 0.0
    try:
        num, den = fps_str.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _extract_ffmpeg_error(stderr: str) -> str:
    """Extract meaningful error line from ffmpeg stderr output."""
    lines = (stderr or "").strip().split("\n")
    skip_prefixes = (
        "ffmpeg version", "built with", "configuration:",
        "lib", "  ", "Metadata:", "Stream #", "Input #",
        "Output #", "Duration:", "Press [q]",
    )
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(skip_prefixes):
            continue
        return stripped[:200]
    return (lines[-1].strip()[:200] if lines else "") or "unknown error"


def trim_video(video_path: str, cut_seconds: float) -> dict:
    """Trim a single video to cut_seconds using stream copy.

    Stream copy is lossless, codec-agnostic, and preserves all streams
    (video, audio, subtitles). No re-encoding occurs.

    Returns dict: {path, ok, detail}.
    """
    result = {"path": video_path, "ok": False, "detail": ""}

    # Create tmp file on same filesystem for atomic os.replace
    video_dir = os.path.dirname(video_path) or "."
    ext = os.path.splitext(video_path)[1]
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=video_dir)
        os.close(fd)
    except OSError as e:
        result["detail"] = f"cannot create temp file: {e}"
        return result

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", f"{cut_seconds:.4f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        tmp_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=_DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result["detail"] = "timeout (300s)"
            _remove_if_exists(tmp_path)
            return result

        if proc.returncode != 0:
            result["detail"] = _extract_ffmpeg_error(stderr)
            _remove_if_exists(tmp_path)
            return result

        # Verify output is valid (non-zero size)
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            result["detail"] = "output file empty or missing"
            _remove_if_exists(tmp_path)
            return result

        os.replace(tmp_path, video_path)
        result["ok"] = True
        return result

    except Exception as e:
        result["detail"] = str(e)
        _remove_if_exists(tmp_path)
        return result


def _remove_if_exists(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def preprocess_videos(
    video_paths: list,
    max_frames: int,
    max_workers: int = None,
    on_progress=None,
) -> dict:
    """Trim multiple videos in parallel using stream copy.

    max_frames must already be snapped to F%8==1 by the caller.

    Args:
        video_paths: list of video file paths.
        max_frames: target frame count (already snapped). Videos longer
                    than this will be trimmed.
        max_workers: parallel workers (default: min(cpu_count, 64)).
        on_progress: callback with signature:
            on_scan_progress(scan_done: int, scan_total: int) during Phase 1
            on_trim_progress(success: int, failed: int, trim_total: int) during Phase 2
            To distinguish phases, Phase 1 calls: callback("scan", done, total)
            Phase 2 calls: callback("trim", success, failed, total)

    Returns: {total, trimmed, failed, skipped, probe_failed,
              details: [{path, ok, detail}]}
    """
    empty = {"total": len(video_paths), "trimmed": 0, "failed": 0,
             "skipped": 0, "probe_failed": 0, "details": []}

    if not video_paths:
        empty["total"] = 0
        return empty

    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 4, len(video_paths), 64))

    # Phase 1: Parallel ffprobe to get fps + frame count
    infos = {}
    scan_done = 0
    interrupted = False
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = {executor.submit(get_video_info, vp): vp for vp in video_paths}
        for f in as_completed(futures):
            vp = futures[f]
            infos[vp] = f.result()
            scan_done += 1
            if on_progress:
                on_progress("scan", scan_done, len(video_paths))
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    # Phase 2: Decide which videos need trimming
    to_trim = []  # (video_path, cut_seconds)
    probe_failed = 0
    for vp in video_paths:
        info = infos.get(vp)
        if info is None:
            probe_failed += 1
            continue

        fps = info["fps"]
        frames = info["frames"]

        if fps <= 0 or frames <= 0:
            probe_failed += 1
            continue

        if frames <= max_frames:
            continue

        cut_seconds = max_frames / fps
        to_trim.append((vp, cut_seconds))

    skipped = len(video_paths) - len(to_trim) - probe_failed
    stats = {
        "total": len(video_paths),
        "trimmed": 0,
        "failed": 0,
        "skipped": skipped,
        "probe_failed": probe_failed,
        "details": [],
    }

    if not to_trim:
        # Signal trim phase done immediately
        if on_progress:
            on_progress("trim", 0, 0, 0)
        return stats

    # Signal trim phase start with correct total
    if on_progress:
        on_progress("trim", 0, 0, len(to_trim))

    # Phase 3: Parallel stream-copy trim
    interrupted = False
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(trim_video, vp, secs): vp
            for vp, secs in to_trim
        }
        for future in as_completed(futures):
            res = future.result()
            stats["details"].append(res)
            if res["ok"]:
                stats["trimmed"] += 1
            else:
                stats["failed"] += 1
            if on_progress:
                on_progress("trim", stats["trimmed"], stats["failed"], len(to_trim))
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    return stats
