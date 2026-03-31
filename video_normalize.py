"""Normalize video files: convert formats, fix mono audio, fix framerate.

Three independent operations, all parallelized:

1. Format normalization: convert gif/webp/avi/mov/mkv/webm/flv/wmv → mp4
   Uses re-encoding with libx264 + aac. Original non-mp4 file is removed.

2. Mono → stereo: duplicate mono channel into L+R via pan filter.
   Video stream is copied (no re-encode). Only audio is re-encoded.
   Videos already stereo or without audio are untouched.

3. Framerate normalization: re-encode video to a fixed fps (e.g. 25).
   Videos already at the target fps are skipped (no re-encode).
   Audio is stream-copied.

All operations work in-place with atomic replace via temp files.
"""

import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# Formats that need conversion to mp4
CONVERTIBLE_EXTS = {".gif", ".webp", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

_DEVNULL = subprocess.DEVNULL


def get_audio_channels(video_path: str) -> Optional[int]:
    """Return number of audio channels, or None if no audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, stdin=_DEVNULL,
                           capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if not out:
            return None
        return int(out)
    except Exception:
        return None


def convert_to_mp4(src_path: str) -> dict:
    """Convert a non-mp4 video to mp4 format. Removes original on success.

    Returns {path, ok, detail, new_path}.
    """
    result = {"path": src_path, "ok": False, "detail": "", "new_path": src_path}

    dst_path = os.path.splitext(src_path)[0] + ".mp4"
    video_dir = os.path.dirname(src_path) or "."

    # Guard against overwriting an existing mp4 with the same stem
    if os.path.exists(dst_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
        result["detail"] = f"destination already exists: {dst_path}"
        return result

    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=video_dir)
        os.close(fd)
    except OSError as e:
        result["detail"] = f"cannot create temp file: {e}"
        return result

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        tmp_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=_DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result["detail"] = "timeout (600s)"
            return result

        if proc.returncode != 0:
            result["detail"] = _last_error(stderr)
            return result

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            result["detail"] = "output empty"
            return result

        os.replace(tmp_path, dst_path)

        # Remove original non-mp4 file if it's different from destination
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            _remove(src_path)

        # Also update the .txt caption file name if it exists
        old_txt = os.path.splitext(src_path)[0] + ".txt"
        new_txt = os.path.splitext(dst_path)[0] + ".txt"
        if old_txt != new_txt and os.path.exists(old_txt) and not os.path.exists(new_txt):
            try:
                os.replace(old_txt, new_txt)
            except OSError:
                pass  # video converted ok, txt rename is best-effort

        result["ok"] = True
        result["new_path"] = dst_path
        return result

    except Exception as e:
        result["detail"] = str(e)
        return result
    finally:
        # Clean up temp file if it still exists (os.replace succeeded = file gone)
        _remove(tmp_path)


def fix_mono_to_stereo(video_path: str) -> dict:
    """Convert mono audio to stereo. Video is stream-copied (no re-encode).

    Uses pan filter to explicitly duplicate mono channel: L=mono, R=mono.

    Returns {path, ok, detail, action} where action is one of:
    "converted", "already_stereo", "no_audio", "error".
    """
    result = {"path": video_path, "ok": True, "detail": "", "action": "already_stereo"}

    channels = get_audio_channels(video_path)

    if channels is None:
        result["action"] = "no_audio"
        return result

    if channels >= 2:
        result["action"] = "already_stereo"
        return result

    # Mono → stereo
    video_dir = os.path.dirname(video_path) or "."
    ext = os.path.splitext(video_path)[1]

    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=video_dir)
        os.close(fd)
    except OSError as e:
        result["ok"] = False
        result["detail"] = f"cannot create temp file: {e}"
        result["action"] = "error"
        return result

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-af", "pan=stereo|c0=c0|c1=c0",
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
            result["ok"] = False
            result["detail"] = "timeout (300s)"
            result["action"] = "error"
            return result

        if proc.returncode != 0:
            result["ok"] = False
            result["detail"] = _last_error(stderr)
            result["action"] = "error"
            return result

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            result["ok"] = False
            result["detail"] = "output empty"
            result["action"] = "error"
            return result

        os.replace(tmp_path, video_path)
        result["action"] = "converted"
        return result

    except Exception as e:
        result["ok"] = False
        result["detail"] = str(e)
        result["action"] = "error"
        return result
    finally:
        # Clean up temp file if it still exists (os.replace succeeded = file gone)
        _remove(tmp_path)


def get_video_fps(video_path: str) -> Optional[float]:
    """Return the video stream fps, or None on error."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, stdin=_DEVNULL,
                           capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if not out or "/" not in out:
            return None
        num, den = out.split("/")
        d = float(den)
        return float(num) / d if d else None
    except Exception:
        return None


def _fix_fps_single(args: tuple) -> dict:
    """Re-encode a single video to target fps. Receives (video_path, target_fps).

    Returns {path, ok, detail, action} where action is one of:
    "converted", "already_correct", "probe_failed", "error".
    """
    video_path, target_fps = args
    result = {"path": video_path, "ok": True, "detail": "", "action": "already_correct"}

    current_fps = get_video_fps(video_path)
    if current_fps is None:
        result["ok"] = False
        result["action"] = "probe_failed"
        return result

    # Exact match only — no tolerance
    if current_fps == target_fps:
        result["action"] = "already_correct"
        return result

    video_dir = os.path.dirname(video_path) or "."
    _, ext = os.path.splitext(video_path)

    try:
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=video_dir)
        os.close(fd)
    except OSError as e:
        result["ok"] = False
        result["detail"] = f"cannot create temp file: {e}"
        result["action"] = "error"
        return result

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        tmp_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdin=_DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result["ok"] = False
            result["detail"] = "timeout (600s)"
            result["action"] = "error"
            return result

        if proc.returncode != 0:
            result["ok"] = False
            result["detail"] = _last_error(stderr)
            result["action"] = "error"
            return result

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            result["ok"] = False
            result["detail"] = "output empty"
            result["action"] = "error"
            return result

        os.replace(tmp_path, video_path)
        result["action"] = "converted"
        return result

    except Exception as e:
        result["ok"] = False
        result["detail"] = str(e)
        result["action"] = "error"
        return result
    finally:
        # Clean up temp file if it still exists (os.replace succeeded = file gone)
        _remove(tmp_path)


def normalize_videos(
    file_paths: list,
    fix_stereo: bool = True,
    target_fps: Optional[float] = None,
    max_workers: int = None,
    on_progress=None,
) -> dict:
    """Run format conversion, optional mono→stereo fix, and optional fps normalization.

    Phase 1: Convert non-mp4 files to mp4.
    Phase 2: Fix mono→stereo (if enabled).
    Phase 3: Normalize framerate (if target_fps is set).

    Args:
        file_paths: all media files (videos + convertible formats).
        fix_stereo: whether to run mono→stereo pass.
        target_fps: if set, re-encode videos to this fps. None = skip.
        max_workers: parallel workers.
        on_progress: callback with signature:
            ("convert", done, total) for Phase 1
            ("stereo", done, total) for Phase 2
            ("fps", done, total) for Phase 3

    Returns dict with stats.
    """
    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 4, len(file_paths), 64))

    stats = {
        "converted": 0, "convert_failed": 0, "convert_skipped": 0,
        "stereo_converted": 0, "stereo_skipped": 0, "stereo_no_audio": 0,
        "stereo_failed": 0,
        "fps_converted": 0, "fps_skipped": 0, "fps_failed": 0,
        "details": [],
    }

    # Phase 1: Format conversion
    to_convert = [p for p in file_paths if os.path.splitext(p)[1].lower() in CONVERTIBLE_EXTS]
    # All non-convertible files pass through to Phase 2/3 (includes .mp4 and any other format)
    non_convert = [p for p in file_paths if os.path.splitext(p)[1].lower() not in CONVERTIBLE_EXTS]
    stats["convert_skipped"] = len(non_convert)

    if on_progress:
        on_progress("convert", 0, len(to_convert))

    converted_paths = list(non_convert)
    interrupted = False

    if to_convert:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futures = {executor.submit(convert_to_mp4, p): p for p in to_convert}
            done = 0
            for f in as_completed(futures):
                res = f.result()
                done += 1
                if res["ok"]:
                    stats["converted"] += 1
                    converted_paths.append(res["new_path"])
                else:
                    stats["convert_failed"] += 1
                    stats["details"].append(res)
                if on_progress:
                    on_progress("convert", done, len(to_convert))
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    # Phase 2: Mono → stereo
    if fix_stereo and converted_paths:
        if on_progress:
            on_progress("stereo", 0, len(converted_paths))

        executor = ProcessPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = {executor.submit(fix_mono_to_stereo, p): p for p in converted_paths}
            done = 0
            for f in as_completed(futures):
                res = f.result()
                done += 1
                action = res["action"]
                if action == "converted":
                    stats["stereo_converted"] += 1
                elif action == "already_stereo":
                    stats["stereo_skipped"] += 1
                elif action == "no_audio":
                    stats["stereo_no_audio"] += 1
                else:
                    stats["stereo_failed"] += 1
                    stats["details"].append(res)
                if on_progress:
                    on_progress("stereo", done, len(converted_paths))
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    # Phase 3: Framerate normalization
    if target_fps and converted_paths:
        if on_progress:
            on_progress("fps", 0, len(converted_paths))

        executor = ProcessPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = {
                executor.submit(_fix_fps_single, (p, target_fps)): p
                for p in converted_paths
            }
            done = 0
            for f in as_completed(futures):
                res = f.result()
                done += 1
                action = res["action"]
                if action == "converted":
                    stats["fps_converted"] += 1
                elif action == "already_correct":
                    stats["fps_skipped"] += 1
                else:
                    stats["fps_failed"] += 1
                    stats["details"].append(res)
                if on_progress:
                    on_progress("fps", done, len(converted_paths))
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=interrupted)

    return stats


def _remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _last_error(stderr: str) -> str:
    lines = (stderr or "").strip().split("\n")
    for line in reversed(lines):
        s = line.strip()
        if s and not s.startswith(("ffmpeg version", "built with", "configuration:",
                                    "lib", "  ", "Metadata:", "Stream #",
                                    "Input #", "Output #", "Duration:", "Press [q]")):
            return s[:200]
    return (lines[-1].strip()[:200] if lines else "") or "unknown error"
