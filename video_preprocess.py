"""Video preprocessing: frame cutting and resolution normalization.

- Frame count snaps to F % 8 == 1 (1, 9, 17, 25, 33, 41, 49, ...)
- Width/Height snapped to multiples of 32 (scale-to-fit, minimal crop)
- Uses ffmpeg for all operations
- Parallel processing with ProcessPoolExecutor
"""
import os
import signal
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# NOTE: These functions run in subprocess workers — they must NOT import Rich
# (Rich Console is not fork-safe). Logging is returned as results instead.

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def snap_frames(n: int) -> int:
    """Snap frame count to nearest value satisfying F % 8 == 1.

    Valid values: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ...
    """
    if n <= 1:
        return 1
    # Find nearest F where F % 8 == 1
    lower = ((n - 1) // 8) * 8 + 1
    upper = lower + 8
    if abs(n - lower) <= abs(n - upper):
        return lower
    return upper


def snap_dimension(d: int) -> int:
    """Snap dimension to nearest multiple of 32."""
    return max(32, round(d / 32) * 32)


def get_video_info(video_path: str) -> Optional[dict]:
    """Get video width, height, frame count via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        import json
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))

        # nb_frames can be "N/A" for some containers
        nb = stream.get("nb_frames", "N/A")
        if nb == "N/A" or not nb:
            # Estimate from duration * fps
            duration = float(fmt.get("duration", 0))
            fps_str = stream.get("r_frame_rate", "24/1")
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 24.0
            nb = int(duration * fps) if duration else 0
        else:
            nb = int(nb)

        return {"width": w, "height": h, "frames": nb}
    except Exception:
        return None


def _has_audio_stream(video_path: str) -> bool:
    """Check if video has an audio stream via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=10,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def _extract_ffmpeg_error(stderr: str) -> str:
    """Extract meaningful error line from ffmpeg stderr."""
    lines = (stderr or "").strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("ffmpeg version", "built with", "configuration:",
                            "lib", "  ", "Metadata:", "Stream #", "Input #",
                            "Output #", "Duration:", "Press [q]")):
            continue
        return line[:200]
    return lines[-1].strip()[:200] if lines else "unknown error"


def _run_ffmpeg(cmd: list, tmp_path: str) -> tuple:
    """Run ffmpeg command. Returns (success: bool, stderr: str)."""
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return False, "timeout (600s)"
        return proc.returncode == 0, stderr or ""
    except Exception as e:
        return False, str(e)


def _build_ffmpeg_cmd(
    video_path: str, tmp_path: str,
    max_frames: Optional[int],
    target_w: Optional[int], target_h: Optional[int],
    has_audio: bool,
) -> list:
    """Build ffmpeg command with proper pixel format handling."""
    filters = []

    # Force pixel format first — fixes "Conversion failed!" and assertion errors
    # with exotic input formats (yuv444p10le, gbrp, etc.)
    filters.append("format=yuv420p")

    if target_w and target_h:
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )

    cmd = ["ffmpeg", "-y", "-i", video_path]

    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])

    cmd.extend(["-vf", ",".join(filters)])
    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18"])

    if has_audio:
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.append(tmp_path)
    return cmd


def process_single_video(
    video_path: str,
    max_frames: Optional[int],
    target_w: Optional[int],
    target_h: Optional[int],
) -> dict:
    """Process a single video: cut frames and/or resize.

    Strategy: try with resize+format, fallback to format-only on failure.
    Returns dict with status info (no Rich imports — subprocess safe).
    """
    result = {"path": video_path, "ok": False, "detail": ""}
    tmp_path = video_path + ".tmp" + os.path.splitext(video_path)[1]
    has_audio = _has_audio_stream(video_path)

    # Attempt 1: full processing (format + scale/crop if requested)
    cmd = _build_ffmpeg_cmd(video_path, tmp_path, max_frames, target_w, target_h, has_audio)
    ok, stderr = _run_ffmpeg(cmd, tmp_path)

    if not ok and (target_w and target_h):
        # Attempt 2: retry without resize (just frame cut + format fix)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        cmd = _build_ffmpeg_cmd(video_path, tmp_path, max_frames, None, None, has_audio)
        ok, stderr = _run_ffmpeg(cmd, tmp_path)
        if ok:
            result["detail"] = "ok (resize skipped)"

    if not ok and has_audio:
        # Attempt 3: retry without audio (some audio streams are incompatible)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        cmd = _build_ffmpeg_cmd(video_path, tmp_path, max_frames, None, None, False)
        ok, stderr = _run_ffmpeg(cmd, tmp_path)
        if ok:
            result["detail"] = "ok (resize+audio skipped)"

    if ok:
        try:
            os.replace(tmp_path, video_path)
            result["ok"] = True
        except Exception as e:
            result["detail"] = f"replace failed: {e}"
    else:
        result["detail"] = _extract_ffmpeg_error(stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result


def preprocess_videos(
    video_paths: list,
    max_frames: Optional[int] = None,
    resize: bool = False,
    max_workers: int = None,
    on_progress=None,
) -> dict:
    """Preprocess multiple videos in parallel.

    Args:
        video_paths: list of video file paths
        max_frames: if set, snap to F%8==1 and cut. None = skip cutting.
        resize: if True, snap W/H to multiples of 32
        max_workers: parallel workers (default: min(cpu_count, 16))

    Returns: {total, success, failed, skipped, details: [{path, ok, detail}]}
    """
    if not video_paths:
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0, "details": []}

    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 4, len(video_paths), 64))

    snapped_frames = snap_frames(max_frames) if max_frames is not None else None

    # Pre-scan videos for dimensions and frame counts (parallel ffprobe)
    need_probe = resize or (snapped_frames is not None)
    infos = {}  # vp -> info_dict_or_None
    if need_probe:
        with ProcessPoolExecutor(max_workers=max_workers) as probe_pool:
            probe_futures = {probe_pool.submit(get_video_info, vp): vp for vp in video_paths}
            for f in as_completed(probe_futures):
                vp = probe_futures[f]
                infos[vp] = f.result()
                if on_progress:
                    on_progress(0, 0, len(video_paths))  # signal scan progress

    targets = {}
    probe_failures = 0
    for vp in video_paths:
        tw, th = None, None
        needs_cut = snapped_frames is not None
        probe_ok = True
        info = infos.get(vp)

        if resize:
            if info and info["width"] and info["height"]:
                tw = snap_dimension(info["width"])
                th = snap_dimension(info["height"])
                if tw == info["width"] and th == info["height"]:
                    tw, th = None, None
            elif info is None:
                probe_ok = False
                probe_failures += 1

        if snapped_frames is not None and info and info.get("frames", 0) > 0:
            if info["frames"] <= snapped_frames:
                needs_cut = False

        needs_work = needs_cut or (tw is not None)
        if not probe_ok and snapped_frames:
            needs_work = True

        targets[vp] = (tw, th, needs_work, needs_cut)

    to_process = [
        (vp, snapped_frames if needs_cut else None, tw, th)
        for vp, (tw, th, needs, needs_cut) in targets.items()
        if needs
    ]
    skipped = len(video_paths) - len(to_process)

    stats = {"total": len(video_paths), "success": 0, "failed": 0, "skipped": skipped, "details": []}

    if not to_process:
        return stats

    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(process_single_video, vp, mf, tw, th): vp
            for vp, mf, tw, th in to_process
        }
        for future in as_completed(futures):
            res = future.result()
            stats["details"].append(res)
            if res["ok"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            if on_progress:
                on_progress(stats["success"], stats["failed"], len(to_process))
    except KeyboardInterrupt:
        # Kill all running workers immediately
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False)

    return stats
