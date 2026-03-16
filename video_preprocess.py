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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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


def process_single_video(
    video_path: str,
    max_frames: Optional[int],
    target_w: Optional[int],
    target_h: Optional[int],
) -> dict:
    """Process a single video: cut frames and/or resize.

    Returns dict with status info (no Rich imports — subprocess safe).
    """
    result = {"path": video_path, "ok": False, "detail": ""}
    tmp_path = video_path + ".tmp" + os.path.splitext(video_path)[1]

    filters = []

    # Build ffmpeg filter chain
    if target_w and target_h:
        # Scale to fit within target_w x target_h, then crop to exact dimensions
        filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )

    cmd = ["ffmpeg", "-y", "-i", video_path]

    if max_frames is not None:
        cmd.extend(["-vframes", str(max_frames)])

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        tmp_path,
    ])

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _, stderr = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            result["detail"] = "timeout (600s)"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return result

        if proc.returncode != 0:
            result["detail"] = (stderr or "")[:200]
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return result
        os.replace(tmp_path, video_path)
        result["ok"] = True
        return result
    except Exception as e:
        result["detail"] = str(e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return result


def preprocess_videos(
    video_paths: list,
    max_frames: Optional[int] = None,
    resize: bool = False,
    max_workers: int = None,
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
        max_workers = max(1, min(os.cpu_count() or 4, len(video_paths), 16))

    snapped_frames = snap_frames(max_frames) if max_frames is not None else None

    # Pre-scan videos for dimensions and frame counts
    targets = {}  # path -> (target_w, target_h, needs_work)
    probe_failures = 0
    for vp in video_paths:
        tw, th = None, None
        needs_cut = snapped_frames is not None
        probe_ok = True

        info = get_video_info(vp) if (resize or snapped_frames) else None

        if resize:
            if info and info["width"] and info["height"]:
                tw = snap_dimension(info["width"])
                th = snap_dimension(info["height"])
                # Skip resize if already aligned
                if tw == info["width"] and th == info["height"]:
                    tw, th = None, None
            elif info is None:
                probe_ok = False
                probe_failures += 1

        # Skip frame cut if video already has <= snapped_frames
        if snapped_frames is not None and info and info.get("frames", 0) > 0:
            if info["frames"] <= snapped_frames:
                needs_cut = False

        needs_work = needs_cut or (tw is not None)

        # If probe failed but we need to cut, still submit (ffmpeg handles it)
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

    return stats
