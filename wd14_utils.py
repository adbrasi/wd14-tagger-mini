import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


def setup_logging(log_level: Optional[str] = None) -> None:
    if logging.root.handlers:
        return

    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = None
    try:
        from rich.logging import RichHandler
        from rich.console import Console

        handler = RichHandler(console=Console(stderr=True))
    except Exception:
        handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logging.root.setLevel(level)
    logging.root.addHandler(handler)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".avif",
    ".jxl",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
}


def glob_images_pathlib(base_dir: Path, recursive: bool) -> List[Path]:
    if base_dir.is_file():
        return [base_dir]

    if recursive:
        paths: Iterable[Path] = base_dir.rglob("*")
    else:
        paths = base_dir.iterdir()

    images: List[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    return images


def glob_videos_pathlib(base_dir: Path, recursive: bool) -> List[Path]:
    if base_dir.is_file():
        if base_dir.suffix.lower() in VIDEO_EXTENSIONS:
            return [base_dir]
        return []

    if recursive:
        paths: Iterable[Path] = base_dir.rglob("*")
    else:
        paths = base_dir.iterdir()

    videos: List[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
    return videos


def extract_first_frame(video_path: str, output_path: Optional[str] = None) -> Optional[Image.Image]:
    """Extract the first frame from a video file and return as PIL Image."""
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python-headless is required for video support. Install with: pip install opencv-python-headless")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.warning(f"Could not read frame from: {video_path}")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    if output_path:
        pil_image.save(output_path)
        logger.debug(f"Saved frame to: {output_path}")

    return pil_image


def get_pil_interpolation(interpolation: Optional[str]) -> Optional[Image.Resampling]:
    if interpolation is None:
        return None

    value = interpolation.lower()
    if value == "lanczos":
        return Image.Resampling.LANCZOS
    if value == "nearest":
        return Image.Resampling.NEAREST
    if value in {"bilinear", "linear"}:
        return Image.Resampling.BILINEAR
    if value in {"bicubic", "cubic"}:
        return Image.Resampling.BICUBIC
    if value == "area":
        return Image.Resampling.HAMMING
    if value == "box":
        return Image.Resampling.BOX
    return None


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    resized_width: int,
    resized_height: int,
    resize_interpolation: Optional[str] = None,
) -> np.ndarray:
    width = int(width)
    height = int(height)
    resized_width = int(resized_width)
    resized_height = int(resized_height)

    if resize_interpolation is None:
        if width >= resized_width and height >= resized_height:
            resize_interpolation = "area"
        else:
            resize_interpolation = "lanczos"

    interpolation = get_pil_interpolation(resize_interpolation)

    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
    if has_alpha:
        pil_image = Image.fromarray(image[:, :, [2, 1, 0, 3]], mode="RGBA")
    else:
        pil_image = Image.fromarray(image[:, :, ::-1], mode="RGB")

    resized = pil_image.resize((resized_width, resized_height), resample=interpolation)

    if has_alpha:
        out = np.array(resized)[:, :, [2, 1, 0, 3]]
    else:
        out = np.array(resized)[:, :, ::-1]

    return out
