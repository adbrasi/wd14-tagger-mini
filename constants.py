"""Project-wide constants shared across all scripts."""

# Superset of all image extensions used anywhere in the project.
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
)

# Superset of all video extensions used anywhere in the project.
VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
)

# Convenience union for code that handles any media file.
MEDIA_EXTS: frozenset[str] = IMAGE_EXTS | VIDEO_EXTS
