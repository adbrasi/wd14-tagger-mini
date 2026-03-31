#!/usr/bin/env python3
"""Build WebDataset TAR shards from video+txt pairs.

Creates TAR archives in the standard WebDataset format:
  train-0000.tar, train-0001.tar, ...

Each TAR contains paired files with globally unique numeric keys:
  shard 0: 000000.mp4 + 000000.txt, 000001.mp4 + 000001.txt, ...
  shard 1: 001000.mp4 + 001000.txt, ...  (keys are globally unique)

Optimized for HuggingFace Hub streaming with load_dataset("webdataset").
"""

import argparse
import json
import os
import re
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

from constants import VIDEO_EXTS
DEFAULT_SHARD_SIZE_GB = 1.0
DEFAULT_SPLIT = "train"


def find_video_txt_pairs(root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Find all video files and their matching .txt caption files.

    Returns list of (video_path, txt_path_or_None) sorted by full path.
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.suffix.lower() in VIDEO_EXTS:
                txt = p.with_suffix(".txt")
                pairs.append((p, txt if txt.exists() else None))
    pairs.sort(key=lambda x: str(x[0]))
    return pairs


def build_tars(
    root: Path,
    output_dir: Path,
    shard_size_gb: float = DEFAULT_SHARD_SIZE_GB,
    split: str = DEFAULT_SPLIT,
) -> dict:
    """Build WebDataset TAR shards from video+txt pairs.

    Args:
        root: directory containing video files and .txt captions.
        output_dir: where to write the TAR shards.
        shard_size_gb: target size per shard in GB.
        split: shard name prefix (e.g. "train").

    Returns:
        dict with stats: total_pairs, samples_written, shards_created,
                         total_bytes, missing_txt, shard_paths.
    """
    pairs = find_video_txt_pairs(root)
    if not pairs:
        raise RuntimeError(f"no video files found in {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_shard_bytes = int(shard_size_gb * (1024 ** 3))

    stats = {
        "total_pairs": len(pairs),
        "samples_written": 0,
        "shards_created": 0,
        "total_bytes": 0,
        "missing_txt": 0,
        "shard_paths": [],
    }

    shard_idx = 0
    sample_idx = 0
    current_bytes = 0
    tar: Optional[tarfile.TarFile] = None
    tar_path: Optional[Path] = None

    def _close_current_shard():
        """Close current shard and record its stats using actual TAR size."""
        nonlocal tar, tar_path, current_bytes
        if tar is not None:
            tar.close()
            tar = None
            actual_bytes = tar_path.stat().st_size
            stats["shard_paths"].append(str(tar_path))
            stats["shards_created"] += 1
            stats["total_bytes"] += actual_bytes
            current_bytes = 0

    def _open_new_shard():
        """Open a fresh TAR shard."""
        nonlocal tar, tar_path, shard_idx
        tar_path = output_dir / f"{split}-{shard_idx:04d}.tar"
        tar = tarfile.open(tar_path, "w")
        shard_idx += 1

    try:
        _open_new_shard()

        for video_path, txt_path in pairs:
            video_size = video_path.stat().st_size
            txt_size = txt_path.stat().st_size if txt_path else 0
            pair_size = video_size + txt_size

            # Start new shard if current one would exceed target
            if current_bytes > 0 and current_bytes + pair_size > max_shard_bytes:
                _close_current_shard()
                _open_new_shard()

            key = f"{sample_idx:06d}"
            video_ext = video_path.suffix.lower()

            # Add video
            tar.add(str(video_path), arcname=f"{key}{video_ext}")

            # Add txt caption
            if txt_path:
                tar.add(str(txt_path), arcname=f"{key}.txt")
            else:
                stats["missing_txt"] += 1

            current_bytes += pair_size
            sample_idx += 1
            stats["samples_written"] += 1

        # Close final shard
        _close_current_shard()

    finally:
        # Safety: ensure TAR handle is always closed + clean up empty shard
        if tar is not None:
            tar.close()
            tar = None
            # Remove empty/partial shard that was never properly closed
            if tar_path and tar_path.exists() and tar_path.stat().st_size <= 1024:
                try:
                    tar_path.unlink()
                except OSError:
                    pass

    # Write metadata for dataset loading
    _write_metadata(output_dir, split, stats)

    return stats


def _write_metadata(output_dir: Path, split: str, stats: dict):
    """Write a minimal dataset_info.json for HuggingFace compatibility."""
    info = {
        "description": "Video dataset in WebDataset TAR format",
        "split": split,
        "total_samples": stats["samples_written"],
        "total_shards": stats["shards_created"],
        "total_bytes": stats["total_bytes"],
        "missing_captions": stats["missing_txt"],
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Build WebDataset TAR shards from video+txt pairs"
    )
    parser.add_argument(
        "--root", required=True,
        help="directory with video files and .txt captions",
    )
    parser.add_argument(
        "--output", required=True,
        help="output directory for TAR shards",
    )
    parser.add_argument(
        "--shard_size_gb", type=float, default=DEFAULT_SHARD_SIZE_GB,
        help=f"target shard size in GB (default: {DEFAULT_SHARD_SIZE_GB})",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT,
        help=f"shard name prefix (default: {DEFAULT_SPLIT})",
    )
    args = parser.parse_args()

    if args.shard_size_gb <= 0:
        raise SystemExit("--shard_size_gb must be positive")

    if not re.match(r'^[A-Za-z0-9_-]+$', args.split):
        raise SystemExit(f"--split must match [A-Za-z0-9_-]+, got: {args.split!r}")

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    stats = build_tars(
        root=root,
        output_dir=output,
        shard_size_gb=args.shard_size_gb,
        split=args.split,
    )

    print(f"Done: {stats['shards_created']} shards, "
          f"{stats['samples_written']} samples, "
          f"{stats['total_bytes'] / (1024**3):.2f} GB")
    if stats["missing_txt"]:
        print(f"Warning: {stats['missing_txt']} videos without .txt caption")


if __name__ == "__main__":
    main()
