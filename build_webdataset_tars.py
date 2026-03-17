#!/usr/bin/env python3
"""Build WebDataset TAR shards from video+txt pairs.

Creates TAR archives in the standard WebDataset format:
  train-0000.tar, train-0001.tar, ...

Each TAR contains paired files with matching numeric keys:
  000.mp4 + 000.txt, 001.mp4 + 001.txt, ...

Optimized for HuggingFace Hub streaming with load_dataset("webdataset").
"""

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
DEFAULT_SHARD_SIZE_GB = 1.0
DEFAULT_SPLIT = "train"


def find_video_txt_pairs(root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Find all video files and their matching .txt caption files.

    Returns list of (video_path, txt_path_or_None) sorted by name.
    """
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.suffix.lower() in VIDEO_EXTS:
                txt = p.with_suffix(".txt")
                pairs.append((p, txt if txt.exists() else None))
    pairs.sort(key=lambda x: x[0].name)
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
        dict with stats: total_pairs, shards_created, total_bytes,
                         missing_txt, shard_paths.
    """
    pairs = find_video_txt_pairs(root)
    if not pairs:
        raise RuntimeError(f"no video files found in {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_shard_bytes = int(shard_size_gb * (1024 ** 3))

    stats = {
        "total_pairs": len(pairs),
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

    def open_new_shard():
        nonlocal tar, tar_path, shard_idx, current_bytes, sample_idx
        if tar is not None:
            tar.close()
        tar_path = output_dir / f"{split}-{shard_idx:04d}.tar"
        tar = tarfile.open(tar_path, "w")
        current_bytes = 0
        sample_idx = 0
        shard_idx += 1

    open_new_shard()

    for video_path, txt_path in pairs:
        video_size = video_path.stat().st_size
        txt_size = txt_path.stat().st_size if txt_path else 0
        pair_size = video_size + txt_size

        # Start new shard if current one would exceed target
        if current_bytes > 0 and current_bytes + pair_size > max_shard_bytes:
            tar.close()
            stats["shard_paths"].append(str(tar_path))
            stats["shards_created"] += 1
            stats["total_bytes"] += current_bytes
            open_new_shard()

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

    # Close final shard
    if tar is not None:
        tar.close()
        stats["shard_paths"].append(str(tar_path))
        stats["shards_created"] += 1
        stats["total_bytes"] += current_bytes

    # Write metadata for dataset loading
    _write_metadata(output_dir, split, stats)

    return stats


def _write_metadata(output_dir: Path, split: str, stats: dict):
    """Write a minimal dataset_info.json for HuggingFace compatibility."""
    info = {
        "description": "Video dataset in WebDataset TAR format",
        "split": split,
        "total_samples": stats["total_pairs"],
        "total_shards": stats["shards_created"],
        "total_bytes": stats["total_bytes"],
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
          f"{stats['total_pairs']} samples, "
          f"{stats['total_bytes'] / (1024**3):.2f} GB")
    if stats["missing_txt"]:
        print(f"Warning: {stats['missing_txt']} videos without .txt caption")


if __name__ == "__main__":
    main()
