#!/usr/bin/env python3
"""Mini tool: build WebDataset TARs + upload to HuggingFace in one command.

Usage:
    python upload_tars_hf.py \
        --root /path/to/video+txt/pairs \
        --repo usuario/nome-do-dataset \
        [--shard_size_gb 1.0] \
        [--split train] \
        [--private] \
        [--workers 8] \
        [--skip-build]   # skip TAR creation, upload existing TARs only

Keys inside TARs are globally unique across shards (no overwrite on extract).
"""

import argparse
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

from constants import VIDEO_EXTS


# ── TAR builder ──────────────────────────────────────────────────────────────


def find_video_txt_pairs(root: Path) -> List[Tuple[Path, Optional[Path]]]:
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
    shard_size_gb: float = 1.0,
    split: str = "train",
) -> dict:
    pairs = find_video_txt_pairs(root)
    if not pairs:
        raise RuntimeError(f"no video files found in {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_shard_bytes = int(shard_size_gb * (1024**3))

    stats = {
        "total_pairs": len(pairs),
        "samples_written": 0,
        "shards_created": 0,
        "total_bytes": 0,
        "missing_txt": 0,
        "shard_paths": [],
    }

    shard_idx = 0
    sample_idx = 0  # global — never resets per shard
    current_bytes = 0
    tar: Optional[tarfile.TarFile] = None
    tar_path: Optional[Path] = None

    def _close():
        nonlocal tar, tar_path, current_bytes
        if tar is not None:
            tar.close()
            tar = None
            actual = tar_path.stat().st_size
            stats["shard_paths"].append(str(tar_path))
            stats["shards_created"] += 1
            stats["total_bytes"] += actual
            current_bytes = 0

    def _open():
        nonlocal tar, tar_path, shard_idx
        tar_path = output_dir / f"{split}-{shard_idx:04d}.tar"
        tar = tarfile.open(tar_path, "w")
        shard_idx += 1

    try:
        _open()
        for video_path, txt_path in pairs:
            video_size = video_path.stat().st_size
            txt_size = txt_path.stat().st_size if txt_path else 0
            pair_size = video_size + txt_size

            if current_bytes > 0 and current_bytes + pair_size > max_shard_bytes:
                _close()
                _open()

            key = f"{sample_idx:06d}"
            ext = video_path.suffix.lower()
            tar.add(str(video_path), arcname=f"{key}{ext}")
            if txt_path:
                tar.add(str(txt_path), arcname=f"{key}.txt")
            else:
                stats["missing_txt"] += 1

            current_bytes += pair_size
            sample_idx += 1
            stats["samples_written"] += 1

        _close()
    finally:
        if tar is not None:
            tar.close()
            if tar_path and tar_path.exists() and tar_path.stat().st_size <= 1024:
                try:
                    tar_path.unlink()
                except OSError:
                    pass

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(
            {
                "split": split,
                "total_samples": stats["samples_written"],
                "total_shards": stats["shards_created"],
                "total_bytes": stats["total_bytes"],
                "missing_captions": stats["missing_txt"],
            },
            f,
            indent=2,
        )

    return stats


# ── HuggingFace upload ───────────────────────────────────────────────────────


def upload_to_hf(folder: Path, repo_id: str, private: bool, workers: int):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    print(f"Uploading {folder} → hf://{repo_id} ({workers} workers)...")
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        num_workers=workers,
    )
    print(f"Done! https://huggingface.co/datasets/{repo_id}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Build WebDataset TARs + upload to HF")
    p.add_argument("--root", required=True, help="directory with video+txt pairs")
    p.add_argument("--repo", required=True, help="HF repo id (user/dataset-name)")
    p.add_argument("--shard_size_gb", type=float, default=1.0, help="target shard size in GB (default: 1.0)")
    p.add_argument("--split", default="train", help="shard prefix (default: train)")
    p.add_argument("--private", action="store_true", help="create private HF repo")
    p.add_argument("--workers", type=int, default=min(os.cpu_count() * 2, 64), help="upload threads")
    p.add_argument("--skip-build", action="store_true", help="skip TAR creation, upload existing TARs only")
    p.add_argument("--tar-dir", default=None, help="custom output dir for TARs (default: <root>_tars)")
    args = p.parse_args()

    if not re.match(r"^[A-Za-z0-9_-]+$", args.split):
        sys.exit(f"--split must match [A-Za-z0-9_-]+, got: {args.split!r}")

    root = Path(args.root).resolve()
    tar_dir = Path(args.tar_dir).resolve() if args.tar_dir else root.parent / f"{root.name}_tars"

    if not args.skip_build:
        if not root.exists():
            sys.exit(f"root not found: {root}")
        print(f"Building TARs from {root} → {tar_dir} ...")
        stats = build_tars(root, tar_dir, args.shard_size_gb, args.split)
        print(
            f"  {stats['shards_created']} shards, "
            f"{stats['samples_written']} samples, "
            f"{stats['total_bytes'] / (1024**3):.2f} GB"
        )
        if stats["missing_txt"]:
            print(f"  ⚠ {stats['missing_txt']} videos without .txt caption")
    else:
        if not tar_dir.exists():
            sys.exit(f"tar dir not found: {tar_dir}")
        print(f"Skipping build, using existing TARs at {tar_dir}")

    upload_to_hf(tar_dir, args.repo, args.private, args.workers)


if __name__ == "__main__":
    main()
