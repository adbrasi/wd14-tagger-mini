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
        [--hf_token TOKEN]
        [--hf_token_env HF_TOKEN]

Keys inside TARs are globally unique across shards (no overwrite on extract).
"""

import argparse
import os
import re
import sys
from pathlib import Path

from build_webdataset_tars import build_tars


# ── HuggingFace upload ───────────────────────────────────────────────────────


def upload_to_hf(
    folder: Path, repo_id: str, private: bool, workers: int, token: str
):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
    )

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
    p.add_argument("--workers", type=int, default=min((os.cpu_count() or 4) * 2, 64), help="upload threads")
    p.add_argument("--skip-build", action="store_true", help="skip TAR creation, upload existing TARs only")
    p.add_argument("--tar-dir", default=None, help="custom output dir for TARs (default: <root>_tars)")
    p.add_argument("--hf_token", default=None, help="HF token (recommended)")
    p.add_argument("--hf_token_env", default="HF_TOKEN", help="fallback env var for HF token")
    args = p.parse_args()

    # ── Validate token BEFORE any expensive work ──
    token = (args.hf_token or os.getenv(args.hf_token_env, "")).strip()
    if not token:
        sys.exit(
            f"missing token: pass --hf_token or set {args.hf_token_env}"
        )

    if args.shard_size_gb <= 0:
        sys.exit("--shard_size_gb must be positive")

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
            print(f"  Warning: {stats['missing_txt']} videos without .txt caption")
    else:
        if not tar_dir.exists():
            sys.exit(f"tar dir not found: {tar_dir}")
        print(f"Skipping build, using existing TARs at {tar_dir}")

    upload_to_hf(tar_dir, args.repo, args.private, args.workers, token)


if __name__ == "__main__":
    main()
