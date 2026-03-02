#!/usr/bin/env python3
"""Upload a local folder to Hugging Face with chunked commits."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

DEFAULT_CHUNK_GB = 5.0
DEFAULT_MAX_FILES_PER_CHUNK = 5000


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def default_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(4, min(64, cpu * 2))


def sanitize_repo_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip().lower())
    name = re.sub(r"-{2,}", "-", name).strip("-.")
    if not name:
        return "dataset"
    return name[:96].strip("-.") or "dataset"


def auto_repo_id(api: Any, token: str, root: Path, seed_path: Optional[Path]) -> str:
    who = api.whoami(token=token)
    namespace = who.get("name")
    if not namespace:
        raise RuntimeError("could not resolve Hugging Face username from token")
    seed = seed_path.stem if seed_path else root.name
    repo_name = sanitize_repo_name(f"{seed}-{root.name}")
    return f"{namespace}/{repo_name}"


def iter_all_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.is_file():
                yield p


def collect_entries(root: Path, include_file: Optional[Path]) -> List[Tuple[Path, str, int]]:
    entries: List[Tuple[Path, str, int]] = []
    root_resolved = root.resolve()

    for p in iter_all_files(root):
        entries.append((p, p.relative_to(root).as_posix(), p.stat().st_size))

    if include_file and include_file.exists():
        include_resolved = include_file.resolve()
        try:
            rel = include_resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            rel = f"_reports/{include_file.name}"
        entries.append((include_file, rel, include_file.stat().st_size))

    entries.sort(key=lambda item: item[1])
    return entries


def chunk_entries(
    entries: List[Tuple[Path, str, int]],
    max_chunk_bytes: int,
    max_files_per_chunk: int,
) -> List[List[Tuple[Path, str, int]]]:
    chunks: List[List[Tuple[Path, str, int]]] = []
    current: List[Tuple[Path, str, int]] = []
    current_bytes = 0

    for entry in entries:
        _, _, size = entry
        if size > max_chunk_bytes:
            if current:
                chunks.append(current)
                current = []
                current_bytes = 0
            chunks.append([entry])
            continue

        should_split = (
            current
            and (
                current_bytes + size > max_chunk_bytes
                or len(current) >= max_files_per_chunk
            )
        )
        if should_split:
            chunks.append(current)
            current = []
            current_bytes = 0

        current.append(entry)
        current_bytes += size

    if current:
        chunks.append(current)

    return chunks


def upload_folder(
    root: Path,
    token: str,
    repo_type: str,
    private: bool,
    repo_id: Optional[str],
    strategy: str,
    chunk_gb: float,
    max_files_per_chunk: int,
    workers: int,
    include_file: Optional[Path],
    no_xet_high_performance: bool,
) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    if not no_xet_high_performance and not os.getenv("HF_XET_HIGH_PERFORMANCE"):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        logging.info("HF_XET_HIGH_PERFORMANCE=1 enabled")

    if importlib.util.find_spec("hf_xet") is None:
        logging.warning("hf_xet not installed; upload may be slower")
    else:
        logging.info("hf_xet detected")

    api = HfApi(token=token)
    resolved_repo_id = repo_id or auto_repo_id(api=api, token=token, root=root, seed_path=include_file)

    create_repo(
        repo_id=resolved_repo_id,
        token=token,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    logging.info("repo ready: %s (type=%s)", resolved_repo_id, repo_type)

    if strategy == "large-folder":
        logging.info("upload strategy=large-folder")
        api.upload_large_folder(
            repo_id=resolved_repo_id,
            folder_path=str(root),
            repo_type=repo_type,
            num_workers=max(4, min(64, workers)),
        )
        if include_file and include_file.exists():
            from huggingface_hub import upload_file

            try:
                path_in_repo = include_file.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                path_in_repo = f"_reports/{include_file.name}"
            upload_file(
                path_or_fileobj=str(include_file),
                path_in_repo=path_in_repo,
                repo_id=resolved_repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=f"add {include_file.name}",
            )
        return resolved_repo_id

    if chunk_gb <= 0:
        raise ValueError("--chunk_gb must be > 0")
    if max_files_per_chunk <= 0:
        raise ValueError("--max_files_per_chunk must be > 0")

    max_chunk_bytes = int(chunk_gb * (1024**3))
    entries = collect_entries(root=root, include_file=include_file)
    if not entries:
        raise RuntimeError(f"no files found under {root}")

    total_bytes = sum(size for _, _, size in entries)
    chunks = chunk_entries(
        entries=entries,
        max_chunk_bytes=max_chunk_bytes,
        max_files_per_chunk=max_files_per_chunk,
    )
    logging.info(
        "upload strategy=chunked files=%d total=%.2fGB chunks=%d chunk_target=%.2fGB max_files_chunk=%d",
        len(entries),
        total_bytes / (1024**3),
        len(chunks),
        chunk_gb,
        max_files_per_chunk,
    )

    thread_count = max(4, min(64, workers))
    for idx, chunk in enumerate(chunks, start=1):
        chunk_bytes = sum(size for _, _, size in chunk)
        operations = [
            CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(local_path))
            for local_path, rel_path, _ in chunk
        ]
        commit_message = (
            f"chunk {idx}/{len(chunks)} - files={len(chunk)} "
            f"size={chunk_bytes / (1024**3):.2f}GB"
        )
        api.create_commit(
            repo_id=resolved_repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message,
            token=token,
            num_threads=thread_count,
        )
        logging.info(
            "uploaded chunk %d/%d files=%d size=%.2fGB",
            idx,
            len(chunks),
            len(chunk),
            chunk_bytes / (1024**3),
        )

    return resolved_repo_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="local dataset folder to upload")
    parser.add_argument("--hf_repo_id", default=None, help="target repo id (e.g. user/my-dataset)")
    parser.add_argument("--hf_repo_type", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--hf_private", action="store_true", help="create private repo")
    parser.add_argument("--hf_token", default=None, help="HF token (recommended in this flag)")
    parser.add_argument("--hf_token_env", default="HF_TOKEN", help="fallback env var for HF token")
    parser.add_argument(
        "--strategy",
        choices=["chunked", "large-folder"],
        default="chunked",
        help="chunked=5GB commits (default), large-folder=upload_large_folder",
    )
    parser.add_argument("--chunk_gb", type=float, default=DEFAULT_CHUNK_GB, help="chunk size for strategy=chunked")
    parser.add_argument("--max_files_per_chunk", type=int, default=DEFAULT_MAX_FILES_PER_CHUNK)
    parser.add_argument("--workers", type=int, default=default_workers(), help="threads for hashing/upload")
    parser.add_argument("--include_json", default=None, help="optional JSON/state file to include in upload")
    parser.add_argument(
        "--no_xet_high_performance",
        action="store_true",
        help="do not set HF_XET_HIGH_PERFORMANCE=1 automatically",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"invalid --root: {root}")

    include_json = Path(args.include_json).resolve() if args.include_json else None
    if include_json and not include_json.exists():
        raise SystemExit(f"--include_json not found: {include_json}")

    token = (args.hf_token or os.getenv(args.hf_token_env, "")).strip()
    if not token:
        raise SystemExit(
            f"missing token: pass --hf_token or set {args.hf_token_env}"
        )

    repo_id = upload_folder(
        root=root,
        token=token,
        repo_type=args.hf_repo_type,
        private=args.hf_private,
        repo_id=args.hf_repo_id,
        strategy=args.strategy,
        chunk_gb=args.chunk_gb,
        max_files_per_chunk=args.max_files_per_chunk,
        workers=max(1, args.workers),
        include_file=include_json,
        no_xet_high_performance=args.no_xet_high_performance,
    )
    logging.info("upload completed: %s", repo_id)


if __name__ == "__main__":
    main()
