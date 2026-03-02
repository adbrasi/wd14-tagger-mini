#!/usr/bin/env python3
"""Upload a local folder to Hugging Face with chunked commits."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

DEFAULT_CHUNK_GB = 5.0
DEFAULT_MAX_FILES_PER_CHUNK = 5000
DEFAULT_EXCLUDED_DIRS = {".cache", ".git", ".hg", ".svn", "__pycache__"}
DEFAULT_HF_TIMEOUT_SECONDS = 600
DEFAULT_COMMIT_RETRIES = 8
DEFAULT_RETRY_BASE_SECONDS = 4.0
DEFAULT_ZIP_PATH_PREFIX = "chunks"


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


def iter_all_files(
    root: Path,
    excluded_dirs: set[str],
    include_hidden: bool,
) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories early for speed and to avoid invalid repo paths.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in excluded_dirs and (include_hidden or not d.startswith("."))
        ]
        base = Path(dirpath)
        for name in filenames:
            if not include_hidden and name.startswith("."):
                continue
            p = base / name
            if p.is_file():
                yield p


def collect_entries(
    root: Path,
    include_file: Optional[Path],
    excluded_dirs: set[str],
    include_hidden: bool,
) -> List[Tuple[Path, str, int]]:
    entries: List[Tuple[Path, str, int]] = []
    root_resolved = root.resolve()

    for p in iter_all_files(
        root=root,
        excluded_dirs=excluded_dirs,
        include_hidden=include_hidden,
    ):
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


def build_chunk_zip(
    chunk: List[Tuple[Path, str, int]],
    zip_path: Path,
    compression: int,
) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=compression,
        allowZip64=True,
    ) as zf:
        for local_path, rel_path, _ in chunk:
            zf.write(local_path, arcname=rel_path)
    return zip_path.stat().st_size


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
    excluded_dirs: set[str],
    include_hidden: bool,
    hf_timeout_seconds: int,
    commit_retries: int,
    retry_base_seconds: float,
    chunk_payload: str,
    zip_compression: str,
    zip_temp_dir: Path,
    zip_path_prefix: str,
) -> str:
    if hf_timeout_seconds > 0:
        timeout_s = str(int(hf_timeout_seconds))
        if not os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"):
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout_s
        if not os.getenv("HF_HUB_ETAG_TIMEOUT"):
            os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout_s
        logging.info(
            "HF_HUB_*_TIMEOUT set to %ss (download/etag)",
            timeout_s,
        )

    import httpx
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo
    from huggingface_hub.errors import HfHubHTTPError

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
    if chunk_payload not in {"zip", "files"}:
        raise ValueError("--chunk_payload must be 'zip' or 'files'")

    max_chunk_bytes = int(chunk_gb * (1024**3))
    entries = collect_entries(
        root=root,
        include_file=include_file,
        excluded_dirs=excluded_dirs,
        include_hidden=include_hidden,
    )
    if not entries:
        raise RuntimeError(f"no files found under {root}")

    total_bytes = sum(size for _, _, size in entries)
    chunks = chunk_entries(
        entries=entries,
        max_chunk_bytes=max_chunk_bytes,
        max_files_per_chunk=max_files_per_chunk,
    )
    logging.info(
        "upload strategy=chunked payload=%s files=%d total=%.2fGB chunks=%d chunk_target=%.2fGB max_files_chunk=%d",
        chunk_payload,
        len(entries),
        total_bytes / (1024**3),
        len(chunks),
        chunk_gb,
        max_files_per_chunk,
    )

    zip_compression_mode = (
        zipfile.ZIP_DEFLATED if zip_compression == "deflate" else zipfile.ZIP_STORED
    )
    clean_prefix = zip_path_prefix.strip().strip("/")
    if not clean_prefix:
        clean_prefix = DEFAULT_ZIP_PATH_PREFIX

    thread_count = max(4, min(64, workers))
    for idx, chunk in enumerate(chunks, start=1):
        chunk_bytes = sum(size for _, _, size in chunk)
        operations: List[CommitOperationAdd] = []
        commit_message = ""
        temp_zip_path: Optional[Path] = None

        if chunk_payload == "zip":
            zip_name = f"chunk_{idx:05d}.zip"
            temp_zip_path = zip_temp_dir / zip_name
            zip_size = build_chunk_zip(
                chunk=chunk,
                zip_path=temp_zip_path,
                compression=zip_compression_mode,
            )
            zip_repo_path = f"{clean_prefix}/{zip_name}"
            operations = [
                CommitOperationAdd(
                    path_in_repo=zip_repo_path,
                    path_or_fileobj=str(temp_zip_path),
                )
            ]
            commit_message = (
                f"zip chunk {idx}/{len(chunks)} - src_files={len(chunk)} "
                f"src_size={chunk_bytes / (1024**3):.2f}GB "
                f"zip_size={zip_size / (1024**3):.2f}GB"
            )
            logging.info(
                "built zip for chunk %d/%d: %s size=%.2fGB (src=%.2fGB files=%d)",
                idx,
                len(chunks),
                temp_zip_path,
                zip_size / (1024**3),
                chunk_bytes / (1024**3),
                len(chunk),
            )
        else:
            operations = [
                CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(local_path))
                for local_path, rel_path, _ in chunk
            ]
            commit_message = (
                f"chunk {idx}/{len(chunks)} - files={len(chunk)} "
                f"size={chunk_bytes / (1024**3):.2f}GB"
            )

        for attempt in range(1, max(1, commit_retries) + 1):
            try:
                api.create_commit(
                    repo_id=resolved_repo_id,
                    repo_type=repo_type,
                    operations=operations,
                    commit_message=commit_message,
                    token=token,
                    num_threads=thread_count,
                )
                break
            except Exception as exc:
                status_code = None
                retryable = isinstance(
                    exc,
                    (
                        httpx.ReadTimeout,
                        httpx.ConnectTimeout,
                        httpx.ConnectError,
                        httpx.RemoteProtocolError,
                    ),
                )
                if isinstance(exc, HfHubHTTPError):
                    status_code = getattr(exc.response, "status_code", None)
                    retryable = retryable or status_code in {
                        408,
                        409,
                        423,
                        425,
                        429,
                        500,
                        502,
                        503,
                        504,
                        520,
                        522,
                        524,
                    }
                    low_msg = str(exc).lower()
                    if status_code == 400 and (
                        "no files have been modified" in low_msg
                        or "no changes" in low_msg
                        or "nothing to commit" in low_msg
                    ):
                        logging.warning(
                            "chunk %d/%d appears already committed (no changes). continuing.",
                            idx,
                            len(chunks),
                        )
                        break

                if not retryable or attempt >= max(1, commit_retries):
                    raise

                sleep_s = min(90.0, retry_base_seconds * (2 ** (attempt - 1)))
                logging.warning(
                    "chunk %d/%d commit failed (attempt %d/%d status=%s err=%s). retrying in %.1fs",
                    idx,
                    len(chunks),
                    attempt,
                    max(1, commit_retries),
                    status_code,
                    type(exc).__name__,
                    sleep_s,
                )
                time.sleep(sleep_s)
        logging.info(
            "uploaded chunk %d/%d files=%d size=%.2fGB",
            idx,
            len(chunks),
            len(chunk),
            chunk_bytes / (1024**3),
        )
        if temp_zip_path and temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
            except OSError:
                logging.warning("failed to remove temp zip: %s", temp_zip_path)

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
        help="chunked=chunk commits (default), large-folder=upload_large_folder",
    )
    parser.add_argument("--chunk_gb", type=float, default=DEFAULT_CHUNK_GB, help="chunk size for strategy=chunked")
    parser.add_argument("--max_files_per_chunk", type=int, default=DEFAULT_MAX_FILES_PER_CHUNK)
    parser.add_argument("--workers", type=int, default=default_workers(), help="threads for hashing/upload")
    parser.add_argument("--include_json", default=None, help="optional JSON/state file to include in upload")
    parser.add_argument(
        "--chunk_payload",
        choices=["zip", "files"],
        default="zip",
        help="for strategy=chunked: upload one zip per chunk (default) or individual files",
    )
    parser.add_argument(
        "--zip_compression",
        choices=["store", "deflate"],
        default="store",
        help="zip compression mode when --chunk_payload=zip (store is faster)",
    )
    parser.add_argument(
        "--zip_temp_dir",
        default=None,
        help="temporary directory for local zip creation (default: <root>/.hf_upload_tmp)",
    )
    parser.add_argument(
        "--zip_path_prefix",
        default=DEFAULT_ZIP_PATH_PREFIX,
        help="path prefix in repo for zip chunks (default: chunks)",
    )
    parser.add_argument(
        "--hf_timeout_seconds",
        type=int,
        default=DEFAULT_HF_TIMEOUT_SECONDS,
        help="read timeout for HF HTTP operations (default: 600)",
    )
    parser.add_argument(
        "--commit_retries",
        type=int,
        default=DEFAULT_COMMIT_RETRIES,
        help="retry attempts per chunk commit on transient errors",
    )
    parser.add_argument(
        "--retry_base_seconds",
        type=float,
        default=DEFAULT_RETRY_BASE_SECONDS,
        help="base backoff (seconds), exponential",
    )
    parser.add_argument(
        "--exclude_dirs",
        default=",".join(sorted(DEFAULT_EXCLUDED_DIRS)),
        help="comma-separated directory names to skip recursively (default excludes .cache/.git/etc)",
    )
    parser.add_argument(
        "--include_hidden",
        action="store_true",
        help="include hidden files/dirs (disabled by default)",
    )
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
    zip_temp_dir = (
        Path(args.zip_temp_dir).resolve()
        if args.zip_temp_dir
        else (root / ".hf_upload_tmp").resolve()
    )
    zip_temp_dir.mkdir(parents=True, exist_ok=True)
    excluded_dirs = {
        d.strip()
        for d in str(args.exclude_dirs).split(",")
        if d.strip()
    }
    excluded_dirs.add(zip_temp_dir.name)
    logging.info(
        "scan filters: include_hidden=%s excluded_dirs=%s chunk_payload=%s",
        args.include_hidden,
        ",".join(sorted(excluded_dirs)) or "(none)",
        args.chunk_payload,
    )

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
        excluded_dirs=excluded_dirs,
        include_hidden=args.include_hidden,
        hf_timeout_seconds=max(0, args.hf_timeout_seconds),
        commit_retries=max(1, args.commit_retries),
        retry_base_seconds=max(0.1, args.retry_base_seconds),
        chunk_payload=args.chunk_payload,
        zip_compression=args.zip_compression,
        zip_temp_dir=zip_temp_dir,
        zip_path_prefix=args.zip_path_prefix,
    )
    logging.info("upload completed: %s", repo_id)


if __name__ == "__main__":
    main()
