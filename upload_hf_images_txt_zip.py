#!/usr/bin/env python3
"""Upload only image + .txt files to Hugging Face as zip chunks."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import random
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".jxl"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
TEXT_EXTS = {".txt"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS | TEXT_EXTS

DEFAULT_CHUNK_GB = 5.0
DEFAULT_MAX_FILES_PER_CHUNK = 5000
DEFAULT_EXCLUDED_DIRS = {".cache", ".git", ".hg", ".svn", "__pycache__"}
DEFAULT_HF_TIMEOUT_SECONDS = 1200
DEFAULT_COMMIT_RETRIES = 12
DEFAULT_RETRY_BASE_SECONDS = 5.0
DEFAULT_ZIP_PREFIX = "chunks"
DEFAULT_TEMP_SUBDIR = ".hf_upload_tmp_images_txt"
DEFAULT_BOOTSTRAP_FILES = {"README.md", ".gitattributes", ".huggingface.yaml"}


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
        name = "dataset"
    return name[:96].strip("-.") or "dataset"


def auto_repo_id(api: Any, token: str, root: Path) -> str:
    who = api.whoami(token=token)
    namespace = who.get("name")
    if not namespace:
        raise RuntimeError("could not resolve Hugging Face username from token")
    return f"{namespace}/{sanitize_repo_name(root.name + '-images-txt')}"


def iter_files(root: Path, excluded_dirs: Set[str], include_hidden: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
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


def collect_allowed_entries(
    root: Path,
    excluded_dirs: Set[str],
    include_hidden: bool,
) -> Tuple[List[Tuple[Path, str, int]], Dict[str, int]]:
    entries: List[Tuple[Path, str, int]] = []
    stats = {
        "total_seen": 0,
        "images": 0,
        "txt": 0,
        "zip_skipped": 0,
        "other_skipped": 0,
    }

    for p in iter_files(root=root, excluded_dirs=excluded_dirs, include_hidden=include_hidden):
        stats["total_seen"] += 1
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS or suffix in VIDEO_EXTS:
            stats["images"] += 1
            entries.append((p, p.relative_to(root).as_posix(), p.stat().st_size))
        elif suffix in TEXT_EXTS:
            stats["txt"] += 1
            entries.append((p, p.relative_to(root).as_posix(), p.stat().st_size))
        elif suffix == ".zip":
            stats["zip_skipped"] += 1
        else:
            stats["other_skipped"] += 1

    entries.sort(key=lambda item: item[1])
    return entries, stats


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


def detect_uploaded_chunks(
    api: Any,
    repo_id: str,
    repo_type: str,
    zip_prefix: str,
    token: str,
) -> Set[int]:
    clean_prefix = zip_prefix.strip().strip("/")
    if not clean_prefix:
        clean_prefix = DEFAULT_ZIP_PREFIX
    pattern = re.compile(rf"^{re.escape(clean_prefix)}/chunk_(\d+)\.zip$")
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
    out: Set[int] = set()
    for path in files:
        m = pattern.match(path)
        if m:
            out.add(int(m.group(1)))
    return out


def _is_safe_existing_repo_file(path: str, zip_prefix: str) -> bool:
    if path in DEFAULT_BOOTSTRAP_FILES:
        return True
    clean_prefix = zip_prefix.strip().strip("/")
    if not clean_prefix:
        clean_prefix = DEFAULT_ZIP_PREFIX
    return bool(re.match(rf"^{re.escape(clean_prefix)}/chunk_\d+\.zip$", path))


def upload_images_txt_as_zips(
    root: Path,
    token: str,
    repo_type: str,
    private: bool,
    repo_id: Optional[str],
    chunk_gb: float,
    max_files_per_chunk: int,
    workers: int,
    excluded_dirs: Set[str],
    include_hidden: bool,
    zip_compression: str,
    zip_temp_dir: Path,
    zip_path_prefix: str,
    start_chunk: int,
    end_chunk: int,
    resume_auto: bool,
    allow_existing_repo: bool,
    hf_timeout_seconds: int,
    commit_retries: int,
    retry_base_seconds: float,
    sleep_between_chunks: float,
    no_xet_high_performance: bool,
) -> str:
    if chunk_gb < 5.0:
        raise ValueError("--chunk_gb must be >= 5.0")
    if max_files_per_chunk <= 0:
        raise ValueError("--max_files_per_chunk must be > 0")
    if start_chunk < 1:
        raise ValueError("--start_chunk must be >= 1")

    if hf_timeout_seconds > 0:
        timeout_s = str(int(hf_timeout_seconds))
        if not os.getenv("HF_HUB_DOWNLOAD_TIMEOUT"):
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout_s
        if not os.getenv("HF_HUB_ETAG_TIMEOUT"):
            os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout_s
        logging.info("HF_HUB_*_TIMEOUT set to %ss", timeout_s)

    if not no_xet_high_performance and not os.getenv("HF_XET_HIGH_PERFORMANCE"):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        logging.info("HF_XET_HIGH_PERFORMANCE=1 enabled")
    if importlib.util.find_spec("hf_xet") is None:
        logging.warning("hf_xet not installed; upload may be slower")
    else:
        logging.info("hf_xet detected")

    import httpx
    from huggingface_hub import CommitOperationAdd, HfApi, create_repo
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi(token=token)
    resolved_repo_id = repo_id or auto_repo_id(api=api, token=token, root=root)
    clean_prefix = zip_path_prefix.strip().strip("/")
    if not clean_prefix:
        clean_prefix = DEFAULT_ZIP_PREFIX

    create_repo(
        repo_id=resolved_repo_id,
        token=token,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    logging.info("repo ready: %s (type=%s)", resolved_repo_id, repo_type)

    existing_files = api.list_repo_files(repo_id=resolved_repo_id, repo_type=repo_type, token=token)
    unsafe_existing = [p for p in existing_files if not _is_safe_existing_repo_file(p, clean_prefix)]
    if unsafe_existing and not allow_existing_repo:
        raise RuntimeError(
            f"repo '{resolved_repo_id}' has existing content that is not safe to overwrite "
            f"({len(unsafe_existing)} files, e.g. {unsafe_existing[:3]}). "
            "Use a new repo_id or pass --allow_existing_repo."
        )
    if existing_files and not unsafe_existing:
        logging.info(
            "repo has only bootstrap/zip files (%d entries); continuing safely.",
            len(existing_files),
        )

    entries, stats = collect_allowed_entries(
        root=root,
        excluded_dirs=excluded_dirs,
        include_hidden=include_hidden,
    )
    if not entries:
        raise RuntimeError("no image/txt files found to upload")

    total_bytes = sum(size for _, _, size in entries)
    max_chunk_bytes = int(chunk_gb * (1024**3))
    chunks = chunk_entries(
        entries=entries,
        max_chunk_bytes=max_chunk_bytes,
        max_files_per_chunk=max_files_per_chunk,
    )

    logging.info(
        "selection: images=%d txt=%d zip_skipped=%d other_skipped=%d selected_files=%d selected_size=%.2fGB chunks=%d",
        stats["images"],
        stats["txt"],
        stats["zip_skipped"],
        stats["other_skipped"],
        len(entries),
        total_bytes / (1024**3),
        len(chunks),
    )

    if end_chunk and end_chunk < start_chunk:
        raise ValueError("--end_chunk must be 0 or >= --start_chunk")
    if start_chunk > len(chunks):
        raise ValueError(f"--start_chunk={start_chunk} > total chunks={len(chunks)}")
    if end_chunk and end_chunk > len(chunks):
        end_chunk = len(chunks)

    if resume_auto:
        uploaded = detect_uploaded_chunks(
            api=api,
            repo_id=resolved_repo_id,
            repo_type=repo_type,
            zip_prefix=clean_prefix,
            token=token,
        )
        if uploaded:
            auto_start = max(uploaded) + 1
            if auto_start > start_chunk:
                start_chunk = auto_start
        logging.info("resume_auto: uploaded_chunks=%d start_chunk=%d", len(uploaded), start_chunk)

    logging.info(
        "upload window: start=%d end=%s total=%d prefix=%s",
        start_chunk,
        str(end_chunk) if end_chunk else "last",
        len(chunks),
        clean_prefix,
    )

    zip_compression_mode = zipfile.ZIP_STORED if zip_compression == "store" else zipfile.ZIP_DEFLATED
    thread_count = max(4, min(64, workers))
    zip_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, chunk in enumerate(chunks, start=1):
            if idx < start_chunk:
                continue
            if end_chunk and idx > end_chunk:
                break

            src_bytes = sum(size for _, _, size in chunk)
            zip_name = f"chunk_{idx:05d}.zip"
            zip_local_path = zip_temp_dir / zip_name
            zip_repo_path = f"{clean_prefix}/{zip_name}"

            zip_size = build_chunk_zip(
                chunk=chunk,
                zip_path=zip_local_path,
                compression=zip_compression_mode,
            )
            logging.info(
                "built zip %d/%d: %s size=%.2fGB (src=%.2fGB files=%d)",
                idx,
                len(chunks),
                zip_local_path,
                zip_size / (1024**3),
                src_bytes / (1024**3),
                len(chunk),
            )

            operations = [CommitOperationAdd(path_in_repo=zip_repo_path, path_or_fileobj=str(zip_local_path))]
            commit_message = (
                f"zip chunk {idx}/{len(chunks)} "
                f"src_files={len(chunk)} src_size={src_bytes / (1024**3):.2f}GB "
                f"zip_size={zip_size / (1024**3):.2f}GB"
            )

            try:
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
                                408, 409, 423, 425, 429, 500, 502, 503, 504, 520, 522, 524
                            }
                            low_msg = str(exc).lower()
                            if status_code == 400 and (
                                "no files have been modified" in low_msg
                                or "no changes" in low_msg
                                or "nothing to commit" in low_msg
                            ):
                                logging.warning(
                                    "chunk %d/%d appears already committed (no changes), continuing.",
                                    idx,
                                    len(chunks),
                                )
                                break

                        if not retryable or attempt >= max(1, commit_retries):
                            raise

                        retry_after_s = None
                        if isinstance(exc, HfHubHTTPError) and getattr(exc, "response", None) is not None:
                            retry_after = exc.response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    retry_after_s = float(retry_after)
                                except ValueError:
                                    retry_after_s = None

                        sleep_s = (
                            retry_after_s
                            if retry_after_s is not None
                            else min(180.0, retry_base_seconds * (2 ** (attempt - 1)))
                        )
                        sleep_s += random.uniform(0.0, 1.0)
                        logging.warning(
                            "chunk %d/%d commit failed (attempt %d/%d status=%s err=%s). retry in %.1fs",
                            idx,
                            len(chunks),
                            attempt,
                            max(1, commit_retries),
                            status_code,
                            type(exc).__name__,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
            finally:
                if zip_local_path.exists():
                    try:
                        zip_local_path.unlink()
                    except OSError:
                        logging.warning("failed to remove temp zip: %s", zip_local_path)

            logging.info(
                "uploaded chunk %d/%d src_files=%d src_size=%.2fGB",
                idx,
                len(chunks),
                len(chunk),
                src_bytes / (1024**3),
            )
            if sleep_between_chunks > 0:
                time.sleep(sleep_between_chunks)
    except KeyboardInterrupt:
        logging.warning("interrupted by user. Resume with --start_chunk and same parameters.")
        raise SystemExit(130)

    return resolved_repo_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root path")
    parser.add_argument("--hf_repo_id", default=None, help="target repo id (e.g. user/my-dataset)")
    parser.add_argument("--hf_repo_type", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument("--hf_token", default=None, help="HF token (recommended)")
    parser.add_argument("--hf_token_env", default="HF_TOKEN", help="fallback token env var")
    parser.add_argument("--chunk_gb", type=float, default=DEFAULT_CHUNK_GB, help="zip chunk target size in GB (>=5)")
    parser.add_argument("--max_files_per_chunk", type=int, default=DEFAULT_MAX_FILES_PER_CHUNK)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--zip_compression",
        choices=["store", "deflate"],
        default="store",
        help="store is faster, deflate is smaller/slower",
    )
    parser.add_argument("--zip_path_prefix", default=DEFAULT_ZIP_PREFIX, help="repo folder for zip chunks")
    parser.add_argument("--zip_temp_dir", default=None, help="local temp dir for chunk zip files")
    parser.add_argument("--start_chunk", type=int, default=1)
    parser.add_argument("--end_chunk", type=int, default=0, help="0 means until end")
    parser.add_argument("--resume_auto", action="store_true", help="auto-detect highest uploaded chunk and continue")
    parser.add_argument("--allow_existing_repo", action="store_true", help="allow upload to non-empty repo")
    parser.add_argument("--sleep_between_chunks", type=float, default=0.0)
    parser.add_argument("--hf_timeout_seconds", type=int, default=DEFAULT_HF_TIMEOUT_SECONDS)
    parser.add_argument("--commit_retries", type=int, default=DEFAULT_COMMIT_RETRIES)
    parser.add_argument("--retry_base_seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS)
    parser.add_argument(
        "--exclude_dirs",
        default=",".join(sorted(DEFAULT_EXCLUDED_DIRS)),
        help="comma-separated dir names to ignore recursively",
    )
    parser.add_argument("--include_hidden", action="store_true", help="include hidden files/dirs")
    parser.add_argument("--no_xet_high_performance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"invalid --root: {root}")

    token = (args.hf_token or os.getenv(args.hf_token_env, "")).strip()
    if not token:
        raise SystemExit(f"missing token: pass --hf_token or set {args.hf_token_env}")

    excluded_dirs = {d.strip() for d in str(args.exclude_dirs).split(",") if d.strip()}
    zip_temp_dir = (
        Path(args.zip_temp_dir).resolve()
        if args.zip_temp_dir
        else (root / DEFAULT_TEMP_SUBDIR).resolve()
    )
    excluded_dirs.add(zip_temp_dir.name)

    logging.info(
        "filters: include_hidden=%s excluded_dirs=%s",
        args.include_hidden,
        ",".join(sorted(excluded_dirs)) or "(none)",
    )

    repo_id = upload_images_txt_as_zips(
        root=root,
        token=token,
        repo_type=args.hf_repo_type,
        private=args.hf_private,
        repo_id=args.hf_repo_id,
        chunk_gb=float(args.chunk_gb),
        max_files_per_chunk=int(args.max_files_per_chunk),
        workers=max(1, int(args.workers)),
        excluded_dirs=excluded_dirs,
        include_hidden=bool(args.include_hidden),
        zip_compression=args.zip_compression,
        zip_temp_dir=zip_temp_dir,
        zip_path_prefix=args.zip_path_prefix,
        start_chunk=max(1, int(args.start_chunk)),
        end_chunk=max(0, int(args.end_chunk)),
        resume_auto=bool(args.resume_auto),
        allow_existing_repo=bool(args.allow_existing_repo),
        hf_timeout_seconds=max(0, int(args.hf_timeout_seconds)),
        commit_retries=max(1, int(args.commit_retries)),
        retry_base_seconds=max(0.1, float(args.retry_base_seconds)),
        sleep_between_chunks=max(0.0, float(args.sleep_between_chunks)),
        no_xet_high_performance=bool(args.no_xet_high_performance),
    )
    logging.info("upload completed: %s", repo_id)


if __name__ == "__main__":
    main()
