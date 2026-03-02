#!/usr/bin/env python3
"""Scan dataset images quickly and optionally delete broken files in the same pass."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageFile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".jxl"}
PROGRESS_EVERY = 1000
DEFAULT_HF_CHUNK_GB = 5.0


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def default_workers() -> int:
    cpu = os.cpu_count() or 4
    return max(4, min(32, cpu * 2))


def iter_images(root: Path) -> Iterable[Path]:
    """Fast filesystem walk for image files."""
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            suffix = os.path.splitext(name)[1].lower()
            if suffix in IMAGE_EXTS:
                yield base / name


def check_image(path: Path) -> Optional[str]:
    """Return error string if image is broken, else None."""
    try:
        # Single decode pass is much faster than verify()+reopen()+load().
        # Treat DecompressionBombWarning as error so oversized images are removed.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                img.load()
        return None
    except Exception as exc:
        return str(exc)


def _unlink_if_exists(path: Path, dry_run: bool) -> Tuple[bool, bool, Optional[str]]:
    """Return (removed, missing, error)."""
    try:
        if not path.exists():
            return False, True, None
        if dry_run:
            return True, False, None
        path.unlink()
        return True, False, None
    except Exception as exc:
        return False, False, str(exc)


def process_one_image(
    img_path: Path,
    delete_bad: bool,
    delete_txt: bool,
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    """Validate one image and optionally delete bad files immediately."""
    err = check_image(img_path)
    if not err:
        return None

    record: Dict[str, Any] = {"path": str(img_path), "error": err}
    if not delete_bad:
        return record

    removed_img, missing_img, remove_img_err = _unlink_if_exists(img_path, dry_run)
    record["removed_image"] = removed_img
    record["image_missing"] = missing_img
    if remove_img_err:
        record["remove_image_error"] = remove_img_err

    if delete_txt:
        txt_path = img_path.with_suffix(".txt")
        removed_txt, missing_txt, remove_txt_err = _unlink_if_exists(txt_path, dry_run)
        record["removed_txt"] = removed_txt
        record["txt_missing"] = missing_txt
        if remove_txt_err:
            record["remove_txt_error"] = remove_txt_err

    return record


def _log_progress(checked: int, total: int, bad_count: int, start_time: float) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = checked / elapsed
    logging.info(
        "progress: checked=%d/%d bad=%d rate=%.1f img/s",
        checked,
        total,
        bad_count,
        rate,
    )


def scan_and_cleanup(
    root: Path,
    workers: int,
    delete_bad: bool,
    delete_txt: bool,
    dry_run: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    paths = list(iter_images(root))
    total = len(paths)
    logging.info("scan start: root=%s images=%d workers=%d", root, total, workers)

    bad: List[Dict[str, Any]] = []
    start_time = time.monotonic()
    checked = 0
    removed_images = 0
    removed_txt = 0
    missing = 0
    failed = 0

    def _worker(path: Path) -> Optional[Dict[str, Any]]:
        return process_one_image(
            path,
            delete_bad=delete_bad,
            delete_txt=delete_txt,
            dry_run=dry_run,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_worker, paths):
            checked += 1
            if result:
                bad.append(result)
                logging.warning("bad image: %s (%s)", result["path"], result["error"])

                if delete_bad:
                    if result.get("removed_image"):
                        removed_images += 1
                    if result.get("removed_txt"):
                        removed_txt += 1
                    if result.get("image_missing"):
                        missing += 1
                    if result.get("remove_image_error") or result.get("remove_txt_error"):
                        failed += 1

            if checked % PROGRESS_EVERY == 0 or checked == total:
                _log_progress(checked, total, len(bad), start_time)

    stats = {
        "checked": checked,
        "bad": len(bad),
        "removed_images": removed_images,
        "removed_txt": removed_txt,
        "missing": missing,
        "failed": failed,
    }
    return bad, stats


def save_report(report_path: Path, bad: List[Dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("report saved: %s (%d records)", report_path, len(bad))


def delete_from_report(
    report_path: Path,
    delete_txt: bool,
    dry_run: bool,
) -> Dict[str, int]:
    bad = json.loads(report_path.read_text(encoding="utf-8"))
    removed_images = 0
    removed_txt = 0
    missing = 0
    failed = 0

    for rec in bad:
        img = Path(rec["path"])
        r_img, m_img, e_img = _unlink_if_exists(img, dry_run)
        removed_images += int(r_img)
        missing += int(m_img)
        failed += int(e_img is not None)

        if delete_txt:
            txt = img.with_suffix(".txt")
            r_txt, _, e_txt = _unlink_if_exists(txt, dry_run)
            removed_txt += int(r_txt)
            failed += int(e_txt is not None)

    return {
        "checked": len(bad),
        "bad": len(bad),
        "removed_images": removed_images,
        "removed_txt": removed_txt,
        "missing": missing,
        "failed": failed,
    }


def _sanitize_repo_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip().lower())
    name = re.sub(r"-{2,}", "-", name).strip("-.")
    if not name:
        name = "dataset"
    name = name[:96].strip("-.")
    return name or "dataset"


def _auto_repo_id(api: Any, token: str, root: Path, report_path: Path) -> str:
    who = api.whoami(token=token)
    namespace = who.get("name")
    if not namespace:
        raise RuntimeError("could not resolve Hugging Face username from token")
    repo_name = _sanitize_repo_name(f"{report_path.stem}-{root.name}")
    return f"{namespace}/{repo_name}"


def _iter_all_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.is_file():
                yield p


def _collect_upload_entries(root: Path, report_path: Optional[Path]) -> List[Tuple[Path, str, int]]:
    entries: List[Tuple[Path, str, int]] = []
    root_resolved = root.resolve()

    for p in _iter_all_files(root):
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        entries.append((p, rel, size))

    if report_path and report_path.exists():
        report_resolved = report_path.resolve()
        try:
            rel = report_resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            rel = f"_reports/{report_path.name}"
        entries.append((report_path, rel, report_path.stat().st_size))

    entries.sort(key=lambda item: item[1])
    return entries


def _chunk_entries(entries: List[Tuple[Path, str, int]], max_chunk_bytes: int) -> List[List[Tuple[Path, str, int]]]:
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
        if current and current_bytes + size > max_chunk_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += size

    if current:
        chunks.append(current)
    return chunks


def upload_to_hf(
    root: Path,
    report_path: Optional[Path],
    token: str,
    repo_type: str,
    private: bool,
    repo_id: Optional[str],
    strategy: str,
    chunk_gb: float,
    workers: int,
    no_xet_high_performance: bool,
) -> str:
    import importlib.util

    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    if not no_xet_high_performance and not os.getenv("HF_XET_HIGH_PERFORMANCE"):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        logging.info("HF_XET_HIGH_PERFORMANCE=1 enabled for faster upload")
    if importlib.util.find_spec("hf_xet") is None:
        logging.warning("hf_xet is not installed; upload will work but may be slower")
    else:
        logging.info("hf_xet detected (xet backend available)")

    api = HfApi(token=token)
    resolved_repo_id = repo_id or _auto_repo_id(api=api, token=token, root=root, report_path=report_path or root / "state.json")
    create_repo(
        repo_id=resolved_repo_id,
        token=token,
        private=private,
        repo_type=repo_type,
        exist_ok=True,
    )
    logging.info("hf repo ready: %s (type=%s)", resolved_repo_id, repo_type)

    if strategy == "large-folder":
        logging.info("starting upload strategy=large-folder")
        api.upload_large_folder(
            repo_id=resolved_repo_id,
            folder_path=str(root),
            repo_type=repo_type,
            num_workers=max(4, min(64, workers)),
        )
        if report_path and report_path.exists():
            from huggingface_hub import upload_file

            path_in_repo = report_path.relative_to(root).as_posix() if report_path.resolve().is_relative_to(root.resolve()) else f"_reports/{report_path.name}"
            upload_file(
                path_or_fileobj=str(report_path),
                path_in_repo=path_in_repo,
                repo_id=resolved_repo_id,
                repo_type=repo_type,
                token=token,
                commit_message="Add cleanup report",
            )
        return resolved_repo_id

    if chunk_gb <= 0:
        raise ValueError("--hf_chunk_gb must be > 0")
    max_chunk_bytes = int(chunk_gb * (1024**3))

    entries = _collect_upload_entries(root=root, report_path=report_path)
    if not entries:
        raise RuntimeError(f"no files found to upload under {root}")

    total_bytes = sum(size for _, _, size in entries)
    chunks = _chunk_entries(entries, max_chunk_bytes=max_chunk_bytes)
    logging.info(
        "starting upload strategy=chunked files=%d total=%.2fGB chunks=%d chunk_target=%.2fGB",
        len(entries),
        total_bytes / (1024**3),
        len(chunks),
        chunk_gb,
    )

    thread_count = max(4, min(64, workers))
    for idx, chunk in enumerate(chunks, start=1):
        chunk_bytes = sum(size for _, _, size in chunk)
        operations = [
            CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(local_path))
            for local_path, rel_path, _ in chunk
        ]
        commit_message = f"chunk {idx}/{len(chunks)} - {len(chunk)} files - {chunk_bytes / (1024**3):.2f}GB"
        api.create_commit(
            repo_id=resolved_repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message,
            token=token,
            num_threads=thread_count,
        )
        logging.info(
            "uploaded chunk %d/%d (files=%d size=%.2fGB)",
            idx,
            len(chunks),
            len(chunk),
            chunk_bytes / (1024**3),
        )

    return resolved_repo_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset root path")
    parser.add_argument(
        "--report",
        default=None,
        help="output JSON report path (default: <root>/../bad_images.json)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete broken images immediately while scanning (single pass)",
    )
    parser.add_argument("--keep_txt", action="store_true", help="do not remove sidecar .txt files")
    parser.add_argument("--dry_run", action="store_true", help="show actions without deleting files")
    parser.add_argument("--from_report", default=None, help="skip scan and delete from an existing JSON report")
    parser.add_argument("--workers", type=int, default=default_workers(), help="worker threads for validation")
    parser.add_argument("--hf_upload", action="store_true", help="upload folder to Hugging Face after cleanup")
    parser.add_argument("--hf_repo_id", default=None, help="target repo id, e.g. user/my-dataset (auto if omitted)")
    parser.add_argument("--hf_repo_type", choices=["dataset", "model"], default="dataset")
    parser.add_argument("--hf_private", action="store_true", help="create private repo on Hugging Face")
    parser.add_argument(
        "--hf_strategy",
        choices=["chunked", "large-folder"],
        default="chunked",
        help="upload strategy; chunked supports chunk_gb commits",
    )
    parser.add_argument("--hf_chunk_gb", type=float, default=DEFAULT_HF_CHUNK_GB, help="chunk size in GB for strategy=chunked")
    parser.add_argument("--hf_token_env", default="HF_TOKEN", help="environment variable containing HF token")
    parser.add_argument(
        "--hf_no_xet_high_performance",
        action="store_true",
        help="disable automatic HF_XET_HIGH_PERFORMANCE=1",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    report_path = Path(args.report).resolve() if args.report else (root.parent / "bad_images.json")
    delete_txt = not args.keep_txt
    effective_report_path = report_path

    if args.from_report:
        report_in = Path(args.from_report).resolve()
        if not report_in.exists():
            raise SystemExit(f"report not found: {report_in}")
        if not args.delete:
            raise SystemExit("--from_report requires --delete")
        effective_report_path = report_in
        stats = delete_from_report(
            report_path=report_in,
            delete_txt=delete_txt,
            dry_run=args.dry_run,
        )
        logging.info(
            "delete summary: checked=%d removed_images=%d removed_txt=%d missing=%d failed=%d",
            stats["checked"],
            stats["removed_images"],
            stats["removed_txt"],
            stats["missing"],
            stats["failed"],
        )
    else:
        bad, stats = scan_and_cleanup(
            root=root,
            workers=max(1, args.workers),
            delete_bad=args.delete,
            delete_txt=delete_txt,
            dry_run=args.dry_run,
        )
        save_report(report_path, bad)

        if args.delete:
            logging.info(
                "done: checked=%d bad=%d removed_images=%d removed_txt=%d missing=%d failed=%d",
                stats["checked"],
                stats["bad"],
                stats["removed_images"],
                stats["removed_txt"],
                stats["missing"],
                stats["failed"],
            )
        else:
            logging.info("done: checked=%d bad=%d", stats["checked"], stats["bad"])

    if args.hf_upload:
        token = os.getenv(args.hf_token_env, "").strip()
        if not token:
            raise SystemExit(f"{args.hf_token_env} is not set")
        if args.dry_run:
            logging.warning("--dry_run is enabled; uploading current folder state without deletions")
        repo_id = upload_to_hf(
            root=root,
            report_path=effective_report_path if effective_report_path.exists() else None,
            token=token,
            repo_type=args.hf_repo_type,
            private=args.hf_private,
            repo_id=args.hf_repo_id,
            strategy=args.hf_strategy,
            chunk_gb=args.hf_chunk_gb,
            workers=max(1, args.workers),
            no_xet_high_performance=args.hf_no_xet_high_performance,
        )
        logging.info("hugging face upload done: %s", repo_id)


if __name__ == "__main__":
    main()
