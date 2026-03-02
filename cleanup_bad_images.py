#!/usr/bin/env python3
"""Scan dataset images quickly and optionally delete broken files in the same pass."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageFile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".jxl"}
PROGRESS_EVERY = 1000


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
        # With LOAD_TRUNCATED_IMAGES=False, truncated files still fail.
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    report_path = Path(args.report).resolve() if args.report else (root.parent / "bad_images.json")
    delete_txt = not args.keep_txt

    if args.from_report:
        report_in = Path(args.from_report).resolve()
        if not report_in.exists():
            raise SystemExit(f"report not found: {report_in}")
        if not args.delete:
            raise SystemExit("--from_report requires --delete")
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
        return

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


if __name__ == "__main__":
    main()
