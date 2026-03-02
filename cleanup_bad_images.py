#!/usr/bin/env python3
"""Scan a dataset for unreadable images and optionally remove them (+ sidecar .txt)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageFile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".jxl"}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def check_image(path: Path) -> str | None:
    """Return error string if image is broken; otherwise None."""
    try:
        # First pass validates file structure.
        with Image.open(path) as img:
            img.verify()
        # Second pass forces actual decode for truncated/partial files.
        with Image.open(path) as img:
            img.load()
        return None
    except Exception as exc:
        return str(exc)


def scan_bad_images(root: Path) -> List[Dict[str, str]]:
    bad: List[Dict[str, str]] = []
    count = 0
    for img_path in iter_images(root):
        count += 1
        err = check_image(img_path)
        if err:
            bad.append({"path": str(img_path), "error": err})
            logging.warning("bad image: %s (%s)", img_path, err)
    logging.info("scan finished: checked=%d bad=%d", count, len(bad))
    return bad


def save_report(report_path: Path, bad: List[Dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("report saved: %s", report_path)


def delete_bad_images(
    bad: List[Dict[str, str]],
    delete_txt: bool,
    dry_run: bool,
) -> Dict[str, int]:
    removed_images = 0
    removed_txt = 0
    missing = 0
    failed = 0

    for rec in bad:
        img = Path(rec["path"])
        txt = img.with_suffix(".txt")

        try:
            if img.exists():
                if dry_run:
                    logging.info("[dry-run] remove image: %s", img)
                else:
                    img.unlink()
                removed_images += 1
            else:
                missing += 1

            if delete_txt and txt.exists():
                if dry_run:
                    logging.info("[dry-run] remove txt: %s", txt)
                else:
                    txt.unlink()
                removed_txt += 1
        except Exception as exc:
            failed += 1
            logging.error("failed removing %s: %s", img, exc)

    return {
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
    parser.add_argument("--delete", action="store_true", help="remove bad images found in report/scan")
    parser.add_argument("--keep_txt", action="store_true", help="do not remove sidecar .txt files")
    parser.add_argument("--dry_run", action="store_true", help="print deletions without deleting")
    parser.add_argument("--from_report", default=None, help="skip scan and load bad images from this JSON report")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    report_path = Path(args.report).resolve() if args.report else (root.parent / "bad_images.json")

    if args.from_report:
        report_in = Path(args.from_report).resolve()
        if not report_in.exists():
            raise SystemExit(f"report not found: {report_in}")
        bad = json.loads(report_in.read_text(encoding="utf-8"))
        logging.info("loaded %d bad records from report: %s", len(bad), report_in)
    else:
        bad = scan_bad_images(root)
        save_report(report_path, bad)

    if args.delete:
        stats = delete_bad_images(
            bad=bad,
            delete_txt=not args.keep_txt,
            dry_run=args.dry_run,
        )
        logging.info(
            "delete summary: removed_images=%d removed_txt=%d missing=%d failed=%d",
            stats["removed_images"],
            stats["removed_txt"],
            stats["missing"],
            stats["failed"],
        )


if __name__ == "__main__":
    main()
