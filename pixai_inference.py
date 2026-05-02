"""
Standalone PixAI tagger inference.

Downloads `pixai-labs/pixai-tagger-v0.9` from Hugging Face on first run
(requires HF_TOKEN — the repo is gated), builds the PyTorch model,
and tags every image in a directory at a high batch size, writing one
`.txt` next to each image with comma-separated tags.

Usage:
    export HF_TOKEN=hf_xxx
    python pixai_inference.py /path/to/images
    python pixai_inference.py /path/to/images --batch-size 256 --recursive

Designed for big GPUs (RTX PRO 6000 WS / A100 / H100) — defaults to a
large batch size, fp16 autocast, pinned memory, and parallel image
loading via a PyTorch DataLoader.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 500_000_000

REPO_ID = "pixai-labs/pixai-tagger-v0.9"
ENCODER_REPO = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
WEIGHTS_FILE = "model_v0.9.pth"
TAGS_FILE = "tags_v0.9_13k.json"
CHAR_IP_MAP_FILE = "char_ip_map.json"
CATEGORY_THRESHOLDS_FILE = "category_thresholds.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DEFAULT_GENERAL_THRESHOLD = 0.30
DEFAULT_CHARACTER_THRESHOLD = 0.85

logger = logging.getLogger("pixai")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PixAIHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


def build_model(weights_path: str, device: torch.device, num_classes: int) -> nn.Module:
    import timm

    encoder = timm.create_model(ENCODER_REPO, pretrained=False)
    encoder.reset_classifier(0)
    decoder = PixAIHead(1024, num_classes)
    model = nn.Sequential(encoder, decoder)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------

def download_assets(model_dir: Path, hf_token: Optional[str]) -> Tuple[Path, Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    files = [WEIGHTS_FILE, TAGS_FILE, CHAR_IP_MAP_FILE, CATEGORY_THRESHOLDS_FILE]
    for fname in files:
        target = model_dir / fname
        if target.exists():
            continue
        logger.info("downloading %s from %s", fname, REPO_ID)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=str(model_dir),
            token=hf_token,
        )

    return (
        model_dir / WEIGHTS_FILE,
        model_dir / TAGS_FILE,
        model_dir / CHAR_IP_MAP_FILE,
        model_dir / CATEGORY_THRESHOLDS_FILE,
    )


def load_tag_map(tags_path: Path) -> Tuple[Dict[int, str], int, int, int]:
    with tags_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    tag_map = info["tag_map"]
    split = info["tag_split"]
    gen_count = int(split["gen_tag_count"])
    char_count = int(split["character_tag_count"])
    index_to_tag = {int(v): k for k, v in tag_map.items()}
    return index_to_tag, gen_count, char_count, len(tag_map)


def load_char_ip_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_category_thresholds(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return out
    header = [c.strip().lower() for c in rows[0]]
    name_idx = header.index("name") if "name" in header else None
    cat_idx = header.index("category") if "category" in header else None
    th_idx = header.index("threshold") if "threshold" in header else None
    start = 1 if (name_idx is not None or cat_idx is not None) else 0
    for row in rows[start:]:
        if not row:
            continue
        try:
            th = float(row[th_idx]) if th_idx is not None else float(row[-1])
        except (ValueError, IndexError):
            continue
        name = row[name_idx].strip().lower() if name_idx is not None and name_idx < len(row) else ""
        category = row[cat_idx].strip() if cat_idx is not None and cat_idx < len(row) else ""
        if name:
            out[name] = th
        elif category:
            out[category] = th
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        image.load()
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode == "P":
        return pil_to_rgb(image.convert("RGBA"))
    return image.convert("RGB")


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                img = pil_to_rgb(img)
                tensor = self.transform(img)
            return str(path), tensor, True
        except Exception as exc:
            logger.warning("failed to load %s: %s", path, exc)
            return str(path), torch.zeros(3, 448, 448), False


def collate(batch):
    paths = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    valid = [b[2] for b in batch]
    return paths, tensors, valid


def gather_images(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def autocast_dtype(device: torch.device, requested: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp32":
        return None
    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def write_tags(image_path: Path, tags: List[str], suffix: str, separator: str, remove_underscore: bool) -> None:
    if remove_underscore:
        tags = [t.replace("_", " ") if len(t) > 3 else t for t in tags]
    out_path = image_path.with_suffix(suffix)
    out_path.write_text(separator.join(tags), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {input_dir}")

    paths = gather_images(input_dir, args.recursive)
    if not paths:
        logger.warning("no images found under %s (recursive=%s)", input_dir, args.recursive)
        return
    logger.info("found %d images", len(paths))

    if not args.overwrite:
        before = len(paths)
        paths = [p for p in paths if not p.with_suffix(args.out_suffix).exists()]
        skipped = before - len(paths)
        if skipped:
            logger.info("skipping %d already-tagged images (use --overwrite to re-run)", skipped)
        if not paths:
            logger.info("nothing to do")
            return

    model_dir = Path(args.model_dir).expanduser().resolve()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    weights_path, tags_path, ip_map_path, thresholds_path = download_assets(model_dir, hf_token)

    index_to_tag, gen_count, char_count, total = load_tag_map(tags_path)
    char_ip_map = load_char_ip_map(ip_map_path)
    category_thresholds = load_category_thresholds(thresholds_path)

    general_th = args.general_threshold
    if general_th is None:
        general_th = category_thresholds.get("general") or category_thresholds.get("0") or DEFAULT_GENERAL_THRESHOLD
    character_th = args.character_threshold
    if character_th is None:
        character_th = category_thresholds.get("character") or category_thresholds.get("4") or DEFAULT_CHARACTER_THRESHOLD
    logger.info("thresholds: general=%.3f character=%.3f", general_th, character_th)

    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("device=%s batch_size=%d num_workers=%d", device, args.batch_size, args.num_workers)

    model = build_model(str(weights_path), device, total)
    if args.compile and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    dataset = ImageDataset(paths, build_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    amp_dtype = autocast_dtype(device, args.precision)
    if amp_dtype is not None:
        logger.info("using autocast dtype=%s", amp_dtype)

    write_pool = ThreadPoolExecutor(max_workers=max(2, args.num_workers))
    pending = []

    with torch.inference_mode():
        for batch_paths, batch_tensor, valid in tqdm(loader, desc="pixai", smoothing=0.0):
            if device.type == "cuda":
                batch_tensor = batch_tensor.to(device, non_blocking=True)
            else:
                batch_tensor = batch_tensor.to(device)

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    probs = model(batch_tensor).float()
            else:
                probs = model(batch_tensor)

            gen_block = probs[:, :gen_count]
            char_block = probs[:, gen_count : gen_count + char_count]
            gen_mask = (gen_block > general_th).cpu()
            char_mask = (char_block > character_th).cpu()

            for i, image_path in enumerate(batch_paths):
                if not valid[i]:
                    continue
                gen_idx = gen_mask[i].nonzero(as_tuple=True)[0].tolist()
                char_idx = char_mask[i].nonzero(as_tuple=True)[0].tolist()

                general_tags = [index_to_tag[idx] for idx in gen_idx if idx in index_to_tag]
                character_tags = [
                    index_to_tag[idx + gen_count]
                    for idx in char_idx
                    if (idx + gen_count) in index_to_tag
                ]

                ip_tags: List[str] = []
                if not args.no_ip_tags:
                    for tag in character_tags:
                        if tag in char_ip_map:
                            ip_tags.extend(char_ip_map[tag])
                    ip_tags = sorted(set(ip_tags))

                if args.character_first:
                    tags_out = character_tags + ip_tags + general_tags
                else:
                    tags_out = general_tags + character_tags + ip_tags

                pending.append(
                    write_pool.submit(
                        write_tags,
                        Path(image_path),
                        tags_out,
                        args.out_suffix,
                        args.separator,
                        args.remove_underscore,
                    )
                )

    for fut in pending:
        fut.result()
    write_pool.shutdown(wait=True)
    logger.info("done. wrote tags for %d images", sum(1 for _ in pending))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone PixAI tagger inference.")
    p.add_argument("input_dir", help="directory with images")
    p.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    p.add_argument("--batch-size", type=int, default=128, help="GPU batch size (default: 128, crank it for big GPUs)")
    p.add_argument("--num-workers", type=int, default=8, help="dataloader workers for image I/O")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="autocast dtype on CUDA")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (slow first batch, faster later)")
    p.add_argument("--model-dir", default="./models/pixai", help="where to cache model weights")
    p.add_argument("--hf-token", default=None, help="overrides $HF_TOKEN / $HUGGINGFACE_HUB_TOKEN")
    p.add_argument("--general-threshold", type=float, default=None, help="override general-tag threshold")
    p.add_argument("--character-threshold", type=float, default=None, help="override character-tag threshold")
    p.add_argument("--no-ip-tags", action="store_true", help="skip mapping characters to their source IP")
    p.add_argument("--character-first", action="store_true", help="put character/IP tags before general tags")
    p.add_argument("--remove-underscore", action="store_true", default=True, help="replace _ with space in tags (default on)")
    p.add_argument("--keep-underscore", dest="remove_underscore", action="store_false", help="keep underscores in tags")
    p.add_argument("--separator", default=", ", help="tag separator in output file")
    p.add_argument("--out-suffix", default=".txt", help="output file suffix per image")
    p.add_argument("--overwrite", action="store_true", help="re-tag images that already have an output file")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
