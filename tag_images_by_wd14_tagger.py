from __future__ import annotations

import argparse
import base64
import collections
import csv
import gc
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# Allow opening very large images (resize will bring them down to 1024px anyway)
Image.MAX_IMAGE_PIXELS = 500_000_000

from wd14_utils import (
    extract_frame,
    extract_frames,
    glob_images_pathlib,
    glob_videos_pathlib,
    resize_image,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
DEFAULT_CAMIE_REPO = "Camais03/camie-tagger-v2"
DEFAULT_PIXAI_REPO = "pixai-labs/pixai-tagger-v0.9"

WD14_CSV_FILE = "selected_tags.csv"
WD14_ONNX_NAME = "model.onnx"

CAMIE_ONNX_FILE = "camie-tagger-v2.onnx"
CAMIE_META_FILE = "camie-tagger-v2-metadata.json"
CAMIE_CATEGORY_THRESHOLDS_FILE = "category_thresholds.csv"

PIXAI_PTH_FILE = "model_v0.9.pth"
PIXAI_TAGS_JSON_FILE = "tags_v0.9_13k.json"
PIXAI_CHAR_IP_MAP_FILE = "char_ip_map.json"
PIXAI_CATEGORY_THRESHOLDS_FILE = "category_thresholds.csv"


# -------------------------
# Common helpers
# -------------------------

def build_session(onnx_path: str) -> Tuple[Any, str, Optional[int]]:
    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    try:
        batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    except Exception:
        batch_size = None
    del model

    providers: List[str] = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in available:
        providers.append("ROCMExecutionProvider")
    providers.append("CPUExecutionProvider")

    logger.info(f"Using onnxruntime providers: {providers}")
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, input_name, batch_size


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -------------------------
# Preprocess
# -------------------------

def preprocess_wd14(image: Image.Image) -> np.ndarray:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    image = np.array(image)
    image = image[:, :, ::-1]

    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    image = resize_image(image, image.shape[1], image.shape[0], IMAGE_SIZE, IMAGE_SIZE)
    return image.astype(np.float32)


def preprocess_imagenet(image: Image.Image, image_size: int) -> np.ndarray:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    width, height = image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = image_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = image_size
        new_width = int(new_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    pad_color = (124, 116, 104)
    padded = Image.new("RGB", (image_size, image_size), pad_color)
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2
    padded.paste(image, (paste_x, paste_y))

    img = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return img


# -------------------------
# Post-processing
# -------------------------

def apply_remove_underscore(tags: List[str]) -> List[str]:
    return [t.replace("_", " ") if len(t) > 3 else t for t in tags]


def apply_tag_replacement(tags: List[str], tag_replacements_arg: str) -> List[str]:
    escaped = tag_replacements_arg.replace("\\,", "@@@@").replace("\\;", "####")
    pairs = escaped.split(";")

    for pair in pairs:
        parts = pair.split(",", 1)
        if len(parts) != 2:
            continue
        source, target = parts
        source = source.replace("@@@@", ",").replace("####", ";")
        target = target.replace("@@@@", ",").replace("####", ";")
        tags = [target if t == source else t for t in tags]

    return tags


def filter_undesired(tags: List[str], undesired_str: str, sep: str) -> List[str]:
    if not undesired_str:
        return tags
    undesired = set(t.strip() for t in undesired_str.split(sep.strip()) if t.strip())
    return [t for t in tags if t not in undesired]


def apply_always_first(tags: List[str], always_first: str, sep: str) -> List[str]:
    always = [t.strip() for t in always_first.split(sep.strip()) if t.strip()]
    for tag in always:
        if tag in tags:
            tags.remove(tag)
            tags.insert(0, tag)
    return tags


def postprocess_tags(tags: List[str], args) -> List[str]:
    if args.remove_underscore:
        tags = apply_remove_underscore(tags)
    if args.tag_replacement:
        tags = apply_tag_replacement(tags, args.tag_replacement)
    tags = filter_undesired(tags, args.undesired_tags, args.caption_separator)
    if args.always_first_tags:
        tags = apply_always_first(tags, args.always_first_tags, args.caption_separator)
    return tags


def add_tags_to_map(result_map: Dict[str, List[str]], image_path: str, tags: List[str], dedupe: bool) -> None:
    if not tags:
        return

    if not dedupe:
        result_map[image_path].extend(tags)
        return

    seen = set(result_map[image_path])
    for tag in tags:
        if tag not in seen:
            result_map[image_path].append(tag)
            seen.add(tag)


# -------------------------
# WD14 tagger
# -------------------------

def load_wd14_tags(model_location: str, args) -> Tuple[List[str], List[str], List[str]]:
    with open(os.path.join(model_location, WD14_CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    header = rows[0]
    data = rows[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category"

    rating_tags = [row[1] for row in data if row[2] == "9"]
    general_tags = [row[1] for row in data if row[2] == "0"]
    character_tags = [row[1] for row in data if row[2] == "4"]

    if args.remove_underscore:
        rating_tags = apply_remove_underscore(rating_tags)
        general_tags = apply_remove_underscore(general_tags)
        character_tags = apply_remove_underscore(character_tags)

    if args.tag_replacement:
        rating_tags = apply_tag_replacement(rating_tags, args.tag_replacement)
        general_tags = apply_tag_replacement(general_tags, args.tag_replacement)
        character_tags = apply_tag_replacement(character_tags, args.tag_replacement)

    if args.character_tag_expand:
        character_tags = expand_character_tags(character_tags, args.caption_separator)

    return rating_tags, general_tags, character_tags


def expand_character_tags(tags: List[str], sep: str) -> List[str]:
    out = tags[:]
    for i, tag in enumerate(out):
        if tag.endswith(")"):
            parts = tag.split("(")
            character = "(".join(parts[:-1]).rstrip("_")
            series = parts[-1].replace(")", "")
            out[i] = character + sep + series
    return out


def run_wd14(
    paths: List[str],
    args,
    repo_id: str,
    model_dir: str,
    batch_size: int,
    dedupe: bool,
    result_map: Dict[str, List[str]],
    general_threshold: float,
    character_threshold: float,
    progress=None,
) -> None:
    model_location = os.path.join(model_dir, repo_id.replace("/", "_"))
    if not os.path.exists(model_location) or args.force_download:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"downloading WD14 model from HF: {repo_id}")
        hf_hub_download(repo_id=repo_id, filename=WD14_CSV_FILE, local_dir=model_location, force_download=True)
        hf_hub_download(repo_id=repo_id, filename=WD14_ONNX_NAME, local_dir=model_location, force_download=True)

    rating_tags, general_tags, character_tags = load_wd14_tags(model_location, args)

    onnx_path = os.path.join(model_location, WD14_ONNX_NAME)
    session, input_name, fixed_batch = build_session(onnx_path)
    if fixed_batch and fixed_batch > 0 and batch_size != fixed_batch:
        logger.warning(f"WD14 batch {batch_size} != model batch {fixed_batch}; using {fixed_batch}")
        batch_size = fixed_batch

    batches = batch_loader(paths, batch_size, args.max_workers, preprocess_wd14)
    iterable = batches if progress is not None or args.no_progress else tqdm(batches, smoothing=0.0, desc="wd14", total=count_batches(len(paths), batch_size))

    for batch_paths, batch_imgs in iterable:
        probs = session.run(None, {input_name: batch_imgs})[0]
        probs = probs[: len(batch_paths)]

        for image_path, prob in zip(batch_paths, probs):
            tags: List[str] = []
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= general_threshold:
                    tags.append(general_tags[i])
                elif i >= len(general_tags) and p >= character_threshold:
                    tag_name = character_tags[i - len(general_tags)]
                    if args.character_tags_first:
                        tags.insert(0, tag_name)
                    else:
                        tags.append(tag_name)

            if args.use_rating_tags or args.use_rating_tags_as_last_tag:
                rating_index = prob[:4].argmax()
                rating = rating_tags[rating_index]
                if args.use_rating_tags:
                    tags.insert(0, rating)
                else:
                    tags.append(rating)

            tags = postprocess_tags(tags, args)
            add_tags_to_map(result_map, image_path, tags, dedupe)
        if progress is not None:
            progress.update(1)

    del session
    cleanup_memory()


# -------------------------
# Camie tagger
# -------------------------

def load_camie_metadata(path: str) -> Tuple[Dict[int, str], Dict[str, str], int]:
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_info = metadata["dataset_info"]
    tag_mapping = dataset_info["tag_mapping"]
    idx_to_tag = tag_mapping["idx_to_tag"]
    tag_to_category = tag_mapping["tag_to_category"]
    img_size = metadata.get("model_info", {}).get("img_size", IMAGE_SIZE)

    idx_map = {int(k): v for k, v in idx_to_tag.items()}
    return idx_map, tag_to_category, int(img_size)


def load_camie_category_thresholds(model_location: str, override_path: Optional[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    path = override_path or os.path.join(model_location, CAMIE_CATEGORY_THRESHOLDS_FILE)
    if not path or not os.path.exists(path):
        return thresholds

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if not rows:
        return thresholds

    header = [c.strip().lower() for c in rows[0]]
    name_idx = header.index("name") if "name" in header else None
    category_idx = header.index("category") if "category" in header else None
    th_idx = header.index("threshold") if "threshold" in header else None
    start_row = 1 if (name_idx is not None or category_idx is not None) else 0

    for row in rows[start_row:]:
        if not row:
            continue
        try:
            th = float(row[th_idx]) if th_idx is not None else float(row[-1])
        except Exception:
            continue
        name = row[name_idx].strip().lower() if name_idx is not None and name_idx < len(row) else ""
        category = row[category_idx].strip().lower() if category_idx is not None and category_idx < len(row) else ""

        if name:
            thresholds[name] = th
        elif category:
            thresholds[category] = th

    return thresholds


def run_camie(
    paths: List[str],
    args,
    repo_id: str,
    model_dir: str,
    batch_size: int,
    dedupe: bool,
    result_map: Dict[str, List[str]],
    general_threshold: float,
    character_threshold: float,
    progress=None,
) -> None:
    model_location = os.path.join(model_dir, repo_id.replace("/", "_"))
    if not os.path.exists(model_location) or args.force_download:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"downloading Camie model from HF: {repo_id}")
        hf_hub_download(repo_id=repo_id, filename=CAMIE_ONNX_FILE, local_dir=model_location, force_download=True)
        hf_hub_download(repo_id=repo_id, filename=CAMIE_META_FILE, local_dir=model_location, force_download=True)
        try:
            hf_hub_download(repo_id=repo_id, filename=CAMIE_CATEGORY_THRESHOLDS_FILE, local_dir=model_location, force_download=True)
        except Exception:
            pass

    onnx_path = os.path.join(model_location, CAMIE_ONNX_FILE)
    meta_path = os.path.join(model_location, CAMIE_META_FILE)

    idx_to_tag, tag_to_category, img_size = load_camie_metadata(meta_path)
    category_thresholds = load_camie_category_thresholds(model_location, args.camie_category_thresholds_file)
    min_confidence = args.camie_min_confidence
    if not category_thresholds and args.camie_thresh is None and args.camie_general_threshold is None and args.camie_character_threshold is None:
        # Default to macro-optimized threshold (from official docs)
        general_threshold = 0.492
        character_threshold = 0.492
    session, input_name, fixed_batch = build_session(onnx_path)
    if fixed_batch and fixed_batch > 0 and batch_size != fixed_batch:
        logger.warning(f"Camie batch {batch_size} != model batch {fixed_batch}; using {fixed_batch}")
        batch_size = fixed_batch

    batches = batch_loader(paths, batch_size, args.max_workers, lambda img: preprocess_imagenet(img, img_size))
    iterable = batches if progress is not None or args.no_progress else tqdm(batches, smoothing=0.0, desc="camie", total=count_batches(len(paths), batch_size))

    for batch_paths, batch_imgs in iterable:
        outputs = session.run(None, {input_name: batch_imgs})
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]
        probs = sigmoid(logits)

        for image_path, prob in zip(batch_paths, probs):
            tags: List[str] = []
            for idx, p in enumerate(prob):
                if p < min_confidence:
                    continue
                tag = idx_to_tag.get(idx)
                if tag is None:
                    continue
                category = tag_to_category.get(tag, "general")
                category_key = category.lower()
                cat_threshold = category_thresholds.get(category_key)
                if cat_threshold is None:
                    if category_key == "character":
                        cat_threshold = character_threshold
                    else:
                        cat_threshold = general_threshold

                if p < cat_threshold:
                    continue

                if category_key == "character":
                    if args.character_tags_first:
                        tags.insert(0, tag)
                    else:
                        tags.append(tag)
                elif category_key == "rating":
                    if args.use_rating_tags:
                        tags.insert(0, tag)
                    elif args.use_rating_tags_as_last_tag:
                        tags.append(tag)
                else:
                    tags.append(tag)

            tags = postprocess_tags(tags, args)
            add_tags_to_map(result_map, image_path, tags, dedupe)
        if progress is not None:
            progress.update(1)

    del session
    cleanup_memory()


# -------------------------
# PixAI tagger (PyTorch)
# -------------------------


def pixai_pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        image.load()
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    if image.mode == "P":
        return pixai_pil_to_rgb(image.convert("RGBA"))
    return image.convert("RGB")


def build_pixai_transform() -> Any:
    from torchvision import transforms
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_pixai_assets(
    model_location: str,
    repo_id: str,
    force: bool,
    hf_token: Optional[str],
) -> Tuple[str, str, str, Dict[str, str]]:
    download_errors: Dict[str, str] = {}
    if not os.path.exists(model_location) or force:
        from huggingface_hub import hf_hub_download
        os.makedirs(model_location, exist_ok=True)
        logger.info(f"downloading PixAI model from HF: {repo_id}")
        for file in [
            PIXAI_PTH_FILE,
            PIXAI_TAGS_JSON_FILE,
            PIXAI_CHAR_IP_MAP_FILE,
            PIXAI_CATEGORY_THRESHOLDS_FILE,
        ]:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    local_dir=model_location,
                    force_download=True,
                    token=hf_token,
                )
            except Exception as exc:
                msg = str(exc).strip()
                download_errors[file] = msg
                logger.warning(f"PixAI download failed for {file}: {msg}")

    weights_path = os.path.join(model_location, PIXAI_PTH_FILE)
    tags_path = os.path.join(model_location, PIXAI_TAGS_JSON_FILE)
    ip_map_path = os.path.join(model_location, PIXAI_CHAR_IP_MAP_FILE)
    return weights_path, tags_path, ip_map_path, download_errors


def load_pixai_tag_map(tags_path: str) -> Tuple[Dict[int, str], int, int, int]:
    with open(tags_path, "r", encoding="utf-8") as f:
        tag_info = json.load(f)
    tag_map = tag_info["tag_map"]
    tag_split = tag_info["tag_split"]
    gen_count = int(tag_split["gen_tag_count"])
    char_count = int(tag_split["character_tag_count"])
    index_to_tag = {int(v): k for k, v in tag_map.items()}
    return index_to_tag, gen_count, char_count, len(tag_map)


def load_pixai_char_ip_map(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items()}


def load_pixai_category_thresholds(model_location: str, override_path: Optional[str] = None) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    path = override_path or os.path.join(model_location, PIXAI_CATEGORY_THRESHOLDS_FILE)
    if not path or not os.path.exists(path):
        return thresholds

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if not rows:
        return thresholds

    header = [c.strip().lower() for c in rows[0]]
    name_idx = header.index("name") if "name" in header else None
    category_idx = header.index("category") if "category" in header else None
    th_idx = header.index("threshold") if "threshold" in header else None
    start_row = 1 if (name_idx is not None or category_idx is not None) else 0

    for row in rows[start_row:]:
        if not row:
            continue
        try:
            th = float(row[th_idx]) if th_idx is not None else float(row[-1])
        except Exception:
            continue
        name = row[name_idx].strip().lower() if name_idx is not None and name_idx < len(row) else ""
        category = row[category_idx].strip() if category_idx is not None and category_idx < len(row) else ""

        if name:
            thresholds[name] = th
        elif category:
            thresholds[category] = th

    return thresholds


def build_pixai_model(weights_path: str, device: str, num_classes: int) -> Any:
    import timm
    import torch

    class PixAITaggingHead(torch.nn.Module):
        def __init__(self, input_dim: int, num_classes: int) -> None:
            super().__init__()
            self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))

        def forward(self, x):
            return torch.sigmoid(self.head(x))

    base_model_repo = "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3"
    encoder = timm.create_model(base_model_repo, pretrained=False)
    encoder.reset_classifier(0)
    decoder = PixAITaggingHead(1024, num_classes)
    model = torch.nn.Sequential(encoder, decoder)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def batch_loader_torch(
    paths: List[str],
    batch_size: int,
    max_workers: int,
    transform,
) -> Iterable[Tuple[List[str], Any]]:
    import torch
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    def _load(path: str):
        with Image.open(path) as img:
            img = pixai_pil_to_rgb(img)
            tensor = transform(img)
        return path, tensor

    for batch in batches:
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                loaded = list(ex.map(_load, batch))
        else:
            loaded = [_load(p) for p in batch]

        if not loaded:
            yield [], torch.zeros((0, 3, 448, 448), dtype=torch.float32)
            continue

        paths_b, tensors_b = zip(*loaded)
        batch_tensor = torch.stack(tensors_b)
        yield list(paths_b), batch_tensor


def run_pixai(
    paths: List[str],
    args,
    repo_id: str,
    model_dir: str,
    batch_size: int,
    dedupe: bool,
    result_map: Dict[str, List[str]],
    general_threshold: float,
    character_threshold: float,
    progress=None,
) -> None:
    model_location = os.path.join(model_dir, repo_id.replace("/", "_"))
    weights_path, tags_path, ip_map_path, download_errors = load_pixai_assets(
        model_location, repo_id, args.force_download, args.hf_token
    )

    missing = [p for p in (weights_path, tags_path, ip_map_path) if not os.path.exists(p)]
    if missing:
        details = ""
        if download_errors:
            lines = [f"- {k}: {v}" for k, v in download_errors.items()]
            details = "\nDownload errors:\n" + "\n".join(lines)
        raise FileNotFoundError(
            "PixAI assets not found. The repo is gated/private and requires a HF token.\n"
            "Set env HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) or pass --hf_token.\n"
            f"Missing: {', '.join(missing)}{details}"
        )

    index_to_tag, gen_count, char_count, total_tags = load_pixai_tag_map(tags_path)
    char_ip_map = load_pixai_char_ip_map(ip_map_path)

    category_thresholds = load_pixai_category_thresholds(model_location, args.pixai_category_thresholds_file)
    if args.pixai_general_threshold is None and args.pixai_thresh is None:
        general_override = category_thresholds.get("general") or category_thresholds.get("0")
        if general_override is not None:
            general_threshold = general_override
        else:
            general_threshold = 0.30
    if args.pixai_character_threshold is None and args.pixai_thresh is None:
        character_override = category_thresholds.get("character") or category_thresholds.get("4")
        if character_override is not None:
            character_threshold = character_override
        else:
            character_threshold = 0.85

    import torch

    device = args.pixai_device
    if device == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                device = "cuda"
            except Exception:
                device = "cpu"
        else:
            device = "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_pixai_model(weights_path, device, total_tags)
    transform = build_pixai_transform()

    batches = batch_loader_torch(paths, batch_size, args.max_workers, transform)
    iterable = batches if progress is not None or args.no_progress else tqdm(batches, smoothing=0.0, desc="pixai", total=count_batches(len(paths), batch_size))

    mode = args.pixai_mode
    topk_general = max(0, min(int(args.pixai_topk_general), gen_count))
    topk_character = max(0, min(int(args.pixai_topk_character), char_count))

    for batch_paths, batch_tensor in iterable:
        if device == "cuda":
            batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
        else:
            batch_tensor = batch_tensor.to(device)

        with torch.inference_mode():
            probs = model(batch_tensor)

        for i, image_path in enumerate(batch_paths):
            prob = probs[i]
            if mode == "topk":
                gen_scores, gen_idx = (torch.tensor([]), torch.tensor([], dtype=torch.long))
                char_scores, char_idx = (torch.tensor([]), torch.tensor([], dtype=torch.long))
                if topk_general > 0:
                    gen_scores, gen_idx = torch.topk(prob[:gen_count], topk_general)
                if topk_character > 0:
                    char_scores, char_idx = torch.topk(prob[gen_count : gen_count + char_count], topk_character)
                    char_idx = char_idx + gen_count
                combined_idx = torch.cat((gen_idx, char_idx)).cpu()
            else:
                general_mask = prob[:gen_count] > general_threshold
                character_mask = prob[gen_count : gen_count + char_count] > character_threshold
                gen_idx = general_mask.nonzero(as_tuple=True)[0]
                char_idx = character_mask.nonzero(as_tuple=True)[0] + gen_count
                combined_idx = torch.cat((gen_idx, char_idx)).cpu()

            general_tags: List[str] = []
            character_tags: List[str] = []
            for idx in combined_idx.tolist():
                tag = index_to_tag.get(int(idx))
                if tag is None:
                    continue
                if idx < gen_count:
                    general_tags.append(tag)
                else:
                    character_tags.append(tag)

            ip_tags: List[str] = []
            if not args.pixai_no_ip:
                for tag in character_tags:
                    if tag in char_ip_map:
                        ip_tags.extend(char_ip_map[tag])
                ip_tags = sorted(set(ip_tags))

            if args.character_tags_first:
                tags_out = character_tags + ip_tags + general_tags
            else:
                tags_out = general_tags + character_tags + ip_tags

            tags_out = postprocess_tags(tags_out, args)
            add_tags_to_map(result_map, image_path, tags_out, dedupe)

        # Release GPU memory for this batch immediately
        del batch_tensor, probs
        if progress is not None:
            progress.update(1)

    del model
    cleanup_memory()


# -------------------------
# Grok tagger (OpenRouter API)
# -------------------------

DEFAULT_GROK_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_XAI_BATCH_MODEL = "grok-4-1-fast-non-reasoning"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
XAI_API_BASE_URL = "https://api.x.ai"
# xAI Batch API documented limits (docs.x.ai/guides/batch-api):
# - max payload per add-requests call: 25 MB
# - max add-requests calls per team: 100 every 30 seconds (rolling window)
XAI_BATCH_MAX_ADD_PAYLOAD_BYTES = 25 * 1024 * 1024
XAI_BATCH_PAYLOAD_SAFETY_MARGIN_BYTES = 2 * 1024 * 1024
XAI_BATCH_MAX_ADD_CALLS_PER_30S = 100
XAI_BATCH_ADD_WINDOW_SECONDS = 30.0

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
GROK_TAG_CATEGORIES = {"character", "artist", "copyright"}


def load_prompt_file(path: str) -> str:
    """Load a prompt from a .md file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_system_prompt(args) -> str:
    """Load system prompt from file or use CLI override."""
    if args.grok_system_prompt_file:
        return load_prompt_file(args.grok_system_prompt_file)

    mode_dir = "video" if getattr(args, "video", False) else "image"
    candidates = [
        os.path.join(PROMPTS_DIR, mode_dir, "system_prompt.md"),
        os.path.join(PROMPTS_DIR, "system_prompt.md"),  # backward-compatible fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_prompt_file(path)

    return "You are an expert image captioner for AI training datasets."


def get_user_prompt_template(args) -> str:
    """Load user prompt template from file or use CLI override."""
    if args.grok_prompt_file:
        return load_prompt_file(args.grok_prompt_file)

    mode_dir = "video" if getattr(args, "video", False) else "image"
    candidates = [
        os.path.join(PROMPTS_DIR, mode_dir, "user_prompt.md"),
        os.path.join(PROMPTS_DIR, "user_prompt.md"),  # backward-compatible fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_prompt_file(path)

    return "Analyze this image and the following tags:\n\n{tags}"


def _normalize_tag_key(tag: str) -> str:
    return tag.strip().lower()


def _build_grok_tag_category_lookup(tag_to_category: Dict[str, str]) -> Dict[str, str]:
    """Build a normalized lookup map for selected categories used in grok prompts."""
    lookup: Dict[str, str] = {}
    for tag, category in tag_to_category.items():
        cat = str(category).strip().lower()
        if cat not in GROK_TAG_CATEGORIES:
            continue

        raw = str(tag).strip()
        if not raw:
            continue

        # Match both raw booru form and underscore-removed form.
        lookup[_normalize_tag_key(raw)] = cat
        lookup[_normalize_tag_key(raw.replace("_", " "))] = cat
    return lookup


def _load_grok_tag_category_lookup(args) -> Dict[str, str]:
    """Load local Camie metadata and build tag->category lookup for grok prompt enrichment."""
    explicit_path = args.grok_tag_category_metadata_file
    # Prefer metadata committed in this repo root.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(repo_root, CAMIE_META_FILE)

    meta_path = explicit_path or default_path

    if not os.path.exists(meta_path):
        logger.warning(
            "local tag-category metadata not found for grok prompt enrichment: "
            f"{meta_path}. continuing without category suffixes."
        )
        return {}

    try:
        _, tag_to_category, _ = load_camie_metadata(meta_path)
    except Exception as e:
        logger.warning(f"failed to parse tag-category metadata from {meta_path}: {e}")
        return {}

    lookup = _build_grok_tag_category_lookup(tag_to_category)
    if lookup:
        logger.info(f"loaded {len(lookup)} categorized tags for grok prompt enrichment")
    return lookup


def _format_grok_tags_with_categories(tags_list: List[str], tag_category_lookup: Dict[str, str]) -> List[str]:
    """Append category suffix to selected tags before sending context to grok."""
    out: List[str] = []
    for tag in tags_list:
        clean = tag.strip()
        if not clean:
            continue

        # Skip if already annotated by caller.
        lower_clean = clean.lower()
        if lower_clean.endswith("(character)") or lower_clean.endswith("(artist)") or lower_clean.endswith("(copyright)"):
            out.append(clean)
            continue

        cat = tag_category_lookup.get(_normalize_tag_key(clean))
        if cat in GROK_TAG_CATEGORIES:
            out.append(f"{clean} ({cat})")
        else:
            out.append(clean)
    return out


def image_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL image to base64 data URI, resizing if needed for API efficiency."""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_uris: List[str],
    temperature: float = 0.3,
    max_retries: int = 3,
    json_mode: bool = True,
) -> Optional[str]:
    """Call OpenRouter API with one or more images and return the text response.

    Uses json_object response_format + response-healing plugin for reliable JSON output.
    Retries with exponential backoff on 429/5xx errors.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter docs: text first, then images
    content_parts = [{"type": "text", "text": user_prompt}]
    for uri in image_data_uris:
        content_parts.append({"type": "image_url", "image_url": {"url": uri}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        "reasoning": {"effort": "none"},
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        # response-healing auto-fixes malformed JSON from the model
        payload["plugins"] = [{"id": "response-healing"}]

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)

            # Retry on rate limit (429) and server errors (5xx)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"OpenRouter {resp.status_code} (attempt {attempt + 1}), retrying in {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.warning(f"Response: {e.response.text[:500]}")
            if attempt < max_retries:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
            else:
                logger.error(f"OpenRouter API failed after {max_retries + 1} attempts")
                return None
    return None


def parse_grok_json_output(raw: str) -> Optional[Dict]:
    """Parse JSON output from grok, handling markdown code blocks and edge cases."""
    text = raw.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        # Remove ```json or ``` at start and ``` at end
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from grok output: {text[:200]}...")
    return None


def _grok_single_task(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    image_path: str,
    tags_str: str,
    extra_image_paths: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Process a single image through grok. Returns (image_path, caption_or_none).

    The raw API response is JSON. We parse it and extract the caption field.
    The full JSON is returned as the caption string (written to .txt).
    """
    image_data_uris = []
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_data_uris.append(image_to_base64(img))
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return image_path, None

    if extra_image_paths:
        for ep in extra_image_paths:
            try:
                with Image.open(ep) as img:
                    img = img.convert("RGB")
                    image_data_uris.append(image_to_base64(img))
            except Exception as e:
                logger.warning(f"Failed to load extra image {ep}: {e}")

    user_prompt = user_prompt_template.replace("{tags}", tags_str)
    raw = call_openrouter(api_key, model, system_prompt, user_prompt, image_data_uris)

    if not raw:
        return image_path, None

    # Try to parse JSON and extract just the caption for the .txt file
    parsed = parse_grok_json_output(raw)
    if parsed and "caption" in parsed:
        return image_path, parsed["caption"]

    # If JSON parsing worked but no caption field, use the whole JSON
    if parsed:
        logger.debug(f"JSON parsed but no 'caption' key, using full JSON for {image_path}")
        return image_path, json.dumps(parsed, ensure_ascii=False)

    # Fallback: use raw text as-is (model didn't return valid JSON)
    return image_path, raw


def _xai_headers(api_key: str, conv_id: Optional[str] = None) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if conv_id:
        h["x-grok-conv-id"] = conv_id
    return h


def _xai_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict] = None,
    params: Optional[Dict] = None,
    max_retries: int = 5,
    timeout: int = 120,
) -> Dict:
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=payload,
                params=params,
                timeout=timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt, 30)
                logger.warning(f"xAI API {resp.status_code} on {url} (attempt {attempt + 1}), retrying in {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                body_preview = (resp.text or "").strip().replace("\n", " ")
                if len(body_preview) > 800:
                    body_preview = body_preview[:800] + "..."
                logger.error("xAI API %s on %s: %s", resp.status_code, url, body_preview or "<empty body>")
            resp.raise_for_status()
            if not resp.text:
                return {}
            return resp.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                status = e.response.status_code if e.response is not None else None
                # 4xx client errors (except 429) are non-retriable for the same request payload.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
            if attempt >= max_retries:
                raise
            wait = min(2 ** attempt, 30)
            logger.warning(f"xAI API request error on {url} (attempt {attempt + 1}): {e}; retrying in {wait}s...")
            time.sleep(wait)
    return {}


def _resolve_xai_state_file(args) -> str:
    if args.xai_batch_state_file:
        return args.xai_batch_state_file
    base_dir = os.path.abspath(args.train_data_dir)
    parent_dir = os.path.dirname(base_dir)
    dataset_name = os.path.basename(base_dir)
    key = hashlib.md5(base_dir.encode("utf-8")).hexdigest()[:10]
    return os.path.join(parent_dir, f".xai_batch_state_{dataset_name}_{key}.json")


def _load_xai_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"could not read xai batch state from {path}: {e}")
        return {}


def _save_xai_state(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _xai_extract_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part.get("content"), str):
                    chunks.append(part["content"])
        return "\n".join(chunks).strip()
    return ""


def _xai_extract_caption_from_result(result_obj: Dict) -> Optional[str]:
    response = result_obj.get("response")
    if response is None and isinstance(result_obj.get("result"), dict):
        response = result_obj["result"].get("response")
    if response is None and isinstance(result_obj.get("batch_result"), dict):
        response = result_obj["batch_result"].get("response")
    if response is None:
        return None

    message = None
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
        if message is None and isinstance(response.get("message"), dict):
            message = response.get("message")

    if not isinstance(message, dict):
        return None

    content_text = _xai_extract_content_text(message.get("content"))
    if not content_text:
        return None

    parsed = parse_grok_json_output(content_text)
    if parsed and "caption" in parsed:
        return parsed["caption"]
    if parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return content_text


def _estimate_request_bytes(request_body: Dict) -> int:
    """Estimate serialized byte size of one batch request body (fast, no json.dumps)."""
    total = 256  # fixed overhead for model/response_format/reasoning_effort keys
    for msg in request_body.get("chat_get_completion", {}).get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content.encode("utf-8", errors="replace"))
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    total += len(part.get("image_url", {}).get("url", ""))
                elif part.get("type") == "text":
                    total += len(part.get("text", "").encode("utf-8", errors="replace"))
    return total


def _json_byte_size(payload: Dict) -> int:
    """Return serialized JSON size in bytes (compact separators)."""
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _xai_build_user_content(user_prompt: str, image_path: str, extra_image_paths: Optional[List[str]], include_images: bool):
    if not include_images:
        return user_prompt

    content_parts = [{"type": "text", "text": user_prompt}]

    image_paths = [image_path]
    if extra_image_paths:
        image_paths.extend(extra_image_paths)

    for ipath in image_paths:
        try:
            with Image.open(ipath) as img:
                img = img.convert("RGB")
                content_parts.append({"type": "image_url", "image_url": {"url": image_to_base64(img)}})
        except Exception as e:
            logger.warning(f"failed to load image for xai batch request {ipath}: {e}")
    return content_parts


def _xai_strip_images_from_batch_request(batch_item: Dict) -> bool:
    """Drop image parts from one batch request, keeping only text prompt content."""
    try:
        user_msg = batch_item["batch_request"]["chat_get_completion"]["messages"][1]
        content = user_msg.get("content")
        if isinstance(content, str):
            return False
        if not isinstance(content, list):
            return False

        text_chunks: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])

        fallback_text = "\n".join([t for t in text_chunks if t.strip()]).strip()
        user_msg["content"] = fallback_text or "(no prompt)"
        batch_item["_images_stripped"] = True
        return True
    except Exception:
        return False


def _resolve_collect_image_path(meta: Dict, train_data_dir: str) -> Optional[str]:
    """Resolve image path from batch metadata, supporting cross-machine collect."""
    abs_path = meta.get("image_path")
    if abs_path and os.path.exists(abs_path):
        return abs_path

    rel_path = meta.get("image_path_rel")
    if rel_path:
        candidate = os.path.join(os.path.abspath(train_data_dir), rel_path)
        if os.path.exists(candidate):
            return candidate

    return None


def run_grok_xai_batch(
    paths: List[str],
    args,
    dedupe: bool,
    result_map: Dict[str, List[str]],
    existing_tags: Dict[str, List[str]],
    extra_frames: Dict[str, List[str]],
) -> Dict:
    if args.video:
        raise ValueError("xai batch mode is currently supported only for image mode (no --video).")

    api_key = args.xai_api_key
    if not api_key:
        raise ValueError("XAI_API_KEY is required for xai batch mode (or pass --xai_api_key).")

    headers = _xai_headers(api_key)
    model = getattr(args, "xai_batch_model", None) or DEFAULT_XAI_BATCH_MODEL
    system_prompt = get_system_prompt(args)
    user_prompt_template = get_user_prompt_template(args)
    tag_category_lookup = _load_grok_tag_category_lookup(args)
    train_data_dir_abs = os.path.abspath(args.train_data_dir)

    state_file = _resolve_xai_state_file(args)
    state = _load_xai_state(state_file)

    system_prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    if not state:
        state = {
            "version": 1,
            "provider": "xai-batch",
            "batch_id": None,
            "batch_name": None,
            "model": model,
            "system_prompt_sha256": system_prompt_hash,
            "state_file": state_file,
            "request_map": {},
            "created_at": time.time(),
        }

    # Batch API docs do not require x-grok-conv-id; keep headers minimal for compatibility.
    headers = _xai_headers(api_key)

    # Warn if the system prompt changed since the batch was created (would bust cache)
    saved_hash = state.get("system_prompt_sha256")
    if saved_hash and saved_hash != system_prompt_hash:
        logger.warning(
            "System prompt has changed since this batch was created! "
            "Cache hits will be lost. saved_sha256=%s current_sha256=%s",
            saved_hash[:12],
            system_prompt_hash[:12],
        )
    elif not saved_hash:
        state["system_prompt_sha256"] = system_prompt_hash

    action = args.xai_batch_action

    def _create_xai_batch(reason: Optional[str] = None) -> str:
        batch_name = args.xai_batch_name or f"tagger_{int(time.time())}"
        created = _xai_request(
            "POST",
            f"{args.xai_api_base_url}/v1/batches",
            headers,
            payload={"name": batch_name},
        )
        new_batch_id = created.get("batch_id")
        if not new_batch_id:
            raise RuntimeError(f"failed to create xai batch. response: {created}")
        state["batch_id"] = new_batch_id
        state["batch_name"] = created.get("name", batch_name)
        state["batch_created_at"] = created.get("created_at")
        if reason:
            state["batch_reset_reason"] = reason
            state["batch_reset_at"] = time.time()
        _save_xai_state(state_file, state)
        logger.info(f"created xai batch: {new_batch_id}")
        return new_batch_id

    if not state.get("batch_id") and action in ("submit",):
        _create_xai_batch(reason="initial_create")

    batch_id = state.get("batch_id")
    if not batch_id:
        raise ValueError(f"xai batch state has no batch_id: {state_file}. run with --xai_batch_action submit first.")

    if action == "submit":
        try:
            batch_meta = _xai_request("GET", f"{args.xai_api_base_url}/v1/batches/{batch_id}", headers)
            state["last_status"] = batch_meta
            _save_xai_state(state_file, state)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (403, 404):
                logger.warning(
                    "existing batch_id=%s in state file is not writable/visible with this key (HTTP %s). "
                    "creating a fresh batch and resetting request_map for submit.",
                    batch_id,
                    status,
                )
                state["request_map"] = {}
                batch_id = _create_xai_batch(reason=f"preflight_http_{status}")
            else:
                raise

    if action == "status":
        batch_meta = _xai_request("GET", f"{args.xai_api_base_url}/v1/batches/{batch_id}", headers)
        state["last_status"] = batch_meta
        _save_xai_state(state_file, state)
        counters = batch_meta.get("state", {})
        logger.info(
            f"xai batch {batch_id} status: "
            f"total={counters.get('num_requests', 0)} pending={counters.get('num_pending', 0)} "
            f"success={counters.get('num_success', 0)} error={counters.get('num_error', 0)}"
        )
        return {"mode": "xai-batch-status", "completed_paths": []}

    if action == "submit":
        submitted_now = 0
        skipped_payload_too_large = 0
        failed_submit_requests = 0
        request_map = state.setdefault("request_map", {})
        if args.force and request_map:
            logger.info("force mode enabled: clearing prior xai request_map states before submit.")
            request_map.clear()
            state["request_map_force_reset_at"] = time.time()
            _save_xai_state(state_file, state)
        rotated_batch_after_forbidden = False
        chunk_size = args.xai_batch_submit_chunk
        if chunk_size <= 0:
            raise ValueError("--xai_batch_submit_chunk must be >= 1")
        include_images = not args.xai_batch_no_image
        # ThreadPoolExecutor workers for parallel image encoding (I/O + PIL, thread-friendly)
        # I/O + PIL encoding is thread-bound, not CPU-bound — use more threads than cores
        encode_workers = min(64, (os.cpu_count() or 4) * 3) if include_images else 1

        use_pbar = not getattr(args, "no_progress", False)

        # Pass 1: scan all paths, filter already-submitted ones (no I/O)
        scan_pbar = tqdm(total=len(paths), desc="xai-batch scan", smoothing=0.0, unit="img") if use_pbar else None
        to_process: List[Tuple[str, str, Optional[str], str, str, Optional[List[str]]]] = []
        for image_path in paths:
            image_abs = os.path.abspath(image_path)
            rel_path = None
            try:
                rel_path = os.path.relpath(image_abs, train_data_dir_abs)
            except Exception:
                rel_path = None
            req_id_seed = rel_path if rel_path and not rel_path.startswith("..") else image_abs
            req_id = "req_" + hashlib.sha1(req_id_seed.encode("utf-8")).hexdigest()
            if scan_pbar is not None:
                scan_pbar.update(1)
            if request_map.get(req_id, {}).get("state") in ("submitted", "succeeded"):
                continue
            tags_list = existing_tags.get(image_path, [])
            prompt_tags = _format_grok_tags_with_categories(tags_list, tag_category_lookup)
            tags_str = args.caption_separator.join(prompt_tags) if prompt_tags else "(no prior tags)"
            extras = extra_frames.get(image_path)
            to_process.append((image_path, image_abs, rel_path, req_id, tags_str, extras))
        if scan_pbar is not None:
            scan_pbar.close()

        logger.info(
            f"xai-batch submit: {len(to_process)} to submit, "
            f"{len(paths) - len(to_process)} already submitted"
        )

        # Pass 2: encode in parallel per chunk, then POST with size-aware flushing.
        # API limit is 25 MB per add-requests call; keep a safety margin to avoid hard-limit failures.
        MAX_PAYLOAD_BYTES = XAI_BATCH_MAX_ADD_PAYLOAD_BYTES - XAI_BATCH_PAYLOAD_SAFETY_MARGIN_BYTES
        rate_limit_window = XAI_BATCH_ADD_WINDOW_SECONDS
        max_add_calls_per_window = max(1, int(XAI_BATCH_MAX_ADD_CALLS_PER_30S * 0.9))  # 10% safety headroom
        add_call_timestamps = collections.deque()
        submit_pbar = tqdm(total=len(to_process), desc="xai-batch submit", smoothing=0.0, unit="img") if use_pbar else None

        def _encode_item(item: Tuple) -> Tuple[str, str, Optional[str], Any]:
            img_path, img_abs, rel_p, req_id, tags_str, extras = item
            user_prompt = user_prompt_template.replace("{tags}", tags_str)
            content = _xai_build_user_content(
                user_prompt=user_prompt,
                image_path=img_path,
                extra_image_paths=extras,
                include_images=include_images,
            )
            return req_id, img_abs, rel_p, content

        def _mark_submitted(sub_batch: List[Dict]) -> None:
            nonlocal submitted_now
            for item in sub_batch:
                req_id = item["batch_request_id"]
                req_state = request_map.setdefault(req_id, {})
                req_state["state"] = "submitted"
                if item.get("_images_stripped"):
                    req_state["image_payload"] = "stripped_due_to_413"
            submitted_now += len(sub_batch)
            state["submitted_at"] = time.time()
            _save_xai_state(state_file, state)
            if submit_pbar is not None:
                submit_pbar.set_postfix(
                    submitted=submitted_now,
                    skipped=skipped_payload_too_large,
                    failed=failed_submit_requests,
                )

        def _mark_payload_too_large(item: Dict) -> None:
            nonlocal skipped_payload_too_large
            req_id = item["batch_request_id"]
            req_state = request_map.setdefault(req_id, {})
            req_state["state"] = "failed_payload_too_large"
            req_state["error"] = "xai_413_payload_too_large"
            req_state["updated_at"] = time.time()
            skipped_payload_too_large += 1
            _save_xai_state(state_file, state)
            if submit_pbar is not None:
                submit_pbar.set_postfix(
                    submitted=submitted_now,
                    skipped=skipped_payload_too_large,
                    failed=failed_submit_requests,
                )

        def _mark_failed_request(item: Dict, status: Optional[int], message: str) -> None:
            nonlocal failed_submit_requests
            req_id = item["batch_request_id"]
            req_state = request_map.setdefault(req_id, {})
            req_state["state"] = "failed_submit"
            req_state["error"] = f"xai_http_{status or 'unknown'}"
            req_state["error_message"] = message[:1000]
            req_state["updated_at"] = time.time()
            failed_submit_requests += 1
            _save_xai_state(state_file, state)
            if submit_pbar is not None:
                submit_pbar.set_postfix(
                    submitted=submitted_now,
                    skipped=skipped_payload_too_large,
                    failed=failed_submit_requests,
                )

        def _post_sub_batch(sub_batch: List[Dict]) -> None:
            # Respect xAI team-level add-requests rolling limit (100 calls / 30s).
            now = time.monotonic()
            while add_call_timestamps and now - add_call_timestamps[0] >= rate_limit_window:
                add_call_timestamps.popleft()
            if len(add_call_timestamps) >= max_add_calls_per_window:
                wait = rate_limit_window - (now - add_call_timestamps[0]) + 0.05
                if wait > 0:
                    logger.info(
                        "throttling xAI add-requests calls to respect rolling limit: sleeping %.2fs",
                        wait,
                    )
                    time.sleep(wait)
                    now = time.monotonic()
                    while add_call_timestamps and now - add_call_timestamps[0] >= rate_limit_window:
                        add_call_timestamps.popleft()

            payload = {"batch_requests": sub_batch}
            payload_bytes = _json_byte_size(payload)
            if payload_bytes > XAI_BATCH_MAX_ADD_PAYLOAD_BYTES:
                fake_resp = requests.Response()
                fake_resp.status_code = 413
                raise requests.exceptions.HTTPError(
                    f"local payload too large before POST: {payload_bytes} bytes > "
                    f"{XAI_BATCH_MAX_ADD_PAYLOAD_BYTES} bytes",
                    response=fake_resp,
                )
            _xai_request(
                "POST",
                f"{args.xai_api_base_url}/v1/batches/{batch_id}/requests",
                headers,
                payload=payload,
                timeout=300,
            )
            add_call_timestamps.append(time.monotonic())

        def _flush_sub_batch(sub_batch: List[Dict]) -> None:
            """Submit sub-batch with adaptive fallback for payload and client errors."""
            nonlocal batch_id, rotated_batch_after_forbidden

            if not sub_batch:
                return

            status: Optional[int] = None
            err_text = ""
            try:
                _post_sub_batch(sub_batch)
                _mark_submitted(sub_batch)
                return
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                err_text = ((e.response.text or "") if e.response is not None else str(e)).strip()

            # If this state file points to an inaccessible batch from another key/team, rotate once.
            if status in (403, 404) and submitted_now == 0 and not rotated_batch_after_forbidden:
                logger.warning(
                    "received HTTP %s while adding to batch_id=%s before any successful submit. "
                    "creating a fresh batch and retrying current sub-batch once.",
                    status,
                    batch_id,
                )
                batch_id = _create_xai_batch(reason=f"auto_rotate_http_{status}")
                rotated_batch_after_forbidden = True
                try:
                    _post_sub_batch(sub_batch)
                    _mark_submitted(sub_batch)
                    return
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    err_text = ((e.response.text or "") if e.response is not None else str(e)).strip()

            # For payload/content/validation failures on mixed sub-batches, split and isolate.
            if status in (400, 403, 404, 413, 422) and len(sub_batch) > 1:
                mid = len(sub_batch) // 2
                logger.warning(
                    "xAI HTTP %s on sub-batch of %d requests; splitting into %d + %d and retrying.",
                    status,
                    len(sub_batch),
                    mid,
                    len(sub_batch) - mid,
                )
                _flush_sub_batch(sub_batch[:mid])
                _flush_sub_batch(sub_batch[mid:])
                return

            single = sub_batch[0]
            req_id = single["batch_request_id"]

            # For single-request failures, one fallback is to remove images and retry once.
            if include_images and status in (400, 403, 413, 422) and _xai_strip_images_from_batch_request(single):
                logger.warning(
                    "xAI HTTP %s on single request %s; retrying once with images removed for this request.",
                    status,
                    req_id,
                )
                try:
                    _post_sub_batch([single])
                    _mark_submitted([single])
                    return
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    err_text = ((e.response.text or "") if e.response is not None else str(e)).strip()

            if status == 401:
                raise RuntimeError(
                    "xAI API returned 401 Unauthorized while submitting batch requests. "
                    "Verify XAI_API_KEY and retry."
                )

            if status in (403, 404) and submitted_now == 0:
                body_preview = err_text[:600] if err_text else "<empty body>"
                raise RuntimeError(
                    "xAI API refused batch submission before any request was accepted "
                    f"(HTTP {status}). Response: {body_preview}"
                )

            if status == 413:
                logger.error(
                    "xAI 413 persists for request %s even after payload reduction; marking as failed.",
                    req_id,
                )
                _mark_payload_too_large(single)
                return

            body_preview = err_text[:600] if err_text else "<empty body>"
            logger.error(
                "xAI HTTP %s on request %s; marking request as failed and continuing. response=%s",
                status,
                req_id,
                body_preview,
            )
            _mark_failed_request(single, status, body_preview)

        try:
            for chunk_start in range(0, len(to_process), chunk_size):
                chunk = to_process[chunk_start : chunk_start + chunk_size]

                if include_images and len(chunk) > 1:
                    encoded = []
                    with ThreadPoolExecutor(max_workers=encode_workers) as ex:
                        futures = [ex.submit(_encode_item, item) for item in chunk]
                        try:
                            for future in as_completed(futures):
                                encoded.append(future.result())
                        except KeyboardInterrupt:
                            for future in futures:
                                future.cancel()
                            ex.shutdown(wait=False, cancel_futures=True)
                            logger.warning("interrupted by user while encoding xai batch payloads.")
                            raise
                else:
                    encoded = [_encode_item(item) for item in chunk]

                # Size-aware flushing: accumulate items until payload limit, then POST
                sub_batch: List[Dict] = []
                sub_batch_bytes = 0

                for req_id, img_abs, rel_p, user_content in encoded:
                    request_body = {
                        "chat_get_completion": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            "response_format": {"type": "json_object"},
                            "reasoning_effort": "none",
                        }
                    }
                    item_bytes = _estimate_request_bytes(request_body)

                    # Flush before adding if this item would push us over the limit
                    if sub_batch and sub_batch_bytes + item_bytes > MAX_PAYLOAD_BYTES:
                        _flush_sub_batch(sub_batch)
                        sub_batch = []
                        sub_batch_bytes = 0

                    sub_batch.append({
                        "batch_request_id": req_id,
                        "batch_request": request_body,
                    })
                    sub_batch_bytes += item_bytes
                    request_map[req_id] = {
                        "image_path": img_abs,
                        "image_path_rel": rel_p if rel_p and not rel_p.startswith("..") else None,
                        "state": "queued_for_submission",
                    }

                if sub_batch:
                    _flush_sub_batch(sub_batch)

                if submit_pbar is not None:
                    submit_pbar.update(len(chunk))
        except KeyboardInterrupt:
            _save_xai_state(state_file, state)
            logger.warning(
                "xai batch submit interrupted by user (Ctrl+C). "
                f"state saved to {state_file}; you can resume with submit/status/collect."
            )
            raise
        finally:
            if submit_pbar is not None:
                submit_pbar.close()

        logger.info(
            f"xai batch submit finished: batch_id={batch_id} submitted_now={submitted_now} "
            f"skipped_payload_too_large={skipped_payload_too_large} "
            f"failed_submit_requests={failed_submit_requests} "
            f"total_tracked={len(request_map)} state_file={state_file}"
        )
        return {"mode": "xai-batch-submit", "completed_paths": []}

    if action == "collect":
        # Check batch status first so we know totals and can show real progress
        batch_meta = _xai_request("GET", f"{args.xai_api_base_url}/v1/batches/{batch_id}", headers)
        counters = batch_meta.get("state", {})
        total_requests = int(counters.get("num_requests", 0) or 0)
        pending_remote = int(counters.get("num_pending", 0) or 0)
        success_remote = int(counters.get("num_success", 0) or 0)
        error_remote = int(counters.get("num_error", 0) or 0)
        done_remote = success_remote + error_remote
        pct_done = (done_remote / total_requests * 100.0) if total_requests else 0.0

        logger.info(
            f"batch status before collect: total={total_requests} done={done_remote} ({pct_done:.1f}%) "
            f"pending={pending_remote} success={success_remote} error={error_remote}"
        )
        if pending_remote > 0:
            logger.warning(
                f"{pending_remote} requests still pending — collecting {done_remote} completed results. "
                "Run collect again later to retrieve remaining."
            )

        request_map = state.get("request_map", {})
        completed_paths: List[str] = []
        success_count = 0
        error_count = 0
        pagination_token = None
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        use_pbar = not getattr(args, "no_progress", False)
        pbar = tqdm(total=done_remote or None, desc="xai-batch collect", smoothing=0.0, unit="result") if use_pbar else None

        while True:
            params = {"page_size": args.xai_batch_page_size}
            if pagination_token:
                params["pagination_token"] = pagination_token
            page = _xai_request(
                "GET",
                f"{args.xai_api_base_url}/v1/batches/{batch_id}/results",
                headers,
                params=params,
                timeout=180,
            )
            results = page.get("results", [])
            if not results:
                break

            for item in results:
                req_id = item.get("batch_request_id")
                if not req_id:
                    continue
                meta = request_map.get(req_id, {})
                image_path = _resolve_collect_image_path(meta, args.train_data_dir)
                if not image_path:
                    logger.warning(
                        f"could not resolve local image path for batch_request_id={req_id}. "
                        "Ensure dataset is extracted with same folder structure before collect."
                    )
                    continue

                caption = _xai_extract_caption_from_result(item)
                if caption:
                    if image_path not in result_map:
                        result_map[image_path] = []
                    add_tags_to_map(result_map, image_path, [caption], dedupe)
                    request_map[req_id]["state"] = "succeeded"
                    completed_paths.append(image_path)
                    success_count += 1
                else:
                    request_map[req_id]["state"] = "failed"
                    request_map[req_id]["error_message"] = item.get("error_message") or item.get("error") or "unknown"
                    error_count += 1

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ok=success_count, err=error_count)

                response_obj = item.get("response") or {}
                usage = response_obj.get("usage") if isinstance(response_obj, dict) else {}
                if isinstance(usage, dict):
                    usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                    usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                    usage_totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

            pagination_token = page.get("pagination_token")
            state["last_collect_at"] = time.time()
            state["usage_totals_from_collected_results"] = usage_totals
            _save_xai_state(state_file, state)

            if not pagination_token:
                break

        if pbar is not None:
            pbar.close()

        logger.info(
            f"xai batch collect finished: "
            f"collected={success_count} errors={error_count} pending_on_server={pending_remote} "
            f"batch_id={batch_id} state_file={state_file}"
        )
        if pending_remote > 0:
            logger.warning(
                f"{pending_remote} requests still pending on xAI. "
                "Run collect again when they complete to get remaining .txt files."
            )
        if usage_totals["total_tokens"]:
            logger.info(
                f"token usage: prompt={usage_totals['prompt_tokens']} "
                f"completion={usage_totals['completion_tokens']} "
                f"total={usage_totals['total_tokens']}"
            )
        return {"mode": "xai-batch-collect", "completed_paths": sorted(set(completed_paths))}

    raise ValueError(f"unknown xai batch action: {action}")


def run_grok(
    paths: List[str],
    args,
    dedupe: bool,
    result_map: Dict[str, List[str]],
    existing_tags: Optional[Dict[str, List[str]]] = None,
    extra_frames: Optional[Dict[str, List[str]]] = None,
    progress=None,
) -> None:
    """Run grok tagger via OpenRouter API with concurrent batch processing.

    Args:
        paths: list of image paths to process
        args: CLI arguments
        dedupe: whether to deduplicate tags
        result_map: output map of image_path -> tags
        existing_tags: tags from previous taggers (pixai/wd14/camie) per image
        extra_frames: additional frame paths per image (for pro mode)
        progress: tqdm progress bar
    """
    if args.grok_provider == "xai-batch":
        return run_grok_xai_batch(
            paths=paths,
            args=args,
            dedupe=dedupe,
            result_map=result_map,
            existing_tags=existing_tags or {},
            extra_frames=extra_frames or {},
        )

    api_key = args.grok_api_key
    if not api_key:
        raise ValueError(
            "OpenRouter API key is required for grok tagger. "
            "Set OPENROUTER_API_KEY env var or pass --grok_api_key."
        )

    model = args.grok_model
    system_prompt = get_system_prompt(args)
    user_prompt_template = get_user_prompt_template(args)
    max_workers = args.grok_concurrency
    tag_category_lookup = _load_grok_tag_category_lookup(args)

    existing_tags = existing_tags or {}
    extra_frames = extra_frames or {}

    pbar = None
    if progress is None and not args.no_progress:
        pbar = tqdm(total=len(paths), smoothing=0.0, desc="grok")

    futures = {}
    completed_paths: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_path in paths:
            tags_list = existing_tags.get(image_path, [])
            prompt_tags = _format_grok_tags_with_categories(tags_list, tag_category_lookup)
            tags_str = args.caption_separator.join(prompt_tags) if prompt_tags else "(no prior tags)"
            extras = extra_frames.get(image_path)

            future = executor.submit(
                _grok_single_task,
                api_key,
                model,
                system_prompt,
                user_prompt_template,
                image_path,
                tags_str,
                extras,
            )
            futures[future] = image_path

        for future in as_completed(futures):
            try:
                image_path, caption = future.result()
            except Exception as e:
                image_path = futures[future]
                logger.error(f"Grok task failed for {image_path}: {e}")
                caption = None
            if caption:
                add_tags_to_map(result_map, image_path, [caption], dedupe)
                completed_paths.append(image_path)
            else:
                logger.warning(f"No caption returned for {image_path}")

            if pbar is not None:
                pbar.update(1)
            if progress is not None:
                progress.update(1)

    if pbar is not None:
        pbar.close()
    return {"mode": "realtime", "completed_paths": completed_paths}


# -------------------------
# Video processing
# -------------------------

def extract_video_frames(
    video_paths: List[str],
    temp_dir: str,
    frame_number: int = 12,
    pro_mode: bool = False,
    pro_frames: Tuple[int, int] = (6, 30),
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Extract frames from videos for processing.

    Returns:
        frame_to_video: mapping of primary temp_frame_path -> original_video_path
        extra_frames_map: mapping of primary temp_frame_path -> [extra_frame_paths] (pro mode only)
    """
    frame_to_video: Dict[str, str] = {}
    extra_frames_map: Dict[str, List[str]] = {}

    for vpath in tqdm(video_paths, desc="extracting frames", smoothing=0.0):
        video_stem = Path(vpath).stem
        uid = hashlib.md5(vpath.encode()).hexdigest()[:10]

        if pro_mode:
            frames = extract_frames(vpath, list(pro_frames))
            primary = frames[0]
            secondary = frames[1] if len(frames) > 1 else None

            # If primary failed, try secondary as primary
            if primary is None and secondary is not None:
                primary = secondary
                secondary = None

            if primary is None:
                logger.warning(f"Skipping video (no frames extracted): {vpath}")
                continue

            primary_path = os.path.join(temp_dir, f"{video_stem}_{uid}_f{pro_frames[0]}.jpg")
            primary.save(primary_path, "JPEG", quality=95)
            frame_to_video[primary_path] = vpath

            if secondary is not None:
                secondary_path = os.path.join(temp_dir, f"{video_stem}_{uid}_f{pro_frames[1]}.jpg")
                secondary.save(secondary_path, "JPEG", quality=95)
                extra_frames_map[primary_path] = [secondary_path]
        else:
            frame = extract_frame(vpath, frame_number)
            if frame is None:
                logger.warning(f"Skipping video (no frame extracted): {vpath}")
                continue

            frame_path = os.path.join(temp_dir, f"{video_stem}_{uid}_f{frame_number}.jpg")
            frame.save(frame_path, "JPEG", quality=95)
            frame_to_video[frame_path] = vpath

    return frame_to_video, extra_frames_map


# -------------------------
# Shared batch loading
# -------------------------

def batch_loader(
    paths: List[str],
    batch_size: int,
    max_workers: int,
    preprocess_fn,
) -> Iterable[Tuple[List[str], np.ndarray]]:
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    for batch in batches:
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                loaded = list(ex.map(load_image, batch, [preprocess_fn] * len(batch)))
        else:
            loaded = [load_image(p, preprocess_fn) for p in batch]

        if not loaded:
            yield [], np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            continue

        paths_b, images_b = zip(*loaded)
        images_np = np.stack(images_b).astype(np.float32)
        yield list(paths_b), images_np


def load_image(path: str, preprocess_fn) -> Tuple[str, np.ndarray]:
    with Image.open(path) as img:
        arr = preprocess_fn(img)
    return path, arr


# -------------------------
# Output helpers
# -------------------------

def write_json_output(output_path: str, combined: Dict[str, List[str]], sep: str) -> None:
    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump({k: sep.join(v) for k, v in combined.items()}, f, ensure_ascii=False, indent=4)
    logger.info(f"captions saved to {output_path}")


def write_jsonl_output(output_path: str, combined: Dict[str, List[str]], sep: str) -> None:
    with open(output_path, "wt", encoding="utf-8") as f:
        for image_path, tags in combined.items():
            f.write(json.dumps({"image_path": image_path, "caption": sep.join(tags)}) + "\n")
    logger.info(f"captions saved to {output_path}")


# -------------------------
# Processing log
# -------------------------

TAGGER_LOG_FILE = ".tagger_log.json"


def load_processing_log(base_dir: str) -> Dict:
    """Load the processing log from the base directory."""
    log_path = os.path.join(base_dir, TAGGER_LOG_FILE)
    if os.path.exists(log_path):
        try:
            with open(log_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read processing log: {e}")
    return {"processed": {}}


def save_processing_log(base_dir: str, log_data: Dict) -> None:
    """Save the processing log to the base directory."""
    log_path = os.path.join(base_dir, TAGGER_LOG_FILE)
    try:
        with open(log_path, "wt", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"Could not save processing log: {e}")


def filter_already_processed(
    file_paths: List[str],
    log_data: Dict,
    taggers: List[str],
) -> Tuple[List[str], int]:
    """Filter out files that have already been processed with the same taggers.

    Returns (paths_to_process, skipped_count).
    """
    processed = log_data.get("processed", {})
    to_process = []
    skipped = 0

    taggers_set = set(taggers)

    for fpath in file_paths:
        key = os.path.abspath(fpath)
        entry = processed.get(key)
        if entry and set(entry.get("taggers", [])) >= taggers_set:
            skipped += 1
        else:
            to_process.append(fpath)

    return to_process, skipped


def update_processing_log(
    log_data: Dict,
    file_paths: List[str],
    taggers: List[str],
) -> None:
    """Mark files as processed in the log."""
    import datetime

    processed = log_data.setdefault("processed", {})
    timestamp = datetime.datetime.now().isoformat()

    for fpath in file_paths:
        key = os.path.abspath(fpath)
        existing = processed.get(key, {})
        existing_taggers = set(existing.get("taggers", []))
        existing_taggers.update(taggers)
        processed[key] = {
            "taggers": sorted(existing_taggers),
            "timestamp": timestamp,
        }


def write_caption_files(combined: Dict[str, List[str]], args) -> None:
    for image_path, tags in combined.items():
        if not tags:
            continue
        caption_file = os.path.splitext(image_path)[0] + args.caption_extension
        tag_text = args.caption_separator.join(tags)

        if args.append_tags and os.path.exists(caption_file):
            with open(caption_file, "rt", encoding="utf-8") as f:
                existing = [
                    t.strip()
                    for t in f.read().strip("\n").split(args.caption_separator.strip())
                    if t.strip()
                ]
            new_tags = [t for t in tags if t not in existing]
            tag_text = args.caption_separator.join(existing + new_tags)

        with open(caption_file, "wt", encoding="utf-8") as f:
            f.write(tag_text + "\n")


def _first_set(*values) -> float:
    """Return the first non-None value from the arguments."""
    for v in values:
        if v is not None:
            return v
    return 0.35


def count_batches(num_images: int, batch_size: int) -> int:
    if batch_size <= 0:
        return 0
    return (num_images + batch_size - 1) // batch_size


def recommend_batch_by_vram() -> Optional[int]:
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        free_mb = int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None

    if free_mb >= 30000:
        return 16
    if free_mb >= 20000:
        return 8
    if free_mb >= 12000:
        return 4
    return 2


# -------------------------
# Main
# -------------------------

def main(args):
    if not args.hf_token:
        args.hf_token = (
            os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    if not args.grok_api_key:
        args.grok_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.xai_api_key:
        args.xai_api_key = os.environ.get("XAI_API_KEY", "")

    args.general_threshold = args.general_threshold if args.general_threshold is not None else args.thresh
    args.character_threshold = args.character_threshold if args.character_threshold is not None else args.thresh

    if args.one_tagger:
        taggers = [args.one_tagger]
    else:
        taggers = [t.strip() for t in args.taggers.split(",") if t.strip()]

    # Lightweight status check: no dataset scan needed.
    if (
        args.grok_provider == "xai-batch"
        and "grok" in taggers
        and args.xai_batch_action == "status"
    ):
        run_grok_xai_batch(
            paths=[],
            args=args,
            dedupe=not args.no_dedupe,
            result_map={},
            existing_tags={},
            extra_frames={},
        )
        return

    # Video mode setup
    video_mode = args.video
    pro_mode = args.pro
    frame_to_video: Dict[str, str] = {}
    extra_frames_map: Dict[str, List[str]] = {}
    temp_dir_obj = None

    # In pro mode, also include extra frame paths for tagger processing
    pro_extra_paths: List[str] = []

    # Load processing log for skip logic
    base_dir = os.path.abspath(args.train_data_dir)
    processing_log = load_processing_log(base_dir)

    if args.smoke_test_image:
        if not os.path.exists(args.smoke_test_image):
            raise FileNotFoundError(f"smoke_test_image not found: {args.smoke_test_image}")
        paths = [args.smoke_test_image]
        if not args.one_tagger and not args.taggers:
            taggers = ["wd14", "camie", "pixai"]
        logger.info(f"smoke test on image: {args.smoke_test_image}")
    elif video_mode:
        video_paths_list = glob_videos_pathlib(Path(args.train_data_dir), args.recursive)
        logger.info(f"found {len(video_paths_list)} videos")
        if not video_paths_list:
            logger.warning("No videos found in the specified directory.")
            return
        video_str_paths = [str(p) for p in video_paths_list]

        # Skip already-processed videos unless --force
        if not args.force:
            video_str_paths, skipped = filter_already_processed(video_str_paths, processing_log, taggers)
            if skipped:
                logger.info(f"skipping {skipped} already-processed videos (use --force to reprocess)")
            if not video_str_paths:
                logger.info("all videos already processed. nothing to do.")
                return

        temp_dir_obj = tempfile.TemporaryDirectory(prefix="tagger_frames_")

        frame_number = args.frame_number
        pro_frames = (args.pro_frame_a, args.pro_frame_b)

        logger.info(f"extracting frames from {len(video_str_paths)} videos (pro={pro_mode})...")
        frame_to_video, extra_frames_map = extract_video_frames(
            video_str_paths,
            temp_dir_obj.name,
            frame_number=frame_number,
            pro_mode=pro_mode,
            pro_frames=pro_frames,
        )
        if not frame_to_video:
            logger.error("Could not extract frames from any video.")
            temp_dir_obj.cleanup()
            return
        logger.info(f"extracted frames for {len(frame_to_video)} videos")
        paths = list(frame_to_video.keys())

        # In pro mode, collect extra frame paths so taggers process both frames
        if pro_mode:
            for extras in extra_frames_map.values():
                pro_extra_paths.extend(extras)
    else:
        image_paths = glob_images_pathlib(Path(args.train_data_dir), args.recursive)
        logger.info(f"found {len(image_paths)} images")
        all_image_paths = [str(p) for p in image_paths]

        # Skip already-processed images unless --force
        if not args.force:
            all_image_paths, skipped = filter_already_processed(all_image_paths, processing_log, taggers)
            if skipped:
                logger.info(f"skipping {skipped} already-processed images (use --force to reprocess)")
            if not all_image_paths:
                logger.info("all images already processed. nothing to do.")
                return

        paths = all_image_paths

    # For taggers: in pro mode, process both primary + extra frames
    tagger_paths = paths + pro_extra_paths
    combined: Dict[str, List[str]] = {p: [] for p in tagger_paths}

    if args.suggest_batch:
        suggestion = recommend_batch_by_vram()
        if suggestion is not None:
            logger.info(f"Suggested batch_size based on free VRAM: {suggestion}")

    # Load existing .txt files and merge with new tags
    if video_mode:
        # In video mode: check for existing .txt next to the video files
        existing_count = 0
        for frame_path, video_path in frame_to_video.items():
            caption_file = os.path.splitext(video_path)[0] + args.caption_extension
            if not os.path.exists(caption_file):
                continue
            with open(caption_file, "rt", encoding="utf-8") as f:
                existing = [
                    t.strip()
                    for t in f.read().strip("\n").split(args.caption_separator.strip())
                    if t.strip()
                ]
            if existing:
                add_tags_to_map(combined, frame_path, existing, not args.no_dedupe)
                existing_count += 1
        if existing_count:
            logger.info(f"loaded existing tags from {existing_count} .txt files")
    elif args.append_tags or args.grok_context_from_existing:
        existing_count = 0
        for image_path in paths:
            caption_file = os.path.splitext(image_path)[0] + args.caption_extension
            if not os.path.exists(caption_file):
                continue
            with open(caption_file, "rt", encoding="utf-8") as f:
                existing = [
                    t.strip()
                    for t in f.read().strip("\n").split(args.caption_separator.strip())
                    if t.strip()
                ]
            if existing:
                add_tags_to_map(combined, image_path, existing, not args.no_dedupe)
                existing_count += 1
        if existing_count:
            logger.info(f"loaded existing tags from {existing_count} .txt files as grok context")

    # Separate taggers: booru taggers run first, grok runs last (needs booru output)
    booru_taggers = [t for t in taggers if t != "grok"]
    has_grok = "grok" in taggers

    total_batches = 0
    if not args.no_progress:
        for tagger in booru_taggers:
            bs = args.batch_size
            total_batches += count_batches(len(tagger_paths), bs)
        overall = tqdm(total=total_batches, desc="taggers", smoothing=0.0) if total_batches > 0 else None
    else:
        overall = None

    for tagger in booru_taggers:
        if tagger == "wd14":
            general_threshold = _first_set(args.wd14_general_threshold, args.wd14_thresh, args.general_threshold)
            character_threshold = _first_set(args.wd14_character_threshold, args.wd14_thresh, args.character_threshold)
            run_wd14(
                tagger_paths,
                args,
                args.wd14_repo_id,
                args.model_dir,
                args.batch_size,
                not args.no_dedupe,
                combined,
                general_threshold,
                character_threshold,
                overall,
            )
        elif tagger == "camie":
            general_threshold = _first_set(args.camie_general_threshold, args.camie_thresh, args.general_threshold)
            character_threshold = _first_set(args.camie_character_threshold, args.camie_thresh, args.character_threshold)
            run_camie(
                tagger_paths,
                args,
                args.camie_repo_id,
                args.model_dir,
                args.batch_size,
                not args.no_dedupe,
                combined,
                general_threshold,
                character_threshold,
                overall,
            )
        elif tagger == "pixai":
            general_threshold = _first_set(args.pixai_general_threshold, args.pixai_thresh, args.general_threshold)
            character_threshold = _first_set(args.pixai_character_threshold, args.pixai_thresh, args.character_threshold)
            run_pixai(
                tagger_paths,
                args,
                args.pixai_repo_id,
                args.model_dir,
                args.batch_size,
                not args.no_dedupe,
                combined,
                general_threshold,
                character_threshold,
                overall,
            )
        else:
            raise ValueError(f"Unknown tagger: {tagger}")

    if overall is not None:
        overall.close()

    # In pro mode: merge extra frame tags into primary frame (deduplicated)
    if pro_mode and extra_frames_map:
        logger.info("merging tags from multiple frames (pro mode)...")
        for primary_path, extra_paths in extra_frames_map.items():
            for ep in extra_paths:
                extra_tags = combined.get(ep, [])
                add_tags_to_map(combined, primary_path, extra_tags, dedupe=True)
                # Remove extra frame entry from combined
                combined.pop(ep, None)

    # Run grok AFTER booru taggers so it has access to their tags
    grok_completed_paths: Optional[List[str]] = None
    if has_grok:
        logger.info("running grok captioner with booru tags as context...")
        # Build existing_tags map for grok (only primary paths)
        existing_tags_for_grok = {p: combined.get(p, []) for p in paths}
        # Grok result goes into a separate map (it's a full caption, not tags)
        grok_combined: Dict[str, List[str]] = {p: [] for p in paths}
        grok_result = run_grok(
            paths,
            args,
            not args.no_dedupe,
            grok_combined,
            existing_tags=existing_tags_for_grok,
            extra_frames=extra_frames_map if pro_mode else None,
        )
        grok_completed_paths = (grok_result or {}).get("completed_paths", [])
        if args.grok_provider == "xai-batch" and args.xai_batch_action in ("submit", "status"):
            logger.info(
                "xai batch action completed without local caption writes "
                f"(action={args.xai_batch_action}). state saved for later collect."
            )
            return
        # Replace combined with grok output (grok caption is the final output)
        combined = grok_combined

    # In video mode, remap frame paths to original video paths for output
    if video_mode and frame_to_video:
        video_combined: Dict[str, List[str]] = {}
        for frame_path, tags in combined.items():
            video_path = frame_to_video.get(frame_path, frame_path)
            video_combined[video_path] = tags
        combined = video_combined
        if grok_completed_paths:
            grok_completed_paths = [frame_to_video.get(p, p) for p in grok_completed_paths]

    if args.output_path:
        if args.output_path.endswith(".jsonl"):
            write_jsonl_output(args.output_path, combined, args.caption_separator)
        else:
            write_json_output(args.output_path, combined, args.caption_separator)
    else:
        write_caption_files(combined, args)

    # Update processing log with successfully processed files
    if has_grok and grok_completed_paths is not None:
        processed_files = sorted(set(grok_completed_paths))
    elif video_mode:
        processed_files = list(combined.keys())
    else:
        processed_files = paths
    update_processing_log(processing_log, processed_files, taggers)
    save_processing_log(base_dir, processing_log)
    logger.info(f"processing log updated ({len(processed_files)} files logged)")

    # Cleanup temp frames
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    logger.info("done")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, nargs="?", default=".", help="directory for images")

    parser.add_argument("--taggers", type=str, default="wd14", help="comma list: wd14,camie,pixai,grok")
    parser.add_argument("--one_tagger", type=str, choices=["wd14", "camie", "pixai", "grok"], default=None)
    parser.add_argument("--video", action="store_true", help="video mode: extract frames from videos and tag them")
    parser.add_argument("--pro", action="store_true", help="pro mode: use 2 frames per video (better quality, 2x tagger cost)")
    parser.add_argument("--frame_number", type=int, default=12, help="which frame to extract in normal video mode (default: 12)")
    parser.add_argument("--pro_frame_a", type=int, default=6, help="first frame number for pro mode (default: 6)")
    parser.add_argument("--pro_frame_b", type=int, default=30, help="second frame number for pro mode (default: 30)")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--smoke_test_image", type=str, default=None)

    parser.add_argument("--wd14_repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO)
    parser.add_argument("--camie_repo_id", type=str, default=DEFAULT_CAMIE_REPO)
    parser.add_argument("--pixai_repo_id", type=str, default=DEFAULT_PIXAI_REPO)

    parser.add_argument("--wd14_thresh", type=float, default=None)
    parser.add_argument("--camie_thresh", type=float, default=None)
    parser.add_argument("--pixai_thresh", type=float, default=None)
    parser.add_argument("--wd14_general_threshold", type=float, default=None)
    parser.add_argument("--wd14_character_threshold", type=float, default=None)
    parser.add_argument("--camie_general_threshold", type=float, default=None)
    parser.add_argument("--camie_character_threshold", type=float, default=None)
    parser.add_argument("--pixai_general_threshold", type=float, default=None)
    parser.add_argument("--pixai_character_threshold", type=float, default=None)
    parser.add_argument("--camie_min_confidence", type=float, default=0.1)
    parser.add_argument("--camie_category_thresholds_file", type=str, default=None)
    parser.add_argument("--pixai_category_thresholds_file", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--suggest_batch", action="store_true")

    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--caption_extension", type=str, default=".txt")
    parser.add_argument("--caption_separator", type=str, default=", ")

    parser.add_argument("--thresh", type=float, default=0.35)
    parser.add_argument("--general_threshold", type=float, default=None)
    parser.add_argument("--character_threshold", type=float, default=None)

    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--remove_underscore", action="store_true")
    parser.add_argument("--append_tags", action="store_true")

    parser.add_argument("--use_rating_tags", action="store_true")
    parser.add_argument("--use_rating_tags_as_last_tag", action="store_true")
    parser.add_argument("--character_tags_first", action="store_true")
    parser.add_argument("--always_first_tags", type=str, default=None)

    parser.add_argument("--tag_replacement", type=str, default=None)
    parser.add_argument("--character_tag_expand", action="store_true")
    parser.add_argument("--undesired_tags", type=str, default="")

    parser.add_argument("--force", action="store_true", help="reprocess all files even if already in the processing log")
    parser.add_argument("--no_dedupe", action="store_true", help="disable de-duplication (not recommended)")
    parser.add_argument("--pixai_mode", type=str, choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--pixai_topk_general", type=int, default=25)
    parser.add_argument("--pixai_topk_character", type=int, default=10)
    parser.add_argument("--pixai_no_ip", action="store_true")
    parser.add_argument("--pixai_device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models")

    # Grok (OpenRouter) options
    parser.add_argument("--grok_api_key", type=str, default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--grok_provider", type=str, choices=["openrouter", "xai-batch"], default="openrouter")
    parser.add_argument("--grok_model", type=str, default=DEFAULT_GROK_MODEL, help="OpenRouter model ID")
    parser.add_argument("--grok_system_prompt_file", type=str, default=None, help="path to system prompt .md file")
    parser.add_argument("--grok_prompt_file", type=str, default=None, help="path to user prompt template .md file")
    parser.add_argument(
        "--grok_tag_category_metadata_file",
        type=str,
        default=None,
        help="optional path to camie metadata JSON used to annotate tags for grok prompt context",
    )
    parser.add_argument("--grok_concurrency", type=int, default=8, help="max concurrent API calls for grok (default: 8)")
    parser.add_argument("--xai_api_key", type=str, default=None, help="xAI API key (or set XAI_API_KEY env)")
    parser.add_argument("--xai_api_base_url", type=str, default=XAI_API_BASE_URL, help="xAI API base URL")
    parser.add_argument("--xai_batch_action", type=str, choices=["submit", "status", "collect"], default="submit")
    parser.add_argument("--xai_batch_name", type=str, default=None, help="name for xAI batch when creating")
    parser.add_argument("--xai_batch_state_file", type=str, default=None, help="path to persisted xAI batch state JSON")
    parser.add_argument("--xai_batch_submit_chunk", type=int, default=1000, help="how many requests per add-to-batch API call")
    parser.add_argument("--xai_batch_page_size", type=int, default=100, help="page size when collecting xAI batch results")
    parser.add_argument("--xai_batch_no_image", action="store_true", help="send tags-only requests in xAI batch mode (faster/smaller)")
    parser.add_argument("--xai_batch_model", type=str, default=DEFAULT_XAI_BATCH_MODEL, help="model ID for xAI native batch API")
    parser.add_argument("--grok_context_from_existing", action="store_true",
                        help="load existing .txt files as context for grok prompt without affecting output writing")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C). exiting.")
        raise SystemExit(130)
