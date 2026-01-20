import argparse
import csv
import gc
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from wd14_utils import glob_images_pathlib, resize_image, setup_logging

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

def build_session(onnx_path: str) -> Tuple[ort.InferenceSession, str, Optional[int]]:
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
        source, target = pair.split(",")
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
    if not category_thresholds and not args.camie_thresh and not args.camie_general_threshold and not args.camie_character_threshold:
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


class PixAITaggingHead(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        return torch.sigmoid(logits)


def pixai_pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        image.load()
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    if image.mode == "P":
        return pixai_pil_to_rgb(image.convert("RGBA"))
    return image.convert("RGB")


def build_pixai_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_pixai_assets(model_location: str, repo_id: str, force: bool) -> Tuple[str, str, str]:
    if not os.path.exists(model_location) or force:
        os.makedirs(model_location, exist_ok=True)
        logger.info(f"downloading PixAI model from HF: {repo_id}")
        for file in [
            PIXAI_PTH_FILE,
            PIXAI_TAGS_JSON_FILE,
            PIXAI_CHAR_IP_MAP_FILE,
            PIXAI_CATEGORY_THRESHOLDS_FILE,
        ]:
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=model_location, force_download=True)
            except Exception:
                pass

    weights_path = os.path.join(model_location, PIXAI_PTH_FILE)
    tags_path = os.path.join(model_location, PIXAI_TAGS_JSON_FILE)
    ip_map_path = os.path.join(model_location, PIXAI_CHAR_IP_MAP_FILE)
    return weights_path, tags_path, ip_map_path


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


def build_pixai_model(weights_path: str, device: str, num_classes: int) -> torch.nn.Module:
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
) -> Iterable[Tuple[List[str], torch.Tensor]]:
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    def _load(path: str) -> Tuple[str, torch.Tensor]:
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
    weights_path, tags_path, ip_map_path = load_pixai_assets(model_location, repo_id, args.force_download)

    index_to_tag, gen_count, char_count, total_tags = load_pixai_tag_map(tags_path)
    char_ip_map = load_pixai_char_ip_map(ip_map_path)

    category_thresholds = load_pixai_category_thresholds(model_location, args.pixai_category_thresholds_file)
    if not args.pixai_general_threshold and not args.pixai_thresh:
        general_override = category_thresholds.get("general") or category_thresholds.get("0")
        if general_override is not None:
            general_threshold = general_override
        else:
            general_threshold = 0.30
    if not args.pixai_character_threshold and not args.pixai_thresh:
        character_override = category_thresholds.get("character") or category_thresholds.get("4")
        if character_override is not None:
            character_threshold = character_override
        else:
            character_threshold = 0.85

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

        if progress is not None:
            progress.update(1)

    del model
    cleanup_memory()


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


def write_caption_files(combined: Dict[str, List[str]], args) -> None:
    for image_path, tags in combined.items():
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
    args.general_threshold = args.general_threshold if args.general_threshold is not None else args.thresh
    args.character_threshold = args.character_threshold if args.character_threshold is not None else args.thresh

    if args.one_tagger:
        taggers = [args.one_tagger]
    else:
        taggers = [t.strip() for t in args.taggers.split(",") if t.strip()]

    if args.smoke_test_image:
        if not os.path.exists(args.smoke_test_image):
            raise FileNotFoundError(f"smoke_test_image not found: {args.smoke_test_image}")
        paths = [args.smoke_test_image]
        if not args.one_tagger and not args.taggers:
            taggers = ["wd14", "camie", "pixai"]
        logger.info(f"smoke test on image: {args.smoke_test_image}")
    else:
        image_paths = glob_images_pathlib(Path(args.train_data_dir), args.recursive)
        logger.info(f"found {len(image_paths)} images")
        paths = [str(p) for p in image_paths]
    combined: Dict[str, List[str]] = {p: [] for p in paths}

    if args.suggest_batch:
        suggestion = recommend_batch_by_vram()
        if suggestion is not None:
            logger.info(f"Suggested batch_size based on free VRAM: {suggestion}")

    if args.append_tags:
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
            add_tags_to_map(combined, image_path, existing, not args.no_dedupe)

    total_batches = 0
    if not args.no_progress:
        for tagger in taggers:
            bs = args.batch_size
            total_batches += count_batches(len(paths), bs)
        overall = tqdm(total=total_batches, desc="total", smoothing=0.0) if total_batches > 0 else None
    else:
        overall = None

    for tagger in taggers:
        if tagger == "wd14":
            general_threshold = args.wd14_general_threshold or args.wd14_thresh or args.general_threshold
            character_threshold = args.wd14_character_threshold or args.wd14_thresh or args.character_threshold
            run_wd14(
                paths,
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
            general_threshold = args.camie_general_threshold or args.camie_thresh or args.general_threshold
            character_threshold = args.camie_character_threshold or args.camie_thresh or args.character_threshold
            run_camie(
                paths,
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
            general_threshold = args.pixai_general_threshold or args.pixai_thresh or args.general_threshold
            character_threshold = args.pixai_character_threshold or args.pixai_thresh or args.character_threshold
            run_pixai(
                paths,
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

    if args.output_path:
        if args.output_path.endswith(".jsonl"):
            write_jsonl_output(args.output_path, combined, args.caption_separator)
        else:
            write_json_output(args.output_path, combined, args.caption_separator)
    else:
        write_caption_files(combined, args)

    logger.info("done")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, nargs="?", default=".", help="directory for images")

    parser.add_argument("--taggers", type=str, default="wd14", help="comma list: wd14,camie,pixai")
    parser.add_argument("--one_tagger", type=str, choices=["wd14", "camie", "pixai"], default=None)
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

    parser.add_argument("--no_dedupe", action="store_true", help="disable de-duplication (not recommended)")
    parser.add_argument("--pixai_mode", type=str, choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--pixai_topk_general", type=int, default=25)
    parser.add_argument("--pixai_topk_character", type=int, default=10)
    parser.add_argument("--pixai_no_ip", action="store_true")
    parser.add_argument("--pixai_device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    main(args)
