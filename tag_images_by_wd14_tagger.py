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
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from wd14_utils import glob_images_pathlib, resize_image, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_SIZE = 448

DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
DEFAULT_CAMIE_REPO = "Camais03/camie-tagger-v2"
DEFAULT_PIXAI_REPO = "deepghs/pixai-tagger-v0.9-onnx"

WD14_CSV_FILE = "selected_tags.csv"
WD14_ONNX_NAME = "model.onnx"

CAMIE_ONNX_FILE = "camie-tagger-v2.onnx"
CAMIE_META_FILE = "camie-tagger-v2-metadata.json"

PIXAI_ONNX_FILE = "model.onnx"
PIXAI_TAGS_FILE = "selected_tags.csv"
PIXAI_CATEGORIES_FILE = "categories.json"
PIXAI_PREPROCESS_FILE = "preprocess.json"
PIXAI_THRESHOLDS_FILE = "thresholds.csv"


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


def preprocess_pixai(image: Image.Image, image_size: int, mean: List[float], std: List[float]) -> np.ndarray:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    img = np.array(image).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img = (img - mean_arr) / std_arr
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


def dedupe_tags(tags: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def add_tags_to_map(result_map: Dict[str, List[str]], image_path: str, tags: List[str], dedupe: bool) -> None:
    if dedupe:
        tags = dedupe_tags(tags)
    result_map[image_path] = tags


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

    for batch_paths, batch_imgs in tqdm(batches, smoothing=0.0, desc="wd14"):
        probs = session.run(None, {input_name: batch_imgs})[0]
        probs = probs[: len(batch_paths)]

        for image_path, prob in zip(batch_paths, probs):
            tags: List[str] = []
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tags.append(general_tags[i])
                elif i >= len(general_tags) and p >= args.character_threshold:
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


def run_camie(
    paths: List[str],
    args,
    repo_id: str,
    model_dir: str,
    batch_size: int,
    dedupe: bool,
    result_map: Dict[str, List[str]],
) -> None:
    model_location = os.path.join(model_dir, repo_id.replace("/", "_"))
    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"downloading Camie model from HF: {repo_id}")
        hf_hub_download(repo_id=repo_id, filename=CAMIE_ONNX_FILE, local_dir=model_location, force_download=True)
        hf_hub_download(repo_id=repo_id, filename=CAMIE_META_FILE, local_dir=model_location, force_download=True)

    onnx_path = os.path.join(model_location, CAMIE_ONNX_FILE)
    meta_path = os.path.join(model_location, CAMIE_META_FILE)

    idx_to_tag, tag_to_category, img_size = load_camie_metadata(meta_path)
    session, input_name, fixed_batch = build_session(onnx_path)
    if fixed_batch and fixed_batch > 0 and batch_size != fixed_batch:
        logger.warning(f"Camie batch {batch_size} != model batch {fixed_batch}; using {fixed_batch}")
        batch_size = fixed_batch

    batches = batch_loader(paths, batch_size, args.max_workers, lambda img: preprocess_imagenet(img, img_size))

    for batch_paths, batch_imgs in tqdm(batches, smoothing=0.0, desc="camie"):
        outputs = session.run(None, {input_name: batch_imgs})
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]
        probs = sigmoid(logits)

        for image_path, prob in zip(batch_paths, probs):
            tags: List[str] = []
            for idx, p in enumerate(prob):
                tag = idx_to_tag.get(idx)
                if tag is None:
                    continue
                category = tag_to_category.get(tag, "general")
                if category.lower() == "character":
                    if p >= args.character_threshold:
                        if args.character_tags_first:
                            tags.insert(0, tag)
                        else:
                            tags.append(tag)
                elif category.lower() == "rating":
                    if args.use_rating_tags:
                        tags.insert(0, tag)
                    elif args.use_rating_tags_as_last_tag:
                        tags.append(tag)
                else:
                    if p >= args.general_threshold:
                        tags.append(tag)

            tags = postprocess_tags(tags, args)
            add_tags_to_map(result_map, image_path, tags, dedupe)

    del session
    cleanup_memory()


# -------------------------
# PixAI tagger (ONNX)
# -------------------------

def load_pixai_tags(model_location: str) -> Tuple[List[str], Dict[str, str]]:
    categories: Dict[str, str] = {}
    categories_path = os.path.join(model_location, PIXAI_CATEGORIES_FILE)
    if os.path.exists(categories_path):
        with open(categories_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            categories = {k: v for k, v in data.items()}

    tags: List[str] = []
    tags_path = os.path.join(model_location, PIXAI_TAGS_FILE)
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if rows and rows[0][0] in {"tag_id", "id"}:
        rows = rows[1:]

    for row in rows:
        if not row:
            continue
        tags.append(row[1] if len(row) > 1 else row[0])

    return tags, categories


def load_pixai_preprocess(model_location: str) -> Tuple[int, List[float], List[float]]:
    image_size = IMAGE_SIZE
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    preprocess_path = os.path.join(model_location, PIXAI_PREPROCESS_FILE)
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("image_size", "img_size", "input_size", "size"):
            if key in data:
                value = data[key]
                if isinstance(value, list) and value:
                    image_size = int(value[0])
                elif isinstance(value, int):
                    image_size = int(value)
        if "mean" in data:
            mean = data["mean"]
        if "std" in data:
            std = data["std"]

    return image_size, mean, std


def load_pixai_thresholds(model_location: str) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    thresholds_path = os.path.join(model_location, PIXAI_THRESHOLDS_FILE)
    if not os.path.exists(thresholds_path):
        return thresholds

    with open(thresholds_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if not rows:
        return thresholds

    header = [c.lower() for c in rows[0]]
    if "tag" in header or "name" in header:
        tag_idx = header.index("tag") if "tag" in header else header.index("name")
        th_idx = header.index("threshold") if "threshold" in header else None
        for row in rows[1:]:
            if not row:
                continue
            tag = row[tag_idx]
            try:
                th = float(row[th_idx]) if th_idx is not None else float(row[-1])
            except Exception:
                continue
            thresholds[tag] = th
    else:
        for row in rows:
            if len(row) < 2:
                continue
            try:
                thresholds[row[0]] = float(row[1])
            except Exception:
                continue

    return thresholds


def run_pixai(
    paths: List[str],
    args,
    repo_id: str,
    model_dir: str,
    batch_size: int,
    dedupe: bool,
    result_map: Dict[str, List[str]],
) -> None:
    model_location = os.path.join(model_dir, repo_id.replace("/", "_"))
    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"downloading PixAI ONNX model from HF: {repo_id}")
        for file in [PIXAI_ONNX_FILE, PIXAI_TAGS_FILE, PIXAI_PREPROCESS_FILE, PIXAI_CATEGORIES_FILE, PIXAI_THRESHOLDS_FILE]:
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=model_location, force_download=True)
            except Exception:
                pass

    onnx_path = os.path.join(model_location, PIXAI_ONNX_FILE)
    session, input_name, fixed_batch = build_session(onnx_path)
    if fixed_batch and fixed_batch > 0 and batch_size != fixed_batch:
        logger.warning(f"PixAI batch {batch_size} != model batch {fixed_batch}; using {fixed_batch}")
        batch_size = fixed_batch

    tags, categories = load_pixai_tags(model_location)
    image_size, mean, std = load_pixai_preprocess(model_location)
    thresholds = load_pixai_thresholds(model_location) if args.pixai_use_thresholds else {}

    batches = batch_loader(paths, batch_size, args.max_workers, lambda img: preprocess_pixai(img, image_size, mean, std))

    for batch_paths, batch_imgs in tqdm(batches, smoothing=0.0, desc="pixai"):
        logits = session.run(None, {input_name: batch_imgs})[0]
        probs = sigmoid(logits)

        for image_path, prob in zip(batch_paths, probs):
            tags_out: List[str] = []
            for idx, p in enumerate(prob):
                if idx >= len(tags):
                    continue
                tag = tags[idx]
                category = categories.get(tag, "general") if categories else "general"

                th = thresholds.get(tag)
                if th is None:
                    if category.lower() == "character":
                        th = args.character_threshold
                    else:
                        th = args.general_threshold
                if p >= th:
                    if category.lower() == "character" and args.character_tags_first:
                        tags_out.insert(0, tag)
                    else:
                        tags_out.append(tag)

            tags_out = postprocess_tags(tags_out, args)
            add_tags_to_map(result_map, image_path, tags_out, dedupe)

    del session
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

def output_suffix_for_tagger(tagger: str) -> str:
    return f".{tagger}"


def write_json_outputs(output_path: str, per_tagger: Dict[str, Dict[str, List[str]]]) -> None:
    base, ext = os.path.splitext(output_path)
    for tagger, tag_map in per_tagger.items():
        path = f"{base}{output_suffix_for_tagger(tagger)}{ext or '.json'}"
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(tag_map, f, ensure_ascii=False, indent=4)
        logger.info(f"captions saved to {path}")


def write_jsonl_outputs(output_path: str, per_tagger: Dict[str, Dict[str, List[str]]], sep: str) -> None:
    base, _ = os.path.splitext(output_path)
    for tagger, tag_map in per_tagger.items():
        path = f"{base}{output_suffix_for_tagger(tagger)}.jsonl"
        with open(path, "wt", encoding="utf-8") as f:
            for image_path, tags in tag_map.items():
                f.write(json.dumps({"image_path": image_path, "caption": sep.join(tags)}) + "\n")
        logger.info(f"captions saved to {path}")


def write_caption_files(per_tagger: Dict[str, Dict[str, List[str]]], args) -> None:
    for tagger, tag_map in per_tagger.items():
        for image_path, tags in tag_map.items():
            caption_file = os.path.splitext(image_path)[0] + output_suffix_for_tagger(tagger) + args.caption_extension
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


# -------------------------
# Main
# -------------------------

def main(args):
    image_paths = glob_images_pathlib(Path(args.train_data_dir), args.recursive)
    logger.info(f"found {len(image_paths)} images")
    paths = [str(p) for p in image_paths]

    args.general_threshold = args.general_threshold if args.general_threshold is not None else args.thresh
    args.character_threshold = args.character_threshold if args.character_threshold is not None else args.thresh

    taggers = [t.strip() for t in args.taggers.split(",") if t.strip()]
    per_tagger: Dict[str, Dict[str, List[str]]] = {
        tagger: {p: [] for p in paths} for tagger in taggers
    }

    for tagger in taggers:
        if tagger == "wd14":
            run_wd14(paths, args, args.wd14_repo_id, args.model_dir, args.batch_size, args.dedupe, per_tagger[tagger])
        elif tagger == "camie":
            run_camie(paths, args, args.camie_repo_id, args.model_dir, args.batch_size, args.dedupe, per_tagger[tagger])
        elif tagger == "pixai":
            run_pixai(paths, args, args.pixai_repo_id, args.model_dir, args.batch_size, args.dedupe, per_tagger[tagger])
        else:
            raise ValueError(f"Unknown tagger: {tagger}")

    if args.output_path:
        if args.output_path.endswith(".jsonl"):
            write_jsonl_outputs(args.output_path, per_tagger, args.caption_separator)
        else:
            write_json_outputs(args.output_path, per_tagger)
    else:
        write_caption_files(per_tagger, args)

    logger.info("done")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for images")

    parser.add_argument("--taggers", type=str, default="wd14", help="comma list: wd14,camie,pixai")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--force_download", action="store_true")

    parser.add_argument("--wd14_repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO)
    parser.add_argument("--camie_repo_id", type=str, default=DEFAULT_CAMIE_REPO)
    parser.add_argument("--pixai_repo_id", type=str, default=DEFAULT_PIXAI_REPO)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=4)

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

    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--pixai_use_thresholds", action="store_true")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    main(args)
