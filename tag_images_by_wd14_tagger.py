import argparse
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
CSV_FILE = "selected_tags.csv"
TAG_JSON_FILE = "tag_mapping.json"
ONNX_DEFAULT_NAME = "model.onnx"
ONNX_SUBDIR_NAME = "model_optimized.onnx"


def preprocess_image(image: Image.Image) -> np.ndarray:
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


def download_model(args) -> Tuple[str, bool, Optional[str]]:
    tokens = args.repo_id.split("/")
    if len(tokens) > 2:
        repo_id = "/".join(tokens[:2])
        subdir = "/".join(tokens[2:])
        model_location = os.path.join(args.model_dir, repo_id.replace("/", "_"), subdir)
        onnx_name = ONNX_SUBDIR_NAME
        default_format = False
    else:
        repo_id = args.repo_id
        subdir = None
        model_location = os.path.join(args.model_dir, repo_id.replace("/", "_"))
        onnx_name = ONNX_DEFAULT_NAME
        default_format = True

    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from HF: {args.repo_id}")

        if subdir is None:
            files = [CSV_FILE, onnx_name]
            for file in files:
                hf_hub_download(repo_id=args.repo_id, filename=file, local_dir=model_location, force_download=True)
        else:
            files = [onnx_name, TAG_JSON_FILE]
            for file in files:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    subfolder=subdir,
                    local_dir=os.path.join(args.model_dir, repo_id.replace("/", "_")),
                    force_download=True,
                )
    else:
        logger.info("using existing wd14 tagger model")

    return model_location, default_format, onnx_name


def load_tags_default(model_location: str, args) -> Tuple[List[str], List[str], List[str]]:
    with open(os.path.join(model_location, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    header = rows[0]
    data = rows[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category"

    rating_tags = [row[1] for row in data if row[2] == "9"]
    general_tags = [row[1] for row in data if row[2] == "0"]
    character_tags = [row[1] for row in data if row[2] == "4"]

    if args.remove_underscore:
        rating_tags = [t.replace("_", " ") if len(t) > 3 else t for t in rating_tags]
        general_tags = [t.replace("_", " ") if len(t) > 3 else t for t in general_tags]
        character_tags = [t.replace("_", " ") if len(t) > 3 else t for t in character_tags]

    if args.tag_replacement:
        rating_tags = apply_tag_replacement(rating_tags, args.tag_replacement)
        general_tags = apply_tag_replacement(general_tags, args.tag_replacement)
        character_tags = apply_tag_replacement(character_tags, args.tag_replacement)

    if args.character_tag_expand:
        character_tags = expand_character_tags(character_tags, args.caption_separator)

    return rating_tags, general_tags, character_tags


def load_tags_mapping(model_location: str, args) -> Tuple[Dict[int, str], Dict[int, str]]:
    with open(os.path.join(model_location, TAG_JSON_FILE), "r", encoding="utf-8") as f:
        tag_mapping = json.load(f)

    id_to_tag: Dict[int, str] = {}
    id_to_category: Dict[int, str] = {}
    for tag_id, tag_info in tag_mapping.items():
        tag = tag_info["tag"]
        category = tag_info["category"]

        if args.remove_underscore:
            tag = tag.replace("_", " ") if len(tag) > 3 else tag
        if args.tag_replacement:
            tag = apply_tag_replacement([tag], args.tag_replacement)[0]
        if category == "Character" and args.character_tag_expand:
            tag = expand_character_tags([tag], args.caption_separator)[0]

        id_to_tag[int(tag_id)] = tag
        id_to_category[int(tag_id)] = category

    return id_to_tag, id_to_category


def expand_character_tags(tags: List[str], sep: str) -> List[str]:
    out = tags[:]
    for i, tag in enumerate(out):
        if tag.endswith(")"):
            parts = tag.split("(")
            character = "(".join(parts[:-1]).rstrip("_")
            series = parts[-1].replace(")", "")
            out[i] = character + sep + series
    return out


def apply_tag_replacement(tags: List[str], tag_replacements_arg: str) -> List[str]:
    escaped = tag_replacements_arg.replace("\\,", "@@@@").replace("\\;", "####")
    pairs = escaped.split(";")

    for pair in pairs:
        source, target = pair.split(",")
        source = source.replace("@@@@", ",").replace("####", ";")
        target = target.replace("@@@@", ",").replace("####", ";")
        tags = [target if t == source else t for t in tags]

    return tags


def build_session(onnx_path: str) -> Tuple[ort.InferenceSession, str, Optional[int]]:
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    try:
        batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    except Exception:
        batch_size = None
    del model

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "ROCMExecutionProvider" in available:
        providers.append("ROCMExecutionProvider")
    providers.append("CPUExecutionProvider")

    logger.info(f"Using onnxruntime providers: {providers}")
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, input_name, batch_size


def load_image(path: str) -> Tuple[str, np.ndarray, Tuple[int, int]]:
    with Image.open(path) as img:
        size = img.size
        arr = preprocess_image(img)
    return path, arr, size


def batch_loader(paths: List[str], batch_size: int, max_workers: int) -> List[Tuple[List[str], np.ndarray, List[Tuple[int, int]]]]:
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    results = []
    for batch in batches:
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                loaded = list(ex.map(load_image, batch))
        else:
            loaded = [load_image(p) for p in batch]

        paths_b, images_b, sizes_b = zip(*loaded) if loaded else ([], [], [])
        images_np = np.stack(images_b) if images_b else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        results.append((list(paths_b), images_np, list(sizes_b)))

    return results


def main(args):
    model_location, default_format, onnx_name = download_model(args)
    onnx_path = os.path.join(model_location, onnx_name)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session, input_name, fixed_batch = build_session(onnx_path)
    if fixed_batch and fixed_batch > 0 and args.batch_size != fixed_batch:
        logger.warning(
            f"Batch size {args.batch_size} doesn't match ONNX model batch size {fixed_batch}; using {fixed_batch}"
        )
        args.batch_size = fixed_batch

    if default_format:
        rating_tags, general_tags, character_tags = load_tags_default(model_location, args)
        id_to_tag = {}
        id_to_category = {}
    else:
        rating_tags, general_tags, character_tags = [], [], []
        id_to_tag, id_to_category = load_tags_mapping(model_location, args)

    train_data_dir_path = Path(args.train_data_dir)
    image_paths = glob_images_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"found {len(image_paths)} images")
    image_paths = [str(p) for p in image_paths]

    args.general_threshold = args.general_threshold if args.general_threshold is not None else args.thresh
    args.character_threshold = args.character_threshold if args.character_threshold is not None else args.thresh

    undesired = set(t.strip() for t in args.undesired_tags.split(args.caption_separator.strip()) if t.strip())
    always_first = None
    if args.always_first_tags:
        always_first = [t.strip() for t in args.always_first_tags.split(args.caption_separator.strip()) if t.strip()]

    tag_freq: Dict[str, int] = {}
    results: Dict[str, Dict] = {}

    batches = batch_loader(image_paths, args.batch_size, args.max_workers)

    for paths, images, sizes in tqdm(batches, smoothing=0.0):
        if len(paths) == 0:
            continue

        probs = session.run(None, {input_name: images})[0]
        probs = probs[: len(paths)]

        for image_path, image_size, prob in zip(paths, sizes, probs):
            combined_tags: List[str] = []

            if default_format:
                for i, p in enumerate(prob[4:]):
                    if i < len(general_tags) and p >= args.general_threshold:
                        tag_name = general_tags[i]
                        if tag_name not in undesired:
                            tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                            combined_tags.append(tag_name)
                    elif i >= len(general_tags) and p >= args.character_threshold:
                        tag_name = character_tags[i - len(general_tags)]
                        if tag_name not in undesired:
                            tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                            if args.character_tags_first:
                                combined_tags.insert(0, tag_name)
                            else:
                                combined_tags.append(tag_name)

                if args.use_rating_tags or args.use_rating_tags_as_last_tag:
                    rating_index = prob[:4].argmax()
                    rating = rating_tags[rating_index]
                    if rating not in undesired:
                        tag_freq[rating] = tag_freq.get(rating, 0) + 1
                        if args.use_rating_tags:
                            combined_tags.insert(0, rating)
                        else:
                            combined_tags.append(rating)
            else:
                prob = 1 / (1 + np.exp(-prob))
                tagged: List[Tuple[str, float]] = []
                character_first: List[Tuple[str, float]] = []

                for i, p in enumerate(prob):
                    if i not in id_to_tag:
                        continue
                    tag = id_to_tag[i]
                    category = id_to_category[i]
                    if tag in undesired:
                        continue

                    if category == "Rating":
                        if args.use_rating_tags:
                            tagged.append((tag, p))
                        continue

                    if category == "General" and p >= args.general_threshold:
                        tag_freq[tag] = tag_freq.get(tag, 0) + 1
                        tagged.append((tag, p))
                    elif category == "Character" and p >= args.character_threshold:
                        tag_freq[tag] = tag_freq.get(tag, 0) + 1
                        if args.character_tags_first:
                            character_first.append((tag, p))
                        else:
                            tagged.append((tag, p))

                tagged.sort(key=lambda x: x[1], reverse=True)
                if character_first:
                    character_first.sort(key=lambda x: x[1], reverse=True)
                    tagged = character_first + tagged

                combined_tags = [t[0] for t in tagged]

            if always_first:
                for tag in always_first:
                    if tag in combined_tags:
                        combined_tags.remove(tag)
                    combined_tags.insert(0, tag)

            caption_file = os.path.splitext(image_path)[0] + args.caption_extension
            tag_text = args.caption_separator.join(combined_tags)

            if args.append_tags and os.path.exists(caption_file):
                with open(caption_file, "rt", encoding="utf-8") as f:
                    existing = [t.strip() for t in f.read().strip("\n").split(args.caption_separator.strip()) if t.strip()]
                new_tags = [t for t in combined_tags if t not in existing]
                tag_text = args.caption_separator.join(existing + new_tags)

            if args.output_path:
                results[image_path] = {"tags": tag_text, "image_size": list(image_size)}
            else:
                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")

    if args.output_path:
        if args.output_path.endswith(".jsonl"):
            with open(args.output_path, "wt", encoding="utf-8") as f:
                for image_path, entry in results.items():
                    f.write(
                        json.dumps(
                            {"image_path": image_path, "caption": entry["tags"], "image_size": entry["image_size"]}
                        )
                        + "\n"
                    )
        else:
            with open(args.output_path, "wt", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"captions saved to {args.output_path}")

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("Tag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    logger.info("done")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for images")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO)
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--caption_extension", type=str, default=".txt")
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
    parser.add_argument("--caption_separator", type=str, default=", ")
    parser.add_argument("--tag_replacement", type=str, default=None)
    parser.add_argument("--character_tag_expand", action="store_true")
    parser.add_argument("--undesired_tags", type=str, default="")
    parser.add_argument("--frequency_tags", action="store_true")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
