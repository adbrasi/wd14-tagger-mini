"""Image similarity computation using CLIP, SSCD, and SSIM.

Compares pairs of images (A, B) using three complementary methods and
returns an averaged similarity score.  All heavy lifting runs on GPU
with batched inference; graceful fallback to CPU when CUDA is unavailable.

Methods
-------
1. **CLIP / SigLIP** – semantic similarity via ``open-clip-torch`` + ``timm``.
2. **SSCD** – copy-detection similarity via Meta's TorchScript model.
3. **SSIM** – structural similarity via ``pytorch-msssim``.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SSCD_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
    "sscd_disc_mixup.torchscript.pt"
)
_SSCD_FILENAME = "sscd_disc_mixup.torchscript.pt"

_CLIP_MODEL_NAME = "hf-hub:timm/ViT-B-16-SigLIP"


def _resolve_device(device: str) -> torch.device:
    """Return *device* when available, otherwise fall back to CPU."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def _load_image(path: str) -> Optional[Image.Image]:
    """Open an image as RGB, returning ``None`` on any error."""
    try:
        img = Image.open(path).convert("RGB")
        # Force full decode so corrupt files are caught here.
        img.load()
        return img
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def _batched_indices(total: int, batch_size: int):
    """Yield ``(start, end)`` index pairs for batched iteration."""
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


# ---------------------------------------------------------------------------
# SSCD model download
# ---------------------------------------------------------------------------


def download_sscd_model(model_dir: str = "models") -> str:
    """Download SSCD TorchScript model if not already present.

    Returns the local file path to the model.
    """
    os.makedirs(model_dir, exist_ok=True)
    dest = os.path.join(model_dir, _SSCD_FILENAME)
    if os.path.isfile(dest):
        logger.info("SSCD model already present at %s", dest)
        return dest

    logger.info("Downloading SSCD model to %s …", dest)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(_SSCD_URL, tmp)
        os.replace(tmp, dest)
        logger.info("SSCD model downloaded successfully")
    except Exception:
        # Clean up partial download.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest


# ---------------------------------------------------------------------------
# Method 1: CLIP / SigLIP
# ---------------------------------------------------------------------------


def compute_clip_similarity(
    paths_a: List[str],
    paths_b: List[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> List[float]:
    """Compute CLIP/SigLIP cosine similarity for image pairs.

    Returns a list of similarity scores in ``[0.0, 1.0]`` (one per pair).
    Pairs where either image cannot be loaded score ``0.0``.
    """
    import open_clip  # lazy import — heavy dependency

    n = len(paths_a)
    assert n == len(paths_b), "paths_a and paths_b must have the same length"
    dev = _resolve_device(device)

    logger.info("Loading CLIP model %s …", _CLIP_MODEL_NAME)
    model, preprocess = open_clip.create_model_from_pretrained(_CLIP_MODEL_NAME)
    model = model.eval().to(dev)

    scores: List[float] = [0.0] * n

    for start, end in _batched_indices(n, batch_size):
        tensors_a: List[torch.Tensor] = []
        tensors_b: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for i in range(start, end):
            img_a = _load_image(paths_a[i])
            img_b = _load_image(paths_b[i])
            if img_a is None or img_b is None:
                continue
            tensors_a.append(preprocess(img_a))
            tensors_b.append(preprocess(img_b))
            valid_indices.append(i)

        if not valid_indices:
            continue

        batch_a = torch.stack(tensors_a).to(dev)
        batch_b = torch.stack(tensors_b).to(dev)

        with torch.no_grad(), torch.amp.autocast(dev.type):
            feat_a = model.encode_image(batch_a)
            feat_b = model.encode_image(batch_b)
            feat_a = F.normalize(feat_a, dim=-1)
            feat_b = F.normalize(feat_b, dim=-1)
            sims = (feat_a * feat_b).sum(dim=-1)  # per-pair cosine

        sims_cpu = sims.float().cpu().tolist()
        for idx, sim in zip(valid_indices, sims_cpu):
            scores[idx] = max(0.0, min(float(sim), 1.0))

        logger.info(
            "CLIP: processed %d/%d pairs", min(end, n), n
        )

    return scores


# ---------------------------------------------------------------------------
# Method 2: SSCD (copy detection)
# ---------------------------------------------------------------------------

_SSCD_PREPROCESS = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def compute_sscd_similarity(
    paths_a: List[str],
    paths_b: List[str],
    model_path: str,
    device: str = "cuda",
    batch_size: int = 128,
) -> List[float]:
    """Compute SSCD cosine similarity for image pairs.

    Returns a list of similarity scores in ``[0.0, 1.0]`` (one per pair).
    Pairs where either image cannot be loaded score ``0.0``.
    """
    n = len(paths_a)
    assert n == len(paths_b), "paths_a and paths_b must have the same length"
    dev = _resolve_device(device)

    logger.info("Loading SSCD model from %s …", model_path)
    model = torch.jit.load(model_path, map_location=dev).eval()

    scores: List[float] = [0.0] * n

    for start, end in _batched_indices(n, batch_size):
        tensors_a: List[torch.Tensor] = []
        tensors_b: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for i in range(start, end):
            img_a = _load_image(paths_a[i])
            img_b = _load_image(paths_b[i])
            if img_a is None or img_b is None:
                continue
            tensors_a.append(_SSCD_PREPROCESS(img_a))
            tensors_b.append(_SSCD_PREPROCESS(img_b))
            valid_indices.append(i)

        if not valid_indices:
            continue

        batch_a = torch.stack(tensors_a).to(dev)
        batch_b = torch.stack(tensors_b).to(dev)

        with torch.no_grad(), torch.amp.autocast(dev.type):
            feat_a = F.normalize(model(batch_a), dim=-1)
            feat_b = F.normalize(model(batch_b), dim=-1)
            sims = (feat_a * feat_b).sum(dim=-1)

        sims_cpu = sims.float().cpu().tolist()
        for idx, sim in zip(valid_indices, sims_cpu):
            scores[idx] = max(0.0, min(float(sim), 1.0))

        logger.info(
            "SSCD: processed %d/%d pairs", min(end, n), n
        )

    return scores


# ---------------------------------------------------------------------------
# Method 3: SSIM
# ---------------------------------------------------------------------------

_SSIM_PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def compute_ssim_similarity(
    paths_a: List[str],
    paths_b: List[str],
    device: str = "cuda",
    batch_size: int = 256,
    resize: Tuple[int, int] = (256, 256),
) -> List[float]:
    """Compute SSIM for image pairs.

    Returns a list of similarity scores in ``[0.0, 1.0]`` (one per pair).
    Pairs where either image cannot be loaded score ``0.0``.
    """
    from pytorch_msssim import ssim  # lazy import

    n = len(paths_a)
    assert n == len(paths_b), "paths_a and paths_b must have the same length"
    dev = _resolve_device(device)

    # Build transform dynamically when a non-default resize is requested.
    if resize == (256, 256):
        preprocess = _SSIM_PREPROCESS
    else:
        preprocess = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
        ])

    scores: List[float] = [0.0] * n

    for start, end in _batched_indices(n, batch_size):
        tensors_a: List[torch.Tensor] = []
        tensors_b: List[torch.Tensor] = []
        valid_indices: List[int] = []

        for i in range(start, end):
            img_a = _load_image(paths_a[i])
            img_b = _load_image(paths_b[i])
            if img_a is None or img_b is None:
                continue
            tensors_a.append(preprocess(img_a))
            tensors_b.append(preprocess(img_b))
            valid_indices.append(i)

        if not valid_indices:
            continue

        batch_a = torch.stack(tensors_a).to(dev)
        batch_b = torch.stack(tensors_b).to(dev)

        with torch.no_grad():
            sims = ssim(batch_a, batch_b, data_range=1.0, size_average=False)

        sims_cpu = sims.float().cpu().tolist()
        for idx, sim in zip(valid_indices, sims_cpu):
            scores[idx] = max(0.0, min(float(sim), 1.0))

        logger.info(
            "SSIM: processed %d/%d pairs", min(end, n), n
        )

    return scores


# ---------------------------------------------------------------------------
# Combined similarity
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "clip": 0.4,
    "sscd": 0.35,
    "ssim": 0.25,
}


def compute_combined_similarity(
    paths_a: List[str],
    paths_b: List[str],
    device: str = "cuda",
    sscd_model_path: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Run all three methods and return the weighted average.

    Parameters
    ----------
    paths_a, paths_b:
        Parallel lists of image file paths to compare.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-falls-back to CPU when CUDA is
        unavailable.
    sscd_model_path:
        Path to the SSCD TorchScript model.  If ``None``, the model is
        downloaded automatically into ``./models/``.
    weights:
        Per-method weights.  Defaults to
        ``{"clip": 0.4, "sscd": 0.35, "ssim": 0.25}``.

    Returns
    -------
    List[float]
        Similarity percentages in ``[0, 100]`` (one per pair).
    """
    n = len(paths_a)
    assert n == len(paths_b), "paths_a and paths_b must have the same length"

    w = dict(weights or _DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Run each method; skip gracefully on total failure.
    # ------------------------------------------------------------------
    results: Dict[str, List[float]] = {}

    # CLIP / SigLIP
    try:
        logger.info("--- Running CLIP similarity ---")
        results["clip"] = compute_clip_similarity(
            paths_a, paths_b, device=device,
        )
    except Exception:
        logger.exception("CLIP similarity failed — skipping method")

    # SSCD
    try:
        logger.info("--- Running SSCD similarity ---")
        model_path = sscd_model_path or download_sscd_model()
        results["sscd"] = compute_sscd_similarity(
            paths_a, paths_b, model_path=model_path, device=device,
        )
    except Exception:
        logger.exception("SSCD similarity failed — skipping method")

    # SSIM
    try:
        logger.info("--- Running SSIM similarity ---")
        results["ssim"] = compute_ssim_similarity(
            paths_a, paths_b, device=device,
        )
    except Exception:
        logger.exception("SSIM similarity failed — skipping method")

    if not results:
        logger.error("All similarity methods failed — returning zeros")
        return [0.0] * n

    # ------------------------------------------------------------------
    # Weighted average (renormalise weights to the methods that succeeded)
    # ------------------------------------------------------------------
    active_weight = sum(w.get(k, 0.0) for k in results)
    if active_weight <= 0:
        active_weight = float(len(results))
        norm = {k: 1.0 / active_weight for k in results}
    else:
        norm = {k: w.get(k, 0.0) / active_weight for k in results}

    combined: List[float] = []
    for i in range(n):
        avg = sum(norm[k] * results[k][i] for k in results)
        # Convert 0-1 → 0-100 percentage.
        combined.append(round(max(0.0, min(avg * 100.0, 100.0)), 4))

    return combined
