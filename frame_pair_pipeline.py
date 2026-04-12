"""Frame-pair captioning pipeline.

Orchestrates the full pipeline for creating scene-transition captions from
paired images (A → B).  Steps:

1. Organize pairs from source directory into datasets
2. Run WD/PixAI tagging on all images
3. Describe A images via xAI Batch API
4. Compute similarity between A and B images
5. Caption B images via xAI Batch API (using A context + similarity)
6. Upload results to HuggingFace
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constants import IMAGE_EXTS
from ui import (
    ask_yes_no,
    console,
    make_progress,
    print_error,
    print_info,
    print_section,
    print_success,
    print_summary_table,
    print_warning,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")

# Suffixes that identify each role in a pair group
_SUFFIXES = ("_A", "_B", "_C", "_image_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_suffix(stem: str) -> Optional[Tuple[str, str]]:
    """Return (base_name, suffix) if *stem* ends with a known suffix, else None."""
    for suf in _SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], suf
    return None


# ---------------------------------------------------------------------------
# Step 1: Organize pairs
# ---------------------------------------------------------------------------


def organize_pairs(
    source_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Scan source_dir for frame-pair groups and copy into dataset subdirs.

    Returns a dict with:
        counts: {dataset_1: int, dataset_2: int, dataset_3: int}
        pairs: list of (path_a, path_b) tuples across all datasets
    """
    source = Path(source_dir)
    out = Path(output_dir)

    # Scan all image files and group by relative path + base name
    groups: Dict[str, Dict[str, str]] = {}  # group_key -> {suffix -> full_path}
    all_files = []
    for root, _, files in os.walk(source):
        rel_root = os.path.relpath(root, source)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            stem = os.path.splitext(f)[0]
            parsed = _strip_suffix(stem)
            if parsed is None:
                continue
            base_name, suffix = parsed
            full_path = os.path.join(root, f)
            group_key = os.path.join(rel_root, base_name) if rel_root != "." else base_name
            groups.setdefault(group_key, {})[suffix] = full_path
            all_files.append(full_path)

    # Classify groups into datasets
    ds1_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B + C
    ds2_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B + image_base
    ds3_groups: List[Tuple[str, Dict[str, str]]] = []  # A + B only

    for group_key, files_map in sorted(groups.items()):
        has_a = "_A" in files_map
        has_b = "_B" in files_map
        has_c = "_C" in files_map
        has_ib = "_image_base" in files_map

        if not (has_a and has_b):
            continue  # Must have at least A and B

        # Use a flat safe name for destination files (replace path separators)
        safe_name = group_key.replace(os.sep, "__")
        if has_c:
            ds1_groups.append((safe_name, files_map))
        elif has_ib:
            ds2_groups.append((safe_name, files_map))
        else:
            ds3_groups.append((safe_name, files_map))

    # Create output directories and copy files
    datasets = {
        "dataset_1": ds1_groups,
        "dataset_2": ds2_groups,
        "dataset_3": ds3_groups,
    }

    pairs: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    total_files = sum(len(g) for gs in datasets.values() for _, g in gs)

    with make_progress() as progress:
        task = progress.add_task("Organizing pairs", total=total_files)

        for ds_name, ds_groups in datasets.items():
            ds_dir = out / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)
            counts[ds_name] = len(ds_groups)

            for base_name, files_map in ds_groups:
                for suffix, src_path in files_map.items():
                    ext = os.path.splitext(src_path)[1]
                    dest = ds_dir / f"{base_name}{suffix}{ext}"
                    shutil.copy2(src_path, dest)
                    progress.advance(task)

                # Record pair (A, B)
                ext_a = os.path.splitext(files_map["_A"])[1]
                ext_b = os.path.splitext(files_map["_B"])[1]
                path_a = str(ds_dir / f"{base_name}_A{ext_a}")
                path_b = str(ds_dir / f"{base_name}_B{ext_b}")
                pairs.append((path_a, path_b))

    return {
        "counts": counts,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Step 2: WD / PixAI tagging
# ---------------------------------------------------------------------------


def _run_tagger_subprocess(
    python: str,
    tagger_script: str,
    directory: str,
    tagger_name: str,
    batch_size: int,
    force: bool = False,
) -> int:
    """Run a tagger subprocess on a directory. Returns exit code."""
    import subprocess

    cmd = [
        python, tagger_script, directory,
        "--taggers", tagger_name,
        "--batch_size", str(batch_size),
        "--remove_underscore",
        "--thresh", "0.30",
    ]
    if force:
        cmd.append("--force")
    # Pass HF token explicitly for gated models (PixAI)
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    )
    if hf_token:
        cmd.extend(["--hf_token", hf_token])
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print_warning("Interrupted — killing tagger process...")
        proc.kill()
        proc.wait()
        raise

    return proc.returncode


def run_wd_tagging(
    pairs: List[Tuple[str, str]],
    python: str,
    batch_size: int = 4,
    force: bool = False,
) -> bool:
    """Run PixAI tagger (fallback to WD14) on all unique A and B images.

    Tries PixAI first. If it fails (e.g. gated model), falls back to WD14.
    Returns True if tagging succeeded, False otherwise.
    """
    # Collect unique directories
    dirs: set = set()
    for path_a, path_b in pairs:
        dirs.add(os.path.dirname(path_a))
        dirs.add(os.path.dirname(path_b))

    tagger_script = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
    sorted_dirs = sorted(dirs)
    total_dirs = len(sorted_dirs)
    any_success = False

    for idx, d in enumerate(sorted_dirs, 1):
        print_info(f"[{idx}/{total_dirs}] Tagging images in: {d}")

        # Try PixAI first
        print_info("Trying PixAI tagger...")
        rc = _run_tagger_subprocess(python, tagger_script, d, "pixai", batch_size, force=force)

        if rc != 0:
            # Fallback to WD14
            print_warning(f"PixAI failed (code {rc}), falling back to WD14...")
            rc = _run_tagger_subprocess(python, tagger_script, d, "wd14", batch_size, force=force)

            if rc != 0:
                print_error(f"WD14 also failed (code {rc}) for {d}")
            else:
                print_success(f"WD14 tagging complete for {d}")
                any_success = True
        else:
            print_success(f"PixAI tagging complete for {d}")
            any_success = True

    if not any_success:
        print_error("All taggers failed for all directories. Cannot proceed without tags.")

    return any_success


# ---------------------------------------------------------------------------
# Step 3: Describe A images via xAI Batch (reuses tagger subprocess)
# ---------------------------------------------------------------------------


def _run_grok_batch_subprocess(
    python: str,
    directory: str,
    action: str,
    xai_api_key: str,
    model: str,
    system_prompt_file: Optional[str] = None,
    user_prompt_file: Optional[str] = None,
    prompt_profile: Optional[str] = None,
    batch_state_file: Optional[str] = None,
    caption_extension: str = ".txt",
) -> int:
    """Run grok xAI batch via the existing tagger subprocess.

    This reuses 100% of the battle-tested batch API implementation in
    tag_images_by_wd14_tagger.py (state JSON, resume, progress bars,
    payload size management, 413 fallback, rate limiting, etc.).
    """
    import subprocess

    tagger_script = os.path.join(SCRIPT_DIR, "tag_images_by_wd14_tagger.py")
    cmd = [
        python, tagger_script, directory,
        "--taggers", "grok",
        "--grok_provider", "xai-batch",
        "--xai_batch_action", action,
        "--xai_batch_model", model,
        "--caption_extension", caption_extension,
    ]
    if system_prompt_file:
        cmd.extend(["--grok_system_prompt_file", system_prompt_file])
    if user_prompt_file:
        cmd.extend(["--grok_prompt_file", user_prompt_file])
    if prompt_profile:
        cmd.extend(["--prompt_profile", prompt_profile])
    if batch_state_file:
        cmd.extend(["--xai_batch_state_file", batch_state_file])

    env = os.environ.copy()
    env["XAI_API_KEY"] = xai_api_key

    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print_warning("Interrupted — killing grok batch process...")
        proc.kill()
        proc.wait()
        raise

    return proc.returncode


def run_describe_a(
    pairs: List[Tuple[str, str]],
    xai_api_key: str,
    model: str,
    output_dir: str,
    python: str,
) -> None:
    """Describe all A images via xAI Batch using the existing tagger infrastructure.

    Creates a temporary system prompt for image description, then calls the
    tagger with --taggers grok --grok_provider xai-batch for each dataset dir.
    Uses submit → status → collect flow with full resume support.
    """
    print_section("STEP 3: DESCRIBE A IMAGES")

    # Create temporary describe prompt files
    import tempfile
    describe_system = (
        "You are an image describer for an AI dataset. Describe this image in complete "
        "detail (~150+ words). Describe everything you see: characters and their appearance "
        "(skin tone, hair color/length/style, eye color, body type, expression, clothing), "
        "pose, action, interaction with others, background, setting, lighting, colors, mood, "
        "and atmosphere. Output only valid JSON: {\"description\": \"...\"}. No other text."
    )
    describe_user = (
        "Describe this image in complete detail. Be thorough — describe every visual "
        "element you can identify.\n\n"
        "Booru tags for reference (verify against image):\n{tags}\n\n"
        "Produce the JSON output."
    )

    sys_prompt_file = os.path.join(output_dir, "_describe_system_prompt.md")
    usr_prompt_file = os.path.join(output_dir, "_describe_user_prompt.md")
    with open(sys_prompt_file, "w", encoding="utf-8") as f:
        f.write(describe_system)
    with open(usr_prompt_file, "w", encoding="utf-8") as f:
        f.write(describe_user)

    # Collect unique directories containing A images
    a_dirs: set = set()
    for path_a, _ in pairs:
        a_dirs.add(os.path.dirname(path_a))

    sorted_dirs = sorted(a_dirs)
    total_dirs = len(sorted_dirs)

    for idx, d in enumerate(sorted_dirs, 1):
        state_file = os.path.join(output_dir, f".xai_batch_state_describe_a_{idx}.json")

        # Submit
        print_info(f"[{idx}/{total_dirs}] Submitting A descriptions for: {d}")
        rc = _run_grok_batch_subprocess(
            python, d, "submit", xai_api_key, model,
            system_prompt_file=sys_prompt_file,
            user_prompt_file=usr_prompt_file,
            batch_state_file=state_file,
            caption_extension="_description.txt",
        )
        if rc != 0:
            print_warning(f"Submit returned code {rc} for {d}")

        # Collect (in separate call — batch processes on xAI servers)
        print_info(f"[{idx}/{total_dirs}] Collecting A descriptions for: {d}")
        rc = _run_grok_batch_subprocess(
            python, d, "collect", xai_api_key, model,
            system_prompt_file=sys_prompt_file,
            user_prompt_file=usr_prompt_file,
            batch_state_file=state_file,
            caption_extension="_description.txt",
        )
        if rc != 0:
            print_warning(f"Collect returned code {rc} for {d}")
        else:
            print_success(f"Descriptions complete for {d}")


# ---------------------------------------------------------------------------
# Step 4: Similarity computation
# ---------------------------------------------------------------------------


def run_similarity(
    pairs: List[Tuple[str, str]],
    device: str = "cuda",
) -> List[float]:
    """Compute combined similarity for all A-B pairs."""
    print_section("STEP 4: COMPUTE SIMILARITY")

    from frame_pair_similarity import compute_combined_similarity

    paths_a = [a for a, _ in pairs]
    paths_b = [b for _, b in pairs]

    print_info(f"Computing similarity for {len(pairs)} pairs (CLIP + SSCD + SSIM)...")
    similarities = compute_combined_similarity(paths_a, paths_b, device=device)

    # Save individual similarity JSONs
    written = 0
    for i, (path_a, path_b) in enumerate(pairs):
        sim_data = {"combined": round(similarities[i], 2)}

        stem_b = os.path.splitext(path_b)[0]
        # Also derive the base name for the similarity file
        stem_a = os.path.splitext(path_a)[0]
        base_parsed = _strip_suffix(os.path.basename(stem_a))
        if base_parsed:
            base_name = base_parsed[0]
            sim_file = os.path.join(os.path.dirname(path_a), f"{base_name}_similarity.json")
        else:
            sim_file = stem_b + "_similarity.json"

        with open(sim_file, "w", encoding="utf-8") as f:
            json.dump(sim_data, f, ensure_ascii=False, indent=2)
        written += 1

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    print_success(f"Similarity computed: avg={avg_sim:.1f}%, wrote {written} JSON files")

    return similarities


# ---------------------------------------------------------------------------
# Step 5: Caption B images via xAI Batch
# ---------------------------------------------------------------------------


def run_caption_b(
    pairs: List[Tuple[str, str]],
    similarities: List[float],
    xai_api_key: str,
    model: str,
    output_dir: str,
    python: str,
) -> None:
    """Caption B images via xAI Batch using the existing tagger infrastructure.

    Pre-generates context .txt files for each B image (containing A tags, A description,
    similarity, B tags) so the tagger's {tags} placeholder receives the full context.
    Then calls the tagger with --prompt_profile frame-pair for submit + collect.
    """
    print_section("STEP 5: CAPTION B IMAGES")

    user_prompt_path = os.path.join(PROMPTS_DIR, "image", "frame-pair", "user_prompt.md")
    if not os.path.exists(user_prompt_path):
        raise FileNotFoundError(f"User prompt not found: {user_prompt_path}")
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()

    # Pre-generate context files for B images that the tagger will use as "tags"
    print_info("Preparing context files for B images...")
    prepared = 0
    skipped_existing = 0

    with make_progress() as progress:
        task = progress.add_task("Preparing B context", total=len(pairs))

        for i, (path_a, path_b) in enumerate(pairs):
            # The tagger will read .txt as "existing tags" for the image.
            # We write the full context (A tags, A desc, similarity, B tags)
            # into a temporary .txt next to B that the user_prompt template uses.
            caption_file = os.path.splitext(path_b)[0] + "_caption.txt"
            if os.path.exists(caption_file):
                skipped_existing += 1
                progress.advance(task)
                continue

            # Read WD tags for A
            tags_a_file = os.path.splitext(path_a)[0] + ".txt"
            wd_tags_a = "(no tags)"
            if os.path.exists(tags_a_file):
                with open(tags_a_file, "r", encoding="utf-8") as f:
                    wd_tags_a = f.read().strip() or "(no tags)"

            # Read description for A
            desc_a_file = os.path.splitext(path_a)[0] + "_description.txt"
            description_a = "(no description)"
            if os.path.exists(desc_a_file):
                with open(desc_a_file, "r", encoding="utf-8") as f:
                    description_a = f.read().strip() or "(no description)"

            # Read WD tags for B
            tags_b_file = os.path.splitext(path_b)[0] + ".txt"
            wd_tags_b = "(no tags)"
            if os.path.exists(tags_b_file):
                with open(tags_b_file, "r", encoding="utf-8") as f:
                    wd_tags_b = f.read().strip() or "(no tags)"

            similarity_percent = round(similarities[i], 1)

            # Build the full context that replaces {tags} in the user prompt
            context = user_prompt_template
            context = context.replace("{wd_tags_a}", wd_tags_a)
            context = context.replace("{description_a}", description_a)
            context = context.replace("{similarity_percent}", str(similarity_percent))
            context = context.replace("{wd_tags_b}", wd_tags_b)

            # Write context as _context.txt (the tagger reads .txt as tags)
            context_file = os.path.splitext(path_b)[0] + "_context.txt"
            with open(context_file, "w", encoding="utf-8") as f:
                f.write(context)
            prepared += 1
            progress.advance(task)

    if skipped_existing:
        print_info(f"Skipped {skipped_existing} B images (already have captions)")

    if prepared == 0:
        print_success("All B images already have captions — nothing to submit")
        return

    print_success(f"Prepared context for {prepared} B images")

    # Now call the tagger with grok xai-batch on each dataset directory.
    # The tagger will read existing .txt tags and pass them via {tags} to grok.
    # We use --caption_extension _caption.txt so it doesn't overwrite WD tags.
    b_dirs: set = set()
    for _, path_b in pairs:
        b_dirs.add(os.path.dirname(path_b))

    sorted_dirs = sorted(b_dirs)
    total_dirs = len(sorted_dirs)

    # Create a user prompt that just passes the pre-built context through
    # (the context file already has everything, tagger passes it as {tags})
    passthrough_user_prompt = "{tags}"
    passthrough_file = os.path.join(output_dir, "_caption_user_prompt.md")
    with open(passthrough_file, "w", encoding="utf-8") as f:
        f.write(passthrough_user_prompt)

    system_prompt_path = os.path.join(PROMPTS_DIR, "image", "frame-pair", "system_prompt.md")

    for idx, d in enumerate(sorted_dirs, 1):
        state_file = os.path.join(output_dir, f".xai_batch_state_caption_b_{idx}.json")

        # Submit
        print_info(f"[{idx}/{total_dirs}] Submitting B captions for: {d}")
        rc = _run_grok_batch_subprocess(
            python, d, "submit", xai_api_key, model,
            system_prompt_file=system_prompt_path,
            user_prompt_file=passthrough_file,
            batch_state_file=state_file,
            caption_extension="_caption.txt",
        )
        if rc != 0:
            print_warning(f"Submit returned code {rc} for {d}")

        # Collect
        print_info(f"[{idx}/{total_dirs}] Collecting B captions for: {d}")
        rc = _run_grok_batch_subprocess(
            python, d, "collect", xai_api_key, model,
            system_prompt_file=system_prompt_path,
            user_prompt_file=passthrough_file,
            batch_state_file=state_file,
            caption_extension="_caption.txt",
        )
        if rc != 0:
            print_warning(f"Collect returned code {rc} for {d}")
        else:
            print_success(f"Captions complete for {d}")


# ---------------------------------------------------------------------------
# Step 6: Upload to HuggingFace
# ---------------------------------------------------------------------------


def run_upload(
    output_dir: str,
    hf_token: str,
    hf_repo: str,
    python: str,
) -> None:
    """Upload B images + their .txt caption files to HuggingFace."""
    print_section("STEP 6: UPLOAD TO HUGGINGFACE")

    import subprocess

    # Collect B images and their .txt files
    upload_files: List[str] = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            stem = os.path.splitext(f)[0]
            if stem.endswith("_B"):
                full_path = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    upload_files.append(full_path)
                    # Include its _caption.txt file
                    caption_path = os.path.splitext(full_path)[0] + "_caption.txt"
                    if os.path.exists(caption_path):
                        upload_files.append(caption_path)

    if not upload_files:
        print_warning("No B images found to upload")
        return

    print_info(f"Uploading {len(upload_files)} files to {hf_repo}")

    # Create a temporary directory with just the B files for upload
    import tempfile
    with tempfile.TemporaryDirectory(prefix="araknideo_fp_upload_") as tmp_dir:
        for fpath in upload_files:
            rel = os.path.relpath(fpath, output_dir)
            dest = os.path.join(tmp_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(fpath, dest)

        venv_dir = os.path.join(SCRIPT_DIR, ".venv")
        pip = os.path.join(venv_dir, "bin", "pip")
        subprocess.run([pip, "install", "-q", "hf_xet"], capture_output=True, text=True)

        num_workers = max(4, min(64, (os.cpu_count() or 4) * 2))

        upload_script = (
            "from huggingface_hub import HfApi\n"
            "import sys, os\n"
            "api = HfApi(token=os.environ['HF_TOKEN'])\n"
            "api.create_repo(sys.argv[1], repo_type='dataset', "
            "private=True, exist_ok=True)\n"
            "api.upload_large_folder(\n"
            "    repo_id=sys.argv[1],\n"
            "    repo_type='dataset',\n"
            "    folder_path=sys.argv[2],\n"
            "    num_workers=int(sys.argv[3]),\n"
            ")\n"
            "print('__done__')\n"
        )

        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token
        env["HF_XET_HIGH_PERFORMANCE"] = "1"

        proc = subprocess.Popen(
            [python, "-c", upload_script, hf_repo, tmp_dir, str(num_workers)],
            env=env,
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            print_warning("Interrupted — killing upload process...")
            proc.kill()
            proc.wait()
            print_info("Re-run to resume automatically")
            return

        if proc.returncode == 0:
            print_success(f"Uploaded to https://huggingface.co/datasets/{hf_repo}")
        else:
            print_error("Upload failed — check logs above")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_frame_pair_pipeline(
    source_dir: str,
    output_dir: str,
    xai_api_key: str,
    model: str = "grok-4-1-fast-reasoning",
    device: str = "cuda",
    python: str = "",
    hf_token: str = "",
    hf_repo: str = "",
) -> None:
    """Run the full frame-pair captioning pipeline."""

    if not python:
        python = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")

    # Step 1: Organize pairs
    print_section("STEP 1: ORGANIZE PAIRS")
    result = organize_pairs(source_dir, output_dir)
    pairs = result["pairs"]
    counts = result["counts"]

    print_summary_table("Pair Organization", [
        ("Dataset 1 (A+B+C)", str(counts.get("dataset_1", 0))),
        ("Dataset 2 (A+B+base)", str(counts.get("dataset_2", 0))),
        ("Dataset 3 (A+B only)", str(counts.get("dataset_3", 0))),
        ("Total pairs", str(len(pairs))),
    ])

    if not pairs:
        print_error("No valid pairs found. Aborting.")
        return

    # Step 2: WD/PixAI tagging
    print_section("STEP 2: WD/PIXAI TAGGING")
    print_info(f"Running PixAI tagger on {len(pairs) * 2} images...")
    run_wd_tagging(pairs, python)
    print_success("Tagging complete")

    # Step 3: Describe A images
    run_describe_a(pairs, xai_api_key, model, output_dir)

    # Step 4: Similarity
    similarities = run_similarity(pairs, device)

    # Step 5: Caption B images
    run_caption_b(pairs, similarities, xai_api_key, model, output_dir)

    # Step 6: Upload (if credentials provided)
    if hf_token and hf_repo:
        run_upload(output_dir, hf_token, hf_repo, python)
    else:
        print_info("Skipping HuggingFace upload (no token/repo provided)")

    print_section("PIPELINE COMPLETE")
    print_success(f"Processed {len(pairs)} frame pairs")
    print_success(f"Output directory: {output_dir}")
