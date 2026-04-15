# Design: Video Caption Pipeline (Gemini + Grok)

## Understanding Summary

- **What**: New pipeline for video captioning using Gemini Flash (video understanding) + PixAI (5 keyframe tags) + Grok (text-only synthesis into final caption)
- **Why**: Current pipeline sees only 1 frame. This sees the entire video (Gemini) + 5 keyframes (PixAI), producing much more accurate captions for video generation model training
- **Constraints**: Gemini Batch API + xAI Batch API (both async, cheap). Grok never receives images — always text-only. Videos already segmented (~30s)
- **Menu**: New CLI option — "Video Caption (Gemini + Grok)"
- **Output**: One `.txt` per video with Grok's final caption only

## Architecture

### File Structure

```
data_araknideo/
├── gemini_batch.py              # Gemini Batch API (upload, submit, poll, collect)
├── video_caption_pipeline.py    # Orchestrator: Gemini → PixAI → Grok
├── prompts/
│   └── video-caption/
│       └── default/
│           ├── gemini_system_prompt.md
│           ├── grok_system_prompt.md
│           └── grok_user_prompt.md
└── cli.py                       # New option 6 in main menu
```

### Phase 1: Gemini (Video Understanding)

1. Upload all videos via File API (`client.files.upload`)
2. Create context cache for system prompt (90% discount on cached tokens)
3. Submit batch job with all videos + system prompt
4. Poll until complete
5. Collect descriptions, track which videos Gemini refused/failed
6. Cleanup: delete uploaded files + cache

**Key decisions:**
- Use `MEDIA_RESOLUTION_LOW` to reduce tokens (~100 tok/s vs 300)
- Gemini model: `gemini-3.1-flash-lite-preview`
- Batch API for 50% discount on tokens
- Context cache for system prompt (90% discount)
- File API upload handles videos up to 2GB each

### Phase 2: PixAI (5 Keyframe Tags)

1. Extract 5 frames per video at positions: 0%, 25%, 50%, 75%, 100%
2. Run PixAI tagger on all extracted frames
3. Group tags by video, preserving per-frame structure

**Reuses:** `extract_frames()` from `wd14_utils.py`, PixAI tagger via subprocess

### Phase 3: Grok (Text-Only Synthesis)

For each video, build text-only prompt containing:
- Gemini description (if available)
- PixAI tags per frame (5 sets, labeled FRAME_1 through FRAME_5)

Submit via xAI Batch API → collect → write `.txt` files.

**Fallback:** If Gemini failed/refused for a video, Grok receives only PixAI tags (no Gemini description). Automatic, no user intervention.

### Data Flow

```
Video (MP4)
  │
  ├──► Gemini Batch API ──► description text (or failure marker)
  │     (sees full video)
  │
  ├──► Extract 5 frames ──► PixAI tagger ──► 5 sets of tags
  │     (0% 25% 50% 75% 100%)
  │
  └──► [all text combined] ──► Grok xAI Batch ──► final caption ──► .txt
```

### CLI Wizard Flow

1. Scan videos in dataset directory
2. Ask for Gemini API key (or detect from env `GEMINI_API_KEY`)
3. Ask for xAI API key (or detect from env `XAI_API_KEY`)
4. Ask for prompt profile (default)
5. Ask for HF token if needed (PixAI)
6. Batch size for PixAI (auto from VRAM)
7. Show config summary table
8. Run Phase 1 (Gemini) → Phase 2 (PixAI) → Phase 3 (Grok)
9. Auto-collect results and write .txt files

### Intermediate State

Each phase saves its output to a temp state directory inside the dataset:
- `.gemini_descriptions/` — JSON mapping video_path → description text
- `.pixai_frame_tags/` — JSON mapping video_path → {frame_1: [...], frame_2: [...], ...}

These are cleaned up after successful completion, or preserved for debugging on failure.

## Decision Log

| Decision | Alternatives | Why |
|----------|-------------|-----|
| Dedicated script (approach A) | Extend tagger (approach B) | Tagger is 3200 lines, Gemini File API is different pattern |
| Gemini Batch API | Real-time API | 50% cheaper, same pattern as xAI batch |
| Context cache for system prompt | No cache | 90% discount on repeated system prompt tokens |
| MEDIA_RESOLUTION_LOW | HIGH/default | 3x cheaper per video, sufficient for scene description |
| PixAI via subprocess | Inline import | Consistent with frame_pair_pipeline pattern |
| Grok via xAI Batch only | OpenRouter option | User requirement, simplifies pipeline |
| Intermediate state in temp dir | In-memory only | Enables resume on failure, debugging |
| 5 fixed keyframes | Configurable | User requirement |
