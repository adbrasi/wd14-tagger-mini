Analyze the provided art and the booru tags below. Produce the JSON caption output following the 5-section Anima character LoRA format defined in the system prompt.

The caption MUST start with `masterpiece, best quality, score_7, safe,` (or `nsfw` if explicit) followed by the count + trigger + series block, then a 2-4 sentence natural-language paragraph, then the remaining booru tags with **spaces instead of underscores** (except `score_N` tags).

Trigger: `{character_name}`
Series: `{series_name}` (may be empty — render as `{character_name} \({series_name}\)` only if non-empty)

**Input image:** [The attached image]
**Booru Tags (verify against image):**
{tags}

---

Look at the image carefully. Tags are your primary source of truth, but use the image to refine pose, framing, expression, lighting, and composition into the natural-language paragraph (Section 3). Convert remaining tags to space-separated form for Section 4. Front-load anchors — Qwen3 causal attention treats tail tokens as noise. Output only the JSON required by the system instructions.
