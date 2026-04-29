Analyze the provided art and the booru tags below. Produce the JSON caption output following the Anima style-LoRA format defined in the system prompt.

The caption MUST start with `masterpiece, best quality, score_7, safe,` (or `nsfw` for explicit), followed by the style anchor (`@{style_name}.` if `{use_at_prefix}` = yes; otherwise `{style_name},`), then a 3-6 sentence natural-language scene description. Optional short booru-tag tail only if a specific character/act/pose is much more compact as a tag.

**Critical:** Describe the scene as if it were real. Do NOT describe the art style — the style anchor handles that.

Style anchor: `{style_name}`
Use `@` prefix: `{use_at_prefix}`

**Input image:** [The attached image]
**Booru Tags (verify against image):**
{tags}

---

Look at the image carefully. Tags are your primary source of truth, but use the image to refine pose, framing, expression, lighting, and composition into the natural-language scene description. Front-load anchors (quality + style). Output only the JSON required by the system instructions.
