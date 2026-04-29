Analyze the provided art and the booru tags below. Produce the JSON caption output following the Anima style-LoRA format defined in the system prompt.

Write the caption as **a single dense paragraph of natural-language prose**, in tdrussell's training-caption shape. **Do NOT include quality tags, booru tags, trigger words, or `@{style_name}` anywhere** — those are handled by the training TOML's `caption_prefix`. Describe the scene as if writing for someone who can't see the image: what's in it, how it's composed, who is doing what, lighting, palette, atmosphere.

Style anchor (informational, do NOT write into the caption): `{style_name}`

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The booru tags tell you what is factually in the image. Convert every relevant tag to prose. Use the image to refine pose, framing, expression, lighting, palette, and overall atmosphere. Do not describe the art style itself. Output only the JSON required by the system instructions.
