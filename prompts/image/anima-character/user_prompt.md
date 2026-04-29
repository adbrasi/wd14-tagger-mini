Analyze the provided art and the booru tags below. Produce the JSON caption output following the Anima character-LoRA format defined in the system prompt.

Write the caption as **a single dense paragraph of natural-language prose**, in tdrussell's training-caption shape. **Do NOT include quality tags, booru tags, the trigger word, the series name, or `@anything`** — the training TOML's `caption_prefix` handles the trigger. Describe the scene in prose: what's happening, framing, the character's identity (hair, eyes, skin, body, distinctive features), expression, clothing, environment, lighting, atmosphere.

Character (informational — do NOT write into the caption): `{character_name}`
Series (informational): `{series_name}`

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The tags tell you what is factually in the image. Convert every relevant tag to prose. Use the image to refine pose, framing, expression, lighting, and color. Front-load identity details (hair, eyes, body, distinctive markings) so the LoRA learns the character. Output only the JSON required by the system instructions.
