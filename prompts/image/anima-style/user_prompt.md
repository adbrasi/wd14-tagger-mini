Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima style-LoRA format defined in the system prompt.

**Style trigger (begin the caption with `@{trigger_style}.`):** `{trigger_style}`

Write the caption as a **single dense paragraph of natural-language prose**, starting with `@{trigger_style}.` followed by a space and the scene description. Do NOT describe the art style itself anywhere — the trigger handles that. Do NOT include quality tags, booru tags, or any additional `@anything` beyond the opening trigger.

Describe the scene: who is in it, what they are doing, composition and framing, appearance and clothing, environment, lighting, palette, atmosphere.

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The booru tags tell you what is factually in the image. Convert every relevant tag to prose. Use the image to refine pose, framing, expression, lighting, and color that tags can't capture. Output only the JSON required by the system instructions.
