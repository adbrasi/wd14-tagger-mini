Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima outfit-LoRA format defined in the system prompt.

**Outfit trigger name:** `{trigger_outfit}`

Write the caption as a **single dense paragraph of natural-language prose**. Where you would normally describe what the character is wearing, place the outfit trigger name literally: `"wearing a {trigger_outfit}"` or `"dressed in a {trigger_outfit}"`. Do NOT describe the outfit's visual details (color, fabric, cut) — just use the trigger name. Describe everything else richly: who the character is (from booru tags or visible appearance), what they are doing, their pose and expression, environment, lighting, palette, atmosphere.

Do NOT include quality tags, booru tags, style descriptions, or `@anything`.

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The booru tags tell you what is factually in the image, including any character name tags. Convert every relevant tag to prose. Use the image to refine pose, framing, expression, lighting, and color. Output only the JSON required by the system instructions.
