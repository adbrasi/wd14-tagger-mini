Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima character-LoRA format defined in the system prompt.

**Character trigger:** `{trigger_character}`

Write the caption as a **single dense paragraph of natural-language prose**. Weave `{trigger_character}` naturally as the subject of the scene. Do NOT describe the art style itself. Do NOT include quality tags, booru tags, or `@anything`. Describe: what the character is doing, their pose and expression, visible clothing, environment, lighting, palette, atmosphere.

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The tags tell you what is factually in the image. Convert every relevant tag to prose. Use the image to refine pose, framing, expression, lighting, and color. Front-load the character's action and context so the LoRA receives clear training signal around the trigger. Output only the JSON required by the system instructions.
