Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima style-LoRA format defined in the system prompt.

**Style trigger (begin the caption with `@{trigger_style}.`):** `{trigger_style}`

Write the caption as a **single dense line in hybrid format**: starting with `@{trigger_style}.` followed by a space, then booru tags + natural-language prose interleaved (`tag, tag, NL clause, tag, NL clause, …`). Do NOT describe the art style itself anywhere — the trigger handles that. Do NOT include quality tags or any additional `@anything` beyond the opening trigger.

Keep canonical concept tokens as booru tags (sexual positions, body proportions, facial expressions, camera angles, kink scenarios, specific outfits, counts, gaze). Convert generic descriptions to prose (hair, eyes, skin, environment, lighting, mood, spatial relations). See the system prompt for the full policy.

**Input image:** [The attached image]
**Booru Tags (ground truth — keep canonical ones as tags, convert generic ones to prose):**
{tags}

---

Decide per tag: keep as booru tag (canonical concept) OR convert to prose (generic descriptive). Use the image to refine pose, framing, expression, lighting, and color that tags can't capture. Output only the JSON required by the system instructions.
