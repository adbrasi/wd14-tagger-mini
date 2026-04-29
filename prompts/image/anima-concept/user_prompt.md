Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima concept-LoRA format defined in the system prompt.

Write the caption as a **single dense line in hybrid format**: booru tags + natural-language prose interleaved (`tag, tag, NL clause, tag, NL clause, …`). There is no trigger word — the caption is the entire learning signal. Do NOT include quality tags, trigger words, style descriptions, or `@anything`.

Keep canonical concept tokens as booru tags (sexual positions, body proportions, facial expressions, camera angles, kink scenarios, specific outfits, counts, gaze). Convert generic descriptions to prose (hair, eyes, skin, environment, lighting, mood, spatial relations). See the system prompt for the full policy.

**Input image:** [The attached image]
**Booru Tags (ground truth — keep canonical ones as tags, convert generic ones to prose):**
{tags}

---

Decide per tag: keep as booru tag (canonical concept) OR convert to prose (generic descriptive). Use the image to refine pose, framing, expression, lighting, and color that tags can't capture. Output only the JSON required by the system instructions.
