Analyze the provided art and the booru tags below. Produce the JSON caption output following the Anima concept-LoRA format defined in the system prompt.

The caption MUST start with `masterpiece, best quality, score_7, safe,` (or `nsfw`), followed by `1girl, solo, {concept_name},` (trigger up front, within first ~12 tokens), then a 2-4 sentence natural-language description whose FIRST sentence explicitly describes the concept tangibly (per concept_kind = `{concept_kind}`), then a tail of booru tags for the rest of the scene. Repeat `{concept_name}` once in the booru-tag tail for reinforcement.

Trigger: `{concept_name}`
Concept kind: `{concept_kind}` (clothing/tattoo/accessory/pose/prop/object)

**Input image:** [The attached image]
**Booru Tags (verify against image):**
{tags}

---

Look at the image carefully. Tags are your primary source of truth; use the image to refine concrete details about the concept (color, shape, material, placement, fit). Do not bury the trigger. Output only the JSON required by the system instructions.
