Analyze the provided art and the booru tags below. Produce the JSON caption output following the Anima concept-LoRA format defined in the system prompt.

Write the caption as **a single dense paragraph of natural-language prose**, in tdrussell's training-caption shape. **Do NOT include quality tags, booru tags, the concept trigger, or `@anything`** — the training TOML's `caption_prefix` handles the trigger. The first portion of the caption MUST describe the concept itself tangibly (per `{concept_kind}`); the remainder describes the rest of the scene.

Concept (informational — do NOT write into the caption): `{concept_name}`
Concept kind: `{concept_kind}` (clothing/tattoo/pose/prop/accessory/object)

**Input image:** [The attached image]
**Booru Tags (ground truth — convert to prose, do not copy):**
{tags}

---

The tags tell you what is factually in the image. Convert every relevant tag to prose, with extra concrete physical detail on the concept itself (color, material, shape, placement, fit, decoration). Output only the JSON required by the system instructions.
