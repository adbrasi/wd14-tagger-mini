## Pipeline: Video Captioning NSFW

```
┌─────────────────────────────────────────────────────────┐
│              1. EXTRAÇÃO DE INFORMAÇÃO                    │
│                                                         │
│  Para cada clipe (~30s, já segmentado):                 │
│                                                         │
│  ┌──────────────┐       ┌─────────────────────────┐     │
│  │  Gemini Flash │       │  WD14 Tagger            │     │
│  │  Lite Preview │       │                         │     │
│  │              │       │  5 keyframes uniformes:  │     │
│  │  Recebe:     │       │  F1 (0%), F2 (25%),     │     │
│  │  - vídeo     │       │  F3 (50%), F4 (75%),    │     │
│  │  - system    │       │  F5 (100%)              │     │
│  │    prompt    │       │                         │     │
│  │              │       │  Output: tags explícitas │     │
│  │  Output:     │       │  por frame              │     │
│  │  narrativa   │       │                         │     │
│  │  temporal    │       │                         │     │
│  │  (censurada) │       │                         │     │
│  └──────┬───────┘       └────────┬────────────────┘     │
│         │                        │                      │
│         │    Rodam em paralelo   │                      │
└─────────┼────────────────────────┼──────────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│                  2. SÍNTESE (GROK)                       │
│                                                         │
│  Input (texto puro, sem imagens):                       │
│  ┌────────────────────────────────────────────────┐     │
│  │ [TAGS_ORIGINAIS]                               │     │
│  │ rule34, 3d, animated, character_name...         │     │
│  │                                                │     │
│  │ [GEMINI_DESCRIPTION]                           │     │
│  │ "she rhythmically grips a dark object..."      │     │
│  │                                                │     │
│  │ [WD14_FRAME_1] 1girl, brown_hair, smile...     │     │
│  │ [WD14_FRAME_2] 1girl, handjob, penis...        │     │
│  │ [WD14_FRAME_3] ...                             │     │
│  │ [WD14_FRAME_4] ...                             │     │
│  │ [WD14_FRAME_5] ...                             │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  System prompt com:                                     │
│  - Regras de de-codificação (eufemismo → explícito)     │
│  - Estilo narrativo (presente, cinematográfico)         │
│  - Estrutura de output (JSON)                           │
│  - Exemplos few-shot                                    │
│                                                         │
│  Output: caption narrativa explícita                    │
│                                                         │
│  ⚠️  FALLBACK: se Gemini recusar/falhar,               │
│  Grok recebe apenas tags + WD14 + keyframes (visão)    │
│  e gera a caption sozinho                               │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              3. OUTPUT FINAL                              │
│                                                         │
│  clip_001.mp4  ←→  clip_001.txt                         │
│  clip_002.mp4  ←→  clip_002.txt                         │
│  ...                                                    │
│                                                         │
│  Cada .txt = caption narrativa explícita                │
│  pronta pra training loop do video gen model            │
└─────────────────────────────────────────────────────────┘
```

### Fluxo de fallback:

```
Gemini respondeu?
  ├─ SIM → Grok sintetiza (tags + gemini + wd14)
  └─ NÃO → Grok assume sozinho (tags + wd14 + keyframes via visão)
```

No fallback o Grok **passa a receber imagens** dos keyframes, compensando a ausência do Gemini. Custo maior por clipe, mas garante que nenhum vídeo fica sem caption.

---
