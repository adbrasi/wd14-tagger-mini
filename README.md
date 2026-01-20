# Multi Tagger (WD14 + Camie + PixAI) - Mini Tool

Ferramenta enxuta para taguear imagens em lote usando **múltiplos taggers** via ONNX Runtime (GPU):
- WD14: `SmilingWolf/wd-eva02-large-tagger-v3`
- Camie: `Camais03/camie-tagger-v2`
- PixAI: `deepghs/pixai-tagger-v0.9-onnx`

**Saída única por imagem**: `nome_da_imagem.txt` com tags separadas por vírgulas.
Quando você usa vários taggers, eles são executados em sequência e as tags vão sendo **anexadas**. As duplicadas são removidas **preservando as tags existentes primeiro**.

## Requisitos
- Python 3.10+
- GPU NVIDIA (5090 ok) com CUDA instalado

## Instalação
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso básico
```bash
python tag_images_by_wd14_tagger.py /caminho/para/imagens --batch_size 8 --recursive
```

## Usar vários taggers (sem misturar vocabularios)
```bash
python tag_images_by_wd14_tagger.py /caminho/para/imagens \
  --taggers wd14,camie,pixai \
  --batch_size 8 --recursive
```

## Prioridade de tags existentes (append)
Se já existir um `.txt` com tags, use `--append_tags`.
As tags existentes ficam **na frente** e as novas entram depois, sem duplicatas.

Exemplo:
- existente: `boy,girl,hug,kiss`
- taggers: `boy,muscular,lipstick,girl,hug,kiss,couple,dog,mature_female`
- final: `boy,girl,hug,kiss,muscular,lipstick,couple,dog,mature_female`

## Progress
- Por padrão há barra de progresso por tagger e total.
- Para desligar: `--no_progress`

## Rodar apenas um tagger
```bash
python tag_images_by_wd14_tagger.py /caminho/para/imagens --one_tagger wd14
```

## Smoke test (uma imagem)
```bash
python tag_images_by_wd14_tagger.py /caminho/para/imagens \
  --smoke_test_image /caminho/para/uma_imagem.png \
  --taggers wd14,camie,pixai
```

## Thresholds por tagger
- `--wd14_thresh`, `--camie_thresh`, `--pixai_thresh`
- `--wd14_general_threshold`, `--wd14_character_threshold`
- `--camie_general_threshold`, `--camie_character_threshold`
- `--pixai_general_threshold`, `--pixai_character_threshold`

## Script pronto (para /workspace ou /root)
Use `run_tagger.sh`:
```bash
./run_tagger.sh /caminho/para/imagens 8
```

## Observações
- O primeiro uso baixa os modelos para `models/`.
- Para máxima velocidade, aumente `--batch_size` até estourar VRAM.
