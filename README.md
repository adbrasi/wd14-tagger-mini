# Multi Tagger (WD14 + Camie + PixAI) - Mini Tool

Ferramenta enxuta para taguear imagens em lote usando **múltiplos taggers** via ONNX Runtime (GPU):
- WD14: `SmilingWolf/wd-eva02-large-tagger-v3`
- Camie: `Camais03/camie-tagger-v2`
- PixAI: `deepghs/pixai-tagger-v0.9-onnx`

A saída é **separada por tagger** (não mistura vocabularios diferentes). Cada tagger gera seus próprios arquivos.

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

## Usar vários taggers (sem misturar)
```bash
python tag_images_by_wd14_tagger.py /caminho/para/imagens \
  --taggers wd14,camie,pixai \
  --batch_size 8 --recursive
```

Saída padrão (sem `--output_path`):
- `imagem.wd14.txt`
- `imagem.camie.txt`
- `imagem.pixai.txt`

Se usar `--output_path` (JSON/JSONL), gera um arquivo por tagger:
- `saida.wd14.json` / `saida.wd14.jsonl`
- `saida.camie.json` / `saida.camie.jsonl`
- `saida.pixai.json` / `saida.pixai.jsonl`

## Opções principais
- `--taggers`: lista separada por vírgula (`wd14,camie,pixai`)
- `--wd14_repo_id`, `--camie_repo_id`, `--pixai_repo_id`: repos HF
- `--batch_size`: tamanho do batch no ONNX
- `--recursive`: varre subpastas
- `--append_tags`: adiciona tags ao invés de sobrescrever
- `--output_path`: salva JSON/JSONL ao invés de .txt por imagem
- `--thresh`, `--general_threshold`, `--character_threshold`: controle de cortes
- `--character_tags_first`: coloca tags de personagem antes das gerais
- `--always_first_tags`: força tags iniciais (ex: `1girl, 1boy`)

## Observações
- O primeiro uso baixa os modelos para `models/`.
- Para máxima velocidade, aumente `--batch_size` até estourar VRAM.
