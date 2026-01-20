# WD14 Tagger (ONNX) - Mini Tool

Ferramenta enxuta para taguear imagens em lote usando o modelo **SmilingWolf/wd-eva02-large-tagger-v3** via ONNX Runtime (GPU).

## Requisitos
- Python 3.10+
- GPU NVIDIA (5090 ok) com CUDA instalado
## Notas do modelo
- O ONNX do v3 aceita batch variável (você pode aumentar o batch).
- O runtime recomendado é `onnxruntime >= 1.17.0`.

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

## Opções principais
- `--repo_id`: modelo HF (padrão: `SmilingWolf/wd-eva02-large-tagger-v3`)
- `--batch_size`: tamanho do batch no ONNX
- `--recursive`: varre subpastas
- `--append_tags`: adiciona tags ao invés de sobrescrever
- `--output_path`: salva JSON/JSONL ao invés de .txt por imagem
- `--thresh`, `--general_threshold`, `--character_threshold`: controle de cortes
- `--character_tags_first`: coloca tags de personagem antes das gerais
- `--always_first_tags`: força tags iniciais (ex: `1girl, 1boy`)

## Observações
- O primeiro uso baixa o modelo para `wd14_tagger_model/`.
- Para máxima velocidade, aumente `--batch_size` até estourar VRAM.
