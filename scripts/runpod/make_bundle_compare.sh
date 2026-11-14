#!/bin/bash
# Krok 88: paczka dla compare_kadry.sh (jedna dziedzina, adapter MLX -> PEFT bez layers_*).
# Użycie: bash scripts/runpod/make_bundle_compare.sh <wyjście.tgz> <adapter_mlx_dir|-> <nazwa_wariantu> [dziedzina]
# Adapter "-" = bez adaptera (sama baza, np. dla compare_models.sh); nazwa wariantu wtedy bez znaczenia.
set -e
cd "$(dirname "$0")/../.."
OUT=${1:?wyjście.tgz}; ADAPTER=${2:?adapter}; VARIANT=${3:?wariant}; D=${4:-kadry_i_place}
STAGE=$(mktemp -d); PEFT=$(mktemp -d)
[ "$ADAPTER" = "-" ] || .venv/bin/python scripts/convert_lora.py mlx-to-peft "$ADAPTER" "$PEFT" >/dev/null
mkdir -p "$STAGE/data/processed" "$STAGE/data/distill/$D" "$STAGE/adapters-vllm/$VARIANT"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp data/*.json "$STAGE/data/"; cp requirements-common.txt requirements-cuda.txt "$STAGE/"
cp data/processed/all_articles.json data/processed/rag_index_${D}.npy data/processed/rag_index_${D}_meta.json "$STAGE/data/processed/"
cp data/distill/$D/eval_holdout.json data/distill/$D/compare_questions.json "$STAGE/data/distill/$D/"
if [ "$ADAPTER" != "-" ]; then
cp "$PEFT/adapter_model.safetensors" "$STAGE/adapters-vllm/$VARIANT/"
.venv/bin/python -c "
import json; c=json.load(open('$PEFT/adapter_config.json')); c.pop('layers_to_transform',None); c.pop('layers_pattern',None)
json.dump(c,open('$STAGE/adapters-vllm/$VARIANT/adapter_config.json','w'),indent=1)"
else rmdir "$STAGE/adapters-vllm/$VARIANT" "$STAGE/adapters-vllm"; fi
.venv/bin/python - "$STAGE" "$D" <<'PY'
import json, shutil, sys
from pathlib import Path
stage, d = Path(sys.argv[1]), sys.argv[2]
for e in json.loads(Path("data/acts_config.json").read_text(encoding="utf-8")):
    if d in (e.get("domains") or []):
        for suf in ("", "_history"):
            p = Path("data/processed") / f"{e['short']}{suf}.json"
            if p.exists(): shutil.copy(p, stage / "data" / "processed" / p.name)
PY
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE" "$PEFT"; ls -la "$OUT"
