#!/bin/bash
# Krok 122: paczka dla prod_vs_v31.sh (adaptery PRODUKCYJNE budowlane/cywilne/mieszkaniowe skonwertowane MLX->PEFT + gotowe konteksty z Kroku 112).
# Użycie: [DOMAINS="konsumenckie"] bash scripts/runpod/make_bundle_k122.sh <wyjście.tgz>  (konsumenckie: konteksty z eval.json, bez odpowiedzi Qwena -- generuje je prod_vs_v31.sh)
set -e
cd "$(dirname "$0")/../.."
OUT=${1:?wyjście.tgz}
src() { case $1 in budowlane) echo bielik11b-budowlane-distill-lora-iter584;; cywilne) echo bielik11b-cywilne-distill-k78-lora-iter550;; mieszkaniowe) echo bielik11b-mieszkaniowe-distill-k78-lora-iter413;; konsumenckie) echo bielik11b-konsumenckie-distill-lora-iter200;; esac; }
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/distill" "$STAGE/adapters-vllm"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp data/*.json "$STAGE/data/"
for d in ${DOMAINS:-budowlane cywilne mieszkaniowe}; do
  mkdir -p "$STAGE/data/distill/$d" "$STAGE/adapters-vllm/prod_$d"
  if [ -f data/distill/$d/eval_vs_adapter.json ]; then cp data/distill/$d/eval_vs_adapter.json "$STAGE/data/distill/$d/"
  else .venv/bin/python -c "
import json; e=json.load(open('data/distill/$d/eval.json',encoding='utf-8'))
json.dump({'contexts': e['contexts'], 'answers': {}}, open('$STAGE/data/distill/$d/eval_vs_adapter.json','w',encoding='utf-8'), ensure_ascii=False)"; fi
  PEFT=$(mktemp -d)
  .venv/bin/python scripts/convert_lora.py mlx-to-peft adapters/$(src $d) "$PEFT" > /dev/null
  cp "$PEFT/adapter_model.safetensors" "$STAGE/adapters-vllm/prod_$d/"
  .venv/bin/python -c "
import json; c=json.load(open('$PEFT/adapter_config.json')); c.pop('layers_to_transform',None); c.pop('layers_pattern',None)
json.dump(c,open('$STAGE/adapters-vllm/prod_$d/adapter_config.json','w'),indent=1)"
  rm -rf "$PEFT"
done
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"; ls -la "$OUT"
