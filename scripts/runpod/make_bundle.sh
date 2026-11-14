#!/bin/bash
# Krok 79: paczka dla poda (tylko to, czego potrzebuje longtrain.sh dla DOMAINS).
# Użycie: bash scripts/runpod/make_bundle.sh [wyjście.tgz]   (z korzenia repo)
set -e
cd "$(dirname "$0")/../.."
OUT=${1:-/tmp/bundle.tgz}
DOMAINS=${DOMAINS:-"cywilne mieszkaniowe"}
old() { case $1 in cywilne) echo 550;; mieszkaniowe) echo 413;; esac; }
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/processed" "$STAGE/data/finetune" "$STAGE/data/distill" "$STAGE/k78"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp data/*.json "$STAGE/data/"
cp requirements-common.txt requirements-cuda.txt "$STAGE/"
cp data/processed/all_articles.json "$STAGE/data/processed/"
for d in $DOMAINS; do
  cp -R data/finetune/${d}_distill "$STAGE/data/finetune/"
  mkdir -p "$STAGE/data/distill/$d"
  for f in eval_holdout.json questions.jsonl answers.jsonl prompts.jsonl; do
    [ -f data/distill/$d/$f ] && cp data/distill/$d/$f "$STAGE/data/distill/$d/"
  done
  cp data/processed/rag_index_${d}.npy data/processed/rag_index_${d}_meta.json "$STAGE/data/processed/"
  mkdir -p "$STAGE/k78/bielik11b-$d-distill"
  cp -R ../bielik-lora-adapters/cuda/krok78/bielik11b-$d-distill/checkpoint-$(old $d) "$STAGE/k78/bielik11b-$d-distill/"
done
# akty źródłowe (build_context czyta pełne ustępy artykułów)
python3 - "$STAGE" $DOMAINS <<'EOF'
import json, shutil, sys
from pathlib import Path
stage, domains = Path(sys.argv[1]), sys.argv[2:]
cfg = json.loads(Path("data/acts_config.json").read_text(encoding="utf-8"))
for e in cfg:
    ed = e.get("domains") or []
    ed = [ed] if isinstance(ed, str) else ed
    if set(ed) & set(domains) and e.get("short"):
        for suf in ("", "_history"):
            p = Path("data/processed") / f"{e['short']}{suf}.json"
            if p.exists():
                shutil.copy(p, stage / "data" / "processed" / p.name)
EOF
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"
ls -la "$OUT"
