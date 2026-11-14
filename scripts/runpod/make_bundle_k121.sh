#!/bin/bash
# Krok 121: paczka dla poda (kadry_i_place: adaptery Bielik v3.1). Użycie: bash scripts/runpod/make_bundle_k121.sh <wyjście.tgz>
set -e
cd "$(dirname "$0")/../.."
OUT=${1:?wyjście.tgz}; D=kadry_i_place
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/processed" "$STAGE/data/distill/$D"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp data/*.json "$STAGE/data/"; cp requirements-common.txt requirements-cuda.txt "$STAGE/"
cp data/processed/all_articles.json "$STAGE/data/processed/"  # indeks RAG buduje setup_k121.sh na podzie (łącze Mac->pod ok. 200 KB/s)
for f in questions.jsonl answers.jsonl eval_holdout.json compare_questions.json eval_models.json; do cp data/distill/$D/$f "$STAGE/data/distill/$D/"; done
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
rm -rf "$STAGE"; ls -la "$OUT"
