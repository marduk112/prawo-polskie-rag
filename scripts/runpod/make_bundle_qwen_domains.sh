#!/bin/bash
# Krok 109: paczka dla compare_qwen_domains.sh -- skrypty + eval_vllm.json (konteksty RAG i odpowiedzi Bielika base)
# wybranych dziedzin. Bez indeksów RAG i bez adapterów: kontekstów nie liczymy od nowa.
# Użycie: bash scripts/runpod/make_bundle_qwen_domains.sh <wyjście.tgz> <dziedzina> [<dziedzina> ...]
set -e
cd "$(dirname "$0")/../.."
OUT=${1:?wyjście.tgz}; shift
[ $# -ge 1 ] || { echo "podaj dziedziny"; exit 1; }
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/processed"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp data/*.json "$STAGE/data/"; cp requirements-common.txt requirements-cuda.txt "$STAGE/"
cp data/processed/all_articles.json "$STAGE/data/processed/"
for D in "$@"; do
  mkdir -p "$STAGE/data/distill/$D"
  cp "data/distill/$D/eval_vllm.json" "$STAGE/data/distill/$D/"
  [ -f "data/distill/$D/eval_vs_adapter.json" ] && cp "data/distill/$D/eval_vs_adapter.json" "$STAGE/data/distill/$D/"
done
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"; ls -la "$OUT"
