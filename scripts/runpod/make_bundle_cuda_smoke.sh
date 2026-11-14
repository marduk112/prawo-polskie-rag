#!/bin/bash
# Krok 124: paczka dla cuda_smoke.sh (test chat_cuda.py z Bielikiem v3.1 bez adaptera na realnej karcie NVIDIA).
# Użycie: bash scripts/runpod/make_bundle_cuda_smoke.sh <wyjście.tgz>   (indeks RAG: tylko dziedzina cudzoziemcy)
set -e
cd "$(dirname "$0")/../.."
OUT=${1:?wyjście.tgz}
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data/processed"
cp -R scripts "$STAGE/"; rm -rf "$STAGE"/scripts/__pycache__
cp requirements-common.txt requirements-cuda.txt "$STAGE/"
cp data/*.json "$STAGE/data/"
cp data/processed/rag_index_cudzoziemcy.npy data/processed/rag_index_cudzoziemcy_meta.json "$STAGE/data/processed/"
tar czf "$OUT" -C "$STAGE" .
rm -rf "$STAGE"; ls -la "$OUT"
