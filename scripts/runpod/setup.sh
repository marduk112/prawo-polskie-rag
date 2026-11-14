#!/bin/bash
# Uruchamiane na podzie (vllm/vllm-openai). Bielik (gated) pobierany osobno z tokenem HF.
cd /workspace/work
tar xzf bundle.tgz --no-same-owner
mkdir -p logs data/distill
python3 -m venv --system-site-packages /opt/train > logs/venv.log 2>&1 || uv venv --system-site-packages /opt/train >> logs/venv.log 2>&1
if [ -x /opt/train/bin/pip ]; then PIPI="/opt/train/bin/pip install"; else PIPI="uv pip install -p /opt/train/bin/python"; fi
$PIPI sentence-transformers==5.7.0 pypdf trl peft datasets > logs/pip.log 2>&1 &
hf download Qwen/Qwen3.8-27B-FP8 > logs/dl_qwen.log 2>&1 &
hf download sdadas/mmlw-retrieval-roberta-large > logs/dl_emb.log 2>&1 &
hf download sdadas/polish-reranker-base-ranknet > logs/dl_rr.log 2>&1 &
wait
echo SETUP_DONE
