#!/bin/bash
cd /workspace/work
export HF_HOME=/workspace/hf
mkdir -p logs
python3 -m venv --system-site-packages /opt/train > logs/venv.log 2>&1 || uv venv --system-site-packages /opt/train >> logs/venv.log 2>&1
if [ -x /opt/train/bin/pip ]; then PIPI="/opt/train/bin/pip install"; else PIPI="uv pip install -p /opt/train/bin/python"; fi
$PIPI sentence-transformers==5.7.0 pypdf pdfplumber==0.11.10 requests psutil > logs/pip.log 2>&1 &
hf download Qwen/Qwen3.8-27B-FP8 > logs/dl_qwen.log 2>&1 &
hf download sdadas/mmlw-retrieval-roberta-large > logs/dl_emb.log 2>&1 &
hf download sdadas/polish-reranker-base-ranknet > logs/dl_rr.log 2>&1 &
wait
echo SETUP_DONE
