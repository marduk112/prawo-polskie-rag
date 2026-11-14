#!/bin/bash
# Krok 121: konfiguracja poda (vllm/vllm-openai). Token HF czytany z /workspace/.hf_token (wysłany przez SSH stdin), potem usuwany.
cd /workspace/work
export HF_HOME=/workspace/hf
mkdir -p logs data/distill "$HF_HOME"
tar xzf bundle.tgz --no-same-owner
python3 -m venv --system-site-packages /opt/train > logs/venv.log 2>&1 || uv venv --system-site-packages /opt/train >> logs/venv.log 2>&1
# Krok 121: pip na podzie bywał ~100 kB/s (utknął na pyarrow); uv pobiera równolegle, ale NIE widzi pakietów systemowych
# (venv --system-site-packages) i ściąga całego torcha + biblioteki CUDA (ok. 3 GB, kilka minut). Potem trzeba dopasować
# torchvision/torchaudio z systemu (od starszego torcha), inaczej `import transformers` wywala "operator torchvision::nms does not exist".
PIPI="uv pip install -p /opt/train/bin/python"
$PIPI sentence-transformers==5.7.0 pypdf trl peft datasets > logs/pip.log 2>&1 &
export HF_TOKEN=$(cat /workspace/.hf_token)
hf download Qwen/Qwen3.8-27B-FP8 > logs/dl_qwen.log 2>&1 &
hf download sdadas/mmlw-retrieval-roberta-large > logs/dl_emb.log 2>&1 &
hf download sdadas/polish-reranker-base-ranknet > logs/dl_rr.log 2>&1 &
hf download speakleash/Bielik-11B-v3.0-Instruct > logs/dl_b30.log 2>&1 &
hf download speakleash/Bielik-11B-v3.1-Instruct > logs/dl_b31.log 2>&1 &
wait
uv pip install -p /opt/train/bin/python --no-deps -U torchvision torchaudio >> logs/pip.log 2>&1
rm -f /workspace/.hf_token
# Indeks RAG kadr budowany na podzie (ta sama konfiguracja co lokalny: z historią wersji, oczekiwane 31009 wektorów)
HF_HUB_OFFLINE=1 /opt/train/bin/python scripts/build_rag_index.py --domain kadry_i_place --include-history > logs/build_index.log 2>&1 || { echo "SETUP_FAILED: indeks"; touch SETUP_FAILED; exit 1; }
/opt/train/bin/python -c "import numpy as np; a=np.load('data/processed/rag_index_kadry_i_place.npy', mmap_mode='r'); print('indeks', a.shape); assert a.shape == (31009, 1024), a.shape" >> logs/build_index.log 2>&1 || { echo "SETUP_FAILED: kształt indeksu"; touch SETUP_FAILED; exit 1; }
grep -qiE "error|denied|401|403" logs/dl_b31.log logs/dl_b30.log && { echo "SETUP_FAILED: $(tail -c 300 logs/dl_b31.log)"; touch SETUP_FAILED; exit 1; }
echo SETUP_DONE; touch SETUP_DONE
