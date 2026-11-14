#!/bin/bash
# Krok 124 (na podzie): instaluje zależności CUDA z requirements-*.txt i uruchamia chat_cuda.py (Bielik v3.1, 4-bit nf4, BEZ adaptera).
# Wynik: /workspace/work/logs/smoke.log; znaczniki ALL_DONE / FAILED (czyta je watchdog.sh).
cd /workspace/work
export HF_HOME=/workspace/hf
mkdir -p logs "$HF_HOME"
tar xzf bundle.tgz --no-same-owner
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > logs/gpu.txt 2>&1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())" >> logs/gpu.txt 2>&1
# Obraz ubuntu2404 ma pythona "externally-managed" (PEP 668): --break-system-packages, żeby użyć torcha z obrazu (venv ściągałby własnego, ~3 GB)
python -m pip install -q --break-system-packages -r requirements-common.txt -r requirements-cuda.txt > logs/pip.log 2>&1 || { echo "pip failed" > FAILED; exit 1; }
{
  for Q in "Kiedy cudzoziemiec może dostać zezwolenie na pobyt stały w Polsce?" \
           "Czy mogę legalnie przebywać w Polsce, gdy czekam na decyzję w sprawie pobytu czasowego?" \
           "Ile wynosi kara za przejście na czerwonym świetle?"; do
    echo "=== PYTANIE: $Q"
    T0=$(date +%s)
    python scripts/chat_cuda.py --domain cudzoziemcy --prompt "$Q" 2>&1
    echo "[czas: $(( $(date +%s) - T0 )) s, szczyt VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)]"
  done
} > logs/smoke.log 2>&1
[ "$(grep -c "=== PYTANIE" logs/smoke.log)" = 3 ] && [ "$(grep -c "\[czas:" logs/smoke.log)" = 3 ] || { echo "nie wszystkie pytania uruchomione" > FAILED; exit 1; }
grep -q "Traceback" logs/smoke.log && { echo "traceback w smoke.log" > FAILED; exit 1; }
touch ALL_DONE
