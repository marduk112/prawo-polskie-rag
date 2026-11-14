#!/bin/bash
# Krok 112: sędzia (Qwen) porównuje odpowiedzi Qwena z odpowiedziami PRODUKCYJNEGO adaptera (nie base). Wymaga pliku
# data/distill/<d>/eval_vs_adapter.json ({contexts, answers: {base = odpowiedzi adaptera, qwen3.8-27b-fp8}}), setup.sh,
# RUNPOD_API_KEY/RUNPOD_POD_ID. W wynikach "adapter" = Qwen, "base" = adapter produkcyjny.
# Użycie: bash scripts/runpod/judge_qwen_vs_adapter.sh <dziedzina> [<dziedzina> ...]
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
QWEN=qwen3.8-27b-fp8
MAX_MIN=60; GRACE_MIN=${GRACE_MIN:-20}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog_va.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog_va.log 2>&1; sleep 600; }
  while true; do
    now=$(date +%s)
    for f in ALL_DONE FAILED; do
      [ -f $f ] && [ $(( now - $(stat -c %Y $f) )) -gt $(( GRACE_MIN * 60 )) ] && stop_pod "$f od ponad ${GRACE_MIN} min"
    done
    [ $(( now - start )) -gt $(( MAX_MIN * 60 )) ] && stop_pod "limit ${MAX_MIN} min"
    sleep 30
  done
) > /dev/null 2>&1 &
disown
nohup vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name $QWEN --max-model-len 24576 \
  --reasoning-parser qwen3 --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.85 \
  > logs/vllm_8000.log 2>&1 &
echo $! > logs/vllm_8000.pid
until curl -sf localhost:8000/v1/models >/dev/null; do
  sleep 5
  kill -0 "$(cat logs/vllm_8000.pid)" 2>/dev/null || { tail -30 logs/vllm_8000.log; fail "VLLM_DIED"; }
done
mark "vllm gotowy"
for D in "$@"; do
  $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
    --eval-file eval_vs_adapter.json --out-file judge_vs_adapter.json --max-tokens 9000 --concurrency 48 \
    > logs/va_judge_$D.log 2>&1 || fail "sedzia $D"
  mark "$D SEDZIA: $(grep '^\[' logs/va_judge_$D.log | tail -1)"
  cp data/distill/$D/judge_vs_adapter.json RESULT_judge_vs_adapter_$D.json
done
touch ALL_DONE
mark "PIPELINE_DONE"
