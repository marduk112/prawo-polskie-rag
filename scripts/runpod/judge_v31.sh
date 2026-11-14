#!/bin/bash
# Krok 114b: sędzia (Qwen, vLLM) dla Bielika v3.1 8-bit na kadrach: (1) v3.1 vs v3.0, (2) v3.1 vs Qwen. "base" = v3.1.
# Pliki: data/distill/kadry_i_place/eval_v31_vs_v30.json, eval_v31_vs_qwen.json. Wymaga setup.sh, RUNPOD_API_KEY/RUNPOD_POD_ID.
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
QWEN=qwen3.8-27b-fp8
D=kadry_i_place
MAX_MIN=60; GRACE_MIN=${GRACE_MIN:-15}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog_v31.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog_v31.log 2>&1; sleep 600; }
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
$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants bielik-v3.0 \
  --eval-file eval_v31_vs_v30.json --out-file judge_v31_vs_v30.json --max-tokens 9000 --concurrency 48 \
  > logs/judge_v30.log 2>&1 || fail "sedzia v30"
mark "v3.1 vs v3.0: $(grep '^\[' logs/judge_v30.log | tail -1)"
cp data/distill/$D/judge_v31_vs_v30.json RESULT_judge_v31_vs_v30.json
$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
  --eval-file eval_v31_vs_qwen.json --out-file judge_v31_vs_qwen.json --max-tokens 9000 --concurrency 48 \
  > logs/judge_qwen.log 2>&1 || fail "sedzia qwen"
mark "v3.1 vs Qwen: $(grep '^\[' logs/judge_qwen.log | tail -1)"
cp data/distill/$D/judge_v31_vs_qwen.json RESULT_judge_v31_vs_qwen.json
touch ALL_DONE
mark "PIPELINE_DONE"
