#!/bin/bash
# Bateria kadrowa z Qwenem na podzie: vLLM (Qwen3.8-27B-FP8) + run_kadry_battery.py. Strażnik zatrzymuje pod po ALL_DONE/FAILED + GRACE_MIN albo po MAX_MIN.
cd /workspace/work
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
MAX_MIN=${MAX_MIN:-50}; GRACE_MIN=${GRACE_MIN:-3}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog.log 2>&1; sleep 600; }
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
nohup vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 \
  --reasoning-parser qwen3 --max-num-seqs 16 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.80 \
  > logs/vllm.log 2>&1 &
echo $! > logs/vllm.pid
until curl -sf localhost:8000/v1/models >/dev/null; do
  sleep 5
  kill -0 "$(cat logs/vllm.pid)" 2>/dev/null || { tail -30 logs/vllm.log; fail "VLLM_DIED"; }
done
mark "vllm gotowy"
curl -s -m 180 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"Odpowiedz jednym slowem: ile to 2+2?"}],"max_tokens":300}' > logs/probe.json 2>&1
grep -q '"choices"' logs/probe.json || { tail -c 400 logs/probe.json; tail -20 logs/vllm.log; fail "VLLM_NIE_ODPOWIADA"; }
mark "test zapytania OK"
/opt/train/bin/python scripts/run_kadry_battery.py > logs/battery.log 2>&1 || { tail -20 logs/battery.log; fail "BATERIA"; }
tail -8 logs/battery.log
/opt/train/bin/python - <<'PY' || fail "WIELE_BLEDOW"
import json, sys
r = json.load(open("results.json")); e = sum(1 for x in r if x["error"])
print("bledy", e, "z", len(r)); sys.exit(1 if e > len(r) // 4 else 0)
PY
cp results.json RESULT_battery.json
touch ALL_DONE
mark "PIPELINE_DONE"
