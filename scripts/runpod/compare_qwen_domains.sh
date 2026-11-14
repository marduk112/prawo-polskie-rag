#!/bin/bash
# Krok 109: odpowiedzi Qwena3.8-27B-FP8 na gotowych kontekstach (eval_vllm.json) kilku dziedzin + sędzia (Qwen) wobec 'base' (Bielik).
# Wymaga: setup.sh (Qwen pobrany), RUNPOD_API_KEY i RUNPOD_POD_ID w środowisku. Wszystko w /workspace/work.
# Użycie: bash scripts/runpod/compare_qwen_domains.sh <dziedzina> [<dziedzina> ...]
# Markery: ALL_DONE, FAILED. Strażnik: twardy limit 100 min i stop poda GRACE_MIN (domyślnie 20) min po ALL_DONE/FAILED.
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
QWEN=qwen3.8-27b-fp8
MAX_MIN=100; GRACE_MIN=${GRACE_MIN:-20}
DOMAINS="$@"
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog_qd.log; }
  stop_pod() {
    log "STOP: $1"
    curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog_qd.log 2>&1
    sleep 600
  }
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
[ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ] || mark "UWAGA: brak RUNPOD_API_KEY/RUNPOD_POD_ID -- strażnik nie zatrzyma poda"

mark "START dziedziny: $DOMAINS"
nohup vllm serve Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 8000 --served-model-name $QWEN --max-model-len 24576 \
  --reasoning-parser qwen3 --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.85 \
  > logs/vllm_8000.log 2>&1 &
echo $! > logs/vllm_8000.pid
until curl -sf localhost:8000/v1/models >/dev/null; do
  sleep 5
  kill -0 "$(cat logs/vllm_8000.pid)" 2>/dev/null || { tail -30 logs/vllm_8000.log; fail "VLLM_DIED"; }
done
mark "vllm gotowy"

for D in $DOMAINS; do
  F=data/distill/$D
  $PY -c "import json,sys; json.dump(list(json.load(open('$F/eval_vllm.json'))['contexts']), open('$F/qd_questions.json','w'), ensure_ascii=False)"
  $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
    --questions-file $F/qd_questions.json --out-file eval_vllm.json --max-tokens 3000 --reasoning-effort low --concurrency 48 \
    > logs/qd_eval_$D.log 2>&1 || fail "odpowiedzi Qwena $D"
  mark "$D odpowiedzi: $(tail -1 logs/qd_eval_$D.log)"
  cp $F/eval_vllm.json RESULT_eval_$D.json
done
for D in $DOMAINS; do
  $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
    --eval-file eval_vllm.json --out-file judge_qwen.json --max-tokens 9000 --concurrency 48 \
    > logs/qd_judge_$D.log 2>&1 || fail "sedzia $D"
  mark "$D SEDZIA: $(grep '^\[' logs/qd_judge_$D.log | tail -1)"
  cp data/distill/$D/judge_qwen.json RESULT_judge_$D.json
done
touch ALL_DONE
mark "PIPELINE_DONE"
