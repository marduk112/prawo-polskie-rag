#!/bin/bash
# Porównanie modeli: Bielik-11B-v3.0-Instruct (bez adaptera) vs Qwen3.8-27B-FP8 na 76 pytaniach
# kadry_i_place (data/distill/kadry_i_place/compare_questions.json). Paczka: make_bundle_compare.sh ... - ...
# Konteksty RAG (ekspansja zapytania Bielikiem v3.0) liczone raz w fazie 1 i użyte identycznie dla Qwena.
# Fazy SEKWENCYJNE (vLLM nie wystartuje obok innego procesu GPU): 1) Bielik, 2) Qwen + sędzia (Qwen).
# Wymaga: setup.sh (Qwen pobrany), pobrany Bielik (token HF wysłany ręcznie przez stdin SSH), RUNPOD_API_KEY i RUNPOD_POD_ID w środowisku.
# Wszystko w /workspace/work. Markery: ALL_DONE, FAILED. Strażnik w tle (poniżej): twardy limit 1h45
# oraz zatrzymanie poda GRACE_MIN (domyślnie 20) minut po ALL_DONE/FAILED -- czas na pobranie wyników.
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
D=kadry_i_place
Q=data/distill/$D/compare_questions.json
QWEN=qwen3.8-27b-fp8
MAX_MIN=105; GRACE_MIN=${GRACE_MIN:-20}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }

# Strażnik: osobny podproces (setsid), nie wymaga pkill -- kończy się razem ze stopem poda.
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog_cmp.log; }
  stop_pod() {
    log "STOP: $1"
    curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog_cmp.log 2>&1
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

serve() {
  local model=$1 port=$2; shift 2
  nohup vllm serve "$model" --host 127.0.0.1 --port "$port" "$@" > logs/vllm_$port.log 2>&1 &
  echo $! > logs/vllm_$port.pid
  until curl -sf localhost:$port/v1/models >/dev/null; do
    sleep 5
    kill -0 "$(cat logs/vllm_$port.pid)" 2>/dev/null || { tail -30 logs/vllm_$port.log; fail "VLLM_DIED $model"; }
  done
  mark "vllm gotowy: $model (port $port)"
}
unserve() {  # po PID z pliku (nie pkill -f: wzorzec byłby w linii poleceń tego skryptu)
  kill "$(cat logs/vllm_$1.pid)"
  while kill -0 "$(cat logs/vllm_$1.pid)" 2>/dev/null; do sleep 2; done
  sleep 10; mark "vllm $1 zatrzymany"
}

until [ -f logs/dl_bielik.log ] && ! pgrep -f "[h]f download speakleash" >/dev/null; do sleep 10; done
grep -qiE "error|denied|401|403" logs/dl_bielik.log && fail "pobieranie Bielika"
mark "START"

# Faza 1: Bielik v3.0 bez adaptera -- ekspansja zapytań, konteksty, odpowiedzi (max 500 tokenów jak chat.py)
serve speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48 \
  --gpu-memory-utilization 0.85
$PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $D --variants base \
  --questions-file $Q --out-file eval_models.json --concurrency 24 > logs/eval_bielik.log 2>&1 || fail "ewaluacja Bielika"
mark "BIELIK gotowy: $(tail -1 logs/eval_bielik.log)"
cp data/distill/$D/eval_models.json RESULT_eval_models_bielik.json
unserve 8001

# Faza 2: Qwen3.8-27B-FP8 -- odpowiedzi na te same konteksty (myślenie low, 3000 tokenów) + sędzia (też Qwen)
serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name $QWEN --max-model-len 24576 --reasoning-parser qwen3 \
  --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.85
$PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
  --questions-file $Q --out-file eval_models.json --max-tokens 3000 --reasoning-effort low --concurrency 32 \
  > logs/eval_qwen.log 2>&1 || fail "odpowiedzi Qwena"
mark "QWEN gotowy: $(tail -1 logs/eval_qwen.log)"
cp data/distill/$D/eval_models.json RESULT_eval_models.json
# Werdykty: "adapter" w wyniku = Qwen, "base" = Bielik (judge_distill.py zna tylko tę konwencję)
$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN \
  --eval-file eval_models.json --out-file judge_models.json --max-tokens 9000 --concurrency 32 > logs/judge_models.log 2>&1 || fail "sedzia"
mark "SEDZIA: $(grep '^\[' logs/judge_models.log | tail -2 | tr '\n' ' ')"
cp data/distill/$D/judge_models.json RESULT_judge_models.json
touch ALL_DONE
mark "PIPELINE_DONE"
