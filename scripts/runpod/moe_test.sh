#!/bin/bash
# Krok 115: test małych MoE (Gemma-4-26B-A4B, Qwen3.6-35B-A3B) na 76 pytaniach kadr + sędzia Qwen3.8-27B (vLLM, jeden GPU, modele kolejno).
# Wymaga: setup.sh (Qwen 27B pobrany), bundle z data/distill/kadry_i_place/{eval_models.json, eval_v31_4bit_vs_8bit.json}, RUNPOD_API_KEY/RUNPOD_POD_ID.
# Wyniki: RESULT_*.json w /workspace/work; markery ALL_DONE/FAILED; czasy generowania w logs/timing.txt. W parach "base" = testowany model (MoE / v3.1 4-bit).
cd /workspace/work
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
QWEN=qwen3.8-27b-fp8
D=kadry_i_place; F=data/distill/$D
MAX_MIN=${MAX_MIN:-110}; GRACE_MIN=${GRACE_MIN:-15}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog_moe.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog_moe.log 2>&1; sleep 600; }
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

serve() {  # serve <repo> <served-name> [extra vllm args]
  local repo=$1 name=$2; shift 2
  nohup vllm serve $repo --host 127.0.0.1 --port 8000 --served-model-name $name --max-model-len 24576 --max-num-seqs 64 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.85 "$@" > logs/vllm_$name.log 2>&1 &
  echo $! > logs/vllm.pid
  until curl -sf localhost:8000/v1/models >/dev/null; do
    sleep 5
    kill -0 "$(cat logs/vllm.pid)" 2>/dev/null || { tail -20 logs/vllm_$name.log; return 1; }
  done
}
stop_vllm() { kill "$(cat logs/vllm.pid)" 2>/dev/null; sleep 15; pkill -9 -f "vllm serve" 2>/dev/null; sleep 5; }

gen_moe() {  # gen_moe <tag> <repo> <template-kwargs-json>
  local tag=$1 repo=$2 tk=$3
  mark "$tag: pobieranie $repo"
  hf download $repo > logs/dl_$tag.log 2>&1 || { mark "$tag: BLAD pobierania: $(tail -2 logs/dl_$tag.log)"; return 1; }
  $PY -c "import json;d=json.load(open('$F/eval_models.json'));json.dump({'contexts':d['contexts'],'answers':{}},open('$F/eval_moe_$tag.json','w'),ensure_ascii=False)"
  $PY -c "import json;json.dump(list(json.load(open('$F/eval_models.json'))['contexts']),open('$F/moe_questions.json','w'),ensure_ascii=False)"
  serve $repo $tag || { mark "$tag: BLAD startu vLLM"; stop_vllm; return 1; }
  mark "$tag: vllm gotowy"
  local t0=$(date +%s)
  $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8000/v1 --domain $D --variants $tag \
    --questions-file $F/moe_questions.json \
    --out-file eval_moe_$tag.json --max-tokens 3000 --template-kwargs "$tk" --concurrency 48 > logs/gen_$tag.log 2>&1
  echo "$tag generowanie $(( $(date +%s) - t0 )) s (concurrency 48, A100)" >> logs/timing.txt
  mark "$tag odpowiedzi: $(tail -1 logs/gen_$tag.log)"
  stop_vllm
  # para do sędziego: base = MoE, wariant = Qwen 27B (odpowiedzi z Kroku 106)
  $PY - <<PY
import json
d=json.load(open('$F/eval_moe_$tag.json')); q=json.load(open('$F/eval_models.json'))
json.dump({'contexts':d['contexts'],'answers':{'base':d['answers']['$tag'],'$QWEN':q['answers']['$QWEN']}},open('$F/pair_$tag.json','w'),ensure_ascii=False)
PY
  cp $F/eval_moe_$tag.json RESULT_eval_moe_$tag.json
}

gen_moe gemma4_26b_a4b google/gemma-4-26B-A4B-it '{"enable_thinking": false}'
gen_moe qwen36_35b_a3b Qwen/Qwen3.6-35B-A3B-FP8 '{"enable_thinking": false}'

mark "sędzia: start Qwen 27B"
serve Qwen/Qwen3.8-27B-FP8 $QWEN --reasoning-parser qwen3 || fail "VLLM Qwen"
for P in gemma4_26b_a4b qwen36_35b_a3b; do
  [ -f $F/pair_$P.json ] || { mark "$P: brak pary, pomijam"; continue; }
  $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $QWEN --eval-file pair_$P.json --out-file judge_$P.json \
    --max-tokens 9000 --concurrency 48 > logs/judge_$P.log 2>&1 || { mark "$P: BLAD sędziego"; continue; }
  mark "$P (MoE vs Qwen 27B): $(grep '^\[' logs/judge_$P.log | tail -1)"
  cp $F/judge_$P.json RESULT_judge_$P.json
done
$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants bielik-v3.1-8bit --eval-file eval_v31_4bit_vs_8bit.json --out-file judge_v31_4bit_vs_8bit.json \
  --max-tokens 9000 --concurrency 48 > logs/judge_4v8.log 2>&1 || fail "sędzia 4 vs 8"
mark "v3.1 4-bit vs 8-bit: $(grep '^\[' logs/judge_4v8.log | tail -1)"
cp $F/judge_v31_4bit_vs_8bit.json RESULT_judge_v31_4bit_vs_8bit.json
touch ALL_DONE
mark "PIPELINE_DONE"
