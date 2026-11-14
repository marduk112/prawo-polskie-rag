#!/bin/bash
# Krok 78: destylacja kilku dziedzin w jednym przebiegu poda (RTX PRO 6000, 96 GB),
# z maksymalnym wykorzystaniem karty:
# Faza A: Qwen (pytania) || Bielik (holdout + konteksty, dwa tory dziedzin naraz).
# Faza B: sam Qwen z całą pamięcią na KV, 96 sekwencji -- odpowiedzi w dwóch torach;
#         zbiór każdej dziedziny od razu po jej odpowiedziach (CPU).
# Faza C: trening bf16 bez gradient checkpointingu (~2x szybciej, Krok 78), ~1 epoka.
# Faza D: scripts/runpod/distill_eval.sh -- Qwen + Bielik z adapterami: ewaluacja,
#         sędzia, bateria Qwena. (vLLM startujący RÓWNOLEGLE z treningiem nie dostaje
#         pamięci na KV -- profilowanie liczy mu przyrost pamięci treningu, Krok 78.)
# Tory biorą dziedziny ze wspólnej kolejki (claim), żeby żaden nie stał bezczynnie.
# Wszystko w /workspace (trwały dysk). Markery: DATA_DONE, ALL_DONE, FAILED (watchdog.sh).
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
D=scripts/distill_qwen.py
MAXLEN=16384
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }

# dziedzina:pytania/artykuł (0 = już są):konteksty (0 = wszystkie):odpowiedzi
LANE1="budowlane:0:0:500 kadry_i_place:2:1100:700 rodzinne:4:0:500 ruch_drogowy:3:0:500"
LANE2="cywilne:0:1100:700 podatkowe:2:1100:700 mieszkaniowe:6:0:500 wlasnosc_przemyslowa:2:0:500"
DOMAINS="$LANE1 $LANE2"
field() { echo "$1" | cut -d: -f$2; }
claim() { mkdir "logs/claim_$1_$2" 2>/dev/null; }  # atomowe "biorę dziedzinę $2 w fazie $1"

serve() {  # $1 model, $2 port, reszta: argumenty vllm
  local model=$1 port=$2; shift 2
  if curl -sf localhost:$port/v1/models >/dev/null; then  # wznowienie -- serwer z poprzedniego startu
    mark "vllm już działa na porcie $port (pid $(cat logs/vllm_$port.pid))"; return
  fi
  nohup vllm serve "$model" --host 127.0.0.1 --port "$port" "$@" > logs/vllm_$port.log 2>&1 &
  echo $! > logs/vllm_$port.pid
  until curl -sf localhost:$port/v1/models >/dev/null; do
    sleep 5
    kill -0 "$(cat logs/vllm_$port.pid)" 2>/dev/null || { tail -30 logs/vllm_$port.log; fail "VLLM_DIED $model"; }
  done
  mark "vllm gotowy: $model (port $port)"
}
unserve() { kill "$(cat logs/vllm_$1.pid)"; while kill -0 "$(cat logs/vllm_$1.pid)" 2>/dev/null; do sleep 2; done; sleep 5; mark "vllm $1 zatrzymany"; }
QWEN="Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 --reasoning-parser qwen3"
BIELIK="speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48"
NOMM='{"image":0,"video":0}'
Q="--url http://127.0.0.1:8000/v1"

# --- Faza A ---
mark "FAZA A"
serve $QWEN --max-num-seqs 64 --limit-mm-per-prompt "$NOMM" --gpu-memory-utilization 0.56
until [ -f logs/dl_bielik.log ] && ! pgrep -f "[h]f download speakleash" >/dev/null; do sleep 15; done
grep -qiE "error|denied|401|403" logs/dl_bielik.log && fail "pobieranie Bielika"
serve $BIELIK --gpu-memory-utilization 0.30

questions_worker() {
  for e in $DOMAINS; do
    d=$(field $e 1); n=$(field $e 2)
    if [ "$n" != 0 ]; then
      $PY $D questions $Q --domain $d --per-article $n --concurrency 64 > logs/q_$d.log 2>&1 || { touch FAILED; return 1; }
      mark "pytania $d: $(wc -l < data/distill/$d/questions.jsonl)"
    fi
    touch logs/Q_DONE_$d
  done
}
contexts_lane() {
  for e in $DOMAINS; do
    d=$(field $e 1); c=$(field $e 3)
    claim ctx $d || continue
    until [ -f logs/Q_DONE_$d ]; do [ -f FAILED ] && return 1; sleep 10; done
    $PY $D holdout --domain $d --articles 12 > logs/h_$d.log 2>&1 || { touch FAILED; return 1; }
    $PY $D contexts --expand-url http://127.0.0.1:8001/v1 --domain $d --concurrency 24 --sample $c > logs/c_$d.log 2>&1 || { touch FAILED; return 1; }
    mark "konteksty $d: $(wc -l < data/distill/$d/prompts.jsonl)"
  done
}
questions_worker & QW=$!
contexts_lane & C1=$!
contexts_lane & C2=$!
wait $QW && wait $C1 && wait $C2 || fail "faza A"
unserve 8001; unserve 8000

# --- Faza B ---
mark "FAZA B"
serve $QWEN --max-num-seqs 96 --limit-mm-per-prompt "$NOMM" --gpu-memory-utilization 0.92
answers_lane() {
  for e in $DOMAINS; do
    d=$(field $e 1); a=$(field $e 4)
    claim ans $d || continue
    $PY $D answers $Q --domain $d --concurrency 48 --sample $a > logs/a_$d.log 2>&1 || { touch FAILED; return 1; }
    mark "odpowiedzi $d: $(wc -l < data/distill/$d/answers.jsonl)"
    $PY $D dataset --domain $d --tokenizer speakleash/Bielik-11B-v3.0-Instruct --max-example-tokens $MAXLEN > logs/d_$d.log 2>&1 || { touch FAILED; return 1; }
    tail -1 logs/d_$d.log
  done
}
answers_lane & A1=$!
answers_lane & A2=$!
wait $A1 && wait $A2 || fail "faza B"
unserve 8000
touch DATA_DONE

# --- Faza C: trening ---
mark "FAZA C"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for e in $DOMAINS; do
  d=$(field $e 1)
  n=$(wc -l < data/finetune/${d}_distill/train.jsonl)
  mark "TRENING $d: $n przykładów, $n kroków"
  $PY scripts/train_lora_cuda.py --model speakleash/Bielik-11B-v3.0-Instruct \
    --data-dir data/finetune/${d}_distill --output-dir adapters-cuda/bielik11b-${d}-distill \
    --quantization none --gradient-checkpointing off --batch-size 1 --max-steps $n --eval-steps 50 \
    --save-steps 50 --max-eval-samples 20 --learning-rate 1e-5 --max-length $MAXLEN > logs/train_$d.log 2>&1 \
    || fail "trening $d"
  mark "TRENING_OK $d"
done
unset PYTORCH_CUDA_ALLOC_CONF

# --- Faza D: ewaluacja, sędzia, bateria ---
DOMAINS="$DOMAINS" exec bash scripts/runpod/distill_eval.sh
