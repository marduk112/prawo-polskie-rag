#!/bin/bash
# Krok 76 (druga próba): RTX PRO 6000 (96 GB). Qwen i Bielik naraz w fazie 1,
# potem Qwen sam z całą pamięcią na KV (odpowiedzi -- wąskie gardło), potem trening.
# Wszystko w /workspace (trwały dysk -- przeżywa stop poda).
# UWAGA (Krok 76): w przebiegu z 2026-10-17 ten skrypt wykonał tylko fazę 1;
# fazy 2-3 przejął pipeline2.sh (Qwen 24576, przykłady do 16384 tokenów).
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
B_ANS=${B_ANS:-600}; C_CTX=${C_CTX:-1400}; C_ANS=${C_ANS:-900}
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }

serve() {  # $1 model, $2 port, reszta: argumenty vllm
  local model=$1 port=$2; shift 2
  nohup vllm serve "$model" --host 127.0.0.1 --port "$port" "$@" > logs/vllm_$port.log 2>&1 &
  echo $! > logs/vllm_$port.pid
  until curl -sf localhost:$port/v1/models >/dev/null; do
    sleep 5
    kill -0 "$(cat logs/vllm_$port.pid)" 2>/dev/null || { tail -30 logs/vllm_$port.log; fail "VLLM_DIED $model"; }
  done
  mark "vllm gotowy: $model (port $port)"
}
unserve() { kill "$(cat logs/vllm_$1.pid)"; while kill -0 "$(cat logs/vllm_$1.pid)" 2>/dev/null; do sleep 2; done; sleep 5; mark "vllm $1 zatrzymany"; }

Q="--url http://127.0.0.1:8000/v1"
QWEN=(Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 16384 --max-num-seqs 64
      --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}')
D="scripts/distill_qwen.py"

# --- Faza 1: Qwen (0.58) + Bielik (0.27) naraz ---
mark "FAZA 1"
serve "${QWEN[@]}" --gpu-memory-utilization 0.58
$PY $D questions $Q --domain budowlane --per-article 4 --concurrency 48 > logs/q_budowlane.log 2>&1 || fail "questions budowlane"
mark "pytania budowlane: $(wc -l < data/distill/budowlane/questions.jsonl)"
# Bielik pobierany osobno (token HF) -- czekamy, aż proces pobierania się skończy.
until [ -f logs/dl_bielik.log ] && ! pgrep -f "hf download speakleash" >/dev/null; do sleep 15; done
grep -qiE "error|denied|401|403" logs/dl_bielik.log && fail "pobieranie Bielika: $(tail -c 300 logs/dl_bielik.log)"
mark "Bielik pobrany"
serve speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 8192 \
  --gpu-memory-utilization 0.27 --max-num-seqs 32

# Qwen: pytania cywilne  ||  Bielik: konteksty budowlane
$PY $D questions $Q --domain cywilne --per-article 2 --concurrency 48 > logs/q_cywilne.log 2>&1 &
QC=$!
$PY $D contexts --expand-url http://127.0.0.1:8001/v1 --domain budowlane --concurrency 24 > logs/c_budowlane.log 2>&1 || fail "contexts budowlane"
mark "konteksty budowlane: $(wc -l < data/distill/budowlane/prompts.jsonl)"
wait $QC || fail "questions cywilne"
mark "pytania cywilne: $(wc -l < data/distill/cywilne/questions.jsonl)"

# Qwen: odpowiedzi budowlane (część)  ||  Bielik: konteksty cywilne
$PY $D answers $Q --domain budowlane --concurrency 24 --sample $B_ANS > logs/a1_budowlane.log 2>&1 &
AB=$!
$PY $D contexts --expand-url http://127.0.0.1:8001/v1 --domain cywilne --concurrency 24 --sample $C_CTX > logs/c_cywilne.log 2>&1 || fail "contexts cywilne"
mark "konteksty cywilne: $(wc -l < data/distill/cywilne/prompts.jsonl)"
kill $AB 2>/dev/null; wait $AB 2>/dev/null
# Przerwany proces mógł zostawić uciętą ostatnią linię -- dopisek po niej zepsułby plik.
$PY -c "
import json, pathlib
p = pathlib.Path('data/distill/budowlane/answers.jsonl')
if p.exists():
    ok = []
    for l in p.read_text(encoding='utf-8').splitlines():
        try: json.loads(l); ok.append(l)
        except json.JSONDecodeError: pass
    p.write_text(''.join(x + '\n' for x in ok), encoding='utf-8')
"
mark "odpowiedzi budowlane po fazie 1: $(wc -l < data/distill/budowlane/answers.jsonl 2>/dev/null)"
unserve 8001; unserve 8000

# --- Faza 2: sam Qwen z ~0.92 pamięci (KV na kilkadziesiąt promptów po 11 tys. tokenów) ---
mark "FAZA 2"
serve "${QWEN[@]}" --gpu-memory-utilization 0.92
$PY $D answers $Q --domain budowlane --concurrency 48 --sample $B_ANS > logs/a2_budowlane.log 2>&1 || fail "answers budowlane"
mark "odpowiedzi budowlane: $(wc -l < data/distill/budowlane/answers.jsonl)"
$PY $D answers $Q --domain cywilne --concurrency 48 --sample $C_ANS > logs/a_cywilne.log 2>&1 || fail "answers cywilne"
mark "odpowiedzi cywilne: $(wc -l < data/distill/cywilne/answers.jsonl)"
unserve 8000

# --- Faza 3: zbiory + trening (bf16, cała karta) ---
mark "FAZA 3"
for d in budowlane cywilne; do
  $PY $D dataset --domain $d --tokenizer speakleash/Bielik-11B-v3.0-Instruct > logs/d_$d.log 2>&1 || fail "dataset $d"
  tail -2 logs/d_$d.log
done
touch DATA_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in budowlane cywilne; do
  n=$(wc -l < data/finetune/${d}_distill/train.jsonl)
  steps=$(( n * 12 / 10 ))  # ~1,2 epoki -- w pilocie minimum val loss wypadło po ~1 epoce
  mark "TRENING $d: $n przykładów, $steps kroków"
  $PY scripts/train_lora_cuda.py --model speakleash/Bielik-11B-v3.0-Instruct \
    --data-dir data/finetune/${d}_distill --output-dir adapters-cuda/bielik11b-${d}-distill \
    --quantization none --batch-size 1 --max-steps $steps --eval-steps 50 --save-steps 50 \
    --max-eval-samples 20 --learning-rate 1e-5 > logs/train_$d.log 2>&1 || fail "trening $d"
  mark "TRENING_OK $d"
done
touch ALL_DONE
mark "PIPELINE_DONE"
