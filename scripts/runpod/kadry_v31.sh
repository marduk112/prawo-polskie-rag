#!/bin/bash
# Krok 121: adapter LoRA dla Bielika v3.1 (kadry_i_place) -- dokłada odpowiedzi Qwena, trenuje, ocenia i sądzi (obie kolejności A/B).
# Wszystko w /workspace/work. Markery: DATA_DONE, TRAIN_DONE, ALL_DONE, FAILED. Strażnik: limit MAX_MIN, stop poda GRACE_MIN po końcu.
# Wymaga setup_k121.sh (SETUP_DONE), RUNPOD_API_KEY/RUNPOD_POD_ID (z /etc/rp_environment).
cd /workspace/work
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/train/bin/python
D=kadry_i_place; DD=scripts/distill_qwen.py
CTX=${CTX:-1500}; NEW_ANS=${NEW_ANS:-750}; MAXLEN=${MAXLEN:-12288}
MAX_MIN=${MAX_MIN:-210}; GRACE_MIN=${GRACE_MIN:-15}
QWEN=qwen3.8-27b-fp8; B31=speakleash/Bielik-11B-v3.1-Instruct; B30=speakleash/Bielik-11B-v3.0-Instruct
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog.log 2>&1; sleep 600; }
  while true; do
    now=$(date +%s)
    for f in ALL_DONE FAILED SETUP_FAILED; do
      [ -f $f ] && [ $(( now - $(stat -c %Y $f) )) -gt $(( GRACE_MIN * 60 )) ] && stop_pod "$f od ponad ${GRACE_MIN} min"
    done
    [ $(( now - start )) -gt $(( MAX_MIN * 60 )) ] && stop_pod "limit ${MAX_MIN} min"
    sleep 30
  done
) > /dev/null 2>&1 &
disown
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
unserve() { kill "$(cat logs/vllm_$1.pid)"; while kill -0 "$(cat logs/vllm_$1.pid)" 2>/dev/null; do sleep 2; done; sleep 5; mark "vllm $1 zatrzymany"; }
until [ -f SETUP_DONE ] || [ -f SETUP_FAILED ]; do sleep 10; done
[ -f SETUP_FAILED ] && fail "setup"
mark "START ($(wc -l < data/distill/$D/answers.jsonl) odpowiedzi na wejściu)"

# --- Faza 1: konteksty (ekspansja zapytania Bielikiem v3.0, jak w produkcyjnym chat.py) ---
serve $B30 8001 --served-model-name bielik-11b --max-model-len 8192 --gpu-memory-utilization 0.45 --max-num-seqs 32
$PY $DD contexts --expand-url http://127.0.0.1:8001/v1 --domain $D --concurrency 24 --sample $CTX > logs/c_$D.log 2>&1 || fail "contexts"
mark "konteksty: $(wc -l < data/distill/$D/prompts.jsonl)"
unserve 8001

# --- Faza 2: dodatkowe odpowiedzi Qwena (--limit = tylko nowe pytania, bez już policzonych) ---
serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name $QWEN --max-model-len 24576 --max-num-seqs 64 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.92
$PY $DD answers --url http://127.0.0.1:8000/v1 --domain $D --concurrency 48 --limit $NEW_ANS > logs/a_$D.log 2>&1 || fail "answers"
$PY -c "
import json, pathlib
p = pathlib.Path('data/distill/$D/answers.jsonl'); ok = []
for l in p.read_text(encoding='utf-8').splitlines():
    try: json.loads(l); ok.append(l)
    except json.JSONDecodeError: pass
p.write_text(''.join(x + '\n' for x in ok), encoding='utf-8')"
mark "odpowiedzi: $(wc -l < data/distill/$D/answers.jsonl)"
unserve 8000

# --- Faza 3: zbiór (bez pytań bliskich ewaluacji) + trening na Bieliku v3.1 BF16 ---
$PY $DD dataset --domain $D --tokenizer $B31 --max-example-tokens $MAXLEN > logs/d_$D.log 2>&1 || fail "dataset"
tail -2 logs/d_$D.log; touch DATA_DONE
n=$(wc -l < data/finetune/${D}_distill/train.jsonl); steps=$(( n * 12 / 10 ))
mark "TRENING: $n przykładów, $steps kroków"
$PY scripts/train_lora_cuda.py --model $B31 --data-dir data/finetune/${D}_distill --output-dir adapters-cuda/bielik31-$D \
  --quantization none --batch-size 1 --max-steps $steps --eval-steps 50 --save-steps 50 \
  --max-eval-samples 20 --learning-rate 1e-5 --max-length $MAXLEN > logs/train.log 2>&1 || fail "trening"
mark "TRENING_OK"; touch TRAIN_DONE
BEST=$($PY - <<PYX
import json, glob
best = None
for ck in sorted(glob.glob("adapters-cuda/bielik31-$D/checkpoint-*")):
    st = json.load(open(ck + "/trainer_state.json"))
    ev = [(h["eval_loss"], h["step"]) for h in st["log_history"] if "eval_loss" in h]
    if ev and (best is None or min(ev)[0] < best[0]): best = (min(ev)[0], min(ev)[1])
print(best[1] if best else "")
PYX
)
[ -d adapters-cuda/bielik31-$D/checkpoint-$BEST ] || fail "brak checkpointu min val loss ($BEST)"
mark "najlepszy checkpoint (min val loss): $BEST"; echo $BEST > BEST_STEP

# --- Faza 4: odpowiedzi bazy v3.1 BF16 i adaptera na 76 pytaniach (konteksty i Qwen jak w eval_models.json) ---
$PY - <<PYX
import json
src = json.load(open("data/distill/$D/eval_models.json", encoding="utf-8"))
json.dump({"contexts": src["contexts"], "answers": {"$QWEN": src["answers"]["$QWEN"]}},
          open("data/distill/$D/eval_k121.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYX
serve $B31 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48 --gpu-memory-utilization 0.90 \
  --enable-lora --max-lora-rank 8 --max-loras 1 --lora-modules "ad$BEST=adapters-cuda/bielik31-$D/checkpoint-$BEST"
$PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $D --variants base ad$BEST \
  --questions-file data/distill/$D/compare_questions.json --out-file eval_k121.json --max-tokens 1000 --concurrency 24 > logs/eval.log 2>&1 || fail "ewaluacja"
mark "EVAL: $(tail -2 logs/eval.log | tr '\n' ' ')"
cp data/distill/$D/eval_k121.json RESULT_eval_k121.json
unserve 8001
$PY - <<PYX
import json
ev = json.load(open("data/distill/$D/eval_k121.json", encoding="utf-8"))
a = ev["answers"]
for name, own in (("ad_vs_qwen", "ad$BEST"), ("base_vs_qwen", "base")):
    json.dump({"contexts": ev["contexts"], "answers": {"base": a[own], "$QWEN": a["$QWEN"]}},
              open(f"data/distill/$D/eval_k121_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYX

# --- Faza 5: sędzia Qwen, obie kolejności A/B ---
serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name $QWEN --max-model-len 24576 --max-num-seqs 64 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.90
J="$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --both-orders --max-tokens 9000 --concurrency 48"
$J --variants ad$BEST --eval-file eval_k121.json --out-file judge_k121_ad_vs_base.json > logs/judge1.log 2>&1 || mark "BLAD sędzia ad_vs_base"
mark "adapter vs baza v3.1: $(grep '^\[' logs/judge1.log | tail -1)"
$J --variants $QWEN --eval-file eval_k121_ad_vs_qwen.json --out-file judge_k121_ad_vs_qwen.json > logs/judge2.log 2>&1 || mark "BLAD sędzia ad_vs_qwen"
mark "adapter vs Qwen: $(grep '^\[' logs/judge2.log | tail -1)"
$J --variants $QWEN --eval-file eval_k121_base_vs_qwen.json --out-file judge_k121_base_vs_qwen.json > logs/judge3.log 2>&1 || mark "BLAD sędzia base_vs_qwen"
mark "baza v3.1 vs Qwen: $(grep '^\[' logs/judge3.log | tail -1)"
for f in judge_k121_ad_vs_base judge_k121_ad_vs_qwen judge_k121_base_vs_qwen; do cp data/distill/$D/$f.json RESULT_$f.json 2>/dev/null; done
tar czf RESULT_adapter.tgz -C adapters-cuda/bielik31-$D checkpoint-$BEST
touch ALL_DONE
mark "PIPELINE_DONE"
