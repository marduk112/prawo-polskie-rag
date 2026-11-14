#!/bin/bash
# Krok 79: test dłuższego treningu (Krok 78: val loss wciąż spadał po 1 epoce).
# Dane destylacji już są (data/finetune/<d>_distill), więc bez fazy Qwena-generatora:
# Faza C: trening EPOCHS epok kolejno dla każdej dziedziny (bez gradient checkpointingu,
#         ~77 GiB szczyt -- dwóch treningów naraz nie zmieścimy), checkpoint co pół epoki.
# Faza D: Qwen (sędzia) + Bielik z adapterami: stary adapter z Kroku 78 (1 epoka) vs
#         najlepszy val loss / ~2 epoki / ostatni checkpoint, ewaluacja + sędzia (vs baza).
# Wszystko w /workspace. Markery: DATA_DONE (tu: trening skończony), ALL_DONE, FAILED (watchdog.sh).
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
MAXLEN=16384
EPOCHS=${EPOCHS:-3}
DOMAINS=${DOMAINS:-"cywilne mieszkaniowe"}
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }

# --- Faza C: trening ---
mark "FAZA C"
until [ -f logs/dl_bielik.log ] && ! pgrep -f "[h]f download speakleash" >/dev/null; do sleep 10; done
grep -qiE "error|denied|401|403" logs/dl_bielik.log && fail "pobieranie Bielika"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in $DOMAINS; do
  if grep -q "TRENING_OK $d" pipeline.out 2>/dev/null && [ -d adapters-cuda/bielik11b-$d-distill ]; then continue; fi
  n=$(wc -l < data/finetune/${d}_distill/train.jsonl)
  steps=$(( n * EPOCHS )); every=$(( n / 2 ))
  mark "TRENING $d: $n przykładów, $steps kroków ($EPOCHS epoki), checkpoint co $every"
  $PY scripts/train_lora_cuda.py --model speakleash/Bielik-11B-v3.0-Instruct \
    --data-dir data/finetune/${d}_distill --output-dir adapters-cuda/bielik11b-${d}-distill \
    --quantization none --gradient-checkpointing off --batch-size 1 --max-steps $steps \
    --eval-steps $every --save-steps $every --max-eval-samples 20 --learning-rate 1e-5 \
    --max-length $MAXLEN > logs/train_$d.log 2>&1 || fail "trening $d"
  mark "TRENING_OK $d"
done
unset PYTORCH_CUDA_ALLOC_CONF
touch DATA_DONE

# --- Faza D: ewaluacja i sędzia ---
mark "FAZA D"
claim() { mkdir "logs/claim_eval_$1" 2>/dev/null; }
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

pick() {  # najlepszy val loss, checkpoint najbliższy 2 epokom, ostatni
  $PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
d = Path(f"adapters-cuda/bielik11b-{sys.argv[1]}-distill")
n = sum(1 for _ in open(f"data/finetune/{sys.argv[1]}_distill/train.jsonl"))
ckpts = sorted(int(p.name.split("-")[1]) for p in d.glob("checkpoint-*"))
hist = json.loads((d / f"checkpoint-{ckpts[-1]}" / "trainer_state.json").read_text())["log_history"]
ev = {h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h and h["step"] in ckpts}
best = min(ev, key=ev.get)
two = min(ckpts, key=lambda s: abs(s - 2 * n))
print(" ".join(str(s) for s in sorted({best, two, ckpts[-1]})))
print(" ".join(f"{s}:{ev.get(s, float('nan')):.4f}" for s in ckpts), file=sys.stderr)
EOF
}

# stary adapter z Kroku 78 (1 epoka), jeden na dziedzinę -- punkt odniesienia
declare -A OLD=( [cywilne]=550 [mieszkaniowe]=413 )
mkdir -p adapters-vllm
MODS=()
for d in $DOMAINS; do
  steps=$(pick $d 2>logs/eval_pick_$d.log) || fail "wybór checkpointów $d"
  mark "$d: checkpointy $steps (val loss: $(cat logs/eval_pick_$d.log))"
  variants=""
  mk() {  # $1 nazwa wariantu, $2 katalog checkpointu PEFT
    local v=adapters-vllm/$1; mkdir -p $v
    cp $2/adapter_model.safetensors $v/
    $PY -c "import json; c=json.load(open('$2/adapter_config.json')); c.pop('layers_to_transform',None); c.pop('layers_pattern',None); json.dump(c,open('$v/adapter_config.json','w'),indent=1)"
    MODS+=("$1=$v"); variants="$variants $1"
  }
  mk $d-k78-iter${OLD[$d]} k78/bielik11b-$d-distill/checkpoint-${OLD[$d]}
  for s in $steps; do mk $d-iter$s adapters-cuda/bielik11b-$d-distill/checkpoint-$s; done
  echo "$variants" > logs/variants_$d
done

serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 --reasoning-parser qwen3 \
  --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.48
serve speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48 \
  --gpu-memory-utilization 0.40 --enable-lora --max-lora-rank 8 --max-loras 4 --max-cpu-loras 24 --lora-modules "${MODS[@]}"

eval_lane() {
  for d in $DOMAINS; do
    claim $d || continue
    $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $d --variants base $(cat logs/variants_$d) > logs/eval_$d.log 2>&1 || { touch FAILED; return 1; }
    mark "EVAL $d gotowe"
    $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $d --variants $(cat logs/variants_$d) > logs/judge_$d.log 2>&1 || { touch FAILED; return 1; }
    grep "^\[" logs/judge_$d.log | tail -6
  done
}
eval_lane & E1=$!
eval_lane & E2=$!
wait $E1 && wait $E2 || fail "ewaluacja"
unserve 8001; unserve 8000
touch ALL_DONE
mark "PIPELINE_DONE"
