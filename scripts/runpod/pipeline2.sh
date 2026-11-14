#!/bin/bash
# Krok 76: dalszy ciąg pipeline.sh od końca fazy 1, z dłuższym kontekstem
# (Qwen 24576, przykłady treningowe do 16384 tokenów zamiast 11264 z MLX).
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
B_ANS=${B_ANS:-600}; C_ANS=${C_ANS:-900}; MAXLEN=16384
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
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
Q="--url http://127.0.0.1:8000/v1"
D="scripts/distill_qwen.py"

# Koniec fazy 1: czekamy na konteksty cywilne (proces z pipeline.sh), potem zatrzymujemy odpowiedzi fazy 1.
while pgrep -f "distill_qwen.py contexts" >/dev/null; do sleep 10; done
grep -qiE "Traceback" logs/c_cywilne.log && fail "contexts cywilne"
mark "konteksty cywilne: $(wc -l < data/distill/cywilne/prompts.jsonl)"
pkill -f "distill_qwen.py answers"; while pgrep -f "distill_qwen.py answers" >/dev/null; do sleep 2; done
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
mark "odpowiedzi budowlane po fazie 1: $(wc -l < data/distill/budowlane/answers.jsonl)"
unserve 8001; unserve 8000

mark "FAZA 2"
serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 --max-num-seqs 64 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.92
$PY $D answers $Q --domain budowlane --concurrency 48 --sample $B_ANS > logs/a2_budowlane.log 2>&1 || fail "answers budowlane"
mark "odpowiedzi budowlane: $(wc -l < data/distill/budowlane/answers.jsonl)"
$PY $D answers $Q --domain cywilne --concurrency 48 --sample $C_ANS > logs/a_cywilne.log 2>&1 || fail "answers cywilne"
mark "odpowiedzi cywilne: $(wc -l < data/distill/cywilne/answers.jsonl)"
unserve 8000

mark "FAZA 3"
for d in budowlane cywilne; do
  $PY $D dataset --domain $d --tokenizer speakleash/Bielik-11B-v3.0-Instruct --max-example-tokens $MAXLEN > logs/d_$d.log 2>&1 || fail "dataset $d"
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
    --max-eval-samples 20 --learning-rate 1e-5 --max-length $MAXLEN > logs/train_$d.log 2>&1 || fail "trening $d"
  mark "TRENING_OK $d"
done
touch ALL_DONE
mark "PIPELINE_DONE"
