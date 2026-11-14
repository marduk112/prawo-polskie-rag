#!/bin/bash
# Krok 88: kadry_i_place -- produkcyjny adapter hedge-iter50 vs baza na większej próbie (77 pytań),
# sędzia Qwen z większym limitem myślenia. Qwen + Bielik z adapterem naraz (jak distill_eval.sh).
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
D=kadry_i_place
V=kadry-hedge50
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
until [ -f logs/dl_bielik.log ] && ! pgrep -f "[h]f download speakleash" >/dev/null; do sleep 10; done
grep -qiE "error|denied|401|403" logs/dl_bielik.log && fail "pobieranie Bielika"
mark "START"
serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 --reasoning-parser qwen3 \
  --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.48
serve speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48 \
  --gpu-memory-utilization 0.40 --enable-lora --max-lora-rank 8 --max-loras 1 --lora-modules "$V=adapters-vllm/$V"
Q=data/distill/$D/compare_questions.json
$PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $D --variants base $V \
  --questions-file $Q --out-file eval_cmp.json --concurrency 24 > logs/eval_cmp.log 2>&1 || fail "ewaluacja"
mark "EVAL gotowe: $(tail -2 logs/eval_cmp.log | tr '\n' ' ')"
cp data/distill/$D/eval_cmp.json RESULT_eval_cmp.json
$PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $D --variants $V \
  --eval-file eval_cmp.json --out-file judge_cmp.json --max-tokens 9000 --concurrency 32 > logs/judge_cmp.log 2>&1 || fail "sedzia"
mark "SEDZIA: $(grep '^\[' logs/judge_cmp.log | tail -2 | tr '\n' ' ')"
cp data/distill/$D/judge_cmp.json RESULT_judge_cmp.json
touch ALL_DONE
mark "PIPELINE_DONE"
