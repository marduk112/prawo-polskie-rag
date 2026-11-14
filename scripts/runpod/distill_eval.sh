#!/bin/bash
# Krok 78: ewaluacja po treningu (zamiennik fazy C-ewaluacja/D z distill_multi.sh).
# Bielik w vLLM startujący RÓWNOLEGLE z treningiem nie dostał pamięci na KV:
# profilowanie vLLM przypisuje mu przyrost pamięci zajmowanej w tym czasie przez
# trening. Dwa statyczne serwery naraz (jak w fazie A) działają -- więc: czekamy
# na koniec treningu, potem Qwen + Bielik ze wszystkimi adapterami, ewaluacja w
# dwóch torach, sędzia zaraz po ewaluacji dziedziny, bateria Qwena równolegle.
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
# Lista dziedzin: z distill_multi.sh (DOMAINS "d:..:..:.."), inaczej domyślna z Kroku 78.
ALL=$(for e in ${DOMAINS:-budowlane kadry_i_place rodzinne ruch_drogowy cywilne podatkowe mieszkaniowe wlasnosc_przemyslowa}; do echo "${e%%:*}"; done)
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

pick() {  # najlepszy val loss + ostatni checkpoint
  $PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
d = Path(f"adapters-cuda/bielik11b-{sys.argv[1]}-distill")
ckpts = sorted(int(p.name.split("-")[1]) for p in d.glob("checkpoint-*"))
hist = json.loads((d / f"checkpoint-{ckpts[-1]}" / "trainer_state.json").read_text())["log_history"]
ev = {h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h and h["step"] in ckpts}
best = min(ev, key=ev.get)
print(" ".join(str(s) for s in sorted({best, ckpts[-1]})))
print(" ".join(f"{s}:{ev.get(s, float('nan')):.4f}" for s in ckpts), file=sys.stderr)
EOF
}

for d in $ALL; do
  until grep -q "TRENING_OK $d" pipeline.out; do
    pgrep -f "[t]rain_lora_cuda.py" >/dev/null || sleep 30
    grep -q "TRENING_OK $d" pipeline.out || pgrep -f "[t]rain_lora_cuda.py" >/dev/null || fail "trening $d przerwany"
    sleep 30
  done
done
mark "FAZA D (ewaluacja po treningu)"

mkdir -p adapters-vllm adapters-mlx
MODS=()
for d in $ALL; do
  steps=$(pick $d 2>logs/eval_pick_$d.log) || fail "wybór checkpointów $d"
  mark "$d: checkpointy $steps (val loss: $(cat logs/eval_pick_$d.log))"
  variants=""
  for s in $steps; do
    src=adapters-cuda/bielik11b-$d-distill/checkpoint-$s; v=adapters-vllm/$d-iter$s; mkdir -p $v
    cp $src/adapter_model.safetensors $v/
    $PY -c "import json; c=json.load(open('$src/adapter_config.json')); c.pop('layers_to_transform'); c.pop('layers_pattern'); json.dump(c,open('$v/adapter_config.json','w'),indent=1)"
    $PY scripts/convert_lora.py peft-to-mlx $src adapters-mlx/bielik11b-$d-distill-lora-iter$s >> logs/convert.log 2>&1 || fail "konwersja $d $s"
    MODS+=("$d-iter$s=$v"); variants="$variants $d-iter$s"
  done
  echo "$variants" > logs/variants_$d
done

serve Qwen/Qwen3.8-27B-FP8 8000 --served-model-name qwen3.8-27b-fp8 --max-model-len 24576 --reasoning-parser qwen3 \
  --max-num-seqs 64 --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.48
serve speakleash/Bielik-11B-v3.0-Instruct 8001 --served-model-name bielik-11b --max-model-len 20480 --max-num-seqs 48 \
  --gpu-memory-utilization 0.40 --enable-lora --max-lora-rank 8 --max-loras 4 --max-cpu-loras 24 --lora-modules "${MODS[@]}"

$PY scripts/check_qwen_answer_regressions.py --url http://127.0.0.1:8000/v1 > logs/qwen_battery.log 2>&1 & QB=$!
eval_lane() {
  for d in $ALL; do
    claim $d || continue
    $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $d --variants base $(cat logs/variants_$d) > logs/eval_$d.log 2>&1 || { touch FAILED; return 1; }
    mark "EVAL $d gotowe"
    $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $d --variants $(cat logs/variants_$d) > logs/judge_$d.log 2>&1 || { touch FAILED; return 1; }
    grep "^\[" logs/judge_$d.log | tail -2
  done
}
eval_lane & E1=$!
eval_lane & E2=$!
wait $E1 && wait $E2 || fail "ewaluacja"
wait $QB; mark "bateria Qwena: $(grep -c 'status: OK' logs/qwen_battery.log) OK, $(grep -c 'status: ZNANY' logs/qwen_battery.log) znane, $(grep -c 'status: NOWA' logs/qwen_battery.log) nowe regresje"
unserve 8001; unserve 8000
touch ALL_DONE
mark "PIPELINE_DONE"
