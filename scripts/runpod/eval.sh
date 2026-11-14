#!/bin/bash
# Krok 76: ewaluacja na podzie (vLLM + LoRA) zaraz po treningu każdej dziedziny,
# równolegle z treningiem następnej; konwersja wybranych checkpointów do MLX.
cd /workspace/work
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/opt/train/bin/python
mark() { echo "### $(date -u +%H:%M:%S) EVAL $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }

pick() {  # $1 dziedzina -> nazwy checkpointów: najlepszy val loss, ostatni, ~60% najlepszego
  $PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
d = Path(f"adapters-cuda/bielik11b-{sys.argv[1]}-distill")
ckpts = sorted((int(p.name.split("-")[1]) for p in d.glob("checkpoint-*")))
hist = json.loads((d / f"checkpoint-{ckpts[-1]}" / "trainer_state.json").read_text())["log_history"]
ev = {h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h and h["step"] in ckpts}
best = min(ev, key=ev.get)
mid = min(ckpts, key=lambda s: abs(s - round(best * 0.6 / 50) * 50))
print(" ".join(str(s) for s in sorted({best, ckpts[-1], mid})))
print(" ".join(f"{s}:{ev.get(s, float('nan')):.4f}" for s in ckpts), file=sys.stderr)
EOF
}

run_domain() {
  local d=$1 steps mods=() variants=()
  until grep -q "TRENING_OK $d" pipeline.out; do [ -f FAILED ] && exit 1; sleep 30; done
  steps=$(pick $d 2>logs/eval_pick_$d.log) || fail "wybór checkpointów $d"
  mark "$d: checkpointy $steps (val loss: $(cat logs/eval_pick_$d.log))"
  for s in $steps; do
    src=adapters-cuda/bielik11b-$d-distill/checkpoint-$s
    v=adapters-vllm/$d-iter$s; mkdir -p $v
    cp $src/adapter_model.safetensors $v/
    $PY -c "import json,sys; c=json.load(open('$src/adapter_config.json')); c.pop('layers_to_transform'); c.pop('layers_pattern'); json.dump(c,open('$v/adapter_config.json','w'),indent=1)"
    mods+=("$d-iter$s=$v"); variants+=("$d-iter$s")
    $PY scripts/convert_lora.py peft-to-mlx $src adapters-mlx/bielik11b-$d-distill-lora-iter$s >> logs/convert.log 2>&1 || fail "konwersja $d $s"
  done
  nohup vllm serve speakleash/Bielik-11B-v3.0-Instruct --host 127.0.0.1 --port 8001 --served-model-name bielik-11b \
    --max-model-len 20480 --gpu-memory-utilization 0.38 --max-num-seqs 32 \
    --enable-lora --max-lora-rank 8 --max-loras 4 --lora-modules "${mods[@]}" > logs/vllm_eval_$d.log 2>&1 &
  local pid=$!
  until curl -sf localhost:8001/v1/models >/dev/null; do
    sleep 5; kill -0 $pid 2>/dev/null || { tail -20 logs/vllm_eval_$d.log; fail "vllm eval $d"; }
  done
  $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain $d --variants base "${variants[@]}" > logs/eval_$d.log 2>&1 \
    || { kill $pid; fail "eval $d"; }
  cat logs/eval_$d.log | grep odpowiedzi
  kill $pid; while kill -0 $pid 2>/dev/null; do sleep 2; done
  mark "$d gotowe"
}

mkdir -p adapters-vllm adapters-mlx
run_domain budowlane
run_domain cywilne
touch EVAL_DONE ALL_DONE
mark "EVAL_DONE"
