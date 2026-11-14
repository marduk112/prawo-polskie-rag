#!/bin/bash
# Krok 122: adaptery produkcyjne (Bielik v3.0 + LoRA) vs Bielik v3.1 bazowy vs Gemma-4-26B-A4B vs Qwen3.8-27B, 3 dziedziny (budowlane, cywilne, mieszkaniowe),
# te same konteksty (z Kroku 112), odpowiedzi wszystkich modeli z limitem 1000 tok. (Gemma 3000, bez myślenia), sędzia Qwen w obu kolejnościach A/B.
# Wymaga bundle.tgz (make_bundle_k122.sh) i /workspace/.hf_token (usuwany po pobraniu); RUNPOD_API_KEY/RUNPOD_POD_ID z /etc/rp_environment.
cd /workspace/work
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/work/stub
PY=python3
DOMAINS="${DOMAINS:-budowlane cywilne mieszkaniowe}"
QWEN=qwen3.8-27b-fp8; B30=speakleash/Bielik-11B-v3.0-Instruct; B31=speakleash/Bielik-11B-v3.1-Instruct; GEMMA=google/gemma-4-26B-A4B-it
MAX_MIN=${MAX_MIN:-110}; GRACE_MIN=${GRACE_MIN:-15}
mkdir -p logs
mark() { echo "### $(date -u +%H:%M:%S) $*"; }
fail() { mark "BLAD $*"; touch FAILED; exit 1; }
(
  start=$(date +%s)
  log() { echo "$(date -u +%H:%M:%S) $*" >> logs/watchdog.log; }
  stop_pod() { log "STOP: $1"; curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> logs/watchdog.log 2>&1; sleep 600; }
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
# --- setup: paczka, stub sentence_transformers (sędzia i eval_distill_vllm nie liczą embeddingów, Krok 112), modele ---
tar xzf bundle.tgz --no-same-owner 2>/dev/null
mkdir -p stub/sentence_transformers stub/pypdf data/distill
cat > stub/sentence_transformers/__init__.py <<'PYX'
class SentenceTransformer:  # stub: kontekstów nie przeliczamy
    def __init__(self, *a, **k): raise RuntimeError("stub")
class CrossEncoder(SentenceTransformer): pass
PYX
printf "class PdfReader:\n    def __init__(self, *a, **k): raise RuntimeError(\"stub\")\n" > stub/pypdf/__init__.py  # file_index importuje pypdf
export HF_TOKEN=$(cat /workspace/.hf_token)
for m in Qwen/Qwen3.8-27B-FP8 $B30 $B31 $GEMMA; do hf download $m > logs/dl_$(basename $m).log 2>&1 & done
wait
rm -f /workspace/.hf_token; unset HF_TOKEN
export HF_HUB_OFFLINE=1
for m in Qwen3.8-27B-FP8 Bielik-11B-v3.0-Instruct Bielik-11B-v3.1-Instruct gemma-4-26B-A4B-it; do grep -qiE "error|denied|401|403" logs/dl_$m.log && fail "pobieranie $m: $(tail -c 200 logs/dl_$m.log)"; done
mark "modele pobrane"
serve() {  # serve <repo> <served-name> [extra]
  local repo=$1 name=$2; shift 2
  nohup vllm serve $repo --host 127.0.0.1 --port 8000 --served-model-name $name --max-model-len 20480 --max-num-seqs 48 \
    --limit-mm-per-prompt '{"image":0,"video":0}' --gpu-memory-utilization 0.88 "$@" > logs/vllm_$name.log 2>&1 &
  echo $! > logs/vllm.pid
  until curl -sf localhost:8000/v1/models >/dev/null; do
    sleep 5
    kill -0 "$(cat logs/vllm.pid)" 2>/dev/null || { tail -30 logs/vllm_$name.log; fail "VLLM_DIED $name"; }
  done
  mark "vllm gotowy: $name"
}
unserve() { kill "$(cat logs/vllm.pid)" 2>/dev/null; sleep 15; pkill -9 -f "vllm serve" 2>/dev/null; sleep 5; mark "vllm zatrzymany"; }
# plik wynikowy: konteksty i odpowiedzi Qwena z Kroku 112 (adapterowe odpowiedzi 500 tok. NIE są używane -- generujemy od nowa z 1000)
for d in $DOMAINS; do
  $PY - <<PYX
import json
src = json.load(open("data/distill/$d/eval_vs_adapter.json", encoding="utf-8"))
json.dump({"contexts": src["contexts"], "answers": ({"$QWEN": src["answers"]["$QWEN"]} if "$QWEN" in src["answers"] else {})},
          open("data/distill/$d/eval_k122.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(list(src["contexts"]), open("data/distill/$d/k122_questions.json", "w", encoding="utf-8"), ensure_ascii=False)
PYX
done
gen() {  # gen <domena> <wariant> <max-tokens> [template-kwargs] [reasoning-effort]
  local d=$1 v=$2 mt=$3 tk=$4 re=$5
  $PY scripts/eval_distill_vllm.py --url http://127.0.0.1:8000/v1 --domain $d --variants $v --questions-file data/distill/$d/k122_questions.json \
    --out-file eval_k122.json --max-tokens $mt ${tk:+--template-kwargs "$tk"} ${re:+--reasoning-effort "$re"} --concurrency 32 > logs/gen_${d}_$v.log 2>&1 || fail "gen $d $v"
  mark "$d $v: $(tail -1 logs/gen_${d}_$v.log)"
}
# --- Faza 1: Bielik v3.0 BF16 + 3 adaptery produkcyjne ---
mods=""; for d in $DOMAINS; do mods="$mods prod_$d=adapters-vllm/prod_$d"; done
serve $B30 bielik-v30 --enable-lora --max-lora-rank 8 --max-loras 3 --lora-modules $mods
for d in $DOMAINS; do gen $d prod_$d 1000; done
unserve
# --- Faza 2: Bielik v3.1 BF16 bazowy ---
serve $B31 v31
for d in $DOMAINS; do gen $d v31 1000; done
unserve
# --- Faza 3: Gemma-4-26B-A4B (bez myślenia, 3000 tok. jak w Kroku 115) ---
serve $GEMMA gemma26
for d in $DOMAINS; do gen $d gemma26 3000 '{"enable_thinking": false}'; done
unserve
# --- Faza 4: sędzia Qwen, obie kolejności A/B ---
serve Qwen/Qwen3.8-27B-FP8 $QWEN --max-model-len 24576 --max-num-seqs 64 --reasoning-parser qwen3 --gpu-memory-utilization 0.90
# odpowiedzi Qwena, gdy ich brak w pliku (np. konsumenckie): jak Krok 109 (3000 tok., reasoning-effort low)
for d in $DOMAINS; do
  $PY -c "import json,sys; sys.exit(0 if '$QWEN' in json.load(open('data/distill/$d/eval_k122.json',encoding='utf-8'))['answers'] else 1)" || gen $d $QWEN 3000 "" low
  cp data/distill/$d/eval_k122.json RESULT_eval_k122_$d.json
done
# --- pary dla sędziego ---
for d in $DOMAINS; do
  $PY - <<PYX
import json
ev = json.load(open("data/distill/$d/eval_k122.json", encoding="utf-8")); a = ev["answers"]
a["prod"] = a["prod_$d"]; a["qwen"] = a["$QWEN"]
for base, var in (("prod", "v31"), ("prod", "gemma26"), ("v31", "gemma26"), ("v31", "qwen"), ("gemma26", "qwen")):
    json.dump({"contexts": ev["contexts"], "answers": {"base": a[base], var: a[var]}},
              open(f"data/distill/$d/pair_k122_{base}_{var}.json", "w", encoding="utf-8"), ensure_ascii=False)
PYX
done
for d in $DOMAINS; do
  for pr in prod:v31 prod:gemma26 v31:gemma26 v31:qwen gemma26:qwen; do
    base=${pr%%:*}; var=${pr##*:}; vname=$var; [ $var = qwen ] && vname=qwen
    $PY scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain $d --variants $vname --eval-file pair_k122_${base}_${var}.json \
      --out-file judge_k122_${base}_${var}.json --both-orders --max-tokens 9000 --concurrency 48 > logs/judge_${d}_${base}_${var}.log 2>&1 || { mark "BLAD sędzia $d $base $var"; continue; }
    mark "$d $base vs $var: $(grep '^\[' logs/judge_${d}_${base}_${var}.log | tail -1)"
    cp data/distill/$d/judge_k122_${base}_${var}.json RESULT_judge_k122_${d}_${base}_${var}.json
  done
done
tar czf RESULTS_k122.tgz RESULT_*.json logs/*.log 2>/dev/null
touch ALL_DONE
mark "PIPELINE_DONE"
