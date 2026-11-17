"""(Krok 142) Lokalna generacja odpowiedzi modelu z LM Studio na TYCH SAMYCH kontekstach RAG co Krok 106/115
(data/distill/kadry_i_place/eval_models.json, 76 pytań), z pomiarem czasu i tokenów.
Wynik: kopia eval_models.json + nowy wariant w answers[<model>] (tekst + metryki).
Uruchamiać z -I, jeden proces naraz (LM Studio ładuje jeden model).

    python -I scripts/eval_lmstudio_local_answers.py <model_id> <out.json> [--limit N] [--max-tokens 3000] [--template-kwargs JSON]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path("<repo>")
SRC = ROOT / "data/distill/kadry_i_place/eval_models.json"
QUESTIONS = ROOT / "data/distill/kadry_i_place/compare_questions.json"
URL = "http://localhost:1234/v1/chat/completions"

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("--limit", type=int, default=None)
p.add_argument("--max-tokens", type=int, default=3000)
p.add_argument("--template-kwargs", default=None)
a = p.parse_args()

data = json.loads(SRC.read_text(encoding="utf-8"))
questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
if a.limit:
    questions = questions[: a.limit]
out_path = Path(a.out)
if out_path.exists():
    data = json.loads(out_path.read_text(encoding="utf-8"))
answers = data["answers"].setdefault(a.model, {})
kwargs = json.loads(a.template_kwargs) if a.template_kwargs else None

for n, q in enumerate(questions, 1):
    if q in answers:
        continue
    payload = {
        "model": a.model,
        "messages": data["contexts"][q]["messages"],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
    }
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=1800)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    usage = j.get("usage", {})
    toks = usage.get("completion_tokens")
    answers[q] = {
        "text": text,
        "seconds": round(dt, 1),
        "completion_tokens": toks,
        "tok_per_s": round(toks / dt, 1) if toks and dt else None,
        "has_reasoning": bool(msg.get("reasoning_content")),
        "finish": j["choices"][0].get("finish_reason"),
        "outside_context": [],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{n}/{len(questions)}] {dt:5.1f}s {toks} tok {answers[q]['tok_per_s']} tok/s słów={len(text.split())} finish={answers[q]['finish']}", flush=True)

vals = [v for v in answers.values() if v.get("seconds")]
if vals:
    print("ŚREDNIO: %.1f s/odp, %.1f tok/s, %.0f słów, ucięte: %d/%d" % (
        sum(v["seconds"] for v in vals) / len(vals),
        sum(v["tok_per_s"] or 0 for v in vals) / len(vals),
        sum(len(v["text"].split()) for v in vals) / len(vals),
        sum(1 for v in vals if v["finish"] == "length"), len(vals)))
