"""Krok 116: lokalny test prędkości modelu w LM Studio na gotowych kontekstach kadr (eval_models.json).
Zasady po Krokach 34/56 (warianty 4-bit/8-bit mają to samo id): `lms unload --all`, ładowanie po KLUCZU wariantu z unikalnym
--identifier i jawnym --context-length, weryfikacja przez `lms ps` (nazwa + rozmiar), zapytania tylko na ten identyfikator.
Jeden ciężki model naraz. Wynik: data/distill/kadry_i_place/speed_<identifier>.json (odpowiedzi + czasy).
Użycie: python scripts/eval_gemma_lmstudio_speed.py --key gemma-4-26b-a4b-it-mlx@4bit --identifier gemma26-4bit --n 10"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from distill_qwen import cited_outside_context, post_chat  # noqa: E402

URL = "http://127.0.0.1:1234/v1"


def sh(*cmd: str, timeout: int = 900) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (out.stdout + out.stderr).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", required=True, help="klucz wariantu z `lms ls` (np. gemma-4-31b-it-mlx@8bit)")
    p.add_argument("--identifier", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--context-length", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--keep-loaded", action="store_true")
    args = p.parse_args()

    d = ROOT / "data/distill/kadry_i_place"
    ctxs = json.loads((d / "eval_models.json").read_text(encoding="utf-8"))["contexts"]
    qs = list(ctxs)
    step = max(1, len(qs) // args.n)
    sample = qs[::step][: args.n]

    print(sh("lms", "unload", "--all"), flush=True)
    t0 = time.time()
    print(sh("lms", "load", args.key, "--identifier", args.identifier, "--context-length", str(args.context_length), "-y"), flush=True)
    load_s = time.time() - t0
    ps = sh("lms", "ps")
    print(ps, flush=True)
    if args.identifier not in ps:
        sys.exit(f"BLAD: identyfikator {args.identifier} nie jest załadowany -- przerywam")

    rows = []
    for i, q in enumerate(sample, 1):
        c = ctxs[q]
        payload = {"model": args.identifier, "messages": c["messages"], "max_tokens": args.max_tokens, "temperature": 0.0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        r = post_chat(URL, payload, timeout=1800)
        dt = time.time() - t
        toks = (r.get("usage") or {}).get("completion_tokens")
        text = r["content"].strip()
        rows.append({"question": q, "seconds": round(dt, 1), "completion_tokens": toks, "tok_per_s": round(toks / dt, 1) if toks else None,
                     "reasoning_chars": len(r.get("reasoning") or ""), "finish_reason": r.get("finish_reason"), "text": text,
                     "outside_context": cited_outside_context(text, c["context_articles"], c["messages"][-1]["content"])})
        print(f"[{args.identifier}] {i}/{len(sample)} {dt:.0f} s, {toks} tok, {rows[-1]['tok_per_s']} tok/s, myślenie {rows[-1]['reasoning_chars']} zn.", flush=True)
    secs = [r["seconds"] for r in rows]
    summary = {"identifier": args.identifier, "key": args.key, "context_length": args.context_length, "load_seconds": round(load_s, 1),
               "lms_ps": ps, "n": len(rows), "mean_seconds": round(sum(secs) / len(secs), 1), "max_seconds": max(secs),
               "mean_tok_per_s": round(sum(r["tok_per_s"] or 0 for r in rows) / len(rows), 1)}
    (d / f"speed_{args.identifier}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PODSUMOWANIE", json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.keep_loaded:
        sh("lms", "unload", "--all")


if __name__ == "__main__":
    main()
