"""Qwen jako sędzia ewaluacji destylacji (PROGRESS.md, Krok 77): porównanie
parami "sam Bielik vs adapter" na odpowiedziach z eval_vllm.json
(scripts/eval_distill_vllm.py) -- ten sam kontekst RAG, kolejność A/B losowa
(stałe ziarno), sędzia widzi kontekst i ewentualną notatkę 'expect'
(data/adapter_vs_base_questions.json). Wynik: data/distill/<d>/judge.json
(wznawialne) + podsumowanie wygranych/remisów/przegranych. Sędzia to nie
wyrocznia -- rozbieżności i błędy "poważne" trzeba przeczytać ręcznie.

Użycie:
    python scripts/judge_distill.py --url http://127.0.0.1:8000/v1 --domain budowlane \\
        --variants budowlane-iter450 budowlane-iter584
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from distill_qwen import Teacher, domain_dir  # noqa: E402

JUDGE_SYSTEM = (
    "Jesteś doświadczonym polskim prawnikiem i oceniasz odpowiedzi asystenta prawnego. Asystent "
    "dostał fragmenty ustaw (kontekst) i pytanie zwykłej osoby. Oceniasz WYŁĄCZNIE merytorykę: "
    "czy wniosek jest poprawny w świetle przepisów, czy podstawa prawna (artykuł, ustęp) jest "
    "właściwa i pochodzi z kontekstu, czy odpowiedź nie zmyśla przepisów ani terminów, czy nie "
    "unika odpowiedzi, gdy kontekst ją rozstrzyga. Styl i długość są drugorzędne."
)
JUDGE_USER = """{context}

---
Notatka eksperta o oczekiwanej odpowiedzi (jeśli jest): {expect}

ODPOWIEDŹ A:
{a}

ODPOWIEDŹ B:
{b}

Zwróć WYŁĄCZNIE obiekt JSON:
{{"lepsza": "A" | "B" | "remis", "powazny_blad_A": "<krótko albo pusty napis>", "powazny_blad_B": "<krótko albo pusty napis>", "uzasadnienie": "<1-2 zdania>"}}
Poważny błąd = zły wniosek, zmyślony przepis lub termin, odmowa odpowiedzi mimo wystarczającego kontekstu."""


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return v if v.get("lepsza") in ("A", "B", "remis") else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variants", nargs="+", required=True, help="Warianty porównywane z 'base'")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=6000, help="Limit myślenia sędziego (Krok 88: 11/40 werdyktów ucięte przy 6000)")
    parser.add_argument("--context-window", type=int, default=24000, help="Okno modelu sędziego w tokenach (z zapasem)")
    parser.add_argument("--eval-file", default="eval_vllm.json", help="Plik z odpowiedziami w data/distill/<d>/ (np. eval.json z MLX)")
    parser.add_argument("--out-file", default="judge.json", help="Plik z werdyktami w data/distill/<d>/")
    parser.add_argument("--both-orders", action="store_true",
                        help="Każdą parę sądź w obu kolejnościach A/B; werdykt liczy się tylko gdy zgodny (inaczej remis, 'niespojne'), "
                             "poważny błąd tylko gdy zgłoszony w obu kolejnościach (Krok 121: bias pozycji i długości)")
    args = parser.parse_args()

    ev = json.loads((domain_dir(args.domain) / args.eval_file).read_text(encoding="utf-8"))
    compare = json.loads((ROOT / "data" / "adapter_vs_base_questions.json").read_text(encoding="utf-8"))
    expect = {i["q"]: i.get("expect", "brak") for i in compare.get(args.domain, [])}
    out = domain_dir(args.domain) / args.out_file
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    teacher = Teacher(args.url, None)
    rng = random.Random(0)

    jobs = []
    for variant in args.variants:
        done = data.setdefault(variant, {})
        for q, ctx in ev["contexts"].items():
            if q in done or q not in ev["answers"]["base"] or q not in ev["answers"].get(variant, {}):
                continue
            base, adapter = ev["answers"]["base"][q]["text"], ev["answers"][variant][q]["text"]
            for swap in ([False, True] if args.both_orders else [rng.random() < 0.5]):
                a, b = (adapter, base) if swap else (base, adapter)
                jobs.append((variant, q, swap, ctx["messages"][-1]["content"], expect.get(q, "brak"), a, b))

    def judge(job):
        variant, q, swap, context, exp, a, b = job
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(context=context, expect=exp, a=a, b=b)},
        ]
        # Krok 88: limit myślenia przycięty do wolnego miejsca w oknie modelu (24576 tokenów;
        # prompt = kontekst + dwie odpowiedzi, ~2,3 znaku/token zachowawczo), a błąd jednej pary
        # nie przerywa całości (sędzia zapisuje werdykty dopiero na końcu).
        est_prompt = int((len(JUDGE_SYSTEM) + len(context) + len(a) + len(b) + 700) / 2.3)
        limit = max(2000, min(args.max_tokens, args.context_window - est_prompt))
        v = None
        for _ in range(2):
            try:
                v = parse_verdict(teacher.chat(messages, max_tokens=limit, temperature=0.0, thinking=True)["content"])
            except Exception as exc:  # noqa: BLE001 -- jedna para nie może wywrócić całego przebiegu
                print(f"[{variant}] błąd zapytania ({type(exc).__name__}): {q[:50]}", flush=True)
                break
            if v:
                break
        return variant, q, swap, v

    pending: dict[tuple[str, str], list[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for variant, q, swap, v in pool.map(judge, jobs):
            if v is None:
                print(f"[{variant}] brak werdyktu: {q[:60]}", flush=True)
                continue
            # Z punktu widzenia adaptera: "adapter" / "base" / "remis".
            win = {"A": "base", "B": "adapter"} if not swap else {"A": "adapter", "B": "base"}
            pending.setdefault((variant, q), []).append({
                "lepsza": win.get(v["lepsza"], "remis"),
                "blad_adapter": v["powazny_blad_B" if not swap else "powazny_blad_A"],
                "blad_base": v["powazny_blad_A" if not swap else "powazny_blad_B"],
                "uzasadnienie": v.get("uzasadnienie", ""),
            })
    for (variant, q), rs in pending.items():
        if args.both_orders and len(rs) < 2:
            continue  # brak werdyktu w jednej z kolejności -- wznowienie dobierze
        consistent = len({r["lepsza"] for r in rs}) == 1
        data[variant][q] = {
            "lepsza": rs[0]["lepsza"] if consistent else "remis",
            "niespojne": not consistent,
            "blad_adapter": rs[0]["blad_adapter"] if all(r["blad_adapter"].strip() for r in rs) else "",
            "blad_base": rs[0]["blad_base"] if all(r["blad_base"].strip() for r in rs) else "",
            "uzasadnienie": " | ".join(r["uzasadnienie"] for r in rs),
        }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for variant in args.variants:
        res = data.get(variant, {}).values()
        count = {k: sum(r["lepsza"] == k for r in res) for k in ("adapter", "remis", "base")}
        err_a = sum(bool(r["blad_adapter"].strip()) for r in res)
        err_b = sum(bool(r["blad_base"].strip()) for r in res)
        print(f"[{args.domain} {variant}] adapter lepszy {count['adapter']}, remis {count['remis']}, "
              f"baza lepsza {count['base']} | poważne błędy: adapter {err_a}, baza {err_b}"
              + (f" | niespójne (liczone jako remis): {sum(bool(r.get('niespojne')) for r in res)}" if args.both_orders else ""), flush=True)


if __name__ == "__main__":
    main()
