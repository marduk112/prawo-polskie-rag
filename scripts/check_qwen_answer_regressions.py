"""
Zestaw testów regresyjnych dla JAKOŚCI ODPOWIEDZI (nie tylko rankingu RAG)
przez model w LM Studio (np. Qwen3.8-27B) -- data/qwen_answer_reference_cases.json.

Powstał jako standaryzacja ad-hoc sond z Kroków 30/32/34/35/36
(PROGRESS.md) -- do tej pory te testy żyły tylko w jednorazowych scratch-
skryptach i ręcznym czytaniu wydruków, więc nie było jak automatycznie
wykryć regresji przy zmianie promptu/parametrów w kolejnej sesji. Ten
skrypt uruchamia wszystkie znane sondy automatycznie przez prawdziwy
`chat_lmstudio.answer()` (dokładnie tę samą ścieżkę co produkcyjny
`scripts/chat_lmstudio.py`) i sprawdza proste, strukturalne kryteria
(obecność/brak oczekiwanych fragmentów tekstu w odpowiedzi).

WAŻNE ograniczenie, celowe: to NIE jest pełna weryfikacja merytorycznej
poprawności -- substring-checki (`expect_contains_any`/`expect_not_contains`)
łapią tylko grube regresje (zniknięcie oczekiwanego numeru artykułu,
pojawienie się znanej błędnej liczby/artykułu). Skrypt zawsze drukuje
PEŁNĄ odpowiedź dla każdego przypadku -- ostateczna ocena jakości
(spójność rozumowania, poprawne zastosowanie Zasady 6 itd., patrz Krok 15/16/30)
wciąż wymaga ręcznego przeczytania, tak jak dotychczas. Traktuj to jako
automatyczny "kanarek" na duże regresje, nie zamiennik ręcznej oceny.

Wymaga: uruchomionego serwera LM Studio (patrz chat_lmstudio.py) z
załadowanym modelem (Krok 32-36 używały qwen/qwen3.8-27b w wariancie
8-bit, `reasoning_effort=low`) oraz zbudowanych indeksów RAG dla
domen użytych w przypadkach. Każdy przypadek z `reasoning_effort=low`
trwa ~150-250s (Krok 32) -- pełny przebieg to kilkanaście minut, nie
część `pytest tests/`.

Użycie:
    python scripts/check_qwen_answer_regressions.py
    python scripts/check_qwen_answer_regressions.py --case karne_148_kara_zabojstwo_odpowiedz
    python scripts/check_qwen_answer_regressions.py --model qwen/qwen3.8-27b
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chat_lmstudio  # noqa: E402
from file_index import SessionFileIndex  # noqa: E402
from rag_search import RagIndex, release_accelerator_cache  # noqa: E402
from router import effective_domains  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "qwen_answer_reference_cases.json"


def load_cases(case_id: str | None = None, domain: str | None = None) -> list[dict]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if domain:
        cases = [c for c in cases if c["domain"] == domain]
        if not cases:
            raise SystemExit(f"Brak przypadków dla dziedziny {domain!r} w {CASES_PATH}")
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise SystemExit(f"Nie znaleziono przypadku o id={case_id!r} w {CASES_PATH}")
    return cases


def check_text(text: str, contains_any: list[str], not_contains: list[str]) -> tuple[bool, str]:
    lowered = text.lower()
    if contains_any and not any(s.lower() in lowered for s in contains_any):
        return False, f"brak żadnego z oczekiwanych fragmentów: {contains_any}"
    hit = [s for s in not_contains if s.lower() in lowered]
    if hit:
        return False, f"znaleziono zakazany fragment: {hit}"
    return True, "OK"


def run_case(
    base_url: str, model_id: str, idx_cache: dict, case: dict, domains: list[str] | None = None,
    template_kwargs: dict | None = None,
) -> tuple[bool, str, str]:
    """`domains` (z routera, patrz router.effective_domains): [primary] albo [primary, secondary];
    domyślnie dziedzina z przypadku."""
    domains = domains or [case["domain"]]
    domain = domains[0]
    cache_key = tuple(domains)
    if cache_key not in idx_cache:
        # Współdzielimy model embeddingowy i reranker MIĘDZY dziedzinami --
        # to te same modele niezależnie od domeny, tylko embeddingi/meta się
        # różnią. Bez tego każda nowa dziedzina dokładała własną, nigdy
        # nieuwalnianą kopię obu modeli do pamięci procesu -- przy ~18
        # dziedzinach w jednym przebiegu urosło to do ~30 GB i omal nie
        # zawiesiło maszyny (patrz PROGRESS.md, Krok 56).
        shared = next(iter(idx_cache.values()), None)
        idx_cache[cache_key] = RagIndex(
            domain=domain,
            extra_domains=domains[1:],
            model=shared.model if shared else None,
            cross_encoder=shared.cross_encoder if shared else None,
        )
    rag = idx_cache[cache_key]
    file_index = SessionFileIndex(model=rag.model)

    history: list[dict] = []
    for turn in case.get("history", []):
        prior_text = chat_lmstudio.answer(
            base_url, model_id, rag, file_index, history, turn["question"], top_k=8,
            reasoning_effort=case.get("reasoning_effort"), max_tokens=case.get("max_tokens", 500),
            template_kwargs=template_kwargs,
        )
        ok, reason = check_text(prior_text, turn.get("expect_contains_any", []), turn.get("expect_not_contains", []))
        if not ok:
            return False, f"tura wstępna ({turn['question']!r}) nie przeszła: {reason}", prior_text

    text = chat_lmstudio.answer(
        base_url, model_id, rag, file_index, history, case["question"], top_k=8,
        as_of=case.get("as_of"), reasoning_effort=case.get("reasoning_effort"),
        max_tokens=case.get("max_tokens", 500), template_kwargs=template_kwargs,
    )
    ok, reason = check_text(text, case.get("expect_contains_any", []), case.get("expect_not_contains", []))
    return ok, reason, text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=str, default=None, help="Ogranicz do jednego przypadku (pole 'id' w danych)")
    parser.add_argument("--domain", type=str, default=None, help="Ogranicz do przypadków jednej dziedziny")
    parser.add_argument("--url", type=str, default=chat_lmstudio.DEFAULT_URL, help="Bazowy URL serwera LM Studio")
    parser.add_argument("--model", type=str, default=None, help="ID modelu w LM Studio (domyślnie: auto-wykrycie)")
    parser.add_argument("--routing-file", type=Path, default=None,
                        help="Wynik router.py --battery --single-threshold 1.01: dziedziny z routera zamiast ręcznych")
    parser.add_argument("--secondary-below", type=float, default=0.85,
                        help="Z --routing-file: druga dziedzina gdy pewność < tej wartości (> 1 = zawsze)")
    parser.add_argument("--only-changed", action="store_true",
                        help="Z --routing-file: tylko przypadki, w których dziedziny z routera != [ręczna dziedzina]")
    parser.add_argument("--template-kwargs", default=None,
                        help='JSON z chat_template_kwargs (Krok 117), np. \'{"enable_thinking": false}\' dla Gemma 4')
    args = parser.parse_args()
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else None
    routing = json.loads(args.routing_file.read_text(encoding="utf-8")) if args.routing_file else None

    cases = load_cases(args.case, args.domain)
    if routing is not None and args.only_changed:
        cases = [c for c in cases if effective_domains(routing[c["id"]], args.secondary_below) != [c["domain"]]]
        print(f"[INFO] Przypadki ze zmienionymi dziedzinami: {len(cases)}")
    model_id = args.model or chat_lmstudio.detect_model(args.url)
    print(f"[INFO] LM Studio: {args.url}, model={model_id}\n")

    idx_cache: dict[tuple, RagIndex] = {}
    any_new_regression = False
    any_known_issue = False

    for case in cases:
        domains = effective_domains(routing[case["id"]], args.secondary_below) if routing else None
        print(f"[{case['id']}] domena={case['domain']}" + (f" -> router: {domains}" if domains else ""))
        print(f"  pytanie: {case['question']}")
        passed, reason, text = run_case(args.url, model_id, idx_cache, case, domains, template_kwargs)
        release_accelerator_cache()
        known_issue = case.get("known_issue")

        if passed:
            status = "OK"
        elif known_issue:
            status = "ZNANY PROBLEM (nie nowy)"
            any_known_issue = True
        else:
            status = "NOWA REGRESJA"
            any_new_regression = True

        print(f"  status: {status} ({reason})")
        if known_issue:
            print(f"  known_issue: {known_issue}")
        print(f"  odpowiedź:\n{text}\n")
        print("-" * 70)

    print("=" * 70)
    if any_new_regression:
        print("ZNALEZIONO NOWĄ REGRESJĘ -- patrz 'NOWA REGRESJA' wyżej.")
        sys.exit(1)
    if any_known_issue:
        print(
            "Wszystkie niepowodzenia to ZNANE, udokumentowane problemy (patrz "
            "PROGRESS.md) -- brak nowej regresji, ale są otwarte zadania do naprawy."
        )
        sys.exit(1)
    print(f"Wszystkie {len(cases)} przypadków referencyjnych przechodzi automatyczny check.")
    print("Pamiętaj: to tylko strukturalny check -- przeczytaj pełne odpowiedzi powyżej dla oceny merytorycznej.")


if __name__ == "__main__":
    main()
