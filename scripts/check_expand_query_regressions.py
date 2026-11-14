"""
Zestaw testów regresyjnych dla `expand_query()` / `QUERY_EXPANSION_SYSTEM_PROMPT`
(patrz `prompt.py`) na znanych przypadkach referencyjnych ze wszystkich
dziedzin -- data/expand_query_reference_cases.json.

Powstał po Kroku 26 (PROGRESS.md): dwie kolejne próby naprawy Kroku 25
("expand_query() odpowiada zamiast przeformułować przy pytaniach o
wysokość kary") pokazały, że RĘCZNE sprawdzanie kilku wybranych
przypadków po każdej zmianie tego współdzielonego promptu NIE WYSTARCZA
-- VAT art. 86 (flagowy przypadek Kroku 14) cicho zregresował z rangi #6
na #27 przez dwie kolejne, niepowiązane zmiany (Krok 21, Krok 22), i
żadna z nich nie miała go na liście do ponownej weryfikacji. Ten skrypt
uruchamia WSZYSTKIE znane przypadki automatycznie, za każdym razem.

Dla każdego przypadku: woła prawdziwy `chat.expand_query()` (zawsze na
modelu bazowym, bez LoRA -- dokładnie jak w produkcyjnym chat.py, patrz
Krok 21), potem `RagIndex.search()` z oryginalnym pytaniem I tym
rozszerzeniem jako `extra_queries` (domyślny reranking + fuzja rang,
dokładnie jak w produkcji) i sprawdza rangę oczekiwanego artykułu.

Przypadki oznaczone w danych polem `known_issue` (niepuste) to ZNANE,
udokumentowane w PROGRESS.md problemy -- oczekuje się, że będą się
czerwienić, dopóki ktoś ich nie naprawi; skrypt i tak kończy się kodem 1,
jeśli którykolwiek przypadek nie przechodzi (żeby regresja NOWEGO
przypadku nie zginęła w tle znanych problemów), ale w raporcie jasno
odróżnia "ZNANY problem" od "NOWA regresja".

Wymaga lokalnie: modelu MLX (`models/Bielik-11B-v3.0-Instruct-mlx`) oraz
zbudowanych indeksów RAG dla wszystkich dziedzin (`--domain <x>` w
`build_rag_index.py`) -- nie jest częścią szybkiego `pytest tests/`
(tamten nie ładuje modelu w ogóle), tylko osobnym, ręcznie odpalanym
narzędziem, analogicznie do `check_acts_freshness.py`.

Użycie:
    python scripts/check_expand_query_regressions.py
    python scripts/check_expand_query_regressions.py --case vat_86_zakupy_prywatne
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlx_lm import load  # noqa: E402

import chat  # noqa: E402
from rag_search import RERANK_CANDIDATE_POOL, RagIndex, release_accelerator_cache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "expand_query_reference_cases.json"  # --cases nadpisuje
# Fuzja liczona na produkcyjnej puli (RERANK_CANDIDATE_POOL na wariant zapytania,
# jak przy top_k=8 w chat.py) -- wcześniej pula 300 dawała inne rangi niż
# produkcja (Krok 93). Zwracamy całą scaloną listę (do 2x pula), żeby zmierzyć
# rangę także poza top-8; dalej -- "poza pulą".
CANDIDATE_POOL = 2 * RERANK_CANDIDATE_POOL


def load_cases(case_id: str | None = None, domain: str | None = None, path: Path | None = None) -> list[dict]:
    cases = json.loads((path or CASES_PATH).read_text(encoding="utf-8"))
    if domain:
        cases = [c for c in cases if c["domain"] == domain]
        if not cases:
            raise SystemExit(f"Brak przypadków dla domain={domain!r} w {CASES_PATH}")
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise SystemExit(f"Nie znaleziono przypadku o id={case_id!r} w {CASES_PATH}")
    return cases


def find_rank(results: list[dict], act_short: str, article: str) -> int | None:
    for i, r in enumerate(results):
        if r["act_short"] == act_short and r["article"] == article:
            return i + 1
    return None


def run_case(
    model, tokenizer, idx_cache: dict, case: dict, routing: dict | None = None, primary_only: bool = False
) -> tuple[int | None, str]:
    """`routing` (z router.py --battery): dziedziny z routera zamiast `case["domain"]`;
    `primary_only` ignoruje `secondary` (wariant minimalny, Krok 99)."""
    domain = case["domain"]
    extra: tuple[str, ...] = ()
    if routing is not None:
        decision = routing[case["id"]]
        domain = decision["primary"]
        if decision["secondary"] and not primary_only:
            extra = (decision["secondary"],)
    cache_key = (domain, *extra)
    if cache_key not in idx_cache:
        # Współdzielimy model embeddingowy i reranker MIĘDZY dziedzinami --
        # bez tego każda nowa dziedzina dokłada własną, nigdy nieuwalnianą
        # kopię obu modeli do pamięci procesu (patrz PROGRESS.md, Krok 56,
        # ten sam problem znaleziony w check_qwen_answer_regressions.py --
        # tam przy ~18 dziedzinach urosło to do ~30 GB).
        shared = next(iter(idx_cache.values()), None)
        idx_cache[cache_key] = RagIndex(
            domain=domain,
            extra_domains=list(extra),
            rerank=True,
            fuse=True,
            model=shared.model if shared else None,
            cross_encoder=shared.cross_encoder if shared else None,
        )
    idx = idx_cache[cache_key]

    expansion = chat.expand_query(
        model, tokenizer, case["question"], lora_layers=None, act_titles=idx.act_titles(), domain=domain
    )
    results = idx.search(case["question"], top_k=CANDIDATE_POOL, extra_queries=[expansion], candidate_pool=RERANK_CANDIDATE_POOL)
    rank = find_rank(results, case["act_short"], case["article"])
    return rank, expansion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=str, default=None, help="Ogranicz do jednego przypadku (pole 'id' w danych)")
    parser.add_argument("--domain", type=str, default=None, help="Ogranicz do przypadków jednej dziedziny")
    parser.add_argument(
        "--model", type=str, default=str(chat.MODEL_PATH), help="Ścieżka do modelu bazowego w formacie MLX"
    )
    parser.add_argument("--cases", type=Path, default=None,
                        help="Plik z przypadkami zamiast bazy 129 (np. data/routing_holdout_cases.json)")
    parser.add_argument("--routing-file", type=Path, default=None,
                        help="Wynik router.py --battery: szukaj w dziedzinach wskazanych przez router")
    parser.add_argument("--primary-only", action="store_true", help="Z --routing-file: pomiń `secondary`")
    args = parser.parse_args()
    routing = json.loads(args.routing_file.read_text(encoding="utf-8")) if args.routing_file else None

    cases = load_cases(args.case, args.domain, args.cases)

    print(f"Ładowanie modelu bazowego ({args.model})...")
    model, tokenizer = load(args.model)
    idx_cache: dict[tuple, RagIndex] = {}

    any_new_regression = False
    any_known_issue = False

    for case in cases:
        rank, expansion = run_case(model, tokenizer, idx_cache, case, routing, args.primary_only)
        release_accelerator_cache()
        max_rank = case["max_acceptable_rank"]
        passed = rank is not None and rank <= max_rank
        known_issue = case.get("known_issue")

        print(f"[{case['id']}] {case['act_short']} art.{case['article']}")
        print(f"  pytanie: {case['question']}")
        print(f"  rozszerzenie: {expansion[:150]!r}")
        print(f"  ranga: {rank if rank is not None else 'poza pulą ' + str(CANDIDATE_POOL)} (próg: top-{max_rank})")

        if passed:
            status = "OK"
        elif known_issue:
            status = "ZNANY PROBLEM (nie nowy)"
            any_known_issue = True
        else:
            status = "NOWA REGRESJA"
            any_new_regression = True
        print(f"  status: {status}")
        if known_issue:
            print(f"  known_issue: {known_issue}")
        print()

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
    print(f"Wszystkie {len(cases)} przypadków referencyjnych przechodzi.")


if __name__ == "__main__":
    main()
