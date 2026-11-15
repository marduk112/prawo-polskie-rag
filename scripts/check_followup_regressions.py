"""
Bateria dla pytań uzupełniających w rozmowie ("A po 15 latach?") -- tylko wyszukiwanie (bez odpowiedzi modelu) --
PROGRESS.md, Krok 140. Porównuje dwa warianty szukania z historią jednej poprzedniej tury:
  stary  -- prompt.search_with_history z fallbackiem opartym na progu kosinusa 0,75
  nowy   -- model w LM Studio przepisuje pytanie na samodzielne (chat_lmstudio.rewrite_followup), szukamy przepisanym
Zbiory: data/followup_cases.json (pytania uzupełniające ze znanym artykułem, sprawdzone po treści) oraz próbka pytań
SAMODZIELNYCH z expand_query_reference_cases.json po historii z INNEJ dziedziny (kontrola szkód: ranga nie powinna się
pogorszyć względem zwykłego wyszukiwania). Rozszerzanie zapytań jest wyłączone w obu wariantach (izolacja efektu).

    python scripts/check_followup_regressions.py [--model ID] [--reasoning-effort low] [--negatives-step 12]
"""

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import chat_lmstudio  # noqa: E402
import prompt  # noqa: E402
from rag_search import EMBEDDING_MODEL, RERANKER_MODEL, RagIndex  # noqa: E402

LIST_LEN = 30


def hist(question: str) -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": "(odpowiedź)"}]


def rank_of(results: list[dict], act: str, article: str) -> int | None:
    return next((i + 1 for i, r in enumerate(results) if (r["act_short"], r["article"]) == (act, article)), None)


def summarize(ranks: list[int | None]) -> str:
    return (
        f"top-1 {sum(r == 1 for r in ranks)}, top-3 {sum(r is not None and r <= 3 for r in ranks)}, "
        f"top-8 {sum(r is not None and r <= 8 for r in ranks)} z {len(ranks)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=chat_lmstudio.DEFAULT_URL)
    parser.add_argument("--model", default=None, help="ID modelu w LM Studio (domyślnie: auto-wykrycie)")
    parser.add_argument("--reasoning-effort", default=None, help="np. low dla Qwena (jak w chat_lmstudio.py)")
    parser.add_argument("--negatives-step", type=int, default=12, help="co ile-ty przypadek z expand_query_reference_cases.json")
    args = parser.parse_args()

    model_id = args.model or chat_lmstudio.detect_model(args.url)
    emb, cross = SentenceTransformer(EMBEDDING_MODEL), CrossEncoder(RERANKER_MODEL)
    indexes: dict[str, RagIndex] = {}

    def index(domain: str) -> RagIndex:
        if domain not in indexes:
            indexes[domain] = RagIndex(domain=domain, model=emb, cross_encoder=cross)
        return indexes[domain]

    def new_results(domain: str, history: list[dict], question: str) -> list[dict]:
        raw = chat_lmstudio.rewrite_followup(args.url, model_id, history, question, args.reasoning_effort)
        rewritten = prompt.clean_rewritten_question(raw, question)
        print(f"      przepisane: {rewritten!r}")
        return index(domain).search(rewritten, top_k=LIST_LEN, candidate_pool=30)

    def old_results(domain: str, history: list[dict], question: str) -> list[dict]:
        return prompt.search_with_history(index(domain), history, question, LIST_LEN)

    follow = json.loads((ROOT / "data" / "followup_cases.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "data" / "expand_query_reference_cases.json").read_text(encoding="utf-8"))
    negatives = cases[:: args.negatives_step]

    print("== pytania uzupełniające")
    old_f, new_f = [], []
    for c in follow:
        o = rank_of(old_results(c["domain"], hist(c["prior"]), c["question"]), c["act_short"], c["article"])
        print(f"  {c['id']}: {c['question']!r}")
        n = rank_of(new_results(c["domain"], hist(c["prior"]), c["question"]), c["act_short"], c["article"])
        old_f.append(o)
        new_f.append(n)
        print(f"      stary {o} -> nowy {n}")
    print("\n== pytania samodzielne po historii z innej dziedziny (kontrola szkód)")
    worse = better = 0
    for i, c in enumerate(negatives):
        other = negatives[(i + 1) % len(negatives)]
        base = rank_of(index(c["domain"]).search(c["question"], top_k=LIST_LEN, candidate_pool=30), c["act_short"], c["article"])
        old = rank_of(old_results(c["domain"], hist(other["question"]), c["question"]), c["act_short"], c["article"])
        new = rank_of(new_results(c["domain"], hist(other["question"]), c["question"]), c["act_short"], c["article"])
        def key(r): return LIST_LEN + 1 if r is None else r
        worse += key(new) > key(base)
        better += key(new) < key(base)
        print(f"  {c['id']}: bez historii {base}, stary {old}, nowy {new}")
    print(f"\nPytania uzupełniające -- stary: {summarize(old_f)}; nowy: {summarize(new_f)}")
    print(f"Pytania samodzielne ({len(negatives)}): nowy gorszy niż bez historii: {worse}, lepszy: {better}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
