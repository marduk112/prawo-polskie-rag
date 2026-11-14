"""Krok 104: prototyp oceny "agentic RAG" w wersji minimalnej -- pętla "jeśli wynik wygląda słabo,
poszerz wyszukiwanie o kolejne dziedziny".

Skrypt NIE implementuje pętli, tylko zbiera dane do jej oceny OFFLINE: dla każdego przypadku jedno
rozszerzenie zapytania (Bielik, dziedzina `primary` routera), a potem wyszukiwanie z tym samym
rozszerzeniem w kilku konfiguracjach dziedzin -- P (primary), S (primary + drugi z routera),
T3 (trzy najbardziej prawdopodobne), ALL (wszystkie) -- te z rozszerzeniem z dziedziny `primary` --
oraz N2 (sama druga dziedzina routera) i G (dziedzina przypadku, odniesienie), które dostają WŁASNE
rozszerzenie wygenerowane dla swojej dziedziny, tak jak zrobiłaby to prawdziwa pętla po przełączeniu.
Dla każdej zapisuje rangę właściwego artykułu i sygnały pewności wyniku (rerank_score, fused_score
pierwszych trafień). Polityki typu "gdy sygnał P < próg, użyj S/T3/ALL" liczy się potem bez
ponownego uruchamiania modeli. Różnica względem produkcji: w P/S/T3/ALL rozszerzenie jest
generowane raz (z act_titles dziedziny primary).

Wejście: JSON z listą {"set", "case", "routing"} (case jak w expand_query_reference_cases.json,
routing jak z router.py --battery, z "probabilities"). Wyjście: JSONL, po jednej linii na przypadek;
wznawia pracę (pomija id już zapisane).

    python scripts/eval_agentic_fallback.py JOBS.json WYNIK.jsonl
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlx_lm import load  # noqa: E402

import chat  # noqa: E402
from check_expand_query_regressions import CANDIDATE_POOL, find_rank  # noqa: E402
from rag_search import RERANK_CANDIDATE_POOL, RagIndex, release_accelerator_cache  # noqa: E402
from router import available_domains  # noqa: E402


def configs(case: dict, routing: dict, all_domains: list[str]) -> dict[str, list[str]]:
    probs = routing["probabilities"]
    ranked = sorted(probs, key=lambda d: (-probs[d], d))
    primary = routing["primary"]
    second = next(d for d in ranked if d != primary)
    return {
        "P": [primary],
        "S": [primary, second],
        "T3": ranked[:3],
        "ALL": [primary] + [d for d in all_domains if d != primary],
        "N2": [second],
        "G": [case["domain"]],
    }


OWN_EXPANSION = ("N2", "G")  # konfiguracje z rozszerzeniem wygenerowanym dla własnej dziedziny


def signals(results: list[dict]) -> dict:
    top = results[:3]
    return {
        "top1": f'{results[0]["act_short"]}:{results[0]["article"]}' if results else None,
        "top1_rerank": results[0].get("rerank_score") if results else None,
        "top1_fused": results[0].get("fused_score") if results else None,
        "max_rerank": max((r.get("rerank_score", 0.0) for r in results[:8]), default=None),
        "mean3_rerank": sum(r.get("rerank_score", 0.0) for r in top) / len(top) if top else None,
    }


def main():
    jobs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    all_domains = available_domains()

    print(f"Ładowanie modelu bazowego ({chat.MODEL_PATH}); do zrobienia {len(jobs) - len(done)} z {len(jobs)}")
    model, tokenizer = load(str(chat.MODEL_PATH))
    primaries: dict[str, RagIndex] = {}

    def index_for(domains: list[str]) -> RagIndex:
        shared = next(iter(primaries.values()), None)
        return RagIndex(
            domain=domains[0],
            extra_domains=domains[1:],
            rerank=True,
            fuse=True,
            model=shared.model if shared else None,
            cross_encoder=shared.cross_encoder if shared else None,
        )

    with out_path.open("a", encoding="utf-8") as out:
        for n, job in enumerate(jobs, 1):
            case, routing = job["case"], job["routing"]
            if case["id"] in done:
                continue
            started = time.time()
            primary = routing["primary"]
            if primary not in primaries:
                primaries[primary] = index_for([primary])
            def expand(domain: str) -> str:
                if domain not in primaries:
                    primaries[domain] = index_for([domain])
                return chat.expand_query(
                    model, tokenizer, case["question"], lora_layers=None,
                    act_titles=primaries[domain].act_titles(), domain=domain,
                )

            expansion = expand(primary)
            row = {"id": case["id"], "set": job["set"], "gold_domain": case["domain"], "primary": primary,
                   "router_confidence": routing["confidence"], "router_secondary": routing["secondary"],
                   "max_rank": case["max_acceptable_rank"], "expansion": expansion, "cfg": {}}
            for name, domains in configs(case, routing, all_domains).items():
                own = name in OWN_EXPANSION and domains[0] != primary
                config_expansion = expand(domains[0]) if own else expansion
                idx = primaries[domains[0]] if len(domains) == 1 else index_for(domains)
                results = idx.search(case["question"], top_k=CANDIDATE_POOL, extra_queries=[config_expansion],
                                     candidate_pool=RERANK_CANDIDATE_POOL)
                row["cfg"][name] = {"domains": domains, "rank": find_rank(results, case["act_short"], case["article"]),
                                    "own_expansion": config_expansion if own else None, **signals(results)}
                del idx
                release_accelerator_cache()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            ranks = {k: v["rank"] for k, v in row["cfg"].items()}
            print(f"[{n}/{len(jobs)}] {case['id']} {time.time() - started:.0f}s ranks={ranks}", flush=True)


if __name__ == "__main__":
    main()
