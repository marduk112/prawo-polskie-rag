"""Porównanie odpowiedzi Bielika z adapterem LoRA i bez (ten sam kontekst RAG) -- PROGRESS.md, Krok 61.

Dla każdej dziedziny: model ładowany raz z adapterem, dla każdego pytania
jedna ekspansja (zawsze bazą, jak w chat.answer) + jedno wyszukiwanie, potem
dwie generacje -- LoRA włączona / wyłączona (set_lora_enabled). Wyniki do JSON
(wznawialne -- już policzone pary są pomijane). Automatycznie oznacza tylko
cytaty spoza kontekstu RAG; błędne przypisanie treści do numeru przepisu
z kontekstu wymaga ręcznego sprawdzenia z tekstem ustawy.

Pytania (z notatką 'expect' do ręcznej oceny) w data/adapter_vs_base_questions.json;
pytania bliskie treningowym (kosinus >= 0.85) są pomijane.

Użycie (jeden proces MLX naraz):
    python scripts/compare_adapter_vs_base.py
"""
import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402
from mlx_lm import generate, load  # noqa: E402

from chat import LEGACY_ADAPTER_PATHS, MODEL_PATH, expand_query, find_lora_layers, set_lora_enabled  # noqa: E402
from prompt import build_system_prompt, build_user_message, search_with_history  # noqa: E402
from rag_search import QUERY_PREFIX, RagIndex, release_accelerator_cache  # noqa: E402

OUT = ROOT / "data" / "adapter_vs_base_comparison.json"

QUESTIONS_PATH = ROOT / "data" / "adapter_vs_base_questions.json"
# Pytanie zbyt podobne do treningowego zawyżyłoby wynik adaptera (zapamiętana
# odpowiedź) -- ten sam próg co w distill_qwen.py (Krok 72).
NEAR_DUPLICATE_COSINE = 0.85


def load_questions() -> dict[str, list[dict]]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def training_questions(domain: str) -> list[str]:
    out = []
    for name in ("train.jsonl", "valid.jsonl"):
        path = ROOT / "data" / "finetune" / domain / name
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                msgs = json.loads(line).get("messages", [])
                out += [m["content"] for m in msgs if m["role"] == "user"][:1]
    return out


def near_duplicates(rag: RagIndex, questions: list[str], train: list[str]) -> dict[str, tuple[float, str]]:
    if not train:
        return {}
    enc = lambda texts: rag.model.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True)  # noqa: E731
    sims = enc(questions) @ enc(train).T
    return {
        q: (float(sims[i].max()), train[int(sims[i].argmax())])
        for i, q in enumerate(questions)
        if sims[i].max() >= NEAR_DUPLICATE_COSINE
    }


CITE_RE = re.compile(r"art\.\s*(\d+[a-ząćęłńóśźż]{0,3}[⁰¹²³⁴⁵⁶⁷⁸⁹]{0,2})", re.IGNORECASE)


def main() -> None:
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["domain"], r["question"]) for r in results}
    shared_rag = None
    for domain, items in load_questions().items():
        expect = {it["q"]: it.get("expect") for it in items}
        questions = list(expect)
        if all((domain, q) in done for q in questions):
            continue
        adapter = str(LEGACY_ADAPTER_PATHS[domain])
        print(f"\n##### {domain} (adapter={Path(adapter).name})", flush=True)
        model, tokenizer = load(str(MODEL_PATH), adapter_path=adapter)
        lora_layers = find_lora_layers(model)
        rag = RagIndex(
            domain=domain,
            model=shared_rag.model if shared_rag else None,
            cross_encoder=shared_rag.cross_encoder if shared_rag else None,
        )
        shared_rag = rag
        system = build_system_prompt(domain)
        dups = near_duplicates(rag, [q for q in questions if (domain, q) not in done], training_questions(domain))
        for q, (sim, tq) in dups.items():
            print(f"[{domain}] POMINIĘTE (kosinus {sim:.2f} do treningowego: {tq[:70]}): {q[:70]}", flush=True)
        for q in questions:
            if (domain, q) in done or q in dups:
                continue
            expanded = expand_query(model, tokenizer, q, lora_layers=lora_layers, act_titles=rag.act_titles(), domain=domain)
            extra = [expanded] if expanded and expanded.strip().lower() != q.strip().lower() else None
            hits = search_with_history(rag, [], q, 8, extra_queries=extra)
            context_articles = sorted({h["article"] for h in hits})
            user = {"role": "user", "content": build_user_message(q, hits, [])}
            prompt = tokenizer.apply_chat_template([{"role": "system", "content": system}, user], add_generation_prompt=True)
            out = {}
            for label, enabled in (("adapter", True), ("base", False)):
                set_lora_enabled(model, lora_layers, enabled=enabled)
                out[label] = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False).strip()
            set_lora_enabled(model, lora_layers, enabled=True)
            rec = {
                "domain": domain,
                "question": q,
                "expect": expect[q],
                "expanded": expanded,
                "context_articles": context_articles,
                "context": [f"{h['act_short']} art. {h['article']}" for h in hits],
            }
            for label, text in out.items():
                cited = sorted({m.group(1) for m in CITE_RE.finditer(text)})
                rec[label] = text
                rec[f"{label}_cited"] = cited
                rec[f"{label}_cited_outside_context"] = [c for c in cited if c not in context_articles]
            results.append(rec)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{domain}] {q[:60]} | adapter poza kontekstem: {rec['adapter_cited_outside_context']} "
                  f"| baza poza kontekstem: {rec['base_cited_outside_context']}", flush=True)
            release_accelerator_cache()
        del model, tokenizer, lora_layers
        gc.collect()
        mx.clear_cache()
    print("\nGOTOWE", flush=True)


if __name__ == "__main__":
    main()
