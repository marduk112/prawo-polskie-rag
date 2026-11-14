"""Ewaluacja adaptera po destylacji (PROGRESS.md, Krok 72): te same pytania
ewaluacyjne (distill_qwen.eval_questions -- trzymane z dala od zbioru
treningowego) dla kilku wariantów Bielika, z identycznym kontekstem RAG.

Kontekst (expand_query na bazie + wyszukiwanie) liczony raz na pytanie i
zapisywany, więc każdy wariant odpowiada na DOKŁADNIE te same wiadomości.
Wyniki dopisywane do data/distill/<dziedzina>/eval.json (wznawialne).
Automatycznie oznacza tylko cytaty spoza kontekstu -- merytoryczną
poprawność trzeba przeczytać ręcznie.

Użycie (jeden proces MLX naraz):
    python scripts/eval_distill.py --domain konsumenckie --variant base
    python scripts/eval_distill.py --domain konsumenckie --variant adapter:adapters/bielik11b-konsumenckie-lora-iter25
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mlx_lm import generate, load  # noqa: E402

from chat import MODEL_PATH, expand_query  # noqa: E402
from distill_qwen import cited_outside_context, domain_dir, eval_questions  # noqa: E402
from prompt import build_system_prompt, build_user_message, search_with_history  # noqa: E402
from rag_search import RagIndex, release_accelerator_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variant", required=True, help="'base' albo 'adapter:<ścieżka>'")
    args = parser.parse_args()

    out = domain_dir(args.domain) / "eval.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"contexts": {}, "answers": {}}
    questions = eval_questions(args.domain)
    adapter = args.variant.split(":", 1)[1] if args.variant.startswith("adapter:") else None
    answers = data["answers"].setdefault(args.variant, {})

    model, tokenizer = load(str(MODEL_PATH))
    rag = None
    for q in questions:
        if q in data["contexts"]:
            continue
        rag = rag or RagIndex(domain=args.domain)
        expanded = expand_query(model, tokenizer, q, lora_layers=None, act_titles=rag.act_titles(), domain=args.domain)
        extra = [expanded] if expanded and expanded.strip().lower() != q.strip().lower() else None
        hits = search_with_history(rag, [], q, 8, extra_queries=extra)
        data["contexts"][q] = {
            "context_articles": [[h["act_short"], h["article"]] for h in hits],
            "messages": [
                {"role": "system", "content": build_system_prompt(args.domain)},
                {"role": "user", "content": build_user_message(q, hits, [])},
            ],
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        release_accelerator_cache()

    if adapter:
        del model
        model, tokenizer = load(str(MODEL_PATH), adapter_path=adapter)
    for q in questions:
        if q in answers:
            continue
        ctx = data["contexts"][q]
        prompt = tokenizer.apply_chat_template(ctx["messages"], add_generation_prompt=True)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False).strip()
        answers[q] = {"text": text, "outside_context": cited_outside_context(text, ctx["context_articles"], ctx["messages"][-1]["content"])}
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{args.variant}] {q[:60]} | poza kontekstem: {answers[q]['outside_context']}", flush=True)
        release_accelerator_cache()


if __name__ == "__main__":
    main()
