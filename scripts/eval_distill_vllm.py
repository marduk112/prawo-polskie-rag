"""Ewaluacja adapterów po destylacji na podzie CUDA (PROGRESS.md, Krok 76) --
odpowiednik scripts/eval_distill.py (MLX) dla Bielika na vLLM z adapterami
LoRA ładowanymi przez --lora-modules.

Te same pytania (distill_qwen.eval_questions -- trzymane z dala od zbioru
treningowego) i ten sam kontekst dla każdego wariantu: ekspansja zapytania
Bielikiem bazowym (jak chat.expand_query, zachłannie, max 80 tokenów),
wyszukiwanie liczone raz. Odpowiedzi zachłannie, domyślnie max 500 tokenów --
jak chat.py (--max-tokens/--reasoning-effort dla Qwena, scripts/runpod/compare_models.sh). Wyniki w data/distill/<dziedzina>/eval_vllm.json (wznawialne, ten sam
format co eval.json). Merytoryczną poprawność trzeba przeczytać ręcznie.

Serwer (przykład):
    vllm serve speakleash/Bielik-11B-v3.0-Instruct --served-model-name bielik-11b \\
        --enable-lora --max-lora-rank 8 --lora-modules ckpt300=adapters-cuda/.../checkpoint-300

Użycie:
    python scripts/eval_distill_vllm.py --url http://127.0.0.1:8001/v1 --domain budowlane \\
        --variants base ckpt300 ckpt500
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from chat_lmstudio import post_chat  # noqa: E402
from distill_qwen import cited_outside_context, domain_dir, eval_questions  # noqa: E402
from prompt import build_query_expansion_messages, build_system_prompt, build_user_message, search_with_history  # noqa: E402
from rag_search import RagIndex  # noqa: E402

BASE_MODEL_NAME = "bielik-11b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Bazowy URL vLLM (z /v1)")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variants", nargs="+", required=True, help="'base' albo nazwy z --lora-modules")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--questions-file", help="Plik JSON z listą pytań (zamiast eval_questions; Krok 88)")
    parser.add_argument("--out-file", default="eval_vllm.json", help="Plik wynikowy w data/distill/<d>/")
    parser.add_argument("--max-tokens", type=int, default=500, help="Limit tokenów odpowiedzi (Qwen z myśleniem: ~3000)")
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default=None,
        help="Dla modeli z trybem thinking (Qwen3.8) -- jak w chat_lmstudio.py; dotyczy tylko odpowiedzi, nie ekspansji zapytania",
    )
    parser.add_argument(
        "--template-kwargs", default=None,
        help='JSON z chat_template_kwargs dla odpowiedzi (np. \'{"enable_thinking": false}\' dla Gemma 4 / Qwen3.6); łączony z --reasoning-effort',
    )
    args = parser.parse_args()
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else {}

    out = domain_dir(args.domain) / args.out_file
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"contexts": {}, "answers": {}}
    questions = (
        json.loads(Path(args.questions_file).read_text(encoding="utf-8"))
        if args.questions_file
        else eval_questions(args.domain)
    )

    def chat(model: str, messages: list[dict], max_tokens: int, reasoning_effort: str | None = None) -> str:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
            # vLLM przekazuje parametry szablonu czatu tylko przez chat_template_kwargs (jak chat_lmstudio.py)
            payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        if template_kwargs and max_tokens > 80:  # tylko odpowiedzi, nie ekspansja zapytania
            payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), **template_kwargs}
        return post_chat(args.url, payload, timeout=600)["content"].strip()

    todo = [q for q in questions if q not in data["contexts"]]
    if todo:
        rag = RagIndex(domain=args.domain)
        act_titles = rag.act_titles()
        system = build_system_prompt(args.domain)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            expansions = list(pool.map(
                lambda q: chat(BASE_MODEL_NAME, build_query_expansion_messages(q, act_titles=act_titles, domain=args.domain), 80),
                todo,
            ))
        for q, expanded in zip(todo, expansions):
            extra = [expanded] if expanded and expanded.strip().lower() != q.strip().lower() else None
            hits = search_with_history(rag, [], q, 8, extra_queries=extra)
            data["contexts"][q] = {
                "expanded": expanded,
                "context_articles": [[h["act_short"], h["article"]] for h in hits],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": build_user_message(q, hits, [])}],
            }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for variant in args.variants:
        answers = data["answers"].setdefault(variant, {})
        model = BASE_MODEL_NAME if variant == "base" else variant
        missing = [q for q in questions if q not in answers]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            texts = list(pool.map(lambda q: chat(model, data["contexts"][q]["messages"], args.max_tokens, args.reasoning_effort), missing))
        for q, text in zip(missing, texts):
            ctx = data["contexts"][q]
            answers[q] = {"text": text, "outside_context": cited_outside_context(text, ctx["context_articles"], ctx["messages"][-1]["content"])}
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        flagged = sum(bool(a["outside_context"]) for a in answers.values())
        print(f"[{variant}] odpowiedzi: {len(answers)}, z cytatem spoza kontekstu: {flagged}", flush=True)


if __name__ == "__main__":
    main()
