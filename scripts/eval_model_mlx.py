"""Krok 114: odpowiedzi DOWOLNEGO modelu MLX (np. Bielik 11B v3.1 8-bit) na GOTOWYCH kontekstach RAG z istniejących plików
ewaluacyjnych (eval_models.json dla kadr, eval_vllm.json dla pozostałych dziedzin) -- te same wiadomości co dla Bielika v3.0 / Qwena,
zachłannie, max 500 tokenów (jak chat.py). Wynik: data/distill/<d>/<out-file> = {contexts, answers: {<wariant>: {...}}} (wznawialne).
Jeden proces MLX naraz.

Użycie: python scripts/eval_model_mlx.py --model-path <katalog MLX> --variant bielik-11b-v3.1-8bit \
          --source kadry_i_place:eval_models.json --source budowlane:eval_vllm.json ...
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mlx_lm import generate, load  # noqa: E402

from distill_qwen import cited_outside_context, domain_dir  # noqa: E402
from rag_search import release_accelerator_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", action="append", required=True, metavar="DZIEDZINA:PLIK", help="plik z kontekstami w data/distill/<d>/")
    parser.add_argument("--out-file", default="eval_v31.json")
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    model, tokenizer = load(args.model_path)
    for spec in args.source:
        domain, fname = spec.split(":", 1)
        src = json.loads((domain_dir(domain) / fname).read_text(encoding="utf-8"))
        out = domain_dir(domain) / args.out_file
        data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"contexts": src["contexts"], "answers": {}}
        answers = data["answers"].setdefault(args.variant, {})
        for q, ctx in src["contexts"].items():
            if q in answers:
                continue
            prompt = tokenizer.apply_chat_template(ctx["messages"], add_generation_prompt=True)
            text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False).strip()
            answers[q] = {"text": text, "outside_context": cited_outside_context(text, ctx["context_articles"], ctx["messages"][-1]["content"])}
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{domain}] {len(answers)}/{len(src['contexts'])} | poza kontekstem: {answers[q]['outside_context']}", flush=True)
            release_accelerator_cache()
        print(f"### {domain} gotowe: {len(answers)} odpowiedzi", flush=True)


if __name__ == "__main__":
    main()
