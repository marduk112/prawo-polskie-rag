"""
Przetestowane end-to-end na realnej karcie NVIDIA (RunPod, A40) --
patrz PROGRESS.md, Krok 26, oraz train_lora_cuda.py po dwa bugi
znalezione i naprawione przy tej okazji. Domyślny model to
11B (v3.1, bez adaptera): 4.5B ma udokumentowaną (Krok 16, Krok 25, Krok 26) tendencję do
przesłaniania niepełnego/brakującego kontekstu z RAG pewną, ale
zmyśloną liczbą z wag -- 11B tego problemu nie miał w żadnym
przetestowanym przypadku, nawet bez adaptera (`--no-adapter`).

Odpowiednik scripts/chat.py dla Linuksa/Windows z kartą NVIDIA:
RAG (rag_search.RagIndex) + Bielik doduczony przez
scripts/train_lora_cuda.py, generacja przez transformers + peft
zamiast MLX. Ten sam prompt systemowy i budowanie kontekstu co
pozostałe warianty (scripts/prompt.py).

Tryb interaktywny pamięta kontekst rozmowy (poprzednie pytania i
odpowiedzi trafiają do promptu przy kolejnych turach) -- wpisz /nowy,
żeby zacząć nowy wątek i wyczyścić historię. Historia jest ucinana do
ostatnich --max-turns par pytanie/odpowiedź, żeby nie przepełnić okna
kontekstu modelu (każda tura dokłada też fragmenty RAG).

Można też wgrać własny plik PDF (tekstowy, nie skan) jako dodatkowy
kontekst -- wpisz /plik <ścieżka> w trybie interaktywnym albo podaj
--file przy starcie. Fragmenty z pliku są wyraźnie oznaczane jako
"treść pliku użytkownika", nie mylone z obowiązującym prawem.

Można też zapytać o stan prawny na konkretny dzień w przeszłości (wymaga
indeksu RAG zbudowanego z --include-history, patrz PROGRESS.md Krok 18/19)
-- wpisz /data RRRR-MM-DD w trybie interaktywnym albo podaj --as-of przy
starcie; samo /data (bez daty) wraca do stanu bieżącego.

Użycie:
    python scripts/chat_cuda.py
    python scripts/chat_cuda.py --prompt "pytanie..."
    python scripts/chat_cuda.py --no-adapter
    python scripts/chat_cuda.py --file umowa.pdf
    python scripts/chat_cuda.py --as-of 2019-06-01
    python scripts/chat_cuda.py \\
        --model speakleash/Bielik-4.5B-v3.0-Instruct \\
        --adapter-path adapters-cuda/bielik4.5b-kadry-lora-hedge/final
"""

import argparse
from contextlib import nullcontext
from datetime import date

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from file_index import SessionFileIndex
from prompt import (
    build_query_expansion_messages,
    build_system_prompt,
    build_user_message,
    looks_like_meta_question,
    search_with_history,
    trim_history,
)
from rag_search import RagIndex

# Krok 118/122: domyślnie Bielik v3.1 BEZ adapterów (adaptery v3.0 nie ładują się na v3.1,
# a w Kroku 122 produkcyjne adaptery v3.0 nie różniły się od v3.1 bazowego). Poniższy
# słownik to stare adaptery (v3.0), dostępne tylko jawnie przez --adapter-path / --legacy-adapter.
DEFAULT_MODEL = "speakleash/Bielik-11B-v3.1-Instruct"
DEFAULT_DOMAIN = "kadry_i_place"
LEGACY_ADAPTER_PATHS = {
    "kadry_i_place": "adapters-cuda/bielik11b-kadry-lora-hedge/final",
    "podatkowe": "adapters-cuda/bielik11b-podatkowe-lora-iter25/final",
    "administracyjne": "adapters-cuda/bielik11b-administracyjne-lora-iter25/final",
    "cywilne": "adapters-cuda/bielik11b-cywilne-lora-iter25/final",
    "rodzinne": "adapters-cuda/bielik11b-rodzinne-lora-iter25/final",
    "karne": "adapters-cuda/bielik11b-karne-lora-iter25/final",
    "gospodarcze": "adapters-cuda/bielik11b-gospodarcze-lora-iter25/final",
    "budowlane": "adapters-cuda/bielik11b-budowlane-lora-iter25/final",
    "autorskie": "adapters-cuda/bielik11b-autorskie-lora-iter25/final",
    "wlasnosc_przemyslowa": "adapters-cuda/bielik11b-wlasnosc-przemyslowa-lora-iter25/final",
    "ochrona_danych": "adapters-cuda/bielik11b-ochrona-danych-lora-iter25/final",
    "zamowienia_publiczne": "adapters-cuda/bielik11b-zamowienia-publiczne-lora-iter25/final",
    "ochrona_srodowiska": "adapters-cuda/bielik11b-ochrona-srodowiska-lora-iter25/final",
    "ruch_drogowy": "adapters-cuda/bielik11b-ruch-drogowy-lora-iter25/final",
    "upadlosciowe": "adapters-cuda/bielik11b-upadlosciowe-lora-iter25/final",
    "lotnicze": "adapters-cuda/bielik11b-lotnicze-lora-iter25/final",
    "geologia_i_gornictwo": "adapters-cuda/bielik11b-geologia-i-gornictwo-lora-iter25/final",
    "wodne": "adapters-cuda/bielik11b-wodne-lora-iter25/final",
    "mieszkaniowe": "adapters-cuda/bielik11b-mieszkaniowe-lora-iter25/final",
    "energetyczne": "adapters-cuda/bielik11b-energetyczne-lora-iter25/final",
    "telekomunikacyjne": "adapters-cuda/bielik11b-telekomunikacyjne-lora-iter25/final",
    "konsumenckie": "adapters-cuda/bielik11b-konsumenckie-lora-iter25/final",
}


def load_model(model_id: str, adapter_path: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def expand_query(
    model, tokenizer, question: str, act_titles: list[str] | None = None, domain: str | None = None
) -> str:
    """Przeformułowuje pytanie na język formalny/prawniczy jednym, krótkim
    wywołaniem modelu -- patrz prompt.build_query_expansion_messages().

    Wyłącza adapter LoRA (jeśli jest załadowany) na czas tego wywołania,
    tym samym mechanizmem co `chat.py::set_lora_enabled` (patrz
    PROGRESS.md, Krok 21) -- w wersji MLX empirycznie potwierdzono, że
    domenowy adapter, mimo odmiennego promptu systemowego każącego mu nie
    cytować przepisów, i tak "przeciekał" swój wyuczony nawyk cytowania
    do tego wywołania, czasem podstawiając pod poprawny numer artykułu
    treść zupełnie innego przepisu. `PeftModel.disable_adapter()` to
    natywny odpowiednik tego mechanizmu dla `peft` -- nieprzetestowane na
    tym wariancie (brak wytrenowanych adapterów CUDA, patrz PROGRESS.md
    "Co dalej"), ale ta sama zasada powinna dotyczyć obu backendów, bo
    przyczyna leży w wagach LoRA, nie w konkretnej bibliotece.

    `act_titles` (z RagIndex.act_titles()): patrz
    prompt.build_query_expansion_messages() (Krok 28) po uzasadnienie.
    `domain` -- patrz DOMAIN_QUERY_EXPANSION_HINTS (Krok 41)."""
    messages = build_query_expansion_messages(question, act_titles=act_titles, domain=domain)
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    no_lora = model.disable_adapter() if isinstance(model, PeftModel) else nullcontext()
    with no_lora, torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer(
    model,
    tokenizer,
    rag: RagIndex,
    file_index: SessionFileIndex,
    history: list[dict],
    question: str,
    top_k: int,
    as_of: str | None = None,
    expand_query_enabled: bool = True,
) -> str:
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        extra_queries = None
        if expand_query_enabled:
            expanded = expand_query(model, tokenizer, question, act_titles=rag.act_titles(), domain=rag.domain)
            if expanded and expanded.strip().lower() != question.strip().lower():
                extra_queries = [expanded]
        results = search_with_history(rag, history, question, top_k, as_of=as_of, extra_queries=extra_queries)
        file_results = file_index.search(question, top_k=3)
        current_turn = {
            "role": "user",
            "content": build_user_message(question, results, file_results, as_of=as_of),
        }
    messages = [{"role": "system", "content": build_system_prompt(rag.domain)}] + history + [current_turn]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # W historii trzymamy samo pytanie (bez fragmentów RAG) -- fragmenty
    # doklejane są od nowa przy każdej turze, więc trzymanie ich też w
    # historii bardzo szybko przepełnia okno kontekstu (patrz PROGRESS.md,
    # krok 10).
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    return text


def main():
    parser = argparse.ArgumentParser(description="Asystent prawa polskiego, wielodziedzinowy (RAG + Bielik LoRA, CUDA)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Repo Hugging Face modelu bazowego")
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help=f"Dziedzina prawa: wybiera indeks RAG (domyślnie: {DEFAULT_DOMAIN}).",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None, help="Ścieżka do adaptera LoRA (domyślnie: bez adaptera)"
    )
    parser.add_argument(
        "--legacy-adapter",
        action="store_true",
        help="Załaduj stary adapter v3.0 danej dziedziny z LEGACY_ADAPTER_PATHS (wymaga --model speakleash/Bielik-11B-v3.0-Instruct)",
    )
    parser.add_argument("--no-adapter", action="store_true", help="(bez efektu; zostaje dla zgodności -- domyślnie nie ma adaptera)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Liczba fragmentów RAG dołączanych do kontekstu (domyślnie 8 -- podniesione z 5, patrz "
        "PROGRESS.md, Krok 21/22/24: przy 5 fragment z nowo dodanych rozporządzeń o konkretnych kwotach "
        "czasem przegrywał rankingiem z kilkoma tematycznie bliskimi artykułami tej samej ustawy bazowej)",
    )
    parser.add_argument("--max-turns", type=int, default=6, help="Ile ostatnich par pytanie/odpowiedź zachować w kontekście rozmowy")
    parser.add_argument("--file", type=str, default=None, help="Ścieżka do pliku PDF (tekstowego) wgrywanego jako dodatkowy kontekst")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        metavar="RRRR-MM-DD",
        help="Pytaj o stan prawny na ten dzień zamiast bieżącego (wymaga indeksu "
        "zbudowanego z build_rag_index.py --include-history)",
    )
    parser.add_argument(
        "--no-expand-query",
        action="store_true",
        help="Wyłącz automatyczne przeformułowanie pytania na język formalny/prawniczy "
        "przed wyszukiwaniem RAG (domyślnie włączone).",
    )
    args = parser.parse_args()

    legacy = LEGACY_ADAPTER_PATHS.get(args.domain) if args.legacy_adapter else None
    adapter_path = None if args.no_adapter else (args.adapter_path or legacy)
    print(f"[INFO] Ładowanie modelu ({args.model}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load_model(args.model, adapter_path)

    print(f"[INFO] Ładowanie indeksu RAG (dziedzina: {args.domain})...")
    rag = RagIndex(domain=args.domain)
    file_index = SessionFileIndex(model=rag.model)

    if args.file:
        n = file_index.add_pdf(args.file)
        print(f"[INFO] Wgrano plik {args.file} ({n} fragmentów)")

    if args.as_of:
        print(f"[INFO] Stan prawny na dzień: {args.as_of}")

    if args.prompt:
        print(answer(model, tokenizer, rag, file_index, [], args.prompt, args.top_k, as_of=args.as_of, expand_query_enabled=not args.no_expand_query))
        return

    print(f"\nAsystent (CUDA) gotowy (dziedzina: {args.domain}). Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek, /plik <ścieżka> aby wgrać PDF, /data RRRR-MM-DD aby zapytać o stan prawny na dany dzień).\n")
    history: list[dict] = []
    as_of = args.as_of
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/nowy":
            # Celowo NIE czyści `as_of` -- tak samo jak nie czyści wgranego
            # pliku (file_index) -- osobny "tryb" sesji, czyści go
            # wyłącznie jawne /data.
            history = []
            print("[INFO] Rozpoczęto nowy wątek rozmowy.\n")
            continue
        if question.startswith("/plik "):
            path = question[len("/plik "):].strip()
            try:
                n = file_index.add_pdf(path)
                print(f"[INFO] Wgrano plik {path} ({n} fragmentów)\n")
            except (FileNotFoundError, ValueError) as e:
                print(f"[BŁĄD] {e}\n")
            continue
        if question == "/data" or question.startswith("/data "):
            arg = question[len("/data"):].strip()
            if not arg:
                as_of = None
                print("[INFO] Wrócono do stanu bieżącego.\n")
            else:
                try:
                    date.fromisoformat(arg)
                except ValueError as e:
                    print(f"[BŁĄD] Nieprawidłowa data ({e}), oczekiwano RRRR-MM-DD.\n")
                    continue
                as_of = arg
                print(f"[INFO] Stan prawny na dzień: {as_of}\n")
            continue
        text = answer(model, tokenizer, rag, file_index, history, question, args.top_k, as_of=as_of, expand_query_enabled=not args.no_expand_query)
        print("Asystent:", text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    main()
