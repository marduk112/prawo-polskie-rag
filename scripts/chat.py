"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + douczony Bielik (LoRA).

RAG dostarcza aktualny, źródłowy tekst artykułów ustaw jako kontekst w
promptcie -- LoRA uczy model odpowiedniego stylu odpowiedzi (zwięzłość,
cytowanie artykułów, przyznawanie się do niewiedzy, gdy kontekst nie
wystarcza). Fakty mają pochodzić z kontekstu RAG, a nie z pamięci wag
modelu -- to naprawia niedokładności zaobserwowane w testach samego
douczonego modelu bez kontekstu (patrz PROGRESS.md, krok 4b).

Domyślny model odpowiedzi: Bielik-11B-v3.1-Instruct 8-bit (Krok 118; v3.0 4-bit: `--model models/Bielik-11B-v3.0-Instruct-mlx`);
ekspansja zapytań nadal na v3.0 4-bit (`--expand-model`, bateria 139 z v3.1: 9 nowych regresji).
Adaptery LoRA z DEFAULT_ADAPTER_PATHS były trenowane na v3.0 -- domyślnie ładują się TYLKO z modelem v3.0
(`resolve_adapter_path`); z v3.1 działa sam model bazowy, chyba że podasz jawnie --adapter-path.
Adapter LoRA tylko w dziedzinach z DEFAULT_ADAPTER_PATHS (Krok 73: stare adaptery iter25 wypadały gorzej od bazy).
Dla kadr: bielik11b-kadry-lora-hedge-iter50 (patrz PROGRESS.md, Krok 27 -- douczony
na zbiorze z Kroku 26 rozszerzonym o przykłady hedgingu). Zastrzeżenie:
ten adapter NIE jest w 100% niezawodny w przyznawaniu się do niewiedzy
poza korpusem RAG (patrz Krok 27, test VAT) -- czysty model bazowy bez
adaptera (--no-adapter) wypadał na tym wymiarze czyściej we wszystkich
dotąd przetestowanych przypadkach, kosztem gorszego stylu/cytowania na
faktach, które RAG dostarcza poprawnie.

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
    python scripts/chat.py                          # tryb interaktywny
    python scripts/chat.py --prompt "pytanie..."     # pojedyncze pytanie
    python scripts/chat.py --no-adapter              # bazowy model bez LoRA
    python scripts/chat.py --file umowa.pdf          # z wgranym plikiem od startu
    python scripts/chat.py --as-of 2019-06-01        # stan prawny na dany dzień
    python scripts/chat.py \\
        --model models/Bielik-4.5B-v3.0-Instruct-mlx \\
        --adapter-path adapters/bielik-kadry-lora-v3-iter25  # szybszy wariant 4.5B
"""

import argparse
from datetime import date
from pathlib import Path

from mlx.utils import tree_unflatten
from mlx_lm import generate, load
from mlx_lm.tuner.lora import LoRALinear

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

ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH = baza v3.0 4-bit: to ją importują skrypty ewaluacji/destylacji i adaptery LoRA (trenowane na v3.0) -- NIE zmieniać po cichu.
MODEL_PATH = ROOT / "models" / "Bielik-11B-v3.0-Instruct-mlx"
# Krok 118: domyślny model ODPOWIEDZI w CLI chat.py (--model): Bielik v3.1 8-bit.
ANSWER_MODEL_PATH = ROOT / "models" / "Bielik-11B-v3.1-Instruct-mlx-8bit"
# Adaptery z DEFAULT_ADAPTER_PATHS są dopasowane do wag Bielika v3.0 (Krok 118) -- na innej bazie nie ładujemy ich domyślnie.
ADAPTER_BASE_MODEL_PREFIX = "Bielik-11B-v3.0"
# Krok 118: 500 tokenów ucinało odpowiedzi Bielika (v3.0 i v3.1: max ok. 182 słów, mediana ok. 160); 1000 to ok. 2x dłuższy czas generowania.
DEFAULT_MAX_TOKENS = 1000
# Krok 118: ekspansja zapytań zostaje na Bieliku v3.0 4-bit -- bateria 139 z v3.1 8-bit dała 9 NOWYCH regresji retrievalu
# (129 OK + 1 znany + 9 nowych vs 136 OK + 3 znane z v3.0). Odpowiedzi generuje model główny (--model), ekspansję model --expand-model.
EXPAND_MODEL_PATH = MODEL_PATH

DEFAULT_DOMAIN = "kadry_i_place"

# Stare adaptery LoRA per dziedzina (mały ręczny zbiór pytanie -> odpowiedź BEZ
# kontekstu RAG, iter25). Krok 73: na 89 pytaniach spoza treningu w 17
# dziedzinach wypadły średnio GORZEJ od bazy (błędne wnioski ~10 vs 7, poważne
# błędy podstawy 11 vs 5; wcześniej Krok 61 i 68/72) -- dlatego nie są już
# domyślne. Zostają do porównań (scripts/compare_adapter_vs_base.py) i ręcznego
# użycia przez --adapter-path.
LEGACY_ADAPTER_PATHS = {
    "kadry_i_place": ROOT / "adapters" / "bielik11b-kadry-lora-hedge-iter50",
    "podatkowe": ROOT / "adapters" / "bielik11b-podatkowe-lora-iter25",
    "administracyjne": ROOT / "adapters" / "bielik11b-administracyjne-lora-iter25",
    "cywilne": ROOT / "adapters" / "bielik11b-cywilne-lora-iter25",
    "rodzinne": ROOT / "adapters" / "bielik11b-rodzinne-lora-iter25",
    "karne": ROOT / "adapters" / "bielik11b-karne-lora-iter25",
    "gospodarcze": ROOT / "adapters" / "bielik11b-gospodarcze-lora-iter25",
    "budowlane": ROOT / "adapters" / "bielik11b-budowlane-lora-iter25",
    "autorskie": ROOT / "adapters" / "bielik11b-autorskie-lora-iter25",
    "wlasnosc_przemyslowa": ROOT / "adapters" / "bielik11b-wlasnosc-przemyslowa-lora-iter25",
    "ochrona_danych": ROOT / "adapters" / "bielik11b-ochrona-danych-lora-iter25",
    "zamowienia_publiczne": ROOT / "adapters" / "bielik11b-zamowienia-publiczne-lora-iter25",
    "ochrona_srodowiska": ROOT / "adapters" / "bielik11b-ochrona-srodowiska-lora-iter25",
    "ruch_drogowy": ROOT / "adapters" / "bielik11b-ruch-drogowy-lora-iter25",
    "upadlosciowe": ROOT / "adapters" / "bielik11b-upadlosciowe-lora-iter25",
    "lotnicze": ROOT / "adapters" / "bielik11b-lotnicze-lora-iter25",
    "geologia_i_gornictwo": ROOT / "adapters" / "bielik11b-geologia-i-gornictwo-lora-iter25",
    "wodne": ROOT / "adapters" / "bielik11b-wodne-lora-iter25",
    "mieszkaniowe": ROOT / "adapters" / "bielik11b-mieszkaniowe-lora-iter25",
    "energetyczne": ROOT / "adapters" / "bielik11b-energetyczne-lora-iter25",
    "telekomunikacyjne": ROOT / "adapters" / "bielik11b-telekomunikacyjne-lora-iter25",
    "konsumenckie": ROOT / "adapters" / "bielik11b-konsumenckie-lora-iter25",
}

# Adapter domyślny per dziedzina -- tylko tam, gdzie wypadł nie gorzej od bazy.
# Dziedziny spoza słownika działają na samym modelu bazowym (bez LoRA).
DEFAULT_ADAPTER_PATHS = {
    # `kadry_i_place` (hedge-iter50) usunięty w Kroku 89: na 76 pytaniach (Krok 88) brak
    # mierzalnej przewagi nad bazą (33/11/25, błędy 26/25, p~0,36), a bywa zbyt skłonny do
    # odmowy. Adapter zostaje w LEGACY_ADAPTER_PATHS (--adapter-path).
    # Krok 72: adapter z destylacji Qwena (przykłady z kontekstem RAG)
    "konsumenckie": ROOT / "adapters" / "bielik11b-konsumenckie-distill-lora-iter200",
    # Krok 76: destylacja na RunPodzie (trening CUDA, konwersja peft-to-mlx); w Kroku 78
    # sędzia Qwen: 9 lepszy / 4 remis / 6 gorszy od bazy (nowy iter407 przegrał 5:11)
    "budowlane": ROOT / "adapters" / "bielik11b-budowlane-distill-lora-iter584",
    # Krok 78 (8 dziedzin naraz, pełne ustępy długich artykułów): tylko tam, gdzie sędzia
    # Qwen nie wskazał przewagi bazy -- cywilne 12/4/11, mieszkaniowe 8/7/4
    "cywilne": ROOT / "adapters" / "bielik11b-cywilne-distill-k78-lora-iter550",
    "mieszkaniowe": ROOT / "adapters" / "bielik11b-mieszkaniowe-distill-k78-lora-iter413",
}


def find_lora_layers(model) -> list[tuple[str, LoRALinear]]:
    """Znajduje wszystkie warstwy LoRA w załadowanym modelu (puste, jeśli
    model działa bez adaptera) -- do jednorazowego wywołania po load(),
    wynik przekazywany do expand_query() (patrz set_lora_enabled())."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]


def set_lora_enabled(model, lora_layers: list[tuple[str, LoRALinear]], enabled: bool) -> None:
    """Włącza/wyłącza wagi LoRA w miejscu, bez ponownego wczytywania z dysku
    (podmienia już wczytane moduły `LoRALinear` na ich bazowe `.linear` i
    z powrotem) -- patrz expand_query()."""
    if not lora_layers:
        return
    replacement = lora_layers if enabled else [(name, module.linear) for name, module in lora_layers]
    model.update_modules(tree_unflatten(replacement))


def expand_query(
    model,
    tokenizer,
    question: str,
    lora_layers: list[tuple[str, LoRALinear]] | None = None,
    act_titles: list[str] | None = None,
    domain: str | None = None,
) -> str:
    """Przeformułowuje pytanie na język formalny/prawniczy jednym, krótkim
    wywołaniem modelu -- patrz prompt.build_query_expansion_messages().

    `lora_layers` (z find_lora_layers(), None gdy działamy bez adaptera):
    jeśli podane, LoRA jest na czas tego wywołania WYŁĄCZONA. Znalezisko z
    pilotażu `budowlane` (PROGRESS.md, Krok 21) -- domenowy adapter, mimo
    wprost sformułowanego zakazu w QUERY_EXPANSION_SYSTEM_PROMPT ("NIE
    cytuj przepisów"), i tak zaczynał przeformułowanie od "Zgodnie z art.
    X...", po czym pod tym poprawnym numerem artykułu podstawiał TREŚĆ
    zupełnie innego, pokrewnego przepisu -- efekt silnego, wyuczonego
    nawyku cytowania, którego żadna wersja samego promptu nie przełamała.
    Ten sam model BEZ adaptera poprawnie stosował się do instrukcji i
    zwracał czyste przeformułowanie -- stąd decyzja, żeby ekspansję
    zapytania zawsze wykonywać bazowym modelem, niezależnie od tego, czy
    do właściwej odpowiedzi używany jest adapter.

    `act_titles` (z RagIndex.act_titles(), patrz answer() niżej): nazwy
    ustaw bieżącej dziedziny, patrz prompt.build_query_expansion_messages()
    (Krok 28) po uzasadnienie. `domain` -- patrz DOMAIN_QUERY_EXPANSION_HINTS
    (Krok 41)."""
    messages = build_query_expansion_messages(question, act_titles=act_titles, domain=domain)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    set_lora_enabled(model, lora_layers, enabled=False)
    try:
        return generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False).strip()
    finally:
        set_lora_enabled(model, lora_layers, enabled=True)


def resolve_adapter_path(
    model_path: str | Path,
    domain: str,
    no_adapter: bool = False,
    explicit_adapter: str | None = None,
    has_extra_domains: bool = False,
) -> str | None:
    """Który adapter LoRA załadować (Krok 118). Jawny --adapter-path zawsze wygrywa (odpowiedzialność użytkownika);
    domyślny adapter dziedziny tylko dla modelu v3.0 (na nim trenowane), nigdy przy --no-adapter ani przy drugiej
    dziedzinie z routera (niska pewność: adapter jednej dziedziny mógłby zniekształcić odpowiedź z drugiej)."""
    if no_adapter:
        return None
    if explicit_adapter:
        return explicit_adapter
    if has_extra_domains or not Path(model_path).name.startswith(ADAPTER_BASE_MODEL_PREFIX):
        return None
    default = DEFAULT_ADAPTER_PATHS.get(domain)
    return str(default) if default else None


def answer(
    model,
    tokenizer,
    rag: RagIndex,
    file_index: SessionFileIndex,
    history: list[dict],
    question: str,
    top_k: int,
    verbose: bool,
    as_of: str | None = None,
    expand_query_enabled: bool = True,
    lora_layers: list[tuple[str, LoRALinear]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    expand_model=None,
    expand_tokenizer=None,
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
            if expand_model is not None:  # osobny model do ekspansji (bez LoRA)
                expanded = expand_query(expand_model, expand_tokenizer, question, act_titles=rag.act_titles(), domain=rag.domain)
            else:
                expanded = expand_query(
                    model, tokenizer, question, lora_layers=lora_layers, act_titles=rag.act_titles(), domain=rag.domain
                )
            if expanded and expanded.strip().lower() != question.strip().lower():
                extra_queries = [expanded]
        results = search_with_history(rag, history, question, top_k, as_of=as_of, extra_queries=extra_queries)
        file_results = file_index.search(question, top_k=3)
        current_turn = {
            "role": "user",
            "content": build_user_message(question, results, file_results, as_of=as_of),
        }
    messages = [{"role": "system", "content": build_system_prompt(rag.domain, rag.domains[1] if len(rag.domains) > 1 else None)}] + history + [current_turn]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=verbose)
    # W historii trzymamy samo pytanie (bez fragmentów RAG) -- fragmenty
    # doklejane są od nowa przy każdej turze, więc trzymanie ich też w
    # historii bardzo szybko przepełnia okno kontekstu (patrz PROGRESS.md,
    # krok 10).
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    return text


def main():
    parser = argparse.ArgumentParser(description="Asystent prawa polskiego, wielodziedzinowy (RAG + Bielik LoRA)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--model", type=str, default=str(ANSWER_MODEL_PATH), help="Ścieżka do modelu odpowiedzi w formacie MLX (Krok 118: domyślnie Bielik v3.1 8-bit)")
    parser.add_argument(
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN,
        help=f"Dziedzina prawa: wybiera indeks RAG (rag_index_<domena>.npy) i domyślny adapter LoRA "
        f"(domyślnie: {DEFAULT_DOMAIN}). Znane dziedziny z domyślnym adapterem: "
        f"{', '.join(sorted(DEFAULT_ADAPTER_PATHS))} -- inne działają, ale bez auto-doboru adaptera "
        "(podaj --adapter-path ręcznie albo użyj --no-adapter). `auto` (tylko z --prompt): dziedzinę wybiera "
        "router Basala (scripts/router.py); basal-serve musi działać PRZED startem, a przed ładowaniem "
        "Bielika najlepiej go zatrzymać -- jeden proces MLX naraz.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None, help="Nadpisuje domyślny adapter dobrany na podstawie --domain"
    )
    parser.add_argument("--no-adapter", action="store_true", help="Użyj bazowego modelu bez LoRA")
    parser.add_argument(
        "--expand-model", type=str, default=str(EXPAND_MODEL_PATH),
        help="Model MLX do ekspansji zapytań (Krok 118: domyślnie Bielik v3.0 4-bit); 'same' = ten sam co --model",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Limit tokenów odpowiedzi (Krok 118: domyślnie 1000; 500 ucinało odpowiedzi)")
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
        "przed wyszukiwaniem RAG (domyślnie włączone -- kosztuje jedno dodatkowe, krótkie "
        "wywołanie modelu na pytanie, ale naprawia realne przypadki, w których potoczne "
        "sformułowanie nie trafiało we właściwy artykuł ani w rankingu kosinusowym, ani po "
        "rerankingu).",
    )
    args = parser.parse_args()

    extra_domains = None
    if args.domain == "auto":
        if not args.prompt:
            parser.error("--domain auto działa tylko z --prompt (dziedzina jest wybierana raz, na podstawie pytania)")
        from router import auto_domain

        decision = auto_domain(args.prompt)
        args.domain = decision.primary
        extra_domains = [decision.secondary] if decision.secondary else None
        print(f"[INFO] Router: {decision.primary} (pewność {decision.confidence:.2f}"
              + (f", druga: {decision.secondary}" if decision.secondary else "") + ")")

    adapter_path = resolve_adapter_path(
        args.model, args.domain, no_adapter=args.no_adapter, explicit_adapter=args.adapter_path,
        has_extra_domains=bool(extra_domains),
    )
    if args.domain in DEFAULT_ADAPTER_PATHS and not adapter_path and not args.no_adapter and not extra_domains:
        print(f"[INFO] Domyślny adapter pominięty: pasuje tylko do {ADAPTER_BASE_MODEL_PREFIX}* (model: {Path(args.model).name}).")
    print(f"[INFO] Ładowanie modelu ({Path(args.model).name}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load(args.model, adapter_path=adapter_path)
    lora_layers = find_lora_layers(model) if adapter_path else None
    expand_model = expand_tokenizer = None
    if args.expand_model != "same" and Path(args.expand_model).resolve() != Path(args.model).resolve():
        print(f"[INFO] Ładowanie modelu do ekspansji zapytań ({Path(args.expand_model).name})...")
        expand_model, expand_tokenizer = load(args.expand_model)

    print(f"[INFO] Ładowanie indeksu RAG (dziedzina: {args.domain})...")
    rag = RagIndex(domain=args.domain, extra_domains=extra_domains)
    file_index = SessionFileIndex(model=rag.model)

    if args.file:
        n = file_index.add_pdf(args.file)
        print(f"[INFO] Wgrano plik {args.file} ({n} fragmentów)")

    if args.as_of:
        print(f"[INFO] Stan prawny na dzień: {args.as_of}")

    if args.prompt:
        answer(
            model,
            tokenizer,
            rag,
            file_index,
            [],
            args.prompt,
            args.top_k,
            verbose=True,
            as_of=args.as_of,
            expand_query_enabled=not args.no_expand_query,
            lora_layers=lora_layers,
            max_tokens=args.max_tokens,
            expand_model=expand_model,
            expand_tokenizer=expand_tokenizer,
        )
        return

    print(f"\nAsystent gotowy (dziedzina: {args.domain}). Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek, /plik <ścieżka> aby wgrać PDF, /data RRRR-MM-DD aby zapytać o stan prawny na dany dzień).\n")
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
            # pliku (file_index) -- to osobny "tryb" sesji, nie część
            # historii rozmowy; czyści go wyłącznie jawne /data.
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
        print("Asystent: ", end="", flush=True)
        text = answer(
            model,
            tokenizer,
            rag,
            file_index,
            history,
            question,
            args.top_k,
            verbose=False,
            as_of=as_of,
            expand_query_enabled=not args.no_expand_query,
            lora_layers=lora_layers,
            max_tokens=args.max_tokens,
            expand_model=expand_model,
            expand_tokenizer=expand_tokenizer,
        )
        print(text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    main()
