"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + model załadowany w
LM Studio (lokalny serwer zgodny z API OpenAI), zamiast lokalnego MLX.

Przydatne, gdy masz już w LM Studio załadowany model (np. Bielik w
formacie GGUF) i nie chcesz go dublować w MLX. Ponownie wykorzystuje
dokładnie ten sam prompt systemowy i sposób budowania kontekstu z RAG
co scripts/chat.py (wspólny moduł scripts/prompt.py) -- różni się
tylko backendem generacji. Ten skrypt NIE zależy od mlx_lm, więc
działa też na Linuksie/Windows (wszystko, co robi, to zapytania HTTP
do lokalnego serwera LM Studio + embedding modelu RAG przez
sentence-transformers, który jest wieloplatformowy).

Uwaga: LoRA wytrenowane w scripts/chat.py (adapters/*) NIE działa tutaj
-- to wagi w formacie MLX, a LM Studio ładuje osobny plik GGUF. Ten
skrypt daje sam RAG + Twój model z LM Studio, bez dotrenowanego stylu.

Wymaga uruchomionego serwera lokalnego w LM Studio: zakładka
"Developer" -> "Start Server" (domyślnie http://localhost:1234).

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
    python scripts/chat_lmstudio.py                      # tryb interaktywny
    python scripts/chat_lmstudio.py --prompt "pytanie..."
    python scripts/chat_lmstudio.py --model bielik-11b-v3.0-instruct  # jeśli LM Studio ma kilka modeli naraz
    python scripts/chat_lmstudio.py --url http://localhost:1234/v1
    python scripts/chat_lmstudio.py --file umowa.pdf     # z wgranym plikiem od startu
    python scripts/chat_lmstudio.py --as-of 2019-06-01    # stan prawny na dany dzień

Modele z trybem "thinking" (np. Qwen3.8) zużywają tokeny z tego samego
budżetu --max-tokens na wewnętrzne rozumowanie -- bez ograniczenia
wysiłku rozumowania mogą zwrócić pustą odpowiedź (patrz PROGRESS.md
Krok 32):
    python scripts/chat_lmstudio.py --model "qwen/qwen3.8-27b" --domain cywilne \\
        --max-tokens 3000 --reasoning-effort low --prompt "pytanie..."
"""

import argparse
import json
import os
import sys
from datetime import date

import requests

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

DEFAULT_URL = "http://localhost:1234/v1"


def _headers() -> dict:
    """Opcjonalny klucz API (np. vLLM z --api-key na RunPodzie, PROGRESS.md
    Krok 72) -- celowo osobna zmienna, nie OPENAI_API_KEY, żeby prawdziwy
    klucz OpenAI nigdy nie trafił na obcy serwer. LM Studio lokalnie: brak."""
    key = os.environ.get("LLM_SERVER_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def post_chat(base_url: str, payload: dict, timeout: int) -> dict:
    """POST /chat/completions ze strumieniowaniem (SSE). Bez strumienia proxy
    RunPoda (Cloudflare) zrywa zapytanie błędem 524, gdy przez ~100 s nie
    przyjdzie żaden bajt -- a tyle trwa samo rozumowanie Qwena przy dłuższej
    odpowiedzi (PROGRESS.md, Krok 72). Zwraca content, reasoning (osobno,
    jak `reasoning_content` w odpowiedzi bez strumienia), finish_reason, usage."""
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    content, reasoning, finish_reason, usage = [], [], None, {}
    with requests.post(
        f"{base_url}/chat/completions", json=body, headers=_headers(), timeout=timeout, stream=True
    ) as resp:
        resp.raise_for_status()
        # Linie dzielone na BAJTACH i dekodowane jako UTF-8: LM Studio wysyła
        # "text/event-stream" bez charset, więc requests przyjmuje ISO-8859-1,
        # a wtedy bajt 0x85 z "ą" (C4 85) to NEL, na którym splitlines() tnie
        # linię JSON w pół (Krok 73; vLLM podawał charset, więc tam działało).
        for raw in resp.iter_lines():
            line = raw.decode("utf-8")
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish_reason": finish_reason, "usage": usage}


def resolve_generation_defaults(model_id: str, max_tokens: int | None, reasoning_effort: str | None) -> tuple[int, str | None]:
    """Krok 110: Qwen (tryb thinking) z domyślnym max_tokens=500 i bez reasoning_effort zwraca PUSTE odpowiedzi
    (myślenie zjada cały budżet, Krok 32/107). Dla modeli Qwen domyślnie 3000 tokenów i reasoning_effort=low;
    jawnie podane argumenty zawsze wygrywają. Pozostałe modele (Bielik): 500 i brak reasoning_effort, jak dotąd."""
    is_qwen = "qwen" in model_id.lower()
    if max_tokens is None:
        max_tokens = 3000 if is_qwen else 500
    if reasoning_effort is None and is_qwen:
        reasoning_effort = "low"
    return max_tokens, reasoning_effort


def detect_model(base_url: str) -> str:
    try:
        resp = requests.get(f"{base_url}/models", headers=_headers(), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Nie mogę połączyć się z LM Studio pod {base_url}. "
            "Upewnij się, że lokalny serwer jest uruchomiony "
            "(zakładka 'Developer' -> 'Start Server')."
        ) from e
    ids = [m["id"] for m in resp.json().get("data", [])]
    if not ids:
        raise RuntimeError("LM Studio nie zgłasza żadnego załadowanego modelu.")
    if len(ids) > 1:
        print(f"[INFO] LM Studio ma kilka załadowanych modeli: {ids}. Użyję: {ids[0]!r} (podaj --model, żeby wybrać inny).")
    return ids[0]


EMPTY_EXPANSION_RETRY_MAX_TOKENS = 2000


def expand_query(
    base_url: str,
    model_id: str,
    question: str,
    act_titles: list[str] | None = None,
    domain: str | None = None,
    reasoning_effort: str | None = None,
    template_kwargs: dict | None = None,
) -> str:
    """Przeformułowuje pytanie na język formalny/prawniczy jednym, krótkim
    zapytaniem do LM Studio -- patrz prompt.build_query_expansion_messages()
    (`act_titles` z RagIndex.act_titles(), patrz Krok 28; `domain` -- patrz
    DOMAIN_QUERY_EXPANSION_HINTS, Krok 41).

    `reasoning_effort` (patrz PROGRESS.md Krok 32): dla modeli z trybem
    "thinking" (np. Qwen3.8) -- podany tu tak samo jak w `answer()`, bo
    nawet krótkie przeformułowanie potrafi wywołać długie wewnętrzne
    rozumowanie, jeśli w ogóle się nie ograniczy; `None` (domyślnie) nie
    dodaje tego pola do zapytania, więc modele bez trybu "thinking"
    (Bielik) mają zachowanie identyczne jak wcześniej.

    `max_tokens=800` (patrz PROGRESS.md Krok 52 -- podniesione z 80):
    `reasoning_effort` samo w sobie NIE wystarcza, żeby ograniczyć zużycie
    budżetu tokenów przez wewnętrzne rozumowanie -- zweryfikowane wprost,
    że przy `max_tokens=80` (nawet z `reasoning_effort="low"`) Qwen3.8
    czasem zużywa CAŁY budżet na niewidoczne rozumowanie i zwraca PUSTY
    string, co w `answer()` ciszej niż cicho wyłącza rozszerzanie zapytania
    (`if expanded and ...` traktuje pusty string jak brak rozszerzenia,
    bez żadnego ostrzeżenia) -- dokładnie dla najtrudniejszych przypadków
    (VAT art.86, KSH art.210), które rozszerzanie miało naprawiać. Samo
    podniesienie do 500 NIE wystarczyło (nadal ~50% pustych odpowiedzi na
    4 próby dla KSH art.210) -- dopiero 800 dało 0 pustych na 4 próby przy
    trzech różnych wartościach testowych (800/1000/1500 -- 800 wybrane
    jako najmniejsze z potwierdzoną niezawodnością, żeby niepotrzebnie nie
    wydłużać czasu odpowiedzi)."""
    messages = build_query_expansion_messages(question, act_titles=act_titles, domain=domain)
    payload = {"model": model_id, "messages": messages, "max_tokens": 800, "temperature": 0.2}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
        # vLLM przekazuje parametry szablonu czatu tylko przez chat_template_kwargs
        # (LM Studio czyta pole z najwyższego poziomu, nieznane pola ignoruje).
        payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
    if template_kwargs:
        # Krok 117: np. {"enable_thinking": false} dla Gemma 4 (LM Studio: pole "Enable Thinking")
        payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), **template_kwargs}
    expanded = post_chat(base_url, payload, timeout=300)["content"].strip()
    if not expanded and reasoning_effort is not None:
        # Krok 108: na baterii 129 Qwen (low, 800 tokenów) zwrócił 7 pustych ekspansji (5%), z czego
        # jedna kosztowała utratę przypadku (kc_415: ranga 11 zamiast 1). Jedno ponowienie z większym
        # budżetem tylko przy pustym wyniku -- nie zmienia pozostałych przypadków.
        payload["max_tokens"] = EMPTY_EXPANSION_RETRY_MAX_TOKENS
        expanded = post_chat(base_url, payload, timeout=600)["content"].strip()
    return expanded


def answer(
    base_url: str,
    model_id: str,
    rag: RagIndex,
    file_index: SessionFileIndex,
    history: list[dict],
    question: str,
    top_k: int,
    as_of: str | None = None,
    expand_query_enabled: bool = True,
    max_tokens: int = 500,
    reasoning_effort: str | None = None,
    template_kwargs: dict | None = None,
) -> str:
    """`max_tokens`/`reasoning_effort` (patrz PROGRESS.md Krok 32): modele z
    trybem "thinking" (np. Qwen3.8) zużywają tokeny z TEGO SAMEGO budżetu
    na wewnętrzne rozumowanie (zwracane osobno w `reasoning_content`,
    niewidoczne tutaj) -- przy domyślnym `max_tokens=500` i braku kontroli
    wysiłku rozumowania model potrafi zużyć CAŁY budżet na rozumowanie i
    zwrócić pustą odpowiedź. Zweryfikowane: `reasoning_effort="low"` +
    `max_tokens=3000` daje pełne, poprawne odpowiedzi w ~150-250s zamiast
    700+s lub braku odpowiedzi w ogóle. Dla modeli bez trybu "thinking"
    (Bielik) `reasoning_effort=None` (domyślnie) nie dodaje tego pola do
    zapytania -- zachowanie identyczne jak przed Krokiem 32."""
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        extra_queries = None
        if expand_query_enabled:
            expanded = expand_query(
                base_url, model_id, question, act_titles=rag.act_titles(), domain=rag.domain,
                reasoning_effort=reasoning_effort, template_kwargs=template_kwargs,
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
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
        # vLLM przekazuje parametry szablonu czatu tylko przez chat_template_kwargs
        # (LM Studio czyta pole z najwyższego poziomu, nieznane pola ignoruje).
        payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
    if template_kwargs:
        # Krok 117: np. {"enable_thinking": false} dla Gemma 4 (LM Studio: pole "Enable Thinking")
        payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), **template_kwargs}
    text = post_chat(base_url, payload, timeout=1800)["content"]
    # W historii trzymamy samo pytanie (bez fragmentów RAG) -- fragmenty
    # doklejane są od nowa przy każdej turze, więc trzymanie ich też w
    # historii bardzo szybko przepełnia okno kontekstu (patrz PROGRESS.md,
    # krok 10 -- realny błąd znaleziony w testach: 2 tury = przekroczenie
    # limitu kontekstu LM Studio).
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    return text


def main():
    parser = argparse.ArgumentParser(description="Asystent prawa polskiego, wielodziedzinowy (RAG + model w LM Studio)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Bazowy URL lokalnego serwera LM Studio")
    parser.add_argument("--model", type=str, default=None, help="ID modelu w LM Studio (domyślnie: auto-wykrycie)")
    parser.add_argument(
        "--domain",
        type=str,
        default="kadry_i_place",
        help="Dziedzina prawa: wybiera indeks RAG (rag_index_<domena>.npy). LM Studio nie wspiera "
        "adapterów LoRA (format GGUF), więc tu wpływa tylko na dobór indeksu, nie na model. `auto` "
        "(tylko z --prompt): dziedzinę wybiera router Basala (scripts/router.py, wymaga basal-serve).",
    )
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limit tokenów odpowiedzi (domyślnie 500; 3000 dla modeli Qwen, Krok 110). Modele z trybem \"thinking\" "
        "(np. Qwen3.8) zużywają z TEGO SAMEGO budżetu tokeny na wewnętrzne rozumowanie -- "
        "podnieś to (np. do 3000) razem z --reasoning-effort, patrz PROGRESS.md Krok 32.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Dla modeli z trybem \"thinking\" (np. Qwen3.8) -- ogranicza wewnętrzne "
        "rozumowanie przed odpowiedzią. Bez tego (domyślnie) model może zużyć CAŁY "
        "budżet --max-tokens na rozumowanie i zwrócić pustą odpowiedź (zweryfikowane "
        "empirycznie, patrz PROGRESS.md Krok 32) -- \"low\" dawało pełne, poprawne "
        "odpowiedzi w ~150-250s. Nie ma efektu na modele bez trybu \"thinking\" (Bielik).",
    )
    parser.add_argument(
        "--template-kwargs", default=None,
        help='JSON z chat_template_kwargs (Krok 117), np. \'{"enable_thinking": false}\' dla Gemma 4 -- wyłącza myślenie (domyślnie: brak)',
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

    model_id = args.model or detect_model(args.url)
    args.max_tokens, args.reasoning_effort = resolve_generation_defaults(model_id, args.max_tokens, args.reasoning_effort)
    print(f"[INFO] LM Studio: {args.url}, model={model_id}")

    print(f"[INFO] Ładowanie indeksu RAG (dziedzina: {args.domain})...")
    rag = RagIndex(domain=args.domain, extra_domains=extra_domains)
    file_index = SessionFileIndex(model=rag.model)

    if args.file:
        n = file_index.add_pdf(args.file)
        print(f"[INFO] Wgrano plik {args.file} ({n} fragmentów)")

    if args.as_of:
        print(f"[INFO] Stan prawny na dzień: {args.as_of}")

    if args.prompt:
        print(answer(
            args.url, model_id, rag, file_index, [], args.prompt, args.top_k,
            as_of=args.as_of, expand_query_enabled=not args.no_expand_query,
            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
            template_kwargs=json.loads(args.template_kwargs) if args.template_kwargs else None,
        ))
        return

    print(f"\nAsystent (LM Studio) gotowy (dziedzina: {args.domain}). Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek, /plik <ścieżka> aby wgrać PDF, /data RRRR-MM-DD aby zapytać o stan prawny na dany dzień).\n")
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
        text = answer(
            args.url, model_id, rag, file_index, history, question, args.top_k,
            as_of=as_of, expand_query_enabled=not args.no_expand_query,
            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
            template_kwargs=json.loads(args.template_kwargs) if args.template_kwargs else None,
        )
        print("Asystent:", text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)
