"""
Buduje indeks RAG (Retrieval-Augmented Generation) na fragmentach ustaw.

Co to jest RAG, w skrócie:
  Model językowy (Bielik) sam z siebie NIE zna treści konkretnych,
  polskich ustaw na pamięć w sposób wiarygodny -- a nawet gdyby był
  fine-tunowany na ich tekście, ustawy się zmieniają, więc taka wiedza
  szybko by się zdezaktualizowała. RAG obchodzi ten problem inaczej:
  zamiast "wkuwać" przepisy w wagi modelu, w momencie zadawania pytania
  WYSZUKUJEMY najbardziej pasujące fragmenty ustaw i wklejamy je do
  promptu jako kontekst. Model tylko czyta ten kontekst i formułuje
  odpowiedź na jego podstawie -- tak jak człowiek, który dostaje
  podkreślony fragment ustawy do przeczytania, zamiast cytować z pamięci.

Jak działa wyszukiwanie (semantyczne, nie po słowach kluczowych):
  1. Każdy fragment ustawy zamieniamy na wektor liczb (embedding) za
     pomocą modelu embeddingowego -- wektor reprezentuje "znaczenie"
     tekstu w wielowymiarowej przestrzeni. Podobne znaczeniowo teksty
     mają wektory blisko siebie.
  2. Pytanie użytkownika zamieniamy na wektor tym samym modelem.
  3. Liczymy podobieństwo kosinusowe pytania do wszystkich fragmentów
     i bierzemy np. 5 najbardziej podobnych.
  To działa nawet, gdy pytanie nie zawiera dokładnie tych samych słów
  co ustawa (np. "ile urlopu mi się należy" trafi w art. 154 KP, mimo
  że artykuł nie zawiera słowa "należy").

Ten skrypt tylko BUDUJE indeks (liczy i zapisuje embeddingi).
Samo wyszukiwanie jest w scripts/rag_search.py.

Domyślnie buduje JEDEN indeks ze wszystkich dziedzin naraz. Flaga
--domain ogranicza do jednej dziedziny (pole "domains" w artykułach,
patrz data/acts_config.json) i zapisuje do osobnych plików
rag_index_<domena>.npy / _meta.json -- celowo osobno od domyślnego
rag_index.npy, żeby dało się mierzyć jakość wyszukiwania per dziedzina
zanim ktoś zdecyduje się łączyć indeksy.

Uruchomienie:
    python scripts/build_rag_index.py                    # wszystko razem -> rag_index.npy
    python scripts/build_rag_index.py --domain podatkowe  # tylko ta dziedzina -> rag_index_podatkowe.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from embedding_cache import EmbeddingCache, encode_with_cache

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTICLES_PATH = PROCESSED_DIR / "all_articles.json"


def index_paths(domain: str | None) -> tuple[Path, Path]:
    suffix = f"_{domain}" if domain else ""
    return (
        PROCESSED_DIR / f"rag_index{suffix}.npy",
        PROCESSED_DIR / f"rag_index{suffix}_meta.json",
    )

EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"

# Trwały cache embeddingów fragmentów (patrz scripts/embedding_cache.py,
# PROGRESS.md Krok 63): przebudowa indeksu liczy tylko fragmenty nowe albo o
# zmienionej treści. Klucz obejmuje model i tryb (dokument bez prefiksu,
# znormalizowany) -- zmiana któregokolwiek sama omija stare wpisy.
EMBEDDING_CACHE_PATH = PROCESSED_DIR / "embedding_cache.sqlite"
EMBEDDING_CACHE_KEY = f"{EMBEDDING_MODEL}|dokument-bez-prefiksu|znormalizowany"

# Niektóre artykuły (np. w ustawie o systemie ubezpieczeń społecznych)
# mają po kilka tysięcy znaków -- za dużo, żeby sensownie zmieścić się
# jako jeden precyzyjny wektor i za dużo, żeby wstrzykiwać w całości do
# promptu. Długie artykuły tniemy dodatkowo na nakładające się okna.
MAX_CHUNK_CHARS = 900
OVERLAP_CHARS = 150


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split(" ")
    chunks = []
    current: list[str] = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= max_chars:
            chunks.append(" ".join(current))
            # cofamy się o ~overlap znaków, żeby zachować kontekst na styku okien
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(articles: list[dict]) -> list[dict]:
    chunks = []
    for art in articles:
        pieces = split_long_text(art["text"], MAX_CHUNK_CHARS, OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            chunk_id = art["id"] if len(pieces) == 1 else f"{art['id']}_part{i + 1}"
            chunk = {
                "chunk_id": chunk_id,
                "act_short": art["act_short"],
                "act_title": art["act_title"],
                "eli": art["eli"],
                "article": art["article"],
                "source_url": art["source_url"],
                "chunk_text": piece,
                "full_article_text": art["text"],
            }
            # Wymiar czasowy ("as-of") -- opcjonalny, patrz PROGRESS.md Krok 18
            # i download_acts_history.py. Dopisywany TYLKO gdy artykuł źródłowy
            # faktycznie niesie choć jedno z tych pól (wpis dograny/załatany
            # przez load_articles_with_history() z --include-history) --
            # CELOWO nie `.get(..., None)` bezwarunkowo dla każdego artykułu,
            # bo to zmieniłoby format rag_index_meta.json (trzy dodatkowe
            # klucze z wartością null) nawet przy zwykłym uruchomieniu bez
            # --include-history, łamiąc obietnicę "bajtowo identyczny wynik
            # domyślnie" z planu implementacji. Zwykłe wpisy z all_articles.json
            # (bez historii) w ogóle nie mają tych kluczy w słowniku, więc `in`
            # poprawnie je pomija.
            if "valid_from" in art or "valid_to" in art or "announcement_eli" in art:
                chunk["valid_from"] = art.get("valid_from")
                chunk["valid_to"] = art.get("valid_to")
                chunk["announcement_eli"] = art.get("announcement_eli")
            chunk["domains"] = art.get("domains", [])
            chunks.append(chunk)
    return chunks


def _entry_into_force(short: str) -> str | None:
    """`entryIntoForce` (data wejścia ustawy w życie) z `data/raw/{short}_meta.json`
    -- pobrane i zapisane przez zwykły `download_acts.py`, więc dostępne
    niezależnie od tego, czy dla tej ustawy istnieje `_history.json`. Brak
    pliku (np. świeży klon repo bez lokalnie pobranej bazy) -> `None`,
    bez wywalania się -- wołający po prostu nie dostanie dolnej granicy."""
    path = RAW_DIR / f"{short}_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("entryIntoForce")


def _repeal_date(short: str) -> str | None:
    """`repealDate` (data uchylenia ustawy) z `data/raw/{short}_meta.json`.
    Ustawa uchylona w całości (np. Prawo telekomunikacyjne z 2004 r.,
    uchylone 2024-11-10 przez Prawo komunikacji elektronicznej) ma w ELI
    `expirationDate=None` -- jedyną górną granicą jest `repealDate`. Bez
    niej jej ostatni tekst "U" wyglądałby na obowiązujący dziś (Krok 67).
    Brak pliku albo pola -> `None` (ustawa nieuchylona)."""
    path = RAW_DIR / f"{short}_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("repealDate")


def load_articles_with_history(include_history: bool) -> list[dict]:
    """Wczytuje all_articles.json (bieżący stan, jak zawsze) i -- jeśli
    `include_history` -- dogrywa artykuły z każdego istniejącego
    `data/processed/{short}_history.json` (produkowanego opt-in przez
    `download_acts_history.py`) ORAZ z każdego `data/processed/*_series.json`
    (samodzielne, w pełni zwersjonowane "akty" spoza `ACTS`, np. coroczne
    rozporządzenia o wysokości minimalnego wynagrodzenia --
    `download_wage_regulations.py`, Krok 21 -- każda wersja ma już ustawione
    `valid_from`/`valid_to` przy zapisie, więc dogrywane wprost, bez
    patchowania jak przy `_history.json`), oraz nadaje BIEŻĄCYM artykułom
    KAŻDEJ ustawy z `all_articles.json` realną dolną granicę `valid_from`:
    - jeśli ustawa ma plik `_history.json` -- `current_valid_from` z tego
      pliku (granica wyliczona z łańcucha obwieszczeń, patrz Krok 18);
      inaczej "bieżąca" wersja zachowywałaby się jak "-nieskończoność" i
      mogłaby przegrać w rankingu z wersją historyczną przy zwykłym
      zapytaniu bez podanej daty (patrz `rag_search._version_covers`);
    - NIEZALEŻNIE od powyższego, spodem podbita (max) datą
      `entryIntoForce` z `data/raw/{short}_meta.json` -- bez tego ustawa
      bez pliku `_history.json`, ale wciąż młodsza niż zapytanie `as_of`
      (np. ustawa o rynku pracy, w mocy dopiero od 2025-06-01), zostawałaby
      z `valid_from=None` i błędnie "obowiązywałaby" dla dowolnie starej
      daty -- realny błąd znaleziony przy ręcznej weryfikacji Fazy 2
      (`as_of=2010-01-01` błędnie zwracał jej artykuły). `entryIntoForce`
      brakujące w metadanych (rzadkie) -- bez podbicia, `valid_from`
      zostaje czym było."""
    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
    if not include_history:
        return articles

    history_files = sorted(PROCESSED_DIR.glob("*_history.json"))
    if not history_files:
        print("  [UWAGA] --include-history podane, ale brak plików *_history.json -- "
              "uruchom najpierw scripts/download_acts_history.py")

    current_valid_from_by_short = {
        path.name.removesuffix("_history.json"): json.loads(path.read_text(encoding="utf-8")).get(
            "current_valid_from"
        )
        for path in history_files
    }

    acts_present = {art["act_short"] for art in articles}
    for short in sorted(acts_present):
        history_bound = current_valid_from_by_short.get(short)
        entry_bound = _entry_into_force(short)
        # data/raw/{short}_meta.json to daty ISO ("RRRR-MM-DD"), więc zwykłe
        # porównanie leksykograficzne stringów = porównanie chronologiczne
        candidates = [d for d in (history_bound, entry_bound) if d]
        valid_from = max(candidates) if candidates else None
        # ustawa uchylona: bieżący tekst obowiązywał tylko do repealDate
        # (exclusive, jak expirationDate -- patrz rag_search._version_covers)
        valid_to = _repeal_date(short)
        n_patched = 0
        for art in articles:
            if art["act_short"] == short:
                art["valid_from"] = valid_from
                if valid_to:
                    art["valid_to"] = valid_to
                n_patched += 1
        print(f"  [{short}] valid_from bieżących artykułów -> {valid_from!r} ({n_patched} artykułów)")
        if valid_to:
            print(f"  [{short}] ustawa uchylona -- valid_to bieżących artykułów -> {valid_to!r}")

    for path in history_files:
        short = path.name.removesuffix("_history.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        history_articles = data.get("articles", [])
        articles.extend(history_articles)
        print(f"  [{short}] dograno {len(history_articles)} artykułów historycznych")

    # "Serie" -- akty samodzielne, całkowicie odrębne od czegokolwiek w
    # all_articles.json (np. coroczne rozporządzenia o wysokości minimalnego
    # wynagrodzenia, download_wage_regulations.py, Krok 21) -- KAŻDA wersja,
    # łącznie z bieżącą, ma już ustawione valid_from/valid_to przy zapisie,
    # więc -- w przeciwieństwie do *_history.json -- nie ma tu nic do
    # patchowania, dogrywamy wprost.
    for path in sorted(PROCESSED_DIR.glob("*_series.json")):
        series_articles = json.loads(path.read_text(encoding="utf-8"))
        articles.extend(series_articles)
        print(f"  [{path.stem}] dograno {len(series_articles)} artykułów z serii")

    return articles


def seed_cache_from_index(domain: str | None) -> None:
    """Zasila cache wektorami już zbudowanego indeksu -- bezpieczne, bo indeks
    był liczony tym samym modelem i trybem (EMBEDDING_CACHE_KEY), a klucz to
    hash dokładnie tego tekstu fragmentu, z którego policzono wektor."""
    from embedding_cache import text_hash

    emb_path, meta_path = index_paths(domain)
    embeddings = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(embeddings):
        raise SystemExit(f"Niespójny indeks {emb_path.name}: {len(meta)} metadanych vs {len(embeddings)} wektorów")
    cache = EmbeddingCache(EMBEDDING_CACHE_PATH, EMBEDDING_CACHE_KEY)
    try:
        cache.put_many([(text_hash(m["chunk_text"]), v) for m, v in zip(meta, embeddings)])
    finally:
        cache.close()
    print(f"Cache zasilony {len(meta)} wektorami z {emb_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Dograj też historyczne wersje tekstu (data/processed/*_history.json, "
        "produkowane przez download_acts_history.py) -- domyślnie wyłączone, "
        "bez zmiany w formacie/treści wynikowego indeksu.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Zbuduj indeks tylko z jednej dziedziny (pole 'domains' artykułu) -- "
        "zapisuje do rag_index_<domena>.npy/_meta.json zamiast domyślnego "
        "rag_index.npy/_meta.json.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Licz wszystkie embeddingi od nowa, bez odczytu/zapisu cache'u (np. do weryfikacji, "
        "że wynik z cache'u jest identyczny z pełną przebudową).",
    )
    parser.add_argument(
        "--seed-cache-from-index",
        action="store_true",
        help="Nic nie przelicza -- wpisuje do cache'u wektory z ISTNIEJĄCEGO indeksu (rag_index[_<domena>].npy "
        "+ _meta.json) dla --domain, żeby kolejna przebudowa liczyła tylko zmienione fragmenty.",
    )
    args = parser.parse_args()

    if args.seed_cache_from_index:
        seed_cache_from_index(args.domain)
        return

    print(f"Wczytuję fragmenty z {ARTICLES_PATH} ...")
    articles = load_articles_with_history(args.include_history)
    print(f"  {len(articles)} artykułów")

    if args.domain:
        articles = [a for a in articles if args.domain in a.get("domains", [])]
        print(f"  po filtrze --domain {args.domain}: {len(articles)} artykułów")

    chunks = build_chunks(articles)
    print(f"Po podziale długich artykułów: {len(chunks)} fragmentów do embedowania")

    print(f"Ładuję model embeddingowy: {EMBEDDING_MODEL} (pierwsze uruchomienie pobierze go z HF)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["chunk_text"] for c in chunks]
    print("Liczę embeddingi (bez prefiksu -- to są 'dokumenty', nie zapytania)...")
    cache = None if args.no_cache else EmbeddingCache(EMBEDDING_CACHE_PATH, EMBEDDING_CACHE_KEY)
    try:
        from rag_search import release_accelerator_cache

        embeddings = encode_with_cache(model, texts, cache, release_cache_fn=release_accelerator_cache)
    finally:
        if cache is not None:
            cache.close()

    index_emb_path, index_meta_path = index_paths(args.domain)
    np.save(index_emb_path, embeddings.astype(np.float32))
    index_meta_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nZapisano:")
    print(f"  {index_emb_path}  (macierz embeddingów {embeddings.shape})")
    print(f"  {index_meta_path}  (metadane fragmentów)")


if __name__ == "__main__":
    main()
