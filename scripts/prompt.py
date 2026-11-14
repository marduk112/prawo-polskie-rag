"""
Wspólny prompt systemowy i budowanie kontekstu RAG, używane przez
wszystkie warianty czatu (chat.py / MLX, chat_lmstudio.py / LM Studio,
chat_cuda.py / transformers+peft). Celowo bez ciężkich zależności
(mlx, torch), żeby dało się zaimportować na każdej platformie.
"""

import re

from article_excerpt import article_excerpt

# Krótki opis dziedziny wstawiany do promptu systemowego (patrz
# build_system_prompt() niżej) -- klucze muszą się zgadzać z polem
# "domains" w data/acts_config.json. Dopisz wpis tutaj przy każdej nowej
# dziedzinie; brakujący klucz spada na ogólny opis "polskiego prawa"
# (działa, ale bez precyzyjnego dopasowania -- patrz DEFAULT_DOMAIN_DESCRIPTION).
DOMAIN_DESCRIPTIONS = {
    "kadry_i_place": "polskiego prawa pracy i ubezpieczeń społecznych",
    "podatkowe": "polskiego prawa podatkowego",
    "administracyjne": "polskiego postępowania administracyjnego",
    "cywilne": "polskiego prawa cywilnego",
    "rodzinne": "polskiego prawa rodzinnego i opiekuńczego",
    "karne": "polskiego prawa karnego",
    "gospodarcze": "polskiego prawa spółek handlowych",
    "budowlane": "polskiego prawa budowlanego",
    "autorskie": "polskiego prawa autorskiego",
    "wlasnosc_przemyslowa": "polskiego prawa własności przemysłowej",
    "ochrona_danych": "polskiej ustawy o ochronie danych osobowych",
    "zamowienia_publiczne": "polskiego prawa zamówień publicznych",
    "ochrona_srodowiska": "polskiego prawa ochrony środowiska",
    "ruch_drogowy": "polskiego prawa o ruchu drogowym",
    "upadlosciowe": "polskiego prawa upadłościowego i restrukturyzacyjnego",
    "lotnicze": "polskiego prawa lotniczego",
    "geologia_i_gornictwo": "polskiego prawa geologicznego i górniczego",
    "wodne": "polskiego prawa wodnego",
    "mieszkaniowe": "polskiego prawa mieszkaniowego",
    "energetyczne": "polskiego prawa energetycznego",
    "telekomunikacyjne": "polskiego prawa komunikacji elektronicznej (telekomunikacyjnego)",
    "konsumenckie": "polskiego prawa konsumenckiego (ustawy o prawach konsumenta)",
    "kredyt_konsumencki": "polskiego prawa kredytu konsumenckiego (ustawy o kredycie konsumenckim)",
    "wykroczenia": "polskiego prawa wykroczeń (Kodeksu wykroczeń)",
    "postepowanie_cywilne": "polskiego postępowania cywilnego (Kodeksu postępowania cywilnego)",
    "postepowanie_karne": "polskiego postępowania karnego (Kodeksu postępowania karnego)",
    "cit": "polskiego podatku dochodowego od osób prawnych (ustawa o CIT)",
    "podatki_lokalne": "polskich podatków i opłat lokalnych (ustawa o podatkach i opłatach lokalnych)",
    "ubezpieczenia_obowiazkowe": "obowiązkowych ubezpieczeń (OC posiadaczy pojazdów, OC rolników, UFG, PBUK; ustawa o ubezpieczeniach obowiązkowych)",
    "pcc": "polskiego podatku od czynności cywilnoprawnych (ustawa o podatku od czynności cywilnoprawnych)",
    "prawa_pacjenta": "polskiego prawa pacjenta (ustawy o prawach pacjenta i Rzeczniku Praw Pacjenta)",
    "prawo_bankowe": "polskiego prawa bankowego (rachunki bankowe, tajemnica bankowa, kredyty bankowe; ustawa Prawo bankowe)",
    "cudzoziemcy": "polskiego prawa o cudzoziemcach (wjazd, wizy, zezwolenia na pobyt, karty pobytu, zobowiązanie do powrotu; ustawa o cudzoziemcach)",
}
DEFAULT_DOMAIN_DESCRIPTION = "polskiego prawa"

SYSTEM_PROMPT_TEMPLATE = """Jesteś asystentem odpowiadającym po polsku na pytania \
dotyczące {domain_description}.

Poniżej, przed każdym pytaniem użytkownika, otrzymasz fragmenty aktów \
prawnych wyszukane jako potencjalnie pomocny kontekst, a czasem też \
fragmenty pliku wgranego przez użytkownika (np. jego umowy, wniosku). \
Zasady:
1. Odpowiadaj WYŁĄCZNIE na podstawie dostarczonych fragmentów -- nie \
korzystaj z własnej wiedzy o konkretnych liczbach, kwotach czy terminach.
2. Jeśli dostarczone fragmenty nie zawierają odpowiedzi na pytanie, wprost \
napisz, że nie znalazłeś tego w dostępnych materiałach, i nie zgaduj.
3. Cytuj numer artykułu i nazwę aktu, na którym opierasz odpowiedź. \
Fragmenty z wgranego pliku użytkownika wyraźnie oznaczaj jako treść \
tego pliku, a NIE jako obowiązujące prawo -- to różne źródła.
4. Nie jesteś substytutem porady prawnej ani innej porady specjalistycznej \
(podatkowej, kadrowej itp.) -- w sprawach spornych zasugeruj konsultację \
ze specjalistą.
5. Jeśli zapytanie dotyczy konkretnej daty w przeszłości (zobaczysz to w \
notatce "Data, na którą ma obowiązywać odpowiedź" i/lub w adnotacjach \
"stan prawny" przy fragmentach), odpowiadaj TĄ wersją przepisu i wyraźnie \
zaznacz w odpowiedzi, że dotyczy ona tej konkretnej daty. Jeśli żaden \
dostarczony fragment nie obejmuje tej daty, powiedz to wprost -- nie \
zgaduj i nie podawaj w zamian stanu bieżącego.
6. Odróżniaj to, co przepis DOSŁOWNIE stwierdza, od dalszego wniosku \
prawnego, który z niego wyciągasz (interpretacja, analogia, łączenie \
kilku przepisów). Jeśli odpowiedź wymaga takiego wniosku wykraczającego \
poza literalne brzmienie zacytowanego przepisu, zaznacz to wprost (np. \
"przepis wprost mówi X; z tego wynikałoby Y, ale to już wniosek \
interpretacyjny, nie literalny zapis") -- nie przedstawiaj interpretacji \
z tą samą pewnością co dosłowny cytat."""


def build_system_prompt(domain: str | None = None, extra_domain: str | None = None) -> str:
    """Zwraca prompt systemowy z opisem dziedziny wstawionym wg `domain`
    (patrz DOMAIN_DESCRIPTIONS) -- reszta instrukcji jest identyczna dla
    wszystkich dziedzin. `domain=None` albo nieznany klucz -> ogólny opis
    "polskiego prawa" (nadal działa, tylko mniej precyzyjnie).
    `extra_domain` (router przy niskiej pewności, patrz RagIndex): opis obu dziedzin
    połączony przez "oraz"."""
    description = DOMAIN_DESCRIPTIONS.get(domain, DEFAULT_DOMAIN_DESCRIPTION)
    if extra_domain and extra_domain != domain:
        description += " oraz " + DOMAIN_DESCRIPTIONS.get(extra_domain, DEFAULT_DOMAIN_DESCRIPTION)
    return SYSTEM_PROMPT_TEMPLATE.format(domain_description=description)


# Kompatybilność wsteczna / domyślny wariant (kadry_i_place) dla kodu,
# który nie zna jeszcze koncepcji dziedzin.
SYSTEM_PROMPT = build_system_prompt("kadry_i_place")


QUERY_EXPANSION_SYSTEM_PROMPT = """Twoim JEDYNYM zadaniem jest przeformułowanie pytania użytkownika \
na 1-2 zdania w formalnym, prawniczym języku, jakim posługują się polskie ustawy. \
Nie poprzestawaj na ogólnym, "ładnie brzmiącym" prawniczo sformułowaniu -- użyj \
możliwie precyzyjnej, technicznej nazwy KONKRETNEGO mechanizmu prawnego, którego \
dotyczy pytanie, tak jak nazywa go sama ustawa, a nie tylko ogólnego opisu sytuacji \
(np. zamiast "ile mam czasu na coś" napisz "termin do dokonania czynności"). \
Zwróć szczególną uwagę na POTOCZNE SŁOWNICTWO EKONOMICZNE LUB BIZNESOWE \
-- ustawy rzadko używają takich słów wprost, tylko ich ścisłych prawnych \
odpowiedników (np. zamiast "zwiększyć zadłużenie firmy" napisz "zaciągnięcie \
zobowiązania do świadczenia"; zamiast "stracić pieniądze na inwestycji" \
napisz "poniesienie straty"). Nie poprzestawaj na bardziej formalnie \
brzmiącym słowie -- znajdź KONKRETNY termin, którym posługuje się ustawa. \
Zachowaj DOKŁADNY PRZEDMIOT pytania -- jeśli użytkownik pyta o OBOWIĄZEK zrobienia \
czegoś (czy musi, czy trzeba, kto ma obowiązek) albo o to, KOMU coś domyślnie \
przysługuje, przeformułowanie ma dotyczyć TEGO SAMEGO obowiązku/uprawnienia, a nie \
przesuwać się w stronę pokrewnego, ale odrębnego zagadnienia (np. kwalifikacji, \
uprawnień zawodowych czy warunków dotyczących wspomnianej w pytaniu roli lub \
czynności) tylko dlatego, że to zagadnienie pojawia się w tle pytania. \
NIE odpowiadaj na pytanie, NIE cytuj przepisów (żadnych numerów artykułów ani \
ustępów, nawet jeśli jesteś ich pewien), NIE zaczynaj od "Zgodnie z art. ...", \
NIE dodawaj żadnego wstępu ani wyjaśnienia -- zwróć WYŁĄCZNIE samo przeformułowane \
zapytanie, jako pytanie, a nie jako fragment odpowiedzi."""


DOMAIN_QUERY_EXPANSION_HINTS: dict[str, str] = {
    "cywilne": (
        'zamiast "konsument kupił wadliwy towar, chce wymiany lub zwrotu pieniędzy" napisz '
        '"brak zgodności towaru z umową -- naprawa, wymiana, obniżenie ceny albo odstąpienie od '
        'umowy"; zamiast "sprzedawca twierdzi, że to wina kupującego" napisz "domniemanie '
        'istnienia braku zgodności towaru z umową ujawnionego w ciągu dwóch lat od dostarczenia"'
    ),
    "gospodarcze": (
        'zamiast "czy członek zarządu może głosować nad umową ze spółką" '
        'napisz "kto reprezentuje spółkę w umowie z członkiem zarządu"; '
        'zamiast "jak wystąpić ze spółki jawnej, gdy wspólnicy się nie zgadzają" '
        'napisz "wypowiedzenie przez wspólnika umowy spółki jawnej zawartej na czas nieoznaczony"'
    ),
    "wodne": (
        'zamiast "czyja jest rzeka/potok przepływający przez moją działkę" '
        'napisz "czy śródlądowe wody płynące stanowią własność Skarbu Państwa"; '
        'zamiast "czy mogę nabrać wody z jeziora do podlewania" napisz "czy powszechne '
        'korzystanie z wód obejmuje pobór wody bez specjalnych urządzeń technicznych '
        'na potrzeby gospodarstwa domowego"'
    ),
    "telekomunikacyjne": (
        'zamiast "pliki cookies/ciasteczka" napisz "przechowywanie informacji lub '
        'uzyskiwanie dostępu do informacji w telekomunikacyjnym urządzeniu końcowym"; '
        'zamiast "operator podniósł cenę w trakcie umowy" napisz "jednostronna zmiana '
        'warunków umowy zawartej na czas określony i wypowiedzenie umowy bez '
        'odszkodowania"'
    ),
    "konsumenckie": (
        'zamiast "zwrot towaru kupionego w internecie" napisz "odstąpienie od umowy '
        'zawartej na odległość"; zamiast "od kiedy liczę czas na zwrot" napisz "bieg '
        'terminu do odstąpienia od umowy rozpoczyna się"; zamiast "sklep nie powiedział '
        'mi, że mogę oddać" napisz "konsument nie został poinformowany o prawie '
        'odstąpienia od umowy"; zamiast "chcę zwrotu pieniędzy za wadliwy towar" '
        'napisz "obniżenie ceny albo odstąpienie od umowy z powodu braku zgodności '
        'towaru z umową"'
    ),
    "kredyt_konsumencki": (
        'zamiast "umowa pożyczki była wadliwa, czy mogę oddać tylko to, co pożyczyłem" '
        'napisz "zwrot kredytu bez odsetek i innych kosztów kredytu po naruszeniu przez '
        'kredytodawcę wymogów dotyczących treści umowy o kredyt konsumencki (sankcja '
        'kredytu darmowego)"; zamiast "zwróciłem rzecz kupioną na raty, co z ratami" '
        'napisz "odstąpienie od umowy nabycia towaru jest skuteczne także wobec umowy o '
        'kredyt wiązany"'
    ),
}


# Krok 87: podpowiedzi włączane SŁOWEM KLUCZOWYM w pytaniu (domena -> [(regex, podpowiedź)]).
# Dopisanie kolejnej pary do podpowiedzi całej domeny `cywilne` destabilizowało przypadki
# konsumenckie (3 próby, 3 różne regresje -- model reaguje na długość i układ podpowiedzi).
# Para włączana tylko, gdy pytanie pasuje do wzorca, sprawia, że pytania BEZ tego słowa
# dostają dokładnie ten sam prompt co wcześniej, więc regresja jest niemożliwa z konstrukcji.
DOMAIN_QUERY_EXPANSION_KEYWORD_HINTS: dict[str, list[tuple[str, str]]] = {
    "cywilne": [
        (
            r"\bprzedaw",  # \b: "sprzedawca" zawiera "przedaw" (Krok 87)
            'zamiast "czy moje roszczenie już się przedawniło" napisz "po jakim czasie roszczenie '
            'majątkowe ulega przedawnieniu i kiedy kończy się termin przedawnienia"',
        ),
    ],
    "podatkowe": [
        (
            # Krok 93: zwolnienie z art. 21 ust. 1 pkt 131 PIT leży ~75 tys. znaków w głąb
            # artykułu -- reranker widzi tylko jego początek, więc bez słów "własne cele
            # mieszkaniowe" w rozszerzeniu art. 21 przegrywał z art. 30e (ranga 9).
            r"\bsprzeda\w*\b.*\b(mieszka|dom|nieruchom|działk|lokal)|\b(mieszka|dom|nieruchom|działk|lokal)\w*\b.*\bsprzeda",
            'zamiast "sprzedałem mieszkanie/dom/działkę -- czy zapłacę podatek" napisz "podatek od '
            "dochodu z odpłatnego zbycia nieruchomości przed upływem pięciu lat od nabycia oraz "
            'zwolnienie dochodu wydatkowanego na własne cele mieszkaniowe"',
        ),
        (
            # Krok 94: art. 4a ustawy o podatku od spadków i darowizn mówi "zstępni, wstępni",
            # nie "rodzice/dzieci" -- bez tego darowizna od rodziców trafiała w art. 16
            # (ulga mieszkaniowa), a art. 4a wypadał poza pulę kandydatów.
            r"\b(rodzic|mam[aęoy]|mami|matk|tat|ojc|ojciec|dziad|babci|babk|syn|córk|brat|siostr|mąż|męż|żon)\w*\b.*\b(darow|dał|dali|dostał|podarow|spad|odziedzicz)|\b(darow|dał|dali|dostał|podarow|spad|odziedzicz)\w*\b.*\b(rodzic|mam[aęoy]|mami|matk|tat|ojc|ojciec|dziad|babci|babk|syn|córk|brat|siostr|mąż|męż|żon)",
            'zamiast "rodzice dali mi pieniądze -- czy zapłacę podatek" napisz "zwolnienie od podatku '
            "od spadków i darowizn nabycia przez małżonka, zstępnych, wstępnych lub rodzeństwo pod "
            'warunkiem zgłoszenia nabycia naczelnikowi urzędu skarbowego"',
        ),
    ],
}


def build_query_expansion_messages(
    question: str, act_titles: list[str] | None = None, domain: str | None = None
) -> list[dict]:
    """Wiadomości do osobnego, krótkiego wywołania modelu, które przeformułowuje
    pytanie użytkownika na język zbliżony do ustaw -- naprawia lukę słownikową
    między potocznym pytaniem a formalnym tekstem przepisu, na którą cierpi
    zarówno etap embeddingowy, jak i reranking (zweryfikowane empirycznie:
    "zakupy prywatne" nie trafiało w art. 86 ustawy o VAT ani w rankingu
    kosinusowym, ani po rerankingu -- dopiero sformalizowane przeformułowanie
    wyciągało ten artykuł na 1. miejsce w obu etapach). Wynik generacji z tymi
    wiadomościami trafia jako `extra_queries` do RagIndex.search()/
    search_with_history() -- patrz rag_search.py, apply_rerank().

    `act_titles` (opcjonalnie, patrz RagIndex.act_titles()): nazwy ustaw
    indeksowanych w BIEŻĄCEJ dziedzinie, dopisywane jako jedna, czysto
    faktyczna linia kontekstu PRZED pytaniem w wiadomości UŻYTKOWNIKA --
    celowo NIE w QUERY_EXPANSION_SYSTEM_PROMPT (Krok 28, PROGRESS.md).
    Cztery wcześniejsze próby (Krok 25/26) zmieniały ten jeden, wspólny dla
    WSZYSTKICH dziedzin string i za każdym razem naprawiały jeden przypadek
    kosztem regresji w innym, niepowiązanym -- bo zmiana działała globalnie.
    Ten kontekst jest DOMENOWO-SPECYFICZNY (inna lista ustaw w każdej
    dziedzinie), więc naprawa jednej dziedziny (np. `podatkowe`, VAT art. 86)
    nie może struktu­ralnie wpłynąć na zachowanie w innej (np. `karne`,
    `cywilne`) -- inaczej niż poprzednie cztery próby. Zweryfikowane na 12
    przypadkach referencyjnych ze wszystkich dziedzin
    (`check_expand_query_regressions.py`): naprawia VAT art. 86 (#27 -> #2)
    i `ochrona_danych` art. 101 (#46 -> #6, Krok 25), drobna poprawa na 2
    innych, zero regresji na pozostałych 8 (jedna pozorna regresja w
    `kadry_i_place` okazała się błędem w samych danych referencyjnych --
    patrz Krok 28).

    `domain` (opcjonalnie, patrz RagIndex.domain): klucz do
    DOMAIN_QUERY_EXPANSION_HINTS -- ten sam wzorzec co `act_titles`
    (kontekst w wiadomości UŻYTKOWNIKA, nie w współdzielonym
    QUERY_EXPANSION_SYSTEM_PROMPT), tylko dla PAROWANYCH przykładów
    słownictwa (jak Krok 14/22/23), nie samych nazw ustaw. Powód (Krok
    41, PROGRESS.md): próba dopisania kolejnego przykładu wprost do
    QUERY_EXPANSION_SYSTEM_PROMPT (naprawa `gospodarcze`, art. 210 KSH)
    trzykrotnie regresowała NIEPOWIĄZANY przypadek `ochrona_danych`
    (art. 101, #6 -> #50) niezależnie od dokładnych słów przykładu --
    sam fakt dotknięcia tego globalnego stringa wystarczał. Trzymanie
    przykładu w słowniku kluczowanym domeną i wstrzykiwanie go TYLKO
    przy wywołaniach dla tej jednej domeny czyni regresję w innej
    domenie strukturalnie niemożliwą (ten sam argument co Krok 28 dla
    `act_titles`), kosztem tego, że przykład pomaga tylko w domenie, do
    której został przypisany, nie generalizuje globalnie."""
    user_content = question
    context_parts = []
    if act_titles:
        context_parts.append("Kontekst: pytanie dotyczy jednej z następujących ustaw: " + "; ".join(act_titles) + ".")
    hint_parts = [DOMAIN_QUERY_EXPANSION_HINTS[domain]] if domain in DOMAIN_QUERY_EXPANSION_HINTS else []
    for pattern, keyword_hint in DOMAIN_QUERY_EXPANSION_KEYWORD_HINTS.get(domain or "", []):
        if re.search(pattern, question, re.IGNORECASE):
            hint_parts.append(keyword_hint)
    hint = "; ".join(hint_parts)
    if hint:
        context_parts.append(f"Wskazówka słownikowa: {hint}.")
    if context_parts:
        user_content = "\n".join(context_parts) + f"\n\n{question}"
    return [
        {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


MAX_ARTICLE_CHARS = 6000
ARTICLE_HEAD_CHARS = 600  # początek długiego artykułu przed wycinkiem (Krok 77)


LOW_CONFIDENCE_SCORE_THRESHOLD = 0.75


def article_ref(label: str) -> str:
    """"art. 12" dla artykułu, sama etykieta dla załącznika ("załącznik nr 2",
    `download_acts.split_annexes`, Krok 70)."""
    return label if label.startswith("załącznik") else f"art. {label}"


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        # Niektóre artykuły (np. art. 50 ustawy systemowej, ok. 51k znaków po
        # latach nowelizacji) potrafią same zapełnić okno kontekstu modelu --
        # patrz PROGRESS.md, krok 10. Od Kroku 77 zamiast pierwszych
        # MAX_ARTICLE_CHARS znaków: początek artykułu + ustępy wokół fragmentu
        # trafionego przez wyszukiwanie (art. 30 ust. 5 Pr. bud. leży za 7,5 tys.).
        text, shortened = article_excerpt(r["text"], r.get("chunk_text"), MAX_ARTICLE_CHARS, ARTICLE_HEAD_CHARS)
        if shortened:
            text += f"\n[...treść artykułu skrócona do ustępów związanych z pytaniem, pełny tekst: {r['source_url']}...]"
        header = f"### {r['act_title']} -- {article_ref(r['article'])}"
        # Adnotacja "stan prawny" -- tylko gdy fragment faktycznie niesie
        # wymiar czasowy (indeks zbudowany z --include-history, patrz
        # PROGRESS.md Krok 19). Gdy oba pola None (dotychczasowy, domyślny
        # przypadek), nagłówek zostaje DOKŁADNIE taki jak dziś -- celowo,
        # żeby nie zmieniać formatu promptu dla nikogo, kto tej funkcji
        # nie używa (`as_of` samego `build_context` nie jest tu nawet
        # potrzebne do tej decyzji -- to obecność pól w wyniku decyduje,
        # nie to, czy wołający w ogóle podał `as_of`).
        if r.get("valid_from") or r.get("valid_to"):
            valid_from = r.get("valid_from") or "..."
            valid_to = r.get("valid_to") or "nadal"
            header += f" (stan prawny: {valid_from} – {valid_to})"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_file_context(file_results: list[dict]) -> str:
    parts = []
    for r in file_results:
        parts.append(f"### Plik użytkownika: {r['source_file']}\n{r['text']}")
    return "\n\n".join(parts)


def build_user_message(
    question: str, results: list[dict], file_results: list[dict] | None = None, as_of: str | None = None
) -> str:
    parts = []
    if as_of:
        # Notatka niezależna od adnotacji "stan prawny" na pojedynczych
        # fragmentach w build_context() -- ważne zwłaszcza gdy ŻADEN
        # dostarczony fragment jakiejś ustawy nie pokrywa tej daty (taki
        # fragment nie ma czego adnotować), a model i tak musi wiedzieć,
        # że konkretna data była proszona, żeby mógł to wprost powiedzieć
        # zamiast po cichu odpowiedzieć stanem bieżącym (patrz punkt 5
        # SYSTEM_PROMPT).
        parts.append(f"[Data, na którą ma obowiązywać odpowiedź: {as_of}]")
    if not results or results[0]["score"] < LOW_CONFIDENCE_SCORE_THRESHOLD:
        # Jawny sygnał zamiast liczenia na to, że model sam zauważy słabe
        # dopasowanie -- testy end-to-end pokazały, że nawet z regułą 1/2
        # SYSTEM_PROMPT ("nie korzystaj z własnej wiedzy o liczbach") model
        # potrafi po cichu dopowiedzieć konkretną wartość z pamięci, gdy
        # żaden fragment tematycznie nie pasuje (patrz PROGRESS.md, temat
        # spoza pobranych ustaw). NIE pomaga to w przypadku, gdy fragment
        # jest trafiony tematycznie, ale mimo to nie zawiera szukanej
        # liczby (np. artykuł delegujący do nieobecnego w indeksie
        # rozporządzenia) -- na to nie ma tu obrony, to wymagałoby
        # sprawdzania treści, nie samego score'u dopasowania.
        best_score = results[0]["score"] if results else None
        score_note = f" (najlepszy wynik dopasowania: {best_score:.2f})" if best_score is not None else " (brak wyników)"
        parts.append(
            f"[Uwaga: żaden odnaleziony fragment nie jest wysoko dopasowany do tego pytania{score_note} "
            "-- fragmenty poniżej prawdopodobnie NIE zawierają odpowiedzi. Jeśli faktycznie jej nie "
            "zawierają, powiedz to wprost zamiast zgadywać (patrz zasada 2 wyżej).]"
        )
    parts.append(f"Fragmenty aktów prawnych:\n\n{build_context(results)}")
    if file_results:
        parts.append(f"Fragmenty z wgranego przez użytkownika pliku:\n\n{build_file_context(file_results)}")
    parts.append(f"---\n\nPytanie: {question}")
    return "\n\n".join(parts)


_META_QUESTION_PATTERNS = [
    r"o czym (m[oó]wili[śs]my|rozmawiali[śs]my|by[łl]a mowa)",
    r"co (m[oó]wili[śs]my|napisa[łl]e[śs]|powiedzia[łl]e[śs]|odpowiedzia[łl]e[śs])",
    r"\bprzypomnij\b",
    r"\bpodsumuj\b.*\brozmow",
    r"na (samym )?pocz[ąa]tku.*(rozmowy|pytaniu)",
    r"\bpierwszym pytaniu\b",
    r"wcze[śs]niejszej (odpowiedzi|cz[ęe][śs]ci rozmowy)",
    r"o tym co (m[oó]wili[śs]my|napisa[łl]e[śs]|powiedzia[łl]e[śs])",
    r"zacytuj.*(m[oó]wili[śs]my|wcze[śs]niej|na pocz[ąa]tku)",
]
_META_QUESTION_RE = re.compile("|".join(_META_QUESTION_PATTERNS), re.IGNORECASE)


def looks_like_meta_question(question: str) -> bool:
    """Heurystyka (nie klasyfikator) wykrywająca pytania o przebieg TEJ
    rozmowy (np. "przypomnij", "o czym mówiliśmy") zamiast o nowy fakt
    prawny. Fragmenty RAG są wyszukiwane od nowa na podstawie samej
    treści pytania -- dla pytań meta bywają kompletnie nietrafione i
    eksperymentalnie potrafią zablokować zdolność modelu do korzystania
    z historii rozmowy, zwłaszcza po wcześniejszej odmowie w tej samej
    rozmowie (patrz PROGRESS.md, krok 10). Niewyczerpująca -- pytanie o
    nietypowym sformułowaniu po prostu wraca do domyślnego, sprawdzonego
    zachowania (fragmenty RAG jak zwykle)."""
    return bool(_META_QUESTION_RE.search(question))


MAX_LOOKBACK_QUESTIONS = 3


def search_with_history(
    rag,
    history: list[dict],
    question: str,
    top_k: int,
    as_of: str | None = None,
    extra_queries: list[str] | None = None,
) -> list[dict]:
    """RAG search z fallbackiem dla pytań eliptycznych typu "a po 15
    latach?", które same w sobie nie mają wystarczających słów
    kluczowych do trafnego wyszukania (np. brak słowa "urlop") -- w
    testach takie pytanie w ogóle nie trafiało na właściwy artykuł, a
    model bez groundingu z RAG "dopowiadał" nieistniejące szczegóły z
    pamięci historii (patrz PROGRESS.md, krok 10/11).

    Próbujemy też wyszukań z doklejonym KAŻDYM z ostatnich
    MAX_LOOKBACK_QUESTIONS pytań użytkownika osobno (nie tylko
    bezpośrednio poprzednim -- ono może być z zupełnie innego,
    zakończonego odmową wątku, np. "ile wynosi minimalne wynagrodzenie"
    tuż przed "a po 15 latach?", podczas gdy faktycznie powiązane
    pytanie o urlop było dwie tury wcześniej) i zostajemy przy tym
    wariancie, który dał najlepszy wynik -- PORÓWNANY ze zwykłym
    wyszukaniem samego pytania, biorąc maksimum (ten sam wzorzec
    "policz wszystkie warianty, weź maksimum" co przy rozszerzaniu
    zapytania, `RagIndex.search(extra_queries=...)`). Nie robimy tego
    zbiorczo (całą historią naraz) -- doklejanie niezwiązanego pytania
    do zapytania psuje trafność (sprawdzone empirycznie), dlatego każde
    poprzednie pytanie jest próbowane osobno.

    Próbujemy WARIANTÓW Z HISTORIĄ ZAWSZE, gdy jest jakaś historia --
    nie tylko gdy wynik dla samego pytania jest poniżej jakiegoś progu.
    Wcześniejsza wersja tej funkcji (do Kroku 36) uruchamiała wyszukanie
    z historią TYLKO gdy `results[0]["score"] < 0.75` -- Krok 36
    (PROGRESS.md) znalazł realny przypadek, w którym pytanie eliptyczne
    trafiało PRZYPADKOWO w niepowiązany, ale wysoko punktowany fragment
    (score 0.779 > próg) -- ta wcześniejsza wersja w ogóle nie próbowała
    wtedy wariantów z historią, bo próg "wyglądał" na spełniony, mimo że
    trafienie było błędne tematycznie. Sam próg nie ma jak odróżnić
    "wysoki wynik bo trafne dopasowanie" od "wysoki wynik bo przypadkowe
    podobieństwo do innego tematu" -- jedyny niezawodny sposób to zawsze
    policzyć oba i wziąć obiektywnie lepszy. Koszt tego to tylko
    dodatkowe wyszukiwania w indeksie (tanie, samo embedding+reranking,
    bez wywołania modelu językowego) -- nie generacja, więc nie ma tu
    ryzyka istotnego spowolnienia.

    `extra_queries` (opcjonalnie, patrz RagIndex.search()) trafia TYLKO
    do wyszukania samego `question` -- rozszerzenia (LLM-owe
    przeformułowanie na język formalny/prawniczy) dotyczą jednego,
    konkretnego pytania, nie ma sensu doklejać ich też do wariantów
    "poprzednie pytanie + bieżące" z pętli lookback niżej."""
    results = rag.search(question, top_k=top_k, as_of=as_of, extra_queries=extra_queries)
    if not history or not results:
        return results
    best_results, best_score = results, results[0]["score"]
    prior_questions = [m["content"] for m in history if m["role"] == "user"][-MAX_LOOKBACK_QUESTIONS:]
    for prior in prior_questions:
        candidate = rag.search(f"{prior} {question}", top_k=top_k, as_of=as_of)
        if candidate and candidate[0]["score"] > best_score:
            best_results, best_score = candidate, candidate[0]["score"]
    return best_results


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Zachowuje tylko ostatnie `max_turns` par (user, assistant) w historii
    rozmowy, żeby nie przepełnić okna kontekstu modelu -- każda tura dokłada
    do promptu również fragmenty RAG wstrzyknięte w poprzednich pytaniach
    użytkownika, więc historia rośnie szybciej niż liczba wiadomości."""
    max_messages = max_turns * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
    return history
