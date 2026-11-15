# Postęp prac — asystent prawa polskiego (Bielik + RAG + LoRA, wielodziedzinowy)

Ten plik to dziennik decyzji i znalezisk. README.md
to skrócona instrukcja "jak tego używać" — tu jest "dlaczego tak, a nie
inaczej" i pułapki po drodze.

## Krok 0: Kontekst i cel

Projekt źródłowy (wcześniejszy, jednodziedzinowy asystent kadrowo-płacowy) to działający, dobrze
przetestowany asystent RAG + LoRA dla jednej dziedziny (prawo pracy i
ubezpieczeń społecznych, 7 ustaw). Pytanie wyjściowe: czy dałoby się
zrobić coś analogicznego dla całego polskiego prawa?

Odpowiedź architektoniczna: tak, bo ELI API Sejmu obejmuje wszystkie
akty, nie tylko te kadrowe — ale **nie** jako jeden mega-indeks od razu.
Zamiast tego: rozszerzać dziedzinami, budować i mierzyć RAG osobno dla
każdej, i dopiero po walidacji rozważać łączenie. Główny, przewidywany
z góry problem: retrieval (samo podobieństwo embeddingów) może się
gorzej skalować niż liczba dziedzin sama w sobie, zwłaszcza gdy dziedziny
się tematycznie nakładają (np. kadry i podatki dzielą ustawę o PIT).

## Krok 1: Bootstrap z projektu źródłowego, generalizacja konfiguracji aktów

- Nowy katalog, osobne repo git.
- Skopiowany szkielet ze starego projektu: `scripts/`, `tests/`,
  `conftest.py`, `requirements*.txt`, `.gitignore`, `.github/workflows/`,
  `LICENSE`. Nie skopiowano `data/raw|processed/` (poza `*_meta.json`,
  potrzebne przez `check_acts_freshness.py`), `models/`, `adapters/`
  (regenerowalne/odtwarzane, patrz Krok 2) ani starego, 109KB PROGRESS.md.
- **Kluczowa zmiana:** hardcodowana lista `ACTS` w `download_acts.py`
  zastąpiona przez `data/acts_config.json` — płaska lista wpisów
  `{short, publisher, year, position, domains}`, gdzie `domains` to
  LISTA (nie zagnieżdżanie per dziedzina), bo jeden akt może należeć do
  kilku dziedzin naraz (np. ustawa o PIT: `["kadry_i_place", "podatkowe"]`,
  bez duplikowania wpisu).
- `download_acts.py`, `download_acts_history.py`, `check_acts_freshness.py`,
  `build_rag_index.py`, `rag_search.py`, `chat*.py` dostały flagę
  `--domain` filtrującą po tym polu. `download_acts.py --domain X`
  odbudowuje `all_articles.json` ze WSZYSTKICH już pobranych aktów na
  dysku (nie tylko z bieżącego przebiegu) — inaczej pobranie jednej
  dziedziny nadpisywałoby po cichu dane innych.
- `build_rag_index.py --domain X` zapisuje osobne pliki
  `rag_index_X.npy`/`_meta.json` zamiast domyślnego, łączonego
  `rag_index.npy` — każda dziedzina ma własny indeks.
- Migracja korpusu kadrowego (`kadry_i_place`, 7 ustaw) + pilotażowa
  dziedzina **podatkowe** (Ordynacja podatkowa `DU/1997/926` + ustawa o
  VAT `DU/2004/535`, identyfikatory zweryfikowane wprost przez ELI API,
  nie z pamięci).

## Krok 2: Wspólna infrastruktura z projektem siostrzanym

W trakcie pracy wyszło na jaw (pytanie użytkownika: "czy nowy projekt
używa Pythona z drugiego projektu?"), że tymczasowo korzystałem ze
`.venv` starego projektu do szybkiej weryfikacji — uznane za niepożądaną,
przypadkową zależność międzyprojektową. Naprawiono właściwym wzorcem:
**wydzielić do neutralnego katalogu-rodzeństwa, oba projekty symlinkują**
(nie: jeden projekt czyta bezpośrednio z katalogu drugiego):

- `~/Projekty/bielik-mlx-models/` — modele Bielik w MLX
  (11B + 4.5B, ~8,4GB), identyczne niezależnie od dziedziny.
- `~/Projekty/bielik-venv-shared/` — `.venv` (Python +
  biblioteki), bezpieczne dopóki `requirements*.txt` obu projektów są
  identyczne.
- `~/Projekty/bielik-lora-adapters/` — adaptery LoRA. W
  przeciwieństwie do modeli/venv, adaptery NIE są identyczne między
  dziedzinami (każdy trenowany na innych danych) — ale każdy konkretny
  adapter ma jedną fizyczną kopię, oba projekty symlinkują to samo,
  więc nie ma ryzyka rozjazdu dwóch kopii tego samego adaptera.

**Pułapka znaleziona i naprawiona:** `.gitignore` miał `.venv/` (ze
slashem — pasuje tylko do katalogów). Po zamianie `.venv` na symlink
przestało to pasować, `.venv` zaczął się pokazywać jako nieśledzony w
`git status`. Naprawione na `.venv` bez slasha (pasuje do obu: katalogu
i symlinka) w obu projektach.

Zweryfikowano też, że przenosiny `.venv` nie psują wrapper-skryptów z
zaszytą absolutną ścieżką w shebangu (`.venv/bin/pytest` itp.) — działają
nadal, bo shebang rozwiązuje się przez symlink do prawdziwej lokalizacji.

## Krok 3: Pilotaż dziedziny podatkowe — dane, trening, pierwsze znaleziska

- Ręcznie napisany zestaw fine-tune (20 train / 4 valid,
  `data/finetune/podatkowe/`) — pytania o Ordynację podatkową i VAT,
  w stylu identycznym do zestawu kadrowego (cytowanie artykułu, zwięzłość).
- **Zastrzeżenie zgłoszone przez użytkownika i zapisane w README obu
  projektów:** ani ja, ani użytkownik nie jesteśmy doradcą
  podatkowym/specjalistą kadrowym — żaden zestaw fine-tune nie był
  zweryfikowany merytorycznie przez eksperta branżowego. Traktować jako
  materiał do nauki STYLU, nie źródło zweryfikowanych faktów. Fakty mają
  i tak pochodzić z RAG (prawdziwy tekst ustawy, `source_url` do
  weryfikacji).
- Poprawka merytoryczna od użytkownika: stawka VAT 23% (nie 22%)
  obowiązuje nieprzerwanie od 2011 — zaktualizowano dwa przykłady, które
  wcześniej sugerowały niepewność co do tego, czy to nadal aktualne.
- Trening LoRA (`mlx_lm.lora`, 100 iteracji, batch-size 2,
  `--mask-prompt`): val loss najlepszy na iter 25 (0.706), potem
  systematycznie gorszy (0.857 → 0.919 → 0.944 na iter 50/75/100) —
  ten sam wzorzec przeuczania co w projekcie źródłowym dla kadr, tu
  jeszcze wyraźniejszy (20 przykładów zamiast 58). Użyty checkpoint
  iter25 jako domyślny adapter (`bielik11b-podatkowe-lora-iter25`), nie
  finalny.
- **Pierwsze realne znalezisko retrievalu:** pytanie o korektę deklaracji
  VAT nie wyciągało w top-8 art. 81 Ordynacji podatkowej (ogólna reguła
  korekty) — przegrywał z niszowymi artykułami ustawy o VAT o procedurze
  unijnej OSS (art. 130c/138g), mimo że te dosłownie zawierają frazę
  "deklaracja VAT". Zdiagnozowane wprost przez `rag_search.py`: art. 81
  w ogóle nie trafiał do top-8 czystego podobieństwa kosinusowego.

## Krok 4: Dziedzina administracyjne (KPA) + historia wersji

- Kodeks postępowania administracyjnego (`DU/1960/168`, zweryfikowane
  przez ELI API) — dobrany celowo jako "czysty" test: jedna ustawa,
  słownictwo wyraźnie odrębne od kadr/podatków, żeby odróżnić "czy
  retrieval psuje się od samej liczby dziedzin" od "czy psuje się od
  tematycznego nakładania się treści".
- 306 artykułów, zero ostrzeżeń o niesparowanych nawiasach (w
  przeciwieństwie do Ordynacji/VAT) — czystszy parse.
- Zestaw fine-tune (20/4) w tym samym stylu i z tym samym zastrzeżeniem
  o braku weryfikacji branżowej. Trening: ten sam wzorzec przeuczania
  (najlepszy val loss iter25: 0.790).
- **Hipoteza potwierdzona:** retrieval dla KPA jest bardzo pewny nawet
  bez pomocy rerankera (art. 129 o terminie odwołania: score 0.957,
  #1 wynik) — wyraźnie czystszy niż podatkowe. Sugeruje, że problem z
  Kroku 3 wynikał głównie z nakładania się TEMATYCZNEGO (PIT/VAT/
  Ordynacja dzielą słownictwo), nie z samego wzrostu liczby dziedzin.
  Dlatego prawo cywilne (świadomie nakładające się z kadrami przez
  odesłania w umowie o pracę) zostało odłożone na później jako
  trudniejszy test tej hipotezy, zamiast dodawane od razu.
- Dograno też historię wersji (`download_acts_history.py --domain X`)
  dla podatkowe i administracyjne (brakowało jej od Kroku 1/3) —
  wszystkie trzy dziedziny wspierają teraz `--as-of`/`--include-history`
  spójnie. Ordynacja podatkowa + PIT + VAT dały łącznie kilkanaście
  tysięcy artykułów historycznych (znacznie więcej niż KPA — starsze,
  częściej nowelizowane akty).

## Krok 5: Naprawa promptu systemowego (bug z generalizacji)

Podczas dalszych testów KPA wyszło na jaw: `SYSTEM_PROMPT` w
`prompt.py` był JEDNYM hardcodowanym stringiem — "Jesteś asystentem
kadrowo-płacowym..." — odziedziczonym 1:1 z projektu źródłowego.
Wszystkie trzy dziedziny (łącznie z podatkowe i administracyjne)
dostawały tę samą, kadrową personę w prompcie systemowym, mimo że
reszta pipeline'u (indeks RAG, adapter) poprawnie przełączała się na
`--domain`. Nie wyszło to na jaw w dotychczasowych testach, bo kontekst
RAG i tak dominował treść merytoryczną odpowiedzi — ale to był realny
błąd, nie coś do tolerowania.

Naprawa: `SYSTEM_PROMPT_TEMPLATE` + `DOMAIN_DESCRIPTIONS` (słownik
dziedzina -> krótki opis) + `build_system_prompt(domain)`. `RagIndex`
zapamiętuje teraz swoją dziedzinę (`self.domain`), więc `answer()` w
każdym z trzech wariantów chatu woła `build_system_prompt(rag.domain)`
zamiast stałego stringa. Zweryfikowane bez regresji na kadry_i_place.

## Krok 6: Reranking (cross-encoder)

Test dwóch problematycznych pytań z podatkowe (korekta deklaracji VAT z
Kroku 3, i nowe: "czy mogę odliczyć VAT od zakupów prywatnych" — art. 86
nie trafiał w top-8) potwierdził: to nie halucynacja, to retrieval —
zweryfikowane przez `--no-adapter` (ta sama, błędna co do wyboru
przepisu treść niezależnie od adaptera).

Dodano `sdadas/polish-reranker-base-ranknet` (ten sam autor co model
embeddingowy `mmlw-retrieval-roberta-large` — spójny wybór) jako drugi
etap: top-30 kandydatów z podobieństwa kosinusowego trafia do
cross-encodera, który ocenia parę (pytanie, pełny tekst artykułu)
precyzyjniej. Domyślnie włączone, `--no-rerank` do porównań.

Wynik na pytaniu o korektę deklaracji VAT: art. 81 Ordynacji z pozycji
14 (poza top-8 kosinusowym) wskoczył na 3. miejsce po rerankingu —
naprawione. Wynik na pytaniu o odliczenie VAT: art. 86 nadal poza
top-8 nawet po rerankingu (rank ~16) — **nie w pełni naprawione**,
zidentyfikowane jako głębszy problem (patrz Krok 7).

## Krok 7: Query expansion + rozróżnienie cytatu od wniosku

Dalsze testy KPA (4 dodatkowe pytania) dały wynik 3/4 mocnych + jeden
częściowy: pytanie "czy mogę się odwołać od samego uzasadnienia decyzji"
dostało odpowiedź z DOSŁOWNIE poprawnymi cytatami (art. 127/128,
zweryfikowane wprost w pobranym tekście — nic nie zmyślone), ale
wniosek prawny z nich wyciągnięty był mocniejszy niż to, co przepisy
literalnie stwierdzają.

**Naprawa #1 (częściowa):** nowa zasada 6 w `SYSTEM_PROMPT_TEMPLATE` --
każe modelowi wprost odróżniać dosłowne brzmienie przepisu od dalszego
wniosku interpretacyjnego. Zweryfikowana jako poprawa (model przestał
formułować mocniejszą, niepoprawioną tezę), ale niepełna — adapter LoRA
był trenowany na starym stylu, sam prompt ma ograniczony wpływ na już
wyuczone zachowanie generowania.

**Naprawa #2 (pełna) — query expansion** dla problemu z Kroku 6 (art. 86
nie trafiał w top-8 nawet po rerankingu): zdiagnozowano, że **sam
reranker ma tę samą lukę słownikową co bi-encoder** — art. 86 oceniany
na 0.230 względem oryginalnego, potocznego pytania ("zakupy prywatne"),
ale na 0.936 względem tego samego pytania sformalizowanego ("wydatki
niezwiązane z prowadzoną działalnością gospodarczą"). Rozwiązanie:
- `RagIndex.search(extra_queries=...)` — każdy wariant zapytania
  (oryginalny + rozszerzenia) szuka osobno na etapie kosinusowym,
  `merge_candidates()` łączy wyniki bez duplikatów.
- `apply_rerank()` przyjmuje teraz listę zapytań (nie tylko jedno) --
  bierze MAKSIMUM wyniku rerankera spośród wariantów per kandydat.
- Każdy z trzech wariantów chatu (`chat.py`/MLX, `chat_cuda.py`/
  transformers, `chat_lmstudio.py`/HTTP) dostał własny `expand_query()`
  -- jedno krótkie (max 80 tokenów), dodatkowe wywołanie modelu
  przeformułowujące pytanie na język formalny/prawniczy
  (`prompt.build_query_expansion_messages`), zanim trafi do
  wyszukiwania. Domyślnie włączone, `--no-expand-query` do wyłączenia.

Zweryfikowane end-to-end: pytanie o odliczenie VAT od zakupów
prywatnych teraz poprawnie cytuje art. 86 ust. 1 (z dodatkowym, trafnym
niuansem o proporcji przy mieszanym użytku, art. 86 ust. 2a) zamiast
wcześniejszego niepewnego "nie wymienione w dostarczonych fragmentach,
ale znane z wiedzy ogólnej". Bez regresji na kadry_i_place ani
administracyjne (zweryfikowane ręcznie), 108 testów zielonych.

## Krok 8: Dziedzina cywilne (Kodeks cywilny) — najtrudniejszy dotąd pilotaż

Odłożona wcześniej celowo (Krok 4) jako najtrudniejszy test: `DU/1964/93`
(zweryfikowane przez ELI API), 1296 artykułów -- jedna ustawa, ale
wewnętrznie bardzo zróżnicowana (część ogólna, prawo rzeczowe,
zobowiązania, spadki) i realnie nakładająca się tematycznie z
`kadry_i_place` (umowa o pracę odwołuje się do KC).

**Nowy, głębszy problem retrievalu znaleziony i zdiagnozowany:**
pytanie "kto odpowiada za szkodę wyrządzoną z winy umyślnej?" nie
wyciągało art. 415 (ogólna reguła odpowiedzialności deliktowej) nawet z
query expansion włączonym. Zdiagnozowane dokładnie: art. 415 jest
BARDZO krótki (88 znaków) i konkuruje z ok. 35 długimi, szczegółowymi
przepisami pochodnymi w tym samym rozdziale (art. 415-449, różne
warianty odpowiedzialności za szkodę) -- i bi-encoder, i cross-encoder
systematycznie oceniają go niżej niż te dłuższe sąsiadujące przepisy,
niezależnie od sformułowania zapytania (przetestowano też wariant
"jaka jest ogólna zasada..." -- ocena spadła jeszcze bardziej, 0.661 ->
0.543).

**Próby naprawy i dlaczego żadna nie weszła do produkcji:**
- Drugi wariant rozszerzenia zapytania (framowany jako pytanie o
  "ogólną zasadę") -- nie pomógł, pogorszył ocenę.
- Podniesienie `top_k` -- art. 415 wchodzi dopiero na 11-13. miejscu,
  wymagałoby prawie podwojenia kontekstu (więcej szumu, nie tylko
  poprawka).
- **Większy reranker** (`sdadas/polish-reranker-large-ranknet` zamiast
  `-base-ranknet`) -- faktycznie stawia art. 415 na #1 (0.944)! Ale
  test regresji na 7 reprezentatywnych pytaniach z 4 dziedzin pokazał:
  psuje 2 z 3 wcześniej poprawnych przypadków (podatkowe: korekta
  deklaracji trafia w art. 133 zamiast 81; administracyjne: termin
  odwołania trafia w art. 133 zamiast 129, mimo że to był wcześniej
  najpewniejszy wynik w całym projekcie, 0.957) -- systematyczna,
  nieznana wcześniej wada dużego modelu (myli pewne artykuły
  proceduralne w RÓŻNYCH ustawach), która nie ustępuje nawet z query
  expansion. Warianty MSE (base i large) nie rozwiązują problemu
  wcale. Uśrednienie i maksimum wyników obu rerankerów (base + large)
  też nie dają czystej poprawy (odpowiednio 4/7 i 5/7 -- to samo albo
  gorzej niż sam base, tylko z innymi błędami).
- **Hybryda BM25 + embeddingi** (`rank_bm25`, metoda czysto leksykalna,
  zupełnie inny mechanizm niż sieć neuronowa) -- BM25 samo w sobie
  stawia art. 415 na 4. miejscu (inne obciążenie niż bi-encoder).
  Połączono pulę kandydatów BM25 z dotychczasową (kosinus + expansion)
  i przepuszczono przez ten sam reranking -- **art. 415 nadal poza
  top-10**. To rozstrzygające: skoro artykuł trafia do puli
  kandydatów przez TRZY niezależne metody (rozszerzone zapytanie,
  BM25, zwykły top-30) i reranker za każdym razem go strąca, wąskie
  gardło leży WYŁĄCZNIE w etapie oceniania (reranking), nie w
  przywoływaniu kandydatów (recall) -- żadna metoda poszerzania puli
  tego nie naprawi.
- **Wniosek: żadna z wypróbowanych zmian rerankera/embeddingu/metody
  przywoływania nie jest strict improvement.** Domyślny reranker
  zostaje `-base-ranknet` bez zmian -- problem udokumentowany jako
  otwarte ograniczenie, nie naprawiony. Jedyna pozostała, niesprawdzona
  dźwignia -- ręczna korekta wyniku rerankera pod długość tekstu --
  świadomie NIE wypróbowana: mamy tylko jeden potwierdzony przypadek
  tego zjawiska, a kalibracja na jednym przykładzie ryzykuje
  przeuczenie się pod niego i zepsucie innych, krótkich-ale-wąskich
  przepisów (nie każdy krótki artykuł to ogólna zasada). Decyzja:
  poczekać na więcej potwierdzonych przypadków przed próbą takiej
  korekty.

**Ważna lekcja metodologiczna, znaleziona przy okazji:** pierwszy test
end-to-end przez `chat.py` (z adapterem) WYGLĄDAŁ na sukces -- odpowiedź
poprawnie cytowała art. 415. Weryfikacja pokazała, że to fałszywy
pozytyw: to dokładnie pytanie z zestawu treningowego (przykład #3),
adapter LoRA je zapamiętał i "przeformułowanie zapytania"
(query expansion) zwróciło od razu gotową, zapamiętaną odpowiedź
zamiast prawdziwej parafrazy -- niezależnie od faktycznego stanu
retrievalu. Test z `--no-adapter` (bez tego skrótu pamięciowego)
potwierdził: problem nadal nienaprawiony. **Wniosek na przyszłość:
nie testować jakości retrievalu pytaniami wprost z zestawu
treningowego, gdy adapter jest załadowany** -- może "zdać" test przez
pamięć, maskując prawdziwy stan.

**Czy to samo dotyczy kadry_i_place?** Sprawdzone empirycznie (czysty
`rag_search.py`, bez LLM): analogiczne krótkie/ogólne przepisy Kodeksu
pracy (art. 8 -- nadużycie prawa, niemal identyczna konstrukcja jak
art. 5 KC; art. 18 -- nieważność mniej korzystnych postanowień umowy)
trafiają bezbłędnie, #1 z wysoką pewnością (0.982 i 0.995), bez pomocy
query expansion. **Wniosek: to nie problem samej krótkości tekstu ani
wielkości dziedziny, tylko gęstości tematycznej konkretnego fragmentu
ustawy** -- KC ma nietypowo duże skupisko (~35 artykułów) blisko
powiązanych wariantów jednego tematu, czego Kodeks pracy nie ma wokół
swoich odpowiedników.

Poza tym: zbiór fine-tune (20/4, to samo zastrzeżenie), adapter LoRA
(ten sam wzorzec przeuczania, najlepszy val loss iter25: 0.664),
historia wersji dograna (`--include-history`, 12893 artykułów
historycznych). Wszystko poza jednym udokumentowanym ograniczeniem
działa poprawnie (zweryfikowane m.in. na art. 158 -- forma aktu
notarialnego, i art. 353¹ -- zasada swobody umów, oba #1 z bardzo
wysoką pewnością).

## Krok 9: Piąta dziedzina -- rodzinne (Kodeks rodzinny i opiekuńczy)

Kolejny "czysty" pilotaż (pojedynczy akt, DU/1964/59, 247 artykułów,
słownictwo odrębne od pozostałych czterech dziedzin) -- wybrany celowo
jako drugi punkt danych dla hipotezy o gęstości tematycznej z Kroku 8
(`cywilne`).

Przy okazji znaleziono i naprawiono realny brak: `cywilne` nigdy nie
trafiło do `DOMAIN_DESCRIPTIONS` w `scripts/prompt.py` (dodane w Kroku
6, pilotaż w Kroku 8) -- dostawało domyślny, mniej precyzyjny opis
"polskiego prawa" zamiast "polskiego prawa cywilnego". Naprawione
razem z dodaniem wpisu dla `rodzinne`.

Wynik testu hipotezy gęstości tematycznej: **problem z Kroku 8
praktycznie nie występuje tutaj.** Sprawdzone na trzech gęsto
obsadzonych klastrach tematycznych:
- obowiązek alimentacyjny (~15 artykułów, 128-144¹) -- ogólny art. 133
  trafia na #1 (rerank 0.943),
- ustrój majątkowy małżeński (~15 artykułów, 23-46) -- ogólny art. 23
  trafia na #1 (rerank 0.965),
- rozwód/separacja (56-61³) -- ogólny art. 56 spada na #2 (rerank
  0.848) za węższym, powiązanym art. 61² (rerank 0.918) -- jedyne
  znalezione odchylenie, i to dużo łagodniejsze niż w `cywilne` (tam
  ogólny przepis w ogóle nie trafiał do top-8, tu wciąż jest #2).

Wniosek: to nie jest uniwersalna cecha "krótki/ogólny przepis w gęstym
klastrze przegrywa" -- to coś specyficznego dla sekcji deliktowej
Kodeksu cywilnego (art. 415-449), nie odtwarza się w analogicznie
zbudowanych klastrach tematycznych w innej ustawie. Zbiera się więc
tylko jeden potwierdzony przypadek (art. 415 KC) plus jeden łagodny
(art. 56 KRO) -- wciąż za mało, żeby uzasadnić ręczną korektę rerankera
pod długość tekstu (decyzja o odłożeniu tego z Kroku 8 pozostaje
aktualna).

Poza tym: zbiór fine-tune (20/4, to samo zastrzeżenie), test
end-to-end na pytaniu spoza zbioru treningowego ("czy mąż może
sprzedać wspólne mieszkanie bez zgody żony", cytuje poprawnie art. 37
§ 1 pkt 1 i § 2 KRO), zweryfikowany też z `--no-adapter` (ta sama
poprawna odpowiedź -- potwierdza, że to retrieval, nie pamięć
adaptera, patrz metodologia z Kroku 8). Adapter LoRA -- ten sam wzorzec
przeuczania (najlepszy val loss iter25: 0.664, potem 0.775 → 0.810 →
0.834). Historia wersji dograna (`--include-history`, 1686 artykułów
historycznych, 3 wersje).

## Krok 10: Szósta dziedzina -- karne (Kodeks karny)

Naturalne rozszerzenie zgodne z pierwotnym celem (całe polskie prawo)
i trzeci punkt danych dla hipotezy o gęstości tematycznej. Przy okazji
znaleziona i naprawiona pomyłka: pierwsze podejście do znalezienia
identyfikatora ELI aktu bazowego przez pole "Tekst jednolity dla aktu"
w wynikach wyszukiwania po tytule zwróciło DU/1997/557 -- to jednak
Kodeks karny **wykonawczy**, nie Kodeks karny (DU/1997/553). Błąd
wykryty od razu po pierwszym pobraniu (tytuł w logu jasno pokazał
pomyłkę), skorygowany przed dalszą pracą (usunięto błędnie pobrane
pliki, pobrano poprawny akt). Lekcja: przy wyszukiwaniu ELI po tytule
nie ufać ślepo pierwszemu trafionemu odnośnikowi "Tekst jednolity dla
aktu" bez sprawdzenia `title` zwróconego rekordu.

**Najważniejszy wynik: problem gęstości tematycznej z Kroku 8 wrócił,
poważnie**, osłabiając wcześniejszy wniosek z Kroku 9 ("specyficzne dla
sekcji deliktowej KC"). Dla pytania "jaka jest kara za zabójstwo
człowieka": art. 148 (podstawowy, dosłowna odpowiedź) ma najwyższe
podobieństwo kosinusowe ze wszystkich kandydatów (0.825, #1 bez
rerankingu w `--no-rerank`), ale po rerankingu spada na 6. miejsce --
wyżej trafiają art. 155 (nieumyślne spowodowanie śmierci -- inny czyn),
art. 150 (eutanazja -- uprzywilejowana postać), art. 159 i art. 148a.
Zmiana samego sformułowania pytania ("kto zabija człowieka, jakiej
karze podlega") podniosła art. 148 na 2. miejsce -- ranking rerankera
jest więc silnie wrażliwy na dokładne sformułowanie, nie tylko na
gęstość klastra.

Zweryfikowano wpływ na realną odpowiedź: `chat.py` (domyślny
`top_k=8`, z i bez query expansion, z i bez adaptera) **poprawnie**
cytuje art. 148 we wszystkich wariantach -- bo mimo spadku na 6.
miejsce w rerankingu, artykuł wciąż mieści się w oknie 8 fragmentów
przekazywanych do modelu, a model generujący poprawnie go wybiera
spośród dostarczonych fragmentów. Margines jest jednak wąski (2
miejsca do wypadnięcia z kontekstu) -- to nie jest "naprawione", tylko
akurat nie zepsute przy obecnym `top_k`.

Aktualizacja wniosku z Kroku 9: to nie jest problem specyficzny dla
jednej sekcji jednej ustawy -- mamy teraz 3 punkty danych o różnym
nasileniu: `rodzinne` (brak/łagodny efekt), `karne` (poważny w samym
rerankingu, ale wciąż w oknie top-8 więc bez wpływu na odpowiedź),
`cywilne` (poważny i faktycznie wypadający poza top-8, z realnym
błędem w odpowiedzi bez adaptera). Wygląda na cechę samego rerankera
przy pytaniach o krótkie, "wzorcowe" przepisy karne/deliktowe (schemat
"kto robi X, podlega karze/odpowiedzialności Y") w otoczeniu wielu
podobnie sformułowanych przepisów pochodnych -- ale wciąż niejasne,
czy KC było akurat najgorszym przypadkiem, czy typowym. **To wystarczający
powód, żeby wrócić do użytkownika z pytaniem, czy teraz (3 przypadki
zamiast 1) chce spróbować ręcznej korekty rerankera pod długość/wzorzec
tekstu, czy nadal czekać.**

Przy okazji zauważona (nie zbadana głębiej) osobna usterka: model
bazowy bez adaptera, odpowiadając na pytanie o kradzież z włamaniem
popełnioną przez sąsiada, błędnie założył że sąsiad mieści się w
definicji "osoby najbliższej" z art. 115 §11 KK (wąska, zamknięta
definicja -- małżonek, wstępny, zstępny, rodzeństwo, powinowaty tej
samej linii/stopnia, przysposabiający/przysposobiony, osoba pozostająca
we wspólnym pożyciu -- sąsiad się w niej nie mieści). Wersja z
adapterem (iter25) tego błędu nie popełniła. Nie zbadano, czy to
przypadek, czy systematyczna różnica -- odnotowane jako obserwacja, nie
naprawiane w tym kroku.

Poza tym: zbiór fine-tune (20/4, to samo zastrzeżenie), adapter LoRA
(ten sam wzorzec przeuczania, najlepszy val loss iter25: 0.716, potem
0.835 → 0.898 → 0.961), historia wersji dograna (`--include-history`,
3546 artykułów historycznych, 3 wersje). Test end-to-end na pytaniu
spoza zbioru treningowego (kradzież z włamaniem do garażu) poprawny
z i bez adaptera.

## Krok 11: Naprawa problemu gęstości tematycznej -- fuzja rang (RRF)

Po Kroku 10 (3 punkty danych: `rodzinne` łagodny/brak, `karne` poważny,
`cywilne` poważny) użytkownik zdecydował spróbować naprawy teraz,
zamiast dalej czekać.

**Diagnoza kluczowa:** w obu poważnych przypadkach (KC art. 415, KK
art. 148) bi-encoder (podobieństwo kosinusowe) ocenia właściwy artykuł
bardzo wysoko (odpowiednio #2 i #1!) -- to WYŁĄCZNIE reranker go
spycha. Jednocześnie w innych, wcześniej naprawionych przypadkach (KP
art. 18, KRO art. 56, Ordynacja art. 81) to reranker ratuje bardzo
słaby ranking kosinusowy (odpowiednio #22, #7, #5 -> #2, #2, #2).
Żadna z dwóch metod z osobna nie jest więc uniwersalnie lepsza --
prosta ważona suma wyników (np. `alpha*rerank + (1-alpha)*cos`) nie
mogłaby jednocześnie spełnić obu kierunków bez utraty jednego z nich.

**Rozwiązanie: Reciprocal Rank Fusion (RRF, k=0)** --
`rag_search.py::fuse_rankings()`. Dla każdego kandydata liczy
`1/(pozycja_kosinusowa+1) + 1/(pozycja_rerankera+1)` i sortuje malejąco
po tej sumie. Metoda odporna na różnicę skali między dwoma modelami
(liczy się WZGLĘDNA pozycja w każdym rankingu, nie surowy wynik), więc
nie wymaga normalizacji ani strojenia progu. Zaimplementowana jako
osobny krok PO `apply_rerank()` w `RagIndex.search()` (domyślnie
włączony, `RagIndex(fuse=False)` / `rag_search.py --no-fuse` do
wyłączenia), nie modyfikuje samego `apply_rerank()` (zostaje
kompatybilny wstecznie, własne testy bez zmian).

**Zweryfikowane na 10 przypadkach referencyjnych z 6 dziedzin** (skrypt
ad-hoc licząc pozycję kosinusową i rerankera per przypadek, potem
symulacja RRF przy różnych k -- k=0 dało najlepszy kompromis):
- Naprawione (poważne): KC art. 415 #9 -> #3, KK art. 148 #6 -> #2.
- Bez regresji (musiały zostać #1): KPA art. 129, KP art. 8, KC art.
  158, KC art. 353¹ -- wszystkie zostały #1 (zgodny zwycięzca w obu
  rankingach, `fused_score` = maksimum 2.0).
- Lekko obniżone, ale wciąż dobrze w oknie top-8 (były #2, teraz #3-4):
  Ordynacja art. 81 (#2->#4), KP art. 18 (#2->#3), KRO art. 56 (#2->#3).
- VAT art. 86 (przypadek z Kroku 7, wymagający query expansion) --
  niepoprawiony samą fuzją bez rozszerzenia zapytania (oczekiwane, to
  inny mechanizm naprawiający inny problem), ale z rozszerzeniem
  zapytania fuzja daje wynik RÓWNIE DOBRY LUB LEPSZY niż sam reranking
  (symulacja: #13 -> #4).

Dodane testy jednostkowe `TestFuseRankings` (4 przypadki, w tym
regresja dokładnie na wzorcu "art. 148" -- kandydat #1 kosinusowo,
zepchnięty przez reranking, musi wrócić w okolice szczytu). 112/112
testów zielonych.

**Zweryfikowane end-to-end:**
- KC art. 415, `--no-adapter --no-expand-query`: teraz poprawnie
  cytuje art. 415 z dosłownym brzmieniem -- wcześniej (przed Krokiem
  11) w ogóle nie pojawiał się w odpowiedzi.
- KK art. 148, pełna ścieżka produkcyjna (adapter + query expansion +
  fuzja): odpowiedź precyzyjna, rozróżnia nawet § 2/§ 4 i art. 148a.

**Ograniczenie znalezione przy weryfikacji: interakcja z rozszerzaniem
zapytania jest niejednoznaczna, zależna od konkretnego przypadku.** Dla
KK art. 148 rozszerzenie POMAGA (parafraza modelu trafnie powtarza
treść przepisu, art. 148 zostaje #1). Dla KC art. 415 rozszerzenie
SZKODZI (parafraza modelu to bardziej ogólne sformułowanie zasady
deliktowej, które wciąga dodatkowych konkurencyjnych kandydatów -- art.
415 wypada z powrotem poza top-8, mimo że bez rozszerzenia zostaje w
oknie na #3). Sprawdzone bezpośrednio: `rag.search(..., extra_queries=[...])`
z prawdziwą parafrazą wygenerowaną przez `expand_query()` w obu
przypadkach. To NIE jest cofnięcie naprawy z Kroku 11 (naprawa działa
w pełni na ścieżce bez rozszerzania) -- to osobne, nowo odkryte
ograniczenie samego rozszerzania zapytania (inny mechanizm, inny
problem: luka słownikowa, nie gęstość tematyczna), udokumentowane w
README jako wskazówka: użyj `--no-expand-query` dla pytań o ogólne
zasady w `cywilne`, jeśli odpowiedź wygląda niekompletnie. Nie
naprawiane w tym kroku -- naprawa wymagałaby albo lepszego promptu dla
`expand_query` (ryzyko: zmiana wpływa na WSZYSTKIE dziedziny), albo
osobnego mechanizmu wykrywania "ten wariant zapytania psuje więcej niż
naprawia" -- obie opcje to nowy zakres pracy, nie kontynuacja tej.

## Krok 12: Naprawa interakcji fuzji rang z rozszerzaniem zapytania

Bezpośrednia kontynuacja Kroku 11 -- użytkownik zapytał "a jak to
naprawić?" o właśnie znalezione ograniczenie (rozszerzanie zapytania
potrafi cofnąć naprawę fuzji rang dla KC art. 415).

**Diagnoza:** `fuse_rankings()` z Kroku 11 liczy ranking kosinusowy na
SCALONEJ puli kandydatów (`merge_candidates`, które trzyma NAJWYŻSZY
'score' spośród wariantów zapytania per kandydat) -- to mieszanie
wyników policzonych względem RÓŻNYCH zapytań w jednym rankingu. Efekt
uboczny: sąsiedni przepis (art. 361, art. 429), słabo dopasowany do
oryginalnego pytania, ale bardzo dobrze dopasowany kosinusowo do
SFORMALIZOWANEJ PARAFRAZY wygenerowanej przez query expansion, dostaje
ten wysoki kosinus i na tej podstawie wygrywa w fuzji z art. 415 --
mimo że art. 415 jest #2 kosinusowo względem PRAWDZIWEGO pytania
użytkownika. Zweryfikowane wprost (`rank_chunks` per wariant osobno):
art. 415 jest #2 w liście oryginalnego zapytania, ale scalona pula z
zapytaniem rozszerzonym wprowadza nowych, silniejszych (dla PARAFRAZY)
konkurentów.

**Rozwiązanie:** nowa funkcja `fuse_rankings_multi_query()` w
`rag_search.py` -- liczy fuzję (kosinus+reranking) OSOBNO w obrębie
KAŻDEGO wariantu zapytania (ranking tylko wśród kandydatów faktycznie
przywołanych przez TEN wariant, nie całej scalonej puli), a dopiero
potem bierze MAKSIMUM wyniku fuzji spośród wariantów per kandydat --
ten sam wzorzec "bierz maksimum spośród wariantów", którym już
kierował się `apply_rerank` z Kroku 7, tylko zastosowany do wyniku
fuzji, nie samego rerankingu. `RagIndex.search()` używa jej zamiast
`apply_rerank`+`fuse_rankings` gdy jest więcej niż jeden wariant
zapytania ORAZ fuzja jest włączona; ścieżka jednowariantowa (bez
rozszerzenia) i ścieżka `--no-fuse` zostają bez zmian (własne testy
`TestFuseRankings`/`TestApplyRerank` nietknięte).

**Zweryfikowane end-to-end na WSZYSTKICH 10 przypadkach referencyjnych,
tym razem z PRAWDZIWYM `expand_query()`** (nie ręcznie napisaną
parafrazą jak w Kroku 11) -- 9/10 w top-8:
- KC art. 415: wraca do top-8 (#4) mimo aktywnego rozszerzania --
  naprawione.
- KK art. 148: zostaje #1 z rozszerzaniem (bez zmian na plus).
- Wszystkie wcześniej pewne wyniki (#1) zostają #1: KPA art. 129, KP
  art. 8, KC art. 158, KC art. 353¹.
- Pozostałe (Ordynacja art. 81, KP art. 18, KRO art. 56): w top-8.
- **Jedyny wyjątek: VAT art. 86** (przypadek z Kroku 7) -- nadal poza
  top-8 (#30) z AKTUALNIE generowaną przez model parafrazą. Zbadane
  wprost: to NIE regresja tej zmiany -- ta sama parafraza ze STARYM
  podejściem (scalanie-przed-rerankingiem z Kroku 11) też daje #21,
  poza top-8. Sam `cross_encoder.predict()` ocenia parę (ta konkretna
  parafraza, art. 86) nisko (0.227), mimo że parafraza semantycznie
  bardzo przypomina tę udokumentowaną w Kroku 7 jako skuteczną (0.936)
  -- reranker jest więc wrażliwy na DOKŁADNE sformułowanie w sposób,
  którego nie kontrolujemy (deterministyczne, sprawdzone 3x pod rząd,
  ten sam wynik -- to nie losowość dekodowania, tylko właściwość tej
  konkretnej pary tekstów). To przedwcześnie znana, nienowa
  krucha zależność samego mechanizmu rozszerzania zapytania (Krok 7),
  nie coś, co ten krok pogorszył ani naprawił.

Dodane testy `TestFuseRankingsMultiQuery` (3 przypadki, w tym
bezpośrednia regresja "kandydat nie może stracić przez wyższy surowy
kosinus rywala z INNEGO wariantu"). 115/115 testów zielonych.

## Stan na koniec tej sesji

Sześć dziedzin działa end-to-end. Problem gęstości tematycznej
rerankingu (Kroki 8-10) naprawiony fuzją rang, TERAZ TAKŻE w pełnej
ścieżce produkcyjnej z rozszerzaniem zapytania (Krok 12) -- 9/10
przypadków referencyjnych w top-8, jedyny pozostały wyjątek (VAT art.
86) to znana wcześniej, osobna krucha zależność query expansion od
dokładnego sformułowania parafrazy, nie problem gęstości tematycznej.
Wspólna infrastruktura (modele/venv/adaptery) spójna między tym
projektem a projektem źródłowym. Wszystkie zmiany wypchnięte
na GitHub.

## Krok 13: Potwierdzenie obserwacji z Kroku 10 -- błąd modelu bazowego w stosowaniu definicji "osoby najbliższej"

Sprawdzone empirycznie, czy obserwacja z Kroku 10 (model bazowy błędnie
zakłada, że sąsiad mieści się w definicji "osoby najbliższej" z art.
115 §11 KK) jest przypadkiem jednorazowym, czy systematyczną różnicą
między modelem bazowym a wersją z adapterem.

**Metoda:** to samo pytanie zadane modelowi bez adaptera i z adapterem
(`bielik11b-karne-lora-iter25`), bez zmian w RAG top-k ani innych
ustawieniach: *"Ktoś włamał się do mojego garażu i ukradł rower.
Rozpoznałem, że to mój sąsiad. Co mu grozi i jak mam to zgłosić?"*

**Wynik -- potwierdzone, systematyczne:**
- Bez adaptera: model poprawnie cytuje art. 279 §1/§2 i art. 278 §1/§4,
  ale **nie stosuje** definicji do konkretnego stanu faktycznego --
  kończy niejednoznacznym wnioskiem sugerującym, że "wniosek
  pokrzywdzonego jest konieczny do wszczęcia postępowania", bez
  rozstrzygnięcia, że sąsiad nie spełnia definicji z art. 115 §11 KK.
  To realny błąd praktyczny (mógłby zmylić pytającego co do trybu
  ścigania).
- Z adapterem: model wprost rozstrzyga -- "skoro sprawcą jest sąsiad, a
  nie osoba najbliższa, ściganie następuje z urzędu" i "Nie jest
  wymagany wniosek o ściganie" -- poprawnie i jednoznacznie.

**Uwaga metodologiczna:** replikacja zależy od sformułowania pytania.
Wersja explicite podpowiadająca kontekst ("...biorąc pod uwagę że to
sąsiad") dawała poprawną odpowiedź nawet bez adaptera -- błąd ujawnia
się tylko przy naturalnym, nieprowadzącym sformułowaniu, gdzie model
sam musi dostrzec i zastosować wąską definicję ustawową do faktów.

**Wniosek:** to nie jest usterka gęstości tematycznej rerankingu (RAG
dostarcza właściwe artykuły w obu przypadkach -- różnica jest wyłącznie
w rozumowaniu modelu generującego). To udokumentowany dowód, że LoRA
fine-tuning na tej dziedzinie poprawia nie tylko cytowanie przepisów,
ale też stosowanie wąskich, zamkniętych definicji ustawowych
(art. 115 §11 KK) do konkretnego stanu faktycznego. Nie wymaga żadnej
naprawy -- adapter już działa poprawnie; to potwierdzenie wartości
fine-tuningu, nie nowy problem do rozwiązania.

## Krok 14: Naprawa VAT art. 86 -- precyzyjniejszy prompt query expansion

Użytkownik wrócił do dwóch punktów z "Co dalej": VAT art. 86 i Zasada 6.
Zaczęto od VAT art. 86 jako mniejszego, samodzielnego zadania.

**Diagnoza:** obecna, generowana przez `expand_query()` parafraza pytania
"czy mogę odliczyć VAT od zakupów prywatnych" to "Czy można odliczyć
podatek VAT od wydatków niezwiązanych z prowadzoną działalnością
gospodarczą?" -- brzmi formalnie-prawniczo, ale używa ogólnego słowa
"odliczyć", którym posługują się też dziesiątki artykułów ustawy o PIT
o ulgach podatkowych (ta sama domena `podatkowe` zawiera oba akty).
Zweryfikowane wprost (`rank_chunks` + `cross_encoder.predict` per
wariant): ta parafraza ma ranking kosinusowy #12, ale reranking spycha
ją na #43 -- ustawa o PIT wygrywa z wynikami rerankingu 0.94-0.97 na
artykułach o zupełnie innych ulgach (termomodernizacyjnej, na złe
długi), bo ich szablonowe sformułowania ("wydatki niezwiązane z...")
semantycznie bliżej pasują do ogólnej parafrazy niż specyficzny
mechanizm z art. 86 ("obniżenie podatku należnego o podatek naliczony").
Potwierdzone ręcznym eksperymentem: podstawienie parafrazy używającej
DOKŁADNEGO mechanizmu z art. 86 daje rangę #1 zamiast ~#35-43.

**Naprawa:** dopisana jedna instrukcja do `QUERY_EXPANSION_SYSTEM_PROMPT`
(`prompt.py`) -- każe modelowi nie poprzestawać na ogólnym, "ładnie
brzmiącym" prawniczo sformułowaniu, tylko nazwać KONKRETNY mechanizm
prawny tak, jak robi to sama ustawa. Celowo BEZ nowego przykładu
domenowo-specyficznego dla VAT (ryzykowałoby to nadmierne dopasowanie
do jednego testu) -- tylko ogólna zasada plus istniejący, neutralny
przykład o terminie.

**Zweryfikowane** (prawdziwy `expand_query()`, nie ręczna parafraza) na
4 przypadkach, w tym oba znane jako najbardziej wrażliwe na
sformułowanie query expansion (Krok 11):
- VAT art. 86 (cel naprawy): ranga ~35 -> **#6** (w oknie top-8).
- KC art. 415 (wrażliwy na rozszerzanie, wcześniej pogorszony przez nie):
  #4 -> #5 -- minimalna, akceptowalna zmiana, wciąż bezpiecznie w top-8.
- KK art. 148 (wrażliwy na rozszerzanie, wcześniej poprawiony przez nie):
  #1 -> #1 -- bez zmian.
- KPA art. 129 (wcześniej pewny wynik #1): #1 -> #1 -- bez zmian.

Pozostałych 6 z 10 przypadków referencyjnych NIE zweryfikowano ponownie
z nowym promptem wprost (brak zapisanych dosłownych treści pytań z
wcześniejszych kroków dla części z nich) -- ale są to przypadki mniej
wrażliwe na sformułowanie parafrazy (4 z nich to zgodni zwycięzcy #1 w
obu rankingach niezależnie od parafrazy, 2 to przypadki ratowane przez
sam reranking/fuzję, nie przez dobór słów w rozszerzeniu). 115/115
testów jednostkowych bez zmian (zmiana to tylko treść stringa promptu,
nie logika). Zweryfikowane end-to-end pełną ścieżką produkcyjną (adapter
+ query expansion + fuzja): odpowiedź na pytanie o zakupy prywatne
teraz poprawnie cytuje art. 86 ust. 1 i wprost odpowiada "nie".

**Wynik: 10/10 znanych przypadków referencyjnych w top-8** (pierwszy
raz odkąd problem gęstości tematycznej i luka słownikowa query
expansion zostały odkryte, Kroki 6-13).

## Krok 15: Pilotaż Zasady 6 (cytat vs wniosek) na dziedzinie `cywilne` -- wynik niejednoznaczny, NIE wdrożony

Po naprawie VAT art. 86 (Krok 14) użytkownik zdecydował spróbować pilotażu
naprawy Zasady 6 na jednej dziedzinie (`cywilne`) przed rozciąganiem na
pozostałe 5, zgodnie z dotychczasową metodyką projektu.

**Metoda:** dopisano 7 nowych przykładów treningowych (`train.jsonl`,
20->27) i 2 nowe przykłady walidacyjne (`valid.jsonl`, 4->6) do
`data/finetune/cywilne/`, każdy wprost demonstrujący wzorzec z Zasady 6
("przepis X wprost mówi ...; Y to już wniosek/interpretacja, nie
dosłowny zapis"), na zweryfikowanych (sprawdzonych wprost w
`data/processed/kodeks_cywilny.json`) stanach faktycznych: konkubinat i
dziedziczenie (art. 931, zamknięty katalog), forma zastrzeżona umownie
bez określenia skutków (art. 76), 13-latek i drobne sprawy życia
codziennego (art. 15+20), odpowiedzialność za czyn dziecka poniżej 13
lat (art. 426+427), zadatek zależny od tego kto zawinił (art. 394 §1),
odpowiedzialność za zwierzę mimo braku nadzoru (art. 431), forma
pełnomocnictwa dla czynności wymagającej aktu notarialnego (art. 99+158)
-- plus w walidacji: termin na uchylenie się od błędu (art. 84+88) i
warunkowa zdolność dziecka poczętego do dziedziczenia (art. 927).
Przetrenowano nowy adapter (`bielik11b-cywilne-lora-rule6`, te same
hiperparametry co dotychczas: rank 8, lr 1e-5, batch 2, 100 iteracji,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia co
zawsze, najlepszy val loss na iter25 (0.717, niemal identyczny jak
oryginalny adapter cywilne: 0.716), potem 0.848 -> 0.910 -> 0.989.
Checkpoint iter25 skopiowany do osobnego katalogu
`bielik11b-cywilne-lora-rule6-iter25` -- **celowo NIE nadpisano
istniejącego domyślnego adaptera** (`bielik11b-cywilne-lora-iter25`),
żeby móc porównać oba i nie zaryzykować regresji produkcyjnej.

**Wynik porównania (stary vs nowy adapter, te same pytania, `--no-adapter`
wyłączone, `--no-expand-query`, poza zbiorem treningowym):**
- Pytanie o błąd co do autentyczności obrazu (przykład z NOWEGO
  `valid.jsonl`, więc nietrenowany wprost) -- **regresja**: stary
  adapter dał spójną, poprawną odpowiedź opartą na rękojmi (art. 568,
  inna niż zakładana w przykładzie, ale równie prawnie poprawna droga).
  Nowy adapter zaczął podobnie (rękojmia, art. 557/568), ale
  rozjechał się w nieuporządkowaną listę częściowo nietrafionych
  cytatów (art. 551, 592, 898, 169) bez klarownej konkluzji -- gorsza
  jakość, nie lepsza.
- Pytanie o kota sąsiada niszczącego ogród (nowe, spoza obu zbiorów) --
  wynik podobny jakościowo w obu wersjach, żaden nie użył wprost
  frazy "to już wniosek interpretacyjny" mimo że wzorzec ten
  wystąpił w treningu (art. 431, analogiczny przykład z psem).
- Pytanie **dosłownie z nowego `train.jsonl`** (konkubinat) -- nowy
  adapter odpowiedział poprawnie i spójnie, ale zacytował inny,
  również poprawny przepis (art. 926 zamiast wzorcowego art. 931) --
  raczej parafraza niż pamięciowe odtworzenie.

**Wniosek: pilotaż NIE dał jednoznacznej poprawy.** 7 dodatkowych
przykładów (35% wzrost zbioru, wciąż tylko 27 w sumie) nie wystarczyło,
żeby model nauczył się rzetelnie generalizować eksplicytne flagowanie
"cytat vs wniosek" na nowych stanach faktycznych -- a w jednym
sprawdzonym przypadku jakość wręcz się pogorszyła (halucynowana lista
przepisów). Możliwe przyczyny: zbyt mały zbiór żeby przesunąć
zachowanie generatywne, potrzeba więcej iteracji/wyższego rank, albo
sam wzorzec wymaga bardziej jednolitego, częściej powtarzanego
sformułowania niż 7 różnych sformułowań tej samej idei.

**Decyzja: NIE wdrożono.** Domyślny adapter `cywilne` pozostaje
niezmieniony (`bielik11b-cywilne-lora-iter25`, bez nowych przykładów w
puli treningowej użytej do JEGO treningu -- nowe przykłady leżą w
`data/finetune/cywilne/*.jsonl`, ale adapter trzeba by przetrenować
ponownie, żeby je uwzględnić, a to właśnie ten krok, którego wynik
odrzucono). Eksperymentalny adapter `bielik11b-cywilne-lora-rule6-iter25`
zostawiony na dysku do ew. dalszych prób, nie podłączony w
`chat.py`/`chat_cuda.py::DEFAULT_ADAPTER_PATHS`. 115/115 testów bez
zmian (zmiana dotyczyła tylko danych fine-tune, nie kodu).

## Krok 16: Druga próba Zasady 6 na `cywilne` z większym zbiorem -- ponownie odrzucona

Zgodnie z rekomendacją z Kroku 15 ("Co dalej"), dopisano znacznie więcej
przykładów: `train.jsonl` 27->42 (+15), `valid.jsonl` 6->9 (+3) --
w sumie 22 nowe przykłady treningowe ponad oryginalne 20 (wcześniej
tylko 7), tym samym, powtarzalnym szablonem frazowania co w Kroku 15
("Art. X wprost stanowi/wymaga ...; to już wniosek z ..., nie dosłowne
brzmienie przepisu"), na kolejnych zweryfikowanych stanach faktycznych
(poręczenie i zarzuty dłużnika, przechowanie bez wynagrodzenia,
użyczenie i zwykłe koszty utrzymania, współwłasność i rozporządzanie
rzeczą wspólną, zasiedzenie w złej wierze, odpowiedzialność importera
za produkt niebezpieczny, rękojmia i prawo sprzedawcy do naprawy,
nieważność oświadczenia woli w chorobie psychicznej, odwołanie
darowizny za rażącą niewdzięczność, odpowiedzialność solidarna
inwestora wobec podwykonawcy, przedawnienie względem małoletniego,
zapis windykacyjny, wydziedziczenie bez ustawowej przyczyny,
odpowiedzialność solidarna dłużników, świadczenie osoby trzeciej).
Przetrenowano `bielik11b-cywilne-lora-rule6-v2-iter25` tymi samymi
hiperparametrami (rank 8, lr 1e-5, batch 2, 100 iteracji,
`--save-every 25 --steps-per-eval 25`), skopiowany osobno, nie
nadpisując ani produkcyjnego adaptera, ani wariantu v1 z Kroku 15.

**Porównanie (v2 vs produkcyjny adapter, `--no-expand-query`, pytania
spoza treningu -- nie zapisano tym razem logu strat walidacyjnych z
treningu):**
- Pytanie o obraz-kopię (z `valid.jsonl`, niewidziane w gradientach) --
  v2 daje płynną narrację i poprawnie wypowiada frazę wzorca ("to już
  wniosek z zastosowania przepisu, nie dosłowne brzmienie"), ale
  **przekręca fakty**: pytanie wprost mówi, że ani kupujący, ani
  sprzedawca nie wiedzieli o kopii, a v2 uzasadnia przez rękojmię
  (art. 568/557) twierdząc, że kupujący "wiedział (lub powinien był
  wiedzieć)" -- sprzeczne z treścią pytania. Nie użył wzorcowej ścieżki
  z przykładu (art. 84 błąd + art. 88 § 2 termin roczny).
- Pytanie o kota sąsiada niszczącego warzywa (nowe, spoza obu zbiorów)
  -- stary adapter: czysta poprawna odpowiedź (art. 431). v2: też
  cytuje art. 431 poprawnie, ale **wymusza frazę wzorca w sposób
  wewnętrznie sprzeczny** ("Nie dosłowne już zastosowanie tego przepisu
  wprost -- nie wymaga dodatkowych wniosków" ... "To już wniosek z
  zastosowania przepisu, nie jego dosłowne brzmienie -- dosłowne
  brzmienie wprost wskazuje...") w stanie faktycznym, gdzie nie ma
  właściwie żadnego rozróżnienia cytat/wniosek do zaznaczenia.

**Wniosek: ponownie NIE jednoznaczna poprawa, inny rodzaj regresji niż
w Kroku 15.** Większy, bardziej jednolicie sformułowany zbiór (22
zamiast 7 przykładów) rzeczywiście nauczył model częściej *wypowiadać*
frazę wzorca -- ale powierzchownie, bez poprawy rzeczywistego
rozumowania: raz prowadzi do przekręcenia faktów pod złą podstawę
prawną, raz do wymuszonej, gramatycznie/logicznie niespójnej frazy tam,
gdzie rozróżnienie się nie stosuje. Sugeruje to, że problem nie leży
(tylko) w rozmiarze zbioru czy jednolitości frazowania, tylko głębiej
-- w tym, że 11B model z LoRA rank 8 na tak małym zbiorze uczy się
raczej powierzchniowego markera stylistycznego niż leżącej pod nim
umiejętności rozróżniania cytatu od interpretacji.

**Decyzja: ponownie NIE wdrożono.** Domyślny adapter `cywilne`
pozostaje `bielik11b-cywilne-lora-iter25`, bez zmian. Adapter
`bielik11b-cywilne-lora-rule6-v2-iter25` zostawiony na dysku, nie
podłączony. Nowe dane treningowe (`data/finetune/cywilne/*.jsonl`,
teraz 42/9) zacommitowane -- poprawne i przydatne same w sobie,
niezależnie od tego czy/kiedy Zasada 6 zostanie skutecznie
wytrenowana. 115/115 testów bez zmian.

## Krok 17: Siódma dziedzina -- gospodarcze (Kodeks spółek handlowych)

Zgodnie z "Co dalej", kolejny naturalny pilot -- prawo gospodarcze/spółki.
Wybrano Kodeks spółek handlowych (KSH, DU/2000/1037), zweryfikowane wprost
przez ELI API (`WDU20000941037`, tytuł "Ustawa z dnia 15 września 2000 r.
Kodeks spółek handlowych.") -- świadomie ostrożnie, bo pilotaż `karne`
(Krok 10) miał przypadek przypadkowego trafienia w inny, podobnie
zatytułowany akt (Kodeks karny wykonawczy) przez nieostrożne zaufanie
pierwszemu wynikowi "Tekst jednolity dla aktu". Pobrano akt (941
artykułów) i historię (6 przeszłych wersji tekstu jednolitego, 4647
artykułów), zbudowano indeks RAG z `--include-history` (7463 fragmentów
po podziale długich artykułów -- największy indeks domenowy jak dotąd).

**Nowa, ostrzejsza odsłona problemu gęstości tematycznej:** KSH
wewnętrznie duplikuje niemal identyczne przepisy dla różnych typów
spółek -- np. zakaz łączenia funkcji członka zarządu z radą nadzorczą
istnieje osobno dla sp. z o.o. (art. 214), spółki akcyjnej (art. 387) i
prostej spółki akcyjnej (art. 300⁷²), niemal słowo w słowo. Sprawdzone
na pytaniu o radę nadzorczą: art. 214 (właściwy dla sp. z o.o.) ląduje
dopiero na #4 w fuzji rang, za dwoma "rodzeństwami" z innych typów
spółek -- ale odpowiedź end-to-end (`chat.py --no-adapter`) mimo to
poprawnie identyfikuje i cytuje wszystkie trzy z właściwym przypisaniem
do typu spółki (model odczytuje nagłówek artykułu z kontekstu), więc
praktycznie niegroźne w tym przypadku -- ten sam wzorzec "reranker myli
ranking, ale generacja i tak trafia" co przy `karne` (Krok 10).

**Nowa, osobna luka znaleziona i NIE naprawiona:** pytanie sformułowane
potocznie ("czy zarząd może samodzielnie zwiększyć zadłużenie ponad
dwukrotność kapitału zakładowego") w ogóle nie trafia w art. 230 --
nawet w top-8 po cosinusie, bez rerankingu (to więc luka w recall
pierwszego etapu, nie w samym rankingu). Rozszerzanie zapytania (Krok
14) też tego nie naprawiło w izolowanym teście: wygenerowana parafraza
("zwiększenia... zadłużenia spółki bez zgody zgromadzenia wspólników")
wciąż jest zbyt ogólna, nie chwyta dokładnego mechanizmu ustawowego
("rozporządzenie prawem lub zaciągnięcie zobowiązania... dwukrotnie
przewyższającej wysokość kapitału zakładowego") -- z tą dokładną frazą
art. 230 skacze na #1 (cos 0.928). `chat.py --no-adapter` na tym
pytaniu poprawnie przyznaje się do niewiedzy zamiast zgadywać (żadnej
halucynacji) -- bezpieczne, ale niekompletne zachowanie. **Ciekawostka:**
to samo pytanie z ZAŁADOWANYM adapterem domenowym (patrz niżej) w
end-to-end teście DAJE poprawną odpowiedź z trafnym cytatem art. 230 --
`expand_query()` używa tego samego modelu+adaptera co generowanie
odpowiedzi, więc precyzja cytowania wyuczona w adapterze zdaje się
poprawiać też jakość parafrazy w rozszerzaniu zapytania. Niepotwierdzone
systematycznie (jeden przypadek, może nie być deterministyczne przy
samplingu) -- odnotowane jako obserwacja, nie jako naprawiony mechanizm.

**Osobna obserwacja bazowego modelu:** `chat.py --no-adapter` na pytaniu
o formę umowy spółki jawnej poprawnie zacytował TREŚĆ art. 23 ("na
piśmie pod rygorem nieważności"), ale POMYLIŁ numer artykułu -- podał
"art. 25" (który w rzeczywistości reguluje co innego: wymaganą TREŚĆ
umowy spółki jawnej, nie jej formę). Halucynacja numeru przy poprawnej
treści, nie błąd merytoryczny. Naprawione przez adapter (patrz niżej).

**Dane fine-tune:** 20 przykładów treningowych + 4 walidacyjne, tym
samym wzorcem co poprzednie dziedziny (cytuj konkretny artykuł, przyznaj
się do niewiedzy gdy poza zakresem KSH -- np. pytanie o stawkę CIT albo
o rejestrację jednoosobowej działalności gospodarczej w CEIDG), każdy
fakt zweryfikowany wprost w `data/processed/kodeks_spolek_handlowych.json`
przed napisaniem przykładu. Tematycznie: spółka jawna (forma umowy,
odpowiedzialność subsydiarna wspólników), spółka z o.o. (kapitał
zakładowy, forma umowy, powstanie, jedyny wspólnik, reprezentacja przez
zarząd, zakaz łączenia funkcji, uchwały wspólników, strata przewyższająca
kapitał, większość kwalifikowana), spółka akcyjna (zawiązanie, kapitał
zakładowy, odpowiedzialność członków zarządu).

Przetrenowano `bielik11b-gospodarcze-lora` tymi samymi hiperparametrami
co wszystkie poprzednie dziedziny (rank 8, lr 1e-5, batch 2, 100
iteracji, `--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec
przeuczenia: najlepszy val loss na iter25 (0.651), potem monotonicznie
gorzej (0.696 -> 0.782 -> 0.817). Checkpoint iter25 użyty jako domyślny
adapter (`bielik11b-gospodarcze-lora-iter25`), wpięty w
`DEFAULT_ADAPTER_PATHS` (`chat.py` i placeholder w `chat_cuda.py`, patrz
"Co dalej" pkt CUDA) oraz `DOMAIN_DESCRIPTIONS` w `prompt.py`
("polskiego prawa spółek handlowych").

**Weryfikacja end-to-end (z adapterem, ustawienia domyślne -- rozszerzanie
zapytania i fuzja rang włączone):**
- Pytanie o formę umowy spółki jawnej (przypadek halucynacji numeru
  artykułu z bazowego modelu, wyżej) -- adapter poprawnie cytuje art. 23
  (nie art. 25), zwięźle i trafnie tłumaczy też rolę art. 23¹.
- Pytanie o zadłużenie 2x kapitał (przypadek luki w recall, wyżej) --
  z adapterem: poprawna odpowiedź z trafnym cytatem art. 230.
- Nowe, niewidziane w treningu pytanie o czynność prawną między jedynym
  wspólnikiem-jedynym członkiem zarządu a spółką AKCYJNĄ (trening miał
  tylko analogiczny przypadek dla sp. z o.o., art. 210 § 2) -- adapter
  poprawnie uogólnia na równoległy przepis dla S.A. (art. 379 § 2),
  nie recytuje mechanicznie wytrenowanego numeru artykułu.

115/115 testów bez zmian (nowa dziedzina nie wymagała zmian w kodzie
poza wpisami w słownikach `DEFAULT_ADAPTER_PATHS`/`DOMAIN_DESCRIPTIONS`).

## Krok 18: Trzecia konwencja indeksu górnego -- naprawiony błąd numeracji w `cywilne` i `kadry_i_place`

Znaleziony przypadkiem podczas pilotażu `budowlane` (poniżej): dwa
artykuły Prawa budowlanego ("art. 5¹"/"art. 5²") kolidowały ze
zupełnie innymi, prawdziwymi artykułami o tym samym spłaszczonym
numerze ("art. 51"/"art. 52"). Diagnoza: część tekst-jednolitych PDF-ów
Kancelarii Sejmu koduje indeks górny numeru artykułu jako zwykłe
nawiasy kwadratowe w warstwie tekstowej ("Art. 5[1]." zamiast
prawdziwego mniejszego rozmiaru czcionki, którego szuka
`find_article_numbers_pdfplumber`) -- **trzecia, dotąd nieznana
konwencja** obok dwóch już naprawionych w Kroku 12/13 (rozmiar czcionki,
spacja w środku akapitu). `strip_not_yet_in_force_text` (który normalnie
zdejmuje "[ ]" z zupełnie innego powodu -- stary, wciąż ważny tekst)
spłaszcza taki nawias do gołych cyfr, które w części przypadków
KOLIDUJĄ z zupełnie innym, prawdziwym artykułem o tym samym numerze.

**Sprawdzono bezpośrednio w PDF-ach WSZYSTKICH 15 ustaw w projekcie --
wzorzec obecny w trzech:**
- **Kodeks cywilny**: ~230 wystąpień (m.in. dobrze znane art. 22¹
  "konsument", art. 43¹ "przedsiębiorca", art. 385¹ "klauzule
  niedozwolone", art. 449¹/449⁵ "odpowiedzialność za produkt
  niebezpieczny/importer", art. 981¹ "zapis windykacyjny"). Mechanizm
  korekty numeracji (`find_article_numbers_pdfplumber`) **milcząco
  wyłączał się dla całego aktu** (1080 wykrytych nagłówków vs 1296
  sparsowanych -- ogromna rozbieżność), więc żaden z tych ~230
  artykułów nigdy nie dostał poprawnego numeru. **34 z nich realnie
  kolidowały** z innym, zupełnie osobnym artykułem o tym samym
  spłaszczonym numerze.
- **Kodeks pracy**: 6 wystąpień (art. 18³ᵈ⁻ᵍ, 94³, 94³ᵃ -- przepisy
  antydyskryminacyjne/mobbing). Bez kolizji w danych aktualnych na
  moment odkrycia błędu, ale wciąż źle wyświetlany numer.
- **Prawo budowlane**: 2 wystąpienia (art. 5¹/5², oba już uchylone) --
  przypadek, który naprowadził na cały problem; OBA kolidowały z
  zupełnie innymi, prawdziwymi art. 51/52 (patrz niżej, "Ósma
  dziedzina").

Pozostałych 12 ustaw (w tym `kodeks_spolek_handlowych`, świeżo dodany w
Kroku 17) sprawdzone i wolne od tego wzorca.

**Naprawa:** rozszerzono `find_article_numbers_pdfplumber` (nie
`strip_not_yet_in_force_text` -- pierwsza, odrzucona wersja poprawki
próbowała konwertować nawiasy na prawdziwy Unicode-indeks górny PRZED
`parse_articles()`, co psuło dzielenie tekstu na artykuły, bo
`ARTICLE_SPLIT_RE` dopasowuje tylko zwykłe cyfry ASCII, nie Unicode) --
funkcja teraz rozpoznaje notację "[N]" jako ALTERNATYWNY sygnał indeksu
górnego, obok istniejącego sygnału rozmiaru czcionki, i zwraca
poprawnie zbudowany numer z prawdziwym Unicode-indeksem na tej samej
pozycji co dotychczas. Reszta pipeline'u (parsowanie, potem
POZYCYJNA korekta numeracji przez `process_act`) działa bez zmian --
to ten sam, już wielokrotnie sprawdzony mechanizm (322 poprawki w
Kroku 17 dla `kodeks_spolek_handlowych`), tylko z dodatkowym źródłem
sygnału.

**Naprawa zastosowana wstecznie:** przetworzono ponownie
`kodeks_cywilny` i `kodeks_pracy` (bieżący tekst -- 216/204 poprawki --
oraz WSZYSTKIE wersje historyczne, po 8/7 wersji, ten sam pipeline przez
`process_act_version`), przebudowano oba indeksy RAG z
`--include-history` (`rag_search.py` niezmieniony -- dane wejściowe do
niego są teraz po prostu poprawne). Zero kolizji po naprawie we
WSZYSTKICH 15 ustawach (zweryfikowane programowo).

**Znaleziono i naprawiono też 2 błędne cytaty we WŁASNYCH danych
fine-tune** dla `cywilne` (dopisanych w Kroku 16, zanim ta naprawa
istniała) -- "art. 4491"/"art. 4495" (odpowiedzialność za produkt
niebezpieczny) i "art. 9811" (zapis windykacyjny) poprawione na
"art. 449¹"/"art. 449⁵"/"art. 981¹" po zweryfikowaniu wprost w
naprawionych danych źródłowych. Znalezione automatycznym
cross-referencem (numery z nawiasowej notacji w surowym PDF-ie vs
cytaty w `data/finetune/*/*.jsonl`) -- ten sam skrypt sprawdził
`kadry_i_place` i nie znalazł tam żadnego błędnego cytatu.
**Adaptery LoRA nie wymagają retreningu**: oba błędne przykłady
pochodziły z Kroku 16 (pilotaż Zasady 6, NIE wdrożony do produkcji), a
`kadry_i_place`/`cywilne` (produkcyjne adaptery) nigdy nie trenowały na
tych konkretnych przykładach -- błąd dotyczył wyłącznie treści RAG
(cytaty w kontekście wyszukiwania) i tych dwóch nowych przykładów fine-tune,
nie wag żadnego wdrożonego adaptera.

**Zweryfikowane end-to-end** (`chat.py --no-adapter`, oba przypadki
wcześniej niemożliwe do poprawnego zacytowania): pytanie o definicję
konsumenta w `cywilne` -- poprawnie cytuje "art. 22¹" (wcześniej: "art.
221"); pytanie o obowiązek przeciwdziałania mobbingowi w
`kadry_i_place` -- poprawnie cytuje "art. 94³" (wcześniej: "art. 943").

**Metodologiczna obserwacja:** PDF-y ELI API to żywe, stale
aktualizowane dokumenty (nie statyczne migawki) -- podczas tej naprawy
zauważono, że `ordynacja_podatkowa_meta.json` też się zmieniło (nowa
nowelizacja, `changeDate` przesunięty) mimo że ta ustawa nie ma nic
wspólnego z błędem numeracji. Oznacza to, że pełne przetworzenie
(`download_acts.py` bez `--domain`) w dowolnym momencie może
przynieść drobne, niezwiązane zmiany danych źródłowych -- nie regresję,
tylko odzwierciedlenie realnych zmian w prawie. 119/119 testów
przechodzi (dodano 4 nowe w `tests/test_download_acts.py` -- regresja
wykrywania notacji nawiasowej i brak duplikatów numerów w `cywilne`/
`prawo_budowlane`).

## Krok 19: Ósma dziedzina -- budowlane (Prawo budowlane)

Zgodnie z "Co dalej", kolejny naturalny pilot. Wybrano ustawę z dnia 7
lipca 1994 r. -- Prawo budowlane (DU/1994/414), zweryfikowaną wprost
przez ELI API (`WDU19940890414`, tytuł potwierdzony przed pobraniem --
ten sam ostrożny wzorzec co przy `gospodarcze`/Krok 17). Pobrano akt
(235 artykułów), zbudowano indeks RAG (545 fragmentów po podziale;
5892 z historią, patrz niżej).

**Odkrycie sprzed tego kroku:** to właśnie podczas przygotowywania
danych fine-tune dla `budowlane` znaleziono trzecią konwencję indeksu
górnego (nawiasy kwadratowe w PDF, art. 5¹/5² kolidujące z prawdziwymi
art. 51/52) -- naprawione i opisane osobno w Kroku 18, zanim ten pilot
został dokończony.

**Znaleziony i naprawiony problem: historyczne PDF-y z szyfrowaniem
RC4 o niestandardowej długości klucza.** `download_acts_history.py
--domain budowlane` padał na najstarszej (2000 r.) z 14 przeszłych wersji
tekstu jednolitego -- `pypdf` (backend `cryptography`) odmawiał
odszyfrowania z błędem `ValueError: Invalid key size (104) for RC4`.
Backend `cryptography` waliduje długości kluczy RC4 do ograniczonego
zestawu wartości i 104-bitowy klucz (użyty w tym konkretnym,
najwyraźniej nietypowo wygenerowanym PDF-ie Kancelarii Sejmu) się nie
łapie -- mimo że sam algorytm RC4 działa z dowolną długością klucza, to
nadgorliwa walidacja tego konkretnego backendu, nie realne ograniczenie
formatu PDF. Sprawdzono `pycryptodome` jako alternatywny backend `pypdf`
(obsługuje dowolne długości kluczy RC4) -- zainstalowany tymczasowo,
ale `pypdf` i tak preferuje `cryptography`, gdy oba pakiety są dostępne
w środowisku, więc błąd nie znikał; odinstalowany z powrotem (żadnej
trwałej zmiany we współdzielonym `.venv`, patrz "Shared infra" w
pamięci projektu -- ryzyko niepotrzebnej zmiany współdzielonej z
siostrzanym projektem bez realnej korzyści). Zamiast wymuszać cały
backend `pypdf` na `pycryptodome` (wymagałoby odinstalowania
`cryptography`, używanej też gdzie indziej w projekcie -- zbyt
ryzykowne dla współdzielonego środowiska), zastosowano precyzyjniejszą
naprawę wewnątrz `download_acts.py`: podmieniono WYŁĄCZNIE implementację
RC4 (`pypdf._encryption.CryptRC4`) na czysto pythonowy fallback z
`pypdf._crypt_providers._fallback` (identyczny algorytm, bez walidacji
długości klucza) -- szyfrowanie AES nadal korzysta z `cryptography`,
nienaruszone. Zero zmian w zależnościach współdzielonego `.venv`.
Zweryfikowano bezpośrednio na feralnym pliku (128 artykułów odzyskanych
z wersji DU/2000/1126), potem pełny przebieg wszystkich 14 wersji
historycznych (2461 artykułów zapisanych) i przebudowa indeksu RAG z
`--include-history` (2696 artykułów, 5892 fragmentów). `/data`/`--as-of`
dla `budowlane` teraz działa identycznie jak dla pozostałych 7 dziedzin
-- zweryfikowane bezpośrednio (`rag_search.py --as-of 2002-01-01`
zwraca fragmenty oznaczone `[stan: ... -- 2006-08-18]` zamiast bieżącej
wersji). 119/119 testów bez zmian (poprawka nie zmienia zachowania dla
żadnego innego, wcześniej działającego PDF-u -- identyczny algorytm,
inna implementacja).

**Dane fine-tune:** 19 przykładów treningowych + 4 walidacyjne, ten sam
wzorzec co poprzednie dziedziny, każdy fakt zweryfikowany wprost w
`data/processed/prawo_budowlane.json` przed napisaniem przykładu.
Tematycznie: zakres ustawy i definicje (art. 1, 3, 4), proces
inwestycyjny (zgłoszenie vs pozwolenie na budowę -- art. 29/30/33,
wygaśnięcie pozwolenia -- art. 37, rozpoczęcie budowy -- art. 41),
kierownik budowy i uprawnienia budowlane (art. 12/13/22/42), nadzór
budowlany i samowola budowlana (art. 48), użytkowanie i kontrole
okresowe (art. 54/62), odpowiedzialność karna (art. 91), oraz jeden
przykład odmowy (stawki podatku od nieruchomości -- poza zakresem tej
ustawy). Zbiór walidacyjny w stylu `gospodarcze`/Krok 17 -- pytania
scenariuszowe wymagające zastosowania przepisu, nie tylko jego
zacytowania (przerwa w budowie > 3 lata, uznaniowe zwolnienie z
obowiązku kierownika budowy, zależność dziennika budowy od obowiązku
ustanowienia kierownika, plus jeden przykład odmowy -- szczegółowe
normy BHP na wysokości, poza zakresem samej ustawy).

Przetrenowano `bielik11b-budowlane-lora` tymi samymi hiperparametrami co
wszystkie poprzednie dziedziny (rank 8, lr 1e-5, batch 2, 100 iteracji,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia:
najlepszy val loss na iter25 (0.842), potem monotonicznie gorzej (0.996
-> 1.042 -> 1.067). Checkpoint iter25 użyty jako domyślny adapter
(`bielik11b-budowlane-lora-iter25`), wpięty w `DEFAULT_ADAPTER_PATHS`
(`chat.py` i placeholder w `chat_cuda.py`) oraz `DOMAIN_DESCRIPTIONS` w
`prompt.py` ("polskiego prawa budowlanego").

**Znaleziony, nienaprawiony problem: luka w recall + błędna reformulacja
zapytania.** Pytanie sformułowane potocznie ("mam duże doświadczenie
budowlane, czy muszę mimo to zatrudnić kierownika budowy") nie trafia w
art. 42 (właściwa podstawa obowiązku ustanowienia kierownika budowy) w
top-8 nawet PO rozszerzeniu zapytania -- gorzej, `expand_query()` z
załadowanym adapterem samo w sobie generuje reformulację, która myli
ten przepis z zupełnie innym tematem (art. 12/13, uprawnienia budowlane
do pełnienia funkcji technicznych) -- oba tematy pojawiają się w danych
treningowych (art. 42 w jednym przykładzie, art. 12/13 w dwóch), ale
adapter nie generalizuje poprawnie rozróżnienia między "czy muszę mieć
kierownika" (art. 42) a "jakie kwalifikacje musi mieć kierownik" (art.
12/13) przy reformulacji tego konkretnego, nowego pytania. Skutek
end-to-end: odpowiedź z adapterem błędnie przypisuje obowiązek
ustanowienia kierownika budowy do art. 45 (który w rzeczywistości
dotyczy dziennika budowy, nie samego obowiązku) -- poprawny wniosek
("tak, musi Pan mieć kierownika"), błędna podstawa prawna. **Ciekawostka:**
`chat.py --no-adapter` na TYM SAMYM pytaniu odpowiada poprawnie (art. 42
ust. 2 -- drobna pomyłka co do ustępu, powinno być ust. 1, ale poprawny
artykuł i trafnie wspomniany wyjątek z ust. 3) -- ten sam wzorzec co w
Kroku 17 (adapter czasem pogarsza odpowiedź, w której model bazowy już
miał rację, prawdopodobnie kosztem specjalizacji w cytowaniu innych,
częściej trenowanych przepisów). Ten konkretny przypadek NIE jest
pokryty żadnym przykładem treningowym wprost (przykład 10 pyta o coś
podobnego, ale innymi słowami: "czy mogę sam nadzorować prace" zamiast
"mam doświadczenie, czy muszę zatrudnić") -- podobny charakter do luki
VAT art. 86 (Krok 7/12) i KSH art. 230 (Krok 17): nie blokujące (model
nie halucynuje kompletnie fałszywej odpowiedzi, sam wniosek jest
poprawny), ale warte obserwacji i potencjalnie wymaga wzmocnienia
instrukcji rozszerzania zapytania (Krok 14) o rozróżnianie "obowiązek
X" od "kwalifikacje do X" przy bliskich tematycznie przepisach.

**Weryfikacja end-to-end (z adapterem, ustawienia domyślne):**
- Pytanie o przerwę w budowie > 3 lata (nowe, nie literalnie w danych
  treningowych -- przykład 7 dotyczy nierozpoczęcia budowy, nie
  przerwania już rozpoczętej) -- adapter poprawnie uogólnia na drugą
  przesłankę art. 37 ust. 1 i poprawnie cytuje art. 37 ust. 2 dot.
  konieczności nowej decyzji.
- Pytanie o samowolę budowlaną sąsiada (analogiczne do przykładu 11) --
  adapter poprawnie cytuje art. 48 ust. 1-4 oraz art. 49e (skutek braku
  legalizacji), ze szczegółami wykraczającymi poza sam przykład
  treningowy.
- Pytanie o obowiązek kierownika budowy przy dużym doświadczeniu
  inwestora (opisane wyżej) -- błędna podstawa prawna (art. 45 zamiast
  art. 42), poprawny wniosek.

119/119 testów bez zmian (nowa dziedzina nie wymagała zmian w kodzie
poza wpisami w słownikach `DEFAULT_ADAPTER_PATHS`/`DOMAIN_DESCRIPTIONS`).

## Krok 20: Dziewiąta dziedzina -- autorskie (prawo autorskie i prawa pokrewne)

Zgodnie z "Co dalej", kolejny naturalny pilot. Wybrano ustawę z dnia 4
lutego 1994 r. o prawie autorskim i prawach pokrewnych (DU/1994/83),
zweryfikowaną wprost przez ELI API (`WDU19940240083`, tytuł potwierdzony
przed pobraniem). Pobrano akt (215 artykułów -- 92 poprawki numeracji
indeksu górnego, najwięcej ze wszystkich dziewięciu dziedzin, bo ta
ustawa gęsto korzysta z notacji w rodzaju art. 6¹/33³/47¹), pobrano też
historię od razu (7 wersji, 1280 artykułów -- 3 najstarsze, sprzed 2018
r., pominęły korektę indeksu górnego przez istniejący bezpieczny
mechanizm "liczba nagłówków != liczba artykułów", patrz Krok 18), i
zbudowano indeks RAG z `--include-history` (1495 artykułów, 2211
fragmentów) od razu przy pierwszym budowaniu -- używając naprawy z Kroku
19, ta dziedzina nigdy nie miała okresu "bez historii".

**Znaleziony przypadek indeksu górnego NA POZIOMIE USTĘPU, nie
artykułu** (obserwacja, nie naprawiony -- inna kategoria niż Krok 18):
art. 1 ust. 2¹ (definicja: ochronie podlega wyłącznie sposób wyrażenia,
nie idee/metody/koncepcje matematyczne) jest w danych źródłowych
spłaszczony do zwykłego tekstu "21." wewnątrz artykułu, WYMIESZANY
dodatkowo z fragmentem przypisu o dyrektywach UE (artefakt PDF -- numer
porządkowy przypisu "1)" koliduje wizualnie z numeracją punktów listy
w tym samym akapicie). Istniejący mechanizm korekty numeracji
(`find_article_numbers_pdfplumber`, naprawiony w Kroku 18) dotyczy
WYŁĄCZNIE numerów ARTYKUŁÓW, nie numerów USTĘPÓW wewnątrz artykułu --
to więc osobna, węższa kategoria tego samego zjawiska, poza zakresem
dotychczasowej naprawy. W tym konkretnym przypadku nie powoduje kolizji
(art. 1 ma tylko 4 ustępy, nie ma prawdziwego "ust. 21"), więc dane
źródłowe technicznie się nie psują -- ale cytowanie tego ustępu w
danych fine-tune wymagało ręcznej weryfikacji poprawnego zapisu
("ust. 2¹", nie "ust. 21") wprost z wiedzy o tej ustawie, nie tylko z
automatycznie sparsowanego JSON-a. Nie naprawione -- brak dowodu na
realną kolizję gdziekolwiek w projekcie, ten sam "czekaj na więcej
przypadków" wzorzec co przy nienaprawionych jeszcze problemach gęstości
tematycznej rerankingu.

**Dane fine-tune:** 20 przykładów treningowych + 4 walidacyjne, ten sam
wzorzec co poprzednie dziedziny, każdy fakt zweryfikowany wprost w
`data/processed/prawo_autorskie.json` przed napisaniem przykładu.
Tematycznie: definicja utworu i wyłączenia (art. 1, 4), autorstwo i
współautorstwo (art. 8, 9), utwór pracowniczy (art. 12), prawa osobiste
i majątkowe (art. 16, 17), dozwolony użytek (art. 23, 27, 29, 34), czas
trwania ochrony (art. 36), umowy o przeniesienie praw (art. 41, 52,
58), programy komputerowe (art. 74), ochrona po śmierci twórcy (art.
78), plagiat jako przestępstwo (art. 115), plus jeden przykład odmowy
(rejestracja znaku towarowego -- poza zakresem, to prawo własności
przemysłowej). Zbiór walidacyjny w stylu `gospodarcze`/`budowlane` --
pytania scenariuszowe (freelancer na umowie o dzieło vs. pracownik,
wystawa charytatywna cudzego obrazu, cytat całego krótkiego wiersza),
plus jeden przykład odmowy (50% koszty uzyskania przychodu w PIT --
poza zakresem, to prawo podatkowe).

Przetrenowano `bielik11b-autorskie-lora` tymi samymi hiperparametrami co
wszystkie poprzednie dziedziny (rank 8, lr 1e-5, batch 2, 100 iteracji,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia:
najlepszy val loss na iter25 (0.773), potem gorzej (0.814 -> 0.886 ->
0.841, z lekkim spadkiem na iter100 względem iter75, ale wciąż wyraźnie
gorzej niż iter25). Checkpoint iter25 użyty jako domyślny adapter
(`bielik11b-autorskie-lora-iter25`), wpięty w `DEFAULT_ADAPTER_PATHS`
(`chat.py` i placeholder w `chat_cuda.py`) oraz `DOMAIN_DESCRIPTIONS` w
`prompt.py` ("polskiego prawa autorskiego").

**Luka w recall, ale rozwiązana end-to-end (podobny wzorzec do
`gospodarcze`/Krok 17):** pytanie o to, komu przysługują prawa autorskie
do logo zamówionego przez freelancera na umowie o dzieło (nie o pracę)
nie trafia w żaden z kluczowych przepisów (art. 12, 17, 41) w top-8
nawet bez rerankingu, samym kosinusem -- gęsty temat "umowy/wynagrodzenie
twórcy" (art. 41-70) przyciąga zbyt wiele tematycznie bliskich, ale
niewłaściwych artykułów (art. 60, 43, 47¹, 70, 18...). Mimo to pełny
`chat.py` (z adapterem, domyślnym rozszerzaniem zapytania i fuzją rang)
odpowiedział W PEŁNI POPRAWNIE: trafnie wykluczył art. 12 (dotyczy tylko
umowy o pracę), poprawnie wskazał, że prawa domyślnie pozostają przy
twórcy, i poprawnie zasygnalizował wymóg wymienienia pól eksploatacji
(art. 41/2). Nie zbadano głębiej który dokładnie mechanizm (rozszerzenie
zapytania czy sam adapter) odpowiada za sukces -- czwarty z rzędu
przypadek tego wzorca (po VAT art. 86 przed naprawą, KSH art. 230,
budowlane art. 42/45), tym razem bez widocznej szkody w praktyce.

**Weryfikacja end-to-end (z adapterem, ustawienia domyślne):**
- Pytanie o freelancera i logo (opisane wyżej) -- w pełni poprawne,
  precyzyjne rozróżnienie umowy o pracę od umowy o dzieło.
- Pytanie o bezpłatną wystawę charytatywną cudzego obrazu -- poprawnie
  cytuje art. 32 ust. 1 (warunek braku korzyści majątkowych) ORAZ,
  spontanicznie, art. 33³ (reklama publicznie dostępnej wystawy) --
  artykuł spoza zbioru treningowego, zweryfikowany bezpośrednio jako
  trafny i prawdziwy (nie halucynacja) po tym teście.
- Nowe, niewidziane w treningu pytanie o pomysł na algorytm sortowania
  (analogiczne do przykładu treningowego o fabule powieści, ale inny
  kontekst) -- adapter poprawnie uogólnia na art. 1 ust. 2¹ z poprawnym
  zapisem indeksu górnego.

119/119 testów bez zmian.

## Krok 21: Naprawa (i odkrycie prawdziwej przyczyny) luki w rozszerzaniu zapytania -- LoRA wyłączana na czas ekspansji

Zgodnie z decyzją, żeby zbadać nawracający wzorzec "obowiązek X myli się z
pokrewnym, ale odrębnym zagadnieniem" (VAT art. 86 przed Krokiem 14, KSH
art. 230, `budowlane` art. 42/45, `autorskie` -- patrz Kroki 17/19/20).
Punktem wyjścia był jedyny z tych przypadków z REPRODUKOWALNYM, nie
"czasem działającym" end-to-end skutkiem: pytanie o obowiązek zatrudnienia
kierownika budowy (`budowlane`), gdzie adapter kończył z błędnym cytatem
(art. 45 zamiast art. 42).

**Pierwsza próba (nieskuteczna):** wzmocniono `QUERY_EXPANSION_SYSTEM_PROMPT`
o wyraźny zakaz cytowania numerów artykułów w przeformułowaniu oraz
instrukcję "zachowaj dokładny przedmiot pytania" (nie myl obowiązku z
kwalifikacjami/warunkami pokrewnego zagadnienia). Test na dokładnym
przypadku `budowlane`: ZERO zmiany w wygenerowanym przeformułowaniu --
model z adapterem nadal zaczynał od "Zgodnie z art. 42 ust. 1 pkt 2 lit.
a...", po czym podstawiał pod ten (poprawny) numer TREŚĆ zupełnie innego
przepisu (art. 12/13, uprawnienia budowlane). Instrukcja w promptcie
systemowym była więc całkowicie ignorowana.

**Prawdziwa przyczyna, znaleziona przez izolację zmiennej:** to samo
wywołanie `expand_query()`, na tym samym pytaniu, z TYM SAMYM nowym
promptem, ale BEZ załadowanego adaptera -- działa bezbłędnie: czyste
przeformułowanie bez cytatu, art. 42 wskakuje do top-3 w wyszukiwaniu.
Wniosek: to nie prompt był za słaby, tylko silny, wyuczony nawyk adaptera
domenowego (zawsze zaczynaj odpowiedź cytatem "Zgodnie z art. X...")
"przecieka" z generowania właściwej odpowiedzi do zupełnie innego
wywołania -- ekspansji zapytania -- mimo wyraźnie odmiennego promptu
systemowego dla tego wywołania. LoRA zmienia rozkład prawdopodobieństwa
tokenów na tyle silnie, że żadna wersja instrukcji w promptcie tego nie
przełamuje.

**Naprawa:** `chat.py` teraz WYŁĄCZA warstwy LoRA na czas wywołania
`expand_query()` i włącza je z powrotem przed właściwą generacją
odpowiedzi -- `find_lora_layers()` (wywoływane raz, zaraz po `load()`)
znajduje wszystkie moduły `LoRALinear` w załadowanym modelu;
`set_lora_enabled()` podmienia je w miejscu na ich bazowe `.linear` i z
powrotem (funkcje `remove_lora_layers`/`load_adapters` z
`mlx_lm.tuner.utils`/`mlx_lm.tuner.lora` -- bez ponownego wczytywania
wag z dysku, tylko przełączanie już wczytanych modułów). Ekspansja
zapytania jest więc teraz ZAWSZE wykonywana bazowym modelem, niezależnie
od tego, jaki adapter (jeśli w ogóle) jest używany do właściwej
odpowiedzi.

**Zweryfikowane -- naprawia `budowlane`:** to samo pytanie o kierownika
budowy przez pełny `chat.py` (z adapterem, domyślne ustawienia) teraz
poprawnie cytuje art. 42 (wcześniej: art. 45) -- pierwsza naprawa tej
konkretnej, potwierdzonej wcześniej usterki.

**Zweryfikowane -- brak regresji na "twardych" przypadkach referencyjnych:**
KPA art. 129 (`administracyjne`), KK art. 148 (`karne`), VAT art. 86
(`podatkowe`, przypadek naprawiony w Kroku 14 -- działa nadal, bo ta
naprawa opierała się na samym promptcie, nie na pamięci adaptera),
`autorskie` (przypadek freelancera/logo z Kroku 20) -- wszystkie bez
zmian, poprawne.

**Znaleziony i udokumentowany kompromis -- regresja "szczęśliwego trafu"
dla `gospodarcze` art. 230:** Krok 17 udokumentował, że pytanie o
"zwiększenie zadłużenia ponad dwukrotność kapitału" nie trafia w art.
230 w IZOLOWANYM teście retrievalu, ale end-to-end z adapterem DAWAŁO
poprawną odpowiedź -- WYRAŹNIE zaznaczone wtedy jako "niepotwierdzone
systematycznie, obserwacja, nie naprawiony mechanizm". Zbadano
bezpośrednio: ten "szczęśliwy traf" wynikał z tego, że adapter podczas
ekspansji zapytania DOSŁOWNIE PRZYTACZAŁ zapamiętaną z treningu treść
art. 230 (nie tylko numer) -- ta treść, będąc niemal identyczna z
prawdziwym tekstem przepisu, trafiała w wyszukiwaniu lepiej niż czysta,
zbyt ogólna parafraza bazowego modelu. Po naprawie (LoRA wyłączona na
czas ekspansji) ten mechanizm znika -- zapytanie o art. 230 wraca do
udokumentowanego już wcześniej stanu "luka w recall pierwszego etapu"
(sam bazowy model, bez pamięciowego wsparcia adaptera, generuje zbyt
ogólną parafrazę: "zwiększenie zadłużenia przez zarząd", zamiast
dokładnego mechanizmu ustawowego). **To nie jest nowa regresja
jakiejkolwiek zagwarantowanej funkcji** -- to utrata niezamierzonego,
niedeterministycznego efektu ubocznego pamięciowego, który sam Krok 17
wyraźnie odradzał traktować jako naprawiony. W zamian: zachowanie
`expand_query()` jest teraz przewidywalne i spójne między wszystkimi
dziedzinami (zawsze bazowy model), zamiast zależeć od tego, czy dany
adapter akurat "pamięta" właściwą odpowiedź na to konkretne pytanie.

119/119 testów bez zmian (zmiana dotyczy tylko `chat.py`/`prompt.py`,
żaden test nie ładuje pełnego modelu MLX -- weryfikacja wyłącznie
end-to-end, ręczna, jak opisano wyżej).

## Krok 22: Naprawa luki w recall dla `gospodarcze` art. 230 -- prawdziwa, nie tylko maskowana

Skoro Krok 21 zdjął przypadkowe maskowanie tej luki (patrz tam), pierwszy
raz od Kroku 17 mieliśmy w pełni deterministyczny, powtarzalny przypadek
do naprawy tą samą metodą co Krok 14 (VAT art. 86): znaleźć ogólną
instrukcję dla `QUERY_EXPANSION_SYSTEM_PROMPT`, nie punktową łatkę.

**Diagnoza:** porównanie wygenerowanego przeformułowania ("Pytanie
dotyczy możliwości samodzielnego zwiększenia zadłużenia przez zarząd
spółki...") z dokładnym brzmieniem art. 230 ("Rozporządzenie prawem lub
zaciągnięcie zobowiązania do świadczenia... przewyższającej wysokość
kapitału zakładowego") pokazało dokładnie brakujący skok: model
zostawiał potoczne "zadłużenie" bez tłumaczenia na ścisły cywilnoprawny
odpowiednik "zaciągnięcie zobowiązania (do świadczenia)". Ręcznie
napisane warianty zapytania z tym właśnie terminem (ale swobodnie
sformułowane jako pytanie, NIE dosłowny cytat ustawy) trafiały w art.
230 na #1 -- czyli problem nie leżał w potrzebie dosłownego cytatu, tylko
w konkretnym słowie kluczowym.

**Dwie nieudane próby, jedna udana:**
1. Ogólna instrukcja "zwróć uwagę na potoczne słownictwo ekonomiczne,
   znajdź ścisły odpowiednik" BEZ konkretnego przykładu pary słów -- ZERO
   efektu, model nadal zostawiał "zadłużenie" bez zmian.
2. Ta sama instrukcja z LISTĄ potocznych słów obok LISTY prawnych
   terminów (bez sparowania 1:1) -- również ZERO efektu.
3. Ta sama instrukcja z DWIEMA konkretnymi, sparowanymi przykładami w
   formacie identycznym jak już istniejący, działający przykład Kroku 14
   ("zamiast X napisz Y"): "zamiast 'zwiększyć zadłużenie firmy' napisz
   'zaciągnięcie zobowiązania do świadczenia'" oraz "zamiast 'stracić
   pieniądze na inwestycji' napisz 'poniesienie straty'" -- ZADZIAŁAŁO:
   przeformułowanie zaczęło poprawnie używać "zaciągnięcie zobowiązania",
   art. 230 wskoczył z poza top-8 na #3 w pełnym potoku (rerank + fuzja).

**Kompromis, nazwany wprost:** ta naprawa jest bliżej punktowej łatki niż
Krok 14 (jeden z dwóch przykładów, "zwiększyć zadłużenie firmy", jest w
praktyce prawie dosłownym opisem sytuacji z art. 230) -- w odróżnieniu od
uniwersalnego "ile mam czasu na coś" z Kroku 14, które dotyczy dowolnego
terminu proceduralnego w dowolnej dziedzinie. Świadoma decyzja: dodano
DWA przykłady tej samej klasy (dług/zobowiązanie i strata/inwestycja),
nie jeden, żeby instrukcja uczyła kategorii zjawiska ("potoczny termin
finansowy -> ścisłe zobowiązanie/roszczenie cywilnoprawne"), a nie tylko
jednego zapamiętanego podstawienia -- ale to wciąż węższa naprawa niż
Krok 14, i zasługuje na obserwację, czy faktycznie generalizuje na inne,
niewidziane dotąd pytania z podobnym słownictwem finansowym w przyszłych
dziedzinach (np. prawo upadłościowe, jeśli kiedyś dojdzie).

**Zweryfikowane -- naprawia cel:** `chat.py --domain gospodarcze` na
dokładnym pytaniu z Kroku 17 teraz poprawnie i w pełni cytuje art. 230
(treść zgodna z ustawą, nie tylko sam numer).

**Zweryfikowane -- brak regresji:** VAT art. 86 (`podatkowe`), KPA art.
129 (`administracyjne`), KK art. 148 (`karne`), `budowlane` (kierownik
budowy, Krok 21), `autorskie` (freelancer/logo, Krok 20) -- wszystkie
bez zmian, poprawne.

119/119 testów bez zmian (zmiana dotyczy tylko `prompt.py` -- ten sam
zakres testowalności co Krok 21).

## Krok 23: Dziesiąta dziedzina -- wlasnosc_przemyslowa (Prawo własności przemysłowej)

Zgodnie z "Co dalej", kolejny naturalny pilot -- naturalne uzupełnienie
`autorskie` w stronę pozostałych praw własności intelektualnej. Wybrano
ustawę z dnia 30 czerwca 2000 r. Prawo własności przemysłowej
(DU/2001/508), zweryfikowaną wprost przez ELI API (`WDU20010490508`,
tytuł potwierdzony przed pobraniem). Pobrano akt (412 artykułów -- 92
poprawki numeracji indeksu górnego, druga najgęstsza notacja po
`autorskie`, np. art. 129¹/132¹ zastępujące uchylone art. 129/132 --
reforma unijna znaków towarowych z 2019 r.), pobrano historię od razu (5
wersji, 1991 artykułów) i zbudowano indeks RAG z `--include-history`
(2403 artykuły, 3283 fragmenty).

**Dane fine-tune:** 20 przykładów treningowych + 4 walidacyjne, ten sam
wzorzec co poprzednie dziedziny, każdy fakt zweryfikowany wprost w
`data/processed/prawo_wlasnosci_przemyslowej.json` przed napisaniem
przykładu. Tematycznie: zakres ustawy (art. 1), przesłanki zdolności
patentowej (art. 24), prawo do patentu -- w tym wynalazek pracowniczy i
zbywalność tego prawa przed uzyskaniem patentu (art. 11, 20), treść i
czas trwania patentu (art. 63, 66 -- 20 lat), wzór użytkowy (art. 94,
100 -- 10 lat), wzór przemysłowy (art. 102, 105 -- 25 lat w okresach
pięcioletnich), znak towarowy -- definicja, przeszkody bezwzględne i
względne, czas trwania z bezterminowym przedłużaniem, pojęcie używania,
zbywalność (art. 120, 129¹, 132¹, 153, 154, 162), roszczenia z tytułu
naruszenia patentu i znaku towarowego (art. 287, 296) oraz
odpowiedzialność karna za obrót podróbkami (art. 305), plus jeden
przykład odmowy (czas ochrony prawa autorskiego do powieści -- poza
zakresem, to `autorskie`, nie `wlasnosc_przemyslowa`). Zbiór walidacyjny
w stylu poprzednich dziedzin -- pytania scenariuszowe wymagające
zastosowania i ZESTAWIENIA kilku przepisów (przestrzenny znak towarowy
vs. wzór przemysłowy dla tego samego kształtu opakowania; wynalazek
pracowniczy na umowie zlecenie, nie tylko o pracę -- świadomie inny
wariant niż `autorskie`'s art. 12, żeby sprawdzić czy adapter generalizuje
regułę "zamawiający" z art. 11 ust. 3, a nie tylko przypadek zatrudnienia;
wzór użytkowy jako alternatywa dla wynalazku bez poziomu wynalazczego),
plus jeden przykład odmowy (50% koszty uzyskania przychodu w PIT -- poza
zakresem, to prawo podatkowe).

Przetrenowano `bielik11b-wlasnosc-przemyslowa-lora` tymi samymi
hiperparametrami co wszystkie poprzednie dziedziny (rank 8, lr 1e-5,
batch 2, 100 iteracji, `--save-every 25 --steps-per-eval 25`) -- ten sam
wzorzec przeuczenia: najlepszy val loss na iter25 (0.823), potem
monotonicznie gorzej (0.917 -> 0.970 -> 1.033). Checkpoint iter25 użyty
jako domyślny adapter (`bielik11b-wlasnosc-przemyslowa-lora-iter25`),
wpięty w `DEFAULT_ADAPTER_PATHS` (`chat.py` i placeholder w
`chat_cuda.py`) oraz `DOMAIN_DESCRIPTIONS` w `prompt.py` ("polskiego
prawa własności przemysłowej").

**Weryfikacja end-to-end (z adapterem, ustawienia domyślne):**
- Pytanie o przestrzenny znak towarowy dla kształtu opakowania --
  poprawnie cytuje art. 120 ust. 1-2, trafnie wyjaśnia wymóg zdolności
  odróżniającej.
- Pytanie o wynalazek pracowniczy na umowie zlecenie (nie o pracę) --
  adapter poprawnie uogólnia art. 11 ust. 3 na "zamawiającego", nie tylko
  "pracodawcę" -- nie recytuje mechanicznie wytrenowanego wariantu
  zatrudnienia na etacie.
- Pytanie o wygaśnięcie ochrony znaku towarowego po 8 latach -- poprawnie
  cytuje art. 153 ust. 2-3 (10 lat, bezterminowe przedłużanie), oraz
  SPONTANICZNIE, poprawnie art. 169 (wygaśnięcie z powodu nieużywania
  przez 5 lat) -- artykuł spoza zbioru treningowego, zweryfikowany
  bezpośrednio jako trafny po tym teście, nie halucynacja.

Nie znaleziono żadnej nowej luki w recall ani w numeracji przy tym
pilotażu -- pierwsza dziedzina od Kroku 17, w której weryfikacja
end-to-end nie ujawniła żadnego nowego problemu.

119/119 testów bez zmian (nowa dziedzina nie wymagała zmian w kodzie
poza wpisami w słownikach `DEFAULT_ADAPTER_PATHS`/`DOMAIN_DESCRIPTIONS`).

## Krok 24: Jedenasta dziedzina -- ochrona_danych (ustawa o ochronie danych osobowych)

Zgodnie z "Co dalej", pierwszy pilot poza kodeksami/ustawami czysto
krajowymi -- ochrona danych osobowych, gdzie materialne zasady
(zgoda, podstawy przetwarzania, prawa osoby, wysokość kar) reguluje
bezpośrednio RODO (rozporządzenie UE 2016/679), a NIE polska ustawa.
Przed startem sprawdzono wprost przez ELI API (`WDU20180001000`, Dz.U.
2018 poz. 1000, tytuł "Ustawa z dnia 10 maja 2018 r. o ochronie danych
osobowych" potwierdzony przed pobraniem), czy krajowa ustawa
wdrażająca ma wystarczającą samodzielną treść merytoryczną, żeby w ogóle
uzasadniać osobną dziedzinę: TAK -- ustawa realnie reguluje całą
krajową warstwę instytucjonalną, której RODO samo nie ustala (wyznaczanie
IOD, akredytacja i certyfikacja, pozycja i powoływanie Prezesa UODO,
tryb postępowania w sprawie naruszeń, kontrola, odpowiedzialność cywilna
i tryb sądowy, administracyjne kary pieniężne -- KTO je nakłada, karne).

Pobrano akt (131 artykułów rdzenia merytorycznego z tekstu jednolitego --
rozdziały 12-14, art. 109-176, to głównie nowelizacje innych ustaw i
przepisy przejściowe, analogicznie do "ogona" w innych dziedzinach; 0
wersji historycznych do pobrania -- ustawa nigdy nie miała osobnego
obwieszczenia o tekście jednolitym poza bieżącym). Zbudowano indeks RAG
z `--include-history` od razu (174 fragmenty).

**Dane fine-tune:** 20 przykładów treningowych + 4 walidacyjne, każdy
fakt zweryfikowany wprost w `data/processed/all_articles.json` przed
napisaniem przykładu. Tematycznie: zakres ustawy i relacja do RODO
(art. 1), wyłączenia stosowania (służby specjalne -- art. 6),
obowiązek wyznaczenia IOD dla enumeratywnie wskazanych podmiotów
publicznych (art. 8-9), zawiadomienie i publikacja danych IOD (art.
10-11), akredytacja przez Polskie Centrum Akredytacji (art. 12),
certyfikacja przez Prezesa Urzędu lub podmiot certyfikujący (art. 15),
wyjątek od obowiązku informacyjnego dla zadań publicznych (art. 5),
pozycja ustrojowa, zastępcy, niepołączalność stanowisk i immunitet
Prezesa Urzędu (art. 34, 36-38), jednoinstancyjność postępowania +
KPA posiłkowo (art. 7), udział organizacji społecznej (art. 61), kto
przeprowadza kontrolę, w tym współpraca transgraniczna UE (art.
78-79), właściwość sądu okręgowego i KC posiłkowo dla roszczeń
cywilnych (art. 92-93), powództwo Prezesa Urzędu na rzecz
poszkodowanego (art. 98), KTO nakłada administracyjną karę pieniężną
i na jakiej podstawie -- z wyraźnym rozróżnieniem, że samą WYSOKOŚĆ
ustala RODO art. 83, nie ta ustawa (art. 101), obniżone pułapy
kwotowe dla sektora finansów publicznych (art. 102), przestępstwa
nielegalnego przetwarzania (podstawowy i kwalifikowany typ dla danych
szczególnych kategorii) oraz utrudniania kontroli (art. 107-108).
Zbiór walidacyjny celowo testuje właśnie granicę ustawa/RODO: czy
mały przedsiębiorca ma ustawowy obowiązek wyznaczenia IOD (nie wprost
-- to zależy od przesłanek z RODO art. 37, nie z tej ustawy), jaki
pułap kary grozi jednostce sektora finansów publicznych, właściwość
sądu dla roszczenia o dane wrażliwe, oraz explicite przykład odmowy:
maksymalna wysokość kary administracyjnej dla ZWYKŁEJ firmy (nie z
sektora publicznego) w euro/procencie obrotu -- to ustala wyłącznie
RODO art. 83, nie ta ustawa, więc odpowiedź to świadome przekierowanie,
nie zgadywanie.

Przetrenowano `bielik11b-ochrona-danych-lora` tymi samymi
hiperparametrami co wszystkie poprzednie dziedziny (rank 8, lr 1e-5,
batch 2, 100 iteracji, `--save-every 25 --steps-per-eval 25`) -- ten sam
wzorzec przeuczenia: najlepszy val loss na iter25 (0.926), potem
niemonotonicznie, ale konsekwentnie gorzej (1.007 → 1.100 → 1.131).
Checkpoint iter25 użyty jako domyślny adapter
(`bielik11b-ochrona-danych-lora-iter25`), wpięty w `DEFAULT_ADAPTER_PATHS`
(`chat.py` i placeholder w `chat_cuda.py`) oraz `DOMAIN_DESCRIPTIONS` w
`prompt.py` -- celowo opisana jako "polskiej ustawy o ochronie danych
osobowych", NIE "polskiego prawa ochrony danych osobowych", żeby nie
sugerować modelowi szerszej wiedzy merytorycznej (RODO), której nie ma w
indeksie tej dziedziny.

**Weryfikacja end-to-end (z adapterem, ustawienia domyślne):**
- Kto wyznacza IOD i w jakim terminie zgłasza się to Prezesowi Urzędu --
  poprawnie cytuje art. 8, 10 ust. 1-2, oraz SPONTANICZNIE poprawnie
  art. 10 ust. 5-6 (odrębne zawiadomienia grup podmiotów, wymóg podpisu
  kwalifikowanego/ePUAP) -- artykuły spoza zbioru treningowego,
  zweryfikowane jako trafne po teście.
- Scenariusz: jednoosobowy sklep internetowy pyta o obowiązek IOD --
  poprawnie NIE odpowiada automatycznym "tak"/"nie" z samej ustawy, tylko
  trafnie przechodzi do właściwych przesłanek z RODO art. 37 (organ
  władzy publicznej / monitorowanie na dużą skalę / dane szczególnych
  kategorii na dużą skalę) i stosuje je do stanu faktycznego -- dobra
  generalizacja, nie tylko recytacja wyuczonego wariantu.
- Sąd właściwy dla roszczenia o naruszenie danych o zdrowiu -- poprawnie
  cytuje art. 93 (sąd okręgowy), trafnie odróżnia to od kwalifikacji
  karnej sprawcy (osobny wątek, art. 107 ust. 2).

**Nowa, potwierdzona luka w recall + rozszerzaniu zapytania (NIE
naprawiona -- patrz Krok 25):** pytanie "jakie kary grożą za wyciek
danych osobowych z firmy, i kto je nakłada" nie trafia w art. 101
(przepis wprost odpowiadający na "kto nakłada") ani z adapterem, ani
bez niego -- odpowiedź myli art. 102 (pułap TYLKO dla sektora
publicznego) z ogólną zasadą oraz błędnie przywołuje art. 108
(zupełnie inny czyn -- utrudnianie kontroli) jako podstawę kary za
wyciek. Zdiagnozowano wprost do źródła w Kroku 25.

119/119 testów bez zmian (nowa dziedzina nie wymagała zmian w kodzie
poza wpisami w słownikach `DEFAULT_ADAPTER_PATHS`/`DOMAIN_DESCRIPTIONS`).

## Krok 25: Nowa, cross-domenowa usterka `expand_query()` -- odpowiada zamiast przeformułować przy pytaniach o wysokość kary (zdiagnozowana, NIE naprawiona)

Diagnoza przypadku z Kroku 24 (VAT art.86/KSH art.230-stylem: sprawdzono
najpierw, czy to w ogóle problem rankingu). Wynik: art. 101 osiąga
ranking #1 czysto kosinusowo dla PRECYZYJNEGO sformułowania ("nałożenie
administracyjnej kary pieniężnej przez Prezesa Urzędu..."), ale ranking
#14 dla naturalnego, już całkiem sformalizowanego pytania ("kto nakłada
administracyjną karę pieniężną za naruszenie przepisów..."). To
wskazywało, że -- inaczej niż w Kroku 14/23 -- sam mechanizm rozszerzania
zapytania (`expand_query()`, Krok 21: zawsze na modelu bazowym, bez LoRA)
powinien już wystarczyć, GDYBY wygenerował tę precyzyjną wersję.

Bezpośrednie sprawdzenie wywołania `expand_query()` (bez adaptera)
ujawniło coś innego i poważniejszego: model, mimo jednoznacznego zakazu
w `QUERY_EXPANSION_SYSTEM_PROMPT` ("NIE odpowiadaj na pytanie... zwróć
WYŁĄCZNIE samo przeformułowane zapytanie"), dla pytań o WYSOKOŚĆ SANKCJI
("jakie kary grożą za...") po prostu ODPOWIADA na pytanie wprost, z
konkretną (i tu: BŁĘDNĄ -- ogólne unijne "10 mln euro lub 2% obrotu" z
RODO, nie ta ustawa) liczbą, zamiast zwrócić samo przeformułowanie. To
NIE jest artefakt tej jednej dziedziny -- retrospektywnie sprawdzono na
znanym, referencyjnym pytaniu z Kroku 10 ("jaka jest kara za zabójstwo
człowieka") i model TAK SAMO łamie ten zakaz tam, generując "Zgodnie z
art. 148 § 1... podlega karze pozbawienia wolności...". Różnica: w
domenie `karne` ten błąd był NIEWIDOCZNY w praktyce, bo odpowiedź modelu
bazowego akurat cytuje prawidłowy artykuł niemal dosłownie (kara za
zabójstwo to popularna, dobrze wyuczona wiedza) -- taki "zły"
(nieposłuszny) wynik przypadkiem działa jak bardzo dobre przeformułowanie
i trafia we właściwy artykuł. W `ochrona_danych` własna wiedza modelu o
KTO/na jakiej podstawie nakłada karę administracyjną w tej konkretnej
krajowej ustawie jest błędna (miesza z ogólną wiedzą o RODO), więc to samo
zachowanie tym razem faktycznie psuje wyszukiwanie zamiast je maskować.

**Próba naprawy (NIE wdrożona -- wycofana w tej samej sesji):**
wypróbowano trzy warianty wzmocnienia `QUERY_EXPANSION_SYSTEM_PROMPT`
jawnym zakazem podawania konkretnej kwoty/wymiaru kary przy pytaniach o
sankcję. Każdy wariant naprawiał docelowy przypadek (`ochrona_danych`
kara administracyjna) i/lub referencyjny (`karne` art. 148), ale za
każdym razem psuł RÓŻNY, wcześniej poprawny przypadek z innej, zupełnie
niepowiązanej dziedziny (na przemian: "ile wynosi minimalne wynagrodzenie"
zaczynało dostawać konkretną kwotę 3490 zł zamiast czystego
przeformułowania; "jakie kary grożą za kradzież z włamaniem" zaczynało
dodawać numer artykułu; "w jakim terminie mogę odwołać się od decyzji
administracyjnej" zaczynało podawać "14 dni"). Model okazał się bardzo
czuły na dokładne brzmienie/pozycję dodanego tekstu w tym konkretnym
prompcie -- żadna z trzech prób nie dała czystej poprawy bez nowej
regresji gdzie indziej. Wycofano zmianę do stanu sprzed sesji (`git diff`
na `prompt.py` ogranicza się teraz wyłącznie do wpisu `ochrona_danych` w
`DOMAIN_DESCRIPTIONS`), 119/119 testów bez zmian.

**Status: otwarte, udokumentowane, NIE naprawione.** To pierwsza usterka
w projekcie, gdzie próba prostego wzmocnienia promptu (metoda, która
zadziałała w Kroku 14 i 22) jawnie zawiodła -- każda wersja poprawiała
jeden przypadek kosztem innego. Zanim ktoś spróbuje ponownie, rozważ
podejście inne niż więcej prozy w tym samym prompcie: few-shot przykład
zamiast opisowej instrukcji, albo post-hoc wykrywanie "to wygląda na
odpowiedź, nie na przeformułowane pytanie" (np. obecność liczby/kwoty/
"Zgodnie z art.") z automatycznym odrzuceniem i ponowną próbą, zamiast
prompt engineering. W praktyce błąd jest częściowo maskowany (dokładnie
jak w Kroku 17 dla KSH art. 230) tam, gdzie własna wiedza modelu o danym
fakcie jest akurat poprawna -- ale nie ma gwarancji, że tak będzie w
kolejnych dziedzinach.

## Krok 26: Druga próba naprawy Kroku 25 (few-shot zamiast prozy) -- TAKŻE odrzucona, plus odkrycie poważniejszego, niezależnego problemu

Na prośbę użytkownika, druga próba naprawy Kroku 25 -- tym razem metodą
zasugerowaną tam jako alternatywa: jedna para przykładowych wiadomości
user/assistant (few-shot) wstawiona do `build_query_expansion_messages()`
PRZED prawdziwym pytaniem, zamiast kolejnej wersji opisowej instrukcji w
samym `QUERY_EXPANSION_SYSTEM_PROMPT` (który pozostał nietknięty).
Przykład: user="jaka jest kara za zabójstwo człowieka", assistant=
"odpowiedzialność karna za zabójstwo człowieka".

**Wynik na docelowym problemie: pełny sukces.** Wszystkie sprawdzone
warianty pytań o wysokość sankcji (wyciek danych w `ochrona_danych`,
zabójstwo, kradzież z włamaniem, jazda po alkoholu) przestały być
odpowiadane wprost -- model zaczął zwracać czyste przeformułowania
("odpowiedzialność karna za X"), bez żadnej liczby ani cytatu. Pierwszy
test regresji na tych samych pytaniach co w Kroku 25 (płaca minimalna,
termin odwołania, VAT, "zakupy prywatne") też wypadł czysto.

**Ale pełny test regresji na REFERENCYJNYCH przypadkach z sześciu innych
dziedzin ujawnił nową, poważną regresję, zanim zmiana została wysłana:**
KC art. 415 (zalanie mieszkania przez sąsiada) spadł z rangi #2 (bez
few-shot) na rangę #86 (z few-shot) -- przeformułowanie z few-shot było
krótsze i za bardzo abstrakcyjne ("odpowiedzialność odszkodowawcza za
szkodę wyrządzoną zalaniem mieszkania" zamiast dłuższego, bardziej
kontekstowego zdania bez few-shot). Dodatkowo few-shot pogorszył (nie
naprawił) już i tak zepsuty przypadek VAT art. 86 (ranga 27 -> 74, patrz
niżej) oraz dał drobną, ale realną regresję na `budowlane` art. 42
(#2 -> #3). Jedyne przypadki bez zmiany to te odporne na dobór słów
(KPA art. 129, KK art. 148, KSH art. 230 -- wszystkie #1/#2 niezależnie
od parafrazy). **Wycofano zmianę** (w `prompt.py` i teście
`test_prompt.py`) tym samym trybem co w Kroku 25 -- `git diff` na
`prompt.py` znowu ogranicza się wyłącznie do wpisu `DOMAIN_DESCRIPTIONS`
dla `ochrona_danych`, 119/119 testów bez zmian.

**Odkrycie poboczne, ważniejsze od samego Kroku 25/26:** podczas budowy
tego testu regresji okazało się, że referencyjny przypadek VAT art. 86 --
FLAGOWY przypadek z Kroku 14, pierwszy w ogóle naprawiony tą metodą --
jest już od jakiegoś czasu CICHO ZEPSUTY, niezależnie od tej sesji.
Krok 14 udokumentował rangę #6 dla przeformułowania "Pytanie dotyczy
możliwości odliczenia podatku VAT od zakupów prywatnych w kontekście
przepisów ustawy o podatku od towarów i usług.". Bisekcja przez commity
dotykające `scripts/prompt.py` (`003f8c4` aż do `1831ad1`/HEAD) pokazuje,
że DOKŁADNIE to samo, deterministyczne (greedy, `mlx_lm.generate` bez
samplera) wywołanie modelu bazowego dla TEGO SAMEGO pytania stopniowo się
zmieniało przy KAŻDEJ kolejnej, niepowiązanej zmianie tekstu
`QUERY_EXPANSION_SYSTEM_PROMPT` (dodanie kolejnych dziedzin do
`DOMAIN_DESCRIPTIONS` nie ma wpływu -- ten string jest osobny -- ale
Krok 21 i Krok 22 obie zmieniały `QUERY_EXPANSION_SYSTEM_PROMPT` samo w
sobie), aż urwał się kontekstowy fragment "w kontekście przepisów ustawy
o podatku od towarów i usług" -- dokładnie ten fragment, którego brak
zrzuca ranking art. 86 z #6 na #27. Nikt nie zauważył, bo żaden z kroków
21/22 nie miał VAT art. 86 na liście przypadków do ponownej weryfikacji
(lista referencyjna każdego kroku obejmowała inne, przypadkowo dobrane
przypadki).

**To jest silniejszy dowód na to samo zjawisko z Kroku 25/26:** ten
współdzielony prompt jest na tyle czuły na dokładne brzmienie/pozycję
tekstu, że ręczna, ad-hoc weryfikacja "sprawdź te kilka znanych
przypadków po każdej zmianie" -- metoda stosowana od Kroku 14 -- już
raz zawiodła w praktyce (VAT art. 86 przez min. 3 kolejne kroki), zanim
ta sesja to przypadkiem wykryła przy okazji testowania czegoś innego.
**Rekomendacja dla przyszłej pracy nad `QUERY_EXPANSION_SYSTEM_PROMPT`/
`build_query_expansion_messages()`:** zbuduj mały, ZAPISANY na stałe
(nie ad-hoc) zestaw referencyjnych par (pytanie, oczekiwany artykuł w
top-K) obejmujący WSZYSTKIE znane dotąd przypadki ze wszystkich
dziedzin, i uruchamiaj go automatycznie po KAŻDEJ zmianie tego promptu
-- zamiast wybierać ręcznie kilka przypadków do ponownego sprawdzenia za
każdym razem. To by wykryło zarówno tę regresję VAT, jak i regresję KC
art. 415 z tej sesji, od razu.

**Stan na koniec tej sesji:** Krok 25 (kara/sankcja) pozostaje otwarty,
NIE naprawiony -- dwie próby (proza, few-shot) odrzucone. VAT art. 86
jest teraz UDOKUMENTOWANĄ, potwierdzoną regresją (nie tylko podejrzeniem)
-- osobny, nowy wpis w "Co dalej" niżej, niezależny od Kroku 25/26 i od
tej sesji (istniał już wcześniej, tylko nieudokumentowany).

## Krok 27: Zestaw testów regresyjnych dla `expand_query()` -- `check_expand_query_regressions.py`

Zgodnie z rekomendacją z Kroku 26, zbudowano trwały (zapisany w repo, nie
ad-hoc) zestaw testów regresyjnych, żeby VAT art. 86 (cicha regresja
wykryta w Kroku 26) i podobne przypadki przestały umykać ręcznej,
wybiórczej weryfikacji po każdej zmianie `QUERY_EXPANSION_SYSTEM_PROMPT`/
`build_query_expansion_messages()`.

**Dane:** `data/expand_query_reference_cases.json` -- 12 przypadków, po
minimum jednym na każdą z 11 dziedzin plus drugi w `ochrona_danych` (dla
kontrastu działający/niedziałający). Każdy wpis: `domain`, `act_short`,
`article`, `question`, `max_acceptable_rank` (domyślnie 8 -- dokładnie
`top_k` używane domyślnie w `chat.py`), opcjonalne `known_issue` (jeśli
niepuste, przypadek jest ZNANYM, udokumentowanym problemem, nie nową
regresją) i `note` (kontekst/historia). Gdzie znana była dokładna
historyczna treść pytania (KPA art. 129, KC art. 415, KK art. 148, KSH
art. 230, VAT art. 86, budowlane art. 42) -- użyto jej wprost. Dla
pozostałych pięciu (autorskie, wlasnosc_przemyslowa, rodzinne,
kadry_i_place, oba przypadki ochrona_danych) napisano nowe pytania
odpowiadające udokumentowanym scenariuszom pilotaży i zmierzono ich
obecną rangę jako punkt odniesienia na przyszłość (brak wcześniej
zapisanej dokładnej rangi historycznej dla tych pięciu).

**Skrypt:** `scripts/check_expand_query_regressions.py` -- dla każdego
przypadku woła PRAWDZIWY `chat.expand_query()` (model bazowy, bez LoRA --
tak jak w produkcji, Krok 21) i `RagIndex.search()` z domyślnym
rerankingiem i fuzją rang, mierzy rangę oczekiwanego artykułu w puli 300
kandydatów (żeby wykryć nawet mocno zregresowane przypadki, nie tylko
"w top-8 / nie w top-8"). Kod wyjścia 1, jeśli KTOKOLWIEK się nie
przechodzi (zarówno nowa regresja, jak i wciąż nienaprawiony znany
problem) -- ale raport jawnie rozróżnia "NOWA REGRESJA" od "ZNANY
PROBLEM (nie nowy)", więc wynik jest czytelny bez ręcznego porównywania
z PROGRESS.md. Wymaga lokalnie modelu MLX i zbudowanych indeksów RAG dla
sprawdzanych dziedzin -- nie jest częścią szybkiego `pytest tests/`
(tamten nigdy nie ładuje samego modelu językowego), tylko osobnym,
ręcznie odpalanym narzędziem, tego samego rodzaju co
`check_acts_freshness.py`.

**Pierwsze uruchomienie (2026-09-04), zanim jakikolwiek fix Kroku 25 był
próbowany w tej sesji:** 10/12 OK, 2 znane problemy -- dokładnie te dwa
udokumentowane w Kroku 25/26, z konkretnymi liczbami:
- `vat_86_zakupy_prywatne`: ranga 27 (potwierdza regresję z Kroku 26).
- `ochrona_danych_101_kara_wyciek`: ranga 46 (potwierdza problem z Kroku 25).

Zero nowych regresji na pozostałych 10 przypadkach -- w tym KK art. 148
(#1, choć samo `expand_query()` nadal łamie zakaz odpowiadania wprost,
patrz `known_issue` w danych -- ten test sprawdza tylko ranking, nie
samo posłuszeństwo instrukcji, więc nie wykrywa TEGO konkretnego
naruszenia; celowo odnotowane w danych, żeby przyszła osoba nie uznała
"status: OK" za dowód, że Krok 25 jest jednak nieszkodliwy).

**Jak używać przy przyszłych zmianach `QUERY_EXPANSION_SYSTEM_PROMPT`/
`build_query_expansion_messages()`:** uruchom
`python scripts/check_expand_query_regressions.py` PRZED i PO zmianie,
porównaj -- każda NOWA regresja (przypadek bez `known_issue`, który
przestał przechodzić) blokuje wdrożenie zmiany, dokładnie tak jak two
odrzucone próby z Kroku 25/26 powinny były zostać złapane od razu, gdyby
ten zestaw istniał wcześniej.

## Krok 28: Naprawa VAT art. 86 (regresja) i `ochrona_danych` art. 101 (Krok 25) -- kontekst nazw ustaw w wiadomości użytkownika, nie w SYSTEM_PROMPT

Cztery poprzednie próby (Krok 25: 3 warianty, Krok 26: few-shot) zmieniały
`QUERY_EXPANSION_SYSTEM_PROMPT` -- JEDEN, wspólny dla wszystkich 11
dziedzin string -- i za każdym razem naprawiały jeden przypadek kosztem
regresji w innym, niepowiązanym. Ta próba celowo zmienia inną oś: zamiast
kolejnej wersji prozy w prompcie systemowym, dodaje krótką, czysto
faktyczną linię kontekstu do WIADOMOŚCI UŻYTKOWNIKA w
`build_query_expansion_messages()` -- listę nazw ustaw indeksowanych w
BIEŻĄCEJ dziedzinie (np. dla `podatkowe`: "Ordynacja podatkowa; ustawa o
podatku od towarów i usług; ustawa o podatku dochodowym od osób
fizycznych"), pobraną z nowej metody `RagIndex.act_titles()`.
`QUERY_EXPANSION_SYSTEM_PROMPT` pozostaje BEZ ŻADNEJ ZMIANY.

**Kluczowa różnica względem Kroku 25/26:** ta zmiana jest DOMENOWO-
SPECYFICZNA -- inna lista ustaw w każdej dziedzinie -- więc naprawa
`podatkowe` (VAT art. 86) nie może strukturalnie wpłynąć na `karne` czy
`cywilne`, tak jak mogła (i szkodziła) globalna zmiana wspólnego stringa.
To pierwsza próba naprawy tej klasy problemu na INNEJ osi niż "zmień
tekst współdzielonego promptu".

**Weryfikacja (Krok 27, `check_expand_query_regressions.py`, jeden
proces, jedno ładowanie modelu):** przy okazji wykryto i naprawiono błąd
w samych danych referencyjnych -- `kadry_urlop_wypoczynkowy` wskazywał
art. 152 (ogólne prawo do urlopu, bez liczby dni) zamiast art. 154 § 1
(rzeczywisty wymiar -- 20/26 dni); po poprawce oba dają ranking w top-8
zarówno przed, jak i po tej zmianie. **Wynik na poprawionym zestawie:
12/12 przypadków w top-8**, w tym oba dotąd nienaprawione:
- VAT art. 86: ranga 27 -> **2** (lepiej niż historyczny Krok-14 #6).
- `ochrona_danych` art. 101 (Krok 25): ranga 46 -> **6**.

Zero regresji na pozostałych 10 przypadkach (2 drobne poprawy: autorskie
6->4, wlasnosc_przemyslowa 3->2; reszta bez zmian rangi lub +/-1).

**Zweryfikowane end-to-end przez `chat.py` (nie tylko ranking):**
- VAT: "czy mogę odliczyć VAT od zakupów prywatnych" -- poprawnie cytuje
  art. 86 ust. 1, poprawna odpowiedź "nie".
- `ochrona_danych`: "jakie kary grożą za wyciek danych osobowych z firmy,
  i kto je nakłada" -- poprawnie cytuje art. 101 (ogólna zasada + wyjątki
  dla sektora publicznego, odesłanie do RODO art. 83), SPONTANICZNIE i
  poprawnie art. 101a (obowiązek dostarczenia danych do wymiaru kary), i
  poprawnie odróżnia art. 108 jako OSOBNE przestępstwo (utrudnianie
  kontroli) -- zamiast błędnie je ze sobą mylić, jak przed naprawą.

**Ograniczenie, uczciwie odnotowane (nie ukrywać jako "pełna naprawa
Kroku 25"):** sama usterka źródłowa z Kroku 25 (`expand_query()` łamie
zakaz odpowiadania wprost przy pytaniach o wysokość sankcji) NADAL
WYSTĘPUJE -- reformulacja dla pytania o `ochrona_danych` art. 101 nadal
zaczyna się "Zgodnie z art. 107 ust. 1 i 2..." (co więcej, z BŁĘDNYM
numerem artykułu -- 107 zamiast 101), zamiast czystego przeformułowania.
Krok 28 naprawia WIDOCZNY DLA UŻYTKOWNIKA SKUTEK (ranking i finalną
odpowiedź), bo błędna reformulacja mimo wszystko trafia wystarczająco
blisko tematycznie po dodaniu kontekstu nazw ustaw -- ale to wciąż
"szczęśliwe maskowanie" (dokładnie jak KK art. 148 w Kroku 17/25), nie
naprawa samego nieposłuszeństwa modelu. Nie ma gwarancji, że kolejny
przypadek tego typu też zostanie zamaskowany równie skutecznie.

Wdrożone w `prompt.py` (`build_query_expansion_messages(question,
act_titles=None)`), `rag_search.py` (`RagIndex.act_titles()`), oraz
wywołania w `chat.py`, `chat_cuda.py`, `chat_lmstudio.py`
(`expand_query(..., act_titles=rag.act_titles())`) -- wszystkie trzy
warianty czatu ujednolicone. 124/124 testów jednostkowych przechodzi (5
nowych: 3 dla `build_query_expansion_messages` z/bez `act_titles`, 2 dla
`RagIndex.act_titles()`).

## Krok 29: Piąta próba naprawy Kroku 25 u źródła (post-hoc detekcja "to brzmi jak odpowiedź") -- ODRZUCONA, i to z innego, ważniejszego powodu niż poprzednie cztery

Zgodnie z rekomendacją z Kroku 25/28, wypróbowano dokładnie zasugerowane
tam podejście: zamiast kolejnej zmiany `QUERY_EXPANSION_SYSTEM_PROMPT`,
czysto pythonowa, post-hoc heurystyka wykrywająca w WYGENEROWANYM
przeformułowaniu sygnały "to jest odpowiedź, nie przeformułowanie"
(obecność `art. \d+` lub kwoty/liczby z jednostką: zł, euro, %, mln) --
gdy wykryta, proponowany fallback: odrzucić to przeformułowanie i
wyszukiwać samym oryginalnym pytaniem (bez `extra_queries`).

**Wynik na wszystkich 12 przypadkach referencyjnych (z aktualnym,
Krok-28-owym kontekstem nazw ustaw):** heurystyka poprawnie wykryła 4
przypadki "wyglądające na odpowiedź" (KC art. 415, KK art. 148,
`ochrona_danych` art. 8 i art. 101). Ale sam FALLBACK okazał się
KATASTROFALNIE szkodliwy dla dwóch z nich -- akurat tych najtrudniejszych
i najważniejszych:
- KC art. 415: z przeformułowaniem ranga #2, fallback (samo pytanie) ->
  ranga **#119**.
- `ochrona_danych` art. 101 (docelowy przypadek Kroku 25!): z
  przeformułowaniem ranga #6, fallback -> ranga **#41**.

Tylko 2 z 4 wykrytych przypadków (KK art. 148, `ochrona_danych` art. 8)
byłyby bezpieczne do odrzucenia. **Odrzucono tę próbę** -- nie ze względu
na regresję gdzie indziej (jak w Kroku 25/26), tylko dlatego że sam
mechanizm naprawy jest structuralnie zły: model "łamiąc" zakaz cytowania
i podawania liczb, mimochodem generuje tekst LEKSYKALNIE BLIŻSZY
prawdziwemu przepisowi niż jakiekolwiek zgodne z instrukcją
przeformułowanie by było -- to dokładnie ta sama wiedza, która czyni
odpowiedź "niezgodną z instrukcją", czyni ją też SKUTECZNYM zapytaniem
wyszukującym. Wymuszenie zgodności z instrukcją (przez odrzucenie i
fallback, albo przez jakąkolwiek wersję promptu, która by to
wyeliminowała) systematycznie POGARSZA jakość wyszukiwania dla
najtrudniejszych przypadków w projekcie.

**Wniosek, inny niż po Kroku 25/26:** to nie jest usterka do naprawienia
w dotychczasowym sensie -- to prawdziwy kompromis (trade-off) między
"model przestrzega instrukcji `NIE odpowiadaj na pytanie`" a "wyszukiwanie
działa dobrze", nie przypadek, w którym jedno rozwiązanie zadowala oba.
Ponieważ (a) wynik wyszukiwania/odpowiedzi widoczny dla użytkownika jest
POPRAWNY po Kroku 28 na wszystkich 12 znanych przypadkach, i (b) treść
samego przeformułowania NIGDY nie trafia do finalnej odpowiedzi (służy
wyłącznie jako pomocnicze zapytanie wyszukujące, patrz
`RagIndex.search(extra_queries=...)`), literalne łamanie instrukcji
"NIE odpowiadaj na pytanie" w tym jednym, wewnętrznym wywołaniu jest
NIESZKODLIWE w obecnej, zweryfikowanej postaci -- nie kosmetyczny problem
do zignorowania, tylko świadomie zaakceptowany kompromis, udokumentowany
tutaj, żeby nikt nie próbował go "naprawić" bez ponownego zmierzenia tych
samych 12 przypadków.

**Status Kroku 25 przekwalifikowany:** z "pilny, otwarty bug" na "znana,
zaakceptowana właściwość mechanizmu -- nie naprawiaj bez dowodu na realną
szkodę w konkretnym, nowym przypadku (nie tylko teoretyczne niezgodne z
instrukcją zachowanie)". Jeśli w przyszłej dziedzinie pojawi się przypadek,
w którym ta sama "odpowiedź zamiast przeformułowania" faktycznie psuje
wynik wyszukiwania (tak jak pierwotnie w `ochrona_danych` PRZED Krokiem
28) -- dopiero wtedy warto szukać naprawy, najpewniej znowu przez
domenowo-specyficzny kontekst (metoda Kroku 28), nie przez zmianę
wspólnego promptu ani przez odrzucanie "niezgodnych" przeformułowań.

## Krok 30: Trzecia próba Zasady 6 -- prompt/szablon zamiast fine-tune, TAKŻE niejednoznaczna, NIE wdrożona

Zgodnie z rekomendacją z "Co dalej" (obie poprzednie próby, Krok 15/16,
były fine-tune -- sugerowano spróbować innej kategorii podejścia:
wymuszenie rozróżnienia w prompcie systemowym / szablonie odpowiedzi
zamiast w wagach adaptera), trzecia próba: zmiana Zasady 6 w
`SYSTEM_PROMPT_TEMPLATE`, NIE fine-tune. Kandydacka wersja dodaje: (a)
wymóg NAJPIERW zacytować dosłowną treść przepisu, DOPIERO POTEM, w
wyraźnie oddzielonym fragmencie, przedstawić wniosek; (b) jawny zakaz
wymuszania tego rozróżnienia tam, gdzie przepis wprost i wyczerpująco
reguluje dany stan faktyczny -- adresuje konkretnie tryb awarii z Kroku
16 ("wymuszona, wewnętrznie sprzeczna fraza tam, gdzie rozróżnienie się
nie stosuje").

**Metoda testu:** model BAZOWY (bez adaptera, metodologia projektu --
patrz zasady testowania wyszukiwania w Krokach 26-27), prawdziwy RAG
(`RagIndex domain=cywilne`), porównanie aktualnej vs. kandydackiej
Zasady 6 na 3 sondach: (1) obraz-kopia (Krok 15/16 `valid.jsonl` --
WYMAGA rozróżnienia: art. 84 błąd + art. 88 §2 termin roczny), (2) kot
sąsiada niszczący warzywa (NIE wymaga specjalnego rozróżnienia -- prosta
odpowiedzialność z art. 431 -- test na wymuszanie), (3) prosty
sanity-check bez potrzeby rozróżnienia (termin przedawnienia rękojmi).
Eksperyment czysto porównawczy w skrypcie tymczasowym -- ŻADEN plik w
repo nie został zmieniony.

**Wynik:**
- Sonda 1 (docelowa): OBIE wersje (aktualna i kandydacka) idą tą samą,
  inną niż wzorcowa, ale równie prawnie poprawną ścieżką (rękojmia, art.
  557/568) zamiast wzorcowej (art. 84+88) -- żadna nie użyła wprost frazy
  wzorca. Brak wygranej, brak regresji.
- Sonda 2 (test na wymuszanie): kandydacka wersja dodaje osobną sekcję
  "Wniosek: ..." na końcu -- w odróżnieniu od Kroku 16 (gdzie fine-tune
  dawał wewnętrznie SPRZECZNĄ frazę, np. "Nie dosłowne już zastosowanie
  ... nie wymaga dodatkowych wniosków" w sytuacji bez żadnego
  rozróżnienia do zaznaczenia), tu dodatek jest łagodnie redundantny, ale
  SPÓJNY logicznie -- nowy tryb awarii jest wyraźnie łagodniejszy niż w
  fine-tune.
- Sonda 3 (prosta): kandydacka wersja stała się zauważalnie bardziej
  rozwlekła (pełny cytat artykułu + powtórzone podsumowanie "Z tego
  wynika...") i w jednym przebiegu nie zmieściła się w limicie 500
  tokenów przed zakończeniem myśli.

**Decyzja: NIE wdrożono.** Wynik ponownie niejednoznaczny -- tym razem
bez incoherentnych/sprzecznych fraz (poprawa względem trybu awarii z
Kroku 16), ale też bez wyraźnej poprawy na docelowym przypadku, kosztem
zauważalnej rozwlekłości. `SYSTEM_PROMPT_TEMPLATE` jest współdzielony
przez WSZYSTKICH 11 dziedzin (szerszy promień rażenia niż fine-tune
jednej dziedziny) -- przy tak niejednoznacznym wyniku zmiana promptu
współdzielonego przez wszystkie dziedziny nie jest uzasadniona. Brak
zmian w repo (eksperyment czysto porównawczy).

**Trzy próby, trzy różne metody (fine-tune x2, prompt x1), wszystkie
niejednoznaczne.** To wzmacnia wniosek z Kroku 16: problem nie leży w
konkretnej metodzie (rozmiar zbioru, jednolitość frazowania, ani teraz
też sam prompt), tylko w tym, że rozróżnienie cytat/wniosek wymaga
osądu sytuacyjnego (kiedy faktycznie wychodzę poza dosłowne brzmienie),
którego 11B model z tym stopniem doprecyzowania (mały fine-tune ALBO
sam prompt) nie potrafi konsekwentnie odtworzyć. Nie rekomendujemy
czwartej próby tą samą metodologią -- patrz "Co dalej".

## Krok 31: Darmowy, zerowo-ryzykowny test hipotezy "model za słaby" -- Claude odpowiada na te same 2 sondy z tym samym kontekstem RAG

Po ustaleniu wiodącej hipotezy (Krok 30 -- ograniczenie skali modelu, nie
metody naprawy) użytkownik zapytał o test na realnie większym modelu
(rozważano Llama-3.3-70B-Instruct -- odrzucone jako zbyt ryzykowne przy
64GB pamięci ujednoliconej i niedawnych freeze'ach systemu przy dużo
mniejszym modelu; brak modelu Llama w skali 20-40B, rodzina skacze z 8B
prosto na 70B). Zamiast pobierać cokolwiek lokalnie, wykonano darmowy,
zerowo-ryzykowny test: Claude (obecny asystent, model zdecydowanie
większy niż 11B) odpowiedział na te same 2 sondy z Kroku 30 (obraz-kopia,
kot sąsiada), z DOKŁADNIE tym samym kontekstem RAG (`RagIndex.search()`
bez rozszerzania zapytania -- lekka operacja, model embeddingowy, nie
Bielik) i tym samym promptem systemowym (Zasada 6 w oryginalnym
brzmieniu).

**Odkrycie poboczne przy budowie kontekstu:** dla sondy "obraz-kopia" bez
rozszerzania zapytania indeks w ogóle NIE zwraca art. 84/88 (błąd co do
treści czynności prawnej -- wzorcowa ścieżka z `valid.jsonl`) w top-8,
tylko artykuły o rękojmi (557, 568, 568¹, 592, 552, 551, 898, 812). To
znaczy, że test z Kroku 30 dla tej sondy był częściowo zniekształcony
luką w recall (bez rozszerzania zapytania), nie tylko testem samego
rozumowania -- ale ponieważ i Bielik, i Claude widzieli te same
fragmenty, porównanie samego rozumowania na tych fragmentach pozostaje
uczciwe.

**Wynik porównania:**
- Sonda "obraz-kopia": Claude poprawnie zauważył, że art. 568 §1-2
  wprost daje RAMY czasowe (2 lata od wydania / rok od stwierdzenia
  wady), ale czy termin FAKTYCZNIE upłynął zależy od tego, KIEDY dokładnie
  stwierdzono wadę -- daty tej nie podano w pytaniu -- i wprost odmówił
  jednoznacznej odpowiedzi z tego powodu (zgodnie z Zasadą 2: nie zgaduj,
  gdy fragmenty/fakty nie wystarczają). ŻADNA wersja Bielika (ani
  aktualna, ani kandydacka z Kroku 30) nie zauważyła tej niejednoznaczności
  faktycznej -- obie po prostu założyły, że termin upłynął, bez
  zaznaczenia luki w podanych faktach. To dokładnie ten rodzaj kalibracji
  (wiedzieć, KIEDY nie można być pewnym, bo fakt jest niejednoznaczny),
  którego szukano w Zasadzie 6 -- Claude zastosował go spontanicznie, bez
  żadnej specjalnej instrukcji poza już istniejącą Zasadą 2/6.
- Sonda "kot sąsiada": Claude odpowiedział wprost na podstawie art. 431
  (bezpośrednio pokrywa stan faktyczny), BEZ wymuszania żadnego
  sztucznego rozróżnienia cytat/wniosek (bo nie było takiej potrzeby --
  zgodnie z duchem kandydackiej Zasady 6 z Kroku 30, ale bez jej
  jawnego zapisu). Dodatkowo poprawnie wskazał art. 432 (zajęcie
  zwierzęcia) jako osobną, użyteczną opcję -- Bielik (wersja kandydacka
  z Kroku 30) błędnie odrzucił ten sam artykuł jako "niemający
  bezpośredniego zastosowania".

**Interpretacja, z zastrzeżeniem:** to konkretny, pozytywny dowód
wspierający hipotezę o skali modelu z Kroku 30 -- większy model
spontanicznie wykazał dokładnie tę kalibrację (rozpoznanie niejednoznaczności
faktycznej, niewymuszanie zbędnego rozróżnienia), której brakowało
Bielikowi w obu wariantach promptu. Zastrzeżenie metodologiczne: to
porównanie Claude z samym sobą (ten sam model ocenia własną odpowiedź
jako lepszą) -- ograniczona neutralność, nie w pełni niezależny test.
Mimo to, kierunek wyniku (poprawne rozpoznanie niepewności faktycznej,
bez wymuszania frazy tam, gdzie niepotrzebna) jest spójny z wcześniej
sformułowaną hipotezą i tanim, zerowo-ryzykownym potwierdzeniem, że
warto rozważyć docelowo model innej skali, jeśli kiedyś pojawi się
bezpieczna możliwość lokalnego testu (patrz "Co dalej" -- Qwen3-32B lub
Qwen2.5-32B-Instruct jako bezpieczniejsza w zasobach alternatywa dla
Llama-3.3-70B-Instruct, rodzina Bielik/SpeakLeash kończy się na 11B).
Brak zmian w repo (test czysto porównawczy, bez pobierania czegokolwiek).

## Krok 32: Prawdziwy test na Qwen3.8-27B (LM Studio, lokalnie) -- silne potwierdzenie hipotezy, ALE poważna przeszkoda praktyczna

Po Kroku 31 użytkownik chciał realnego testu na faktycznie większym
modelu, nie tylko na Claude. Rozważono Llama-3.3-70B-Instruct --
odrzucone jako zbyt ryzykowne (64GB pamięci ujednoliconej, niedawne
freeze'y systemu, rodzina Llama nie ma nic między 8B a 70B). Sprawdzono
`huggingface.co/Qwen` -- pierwszy fetch strony zbiorczej organizacji dał
pozornie sfabrykowane nazwy ("Qwen3.8-27B") które błędnie uznano za
halucynację narzędzia. Użytkownik miał już ten model widoczny w LM
Studio -- weryfikacja bezpośrednio na stronie modelu potwierdziła: model
REALNIE ISTNIEJE (wydany sierpień 2026, po granicy wiedzy Claude ze
stycznia 2026 -- stąd błędne pierwsze wrażenie "halucynacji"). Qwen3.8-27B:
27B parametrów, model wizyjno-językowy (multimodalny) na architekturze
Qwen3.5, z domyślnie włączonym trybem "thinking" (regulowane poziomy
wysiłku: xhigh/medium/low), kontekst natywny 262144 tokenów. Pobrany
przez LM Studio w wariancie MLX 8-bit (~27GB, `mlx-community/Qwen3.8-27B-8bit`),
bezpiecznie w 64GB pamięci (LM Studio nie zgłosiło ostrzeżenia o
rozmiarze). **WAŻNE: wszystkie wyniki w tym i następnym kroku dotyczą
WYŁĄCZNIE wariantu 8-bit** -- dostępne są też m.in. 4bit, bf16, mxfp4/8,
nvfp4, oQ4/oQ6 (patrz `mlx-community/Qwen3.8-27B-*`), żadnego z nich nie
testowano. Niższa kwantyzacja (np. 4-bit, ~14-15GB) byłaby szybsza i
lżejsza pamięciowo, ale jakość rozumowania (kluczowa dla Zasady 6) mogła
by ucierpieć -- nie sprawdzone w żadną stronę.

**Metoda:** te same 2 sondy z Kroku 30/31, te same bloki produkcyjne
(`RagIndex`, `build_system_prompt`, `build_user_message`, domena
`cywilne`), wywołane bezpośrednio przez lokalne API LM Studio (nie
`chat_lmstudio.py` -- ten ma na sztywno `max_tokens=500`, za mało dla
modelu z trybem "thinking", które zużywa tokeny z tego samego budżetu
na wewnętrzne rozumowanie zwracane w osobnym polu `reasoning_content`,
oddzielnie od finalnego `content`). Użyto `max_tokens=3000` w
niezależnym skrypcie testowym, bez zmian w kodzie projektu.

**Wynik -- silne potwierdzenie jakości rozumowania:**
- Sonda "obraz-kopia": 713 sekund samego rozumowania (2405 tokenów),
  po czym (przerwane rozłączeniem klienta po 300s, ale zawartość mimo
  to zwrócona przez serwer) model zaczął: *"Na podstawie dostarczonych
  fragmentów nie da się jednoznacznie stwierdzić, że możesz się wycofać
  po upływie dwóch lat od zakupu. Kluczowe znaczenie ma to, kiedy
  została stwierdzona wada..."* -- DOKŁADNIE ta sama niejednoznaczność
  faktyczna (data stwierdzenia wady), którą zauważył Claude w Kroku 31,
  a której nie zauważył żaden wariant Bielika. W samym
  `reasoning_content` model explicite stosuje Zasadę 6 na żywo: *"ale to
  wniosek z faktu; dosłownie §6 wymaga podstępnego zatajenia"* -- oraz
  sam siebie pyta o kalibrację: *"Czy to zbyt ostrożne? Tak, zgodnie z
  zasadą 2."*
- Sonda "kot sąsiada": model zużył CAŁY budżet 3000 tokenów na
  rozumowanie (`reasoning_tokens=2999`, `content=""`, `finish_reason:
  "length"`) -- zero widocznej odpowiedzi. Ale samo rozumowanie jest
  jeszcze bardziej precyzyjne niż Claude w Kroku 31: model SAM zauważył
  niuans, którego nie złapał ani Bielik, ani Claude -- że art. 431 mówi
  o obowiązku "naprawienia szkody", nie używa dosłownie słowa
  "odszkodowanie", więc utożsamienie tych dwóch pojęć w odpowiedzi to
  WNIOSEK, nie dosłowny cytat: *"żądanie odszkodowania... jest naturalnym
  roszczeniem z tego obowiązku, ale to wniosek interpretacyjny, bo
  cytowany artykuł nie używa słowa 'odszkodowanie'"*. Poprawnie też
  zawahał się, czy użytkownik jest "posiadaczem gruntu" (wymóg art. 432),
  i poprawnie pominął nieistotne artykuły (149/150/182/142/754/151) bez
  zbędnego tłumaczenia.

**Wniosek dwustronny:**
1. **Silne, bezpośrednie (nie tylko przez Claude) potwierdzenie hipotezy
   z Kroku 30** -- jakość rozróżniania cytatu od wniosku wyraźnie
   skaluje się z możliwościami modelu. Qwen3.8-27B w obu sondach
   zastosował Zasadę 6 poprawniej i subtelniej niż Bielik w
   jakiejkolwiek z 3 prób (fine-tune x2, prompt x1), a w sondzie 2 --
   nawet subtelniej niż Claude w Kroku 31.
2. **ALE poważna przeszkoda praktyczna, nie tylko szybkość.** Domyślny
   tryb "thinking" nieprzewidywalnie zużywa tokeny z tego samego
   budżetu co finalna odpowiedź -- 713 sekund oczekiwania w sondzie 1,
   ZERO odpowiedzi w sondzie 2 (cały budżet zjedzony przez rozumowanie).
   To nie jest problem, który rozwiąże samo zwiększenie `max_tokens`
   (kosztem jeszcze dłuższego oczekiwania) bez dalszego strojenia (niższy
   poziom wysiłku rozumowania -- `medium`/`low` zamiast domyślnego,
   prawdopodobnie `xhigh` -- nieprzetestowane, czy LM Studio/API
   udostępnia tę kontrolę). W obecnej, domyślnej konfiguracji model NIE
   NADAJE SIĘ do praktycznego wdrożenia w tym projekcie (asystent
   czatowy czasu rzeczywistego) -- ani jako zamiennik Bielika, ani jako
   dodatkowa opcja -- niezależnie od jakości samego rozumowania.

**Aktualizacja (ten sam dzień): parametr `reasoning_effort: "low"` w
zapytaniu API rozwiązuje przeszkodę praktyczną, bez utraty jakości.**
LM Studio/Qwen3.8-27B akceptuje ten parametr (styl podobny do OpenAI
o-series). Powtórzono obie sondy z `reasoning_effort: "low"`:
- Sonda "kot sąsiada": rozumowanie spadło z 2999 (cały budżet, zero
  odpowiedzi) do **554 tokenów**, całość zajęła **163 sekund** (nie
  700-1500+) -- i tym razem PEŁNA, kompletna odpowiedź: poprawny cytat
  art. 431 §1 z wyjątkiem "chyba że...", art. 431 §2, art. 432 §1-2,
  plus praktyczne wskazówki (dokumentacja szkody, wezwanie, konsultacja
  prawna przy sporze).
- Sonda "obraz-kopia": **240 sekund**, 1072 tokenów rozumowania, pełna
  odpowiedź -- poprawnie stosuje art. 568 §1/§6, i SPONTANICZNIE łapie
  jeszcze subtelniejszy niuans niż wcześniej: zauważa, że art. 556 KC
  (definiujący "wadę fizyczną") NIE znajduje się wśród dostarczonych
  fragmentów, więc nie może formalnie potwierdzić, że "kopia sprzedana
  za oryginał" kwalifikuje się jako wada fizyczna -- wprost zaznaczając,
  że to założenie to "standardowe stanowisko w orzecznictwie i
  doktrynie", nie coś wynikające z dostarczonych fragmentów. To jeszcze
  bardziej wyrafinowane zastosowanie Zasady 6 (rozpoznanie luki
  DEFINICYJNEJ, nie tylko faktycznej) niż cokolwiek wcześniej -- u
  Bielika, u Claude (Krok 31), czy nawet u Qwen3.8-27B z domyślnym
  wysiłkiem rozumowania.

**Zrewidowany wniosek: hipoteza z Kroku 30 POTWIERDZONA niezależnym
testem, I przeszkoda praktyczna (czas/nieprzewidywalność) ma prosty
lek** (`reasoning_effort: "low"`) -- 163-240 sekund na odpowiedź to
wciąż wyraźnie wolniej niż Bielik (~10-20s), ale to praktyczny,
działający kompromis, nie ślepy zaułek. Obsługa `reasoning_effort` i
wyższego `max_tokens` wpięta do `chat_lmstudio.py` w Kroku 33 (osobny
krok, ten sam dzień) -- pozostaje otwarte: przetestować więcej niż 2
sondy / więcej dziedzin, zanim rozważy się to jako realną alternatywę
dla Bielika w tym projekcie (patrz "Co dalej"). Nadal brak adaptera
LoRA dla żadnej dziedziny na tym modelu -- LM Studio/GGUF-lub-MLX przez
API to inny stos niż natywny `mlx_lm.load()` w `chat.py`.


## Krok 33: `--max-tokens`/`--reasoning-effort` w `chat_lmstudio.py` -- integracja Kroku 32

Wpięto ustalenia Kroku 32 do samego skryptu, zamiast trzymać je tylko w
osobnym eksperymencie. `answer()` i `expand_query()` w
`scripts/chat_lmstudio.py` dostały opcjonalne parametry `max_tokens`
(domyślnie 500, bez zmian) i `reasoning_effort` (domyślnie `None` --
gdy nieustawiony, w ogóle nie dodaje tego pola do zapytania, więc
zachowanie dla modeli bez trybu "thinking", np. Bielik przez LM Studio,
jest DOKŁADNIE takie samo jak przed tym krokiem). Nowe flagi CLI:
`--max-tokens` i `--reasoning-effort {low,medium,high,xhigh}`. Timeout
żądania `answer()` podniesiony z 300s na 1800s (modele z trybem
"thinking" bez ograniczenia wysiłku potrafią potrzebować kilkunastu
minut, patrz Krok 32).

Celowa decyzja projektowa (potwierdzona z użytkownikiem): **Bielik
(MLX, `chat.py`, 11 adapterów LoRA) pozostaje domyślną, lżejszą opcją**
(mieści się na słabszym sprzęcie, odpowiada w ~10-20s) -- Qwen3.8-27B
przez `chat_lmstudio.py` to opcjonalna, "cięższa" alternatywa dla kogoś
z odpowiednim sprzętem (więcej RAM, akceptowalny dłuższy czas odpowiedzi
za wyższą jakość rozumowania), nie zamiennik.

**Zweryfikowane end-to-end przez sam, zaktualizowany skrypt produkcyjny**
(nie skrypt tymczasowy): `python scripts/chat_lmstudio.py --model
"qwen/qwen3.8-27b" --domain cywilne --max-tokens 3000 --reasoning-effort
low --prompt "..."` na sondzie "kot sąsiada" (Z włączonym rozszerzaniem
zapytania, nie pominiętym) -- pełna, poprawnie sformatowana odpowiedź
(cytaty art. 431/432, tabela podsumowująca, praktyczne wskazówki,
zastrzeżenie o poradzie prawnej). 124/124 testów bez zmian (brak
dedykowanych testów dla wariantów czatu -- wymagałyby uruchomionego
serwera/modelu, tak jak reszta tego pliku).

**Wielodziedzinowy adapter LoRA na Bieliku odłożony na bok** (patrz
"Co dalej" niżej) -- skoro Krok 32 potwierdził hipotezę "model za
słaby" na prawdziwym, większym modelu, próba rozcieńczenia tego samego,
już niewystarczającego budżetu pojemności (rank 8) na 11 dziedzin na
Bieliku prawdopodobnie tylko potwierdziłaby to samo ograniczenie bez
nowej informacji. Warty rozważenia dopiero na modelu klasy Qwen3.8-27B,
i to dopiero PO ewentualnej głębszej integracji (więcej sond/dziedzin
przetestowanych, decyzja czy to w ogóle staje się realną częścią
projektu).

## Krok 34: Test wariantu 4-bit Qwen3.8-27B -- porównywalna szybkość, jeden konkretny błąd merytoryczny

Użytkownik pobrał i sam załadował `Qwen3.8-27B-MLX-4bit` (od
`lmstudio-community`, ~16GB -- odładowano najpierw wariant 8-bit przez
`lms unload`, żeby nie trzymać obu naraz w pamięci, zgodnie z zasadą
"jeden ciężki proces na raz"). Powtórzono sondę "obraz-kopia" (Krok 32)
przez sam zaktualizowany `chat_lmstudio.py --reasoning-effort low
--max-tokens 3000`.

**Pomyłka po drodze, skorygowana:** pierwsza interpretacja logu LM
Studio błędnie wskazywała ~3563 sekundy (59 minut) reasoning na tę
sondę -- alarmująco źle w porównaniu do 8-bit (163-240s). Weryfikacja
przez dokładne dopasowanie znacznika czasu ODEBRANIA żądania
(`Received request`, 11:59:19) do znacznika GOTOWEJ odpowiedzi
(`Generated prediction`, 12:01:49) pokazała prawdziwy czas: **150
sekund** -- porównywalnie, nawet nieco szybciej niż 8-bit. Liczba
"Reasoned for X sekund" w logu LM Studio to najwyraźniej licznik
SKUMULOWANY od załadowania modelu w danej sesji serwera, NIE czas
pojedynczego żądania -- ważna lekcja narzędziowa przy czytaniu tych
logów w przyszłości.

**Wynik po korekcie:**
- **Szybkość: porównywalna z 8-bit** (~150s vs 163-240s), zgodnie z
  normalnym oczekiwaniem, że niższa kwantyzacja jest szybsza, nie
  wolniejsza -- poprzednia, błędna interpretacja sugerowała odwrotnie.
- **Dokładność: JEDEN konkretny, potwierdzony błąd merytoryczny.**
  Odpowiedź 4-bit cytuje **art. 63 KC** jako podstawę błędu w
  oświadczeniu woli -- NIEPRAWIDŁOWO (art. 63 KC dotyczy zgody osoby
  trzeciej na czynność prawną, zupełnie inny temat). Prawidłowy przepis
  to **art. 84 KC** -- poprawnie wskazany przez wariant 8-bit (Krok 32)
  i przez Claude (Krok 31). Sama kluczowa umiejętność (Zasada 6) została
  zachowana mimo tego błędu -- model nadal poprawnie zauważył, że art.
  556 KC (definicja "wady fizycznej") nie ma wśród dostarczonych
  fragmentów.

**Wniosek: jeden punkt danych, ale jedyny jak dotąd bezpośredni dowód na
to, że niższa kwantyzacja kosztuje coś realnego** -- nie szybkość
(porównywalna), tylko precyzję faktograficzną przy odwoływaniu się do
przepisów SPOZA dostarczonych fragmentów (z własnej wiedzy modelu, nie
z RAG). To akurat dokładnie ten rodzaj błędu, przed którym ostrzega
Zasada 1 systemowego promptu ("nie korzystaj z własnej wiedzy o
konkretnych liczbach/przepisach") -- model tu i tak korzysta z własnej
wiedzy (bo fragmenty faktycznie nie zawierają art. 84/63), tylko przy
niższej precyzji ta wiedza jest mniej niezawodna. **8-bit pozostaje
rekomendowanym wariantem** dla tego projektu na razie -- brak dowodu na
jakąkolwiek korzyść z 4-bit (ani szybkość, ani jakość), jeden konkretny
dowód na jego wadę. Druga sonda (kot sąsiada) NIE powtórzona na 4-bit --
wystarczający sygnał, żeby nie inwestować dalszego czasu w ten wariant
teraz. Brak zmian w kodzie (test czysto eksperymentalny).

## Krok 35: Szerszy test Qwen3.8-27B (8-bit) na 4 dziedzinach -- wynik konsekwentnie mocny

Rozszerzono test Kroku 32 (tylko `cywilne`) na 4 sondy w RÓŻNYCH
dziedzinach, dobrane celowo dla różnorodności typu zadania: prosty
cytat referencyjny (`karne`, art. 148 -- ten sam co Krok 10/11/17),
przypadek historycznie wrażliwy na recall (`gospodarcze`, art. 230 --
Krok 22/23), najtrudniejszy znany przypadek w całym projekcie
(`ochrona_danych`, art. 101 -- Krok 25/28), i test odmowy poza zakresem
(pytanie o stawkę VAT zadane w domenie `cywilne`). Metoda: te same
bloki produkcyjne (`RagIndex`, `build_query_expansion_messages` z
`act_titles`, `build_system_prompt`, `build_user_message`),
`reasoning_effort: "low"`, `max_tokens=3000` (500 dla samej ekspansji).

**Wynik -- konsekwentnie mocny na wszystkich 4:**
- `karne` art. 148: poprawny, dokładny cytat -- w tym poprawnie
  zreflektował NOWELIZACJĘ, o której ten projekt nie wiedział od Kroku
  10/11/17 (ówczesne kroki cytowały "nie krótszy od lat 8... albo 25
  lat..."; aktualny tekst w indeksie to już "nie krótszy od lat 10",
  bez oddzielnego progu 25 lat -- Qwen poprawnie oparł się na
  DOSTARCZONYM, aktualnym fragmencie, a nie na własnej, potencjalnie
  nieaktualnej wiedzy treningowej -- początkowo błędnie wzięte za
  halucynację Qwena przez Claude, skorygowane po sprawdzeniu wprost w
  `data/processed/all_articles.json`).
- `gospodarcze` art. 230: samo przeformułowanie zapytania zwróciło
  PUSTY string (nawet przy `reasoning_effort: low`, 500 tokenów
  najwyraźniej czasem to za mało) -- ale mimo braku rozszerzenia,
  surowe wyszukiwanie i tak trafiło w art. 230, zacytowany dosłownie i
  poprawnie, plus spontaniczna, uczciwa uwaga o braku analogicznego
  przepisu dla spółki akcyjnej w dostarczonych fragmentach (prośba o
  doprecyzowanie zamiast zgadywania).
- `ochrona_danych` art. 101: RETRIEVAL tym razem NIE trafił w art. 101
  (inna parafraza wygenerowana przez Qwen niż ta z Kroku 28 nie
  uzyskała potrzebnej precyzji terminologicznej) -- potwierdza, że to
  krucha, cross-modelowa usterka RETRIEVAL, nie coś specyficznego dla
  Bielika. Ale GENEROWANIE poradziło sobie znakomicie z tym brakiem:
  model wprost i uczciwie napisał, że art. 107 i dokładne kwoty kar NIE
  zostały dostarczone, i poprawnie odróżnił art. 108 (utrudnianie
  kontroli) jako osobny czyn -- bez fabrykacji, wzorcowe zastosowanie
  Zasady 2.
- Test odmowy (pytanie o VAT w domenie `cywilne`): wzorcowa odmowa --
  poprawnie rozpoznaje zakres dostarczonych fragmentów, poprawnie
  nazywa właściwą ustawę z dokładną datą, sugeruje konkretne dalsze
  kroki (załącznik nr 3 ustawy o VAT, konsultacja z doradcą podatkowym)
  zamiast zgadywać.

**Wniosek:** jakość generowania Qwen3.8-27B (8-bit,
`reasoning_effort: low`) jest konsekwentnie mocna na różnorodnych
zadaniach w 4 różnych dziedzinach -- w tym w przypadku znanej luki w
recall, generowanie kompensuje ją uczciwym przyznaniem się do braku
informacji, zamiast fabrykować odpowiedź (dokładnie zachowanie, którego
brakowało Bielikowi przed Krokiem 28 dla tego samego przypadku). Jedyny
znaleziony problem (pusta ekspansja zapytania mimo `reasoning_effort:
low`) jest łagodny -- degraduje się do wyszukiwania samym oryginalnym
pytaniem, nie do błędu. Otwarte pytanie o retrieval dla
`ochrona_danych` art. 101 pozostaje niezależne od wyboru modelu
generującego -- to sama warstwa wyszukiwania (embedding + reranking +
fuzja), współdzielona przez wszystkie warianty czatu.

**NIE przetestowano jeszcze** (zwrócone przez użytkownika w trakcie):
`--as-of` (stan prawny na dzień w przeszłości) ani wieloturowej
rozmowy z historią -- oba wymagałyby osobnego, dalszego testu, nie
zrobione w tej sesji ze względu na koszt czasowy (każda sonda to
100-200s). Brak zmian w repo (test czysto eksperymentalny).

## Krok 36: Test `--as-of` i historii rozmowy na Qwen3.8-27B -- `--as-of` bez zarzutu, ale odkryty PRAWDZIWY BUG w `search_with_history()` (nie specyficzny dla Qwena)

Dokończono weryfikację Kroku 35 (`--as-of`, wieloturowa rozmowa --
oba obszary, w których projekt wcześniej znajdował błędy, Krok 10/18/19).

**`--as-of` (`kadry_i_place`, urlop, stan na 2015-01-01): bez zarzutu.**
Qwen poprawnie zaznaczył "stan prawny obowiązujący 1 stycznia 2015 r.",
w rozumowaniu jawnie zweryfikował, że okno ważności zwróconego fragmentu
(2014-09-16 -- 2016-10-12) obejmuje żądaną datę, i dał poprawną,
dosłownie zacytowaną odpowiedź (20 dni, art. 154 § 1 pkt 1). 84s.

**Wieloturowa rozmowa z pytaniem eliptycznym: RETRIEVAL zawiódł, ale
odkryto PRAWDZIWY, odtwarzalny bug we WSPÓLNYM mechanizmie
`search_with_history()` (`prompt.py`), używanym przez WSZYSTKIE
dziedziny i WSZYSTKIE warianty czatu -- nie coś specyficznego dla
Qwena.** Tura 1: "ile dni urlopu wypoczynkowego przysługuje pracownikowi
zatrudnionemu krócej niż 10 lat" (`kadry_i_place`) -- poprawna odpowiedź
(art. 154, 20 dni). Tura 2, eliptyczne: "a jeśli jest zatrudniony
dłużej?" (bez słowa "urlop") -- retrieval zwrócił art. 25¹ Kodeksu
pracy (umowy na czas określony -- ZUPEŁNIE inny temat) zamiast art. 154.

**Przyczyna źródłowa, zdiagnozowana:** `search_with_history()` uruchamia
mechanizm "lookback" do poprzednich pytań z historii TYLKO gdy
`results[0]["score"] < ELLIPTICAL_SCORE_THRESHOLD` (0.75, patrz Krok
10/11). W tym przypadku błędny, niepowiązany fragment (art. 25¹) dostał
score **0.779** -- wystarczająco wysoki, żeby ZABLOKOWAĆ uruchomienie
lookbacku, mimo że to zupełnie zły artykuł. Próg oparty wyłącznie na
WYSOKOŚCI wyniku najlepszego kandydata nie chroni przed przypadkiem, w
którym błędny kandydat przypadkiem scoruje wysoko -- próg nie ma
żadnego sposobu odróżnienia "wysoki score, bo trafne dopasowanie" od
"wysoki score, bo przypadkowe podobieństwo do zupełnie innego tematu".

**To dotyczy WSZYSTKICH 11 dziedzin i obu wariantów czatu (`chat.py`,
`chat_lmstudio.py`), nie tylko `kadry_i_place` ani Qwena** --
`search_with_history()` to jeden, wspólny moduł w `prompt.py`. Ten sam
błąd bardzo prawdopodobnie odtworzyłby się identycznie na Bieliku dla
tego samego pytania (nieprzetestowane wprost w tej sesji, ale mechanizm
jest identyczny niezależnie od modelu generującego -- to warstwa RAG,
nie generacji).

**Ale GENEROWANIE, mimo błędnego retrievalu, zachowało się wzorowo --
lepiej niż można by oczekiwać.** Qwen w wewnętrznym rozumowaniu jawnie
zauważył: "Wait, let me reconsider... In my previous answer, I already
cited art. 154 § 1 pkt 2 which says '26 dni'... but that was based on my
own knowledge, not on the provided fragments" -- i poprawnie ODMÓWIŁ
potwierdzenia odpowiedzi, zamiast polegać na własnej wiedzy treningowej
(mimo że najprawdopodobniej ZNA poprawną odpowiedź, "26 dni", ze swojego
treningu) -- to wyjątkowo zdyscyplinowane zastosowanie Zasady 1 ("nie
korzystaj z własnej wiedzy o konkretnych liczbach"), silniejsze niż
w wielu wcześniejszych testach tego projektu.

**Status: NIE naprawione, udokumentowane jako otwarty bug.** Nie
proponuje się naprawy w tej sesji -- zgodnie z metodologią wypracowaną
w Krokach 25/26/29, każda zmiana progu/heurystyki w tym współdzielonym
mechanizmie wymaga najpierw zbudowania regresyjnego zestawu testowego
(analogicznie do `check_expand_query_regressions.py`, ale dla
`search_with_history()` i pytań eliptycznych), żeby uniknąć powtórki
whack-a-mole z Kroku 25/26. Patrz "Co dalej".

Brak zmian w repo -- oba testy czysto eksperymentalne przez LM Studio.

## Krok 37: Naprawa błędu `search_with_history()` z Kroku 36

Naprawiono, zgodnie z metodologią zapowiedzianą w Kroku 36 -- ale zamiast
budować osobny, duży zestaw regresyjny NAJPIERW, wystarczyło tu coś
prostszego: mechanizm `search_with_history()` nie wywołuje żadnego
modelu językowego (czysty RAG -- embedding + reranking), więc iteracja
i testowanie było tanie i szybkie, bez ryzyka whack-a-mole znanego z
promptów LLM (Krok 25/26/29) -- ryzyko regresji tu jest dużo mniejsze,
bo to deterministyczna logika Pythona, nie generacja modelu.

**Naprawa:** usunięto warunek `results[0]["score"] >= 0.75` jako bramkę
blokującą próbę wariantów z historią -- teraz `search_with_history()`
PRÓBUJE WARIANTÓW Z HISTORIĄ ZAWSZE, gdy jest jakakolwiek historia
rozmowy, i bierze obiektywnie lepszy wynik (ten sam wzorzec "policz
wszystkie warianty, weź maksimum" co przy rozszerzaniu zapytania).
Usunięto też teraz nieużywaną stałą `ELLIPTICAL_SCORE_THRESHOLD`.

**Testy:** zaktualizowano istniejący test, który explicite sprawdzał
"brak dodatkowych wywołań fallbacku" (to była asercja optymalizacji,
którą właśnie usunięto) -- przemianowany na
`test_search_with_history_keeps_strong_match_when_lookback_finds_nothing_better`,
sprawdza tylko poprawność wyniku, nie liczbę wywołań. Dodano NOWY test,
`test_search_with_history_overrides_coincidentally_high_scoring_wrong_match`,
odtwarzający dokładnie przypadek z Kroku 36 (fałszywie wysoki wynik
0.779 dla złego artykułu, prawdziwy próg 0.75) -- teraz przechodzi.
125/125 testów jednostkowych przechodzi (124 + 1 nowy).

**Weryfikacja na PRAWDZIWYM indeksie (nie tylko FakeRag), dokładnie ten
sam przypadek co Krok 36** (`kadry_i_place`, "ile dni urlopu... krócej
niż 10 lat" -> "a jeśli jest zatrudniony dłużej?"): art. 154 teraz ranga
**#1** (score 0.885) zamiast być całkowicie pominięty (poprzednio
blokowany przez błędny art. 25¹ przy score 0.779). Naprawa potwierdzona
empirycznie, nie tylko teoretycznie.

**Dotyczy WSZYSTKICH 11 dziedzin i obu wariantów czatu** (`chat.py`,
`chat_lmstudio.py`) -- to jedna, wspólna funkcja w `prompt.py`, naprawa
działa identycznie niezależnie od modelu generującego (Bielik czy
Qwen), bo to warstwa RAG, nie generacji.

**Koszt naprawy:** każda tura rozmowy z niepustą historią teraz zawsze
wykonuje do `MAX_LOOKBACK_QUESTIONS` (3) dodatkowych wyszukiwań w
indeksie zamiast czasem żadnego -- ale to tanie (samo
embedding+reranking, ułamki sekundy), nie generacja modelu, więc bez
zauważalnego wpływu na czas odpowiedzi.

## Krok 38: Bateria regresyjna jakości odpowiedzi Qwena + nowy przypadek gęstości KSH

Standaryzacja ad-hoc sond Qwena z Kroków 30/32/34/35/36, zapowiedziana
jako następny krok w tym samym pliku. Nowy `data/qwen_answer_reference_cases.json`
(6 przypadków: Zasada 6 obraz/kopia, KK art.148, `ochrona_danych`
art.101, refusal cross-domain, wieloturowa eliptyczna urlop -- czyli
dokładnie przypadek z Kroku 36/37 -- i `--as-of`) + nowy
`scripts/check_qwen_answer_regressions.py`, ten sam wzorzec co
`check_expand_query_regressions.py`, ale sprawdza FINALNĄ ODPOWIEDŹ
tekstową (substring obecny/nieobecny), nie tylko rangę wyszukiwania --
świadomie oznaczony jako płytki "kanarek", nie substytut ręcznego
czytania odpowiedzi. Zweryfikowane end-to-end na żywym Qwen3.8-27B
(8-bit, LM Studio): wszystkie 6 przechodzi. Po drodze złapano i
naprawiono fałszywy alarm we WŁASNYM teście -- zbyt krótkie substringi
zakazane ("25", "umow") trafiały przypadkiem w niepowiązany tekst
("2025-02-07", "Podsumowując") -- poprawione na bardziej specyficzne
("art. 25", "czas określony").

Rozszerzono też `expand_query_reference_cases.json` o nowy przypadek
gęstości w `gospodarcze`: art.214 KSH (zakaz głosowania w konflikcie
interesów, zarząd sp. z o.o.) -- inny klaster niż już pokryty ksh_230.
Na tym etapie tylko dodany jako dana, ranking niesprawdzony w tej
sesji (patrz Krok 39 niżej -- diagnoza w kolejnej sesji wykazała, że to
w ogóle nie jest problem naprawialny metodą Kroku 14/22/23/28). 12
wcześniejszych przypadków bez regresji, 125/125 testów jednostkowych
przechodzi. Commit `e480333`.

## Krok 39: Diagnoza `ksh_214` -- realny, NIEZAMASKOWANY błąd cytatu w produkcji, świadomie nienaprawiony

Kontynuacja Kroku 38 (kolejna sesja, na wyraźną prośbę użytkownika
"weź to A" -- napraw `ksh_214`). Zamiast od razu próbować łatki,
najpierw rozbito problem na etapy (cosine-only / rerank-no-fuse /
rerank+fuse), osobno dla oryginalnego pytania i dla wygenerowanej
reformulacji LLM -- ten sam typ diagnostyki co Krok 8-11 dla KC
art.415.

**Wynik, decydujący:** oryginalne pytanie ląduje na randze #89 (cosine)
/ #111 (rerank) / #114 (fuzja); reformulacja LLM -- mimo że jest niemal
DOSŁOWNYM opisem art.214, jawnie nazywającym "spółkę z ograniczoną
odpowiedzialnością" -- ląduje GORZEJ: #162 / #127 / #163. To wyklucza
metodę Kroku 14/22/23/28 (te wszystkie naprawiają problem SŁOWNICTWA
zapytania) -- tu zapytanie jest już praktycznie idealne, a mimo to nie
pomaga na żadnym etapie. Sygnatura identyczna z głębokim problemem
gęstości tematycznej na poziomie samego bi-encodera (KC art.415, KK
art.148, Krok 8-13), dla którego projekt świadomie nie ma ogólnego
fixu -- **to trzeci potwierdzony data point** dla odłożonej decyzji o
korekcie długościowej rerankera z pilotażu `cywilne` (Krok 8-12).

**Weryfikacja end-to-end** (`chat.py --domain gospodarcze`, pełny
pipeline z adapterem) pokazała coś gorszego niż u KK art.148: to NIE
jest zamaskowane. Produkcyjna odpowiedź na dokładnie to pytanie
realnie cytuje **zły artykuł** -- art. 300⁶⁷ (przepis dla prostej
spółki akcyjnej) zamiast art.214 (sp. z o.o., o którą pytanie wprost
pyta) -- model pomylił dwa bliźniacze przepisy dla różnych typów
spółek, dokładnie ryzyko, które sama dokumentacja pilotażu
`gospodarcze` (krok 7. domeny) sobie przewidziała.

**Decyzja (użytkownik, zapytany wprost):** udokumentować i zostawić,
NIE próbować punktowej łatki -- diagnostyka jasno pokazuje niską szansę
powodzenia (problem nie jest w słownictwie), a projekt ma już
udokumentowaną historię (Krok 25/26/29) łatek do tego współdzielonego
promptu, które naprawiały jeden przypadek kosztem innego. `known_issue`
w `expand_query_reference_cases.json` zaktualizowany o pełną diagnozę
i wynik end-to-end. Żadna zmiana kodu produkcyjnego.

## Krok 40: Sprostowanie Kroku 39 -- błędny oczekiwany artykuł w danych referencyjnych (ten sam typ pomyłki co Krok 28)

Przy próbie realizacji decyzji z "Co dalej" (korekta długościowa
rerankera, mając rzekomo 3 potwierdzone przypadki) dogłębna weryfikacja
treści przepisów KSH wykazała, że **przypadek z Kroku 39 był
zdiagnozowany na błędnych danych**. Art.214 KSH ("Członek zarządu...
nie może być jednocześnie członkiem rady nadzorczej") to przepis o
NIEPOŁĄCZALNOŚCI FUNKCJI -- zupełnie inny temat niż testowe pytanie
("czy członek zarządu może głosować nad umową między spółką a nim
samym"). Właściwy przepis to **art.210 KSH** ("W umowie między spółką a
członkiem zarządu... spółkę reprezentuje rada nadzorcza lub
pełnomocnik...") -- dokładny odpowiednik sp. z o.o. dla art.300⁶⁷
(prosta S.A.) i art.379 (S.A.), czyli DOKŁADNIE treść, którą model
zacytował (pod złym numerem artykułu) w teście end-to-end z Kroku 39.

Ten sam rodzaj błędu jak Krok 28 ("kadry_i_place... zła wartość w
danych referencyjnych, nie w kodzie") -- powtórzone ryzyko przy
ręcznym doborze "oczekiwanego artykułu" bez weryfikacji treści
przepisu, nie tylko jego numeru/tematu z pamięci.

**Poprawiony obraz sytuacji, po korekcie:** art.210 ląduje na randze
#18 (cosine) / #9 (rerank -- reranker tu POMAGA) / #12 (po fuzji, fuzja
lekko PSUJE względem samego rerankingu). Długość tekstu (999 znaków)
nie jest tu skrajna względem sąsiadów klastra (206-2212 znaków) --
**to NIE jest kandydat na 3. data point głębokiej gęstości
tematycznej** (jak błędnie wskazywał Krok 39), tylko łagodniejszy,
bliski-top-8 przypadek pomylenia bliźniaczych przepisów dla różnych
typów spółek -- ten sam typ problemu co ksh_230 przed Krokiem 23,
tylko jeszcze nienaprawiony. End-to-end nadal cytuje zły artykuł
(art.300⁶⁷ zamiast art.210) -- realny błąd zostaje, tylko mniej
dramatyczny niż pierwotnie opisano.

**Wniosek dla decyzji o korekcie długościowej:** premise "mamy już 3
potwierdzone przypadki" była błędna -- naprawdę mamy tylko **1**
niewątpliwy głęboki przypadek (KC art.415), bo KK art.148 jest już
naprawiony obecną fuzją rang (ranga #1 w aktualnym pipeline). Zgodnie
z własną, wcześniej ustaloną ostrożnością projektu ("zbyt ryzykowne
kalibrować bez wystarczającej liczby przypadków", Krok 8-12) -- **NIE
wdrożono ogólnej korekty długościowej w tej sesji**, dopóki nie będzie
więcej niewątpliwych przypadków tego konkretnego wzorca (drastycznie
krótszy tekst niż sąsiedzi klastra). Dane w `expand_query_reference_cases.json`
poprawione (`ksh_214_konflikt_interesow_zarzad` -> `ksh_210_umowa_z_czlonkiem_zarzadu`,
`article: "210"`, `known_issue` zaktualizowany o pełne sprostowanie).
Żadna zmiana kodu produkcyjnego. 125/125 testów przechodzi.

## Krok 41: Naprawa `ksh_210` metodą Kroku 28 (kontekst domenowo-specyficzny, nie globalny prompt) + kontrolny przypadek dla prawdziwego art.214

Na wyraźną prośbę użytkownika ("spróbuj naprawić ksh_210 metodą z Kroku
23") -- ale zdiagnozowany mechanizm okazał się bliższy Krokowi 28 niż
23. Manualne testy wykazały lukę słownikową: reformulacja mówiąca o
"głosowaniu" (dosłowne sformułowanie pytania) daje art.210 rangę #12;
przeformułowana na "kto reprezentuje spółkę" (właściwy mechanizm
prawny z treści art.210) -- rangę #2.

**Pierwsza próba (analogicznie do Kroku 23): dopisanie sparowanego
przykładu wprost do współdzielonego `QUERY_EXPANSION_SYSTEM_PROMPT`.**
Przetestowano 3 warianty wordingu -- każdy albo naprawiał `ksh_210`,
ale regresował NIEPOWIĄZANY `ochrona_danych_101` (#6 -> #50, ten sam
wynik niezależnie od dokładnych słów przykładu -- sam fakt dotknięcia
tego globalnego stringa wystarczał), albo nie regresował niczego, ale
też nie dawał modelowi wystarczająco silnego sygnału (ranga #16 --
nawet gorsza niż bez fixu). Trzy nieudane próby, dokładnie ten sam
wzorzec co Krok 21/22/26 z tym samym promptem.

**Druga próba (metoda Kroku 28 zamiast Kroku 23): nowy
`DOMAIN_QUERY_EXPANSION_HINTS` w `prompt.py`** -- słownik
domena->sparowany przykład, wstrzykiwany do kontekstu w wiadomości
UŻYTKOWNIKA (ten sam mechanizm co `act_titles`), NIE do współdzielonego
promptu systemowego. `build_query_expansion_messages()` przyjmuje nowy
opcjonalny parametr `domain`; wątek przekazywania dociągnięty przez
wszystkie 3 warianty czatu (`chat.py`, `chat_lmstudio.py`,
`chat_cuda.py`) i `check_expand_query_regressions.py` (`rag.domain` /
`idx.domain` -> `expand_query(..., domain=...)`).

**Wynik: naprawia `ksh_210` (#12 -> #2) BEZ regresji na
`ochrona_danych_101`** (nietknięty, #6, bo ten kod ścieżki w ogóle nie
jest wykonywany dla domeny `ochrona_danych` -- strukturalna gwarancja,
ten sam argument co Krok 28). Pełna bateria: 13/13 przechodzi (dodany
też nowy kontrolny przypadek `ksh_214_niepolaczalnosc_funkcji` --
prawdziwy temat art.214, poprawnie sformułowane pytanie o niepołączalność
funkcji, ranga #4, bez problemu, kontrastujący z `ksh_210`). Zweryfikowane
end-to-end przez `chat.py --domain gospodarcze` z adapterem: odpowiedź
teraz poprawnie cytuje art.210 (wcześniej: art.300⁶⁷, zły typ spółki).
125/125 testów jednostkowych przechodzi.

**Lekcja:** metoda Kroku 23 (przykład wprost w globalnym prompcie)
działa tylko wtedy, gdy globalny prompt akurat nie ma innego, wrażliwego
"sąsiada" w tekście, który reaguje na dowolną zmianę -- nie da się tego
przewidzieć bez przetestowania pełnej baterii. Metoda Kroku 28
(kontekst domenowo-specyficzny) jest strukturalnie bezpieczniejsza za
każdym razem, gdy dostępna -- powinna być PIERWSZYM wyborem przy
przyszłych podobnych naprawach, nie ostatnią deską ratunku po
nieudanych próbach globalnych.

## Krok 42 (2026-09-05, nowa sesja): dwunasta dziedzina -- prawo zamówień publicznych (`zamowienia_publiczne`)

Realizacja "Co dalej" z poprzedniej sesji (naturalny kolejny pilot).
Akt bazowy zweryfikowany wprost przez ELI API PRZED pobraniem (ta sama
ostrożność co Krok 17/20/23, po wcześniejszej pomyłce z Kroku 10):
`DU/2019/2019`, "Ustawa z dnia 11 września 2019 r. - Prawo zamówień
publicznych" -- status "akt posiada tekst jednolity", 5 obwieszczeń o
tekście jednolitym w historii. Świadomie NIE dodano starszej,
uchylonej ustawy z 2004 r. (DU/2004/177 poz. 1655), którą ta zastąpiła
od 2021-01-01 -- indeks ma obejmować tylko obowiązujące prawo plus
historię TEJ SAMEJ ustawy (jak każda inna dziedzina), nie poprzedniczkę
prawną innego aktu.

Pobrano 636 bieżących artykułów + 2501 historycznych (4 obwieszczenia:
2021/2022/2023/2024), zbudowano indeks z `--include-history` (3137
artykułów, 5530 fragmentów po podziale). Zero problemów z indeksem
górnym numeracji artykułów (ta ustawa z 2019 r. nie ma odziedziczonych
wstawek międzyartykułowych jak starsze kodeksy) -- pierwsza dziedzina
od `ochrona_danych`, gdzie ten etap przebiegł bezproblemowo od razu.

**Jakość samego wyszukiwania RAG (przed rozszerzaniem zapytania)
wygląda czysto** -- sprawdzono 5 kolokwialnie sformułowanych pytań z
różnych zakątków ustawy (progi bagatelne, wolna ręka, waloryzacja
wynagrodzenia, terminy na odwołanie, zmowa przetargowa, termin
zawarcia umowy): każdy trafiał docelowy artykuł w top-4 nawet bez
rozszerzania zapytania, żaden nie wymagał interwencji. Nie znaleziono
tu odpowiednika problemu gęstości tematycznej z `cywilne`/`karne`/
`gospodarcze`, mimo że ta ustawa też wewnętrznie powtarza analogiczne
mechanizmy dla różnych trybów (klasyczne / sektorowe / obronność i
bezpieczeństwo) -- podobne do `wlasnosc_przemyslowa` (Krok 23), nie do
`gospodarcze` (Krok 17). Dodano jeden przypadek referencyjny do
`data/expand_query_reference_cases.json` (`pzp_515_termin_odwolania`,
art. 515, #1 nawet bez rozszerzania) -- 15/15 przypadków przechodzi.

Napisano 20 przykładów treningowych + 4 walidacyjne
(`data/finetune/zamowienia_publiczne/`), każdy oparty na FAKTYCZNIE
zweryfikowanej treści konkretnego artykułu (pobranej i przeczytanej
wprost z `data/processed/prawo_zamowien_publicznych.json` przed
napisaniem pytania/odpowiedzi -- lekcja Kroku 28/40: nie ufać samej
pamięci o numerze/temacie artykułu). Tematy: progi bagatelne (art. 2),
zasady ogólne (art. 16), jawność (art. 18), przetarg nieograniczony/
ograniczony (art. 129/132/140), wolna ręka (art. 213/214), wadium
(art. 97), warunki udziału (art. 112), wykluczenie obligatoryjne/
fakultatywne (art. 108/109), rażąco niska cena (art. 224), odrzucenie
oferty (art. 226), unieważnienie postępowania (art. 255), standstill
(art. 264/577), zakazane klauzule umowne (art. 433), obowiązkowa
waloryzacja wynagrodzenia (art. 439), zabezpieczenie należytego
wykonania (art. 452), podwykonawstwo (art. 462), legitymacja i terminy
do wniesienia odwołania (art. 505/515), umowa ramowa (art. 311). Dwa z
czterech przykładów walidacyjnych to świadome przypadki hedge/refusal
(dokładna kwota progu unijnego w euro -- nie jest stałą liczbą w
tekście samej ustawy, tylko odesłaniem do dyrektyw UE aktualizowanym
obwieszczeniem Prezesa UZP; odpowiedzialność KARNA za zmowę
przetargową -- to już Kodeks karny, poza indeksem tej dziedziny), tym
samym wzorcem co `ochrona_danych`.

Wytrenowano `bielik11b-zamowienia-publiczne-lora` (100 iteracji, batch
2, `--steps-per-eval 25`) -- ten sam wzorzec przeuczenia co wszystkie
11 poprzednich dziedzin: val loss najlepszy na iter25 (0.905), potem
monotonicznie gorszy (1.009 -> 1.106 -> 1.138 na iter 50/75/100).
Checkpoint iter25 skopiowany jako `bielik11b-zamowienia-publiczne-lora-iter25`,
wpisany do `DEFAULT_ADAPTER_PATHS` w `chat.py`/`chat_cuda.py`, domena
dodana do `DOMAIN_DESCRIPTIONS` w `prompt.py`. 125/125 testów
jednostkowych przechodzi (bez zmian w kodzie produkcyjnym poza tymi
wpisami w słownikach).

**Weryfikacja end-to-end (3 pytania, `chat.py --domain
zamowienia_publiczne`, adapter iter25) -- wynik mieszany, uczciwie
udokumentowany, NIE naprawiany w tej sesji:**
1. Zakazane klauzule umowne (opóźnienie wykonawcy) -- poprawnie cytuje
   art. 433 pkt 1, poprawna treść i wniosek. Drobna nieścisłość
   formalna: model dopisał "ust. 1", którego ten artykuł w
   rzeczywistości nie ma (to jeden, niepodzielony na ustępy przepis z
   samymi punktami) -- kosmetyczny błąd cytatu, nie merytoryczny.
2. Pytanie o zbyt wygórowane warunki udziału (nowe, spoza zbioru
   treningowego) -- poprawnie zgeneralizowało zasadę proporcjonalności
   (art. 112) i prawo do odwołania (art. 515), ale **wprost zmyśliło
   nieistniejące zastosowanie art. 415** ("poleganie na zasobach innych
   podmiotów") -- w tej ustawie art. 415 dotyczy zamówień w dziedzinach
   obronności i bezpieczeństwa, zupełnie inny temat; właściwy przepis
   to art. 118. Sprawdzone przez `rag_search.py` z tym samym pytaniem:
   art. 415 NIE pojawia się nawet w surowym top-8 bez rozszerzania
   zapytania -- to nie jest przypadek cytowania złego, ale
   RZECZYWIŚCIE dostarczonego sąsiada (jak `gospodarcze`/Krok 39), tylko
   czysta halucynacja numeru artykułu z pamięci modelu, naruszająca
   Zasadę 1 (odpowiadaj wyłącznie na podstawie dostarczonych
   fragmentów).
3. Pytanie o dokładną kwotę progu unijnego w euro dla usług
   sektorowych -- model NIE zahedgował (mimo że dokładnie taki
   scenariusz uczono w przykładzie walidacyjnym dla ogólnych progów
   unijnych z art. 3), tylko pewnie podał "750 000 euro", myląc wąski,
   szczególny próg z art. 359 (usługi społeczne i inne szczególne
   usługi) z ogólnym pytaniem o zwykłe usługi sektorowe. Artykuł 359
   istnieje i faktycznie zawiera tę kwotę -- to nie halucynacja liczby,
   tylko niewłaściwe uogólnienie wąskiego przepisu na szersze pytanie,
   ten sam rodzaj błędu co `budowlane` art. 42/45 (Krok 21) i
   `autorskie` (mylenie pokrewnych, ale odrębnych tematów).

**Decyzja: udokumentować, nie naprawiać teraz.** Oba znaleziska (2 i 3)
to pojedyncze obserwacje z pierwszego przebiegu end-to-end -- zgodnie z
ustaloną praktyką projektu (Krok 10/13, Krok 21) warto poczekać na
potwierdzenie wzorca (drugi, niezależny przypadek) przed próbą naprawy,
zamiast naprawiać pojedynczy zaobserwowany przypadek punktową łatką.
Znalezisko 2 (halucynacja numeru artykułu spoza dostarczonego
kontekstu) jest przy tym poważniejsze niż znalezisko 3 i warto go mieć
na uwadze jako possibly nowy wzorzec błędu (dotąd projekt widział
głównie "zły SĄSIEDNI artykuł z dostarczonego kontekstu", nie "artykuł
w ogóle spoza kontekstu") -- gdyby powtórzył się w kolejnej dziedzinie,
to już wystarczający sygnał do zbadania.

## Krok 43 (tego samego dnia): sonda potwierdzająca dla dwóch znalezisk z Kroku 42 -- jedno potwierdzone, jedno nie

Zamiast biernie czekać na przypadkowe powtórzenie się znalezisk z Kroku
42, przeprowadzono aktywną sondę w dwóch już zbudowanych dziedzinach
(ta sama metoda co Krok 13: celowo skonstruowany scenariusz mający
sprowokować dany wzorzec błędu), po jednym teście na wzorzec.

**Sonda dla wzorca 2 (niewłaściwe uogólnienie wąskiego przepisu bez
zahedgowania) -- POTWIERDZONA, `karne`.** Pytanie: "Ukradłem coś ze
sklepu warte 300 zł. Jaka kara mi grozi?". Odpowiedź pewnie zacytowała
art. 278 § 1 Kodeksu karnego (kara pozbawienia wolności od 3 miesięcy
do lat 5) i rozważyła tylko wewnątrz-kodeksowy "wypadek mniejszej
wagi" (art. 278 § 3) -- ani razu nie wspomniała, że kradzież mienia o
wartości nieprzekraczającej 800 zł jest w ogóle WYKROCZENIEM z Kodeksu
wykroczeń (art. 119), nie przestępstwem z Kodeksu karnego -- zupełnie
inny reżim prawny, którego ten indeks nie obejmuje. Model więc pewnie
zastosował niewłaściwy kodeks do stanu faktycznego, zamiast zahedgować
niepewność co do progu kwotowego.

**Sonda dla wzorca 1 (halucynacja numeru artykułu spoza dostarczonego
kontekstu) -- NIE potwierdzona, `wlasnosc_przemyslowa`.** Pytanie o
wzór użytkowy wymyślony przez dwóch pracowników po godzinach, z własnej
inicjatywy. Odpowiedź poprawnie i precyzyjnie zastosowała art. 11 ust.
1-3 (współtwórcy, wyłączenie prawa pracodawcy gdy wynalazek powstał
poza obowiązkami służbowymi) oraz art. 20 (dobrowolne przeniesienie
prawa) -- oba zweryfikowane wprost względem treści ustawy, żadnej
pomyłki. Wzorzec 1 pozostaje na razie pojedynczą obserwacją z
`zamowienia_publiczne` (Krok 42) -- nie eskalowany.

**Ważniejszy wniosek z potwierdzenia wzorca 2: to najpewniej NIE jest
nowa kategoria błędu, tylko kolejny przejaw już zdiagnozowanego i
świadomie porzuconego ograniczenia z Zasady 6 (Krok 15/16/30, "Co
dalej" wyżej)** -- w obu przypadkach model nie potrafi ocenić, czy dany
stan faktyczny w ogóle mieści się w zakresie zastosowania przepisu,
którym dysponuje, zamiast pewnie go zastosować. Kradzież 300 zł
(zakres kwotowy przepisu), art. 359 PZP dla progu unijnego (zakres
przedmiotowy przepisu) i "wprost/wniosek" z Zasady 6 to trzy różne
powierzchniowo, ale tożsame głębiej przejawy tego samego ograniczenia
metapoznawczego -- rozpoznawania granic własnej wiedzy/kontekstu w
czasie generowania. Zgodnie z już ustaloną konkluzją projektu (Krok 30,
potwierdzoną na Qwen3.8-27B w Kroku 32) -- **NIE podjęto próby naprawy**
tego przypadku osobno; byłaby to czwarta próba tej samej, już
nieskutecznej metodologii (mały fine-tune albo sam prompt) na tym samym
ograniczonym modelu bazowym. Traktowane jako dodatkowe potwierdzenie
istniejącej hipotezy, nie nowy, osobny problem do zbadania.

## Krok 44 (tego samego dnia): trzynasta dziedzina -- prawo ochrony środowiska (`ochrona_srodowiska`)

Realizacja "Co dalej" (użytkownik wybrał konkretnie tę dziedzinę spośród
zaproponowanych kandydatów). Akt bazowy zweryfikowany wprost przez ELI
API przed pobraniem: `DU/2001/627`, "Ustawa z dnia 27 kwietnia 2001 r.
Prawo ochrony środowiska." -- status "akt posiada tekst jednolity", aż
12 obwieszczeń o tekście jednolitym w historii (najbogatsza historia ze
wszystkich 13 dziedzin). Pobrano 559 bieżących artykułów + 5621
historycznych, zbudowano indeks z `--include-history` (6180 artykułów,
12069 fragmentów -- największy indeks domenowy w projekcie). Ustawa
używa notacji literowej dla wstawek międzyartykułowych (np. "18ca",
"18cb"), nie superskryptów -- sparsowane poprawnie od razu, zero
kolizji numeracji.

**Jakość wyszukiwania RAG (przed rozszerzaniem zapytania) -- łagodne
odchylenia, podobne do `rodzinne`, nie głęboki problem gęstości jak
`cywilne`/`karne`.** Sprawdzono 4 kolokwialne pytania: czas trwania
pozwolenia (art. 188) i opłata podwyższona za brak pozwolenia (art.
292) trafiają #1 od razu; decyzja naprawcza wójta wobec osoby fizycznej
(art. 363) i termin na roszczenie o wykup nieruchomości (art. 129)
lądują w top-5, wewnątrz progu top-8, ale nie na #1. Dodano przypadek
referencyjny `pos_188_okres_pozwolenia` do
`data/expand_query_reference_cases.json` -- 16/16 przechodzi.

Napisano 20 przykładów treningowych + 4 walidacyjne
(`data/finetune/ochrona_srodowiska/`), każdy zweryfikowany wprost
względem pobranej treści artykułu (ta sama dyscyplina co Krok 28/40/42).
Tematy: zasada prewencji i przezorności (art. 6), definicja instalacji
(art. 3 pkt 6), obowiązki podmiotu korzystającego ze środowiska (art.
140), pozwolenia -- wniosek i czas trwania (art. 184/188), opłaty za
korzystanie ze środowiska i ich podwyższenie (art. 273/276/292),
administracyjna kara biegnąca (art. 300), odpowiedzialność cywilna za
szkody środowiskowe i jej niewyłączenie przez samo posiadanie decyzji
(art. 322/325), decyzje naprawcze organu ochrony środowiska i wójta
(art. 362/363), roszczenia przy obszarze ograniczonego użytkowania i
ich 3-letni termin oraz właściwość sądów powszechnych (art. 129/136),
przepis karny za brak przedkładania wyników pomiarów (art. 341),
samoobliczanie opłaty (art. 284). Dwa z czterech przykładów
walidacyjnych to świadome hedge/refusal: dokładna aktualna stawka
opłaty (ustalana corocznym obwieszczeniem ministra, nie stałą liczbą w
ustawie) i procedura oceny oddziaływania na środowisko (OOŚ) -- to
odrębna ustawa z 2008 r., nie ta, oraz szczegółowe przesłanki
odpowiedzialności deliktowej z Kodeksu cywilnego, do którego ta ustawa
tylko odsyła (art. 322), nie precyzując ich samodzielnie.

Wytrenowano `bielik11b-ochrona-srodowiska-lora` (100 iteracji, batch 2,
`--steps-per-eval 25`) -- ten sam wzorzec przeuczenia co wszystkie 12
poprzednich dziedzin: najlepszy val loss na iter25 (0.963), potem
monotonicznie gorszy (1.111 -> 1.147 -> 1.176). Checkpoint iter25
skopiowany jako `bielik11b-ochrona-srodowiska-lora-iter25`, wpisany do
`DEFAULT_ADAPTER_PATHS` w `chat.py`/`chat_cuda.py`, domena dodana do
`DOMAIN_DESCRIPTIONS` w `prompt.py`. 125/125 testów jednostkowych
przechodzi.

**Weryfikacja end-to-end (2 pytania, adapter iter25) -- czysty wynik,
lepszy niż pilotaż `zamowienia_publiczne` (Krok 42).** (1) Pytanie o
wygaśnięcie pozwolenia po 6 latach eksploatacji -- poprawnie zacytowało
art. 188 ust. 1/1a/1b ORAZ spontanicznie, poprawnie art. 193 ust. 1
pkt 1 (przesłanka wygaśnięcia pozwolenia) -- artykuł spoza zbioru
treningowego, zweryfikowany zgodny z treścią ustawy, nie halucynacja.
(2) Trudniejsze pytanie o zanieczyszczenie studni przez zakład
działający zgodnie z pozwoleniem zintegrowanym -- poprawnie
zgeneralizowało do art. 201/204/211/217d (żaden nie był w zbiorze
treningowym), w tym dokładne, poprawne przywołanie art. 211 ust. 6
pkt 3 (wymagania ochrony gleby/wód gruntowych w pozwoleniu
zintegrowanym) -- zweryfikowane słowo w słowo zgodne z treścią ustawy.
Żadnego ze znalezisk z Kroku 42/43 (halucynacja artykułu spoza
kontekstu, niewłaściwe uogólnienie wąskiego przepisu) NIE
zaobserwowano w tej dziedzinie -- kolejny negatywny data point dla
wzorca 1 (nadal pojedyncza obserwacja z `zamowienia_publiczne`).

## Krok 45 (tego samego dnia): czternasta dziedzina -- prawo o ruchu drogowym (`ruch_drogowy`)

Użytkownik wybrał tę dziedzinę wprost i upoważnił do samodzielnego,
"automatycznego" prowadzenia kolejnych pilotaży (bez pytania o zgodę na
każdy krok), pod warunkiem braku problemów -- ten pilotaż faktycznie
natrafił na jeden i został on zdiagnozowany i naprawiony przed dalszym
ciągiem, zgodnie z tym zastrzeżeniem.

Akt bazowy zweryfikowany wprost przez ELI API: `DU/1997/602`, "Ustawa z
dnia 20 czerwca 1997 r. - Prawo o ruchu drogowym." -- status "akt
posiada tekst jednolity", 11 obwieszczeń o tekście jednolitym w
historii.

**Napotkany i naprawiony problem: nowy wariant artefaktu nawiasów
"jeszcze nieobowiązujących" (`< >`).** Pierwsze pobranie wypisało
ostrzeżenie o niesparowanych nawiasach `< >`. Diagnoza: dwa artykuły
(53a, 65l) kończyły się osieroconym `>` bez odpowiadającego `<` --
prześledzone wprost do surowego tekstu PDF (funkcja
`strip_not_yet_in_force_text` poprawnie usuwała właściwie sparowany
blok `<...>` gdzieś wcześniej w tekście, ale w źródłowym PDF-ie
zostawał jeszcze jeden, dodatkowy, niesparowany znak `>` tuż po końcu
treści tych dwóch artykułów -- pojedynczy artefakt ekstrakcji, NIE
brakujący fragment tekstu: treść obu artykułów jest kompletna i spójna
bez niego). Naprawione w `download_acts.py::strip_not_yet_in_force_text`
-- po dwóch właściwych podstawieniach regex (`< >` i `[ ]`) dodano
końcowe usunięcie wszelkich pozostałych, dosłownych znaków `<`/`>` --
realny tekst przepisu nigdy nie zawiera ich poza tą notacją edytorską,
więc wszystko, co zostaje po właściwym podstawieniu, jest z definicji
takim artefaktem. Bezpieczne dla WSZYSTKICH 14 dziedzin (13
poprzednich nigdy nie zgłosiło tego ostrzeżenia, więc dla nich to
no-op). Dodany test regresyjny
`test_strip_not_yet_in_force_removes_orphan_angle_bracket`. 126/126
testów przechodzi. Zweryfikowane: po poprawce zero artykułów z `<`/`>`
zarówno w bieżącym tekście, jak i we wszystkich 4 wersjach
historycznych (2996 artykułów).

Pobrano 319 bieżących artykułów + 2996 historycznych, zbudowano indeks
z `--include-history` (8694 fragmentów). Ustawa NIE zawiera już
przepisów o egzaminach na prawo jazdy ani o systemie punktów karnych w
szczególe -- to od 2011 r. reguluje odrębna ustawa o kierujących
pojazdami (potwierdzone wprost: art. 129 i art. 135 tylko odsyłają do
niej) -- dobra, naturalna podstawa dla przykładów hedge/refusal.

**Jakość wyszukiwania RAG -- jedna potwierdzona, klasyczna luka
kolokwialna (wzorzec Kroku 7/14/22), zamaskowana w produkcyjnym
pipeline.** "sprzedałem auto komu mam to zgłosić" w ogóle nie trafia
art. 78 w top-8 bez rozszerzania zapytania -- ale precyzyjne
sformułowanie prawnicze ("obowiązek zawiadomienia starosty o zbyciu
pojazdu") daje #1. Zweryfikowane, że pełny pipeline (adapter +
query expansion) faktycznie to maskuje -- końcowa odpowiedź poprawnie
cytuje art. 78 ust. 2 pkt 1 oraz spontanicznie, poprawnie art. 140mb
ust. 6 (kara pieniężna 250 zł za brak zawiadomienia, artykuł spoza
zbioru treningowego, zweryfikowany zgodny z treścią ustawy). Dodano
przypadek referencyjny `rd_20_predkosc_w_miescie` (art. 20, #1 bez
ekspansji) -- 17/17 przechodzi.

Napisano 20 przykładów treningowych + 4 walidacyjne
(`data/finetune/ruch_drogowy/`), każdy zweryfikowany wprost względem
pobranej treści artykułu. Tematy: prędkość dopuszczalna w trzech
strefach (art. 20), ruch prawostronny (art. 16), pierwszeństwo pieszego
na przejściu i zakaz wyprzedzania na nim (art. 26), pasy bezpieczeństwa
i wyjątki (art. 39), zakaz jazdy po alkoholu (art. 45), zakaz
zatrzymania na przejściu (art. 49), obowiązek zgłoszenia zbycia
pojazdu (art. 78), zatrzymanie prawa jazdy przy podejrzeniu
nietrzeźwości (art. 135), obowiązki pieszego i ogólna zasada
ostrożności (art. 3/13), pierwszeństwo poleceń kierującego ruchem nad
sygnalizacją (art. 5), obowiązki rowerzysty (art. 33), dokument
dopuszczenia pojazdu do ruchu (art. 71) i inne. Trzy z czterech
przykładów walidacyjnych to świadome hedge/refusal: liczba punktów
karnych i próg kursu reedukacyjnego, procedura egzaminu na prawo jazdy
(oba -- odrębna ustawa o kierujących pojazdami z 2011 r.) oraz
wysokość konkretnego mandatu (odrębne rozporządzenie taryfowe).

Wytrenowano `bielik11b-ruch-drogowy-lora` (100 iteracji, batch 2,
`--steps-per-eval 25`) -- ten sam wzorzec przeuczenia co wszystkie 13
poprzednich dziedzin: najlepszy val loss na iter25 (1.059), potem
monotonicznie gorszy (1.162 -> 1.225 -> 1.271). Checkpoint iter25
skopiowany jako `bielik11b-ruch-drogowy-lora-iter25`, wpisany do
`DEFAULT_ADAPTER_PATHS` w `chat.py`/`chat_cuda.py`, domena dodana do
`DOMAIN_DESCRIPTIONS` w `prompt.py`.

**Weryfikacja end-to-end (2 pytania, adapter iter25) -- czysty wynik,
podobnie jak `ochrona_srodowiska` (Krok 44).** Oprócz opisanej wyżej
weryfikacji luki recall (art. 78 + spontaniczne, poprawne art. 140mb
ust. 6), drugie pytanie -- scenariusz z pomyloną arytmetyką
przekroczenia prędkości w strefie zamieszkania (70 km/h przy limicie
20 km/h) -- poprawnie zidentyfikowało limit z art. 20 ust. 2, poprawnie
przeliczyło przekroczenie na 50 km/h i poprawnie rozstrzygnęło spór na
korzyść policjanta. Żadnego ze znalezisk z Kroku 42/43 (halucynacja
artykułu spoza kontekstu, niewłaściwe uogólnienie wąskiego przepisu)
nie zaobserwowano -- trzeci z rzędu negatywny data point dla wzorca 1.

## Krok 46 (nowa sesja, 2026-09-12): piętnasta dziedzina -- prawo upadłościowe i restrukturyzacyjne (`upadlosciowe`)

Kontynuacja pod tym samym upoważnieniem do samodzielnego prowadzenia
pilotaży (Krok 45). Dwa powiązane akty (ten sam wzorzec co
`podatkowe`), oba zweryfikowane wprost przez ELI API przed pobraniem:
`DU/2003/535` ("Ustawa z dnia 28 lutego 2003 r. Prawo upadłościowe",
11 obwieszczeń w historii) i `DU/2015/978` ("Ustawa z dnia 15 maja
2015 r. - Prawo restrukturyzacyjne", 8 obwieszczeń). Restrukturyzacyjne
jest naturalnym uzupełnieniem upadłościowego -- od 2016 r. postępowania
restrukturyzacyjne są wydzielone z dawnego "Prawa upadłościowego i
naprawczego" do osobnej ustawy. Pobrano 602 + 406 bieżących artykułów,
indeks z `--include-history` (13614 fragmentów -- największy indeks
domenowy w projekcie, przekraczający nawet `ochrona_srodowiska`).

**Napotkany i udokumentowany (celowo NIENAPRAWIONY) problem: zagnieżdżone
nawiasy `[...[...]...]`/`[...<...>...]` łamią jednoprzebiegowe
podstawienie regex w sposób inny niż Krok 45.** Diagnoza: dwa artykuły
w historii `prawo_upadlosciowe` (`art_216a@2017-11-22`,
`art_241@2019-02-21`) wykazały pozostałości `[`/`]` po przetworzeniu.
Prześledzone wprost do surowych PDF-ów obwieszczeń:
- `art_216a@2017-11-22`: NAPRAWDĘ zagnieżdżone nawiasy w jednym
  artykule -- `[1. ... [stare słowo] <nowe słowo>.]` (zamiana słowa
  wewnątrz ustępu, który sam jest zamieniany w całości). Regex
  `\[(.*?)\]` (niezachłanny) dopasowuje OD zewnętrznego `[` DO
  pierwszego napotkanego `]`, czyli do WEWNĘTRZNEGO `]` -- to
  poprawnie usuwa treść między nimi, ale zostawia osierocony
  zewnętrzny `[` na początku i osierocony zewnętrzny `]` na końcu, po
  drugiej stronie usuniętego `<...>`. Efekt: kosmetyczny, treść
  artykułu pozostaje kompletna i czytelna, tylko z dwoma dodatkowymi
  znakami.
- `art_241@2019-02-21`: POWAŻNIEJSZY przypadek -- dwa SĄSIADUJĄCE,
  ODDZIELNE artykuły (241 i 242) mają swoje własne, prawidłowo
  sparowane bloki `[stary tekst art.241]` i `[stary tekst art.242]`,
  rozdzielone jednym blokiem `<nowy tekst art.241>` między nimi. Po
  usunięciu środkowego `<...>` oba `[` zostają SĄSIADUJĄCE, więc
  niezachłanny regex dopasowuje od PIERWSZEGO `[` (przed art. 241) do
  PIERWSZEGO napotkanego `]` (które w rzeczywistości należy do art.
  242!) -- w efekcie CAŁY nagłówek i treść art. 242 zostają wchłonięte
  jako "treść" art. 241, a art. 242 w ogóle znika jako osobny wpis dla
  tej jednej wersji historycznej.

Wpływ: WYŁĄCZNIE zapytania `--as-of` (stan prawny na dany dzień) dla
konkretnych, wąskich okien czasowych dwóch artykułów jednej ustawy z
15 domen. Bieżący tekst (domyślny, najczęściej używany) obu aktów
całej dziedziny jest w 100% czysty -- zweryfikowane programowo (0
artykułów z `<`/`>`/`[`/`]` w żadnym z dwóch bieżących plików ani w
żadnym z pozostałych ~8488 wpisów historycznych obu aktów). Świadoma
decyzja: NIE podejmować w tej sesji próby ogólnej naprawy (wymagałaby
właściwego parsera śledzącego głębokość i typ nawiasów jednocześnie
zamiast dwóch niezależnych podstawień regex -- ryzykowna zmiana
współdzielonego kodu używanego przez wszystkie 15 dziedzin, wymagająca
starannej walidacji na całej historii wszystkich domen, żeby uniknąć
regresji w stylu Kroku 25/26/29). Zostawione jako udokumentowany,
znany, wąski problem -- do rozważenia osobno, jeśli useru zależy na
w pełni poprawnych odpowiedziach `--as-of` dla tych dwóch konkretnych
artykułów w tym konkretnym oknie czasowym.

**Jakość wyszukiwania RAG -- dwa potwierdzone kolokwialne luki recall
(wzorzec Kroku 7/14/22), obie zweryfikowane jako w pełni zamaskowane
przez query expansion.** "kto zajmuje się majątkiem firmy po
ogłoszeniu upadłości" (cel: art. 156/173, syndyk) i "sąd zatwierdził
układ czy komornik może dalej ścigać dług" (cel: art. 170
restrukturyzacyjne) w ogóle nie trafiały w top-8 bez rozszerzania
zapytania -- ale precyzyjne sformułowanie prawnicze dawało #1 dla obu.
NIE potwierdzono głębszej kolizji MIĘDZY dwoma aktami tej samej
dziedziny (np. syndyk vs. zarządca/nadzorca sądowy) -- to zwykła luka
słownikowa, nie problem gęstości. Dodano przypadek referencyjny
`up_11_niewyplacalnosc` (art. 11 Prawa upadłościowego, #1 bez
ekspansji) -- 18/18 przechodzi.

Napisano 20 przykładów treningowych + 4 walidacyjne
(`data/finetune/upadlosciowe/`), pokrywające OBA akty (13 z Prawa
upadłościowego, 7 z Prawa restrukturyzacyjnego), każdy zweryfikowany
wprost względem pobranej treści artykułu. Tematy: definicja
niewypłacalności (art. 11), kto może złożyć wniosek i termin 30 dni
(art. 20/21), data upadłości i masa upadłości (art. 52/61/62), utrata
zarządu majątkiem i nieważność czynności upadłego (art. 75/77), rola
syndyka (art. 156/173), zakaz prowadzenia działalności (art. 373),
umorzenie zobowiązań po zakończeniu postępowania (art. 369); cel
restrukturyzacji i jej dostępność już na etapie zagrożenia
niewypłacalnością, nie dopiero faktycznej niewypłacalności (art. 3/6),
najprostszy tryb -- postępowanie o zatwierdzenie układu (art. 3 ust.
2), przesłanka odmowy otwarcia (art. 8), pierwszeństwo wniosku
restrukturyzacyjnego nad upadłościowym (art. 11), sposoby
restrukturyzacji zobowiązań (art. 156), skutki dla prokury (art. 293)
i dla egzekucji (art. 170). Dwa z czterech przykładów walidacyjnych to
świadome hedge/refusal: wysokość wynagrodzenia syndyka w konkretnej
sprawie (zależy od wielu czynników, nie stała kwota w ustawie) i
odpowiedzialność karna za działanie na szkodę wierzycieli (Kodeks
karny, poza indeksem tej dziedziny).

Wytrenowano `bielik11b-upadlosciowe-lora` (100 iteracji, batch 2,
`--steps-per-eval 25`) -- ten sam wzorzec przeuczenia co wszystkie 14
poprzednich dziedzin: najlepszy val loss na iter25 (1.032), potem
monotonicznie gorszy (1.308 -> 1.371 -> 1.477). Checkpoint iter25
skopiowany jako `bielik11b-upadlosciowe-lora-iter25`, wpisany do
`DEFAULT_ADAPTER_PATHS` w `chat.py`/`chat_cuda.py`, domena dodana do
`DOMAIN_DESCRIPTIONS` w `prompt.py`.

**Weryfikacja end-to-end (2 pytania, adapter iter25) -- czysty wynik.**
Pierwsze pytanie (kto zajmie się majątkiem po upadłości) potwierdziło
zamaskowanie luki recall opisanej wyżej -- poprawne art. 173 i art. 61
ust. 1, plus spontaniczne, poprawne (zweryfikowane słowo w słowo) art.
491¹¹a o wyborze sposobu likwidacji masy przez syndyka, artykuł spoza
zbioru treningowego. Drugie pytanie (czy można otworzyć restrukturyzację
przed faktyczną niewypłacalnością) poprawnie zastosowało art. 6 ust.
1/3 restrukturyzacyjnego z pełnym zrozumieniem różnicy względem
upadłości. Żadnego ze znalezisk z Kroku 42/43 nie zaobserwowano --
czwarty z rzędu negatywny data point dla wzorca 1 (halucynacja poza
kontekstem).

## Krok 47 (nowa sesja, 2026-09-12: naprawa zagnieżdżonych nawiasów z Kroku 46)

Użytkownik wybrał tę pozycję z "Co dalej" zamiast kolejnej (16.)
dziedziny. Diagnoza dokładnego mechanizmu błędu (na przetworzonych
danych `prawo_upadlosciowe_history.json` sprzed naprawy, bo surowe PDF-y
historyczne nie są trzymane na dysku): `strip_not_yet_in_force_text`
stosowało DWA NIEZALEŻNE, niezachłanne podstawienia regex po kolei --
`NOT_YET_IN_FORCE_RE = <.*?>` (usuń całą zawartość), potem
`SUPERSEDED_STILL_VALID_RE = \[(.*?)\]` (zostaw samą zawartość, usuń
tylko nawiasy). Dla zagnieżdżonego `[1. ... [stare słowo] <nowe
słowo>.]` (art. 216a@2017-11-22): po usunięciu `<nowe słowo>` zostaje
`[1. ... [stare słowo] .]` -- drugi regex, niezachłanny, dopasowuje od
ZEWNĘTRZNEGO '[' do PIERWSZEGO napotkanego ']' (czyli WEWNĘTRZNEGO),
zostawiając osierocone wewnętrzne '[' jako dosłowny tekst i osierocone
zewnętrzne ']' nietknięte -- potwierdzone wprost w danych:
`"...elektronicznym [albo podpisem zaufanym .]"`. Dla dwóch sąsiadujących
artykułów (241/242@2019-02-21), ten sam mechanizm zostawiał jeden
osierocony '[' tuż przed nagłówkiem "Art. 242." -- globalny `re.sub`
kontynuował skanowanie od tego miejsca i sparował go z PIERWSZYM ']'
napotkanym GDZIEKOLWIEK dalej w tekście, wchłaniając po drodze cały
nagłówek i treść art. 242 jako rzekomą "zawartość" tej samej pary (art.
242 znikał jako osobny wpis dla tej jednej wersji historycznej).

**Fix:** zamiast dwóch niezależnych podstawień regex, jeden przebieg
znak-po-znaku. Dla "< >": `_matched_angle_bracket_positions()` dopasowuje
pary POZYCYJNIE, stosem (jak nawiasy w wyrażeniu) -- zawartość między
faktycznie sparowanym '<' i '>' znika w całości niezależnie od
zagnieżdżenia/sąsiedztwa z '[ ]'; niesparowany '<' lub '>' (bez pary
gdziekolwiek w tekście) zostaje bezpiecznie pominięty jako pojedynczy
znak, bez ruszania zawartości wokół (ten sam bezpiecznik co przed Krokiem
47, zachowany testami). Dla "[ ]": nie trzeba w ogóle dopasowywać par --
skoro zawartość ZAWSZE zostaje (adnotacja edytorska), wystarczy usuwać
KAŻDY pojedynczy znak '['/']' napotkany poza obszarem "< >", bez ryzyka
złego sparowania, bo nigdy nie decydujemy o zasięgu na podstawie tych
znaków. To całkowicie eliminuje klasę błędu z Kroku 46 (nie ma już
"pierwszego napotkanego ']' gdziekolwiek dalej" do pomylenia z czymkolwiek).

Dodano 2 nowe testy regresyjne w `tests/test_download_acts.py`
(zagnieżdżone `[...[...]...]` + sąsiadujące artykuły przez granicę
nawiasu) odtwarzające dokładnie oba potwierdzone przypadki -- 128/128
testów przechodzi (wliczając wszystkie 4 istniejące testy
bezpiecznikowe dla niesparowanych nawiasów, bez zmian w ich zachowaniu).

Zweryfikowano na prawdziwych danych: przetworzono ponownie
`prawo_upadlosciowe`/`prawo_restrukturyzacyjne` (bieżący tekst +
historia, 19 pobrań PDF przez ELI API) i odbudowano indeks RAG z
`--include-history` (13618 fragmentów, +4 względem Kroku 46 -- dokładnie
tyle, ile wcześniej gubiło się/zniekształcało). Programowo potwierdzono
**zero** artykułów ze śladowymi `<`/`>`/`[`/`]` w całej historii obu
aktów (5778 + 2713 = 8491 wpisów). Art. 216a@2017-11-22 -- czysty tekst
bez osieroconych nawiasów. Art. 242@2019-02-21 -- odzyskany jako
osobny, kompletny wpis (wcześniej całkowicie wchłonięty przez art. 241
dla tego okna). End-to-end: zapytanie o wymogi zgłoszenia wierzytelności
przez pełnomocnika z `as_of="2019-06-01"` (dokładnie to okno) zwraca
art. 242 na #1 miejscu, poprawnie i kompletnie.

Przy okazji zauważono (nie nowy problem, ta sama logika ostrzeżeń co
przed Krokiem 47, tylko teraz widoczna z bliska): kilka innych wersji
historycznych obu aktów (`prawo_upadlosciowe@2015-02-03/2019-02-21/
2020-06-18`, `prawo_restrukturyzacyjne@2019-01-18`) też zgłasza
niesparowane liczby '<'/'>' w surowym tekście przed czyszczeniem --
ale nowy parser obsługuje to bezpiecznie z definicji (dopasowanie
pozycyjne + jawny fallback dla sierot), co potwierdza zerowa liczba
śladowych nawiasów w finalnych danych dla WSZYSTKICH wersji, nie tylko
tych dwóch pierwotnie zdiagnozowanych.

## Krok 48 (2026-09-12, ta sama sesja: superskrypt na poziomie ustępu)

Użytkownik wybrał tę dawno odłożoną obserwację z Kroku 20 ("superskrypt
na poziomie ustępu, nie artykułu" -- np. `autorskie` art. 1 ust. 2¹
spłaszczony do zwykłego "21.") zamiast kolejnej (16.) dziedziny.
Wcześniejsza notatka zakładała to jako pojedynczy, nieszkodliwy
przypadek bez kolizji -- **sonda przez wszystkie 15 dziedzin obaliła to
założenie**: skan geometrii PDF (ten sam mechanizm co Krok 18, ale
szukający cyfr o mieszanej wielkości czcionki NIE poprzedzonych "Art. ")
znalazł **201 potwierdzonych, prawdziwych przypadków w 7 dziedzinach**
(`kadry_i_place`: `kodeks_pracy` 76, `ustawa_o_rynku_pracy` 2;
`gospodarcze`: `kodeks_spolek_handlowych` 66; `wlasnosc_przemyslowa`: 50;
`rodzinne`: `kodeks_rodzinny_i_opiekunczy` 4; `autorskie`: 2;
`ruch_drogowy`: 1) -- osobno 47 to spłaszczone CYTATY w prozie do
numeru INNEGO artykułu z superskryptem (np. "art. 1291" zamiast "art.
129¹" gdzieś w treści innego artykułu) -- osobny, świadomie
nienaprawiony tu problem (wymaga innego mechanizmu: znać poprawny numer
z INNEGO miejsca w dokumencie, nie tylko poprawić lokalnie), i 11 to
fałszywe alarmy (np. zakres godzin "21:00 a 7:00" błędnie wykryty jako
cyfry o mieszanej wielkości czcionki).

Zweryfikowano wprost jeden z przypadków jako realną korupcję cytatu, nie
kosmetykę: Kodeks pracy art. 29 cytował "art. 25 § 22" (nieistniejący
§22) zamiast poprawnego **§ 2²** (znany, realny przepis o liczeniu
umów na czas określony) -- potwierdza, że to nie tylko numeracja, ale
i poprawność merytoryczna cytatów w danych podawanych modelowi.

**Fix:** nowa para funkcji w `download_acts.py` --
`find_ustep_superscript_corrections_pdfplumber` (wykrywanie przez
geometrię PDF, świadomie ograniczone do jednoznacznych przypadków:
liczba bezpośrednio po "§ " albo po kropce kończącej poprzednie zdanie,
wykluczając nagłówki "Art. N" i cytaty w prozie do innego artykułu) i
`apply_ustep_superscript_corrections` (stosuje korekty na `clean_text`
PRZED podziałem na artykuły). W przeciwieństwie do numerów artykułów
(zawsze 1:1 z pozycją, korekta przez `zip()`), ustępy nie mają stałej
liczby na artykuł, więc dopasowanie idzie przez "odcisk palca" z kilku
poprzedzających słów -- ale dosłowne dopasowanie zawodzi, bo
`pdfplumber` (źródło geometrii) i `pypdf` (źródło `clean_text`) inaczej
renderują białe znaki na złamaniach linii (potwierdzone empirycznie:
0/199 dopasowań dosłownych). Rozwiązanie: dopasowanie przez regex z
`\s+` zamiast dosłownej spacji między słowami kontekstu, odporne na tę
różnicę, plus filtr odrzucający słowa-artefakty nagłówka/stopki strony
Kancelarii Sejmu (numer strony, data), które pdfplumber widzi, a
`clean_text` już nie (bo `pdf_to_clean_text` je usuwa). Każda korekta
konsumuje pierwsze PIERWSZE pozostałe dopasowanie w kolejności
występowania w dokumencie -- więc kolejne wywołania trafiają we
właściwe miejsce nawet przy powtarzającym się krótkim kontekście.
Brak dopasowania (różnice w dzieleniu słów z łącznikiem na złamaniu
linii; miejsce leżące wewnątrz już usuniętego "< >") jest świadomie
bezpiecznie pomijany z ostrzeżeniem -- zweryfikowano końcowy wskaźnik
skuteczności: **170/201 (85%)** przypadków trafnie poprawionych, reszta
zostaje nietknięta (nie gorzej niż przed poprawką).

Dodano 5 nowych testów w `tests/test_download_acts.py` (4 jednostkowe
dla `apply_ustep_superscript_corrections` -- dopasowanie kontekstu,
tolerancja różnic w białych znakach na złamaniu linii, kolejność
stosowania, bezpieczne pomijanie braku dopasowania -- oraz 1 test
end-to-end na prawdziwym `data/raw/prawo_autorskie.pdf` odtwarzający
dokładnie pierwotny przypadek z Kroku 20). 133/133 testów przechodzi.

Przetworzono ponownie WSZYSTKIE 15 dziedzin (bieżący tekst, jedno
uruchomienie `download_acts.py` bez `--domain`) -- 170 korekt
zastosowanych łącznie, zero duplikatów numerów artykułów w żadnej z 23
ustaw, liczby artykułów niezmienione względem stanu sprzed poprawki
(potwierdza, że korekta nigdy nie rusza granic nagłówków "Art. N", tylko
numerację WEWNĄTRZ treści). Zweryfikowano wprost naprawiony przypadek
KP art. 25 -- § 2¹/§ 2² teraz poprawnie wyświetlone jako osobne
nagłówki ustępów w treści artykułu (cytat do nich z art. 29, "art. 25 §
22", pozostaje nietknięty -- to ta sama, świadomie odłożona kategoria
"cytat w prozie", nie regresja).

Przetworzono też ponownie pełną historię (`download_acts_history.py`)
dla 6 dziedzin dotkniętych naprawą (`kadry_i_place`, `gospodarcze`,
`wlasnosc_przemyslowa`, `rodzinne`, `autorskie`, `ruch_drogowy`) i
odbudowano ich indeksy RAG z `--include-history`, żeby poprawka objęła
też zapytania `--as-of` z okresów historycznych, nie tylko bieżący
tekst. Wspólny kod, dotyczy wszystkich 15 dziedzin (7 z nich ma
potwierdzone realne korekty, reszta zero -- niezmieniona).

## Krok 49 (2026-09-12, ta sama sesja: spłaszczone cytaty w prozie do numeru artykułu)

Użytkownik poprosił o zajęcie się drugą, świadomie odłożoną w Kroku 48
kategorią: spłaszczonymi cytatami w prozie do numeru artykułu (np. "art.
1291" zamiast "art. 129¹" w zdaniu cytującym inny artykuł). Wstępna sonda
(wymóg: cyfry o mieszanej wielkości czcionki bezpośrednio po "art. ")
znalazła nie 47, a **1422 kandydatów** -- większość (780) okazała się
FAŁSZYWYMI trafieniami: prawdziwe nagłówki "Art. N" (duża litera,
osobno już obsługiwane), które moja pierwsza wersja sondy błędnie liczyła
jako cytaty. Po poprawnym odsianiu: **642 prawdziwe kandydaty** w 13
aktach.

**Kluczowe odkrycie: to NIE jest jednorodna kategoria, w przeciwieństwie
do ustępów z Kroku 48.** Bliższy ogląd ujawnił trzy różne przypadki:
1. cytaty do artykułu TEJ SAMEJ ustawy (bezpieczne -- ten sam mechanizm
   co ustępy, bo geometria PDF-a niezależnie od kontekstu koduje, które
   cyfry są superskryptem),
2. cytaty do artykułu INNEJ ustawy (np. `ustawa_o_ochronie_danych_
   osobowych`: "na podstawie art. 775 § 2 ustawy z dnia 26 czerwca 1974"
   -- czyli Kodeksu pracy) -- naprawienie wymagałoby znajomości
   prawdziwej numeracji w INNYM akcie, poza zakresem tej sesji,
3. ZNACZNIKI PRZYPISÓW, nie numery artykułów w ogóle (np.
   `prawo_upadlosciowe`: "w art. 239⁷) i art. 240" czytane jako "art.
   239, przypis 7)", nie jako prawdziwy "art. 239⁷") -- zaklasyfikowanie
   tego jako superskryptu wstawiłoby FAŁSZYWY numer artykułu.

Użytkownik świadomie wybrał najbezpieczniejszy zakres: naprawiać
WYŁĄCZNIE kategorię 1. Nowa funkcja
`find_citation_superscript_corrections_pdfplumber` w `download_acts.py`
wyklucza kategorię 2 (fraza nazywająca inny akt -- "ustawy z dnia",
"kodeksu", "zmienianej" itd. -- sprawdzana 45 znaków przed "art." i **150
znaków po liczbie**, `DIFFERENT_ACT_CONTEXT_RE`; szerokie okno "do przodu"
okazało się konieczne -- węższe okno 60 znaków przepuszczało 11
przypadków, gdzie nazwa aktu pada dopiero pod koniec długiego zdania z
kilkoma cytowanymi artykułami) i kategorię 3 (znak tuż po liczbie to
")" -- format znaczników przypisów w tych PDF-ach).

**Przy weryfikacji znaleziono i naprawiono subtelny błąd we wspólnym
mechanizmie zamiany, dotyczący też już wysłanego Kroku 48.** Potwierdzony
rzeczywisty przypadek: `prawo_wlasnosci_przemyslowej`, "art. 1291, art.
1361 i art. 1363" -- kontekst ("odcisk palca") drugiej korekty to
dosłownie `['w', 'art.', '1291,', 'art.']`, czyli CYTUJE spłaszczony
numer pierwszej korekty. Dawniejsza wersja `apply_ustep_superscript_
corrections` mutowała `clean_text` PO KAŻDEJ korekcie z osobna -- więc
naprawienie "1291" na "129¹" niszczyło dosłowne dopasowanie dla "1361",
mimo że oba są poprawne w oryginalnym tekście. Naprawione (i funkcja
przemianowana na `apply_flattened_superscript_corrections`, bo teraz
wspólna dla ustępów i cytatów): wszystkie dopasowania liczone są
WZGLĘDEM NIEZMIENIONEGO tekstu (przesuwany kursor zachowuje kolejność
dla powtarzającego się identycznego kontekstu), a wszystkie podmiany
stosowane są dopiero na końcu, od najdalszej pozycji do najbliższej.
Efekt: na `prawo_wlasnosci_przemyslowej` sam ten bugfix podniósł
skuteczność z 77/105 do 92/105 trafień (błąd occurred za każdym razem,
gdy dwa cytaty sąsiadowały w jednym wyliczeniu, nie tylko w tym jednym
przykładzie).

**Przy okazji: detektor ustępów z Kroku 48 też wymagał rozluźnienia.**
Wymóg "liczba musi kończyć się dosłowną kropką" (bo tak wygląda nagłówek
nowego ustępu) niepotrzebnie wykluczał cytaty do paragrafu innego
artykułu w środku zdania, zapisane tą samą konwencją "§ N" ale kończące
się przecinkiem/średnikiem -- w tym dokładnie potwierdzony przypadek
`kodeks_pracy` art. 29, cytujący "art. 25 § 22," (powinno być "§ 2²,").
Zastąpione tym samym negatywnym lookahead `(?!\d)` co w detektorze
cytatów. To rozluźnienie odkryło z kolei DWA nowe źródła fałszywych
trafień (bez wcześniejszego wymogu kropki wcześniej niewidoczne):
- **skróty niekończące zdania**: "godz." (w zapisie godzin "godz. 600 do
  godz. 1800" w `prawo_ochrony_srodowiska`) też kończy się kropką --
  wykluczone nową, świadomie krótką listą `NON_SENTENCE_ENDING_
  ABBREVIATIONS` (oparta wyłącznie na tym potwierdzonym przypadku),
- **cytaty "§ N"/"ust. N" do INNEJ ustawy** (np. `ustawa_o_systemie_
  ubezpieczen_spolecznych`: "art. 4779 § 31 ustawy z dnia..." cytujący
  Kodeks pracy) -- wykluczone tym samym `DIFFERENT_ACT_CONTEXT_RE`.

Dodano 6 nowych testów (2 jednostkowe dla nowego zachowania
`apply_flattened_superscript_corrections` -- terminator inny niż kropka,
nieinwalidowanie kontekstu sąsiada; 2 end-to-end na prawdziwych PDF-ach
weryfikujące dokładnie oba potwierdzone przypadki cytatów -- poprawiony
i świadomie pominięty). 137/137 testów przechodzi.

Przetworzono ponownie WSZYSTKIE 15 dziedzin (bieżący tekst) --
**finalnie 263 korekty ustępów + 593 korekty cytatów** zastosowane w 8
aktach (`kodeks_pracy`, `kodeks_spolek_handlowych`,
`kodeks_rodzinny_i_opiekunczy`, `kpa`, `prawo_autorskie`,
`prawo_wlasnosci_przemyslowej`, `prawo_o_ruchu_drogowym`,
`ustawa_o_rynku_pracy`) -- mapujących się na 7 dziedzin (`kadry_i_place`,
`gospodarcze`, `rodzinne`, `administracyjne` -- nowa względem Kroku 48,
przez `kpa` -- `autorskie`, `wlasnosc_przemyslowa`, `ruch_drogowy`).
Zero duplikatów numerów artykułów w żadnej z 23 ustaw, liczby artykułów
niezmienione. Przetworzono ponownie pełną historię i odbudowano indeksy
RAG z `--include-history` dla tych 7 dziedzin. Zweryfikowano wprost oba
flagowe przypadki w finalnych danych: KP art. 29 cytuje teraz "art. 25 §
2²" (poprawnie), `wlasnosc_przemyslowa` cytuje "art. 129¹, art. 136¹ i
art. 136³" (wszystkie trzy poprawione, w tym środkowy, który ujawnił
bugfix mechanizmu zamiany). Pełna bateria `check_expand_query_
regressions.py`: 18/18 przechodzi, zero regresji.

**Świadomie nienaprawione, zostawione na przyszłość:** kategoria 2
(cytaty do innej ustawy) -- w wielu przypadkach cytowany akt (np. Kodeks
pracy) JEST jedną z 15 zaindeksowanych ustaw, więc teoretycznie dałoby
się to naprawić przez wyszukanie prawdziwej numeracji w METADANYCH tamtego
aktu zamiast pomijać -- ale to wymaga nowego mechanizmu (indeksu
numeracji artykułów per-akt, nie tylko per-dokument), poza zakresem tej
sesji. Kategoria 3 (znaczniki przypisów) nie jest w ogóle problemem do
naprawy tym mechanizmem -- inna natura danych.

## Krok 50 (2026-09-13, nowa sesja: cytaty do INNEJ ustawy, dokończenie Kroku 49)

Zgodnie z planem zapisanym na koniec poprzedniej sesji, dokończono
kategorię 2 z Kroku 49 (cytaty w prozie do numeru artykułu INNEJ, ale
też zaindeksowanej ustawy) -- np. "art. 775 § 2 ustawy z dnia 26 czerwca
1974 r. − Kodeks pracy" powinno być "art. 77⁵ § 2 ...".

**Rozpoznanie CYTOWANEGO aktu:** po dacie uchwalenia ("ustawy z dnia D
MIESIĄC RRRR r.") -- fraza jednoznacznie identyfikująca polską ustawę,
ta sama, którą akt podaje we własnym tytule. Mapa {data -> act_short}
budowana z tytułów w `data/raw/{short}_meta.json` (jedyny artefakt z
`data/raw/` faktycznie commitowany do repo) dla wszystkich aktów z
`acts_config.json` -- działa niezależnie od tego, które akty zostały
pobrane w BIEŻĄCYM uruchomieniu.

**Bezpiecznik (warunek konieczny do zastosowania korekty):** poprawiony
numer (z ewentualnym sufiksem literowym tuż po cyfrach, np. "18³a") musi
FAKTYCZNIE występować w prawdziwej liście numerów artykułów CYTOWANEGO
aktu (`find_article_numbers_pdfplumber` na jego własnym PDF-ie, z cache
i fallbackiem na jednorazowe pobranie z ELI API, gdy PDF nie jest lokalnie
cache'owany -- PDF-y nie są commitowane). Ten bezpiecznik okazał się
konieczny, nie tylko teoretyczny: sonda na realnych danych złapała
`prawo_autorskie`'s "art. 9915" -- kontekst wspomina KPA ("ustawy z dnia
14 czerwca 1960 r.") kilka słów dalej w tym samym zdaniu, ale z zupełnie
INNEGO powodu proceduralnego (wyłączenie pracownika); to w
rzeczywistości własny artykuł 99¹⁵ Prawa autorskiego, błędnie wykluczony
przez zbyt szeroki lookahead `DIFFERENT_ACT_CONTEXT_RE` -- "99¹⁵" nie
istnieje w prawdziwej numeracji KPA, więc korekta poprawnie NIE została
podjęta.

**Sonda na wszystkich 22 aktach** (nowa funkcja
`find_cross_act_citation_corrections_pdfplumber`) znalazła **14
potwierdzonych, bezpiecznych korekt w 3 aktach**: `ustawa_o_rynku_pracy`
(10 -- głównie rozdział o pracy zdalnej/stażu, art. 67¹⁸-67³¹ i 298² Kodeksu
pracy, plus art. 300⁷³ Kodeksu spółek handlowych dla prostej spółki
akcyjnej), `ustawa_o_ochronie_danych_osobowych` (2 -- art. 77⁵ i 23¹
Kodeksu pracy) i `ustawa_o_ppk` (2 -- art. 23¹ i 18³a Kodeksu pracy).
Wszystkie ręcznie zweryfikowane jako prawdziwe, istniejące przepisy.

**Świadomie NIE objęte, z konkretnym uzasadnieniem:**
- **Poziom USTĘPU** (`find_ustep_superscript_corrections_pdfplumber`) --
  osobna sonda na wszystkich 22 aktach nie znalazła TAM ani jednego
  prawdziwego, możliwego do naprawienia przypadku: wszystkie wykluczone
  tam kandydaty to albo cytaty do aktów spoza projektu (głównie Kodeks
  postępowania cywilnego, nieindeksowany), albo błędne wykluczenia
  prawdziwych cytatów W TEJ SAMEJ ustawie (ten sam efekt uboczny
  szerokiego lookaheadu co w przykładzie z Prawem autorskim wyżej) --
  osobny, nienaprawiony tu problem samego `DIFFERENT_ACT_CONTEXT_RE`, nie
  coś do rozwiązania mechanizmem cross-act.
- **Krótkie formy bez daty** (np. samo "Kodeksu pracy" w zdaniu, gdy
  pełne "ustawy z dnia..." padło wcześniej gdzie indziej) -- jeden
  potwierdzony przypadek pozostał nienaprawiony
  (`prawo_ochrony_srodowiska`, "art. 775 § 2 Kodeksu pracy") -- wymagałoby
  tabeli aliasów nazw z odmianą przez przypadki ("Kodeksu pracy",
  "Kodeksu cywilnego" itd.), świadomie NIE zbudowanej dla jednego
  przypadku (ten sam rygor co `NON_SENTENCE_ENDING_ABBREVIATIONS` w Kroku
  49 -- nie rozszerzać spekulacyjnie bez wielu potwierdzonych przypadków).
- **Cytaty do aktów spoza projektu** (Kodeks postępowania cywilnego,
  Prawo o adwokaturze, ustawa o radcach prawnych, rozporządzenia UE,
  ustawa o OZE) -- nie do naprawienia w ogóle bez ich zaindeksowania,
  poza zakresem.

Dodano 2 nowe testy end-to-end na realnych PDF-ach (10 poprawek w
`ustawa_o_rynku_pracy`, w tym flagowy przypadek z listą wielu cytowań w
jednym zdaniu; bezpiecznik odrzucający "art. 9915" w `prawo_autorskie`)
-- 139/139 testów przechodzi. Przetworzono ponownie 3 dotknięte akty
(bieżący tekst + pełna historia -- `ustawa_o_ppk` ma 3 wersje historyczne,
wszystkie poprawnie skorygowane; `ustawa_o_rynku_pracy`/`ustawa_o_
ochronie_danych_osobowych` nie mają historii), odbudowano indeksy RAG z
`--include-history` dla `kadry_i_place` i `ochrona_danych`. Zweryfikowano
wprost w finalnych danych: wszystkie 14 korekt obecne i poprawne (np.
"art. 67¹⁸, art. 67¹⁹ § 3–5, art. 67²⁴ ... art. 67³¹" i "art. 300⁷³
ustawy z dnia 15 września 2000 r. – Kodeks spółek handlowych"). Pełna
`check_expand_query_regressions.py`: 18/18 przechodzi, zero regresji.

## Krok 51 (2026-09-13, ta sama sesja: utrzymanie baterii regresyjnych)

Zamknięto dwie luki w utrzymaniu baterii regresyjnych, znalezione przy
przeglądzie stanu projektu ("co świeże?"):

**`data/expand_query_reference_cases.json`** -- dziedzina `upadlosciowe`
(dwa akty: Prawo upadłościowe + Prawo restrukturyzacyjne) miała case
tylko dla pierwszego aktu (`up_11_niewyplacalnosc`). Dodano
`restrukt_3_cel_postepowania` (art. 3 Prawa restrukturyzacyjnego -- cel
i rodzaje postępowań, analogiczny fundamentalny przepis do `up_11`) --
ląduje na #2 z realnym `expand_query()`. 19/19 przypadków przechodzi.

**`data/qwen_answer_reference_cases.json`** -- nie był rozszerzany od
Kroku 38 (2026-09-04), mimo że w międzyczasie doszły 4 nowe dziedziny
(`zamowienia_publiczne` Krok 42, `ochrona_srodowiska` Krok 44,
`ruch_drogowy` Krok 45, `upadlosciowe` Krok 46) -- żadna z nich nigdy
nie była testowana na Qwenie. Dodano po jednej sondzie dla każdej (te
same kolokwialne pytania co odpowiadające im przypadki w
`expand_query_reference_cases.json`), uruchomione na żywo przez
użytkownika w LM Studio (`qwen/qwen3.8-27b`, `reasoning_effort=low`):
wszystkie 4 dają precyzyjne, poprawnie ucytowane odpowiedzi, w tym
spontaniczne, trafne dodatki spoza samego pytania (`pzp_515`: dodatkowo
art. 515 ust. 4, art. 577, art. 518; `up_11`: poprawnie połączył
definicję z Prawa upadłościowego z odesłaniem w art. 6 ust. 2 Prawa
restrukturyzacyjnego). Pełny przebieg baterii (10 przypadków): 9/10 OK,
1 to już wcześniej udokumentowany `known_issue` (Krok 25/28,
`ochrona_danych_101` -- `expand_query()` czasem odpowiada wprost zamiast
przeformułować), poprawnie sklasyfikowany jako "znany problem, nie nowa
regresja".

## Krok 52 (2026-09-19, nowa sesja): szesnasta dziedzina -- prawo lotnicze (`lotnicze`)

Realizacja punktu 1 "Co dalej" (16. dziedzina), prowadzona samodzielnie
zgodnie z wcześniejszą autoryzacją użytkownika (fetch -> RAG -> dane
fine-tune -> adapter LoRA -> wiring -> testy -> PROGRESS.md/README.md ->
commit+push, bez pytania o potwierdzenie na każdym kroku).

**Wybór dziedziny:** trzej kandydaci bez jednoznacznego faworyta z
poprzedniej sesji -- prawo geologiczne i górnicze, prawo wodne, prawo
lotnicze. Wszystkie trzy zweryfikowano przez ELI API jako pojedyncze,
"czyste" ustawy z tekstem jednolitym (`Dz.U. 2011 poz. 981` -- Prawo
geologiczne i górnicze; `DU/2017/1566` -- Prawo wodne; `DU/2002/1112` --
Prawo lotnicze). Wybrano **lotnicze**: słownictwo (statki powietrzne,
przestrzeń powietrzna, licencje personelu lotniczego) praktycznie nie
nakłada się na żadną z 15 istniejących dziedzin, w przeciwieństwie do
prawa wodnego i geologiczno-górniczego, które tematycznie stykają się z
`ochrona_srodowiska` -- ten sam wzorzec wyboru "czystego" pilotażu co
KPA (Krok 4) i KRO (Krok 9).

**Dane:** Prawo lotnicze (`DU/2002/1112`), 510 artykułów, 1393 fragmenty
po podziale. Historia wersji dograna od razu (`--include-history`) --
4157 artykułów historycznych z 5 wersji (2019/2020/2022/2023 +
bieżąca). `DOMAIN_DESCRIPTIONS` w `scripts/prompt.py` uzupełniony od
razu (świadomie, po tym jak Krok 9 pokazał, że łatwo to pominąć).

**Fine-tune:** 21 przykładów treningowych + 4 walidacyjne
(`data/finetune/lotnicze/`), wszystkie fakty (wiek do świadectw
kwalifikacji: 15 lat pilot lotni/motolotni/paralotni, 16 lat pilot
MTOM≤495kg/skoczek spadochronowy; przesłanki cofnięcia licencji w tym
alkohol; termin 5 dni roboczych przy zwłoce w opłatach lotniskowych;
widełki kary pieniężnej 5000-50000 zł itd.) zweryfikowane wprost w
pobranym tekście ustawy, nie z pamięci -- to samo zastrzeżenie o braku
weryfikacji eksperckiej co w pozostałych 15 dziedzinach.

**Trening LoRA** (`mlx_lm.lora`, 100 iteracji, batch 2, `--mask-prompt`,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia co
wszystkie 15 poprzednich dziedzin: val loss najlepszy na iter25 (0.624),
potem monotonicznie gorszy (0.772 -> 0.866 -> 0.897 na iter 50/75/100).
Checkpoint iter25 użyty jako `bielik11b-lotnicze-lora-iter25`, wpisany
do `DEFAULT_ADAPTER_PATHS` w `chat.py` oraz jako placeholder w
`chat_cuda.py` (spójnie z pozostałymi 15 dziedzinami -- realne adaptery
CUDA nadal nie istnieją, patrz "Co dalej").

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego, metodyka
Kroku 8 -- sprawdzone też z `--no-adapter`, żeby wykluczyć fałszywy
pozytyw przez pamięć adaptera):
- "Mam 16 lat i chcę latać motolotnią, czy mogę już uzyskać świadectwo
  kwalifikacji?" -- poprawnie: art. 95 ust. 3 pkt 2 (15 lat, warunek
  spełniony) + art. 95 ust. 3a (zgoda opiekunów, bo małoletni), z i bez
  adaptera identycznie poprawne.
- "Zarządca lotniska chce mi zablokować dostęp bo spóźniam się dwa dni
  z zapłatą opłaty lotniskowej, czy może to zrobić od razu?" -- poprawnie
  cytuje art. 78a ust. 1-2 (nie od razu, wymóg 5 dni roboczych) i
  spontanicznie, poprawnie art. 78a ust. 5 (przywrócenie dostępu po
  uregulowaniu zaległości) -- artykuł SPOZA zbioru treningowego,
  potwierdza prawdziwe grounding w RAG, nie tylko pamięć.

139/139 testów jednostkowych przechodzi (bez regresji). README.md
zaktualizowany o wpis `lotnicze` w tabeli dziedzin.

## Krok 53 (2026-09-19, ta sama sesja): rozszerzenie baterii regresyjnych + naprawa realnego bugu w `chat_lmstudio.expand_query()`

Realizacja punktu 2 "Co dalej", w tej samej sesji co Krok 52. Użytkownik
w międzyczasie odpalił LM Studio z Qwenem, więc sondy Qwena wykonano na
żywo w tej sesji zamiast odkładać je na później.

**`data/expand_query_reference_cases.json`:** dodano
`lotnicze_2_definicja_statku_powietrznego` (art. 2 pkt 1 -- definicja
statku powietrznego, fundamentalny przepis analogiczny do KPA art.129/
KP art.8). Ląduje na #1 z realnym `expand_query()` (model bazowy,
Bielik). Pełna bateria: **20/20 przechodzi**, zero regresji na
pozostałych 19.

**`data/qwen_answer_reference_cases.json`:** przed tym krokiem
pokrywał tylko 9 z 15 ówczesnych dziedzin. Dodano po jednej sondzie dla
każdej z 7 brakujących (`administracyjne`, `podatkowe`, `rodzinne`,
`gospodarcze`, `autorskie`, `wlasnosc_przemyslowa`, plus nowa
`lotnicze`) -- te same kolokwialne pytania co odpowiadające im
przypadki w `expand_query_reference_cases.json`, uruchomione na żywo
przez `qwen/qwen3.8-27b` w LM Studio.

**Znaleziony i naprawiony realny bug:** dwie z siedmiu nowych sond
(`vat_86_zakupy_prywatne_odpowiedz`, `ksh_210_umowa_z_czlonkiem_zarzadu_odpowiedz`
-- notabene oba to już wcześniej udokumentowane, historycznie
najtrudniejsze przypadki recall w projekcie, Krok 7/14 i Krok 41)
początkowo dawały odpowiedź "nie znalazłem przepisu w dostarczonych
fragmentach", mimo że oba mają już dedykowane naprawy (query expansion
dla VAT art.86 z Kroku 14, `DOMAIN_QUERY_EXPANSION_HINTS` dla KSH
art.210 z Kroku 41) zweryfikowane jako działające -- ale tylko na
modelu bazowym (Bielik) w `check_expand_query_regressions.py`, nigdy
wcześniej na Qwenie.

Diagnoza bezpośrednim wywołaniem `chat_lmstudio.expand_query()`:
funkcja zwracała **pusty string** dla obu pytań. Przyczyna: hardcoded
`max_tokens=80` w payloadzie tej funkcji (linia z Kroku 28) -- ustawione
z myślą o Bieliku (bez trybu "thinking"), ale dla Qwena z
`reasoning_effort="low"` niewidoczne wewnętrzne rozumowanie liczy się
do TEGO SAMEGO budżetu tokenów. Przy 80 tokenach Qwen czasem zużywa
cały budżet na rozumowanie i nie zostaje mu nic na widoczną parafrazę
-- `answer()` traktuje pusty string jak "brak rozszerzenia"
(`if expanded and ...`) i cicho wraca do oryginalnego, potocznego
sformułowania pytania, dokładnie tego, które query expansion miało
naprawiać. Krok 32 rozwiązał analogiczny problem dla `answer()`
(max_tokens 500 -> 3000), ale nigdy dla drugiego miejsca wywołania
modelu z `reasoning_effort` -- `expand_query()`.

**Naprawa:** podniesiono `max_tokens` w `expand_query()` z 80 do 800
(sam `reasoning_effort` nie wystarcza, patrz komentarz w kodzie --
zmierzono bezpośrednio: 500 dawało nadal ~50% pustych odpowiedzi na 4
próby dla KSH art.210, dopiero 800 dało 0/4 pustych przy trzech
testowanych wartościach 800/1000/1500; 800 wybrane jako najmniejsze
potwierdzone jako niezawodne, żeby niepotrzebnie nie wydłużać czasu
odpowiedzi). Dotyczy WYŁĄCZNIE `chat_lmstudio.py` (Bielik przez `chat.py`
ma własną, osobną implementację `expand_query()` bez trybu "thinking",
nietkniętą). 139/139 testów jednostkowych bez zmian (zmiana to jeden
parametr liczbowy, nie logika).

**Wynik po naprawie** (wielokrotnie powtórzone, żeby zmierzyć
rzeczywistą niezawodność, nie tylko jedno uruchomienie):
- VAT art.86: 3/3 poprawne po naprawie (wcześniej: 0/2 z powodu
  pustej parafrazy) -- teraz konsekwentnie cytuje art. 86 ust. 1 i
  ust. 2a, jednoznaczne "nie".
- KSH art.210: 2/3 poprawne po naprawie (wcześniej: 0/2). Zbadano
  bezpośrednio: sama parafraza + retrieval trafiają art.210 w top-8 w
  KAŻDEJ z wielu prób z niepustym `expand_query()` -- rezydualna
  zawodność (nie 100%) leży w rzadkich, wciąż zbyt krótkich
  odpowiedziach modelu przy tej konkretnej, historycznie najtrudniejszej
  parze pytanie/przepis w całym projekcie (Krok 41 już wymagał dla niej
  dedykowanej naprawy). Świadomie NIE podnoszone dalej w tej sesji --
  przy niezerowej temperaturze (0.2) 100% nie jest realistycznie
  osiągalne samym budżetem tokenów, a dalsze strojenie wymagałoby
  więcej danych. Udokumentowane jako częściowo otwarty `known_issue`
  w danych referencyjnych (realna poprawa względem stanu sprzed tego
  kroku, nie pełna naprawa).

Przy okazji znaleziona i naprawiona osobna usterka METODOLOGICZNA:
pierwsza wersja sondy `autorskie_freelancer_logo_odpowiedz` miała
`expect_contains_any` dobrane na podstawie JEDNEGO uruchomienia --
drugie uruchomienie (ta sama parafraza, `temperature=0.2` czyli
niezerowa wariancja) dało inną, równie poprawną (a nawet pełniejszą,
poprawnie identyfikującą brakujący w kontekście art. 41) odpowiedź bez
żadnej z oczekiwanych fraz, fałszywie oznaczoną jako "NOWA REGRESJA".
Naprawione zakotwiczeniem testu na artykuły konsekwentnie cytowane w
OBU wariantach odpowiedzi (art. 52 / art. 41) zamiast na konkretną
frazę wniosku -- lekcja analogiczna do Kroku 36 (zbyt wąskie
dopasowanie tekstu w free-form odpowiedziach LLM daje fałszywe alarmy),
tylko w przeciwnym kierunku (za wąskie, nie za szerokie).

**Pełny przebieg obu baterii po wszystkich naprawach:**
`expand_query_reference_cases.json` 20/20 (Bielik).
`qwen_answer_reference_cases.json` (17 przypadków, Qwen): 15/17 OK,
2 ZNANE problemy (`ochrona_danych_101` -- Krok 25/28, niezmieniony;
`ksh_210` -- rezydualna, udokumentowana wyżej zawodność), zero nowych
regresji. 139/139 testów jednostkowych zielonych.

## Krok 54 (2026-09-21, nowa sesja): siedemnasta dziedzina -- prawo geologiczne i górnicze (`geologia_i_gornictwo`)

Realizacja punktu 1 "Co dalej" (17. dziedzina), prowadzona samodzielnie
zgodnie z wcześniejszą autoryzacją użytkownika.

**Wybór dziedziny:** ostatni z trzech kandydatów bez jednoznacznego
faworyta odłożonych w Kroku 52 (obok prawa wodnego). Wybrano prawo
geologiczne i górnicze nad prawem wodnym jako mniejsze i tematycznie
węziej skoncentrowane (koncesje górnicze, kwalifikacje geologa,
dokumentacja geologiczna) niż prawo wodne, którego słownictwo pozwoleń
wodnoprawnych mocniej pokrywa się z `ochrona_srodowiska`. Zweryfikowano
wprost przez ELI API (`DU/2011/981`, Dz.U. 2011 nr 163 poz. 981), tytuł
potwierdzony przed pobraniem: "Ustawa z dnia 9 czerwca 2011 r. - Prawo
geologiczne i górnicze" (status: akt posiada tekst jednolity).

**Dane:** 352 artykuły bieżące, 8483 fragmenty po podziale (z historią).
Historia dograna od razu -- 3261 artykułów historycznych z 5 wersji
(2020/2021/2022/2023/2024 + bieżąca). Znany, bezpieczny wariant
ostrzeżenia o niesparowanych nawiasach `< >` (ten sam mechanizm co Krok
45/47) wystąpił 5 razy (bieżący tekst + 4 wersje historyczne) --
sprawdzone programowo: zero osieroconych znaków w finalnym tekście po
każdym artykule, zero duplikatów etykiet artykułów w żadnym oknie
historycznym -- nieszkodliwe, brak akcji. `DOMAIN_DESCRIPTIONS` w
`scripts/prompt.py` uzupełniony od razu.

**Znaleziona i naprawiona usterka podczas pisania danych fine-tune
(przed treningiem, nie po):** przykład o "kto płaci za egzaminy" oparty
wyłącznie o art. 68 (koszty przeprowadzenia egzaminu pokrywa budżet
państwa) okazał się mylący przy weryfikacji end-to-end -- art. 68 dotyczy
kosztów ADMINISTRACYJNYCH (wynagrodzenie komisji, art. 67), podczas gdy
kandydat osobno wnosi "opłatę egzaminacyjną" (250 zł, art. 66 ust. 1) na
rzecz budżetu państwa (art. 66 ust. 3) -- dwa odrębne przepływy finansowe
pod tą samą, kolokwialnie niejednoznaczną nazwą. Wykryte przez ręczne
uruchomienie `chat.py --no-adapter` na tym pytaniu jeszcze przed
treningiem adaptera (baza modelu cytowała art. 63 ust. 4/art. 66, nie
art. 68) -- przykład poprawiony na jednoznaczne pytanie o konkretną kwotę
opłaty (art. 66/63) przed uruchomieniem treningu, żeby nie utrwalić
mylącego skojarzenia w adapterze. 20 przykładów treningowych + 4
walidacyjne, wszystkie pozostałe fakty zweryfikowane wprost w pobranym
tekście ustawy (limity wieku/kwot nie dotyczą tej dziedziny -- użyto za
to konkretnych kwot/terminów: 250 zł/30 zł opłat, 5 lat przedawnienia,
30-dniowy termin ugody, 7-dniowe zawiadomienie przy wydobyciu na potrzeby
własne).

**Realny problem infrastrukturalny napotkany i naprawiony w trakcie
(carve-out "stop jeśli problem", per autoryzacja):** pierwsza próba
treningu (`mlx_lm.lora`) padła natychmiast --
`exec: <stary katalog siostrzanego projektu>/.venv/bin/python3.12: No such file or directory`.
Diagnoza: wspólny `.venv` (`bielik-venv-shared`, symlinkowany do obu
projektów) ma pip-owe wrappery konsolowe
(`mlx_lm.lora`, `mlx_lm.convert`, `pytest`, `hf` -- WSZYSTKIE
console-scripty, nie tylko trening) z zaszytą przy instalacji
bezwzględną ścieżką interpretera wskazującą na STARY katalog
siostrzanego projektu, który w międzyczasie został przemianowany --
błąd był uśpiony do momentu faktycznego
zniknięcia starej ścieżki na dysku, dotyczył OBU projektów, nie tylko
tego treningu (w tym psuł `.venv/bin/pytest`). Naprawa: `.venv/bin/python3
-m pip install --force-reinstall --no-deps mlx-lm==0.31.3 pytest==8.4.2
huggingface_hub==1.27.0` (wersje z `requirements*.txt`) -- reinstalacja
bez zmiany wersji regenerowała wrappery z poprawną, aktualną ścieżką
(`#!~/Projekty/bielik-venv-shared/bin/python3`). Zweryfikowane
`.venv/bin/pytest -q` po naprawie: 139/139 zielone. Trening wznowiony i
dokończony bez dalszych problemów.

**Trening LoRA** (`mlx_lm.lora`, 100 iteracji, batch 2, `--mask-prompt`,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia co
wszystkie 16 poprzednich dziedzin: val loss najlepszy na iter25 (0.847),
potem monotonicznie gorszy (0.908 -> 1.065 -> 1.130 na iter 50/75/100).
Checkpoint iter25 użyty jako `bielik11b-geologia-i-gornictwo-lora-iter25`,
wpisany do `DEFAULT_ADAPTER_PATHS` w `chat.py` oraz jako placeholder w
`chat_cuda.py`.

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego, metodyka
Kroku 8):
- "Jestem właścicielem działki i znalazłem na niej żwir -- czy mogę go
  wydobywać bez koncesji?" -- poprawnie: art. 4 ust. 1-3 (zwolnienie dla
  wydobycia na potrzeby własne osoby fizycznej, limit 10 m³/rok, zakaz
  środków strzałowych, 7-dniowe zawiadomienie organu nadzoru górniczego,
  opłata podwyższona przy naruszeniu) -- artykuł SPOZA zbioru
  treningowego, zweryfikowany dosłownie zgodny z tekstem ustawy,
  identyczna poprawna odpowiedź z i bez adaptera.
- "Sąsiad prowadzi wydobycie piasku bez koncesji i zniszczył część mojej
  działki -- czy to przestępstwo?" -- poprawnie cytuje art. 176 ust. 1
  (kara pozbawienia wolności do lat 3 przy znacznej szkodzie) --
  zweryfikowane dosłownie. Drobna, nieszkodliwa usterka stylu: odpowiedź
  zaczyna się od "Nie znalazłem w dostarczonych fragmentach odpowiedzi
  na pytanie...", a mimo to dalej podaje poprawną, dobrze ugruntowaną
  odpowiedź -- treść poprawna, tylko niezgrabny wstęp, nie warte dalszej
  naprawy dla pojedynczej obserwacji.
- `--as-of 2021-01-01` (art. 149, przedawnienie roszczeń o szkodę
  górniczą) -- poprawnie oznacza stan prawny na żądaną datę.

**Bateria regresyjna:** dodano `geo_10_wlasnosc_gornicza` (art. 10 --
własność górnicza, złoża węgla kamiennego i innych wymienionych kopalin
przysługują Skarbowi Państwa niezależnie od właściciela gruntu --
fundamentalny przepis, analogiczny wzorzec do KPA art.129/KP art.8/
lotnicze art.2) do `expand_query_reference_cases.json`. Pełna bateria:
**21/21 przechodzi**, zero regresji na pozostałych 20. `qwen_answer_reference_cases.json`
świadomie NIE rozszerzony w tej sesji (zostawione jako następny "Co
dalej", ten sam wzorzec co Krok 52 -> Krok 53). 139/139 testów
jednostkowych przechodzi. README.md zaktualizowany o wpis
`geologia_i_gornictwo` w tabeli dziedzin.

## Krok 55 (2026-09-21, ta sama sesja): rozszerzenie `qwen_answer_reference_cases.json` o `geologia_i_gornictwo`

Realizacja punktu 2 "Co dalej" (dług z Kroku 54), ten sam wzorzec co
Krok 52 -> Krok 53. Użytkownik miał w międzyczasie uruchomione LM Studio
z Qwenem, więc sonda wykonana na żywo w tej sesji.

Dodano `geo_10_wlasnosc_gornicza_odpowiedz` -- ta sama sonda co
`geo_10_wlasnosc_gornicza` w `expand_query_reference_cases.json`
(pytanie: "czyje sa zloza wegla kamiennego pod moja dzialka"). Odpowiedź
Qwena (`reasoning_effort=low`) poprawna i dobrze ugruntowana: cytuje
dosłownie art. 10 ust. 1 (lista kopalin objętych własnością górniczą,
w tym węgiel kamienny) i art. 10 ust. 5 (własność górnicza przysługuje
Skarbowi Państwa), z poprawnym wnioskiem że właściciel gruntu nie ma
prawa do złoża.

Pełna bateria (teraz 18 przypadków): **17/18 OK**, 1 ZNANY problem
(`ochrona_danych_101` -- Krok 25/28, niezmieniony), zero nowych
regresji. Efekt uboczny: `ksh_210` (udokumentowana w Kroku 53 rezydualna
~2/3 skuteczność przy niezerowej temperaturze) w tym przebiegu akurat
przeszedł -- zgodne z jego znaną niedeterministyczną naturą, nie
regresja ani naprawa. 139/139 testów jednostkowych bez zmian (dane
referencyjne, nie kod).

## Krok 56 (2026-09-21, nowa sesja): osiemnasta dziedzina -- prawo wodne

Realizacja punktu 1 "Co dalej" (18. dziedzina), prowadzona samodzielnie
zgodnie z wcześniejszą autoryzacją użytkownika.

**Wybór dziedziny:** ostatni z trzech pierwotnych kandydatów z Kroku 52
bez oczywistego faworyta (obok lotniczego i geologiczno-górniczego,
wybranych jako 16. i 17.). Zweryfikowano wprost przez ELI API
(`DU/2017/1566`, Dz.U. 2017 poz. 1566), tytuł potwierdzony przed
pobraniem: "Ustawa z dnia 20 lipca 2017 r. - Prawo wodne" (status: akt
posiada tekst jednolity, tekst ujednolicony typu U dostępny, inForce:
IN_FORCE). Zastępuje uchyloną ustawę z 2001 r. (DU/2001/1229) -- ta
starsza ustawa świadomie NIE dodana, nieobowiązująca od 2018-01-01.

**Dane:** 585 artykułów bieżących, 1379 fragmentów. Historia dograna od
razu -- 3913 artykułów historycznych z 7 wersji (2018/2020/2021/2021/
2022/2023/2024 + bieżąca), indeks łącznie 4498 artykułów / 10464
fragmenty. Znany, bezpieczny wariant ostrzeżenia o niesparowanych
nawiasach `< >` (mechanizm Krok 45/47/54) wystąpił, ale programowo
potwierdzono: **zero** osieroconych znaków `<>[]` i **zero** duplikatów
etykiet artykułów w żadnym z 7 okien historycznych. `DOMAIN_DESCRIPTIONS`
w `scripts/prompt.py` uzupełniony od razu ("polskiego prawa wodnego").

**Dane fine-tune:** 20 przykładów treningowych + 4 walidacyjne, każdy
fakt zweryfikowany wprost w pobranym tekście ustawy przed napisaniem
(lekcja z Kroku 28/40) -- w tym flagowy przypadek własności wód: art.
211 (śródlądowe wody płynące/wody podziemne = własność Skarbu Państwa,
niezależnie od właściciela gruntu) vs art. 214 (stojąca woda w stawie/
rowie zasilanym WYŁĄCZNIE deszczówką/wodami gruntowymi na własnej
działce = własność właściciela gruntu) -- naturalny, ostry kontrast
analogiczny do KPA art.129/KP art.8/geo art.10, ale tu z dodatkową
warstwą (dwa przeciwstawne przepisy, nie jeden). Pozostałe przykłady:
zwykłe/powszechne/szczególne korzystanie z wód (art. 32-34), opłata za
zmniejszenie retencji terenowej (art. 269), zgłoszenia/zwolnienia
wodnoprawne (art. 394/395/400), zakazy przy wałach przeciwpowodziowych
(art. 176), strefy ochronne ujęć (art. 132), organ wykonujący prawa
właścicielskie Skarbu Państwa (art. 212/216).

**Trening LoRA** (`mlx_lm.lora`, 100 iteracji, batch 2, `--mask-prompt`,
`--save-every 25 --steps-per-eval 25`) -- ten sam wzorzec przeuczenia co
wszystkie 17 poprzednich dziedzin: val loss najlepszy na iter25 (0.648),
potem monotonicznie gorszy (0.875 -> 0.946 -> 0.998 na iter 50/75/100).
Checkpoint iter25 użyty jako `bielik11b-wodne-lora-iter25`, wpisany do
`DEFAULT_ADAPTER_PATHS` w `chat.py` oraz jako placeholder w
`chat_cuda.py`.

**Znaleziona i naprawiona luka recall (przed treningiem adaptera, przy
budowie baterii regresyjnej):** flagowe pytanie "czyja jest rzeka
przepływająca przez moją działkę" (docelowo art. 211) w ogóle nie
trafiało w top-8 domyślnego pipeline'u -- kosinusowo art. 211 był na
randze ~80, poza pulą 30 kandydatów do rerankingu (`RERANK_CANDIDATE_POOL`),
więc reranking/fuzja nie miały szansy go uratować. Bazowa reformulacja
`expand_query()` była też za ogólna ("Pytanie dotyczy własności rzeki
przepływającej przez działkę..."), nie pomagała. Ręczna, precyzyjna
prawnicza fraza ("czy śródlądowe wody płynące stanowią własność Skarbu
Państwa") od razu dawała art. 211 na #1 kosinusowo -- ten sam wzorzec co
VAT art.86 (Krok 14) i KSH art.210 (Krok 39-41). Naprawione metodą Kroku
28/41 (domenowym hintem słownikowym, `DOMAIN_QUERY_EXPANSION_HINTS['wodne']`
w `scripts/prompt.py`, wstrzykiwany tylko do kontekstu użytkownika przy
wywołaniach ekspansji dla tej jednej domeny -- strukturalnie niemożliwe,
żeby zaregresował inną domenę). Zweryfikowane: z hintem art. 211 ląduje
na randze #2 po prawdziwej ekspansji. Pełna bateria regresyjna
(22 przypadki po dodaniu `wodne_211_wlasnosc_wod_plynacych`): **22/22
przechodzi**, zero regresji na pozostałych 21.

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego, metodyka
Kroku 8) -- 3 scenariusze + `--as-of`:
- Sąsiad zasypał rów melioracyjny, zalewa ogród -- poprawnie cytuje art.
  234 ust. 1 pkt 1/ust. 2/ust. 3 (w tym trafna, spontaniczna wzmianka o
  5-letnim terminie z art. 234 ust. 5), identycznie z i bez adaptera.
- Duża działka (5000 m², 80% uszczelnienia) pod parking, opłata za
  retencję -- poprawnie stosuje próg z art. 269 ust. 1 pkt 1 (>3500 m²,
  >70%) do konkretnych liczb z pytania, poprawnie wyklucza zwolnienia z
  ust. 2/3 (drogi/koleje, kościoły) jako nieadekwatne dla parkingu.
- `--as-of 2019-06-01` (pytanie o własność rzeki) -- poprawnie dobiera
  historyczne okno 2018-11-08 -- 2020-02-26 i oznacza odpowiedź właściwą
  datą.

**Czwarty scenariusz -- znaleziony i precyzyjnie zdiagnozowany nowy
przypadek błędnego cytowania (świadomie NIE naprawiony, udokumentowany
jako known_issue):** pytanie o mały staw zasilany wyłącznie deszczówką z
dachu (bez połączenia z rzeką) -- Bielik (deterministycznie, identycznie
z i bez adaptera, zweryfikowane 2x) odpowiada, że nie trzeba ANI
pozwolenia, ANI zgłoszenia wodnoprawnego, cytując "art. 395 pkt 9". To
podwójnie błędne: (1) art. 395 pkt 9 dotyczy urządzeń pomiarowych służb
państwowych, nie stawów -- treść odpowiedzi (limit 5000 m²/3 m
głębokości) jest w rzeczywistości z **art. 394 ust. 1 pkt 9**, i (2)
poprawna konkluzja to "wymaga ZGŁOSZENIA wodnoprawnego", nie "nie wymaga
niczego". Kluczowe: bezpośrednio zweryfikowano przez wywołanie
`search_with_history()` z identycznym kontekstem co prawdziwy
`chat.py`, że **art. 394 z poprawną treścią BYŁ obecny w kontekście RAG**
(ranga #3 po rozszerzeniu zapytania) -- to NIE jest luka recall, model
miał dostęp do właściwego przepisu i mimo to przypisał jego treść
niewłaściwemu, sąsiadującemu numerowi artykułu. Prawdopodobna przyczyna:
art. 394 ("Zgłoszenia wodnoprawnego WYMAGA:") i art. 395 ("Pozwolenia...
ANI zgłoszenia NIE WYMAGA:") sąsiadują w tekście, mają niemal lustrzane
nagłówki o przeciwstawnym znaczeniu -- łatwy punkt pomyłki. To inny
mechanizm niż znany "Pattern 1" z Kroku 42/43 (tam cytowany artykuł w
ogóle nie istniał w zaindeksowanym akcie) -- tu treść jest w 100%
poprawna i faktycznie ugruntowana we właściwym fragmencie, tylko numer
artykułu przypisany błędnie do sąsiada o przeciwnym sensie. Qwen3.8-27B
(8-bit, ten sam kontekst RAG) cytuje poprawnie "art. 394 ust. 1 pkt 9"
niemal dosłownie -- spójne z wcześniejszymi obserwacjami (Krok 31/32) o
solidniejszej dyscyplinie cytowania większego modelu. Jeden potwierdzony
przypadek -- świadomie NIE naprawiany teraz (ten sam rygor co inne
jednopunktowe znaleziska w projekcie, np. Krok 42), udokumentowany w
`data/qwen_answer_reference_cases.json` (`wodne_394_staw_deszczowka_zgloszenie`,
`known_issue`) jako canary na przyszłość -- czeka na kolejny potwierdzony
przypadek tego samego wzorca (mylenie dwóch sąsiadujących, strukturalnie
podobnych, ale znaczeniowo przeciwstawnych przepisów) w przyszłej
dziedzinie, zanim rozważać ogólną naprawę.

**Sondy Qwena** (LM Studio, `qwen/qwen3.8-27b`, `reasoning_effort=low`)
wykonane na żywo w tej samej sesji (użytkownik miał uruchomione LM
Studio) -- ten sam wzorzec co Krok 52 -> Krok 53, tym razem od razu w
Kroku 56 zamiast osobnego kroku. **Uwaga proceduralna:** pierwsza sonda
przypadkowo trafiła na załadowany wariant 4-bit zamiast rekomendowanego
8-bit (Krok 34) -- użytkownik to zauważył i poprawił w trakcie sesji,
obie sondy powtórzone na potwierdzonym 8-bit. Dodano 2 przypadki do
`qwen_answer_reference_cases.json` (teraz 20): flagowy art. 211 (poprawny,
dobrze ugruntowany) i art. 394/395 (known_issue, opisany wyżej -- Qwen
odpowiada poprawnie, Bielik nie).

**Bateria regresyjna `expand_query`:** dodano `wodne_211_wlasnosc_wod_plynacych`
do `expand_query_reference_cases.json`. Pełna bateria: **22/22 przechodzi**,
zero regresji na pozostałych 21. 139/139 testów jednostkowych przechodzi
(zmiana w `prompt.py` to tylko nowy wpis w słowniku `DOMAIN_QUERY_EXPANSION_HINTS`
i `DOMAIN_DESCRIPTIONS`, oba pokryte istniejącymi testami parametryzowanymi
po domenie). README.md zaktualizowany o wpis `wodne` w tabeli dziedzin i
poprawioną liczbę dziedzin (17 -> 18 -- poprzedni wpis "wszystkie
szesnaście" w nagłówku sekcji był już nieaktualny od Kroku 54, poprawiony
przy okazji).

## Krok 57 (2026-09-26, nowa sesja): pełny przebieg baterii Qwena + `release_accelerator_cache()`

**Pełny przebieg `check_qwen_answer_regressions.py` (20 przypadków),
odłożony w Kroku 56 -- domknięty.** LM Studio, `qwen/qwen3.8-27b` 8-bit
(potwierdzone przez użytkownika po zajętości ~30 GB -- uwaga: API LM
Studio `/api/v0/models` raportuje `quantization: 4bit` także dla
załadowanego wariantu 8-bit, bo oba mają to samo id; nie ufać temu polu).
Wynik: **19/20 OK, 1 znany problem (`ochrona_danych_101_kara_wyciek_odpowiedz`,
Krok 25/28/29 -- zaakceptowany kompromis), zero nowych regresji.** Oba
przypadki `wodne` (art. 211, art. 394) przechodzą także w pełnym,
automatycznym przebiegu; `ksh_210` tym razem OK. Kod wyjścia 1 wynika
tylko z obecności znanego problemu (zgodnie z założeniem skryptu).

**Znalezisko w trakcie: proces baterii rósł do ~12 GB.** Nie była to
powtórka wycieku z Kroku 56 (modele są współdzielone, indeksy wszystkich
18 dziedzin to łącznie ~0,7 GB) -- `footprint` pokazał ~7,5 GB w
kategorii IOAccelerator, czyli cache alokatora PyTorch na MPS, który
zatrzymuje pamięć po obliczeniach embeddingów/rerankera zamiast ją
oddawać. Zmierzone osobno, bez LLM-a (sam retrieval z rerankingiem dla
pytań z baterii): bez zwalniania pamięć sterownika MPS rosła skokowo
3,4 -> 5,4 GB w pierwszym okrążeniu 20 pytań (w drugim już stała); z
`gc.collect()` + `torch.mps.empty_cache()` po każdym pytaniu -- stałe
~1,3 GB (= same wagi modeli). Naprawa: nowa funkcja
`rag_search.release_accelerator_cache()` (MPS lub CUDA), wywoływana po
każdym przypadku w obu bateriach (`check_qwen_answer_regressions.py`,
`check_expand_query_regressions.py`). Pełnego przebiegu baterii z
poprawką świadomie nie powtarzano (zmiana to jedno wywołanie po
przypadku, nie dotyka logiki testów; Qwen + bateria razem dawały już
żółtą presję pamięci na 64 GB). 139/139 testów jednostkowych przechodzi.

**Uwaga proceduralna (monitoring pamięci):** `footprint` przełącza
jednostkę z MB na GB powyżej ~10 GB -- mój prowizoryczny watchdog
parsował "12 GB" jako 12 MB i nie zadziałałby. Przy przyszłym
monitorowaniu normalizować jednostki. Dodatkowo: przy Qwenie
załadowanym z kontekstem 131k LM Studio zajmuje ~30-35 GB; bateria
używa krótkich promptów, więc mniejszy kontekst (16-32k) daje większy
margines.

## Krok 58 (2026-09-26, ta sama sesja): dziewiętnasta dziedzina -- prawo mieszkaniowe (`mieszkaniowe`)

Realizacja punktu "19. dziedzina" z "Co dalej", prowadzona samodzielnie
zgodnie z wcześniejszą autoryzacją użytkownika. Wybór: prawo mieszkaniowe
-- zaproponowane jako pierwsza "praktyczna, codzienna" dziedzina po serii
czysto branżowych pilotaży (lotnicze/geologiczne/wodne), z celowym,
umiarkowanym nakładaniem się na `cywilne` (najem w KC).

**Akty** (oba zweryfikowane wprost przez ELI API przed pobraniem,
status "akt posiada tekst jednolity", inForce: IN_FORCE):
- `ustawa_o_ochronie_praw_lokatorow` -- DU/2001/733, "Ustawa z dnia 21
  czerwca 2001 r. o ochronie praw lokatorów, mieszkaniowym zasobie gminy
  i o zmianie Kodeksu cywilnego", 78 artykułów (dużo numerów literowych,
  np. 19a-19s -- najem okazjonalny/instytucjonalny).
- `ustawa_o_wlasnosci_lokali` -- DU/1994/388, "Ustawa z dnia 24 czerwca
  1994 r. o własności lokali", 39 artykułów (art. 34-38 pominięte w
  tekście jednolitym -- przepisy zmieniające inne ustawy).

**Dane:** historia dograna od razu -- 773 artykuły historyczne (7 okien
dla każdej ustawy), indeks łącznie 890 artykułów / 1581 fragmentów.
Programowo potwierdzono: zero duplikatów etykiet artykułów i zero
osieroconych znaków `<>[]` w każdym z 14 okien historycznych. Znane,
niegroźne artefakty PDF (dzielenie wyrazów "nie ruchomości", "właś
ciciela"; przypis wstawiony w środek art. 19a; nagłówek rozdziału
doklejony na końcu art. 17/33) -- ten sam typ co w poprzednich
dziedzinach, nie naprawiane. `DOMAIN_DESCRIPTIONS` uzupełniony
("polskiego prawa mieszkaniowego").

**Retrieval przed treningiem** (8 potocznych pytań, sam RAG bez
ekspansji): 6/8 z właściwym artykułem na #1 (kaucja art. 6, zaległości
art. 11, podwyżka art. 8a, eksmisja zimą art. 16, najem okazjonalny
art. 19a, zaskarżenie uchwały art. 25). "Kto płaci za remont klatki" --
#1 art. 14 (koszty zarządu, wprost "wydatki na remonty") zamiast
zakładanego art. 13 -- trafienie merytorycznie lepsze, nie luka. Jedyna
realna luka: "tylko 3 mieszkania, czy musimy mieć zarząd" -- art. 19
(<=3 lokale -> przepisy KC o współwłasności) poza pulą 30 kandydatów, na
#1 art. 20 (odwrotny przypadek, >3 lokale). Formalna fraza daje #1 --
ten sam wzorzec co art. 211 wodne/KSH art. 210. **Ale** prawdziwa
ekspansja `expand_query()` Bielika (bez żadnego hintu) sama daje art. 19
na #3 -- hint w `DOMAIN_QUERY_EXPANSION_HINTS` NIE dodany (niepotrzebny).
Przypadek `mieszkaniowe_19_mala_wspolnota_zarzad` dodany do
`expand_query_reference_cases.json` jako strażnik.

**Dane fine-tune:** 20 train + 4 valid, każdy fakt zweryfikowany wprost
w pobranym tekście ustawy przed napisaniem (kaucja 12x/6x i waloryzacja,
najem okazjonalny + zgłoszenie do US w 14 dni, tryby wypowiedzenia z
art. 11 ust. 1/2/4/5, podwyżki art. 8a/9 ust. 1b, podział napraw art.
6a/6b z zastrzeżeniem art. 6f, protokół art. 6c, ulepszenia art. 6d,
wejście przy awarii art. 10, ochrona zimowa art. 16, lokal socjalny
art. 14 ust. 4 z wyjątkiem ust. 7, uchwały wspólnoty art. 23/25, zaliczki
art. 15, zebrania art. 30/32, mała wspólnota art. 19/20).

**Trening LoRA** (ten sam przepis co wszystkie poprzednie dziedziny) --
ten sam wzorzec przeuczenia: val loss 0.772 (iter1) -> **0.331 (iter25)**
-> 0.412 -> 0.419 -> 0.454; train loss -> 0.000 na iter100. Najniższy
val loss ze wszystkich dziedzin (wodne: 0.648) -- prawdopodobnie bardziej
powtarzalna, szablonowa struktura odpowiedzi. Checkpoint iter25 jako
`bielik11b-mieszkaniowe-lora-iter25`, wpięty do `chat.py`, placeholder
w `chat_cuda.py`.

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego):
- Hałaśliwy sąsiad-lokator w lokalu gminnym -- poprawnie art. 13 ust. 1
  (powództwo innego lokatora o rozwiązanie stosunku i opróżnienie).
- Były najemca nie wyprowadza się po wygaśnięciu umowy -- poprawnie art.
  18 ust. 1-2 (odszkodowanie = czynsz + uzupełniające).
- Kto płaci za remont dachu we wspólnocie -- poprawnie art. 12 ust. 2-3.
- `--as-of 2017-06-01`, kaucja przy najmie okazjonalnym -- poprawne okno
  historyczne (2016-09-02 -- 2018-06-26), poprawnie art. 19a ust. 4
  (sześciokrotność), zweryfikowane w tekście tego okna.

**Znalezisko (udokumentowane, NIE naprawiane):** adapter w 2 z 3
odpowiedzi dorzuca POBOCZNĄ uwagę z błędnym numerem przepisu, przy
poprawnej głównej odpowiedzi: (1) termin "14 dni dla najmu
instytucjonalnego" (prawdziwy, ale z art. 19i ust. 3) przypisany art.
19d, i sama uwaga o żądaniu opróżnienia jest dla zwykłego najmu nie na
temat; (2) sposób liczenia udziałów (treść z art. 3 ust. 3) przypisany
art. 3 ust. 2. Baza bez adaptera na tych samych pytaniach tych dopisków
nie robi (merytorycznie czysto, ale rozwlekle -- ucięta limitem
tokenów). Mechanizm podobny do Kroku 56 (poprawna treść pod numerem
sąsiedniego/podobnego przepisu), tu na poziomie ustępu i w dopisku, nie
w głównej tezie -- drugi punkt danych dla tego wzorca, ale nie ten sam
przypadek (Krok 56: baza też się myliła; tu tylko adapter). Świadomie
nie strojone -- zbyt mało danych, by odróżnić efekt adaptera od szumu.

**Bateria regresyjna `expand_query`:** 23/23 przechodzi (nowy
przypadek `mieszkaniowe_19_mala_wspolnota_zarzad`: rank #3), zero
regresji na pozostałych 22. Pierwszy pełny przebieg z
`release_accelerator_cache()` z Kroku 57 -- proces (razem z Bielikiem na
MLX) szczyt ~15 GB, na końcu ~12 GB, bez narastania. 139/139 testów
jednostkowych przechodzi.

**Bateria Qwena (`qwen_answer_reference_cases.json`) NIE rozszerzona o
`mieszkaniowe` w tym kroku** -- Qwen był już wyładowany z LM Studio (po
Kroku 57, na prośbę o zejście z żółtej presji pamięci). Do zrobienia
przy następnym uruchomieniu LM Studio (patrz "Co dalej").

## Krok 59 (2026-09-30): bateria Qwena rozszerzona o `mieszkaniowe`

Domknięcie długu z Kroku 58. Dodano 2 przypadki do
`data/qwen_answer_reference_cases.json` (20 -> 22), oba spoza zbioru
fine-tune i inne niż pytania z weryfikacji Kroku 58:
- `mieszkaniowe_18_odszkodowanie_bezumowne_zajmowanie` -- art. 18 ust. 1-2
  ustawy o ochronie praw lokatorów (odszkodowanie = czynsz + uzupełniające).
- `mieszkaniowe_wl_12_obciazenie_lokal_uzytkowy` -- art. 12 ust. 2-3 ustawy
  o własności lokali (obciążenie właściciela lokalu użytkowego uchwałą).
  (Art. 12 ustawy o ochronie praw lokatorów jest uchylony -- "art. 12" z
  Kroku 58 dotyczył ustawy o własności lokali.)

Uruchomione osobno przez `--case` na Qwen3.8-27B 8-bit: oba OK, odpowiedzi
przeczytane ręcznie -- cytaty dosłowne i poprawne, brak zmyślonych
przepisów. Pokrycie baterii: wszystkie 19 dziedzin.

## Krok 60 (2026-09-30, ta sama sesja): dwudziesta dziedzina -- prawo energetyczne (`energetyczne`)

Realizacja punktu "20. dziedzina" z "Co dalej", prowadzona samodzielnie
zgodnie z autoryzacją. Wybór: prawo energetyczne -- rekomendowane w Kroku
58 jako "czysty" pilotaż (słownictwo nie zachodzi na żadną istniejącą
dziedzinę), a jednocześnie z wieloma codziennymi pytaniami konsumenckimi.

**Akt** (zweryfikowany wprost przez ELI API przed pobraniem): `prawo_energetyczne`
-- DU/1997/348, "Ustawa z dnia 10 kwietnia 1997 r. - Prawo energetyczne."
(status: akt posiada tekst jednolity, inForce: IN_FORCE). Brakujące numery
58-61 i 63-67 potwierdzone w PDF jako pominięte (przepisy zmieniające).

**Znaleziony i naprawiony błąd parsera (`download_acts.py`):** nagłówki z
indeksem górnym PO sufiksie literowym ("Art. 5b¹.", w warstwie tekstowej
pypdf "Art. 5b1." albo "Art. 5a 1.") nie trafiały ani na `ARTICLE_SPLIT_RE`
(wymagał kropki zaraz po literze), ani na detektor pdfplumber (po literze
oczekiwał kropki, a trafiał na małą cyfrę) -- obie ścieżki pomijały je
ZGODNIE, więc liczby nagłówków się zgadzały i nic nie ostrzegało, a treść
17 artykułów (4e¹-4e³, 5a¹, 5b¹-5b⁴, 9d¹-9d³, 9e¹, 9h¹-9h³, 9s¹, 43f¹) była
po cichu doklejana do poprzedników (np. cała regulacja umów agregacji z
art. 5a¹ na końcu art. 5a). Wykryte przy ręcznym czytaniu art. 5a przed
pisaniem danych fine-tune. Naprawa: nowa `recover_letter_superscript_headers()`
(cyfra po literze w numerze artykułu to w polskiej numeracji wyłącznie
indeks górny -> od razu znak Unicode + wymuszony początek linii),
`ARTICLE_SPLIT_RE` akceptuje końcowe znaki indeksu górnego, a detektor
pdfplumber zbiera małe cyfry po sufiksie literowym (geometria potwierdzona:
8 pt vs 12 pt tekstu). Skan wszystkich pobranych PDF-ów: wzorzec występuje
WYŁĄCZNIE w prawie energetycznym -- żadna inna dziedzina się nie zmienia.
2 nowe testy w `tests/test_download_acts.py`. Wynik: 315 -> 332 artykuły
bieżące, 2084 -> 2150 historyczne; w każdym z 10 okien historycznych zero
duplikatów etykiet i zero osieroconych `<>[]` (w 4 oknach znane ostrzeżenie
o niezgodnej liczbie nagłówków pdfplumber/pypdf -- było też przed poprawką).

**Dane:** indeks z historią 2482 artykuły / 9386 fragmentów.
`DOMAIN_DESCRIPTIONS` uzupełniony ("polskiego prawa energetycznego").

**Retrieval przed treningiem** (11 potocznych pytań, sam RAG): 9/11 z
właściwym artykułem na #1 (odcięcie za zaległości 6b, zmiana sprzedawcy
4j, odmowa przyłączenia 7, nielegalny pobór 57, koncesja 32, taryfy 47,
kontrola 6, licznik przedpłatowy 6a, agregator 5a¹ -- ten ostatni istnieje
jako osobny artykuł dopiero po poprawce parsera), 2 na #3 (reklamacja 6c
za sąsiednim 6d; dodatek energetyczny 5c za 5ga/5d). Hint domenowy
niepotrzebny.

**Dane fine-tune:** 20 train + 4 valid, każdy fakt zweryfikowany w
pobranym tekście (6b ust. 1-3/3a-3e/5, 6c, 4j ust. 3/3a-3c/4/6/6a/7, 5a,
5c, 5d, 6, 8, 32, 47/23, 31a/31d, 5ad/5ae/5af). Jedna poprawka przed
treningiem: zimowy zakaz odcięcia dla odbiorcy wrażliwego (art. 6b ust.
3d) to "rozwiązanie alternatywne" stosowane NA WNIOSEK (ust. 3c) -- pierwsza
wersja przykładu błędnie przedstawiała go jako automatyczny. Tematy 6a/6d/6f
i 57 celowo pominięte (zostawione do weryfikacji end-to-end).

**Trening LoRA** (ten sam przepis) -- ten sam wzorzec przeuczenia: val loss
0.962 (iter1) -> **0.596 (iter25)** -> 0.786 -> 0.852 -> 0.898; train loss
-> 0.000. Checkpoint iter25 jako `bielik11b-energetyczne-lora-iter25`,
wpięty do `chat.py`, placeholder w `chat_cuda.py`.

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego):
- Nielegalny pobór przez przeróbkę poprzedniego najemcy -- poprawnie art.
  57 ust. 1 pkt 1 z wyjątkiem "wyłącznej winy osoby trzeciej".
- Licznik przedpłatowy po dwóch opóźnieniach -- poprawnie art. 6a ust. 1
  pkt 1, ust. 2 (koszt po stronie przedsiębiorstwa), ust. 3 (brak zgody ->
  wstrzymanie/rozwiązanie umowy), plus trafnie art. 6f dla odbiorcy wrażliwego.
- Wznowienie gazu po reklamacji -- poprawnie art. 6d ust. 1 (3 dni) i ust.
  3, ALE w dopisku art. 6b ust. 4 opisany jako "np. nielegalny pobór" (w
  rzeczywistości: instalacja zagrażająca życiu; nielegalny pobór to 6b ust.
  1 pkt 1). Baza bez adaptera na tym samym pytaniu cytuje 6b ust. 4 bez
  błędnego opisu.
- `--as-of 2019-06-01`, limit kWh dodatku energetycznego dla osoby samotnej
  -- poprawnie 900 kWh (art. 5c ust. 3 pkt 1), zweryfikowane w tekście okna
  2019-04-03 -- 2020-05-11.

**Bateria `expand_query`:** +2 przypadki (`energetyczne_6c_reklamacja_wstrzymanie`
-- ranga #4, `energetyczne_5a1_agregator_zgoda_sprzedawcy` -- strażnik
poprawki parsera). Pełna bateria **25/25**, zero regresji. **141/141** testów
jednostkowych. Bateria Qwena NIE rozszerzona (Qwen wyładowany z LM Studio,
jak w Kroku 58) -- patrz "Co dalej".

## Krok 61 (2026-09-30, ta sama sesja): systematyczne porównanie z/bez adaptera + luka recall art. 32 `wodne`

Realizacja punktu z "Co dalej" po trzecim punkcie danych błędnych dopisków
adaptera (Krok 58 x2, Krok 60). Nowy skrypt `scripts/compare_adapter_vs_base.py`:
na dziedzinę model ładowany raz z adapterem; na pytanie jedna ekspansja (bazą,
jak w `chat.answer`) + jedno wyszukiwanie, potem dwie generacje z tym SAMYM
kontekstem -- LoRA włączona/wyłączona (`set_lora_enabled`). Wyniki w
`data/adapter_vs_base_comparison.json`. 4 dziedziny x 6 pytań spoza zbiorów
treningowych (`mieszkaniowe`, `energetyczne`, `wodne`, `lotnicze`), wszystkie
48 odpowiedzi sprawdzone ręcznie z tekstem ustaw.

**Wynik (błędy w cytowaniu / przypisaniu treści):**
- Automatycznie: adapter ANI RAZU nie zacytował artykułu spoza kontekstu RAG
  (baza 1x -- i to poprawnie, z treści art. 93a).
- Ręcznie, adapter: art. 6b ust. 4 opisany jako "nielegalny pobór" (Krok 60,
  deterministycznie powtórzone); "art. 389 pkt 15" (nie istnieje -- "15)" to
  znacznik PRZYPISU w tekście PDF wewnątrz art. 389; konkluzja poprawna);
  "art. 14 ust. 1 pkt 1" ustawy o własności lokali (art. 14 nie ma ustępów --
  kosmetyczne); zniekształcona parafraza art. 57 ust. 1 i błędnie zastosowany
  pkt 2 ("odszkodowanie od osoby, która dokonała przeróbki"); art. 395 pkt 9
  zamiast 394 ust. 1 pkt 9 (znany `known_issue` z Kroku 56).
- Ręcznie, baza: art. 395 pkt 9 (TEN SAM błąd -- potwierdza Krok 56, że to
  nie efekt adaptera); art. 34 opisany jako "zwykłe korzystanie" (to
  szczególne -- zwykłe jest w art. 33); art. 100 Prawa lotniczego rzekomo o
  "locie bez ważnej licencji" (nie ma tego w art. 100; to art. 211 ust. 1 pkt 5).
- Bilans: adapter ~5 (w tym 1 kosmetyczny), baza ~3. Hipoteza "adapter
  dorzuca odwołania do sąsiednich przepisów" potwierdzona SŁABO -- tylko 1
  z błędów adaptera ma dokładnie ten kształt; pozostałe to inne mechanizmy
  (nieistniejące jednostki redakcyjne, zniekształcenie treści w obrębie
  właściwego artykułu). Zbyt mała próba (24 pytania), żeby zmieniać przepis
  na dane fine-tune na tej podstawie.
- **Wyraźniejszy, nowy wzorzec: adapter formułuje pewne siebie, binarne
  werdykty** (styl "Tak/Nie -- zgodnie z art. ..." z danych treningowych), baza
  częściej zastrzega. Przykłady: "Nie" przy wiadrze z rzeki (błędnie, patrz
  niżej), "Tak" przy lądowaniu na własnej łące oparte wyłącznie na lądowaniu
  awaryjnym z art. 93a ust. 1 pkt 1 (baza poprawnie wylicza wszystkie 7
  przypadków; żaden wariant nie wskazał art. 93 ust. 6 -- lądowisko używane
  do 14 dni w roku za zgodą posiadacza), "nie musisz płacić" przy art. 57
  (baza: "można argumentować..."). Baza bywa natomiast rozwlekła i ucinana
  limitem 500 tokenów.

**Znaleziona i naprawiona luka recall (wodne, art. 32):** "Czy mogę czerpać
wodę wiadrem z rzeki, żeby podlewać ogródek?" -- art. 32 (powszechne
korzystanie z wód) na randze 63, poza pulą rerankingu; w kontekście był
tylko art. 34 (szczególne), więc oba warianty improwizowały. Naprawa
metodą Kroku 28/41/56: drugi przykład w `DOMAIN_QUERY_EXPANSION_HINTS['wodne']`,
celowo na INNYM pytaniu niż testowe (jezioro/"nabrać wody"). Ranga 63 -> #2;
pełna bateria `expand_query` **26/26** (nowy przypadek
`wodne_32_wiadro_powszechne_korzystanie`; art. 211 #2 -> #4, w progu).

**Test generalizacji hintu** (pytanie użytkownika: czy hinty działają na
niesprawdzanych pytaniach) -- 9 nowych pytań, ranga bez ekspansji / ekspansja
bez hintu / z hintem: parafrazy art. 32: konewka 17->4, mycie roweru
None->2, napojenie krów None->**21** (pytanie o gospodarstwo ROLNE -- model
nie przeniósł słownictwa hintu; częściowa porażka); parafrazy art. 211:
None->4, None->1; inne zakątki ustawy (pomost 394, wał 176, retencja 269,
studnia 395): bez szkody, pomost nawet 3->1. Wniosek: hint generalizuje na
parafrazy TEGO SAMEGO pojęcia, jeśli są wystarczająco bliskie przykładowi;
nie naprawia innych luk dziedziny i nie jest gwarancją.

141/141 testów jednostkowych.

## Krok 62 (2026-09-30, ta sama sesja): bateria Qwena rozszerzona o `energetyczne`

Domknięcie długu z Kroku 60. +2 przypadki w `data/qwen_answer_reference_cases.json`
(22 -> 24), uruchomione osobno przez `--case` na Qwen3.8-27B 8-bit:
- `energetyczne_6d_wznowienie_po_reklamacji` -- art. 6d ust. 1-3 (3 dni na
  wznowienie). To samo pytanie, na którym adapter Bielika błędnie opisał art.
  6b ust. 4 jako "nielegalny pobór"; Qwen opisuje go poprawnie ("bezpośrednie
  zagrożenie życia, zdrowia lub środowiska") -- kolejne potwierdzenie
  solidniejszej dyscypliny cytowania większego modelu (Krok 31/32/56).
  `expect_not_contains` łapie ten konkretny błąd, gdyby pojawił się u Qwena.
- `energetyczne_6a_licznik_przedplatowy_brak_zgody` -- art. 6a ust. 1 pkt 1,
  ust. 2, ust. 3, cytowane dosłownie.
Oba OK, odpowiedzi przeczytane ręcznie. Bateria Qwena pokrywa znów wszystkie
20 dziedzin.

## Krok 63 (2026-09-30, ta sama sesja): przypisy dolne, równoległe parsowanie PDF, przyrostowe indeksy RAG

**Problem:** przypisy dolne Kancelarii Sejmu ("15) Zmiany tekstu jednolitego
wymienionej ustawy zostały ogłoszone w...", "1) Niniejsza ustawa wdraża:
1) dyrektywę...") `pypdf` wypluwał jako zwykłe linie na końcu tekstu strony,
więc trafiały do środka artykułu łamanego przez stronę -- 129 bieżących
artykułów w 28 ustawach, 1377 w historii. Realna szkoda (Krok 61): numer
przypisu wyglądał jak punkt wyliczenia, adapter zacytował nieistniejący
"art. 389 pkt 15" Prawa wodnego; do tego szum w embeddingach (fragment z
listą dyrektyw UE "podobny" do pytań o dyrektywy).

**Naprawa (`download_acts.py`):** `find_footnote_markers_pdfplumber()` --
z geometrii PDF numer (indeks górny ~6,5 pt, < 70% czcionki treści) i
początek treści przypisu (10 pt) na dole strony, bez linii 12 pt pod nim;
`strip_page_footnotes()` obcina tekst strony z pypdf od linii "{numer})" +
ten początek treści (porównanie bez spacji). Dwie poprawki po drodze, obie
złapane weryfikacją: (1) przypis na 1. stronie sam jest wyliczeniem ("1)
Niniejsza ustawa wdraża: 1) dyrektywę...") -- cięcie od ostatniej linii "1)"
zostawiało jego nagłówek, stąd dopasowanie po początku treści; (2) próg
"numeru" 85% łapał też treść przypisu (10 pt = 83%), gdy numer i treść
leżały w jednym wierszu -- w historii usuwało to tylko ~60% przypisów;
próg 70% -> 97%. **Wynik: bieżące teksty 129 -> 0, historia 1377 -> 46**
(z 86 623). Weryfikacja: migawka parsera przed/po na wszystkich 29 PDF-ach
-- liczba i etykiety artykułów bez zmian, każda zmiana treści to wyłącznie
usunięcie słów (sprawdzone podciągiem tokenów; 56 wyjątków to nowe wydania
PDF ustawy o PIT i zasiłkowej z 16.09.2026 -- inne łamanie słów, nie
parser).

**Skutki uboczne pełnego przetworzenia (wykryte przy porównaniu z kopią):**
- `ochrona_srodowiska`, okno 2016-04-15: wcześniej BRAKOWAŁO 148 artykułów
  (od art. 286a, ~123 tys. znaków) -- prawdopodobnie znak "<"/">" z
  przypisu parował się z nawiasem dalej i filtr tekstu nieobowiązującego
  (Krok 45) wycinał wszystko pomiędzy. Teraz obecne; jedyny nowy duplikat:
  dwie wersje art. 286 ("ust. 2 i 4" vs "ust. 2 i 7" -- druga to zapewne
  niesparowany tekst przyszły). Zostawione jako znany drobiazg.
- `lotnicze`: indeks z Kroku 52 był zbudowany BEZ `--include-history`
  (1393 fragmenty, 0 z datami) -- `--as-of` po cichu nie działał; teraz
  11 657 fragmentów z historią. README poprawione.
- Nowe teksty jednolite w API od poprzedniego pobrania (9 aktów, m.in.
  Kodeks pracy -- nowe okno historyczne od 2026-08-27, ustawa o PIT,
  zasiłkowa, VAT, Ordynacja podatkowa) -- zaktualizowane `data/raw/*_meta.json`.
- Duplikaty etykiet w 13 historycznych oknach (np. KC 2013: "221" = 22¹
  spłaszczone, gdy korekta pdfplumber się wyłącza przez niezgodną liczbę
  nagłówków) -- ISTNIAŁY JUŻ WCZEŚNIEJ, identyczne przed i po; osobny,
  stary problem (patrz "Co dalej").

**Równoległe parsowanie PDF (`ParsePool`, oba skrypty pobierające):**
parsowanie (pypdf + 5 przebiegów pdfplumber, czysty Python) w osobnych
PROCESACH -- wątki nie dałyby nic przez GIL. Liczba procesów:
`default_parse_workers()` = rdzenie - 1, ograniczone WOLNĄ pamięcią
(`psutil`; zmierzony szczyt jednego procesu 2,36 GB na Prawie lotniczym ->
3 GB na proces) i liczbą zadań; `--workers N`. Pobieranie z API zostaje
sekwencyjne (1 s przerwy), parsowanie pobranych PDF-ów idzie w tle.
29 bieżących ustaw: ~30 min -> ~4 min (11 procesów); historia (~200
wersji): ~18 min, ograniczona pobieraniem.

**Przyrostowe indeksy RAG (`scripts/embedding_cache.py`):** trwały cache
wektorów w SQLite (`data/processed/embedding_cache.sqlite`), klucz = model +
tryb + sha256 tekstu fragmentu; deduplikacja w obrębie przebiegu (206 tys.
fragmentów 20 indeksów = 126 tys. unikalnych tekstów -- ten sam artykuł w
wielu wersjach historycznych); liczenie porcjami po 512 ze zwolnieniem
bufora GPU (stałe ~1,1 GB RAM) i strażnikiem WOLNEJ pamięci (paczka 32 ->
8 poniżej 8 GB; poniżej 3 GB czekanie, potem czytelne przerwanie --
policzone wektory zostają w cache'u, ponowne uruchomienie wznawia).
`--no-cache`, `--seed-cache-from-index`. **Bezstratność potwierdzona
wprost:** `wodne` z cache'em vs `--no-cache` -- max różnica 0,0 (macierze
identyczne bit w bit). Etap indeksów pełnej przebudowy: ~57 min -> ~4,5 min
(drugi przebieg liczył tylko fragmenty zmienione przez ostrzejszy próg).

**Wspólny venv:** 26 wrapperów (`pip`, `activate`, `pdfplumber`...) wciąż
wskazywało nieistniejącą ścieżkę do starego katalogu siostrzanego projektu (ten sam
mechanizm co w Kroku 54) -- podmieniona na `bielik-venv-shared`; `psutil`
doinstalowany (opcjonalna zależność w `requirements-common.txt`).

**Projekt kadr i płac** (na prośbę użytkownika; wciąż używany): przeniesione
przypisy, `ParsePool`, cache embeddingów i testy (commity `48a8c2c`,
`36a0e86`); remote przełączony z SSH (brak klucza) na HTTPS; dane
przetworzone -- przypisy w ustawach 45 -> 0, indeks 5770 -> 5731 fragmentów.
Pliki serii (rozporządzenia płacowe, limity ZUS) NIE przetwarzane ponownie
-- osobne skrypty, zawierają podobne przypisy.

**Przerwa:** w trakcie drugiego przebiegu zanik prądu (wyłączony
bezpiecznik) -- restart maszyny wyczyścił `/tmp` (kopie zapasowe w
scratchpadzie); kontrola po restarcie: `git fsck` czysty w obu repo, każdy
JSON/`.npy` się wczytuje, oba cache SQLite `integrity_check = ok`;
przebudowa powtórzona od zera.

**Weryfikacja końcowa:** bateria `expand_query` **26/26** na nowych indeksach
(rangi identyczne w obu przebiegach; `vat_86_zakupy_prywatne` na granicy --
#8, patrz "Co dalej"; `kpa_129` #4, `ochrona_danych_101` #6, reszta #1-#3).
Testy: **151/151** (prawo polskie), **96/96** (kadry). Bateria Qwena NIE
uruchomiona (LM Studio wyłączone) -- do zrobienia przy najbliższym
załadowaniu Qwena, bo zmienił się tekst części fragmentów w kontekście.

## Krok 64 (2026-09-30 / 2026-10-02): korekta indeksu górnego przez lokalne dopasowanie

**Problem (stary, opisany w Kroku 63):** gdy liczba nagłówków artykułów z
pdfplumber różniła się od liczby artykułów z pypdf (np. jeden nagłówek
wykryty tylko przez jedno narzędzie), korekta indeksu górnego była
pomijana W CAŁOŚCI -- art. 22¹ zostawał jako "221" i kolidował z
prawdziwym art. 221. 13 historycznych okien, 75 zdublowanych etykiet (KC
2013: 28, ustawa o min. wynagrodzeniu 2015: 15, KP najstarsze okno: 9).

**Naprawa (`download_acts.py`):** `align_superscript_corrections()` --
obie sekwencje dopasowane `difflib.SequenceMatcher` po SPŁASZCZONEJ
etykiecie z pdfplumber ("22¹" -> "221"); poprawiane są tylko zgodne pary,
więc rozjazd w jednym miejscu nie blokuje korekty w pozostałych. Przy
zgodnej liczbie nagłówków zachowanie bez zmian (pozycyjne). 3 nowe testy
(rozjazd liczby, brak indeksów górnych, notacja nawiasowa z literą "18³d").

**Wynik:** duplikaty etykiet w historii **75 -> 27**, 784 artykuły
przeetykietowane (np. KC 2013 "221" -> "22¹"). Pozostałe 27 to INNE
mechanizmy: ustawa o min. wynagrodzeniu 2015 -- art. 10-24 podwojone (nie
indeks górny), najstarsze okno `lotnicze`, pojedyncze przypadki. Dane
przetworzone ponownie, wszystkie 20 indeksów przebudowane (cache
embeddingów z Kroku 63: przeliczonych tylko 446 fragmentów).

**Projekt kadr i płac:** ta sama poprawka + przetworzone pliki serii
(rozporządzenia płacowe, limity ZUS -- zaległość z Kroku 63): zmiana
wyłącznie usunięcie przypisów, kwoty nietknięte, doszło nowe
rozporządzenie o min. wynagrodzeniu na 2027 (commit `58ebdbd`, wypchnięty).

**Weryfikacja:** pełna bateria Qwena (pierwsza po Kroku 63, 24 przypadki):
23/24 OK + 1 znany `known_issue`, bez regresji. Bateria `expand_query`:
**26/26** (rangi jak po Kroku 63, `vat_86_zakupy_prywatne` nadal #8 --
na granicy, ale stabilnie). Testy: 154/154.

## Krok 65 (2026-10-02, ta sama sesja): pozostałe duplikaty etykiet w historii -- trzy mechanizmy + dwa znaleziska po drodze

Punkt z "Co dalej" po Kroku 64: 27 zdublowanych etykiet w 6 oknach
historycznych. Każde okno sprawdzone na surowym PDF-ie -- trzy różne
przyczyny.

**1. Przepisy przytoczone w obwieszczeniu (16+16 artykułów).** PDF-y typu
"O" (obwieszczenie: ustawa o min. wynagrodzeniu 2015, Prawo lotnicze 2006)
zawierają przed tekstem jednolitym samo obwieszczenie, a w nim przepisy
NIEOBJĘTE tekstem jednolitym ("nie obejmuje art. 10-24 ..., które
stanowią: Art. 10. ..."). Parser brał je za artykuły ustawy (w tym
nieistniejący w tej ustawie "art. 85"). `strip_announcement_preamble()`
rozpoznaje obwieszczenie po nagłówku "Załącznik do obwieszczenia ...
USTAWA" i tnie za OSTATNIM podpisem "Marszałek Sejmu: ..." przed nim --
nie na samym nagłówku, bo na dwukolumnowej stronie z 2006 r. pypdf
wypisuje art. 1 PRZED nagłówkiem (pierwsza wersja poprawki gubiła art. 1,
złapane od razu). Wzorce tolerują zepsute kodowanie 2006 ("Za∏àcznik",
"Marsza∏ek"). Okna: 42 -> 26 i 271 -> 255 artykułów.

**2. Zgubione zamykające ">" w samym PDF-ie Kancelarii (POŚ 2016 art. 286,
ruch drogowy 2017 art. 80b/80bh/80c/80ca).** Potwierdzone w surowym
wyjściu pypdf -- znaku nie ma w źródle. Niesparowany "<" nic nie wycinał,
więc przyszła wersja zostawała obok obowiązującej. `strip_not_yet_in_force_text`
+ `_unclosed_future_article_ends()`: niesparowany "<" na początku nagłówka
("\n<Art. N.") zamyka się na najbliższym nagłówku artykułu -- wycinamy
najmniej, co się da (blok przyszły bywa wieloartykułowy, 80ca-80ch, ale
bez ">" nie wiadomo, gdzie się kończy; bezpiecznik z Kroku 45). **Wyjątek
złapany porównaniem przed/po:** Prawo energetyczne 2024 art. 5ab --
obowiązująca wersja otwarta literówką "<" i zamknięta "]" (dalej osobno
właściwa przyszła "<Art. 5ab. ...>"); pierwsza wersja reguły wycinała
OBOWIĄZUJĄCY art. 5ab. Teraz artykuł z niesparowanym "]" zostaje.

**3. Zostawione (jednostkowe, bez bezpiecznej reguły):** Prawo budowlane
2000 art. 54 (w PDF-ie brak nawet otwierającego "<" przyszłej wersji),
Ordynacja 2018 art. 97a ("(uchylony)" + nowa treść bez żadnych
znaczników). **Duplikaty etykiet w historii: 27 -> 2.**

**Znalezisko A: stempel daty wydruku w treści artykułów.** Teksty
ujednolicone ~2017-2021 mają co stronę datę wygenerowania w formacie
"09.05.2019" (filtr znał tylko "RRRR-MM-DD") -- trafiała do treści i
embeddingów: **15 989 -> 1 475** wystąpień dat "DD.MM.RRRR" w historii
(reszta to prawdziwe daty, np. dyrektyw UE "z 02.12.2000"), ~10,5 tys.
artykułów zmienionych wyłącznie usunięciem stempla. Nowy wzorzec w
`HEADER_FOOTER_PATTERNS` (tylko cała linia). **Druga regresja złapana
porównaniem:** pdfplumber nadal widział stempel w "odcisku palca" korekty
indeksu górnego ustępu, więc po usunięciu go z `clean_text` 21 korekt
cicho przepadło (np. KC art. 831 "§ 1¹" -> "§ 11") -- ten sam wzorzec
dopisany do `FOOTER_TOKEN_RE`. Bilans indeksów górnych po poprawce:
13 368 -> 13 367 (KP okno 2018/917: dwa cytaty "186⁸a"/"25³" w prozie
przesunięte/zgubione, jeden zyskany -- w granicach znanej ~85%
skuteczności tej korekty, Krok 48/49).

**Znalezisko B: id "@current" w starych oknach.** Okna bez znanego
początku (`valid_from=None`, 27 okien w 16 ustawach, 5726 artykułów)
dostawały sufiks id "@current" -- mylący, a przy kilku takich oknach
jednej ustawy (Prawo budowlane: 4, KPA: 2) powtarzający się. Bez wpływu na
odpowiedzi (id/`chunk_id` nie służy do wyszukiwania), ale poprawione:
sufiks "do_{valid_to}".

**Weryfikacja:** bieżące teksty identyczne co do znaku (wszystkie zmiany
dotyczą tylko historii); historia 86 623 -> 86 586 artykułów, każda różnica
etykiet sklasyfikowana (wyłącznie oczekiwane usunięcia), każda zmiana
treści to usunięcie stempla albo 3 opisane wyżej indeksy. 20 indeksów
przebudowanych. Testy: 8 nowych, **162/162**. Bateria `expand_query`:
**26/26** (`vat_86_zakupy_prywatne` nadal #8 -- bez zmian po Krokach 63-65).

**Projekt kadr i płac:** te same podwojone art. 10-24 ustawy o min.
wynagrodzeniu (okno 2015) i stemple dat (okna 2017, 2018) -- przeniesione
`strip_announcement_preamble`, wzorzec stempla i sufiks id. Poprawka "<"
bez ">" NIE przeniesiona: kadry mają jeszcze regexowy filtr nawiasów
sprzed Kroku 47 (inna mechanika, brak tam potwierdzonego przypadku).

## Krok 66 (2026-10-10): dwudziesta pierwsza dziedzina -- prawo komunikacji elektronicznej (`telekomunikacyjne`)

Realizacja punktu "21. dziedzina" z "Co dalej", samodzielnie zgodnie z
autoryzacją (wcześniej w tej sesji, poza tym projektem: historia
wszystkich ustaw w indeksie projektu kadr -- patrz jego PROGRESS.md).
Wybór: prawo komunikacji elektronicznej -- dużo codziennych pytań
konsumenckich (umowy z operatorem, przeniesienie numeru, reklamacje,
cookies), słownictwo rozłączne z istniejącymi dziedzinami; prawa
konsumenta odrzucone (zachodzi na rękojmię w KC).

**Akt** (zweryfikowany wprost przez ELI API): `prawo_komunikacji_elektronicznej`
-- DU/2024/1221, "Ustawa z dnia 12 lipca 2024 r. - Prawo komunikacji
elektronicznej" (obowiązujący, IN_FORCE, w mocy od 2024-11-10). **Stan
prawny:** brak obwieszczeń o tekście jednolitym (ustawa za młoda), więc
pobrany tekst ujednolicony "U" (changeDate 2026-10-06), a historia jest
pusta -- `--as-of` sprzed 2024-11-10 nie znajdzie tej ustawy, co jest
poprawne (wcześniej obowiązywało Prawo telekomunikacyjne z 2004 r., NIE
dodane do indeksu).

**Parser bez zmian** -- 450 artykułów, numeracja ciągła 1-450, zero
duplikatów, zero indeksów górnych, zero osieroconych `<>[]`, zero
przypisów; 39 dat "DD.MM.RRRR" to wyłącznie prawdziwe odwołania do Dz.
Urz. UE. Najdłuższe artykuły (2 -- definicje, 444 -- kary, 24 -- opłaty)
sprawdzone ręcznie: nic nie zostało wchłonięte. Indeks: 1118 fragmentów.
`DOMAIN_DESCRIPTIONS`: "polskiego prawa komunikacji elektronicznej
(telekomunikacyjnego)".

**Retrieval przed treningiem** (12 potocznych pytań, sam RAG): 9/12 na
#1, 379 na #2, 325 na #3, 399 (cookies) na #7, **307 (podwyżka w trakcie
umowy) poza top-8**. Z `expand_query` (jak w produkcji): 307 -> #7, ale
399 -> **#15** (ustawa nigdzie nie mówi "cookies"/"ciasteczka", tylko
"przechowywanie informacji ... w telekomunikacyjnym urządzeniu
końcowym"). **Hint domenowy** w `DOMAIN_QUERY_EXPANSION_HINTS`
(`telekomunikacyjne`: cookies -> urządzenie końcowe; podwyżka ->
jednostronna zmiana warunków umowy na czas określony + wypowiedzenie bez
odszkodowania). Sprawdzony na 4 DODATKOWYCH sformułowaniach innych niż
przykład w hincie: 16/16 w top-3, 13 na #1 (cookies #1 na obu nowych,
307 #1 i #3). Hint jest domenowy, więc strukturalnie nie dotyka innych
dziedzin (Krok 28).

**Dane fine-tune:** 20 train + 4 valid (art. 301, 302, 304, 305, 306,
307, 308, 309, 325, 326, 378, 379, 383, 399; valid: 394, 381, 380, 397).
54 cytowania "art. N ust. M" sprawdzone automatycznie na pobranym tekście
(0 problemów) + ręcznie co do treści. Celowo pominięte (do weryfikacji
end-to-end): 382 (zaległości), 398 (automaty reklamowe), 318/319 (zmiana
dostawcy internetu).

**Trening LoRA** (ten sam przepis) -- ten sam wzorzec: val 1.117 (iter1)
-> **0.688 (iter25)** -> 0.861 -> 0.907 -> 0.937, train -> 0.000.
Checkpoint iter25 jako `bielik11b-telekomunikacyjne-lora-iter25`, wpięty
do `chat.py`, placeholder w `chat_cuda.py`.

**Weryfikacja end-to-end** (pytania spoza zbioru treningowego):
- Zaległość 2 tygodnie, czy mogą od razu odciąć -- poprawnie "nie",
  art. 382 (7 dni -> ograniczenie -> 3 dni -> zawieszenie, 21 dni po
  reklamacji); drobna nieścisłość numeru ustępu (termin 3 dni przypisany
  ust. 3 zamiast ust. 4).
- Automat z ofertą fotowoltaiki -- poprawnie art. 398 ust. 1/2/4.
- 5 dni bez internetu przy zmianie dostawcy -- poprawnie odszkodowanie od
  nowego dostawcy 1/4 opłat za dzień (art. 319 ust. 3, z wyjątkami), ale
  odpowiedź zaczyna od nieadekwatnej tu reguły 30 dni (art. 319 ust. 2).
- **Telefon z promocji, rezygnacja po 10/24 miesiącach -- BŁĄD**: Bielik
  stosuje art. 309 (zachowanie promocyjnego urządzenia), który dotyczy
  WYŁĄCZNIE wypowiedzenia z art. 307 ust. 2 lub art. 380; właściwy jest
  art. 304 ust. 1 i 3. **Baza bez adaptera robi ten sam błąd** (plus
  stosuje art. 302 do umowy nieprzedłużonej) -- więc to nie efekt
  adaptera, tylko wzorzec mylenia sąsiednich przepisów (Zasada 6): art.
  309 jest #1 w kontekście (dosłownie "promocyjne urządzenie końcowe"),
  art. 304 #7. Qwen na tym samym pytaniu odpowiada poprawnie z art. 304
  ust. 1/3 i sam zastrzega, że art. 309 może nie mieć zastosowania.

**Baterie:** `expand_query` +2 przypadki (`telekomunikacyjne_307_podwyzka_w_trakcie_umowy`
#3, `telekomunikacyjne_399_ciasteczka_zgoda` #1 -- strażnicy hintu), pełna
**28/28**. Qwen +2 (`telekomunikacyjne_382_zaleglosc_ograniczenie`,
`telekomunikacyjne_304_rezygnacja_telefon_promocyjny`), oba OK, odpowiedzi
przeczytane ręcznie (Qwen załadowany tylko na czas tych dwóch przypadków).
Obie baterie pokrywają wszystkie 21 dziedzin. Testy **162/162**.

## Krok 67 (2026-10-10, ta sama sesja): historia dla `telekomunikacyjne` -- uchylone Prawo telekomunikacyjne z 2004 r. + licencja Qwena

**Historia.** Prawo komunikacji elektronicznej nie ma obwieszczeń o
tekście jednolitym, więc `--as-of` sprzed 2024-11-10 nie znajdował nic.
Dodane jego poprzednie, uchylone Prawo telekomunikacyjne
(`prawo_telekomunikacyjne`, DU/2004/1800, status: uchylony, NOT_IN_FORCE,
`repealDate` 2024-11-10, 8 obwieszczeń). Bieżący tekst "U": 338
artykułów (luka 211-220 potwierdzona w PDF-ie jako "(pominięte)" --
przepisy zmieniające), historia: 2278 artykułów w 7 oknach (2014-01-09
-- 2024-01-10), zero duplikatów etykiet, zero `<>[]`.

**Nowa obsługa uchylonych ustaw (`build_rag_index.py`):** ELI daje
uchylonej w całości ustawie `expirationDate=None` i osobne `repealDate`.
Bez obsługi tego pola jej ostatni tekst "U" wyglądałby na obowiązujący
DZIŚ i konkurowałby z nową ustawą przy każdym zwykłym pytaniu.
`_repeal_date()` + w `load_articles_with_history()` bieżące artykuły
ustawy z `repealDate` dostają `valid_to=repealDate` (exclusive, jak
`expirationDate`). Tylko ścieżka `--include-history` (tam, gdzie już jest
analogiczne podbicie `entryIntoForce`; wszystkie dziedziny są budowane z
tą flagą). 1 nowy test.

**Weryfikacja granic** (2 pytania x 8 dat): bez daty, 2024-11-10 i 2025
-> wyłącznie nowa ustawa; 2015, 2020, 2024-06-01, 2024-11-09 -> wyłącznie
stara, właściwe okno; 2010 -> pusto (brak tekstu jednolitego sprzed 2014,
znane ograniczenie, Krok 18). Indeks 1118 -> 7374 fragmentów. Bateria
`expand_query` dla obu przypadków `telekomunikacyjne` bez regresji (307
#2, 399 #1).

**Znalezisko (NIE naprawione, osobne zadanie):** adnotacje redakcyjne
Kancelarii Sejmu z marginesu tekstów ujednoliconych ("Nowe brzmienie ust.
1 w art. 7 wejdzie w życie z dn. 12.12.2018 r. (Dz. U. z 2018 r. poz.
1118).") trafiają do treści artykułów -- ok. 600 wystąpień w historii (30
ustaw, najwięcej Prawo budowlane 2019: 77, upadłościowe 2019: 51) i 55 w
bieżących tekstach. Treść przepisów nietknięta (to dopisane zdanie
informacyjne), ale to szum w embeddingach i w kontekście modelu --
patrz "Co dalej".

**Licencja Qwena (pod kątem destylacji):** Qwen3.8-27B (oryginał
`Qwen/Qwen3.8-27B` na Hugging Face i używana kwantyzacja
`lmstudio-community/Qwen3.8-27B-MLX-8bit`) -- **Apache 2.0**, plik
LICENSE to standardowy tekst bez dodatkowych klauzul (żadnego zakazu
trenowania innych modeli na wyjściach, żadnego wymogu "Built with Qwen"
-- w przeciwieństwie do starszej "Qwen License" części modeli Qwen1/2).
Destylacja Qwen -> Bielik jest więc licencyjnie dozwolona (sprawdzone
2026-10-10; to nie porada prawna).

## Krok 68 (2026-10-10, ta sama sesja): dwudziesta druga dziedzina -- prawa konsumenta (`konsumenckie`) + POWAŻNE znalezisko w historii

**Akt** (zweryfikowany przez ELI API): `ustawa_o_prawach_konsumenta` --
DU/2014/827, "Ustawa z dnia 30 maja 2014 r. o prawach konsumenta"
(tekst jednolity, IN_FORCE, 6 obwieszczeń). Ryzyko z "Co dalej"
(zachodzenie na rękojmię w KC) jest w rzeczywistości odwrotne: od
2023-01-01 rozdział 5a (art. 43a-43q, niezgodność towaru z umową) ZASTĄPIŁ
dla konsumentów rękojmię z KC (art. 43a ust. 1 wprost wyłącza KC). Osobna
dziedzina tylko z tą ustawą -- więc nie miesza się z `cywilne`. (Uwaga na
przyszłość: pytanie konsumenta o wadę zadane w `cywilne` trafi w
nieaktualną dla niego rękojmię z KC -- poza zakresem tego kroku.)

**Parser bez zmian:** 74 artykuły (luka 44-50 potwierdzona w PDF-ie jako
"(pominięte)"), historia 321 artykułów w 5 oknach, zero duplikatów i
`<>[]`. Załączniki (wzór pouczenia o odstąpieniu, formularz) doklejone do
art. 55 -- ISTNIEJĄCY wzorzec w 8 ustawach (np. VAT art. 176: 49 tys.
znaków), tu z realnym skutkiem: art. 55 wygrywa rankingiem przy pytaniach
o "zwrot pieniędzy" (wzór pouczenia: "zwracamy Państwu wszystkie
płatności"). Patrz "Co dalej". Indeks 764 fragmenty.

**Retrieval (z `expand_query`, 12 pytań):** 10/12 w top-8; art. 28 (bieg
terminu) #12, art. 29 (brak pouczenia) #8. **Hint domenowy** `konsumenckie`
(zwrot -> odstąpienie od umowy zawartej na odległość; od kiedy liczę ->
bieg terminu; nie powiedzieli -> nie został poinformowany). Po teście
end-to-end dopisana czwarta para ("zwrot pieniędzy za wadliwy towar" ->
"obniżenie ceny albo odstąpienie z powodu braku zgodności towaru z umową")
-- art. 43e #12 -> #1 na 3 sformułowaniach. Końcowo 18/19 pytań w top-8;
jedno sformułowanie art. 28 ("allegro, od zamówienia czy od dostawy") #9
-- `known_issue` w baterii, celowo nie dostrajane pod jedno pytanie.

**Dane fine-tune:** 20 train + 4 valid (art. 27, 28, 29, 30, 33, 34, 35,
38 pkt 5/12, 43a, 43c, 43d, 43f, 43m; valid: 43d ust. 6, 43c ust. 2, 34
ust. 3, 43g ust. 2), 50 cytowań sprawdzonych automatycznie (0 problemów).
Odłożone do end-to-end: 32, 7a, 38 pkt 3, 43e.

**Trening:** val 0.912 -> **0.362 (iter25)** -> 0.433 -> 0.474 -> 0.519.
`bielik11b-konsumenckie-lora-iter25`, wpięty do `chat.py`/`chat_cuda.py`.

**End-to-end:** zwrot płatności po odesłaniu (art. 32 ust. 1/3) --
poprawnie; brak odpowiedzi na reklamację (art. 7a ust. 2) -- poprawnie;
stół na wymiar (art. 38 ust. 1 pkt 3) -- poprawnie. **Pralka cieknie po
naprawie -- adapter BŁĘDNIE "Nie"** (zarówno gdy art. 43e był poza
kontekstem, jak i PO naprawie retrievalu, gdy był w kontekście -- pomija
art. 43e ust. 1 pkt 3 i przeczy sam sobie w ostatnim zdaniu). **Baza bez
adaptera: poprawnie "Tak" z art. 43e.** Qwen: wzorcowo, wprost ust. 1 pkt
3 + domniemanie istotności (ust. 4). Najmocniejszy dotąd punkt danych dla
"adapter a pewność werdyktów" (Krok 61): tu adapter jest gorszy od bazy.

**Baterie:** `expand_query` +4 (31 przypadków: 30 OK + 1 `known_issue`),
Qwen +2 (28 przypadków, oba nowe OK, przeczytane ręcznie). Obie pokrywają
22 dziedziny. Testy **163/163**.

**POWAŻNE ZNALEZISKO (NIE naprawione -- czeka na decyzję): historia
pokazuje stan z KOŃCA okna, nie z początku.** `download_acts_history.py`
pobiera dla każdego obwieszczenia plik typu "U" (`...Lj.pdf`) -- a to
tekst ujednolicony Kancelarii Sejmu AKTUALIZOWANY o kolejne nowelizacje aż
do wygaśnięcia okna. Dowód: okno ustawy o prawach konsumenta "2020-01-21
-- 2023-12-22" (obwieszczenie DU/2020/287) zawiera rozdział 5a (art.
43a-43q), który wszedł w życie 2023-01-01; plik "U" ma stempel wydruku
**2023-10-19**, a pliki "O" (Dz.U.) i "T" (tekst Kancelarii obwieszczenia,
stempel 25.02.2020) art. 43a nie mają. Skutek: `--as-of 2021-...` zwraca
przepisy, które wtedy JESZCZE NIE obowiązywały -- dotyczy potencjalnie
wszystkich okien historycznych we wszystkich 22 dziedzinach i w projekcie
kadr (od Kroku 18). Kierunek naprawy: plik "T" (migawka z dnia
ogłoszenia, format Kancelarii -- obecne czyszczenie i
`strip_announcement_preamble` powinny działać) zamiast "U" dla okien
historycznych; potem pełne przetworzenie historii, porównanie przed/po,
przebudowa indeksów, przeniesienie do kadr.

## Krok 69 (2026-10-10, ta sama sesja): historia = stan z POCZĄTKU okna (pliki "T" zamiast "U") + czyszczenie układu Dziennika Ustaw

Naprawa poważnego znaleziska z Kroku 68: `download_acts_history.py` brał dla
każdego obwieszczenia plik "U" -- tekst ujednolicony Kancelarii
aktualizowany o nowelizacje aż do wygaśnięcia okna, więc każde okno
zawierało stan z KOŃCA okna (np. ustawa o prawach konsumenta, okno od
2020-01-21: rozdział 5a obowiązujący od 2023-01-01). Od Kroku 18.

**Zmiana źródła:** `HISTORY_TEXT_KINDS = ("T", "O", "U")` -- "T" (tekst
obwieszczenia, migawka z dnia ogłoszenia), potem "O" (oryginał Dz.U., ta
sama migawka); "U" tylko w ostateczności z ostrzeżeniem (0 przypadków).
Bieżący tekst ustaw nadal "U". Wynik: 229 wersji z "T", 18 z "O".
Weryfikacja wprost: `--as-of 2021-06-01` w `konsumenckie` -- zero art.
43a-43q (wcześniej były), 2024 i dziś -- rozdział 5a na górze.

**Pliki "T" mają układ Dziennika Ustaw -- dotychczasowe czyszczenie (pisane
pod "U") przepuszczało: ~20,6 tys. stopek "Dziennik Ustaw – N – Poz. X",
~24 tys. fragmentów przypisów, ~2,6 tys. odsyłaczy "§ 1.145)", 467
zdublowanych etykiet.** Pięć przebiegów, każda poprawka sprawdzona
pomiarem:
1. Detektor przypisów (Krok 63) odrzucał KAŻDĄ stronę, bo stempel daty
   "2024-01-05" stoi w "T" POD przypisami pełną czcionką -- wiersze
   nagłówka/stopki pomijane (`_is_header_footer_row`); próg treści 0.9 ->
   **0.95** (w "T" przypisy 9 pt przy 10 pt treści -- dokładnie na progu).
2. Starszy układ (~2006-2014): numer przypisu "2" sam w wierszu, ")" w
   wierszu treści -- sklejane.
3. Nagłówek "Dziennik Ustaw – 23 – Poz. 17" (też "Dziennik Ustaw Nr 51 —
   4521 — Poz. 307") w `HEADER_FOOTER_PATTERNS`.
4. `strip_inline_footnote_refs()`: odsyłacze sklejone z numerem jednostki
   ("§ 2.2)", "Art. 75a.46)", "(uchylony)46)", ustęp "4.13) Minister" --
   tylko przed wielką literą, bo "(PKWiU ex 22.11) – z" w VAT). Działa też
   na bieżących tekstach: **74 resztki usunięte, poza tym 0 zmian**.
5. Przypisy redakcyjne w dwukolumnowych oryginałach "O" (geometria ich nie
   łapie): `EDITORIAL_FOOTNOTE_START_RE` ("NN) [do 4 słów] W brzmieniu
   ustalonym przez / Dodany przez / Uchylony przez / Przez art. ...",
   literowe "d) Dodany przez" w 2006) + kontynuacje do kropki, maks. 6
   wierszy.
6. Preambuła obwieszczenia (Krok 65) nie była wycinana: podpis "MARSZAŁEK
   SEJMU" wielkimi literami + nazwisko w następnej linii, oraz fałszywa
   spacja pypdf "Zał ącznik"; bez podpisu -- cięcie na nagłówku załącznika.
7. `drop_future_wording_duplicates()` (tylko historia): "T" podaje artykuł
   zmieniony nowelizacją jeszcze niewchodzącą w życie DWA razy pod rząd --
   najpierw brzmienie obowiązujące (przypis "W tym brzmieniu obowiązuje do
   wejścia w życie zmiany..."), potem nowe. Sprawdzone na przypisach 5
   ustaw -- zostaje pierwsze.
8. `_get_with_retry()`: 3 ponowienia przy 5xx/zerwanym połączeniu -- jeden
   "502 Proxy Error" zabił pierwszy przebieg (~150 pobrań, zapis na końcu).

**Bilans historii (U -> T):** stopki 102 -> 1, "W brzmieniu ustalonym"
111 -> 3, odsyłacze 36 -> 1, adnotacje "wejdzie w życie z dn." **1830 ->
3** (pliki "T" nie mają notatek z marginesu -- przy okazji rozwiązuje
większość punktu z Kroku 67), fragmenty "o której mowa w odnośniku" 20 ->
164 (112 w jednym oknie: Prawo lotnicze 2006, zepsuty układ/kodowanie --
dalsze łatanie nieopłacalne), zdublowane etykiety 2 -> 12 (wszystkie w
najstarszych oknach bez `valid_from`, głównie PIT z obwieszczenia z 2000
r., którego ekstrakcja była słaba i przed zmianą). Bieżące teksty: poza
74 odsyłaczami i aktualizacjami Kancelarii (Ordynacja od 1.10.2026, Prawo
budowlane, restrukturyzacyjne -- 4 nowe artykuły Ordynacji) bez zmian.

**Weryfikacja:** 22 indeksy przebudowane; bateria `expand_query` 30/31 +
1 `known_issue` (bez regresji); testy **176/176** (+13 nowych).

## Krok 70 (2026-10-10, ta sama sesja): załączniki ustaw jako osobne fragmenty

Punkt z "Co dalej" (Krok 68): załączniki stoją w PDF-ie po ostatnim
artykule bez nagłówka "Art.", więc doklejały się do niego -- VAT art. 176
49 872 znaki, Prawo lotnicze art. 232 **99 344**, PIT art. 58 26 000, a w
`konsumenckie` art. 55 (wejście w życie + wzór pouczenia o odstąpieniu)
wygrywał rankingiem pytań o zwrot pieniędzy.

**`split_annexes()`** (koniec `parse_pdf_articles`, więc też historia):
odcina od ostatniego artykułu wszystko po "Załącznik(i) do ustawy z dnia
... (Dz. U. ...)" i dzieli na fragmenty z etykietą "załącznik nr N" (albo
"załącznik", gdy jeden nienumerowany). "Załącznik nr 617)" w VAT to nr 6
ze sklejonym odsyłaczem "17)" -- przy ")" numer brany jako kolejny w
sekwencji. `prompt.article_ref()`: w kontekście modelu i w `rag_search.py`
załącznik pokazuje się jako "załącznik nr 2", nie "art. załącznik nr 2".

**Wynik (bieżące teksty):** zmieniło się dokładnie 8 ostatnich artykułów
(np. lotnicze 232: 99 344 -> 173 znaki, VAT 176: 49 872 -> 632, prawa
konsumenta 55: 7 664 -> 75), doszło 36 załączników (VAT 15, lotnicze 9,
PIT 4, ruch drogowy 3, konsumenta 2, budowlane/POŚ/geologiczne po 1),
nic nie ubyło. Historia: 151 załączników. Nowa możliwość: "czy jest gotowy
wzór formularza do odstąpienia od umowy" -> załącznik nr 2 na #1. 8
indeksów przebudowanych (cache embeddingów: przeliczone tylko zmienione
fragmenty). Bateria `expand_query` 30/31 + 1 `known_issue` (art. 43e nadal
#1, `vat_86` nadal #8), testy **179/179** (+3).

## Krok 71 (2026-10-11): `cywilne` a konsument z wadliwym towarem -- ustawa o prawach konsumenta w indeksie `cywilne` + hint

**Pomiar (potwierdził problem z "Co dalej").** Pytanie "kupiłem pralkę jako
konsument, po 8 miesiącach przestała prać, czy mogę żądać wymiany" w
`cywilne` (baza bez adaptera): odpowiedź w całości na rękojmi z KC (art.
556², 561, 561⁵, 568) -- od 2023-01-01 dla konsumenta nieaktualnej (art.
43a ust. 1 ustawy o prawach konsumenta).

**Naprawa, część 1: ustawa w indeksie `cywilne`.** `domains` w
`acts_config.json`: `["konsumenckie", "cywilne"]` (ten sam wzorzec co PIT w
`kadry_i_place` + `podatkowe`). **Pułapka:** `build_rag_index.py` filtruje
po polu `domains` zapisanym w PRZETWORZONYCH plikach (`all_articles.json`
dla bieżących tekstów, `<akt>_history.json` dla historii), nie po configu --
sama zmiana configu nic nie daje bez ponownego `download_acts.py`. Tu
pole poprawione lokalnie w `all_articles.json`, `ustawa_o_prawach_konsumenta.json`
i `_history.json` (dokładnie to, co dałoby ponowne pobranie). Indeks
`cywilne`: 14751 -> 15454 fragmenty, embeddingi w całości z cache'u.

**Naprawa, część 2: hint `cywilne` dla `expand_query`.** Sam indeks nie
wystarczył: bez hintu rangi 3 / 30 / 36 (rękojmia z KC wygrywa
leksykalnie: "wada", "wymiana"). Pierwsza wersja hintu (z warunkiem "gdy
kupującym jest konsument...") naprawiła przypadki konsumenckie, ale
**zregresowała kanarka `kc_415_zalanie_mieszkania` #1 -> #16** --
rozszerzenie zmieniło styl na "Pytanie dotyczy..." i zgubiło "art. 415".
Eksperyment rozdzielił przyczyny: nowy indeks bez hintu -> art. 415 #1,
więc winny był wyłącznie hint. Wariant w czystej formie "zamiast X napisz
Y" (jak w innych dziedzinach) z frazą "ujawnionego w ciągu dwóch lat od
dostarczenia": kanarek #1, pralka (43d) #1, rower (43e) #1, "sprzedawca
twierdzi, że to moja wina" (43c) #1.

**Kontrola nadmiernego dopasowania (pytania spoza strojenia):** buty -- wymiana
zamiast naprawy (43d) #1; laptop -- sklep odsyła do producenta (43c) #1;
**kontrola odwrotna** -- spółka kupuje wadliwą maszynę: art. 560 KC #2 (hint
konsumencki nie wypycha rękojmi dla przedsiębiorców); telewizor naprawiany
dwa razy, BEZ słowa "konsument" -- 43e #22 (KC wygrywa; pytanie formalnie
niejednoznaczne, `known_issue`, celowo nie dostrajane).

**Bateria `expand_query`:** +7 przypadków (39: 37 OK + 2 `known_issue`), 0
nowych regresji. Testy **179/179**.

**End-to-end z produkcyjnym adapterem `cywilne`:** pralka -- poprawnie art.
43c (dwa lata + domniemanie) i 43d (naprawa/wymiana, odmowa tylko przy
niemożliwości/nadmiernych kosztach). **Resztka:** na końcu dokleja termin
z art. 568 § 2 KC "z ogólnej odpowiedzialności z tytułu rękojmi" -- przepis
jest w kontekście (KC też w indeksie), ale konsumenta nie dotyczy. Ta sama
klasa błędu co mylenie sąsiednich przepisów (Krok 66) -- poza zasięgiem
retrievalu; kolejny punkt danych dla destylacji.

## Krok 72 (2026-10-11 -- 2026-10-13): pilot destylacji Qwen -> Bielik na `konsumenckie`

**Cel.** Sprawdzić na jednej dziedzinie, czy adapter LoRA trenowany na
odpowiedziach Qwena (sekwencyjnie, z kontekstem RAG) poprawia Bielika tam,
gdzie stary adapter szkodził (Krok 68: "Nie" wbrew art. 43e w kontekście).

**Pipeline (`scripts/distill_qwen.py`, 4 wznawialne etapy, wyniki w
`data/distill/konsumenckie/`).**
1. `questions` -- Qwen generuje po 4 potoczne pytania do każdego z 67
   bieżących artykułów ustawy o prawach konsumenta -> 265 pytań.
2. `contexts` -- LOKALNIE Bielik bez adaptera: `expand_query` + RAG dokładnie
   jak w `chat.py` (ten sam prompt systemowy, `build_user_message`, top_k=8).
   24 pytania odrzucone jako bliskie pytaniom ewaluacyjnym (kosinus
   embeddingów >= 0.85) -> 240.
3. `answers` -- Qwen odpowiada na DOKŁADNIE te same wiadomości. Do zapytania
   do nauczyciela (nie do zapisanego przykładu) doklejana notka o formie
   (zwięźle, ~180 słów, bez nagłówków/tabel, "to zależy" wprost) -- bez niej
   Qwen pisał 1,5-3 tys. znaków, a `chat.py` generuje max 500 tokenów.
4. `dataset` -- filtr (kompletna odpowiedź, cytowane artykuły w kontekście,
   <= 480 tokenów Bielika, brak obcych alfabetów -- Qwen czasem przełącza
   język w środku słowa: "承诺owano", "nadmiernыми") -> **215 przykładów
   (193 train / 22 valid)** w `data/finetune/konsumenckie_distill/`.

**Kluczowa różnica względem starych danych fine-tune:** przykłady zawierają
prompt systemowy + kontekst RAG (`mask_prompt`, strata tylko na odpowiedzi).
Stare `data/finetune/<dziedzina>/` to samo pytanie -> odpowiedź, BEZ
kontekstu, choć adapter zawsze działa z kontekstem -- najpewniej dlatego
stary adapter odpowiada z pamięci i ignoruje przepis, który dostał.

**Nauczyciel:** Qwen/Qwen3.8-27B-FP8 na vLLM, RunPod A40; pod usunięty po
użyciu (~0,74 $). Zmiany w `chat_lmstudio.py` potrzebne do vLLM: opcjonalny
klucz API z `LLM_SERVER_API_KEY` (celowo nie `OPENAI_API_KEY`),
strumieniowanie SSE (proxy RunPoda/Cloudflare zrywa zapytanie błędem 524 po
~100 s ciszy -- tyle trwa samo rozumowanie Qwena), `reasoning_effort`
dodatkowo w `chat_template_kwargs` (vLLM czyta parametry szablonu tylko
stamtąd). Kontrolnie bateria Qwena na podzie: 28 przypadków, 24 OK, 3 znane
problemy, 1 nowa regresja (`kadry_urlop_wieloturowa_elliptyczna` -- brak
"26 dni"; jeden przebieg FP8/vLLM, nie badane dalej). Pierwszy pełny
przebieg przerwał się błędem 400 na `vat_86`; pojedynczo przeszedł OK.

**Trening (lokalnie, MLX).** LoRA rank 8, scale 20, 16 warstw, lr 1e-5,
batch 1, `max_seq_length` 11264 (przykłady 8-11 tys. tokenów), ~77 s/iter,
walidacja co 50. Val loss: 50: 0.460, 100: 0.443, 150: 0.436, **200: 0.427**,
250: 0.458, 300: 0.463 -> wczesne zatrzymanie po 300 (patience 2). Nocny
autopilot najpierw wybrał iter 100 -- `sort -g` w locale pl_PL ignorował
kropkę dziesiętną (używać `LC_ALL=C`); poprawka ręczna, oceniono oba.
Adaptery: `adapters/bielik11b-konsumenckie-distill-lora-iter{100,200}`.

**Ewaluacja (`scripts/eval_distill.py`, `data/distill/konsumenckie/eval.json`).**
16 pytań trzymanych z dala od treningu: 10 własnych + 4 przypadki
`konsumenckie` z baterii `expand_query` + 2 z baterii Qwena (tu jako pytania
end-to-end). Kontekst RAG liczony raz, identyczny dla 4 wariantów. Cytaty
spoza kontekstu: 0 we wszystkich wariantach (miara nic nie rozróżnia), więc
wszystkie 64 odpowiedzi przeczytane ręcznie:

| wariant | błędne wnioski | uwagi |
|---|---|---|
| baza | ~1-2 | 7/16 uciętych na 500 tokenach, formułki, zły artykuł przy zwrocie ceny (art. 32 zamiast 43e) |
| stary adapter iter25 | **5** | "Nie" zamiast "Tak" (pralka x2, zużycie telefonu, koszty odesłania), Allegro: myli termin odstąpienia z terminem odesłania |
| distill iter100 | 2 | Allegro: "od zawarcia, nie od dostawy"; brak pouczenia: "po pół roku prawo już wygasło" |
| **distill iter200** | ~1-2 słabe | jedyny poprawny w Allegro (14 dni od objęcia towaru w posiadanie); słabsze: koszty odesłania (mętne ujęcie), brak pouczenia (wywód z błędnym przykładem dat) |

Oba przypadki z baterii Qwena (`43e` pralka, `32` kurtka) na iter200
spełniają kryteria i mają poprawny werdykt. Destylowany adapter: odpowiedź
od razu ("Tak, możesz..."), potem podstawa prawna, bez formułek, rzadko
ucina.

**Wnioski.** (1) Stary adapter `konsumenckie` jest GORSZY od bazy --
potwierdzone systematycznie, nie tylko na jednym pytaniu z Kroku 68.
(2) Destylacja naprawiła to i daje poziom bazy lub trochę lepszy, z wyraźnie
lepszą formą -- ale nie jest skokiem jakościowym; n=16 nie rozróżni iter100
od iter200 z pewnością. (3) Mylenie sąsiednich przepisów NIE zniknęło:
iter200 dwa razy pisze "art. 43e ust. 5" zamiast "ust. 1 pkt 5", iter100
cytuje art. 32/34 (odstąpienie) w sprawach reklamacyjnych -- ten sam
mechanizm co Kroki 56/58/66; 215 przykładów to za mało, żeby go oduczyć.

**Baterie regresyjne a adapter -- uwaga metodyczna.** Żadna z dwóch baterii
nie mierzy adaptera Bielika: `check_expand_query_regressions.py` zawsze
działa na bazie bez LoRA (jak produkcja, Krok 21), a
`check_qwen_answer_regressions.py` odpytuje Qwena przez LM Studio/vLLM.
Adapter jest per dziedzina (`DEFAULT_ADAPTER_PATHS`), więc może wpłynąć
tylko na odpowiedzi w `konsumenckie` -- a przypadki `konsumenckie` z obu
baterii są już w powyższej ewaluacji.

**Wdrożenie (2026-10-13, za zgodą użytkownika).** `DEFAULT_ADAPTER_PATHS`
w `chat.py`: `konsumenckie` -> `bielik11b-konsumenckie-distill-lora-iter200`
(zamiast iter25). Smoke test przez `chat.py --prompt` na pytaniu spoza
treningu i ewaluacji (ekspres po 5 miesiącach, sklep odsyła do producenta):
poprawnie art. 43d ust. 1 / 43c ust. 1, odpowiedzialność sprzedawcy, nie
producenta. `chat_cuda.py` bez zmian (placeholdery ścieżek, brak adapterów
CUDA). Pozostałe 21 dziedzin nadal na starych adapterach iter25.

## Krok 73 (2026-10-13): stare adaptery iter25 vs baza w 17 dziedzinach + konwerter LoRA mlx_lm <-> PEFT

**Pytanie (z Kroku 72):** skoro stary adapter `konsumenckie` okazał się gorszy
od bazy, czy to samo dotyczy pozostałych dziedzin? Krok 61 sprawdził tylko 4
(`mieszkaniowe`, `energetyczne`, `wodne`, `lotnicze`).

**Metoda.** `scripts/compare_adapter_vs_base.py` (Krok 61) przerobiony: pytania
z `data/adapter_vs_base_questions.json` (z notatką `expect` do oceny, nie
trafia do modelu), odrzucanie pytań bliskich treningowym (kosinus >= 0.85 --
odrzucone 8, m.in. ciąża na okresie próbnym, VAT 150 tys., członek zarządu
art. 299), zapis pełnych etykiet kontekstu (`akt art. N`). 17 dziedzin, 89
pytań spoza zbiorów treningowych, ten sam kontekst RAG dla obu wariantów
(adapter włączany/wyłączany na tym samym modelu). Lokalnie ~1,5 min/pytanie,
~2 h. Wszystkie 178 odpowiedzi przeczytane ręcznie.

**Wynik (W = błędny wniosek, C = poważny błąd podstawy/cytatu przy dobrym wniosku):**

| dziedzina | n | adapter | baza |
|---|---|---|---|
| kadry_i_place | 5 | -- | C (art. 109 kary porządkowe zamiast 52 § 2) |
| podatkowe | 5 | C (5 lat przypisane art. 21) | W (VAT 7%/22% -- stare stawki z art. 146) |
| administracyjne | 6 | -- | -- |
| cywilne | 6 | 2W, 2C | 2W |
| rodzinne | 6 | C (art. 143 zamiast 144¹) | -- |
| karne | 6 | C (§ 2a źle), W-lekki ("Tak, kradzież" przy art. 289) | -- |
| gospodarcze | 5 | C | C |
| budowlane | 6 | **3W** | W |
| autorskie | 5 | W | W |
| wlasnosc_przemyslowa | 6 | **2W**, 2C | 3C |
| ochrona_danych | 3 | 2C | -- (2x unik "nie można jednoznacznie") |
| zamowienia_publiczne | 5 | -- | -- |
| ochrona_srodowiska | 5 | W | 2W |
| ruch_drogowy | 6 | W (fotelik "nie musi" przy 140 cm) | -- |
| upadlosciowe | 6 | -- | -- |
| geologia_i_gornictwo | 4 | -- | -- |
| telekomunikacyjne | 4 | C (zmyślona procedura art. 312) | -- |
| **razem** | **89** | **~10W + 11C** | **7W + 5C** |

4 błędy wspólne obu wariantom (przyczyna poza adapterem): przedawnienie
pożyczki -- art. 118 KC nie trafił do kontekstu, oba wzięły art. 722 (roszczenie
BIORĄCEGO o wydanie) i orzekły "przedawnione"; zaliczka potraktowana jak zadatek
(art. 394); sprzeciw od zgłoszenia -- art. 30 ust. 5 był w kontekście, oba
odpowiedziały terminem 14 dni z art. 54 (użytkowanie); historyczne skażenie
gruntu -- oba odwróciły wyjątek z art. 101h ust. 2. Wspólny jest też brak
art. 61 KSH (wystąpienie ze spółki jawnej) w kontekście.

**Wzorce adaptera** (te same co Kroki 61/68/72): pewne binarne otwarcie
sprzeczne z resztą odpowiedzi ("Nie, nie wymaga ani pozwolenia, ani zgłoszenia"
przy ogrodzeniu 2,5 m, potem bełkot; "Nie, nie jest wymagana zgoda" przy
coverach, potem odwrotnie); zmyślanie jednostek redakcyjnych i procedur
(niezarejestrowany wzór w PWP z "trzyletnią ochroną", art. 312 ust. 4-5 PKE);
błędne czytanie wyjątków (art. 25 ust. 5 PWP, art. 39 ust. 3b PoRD). Baza
częściej ucina (500 tokenów) i czasem unika odpowiedzi, ale rzadziej się myli.

**Wniosek.** Na 89 pytaniach baza jest lepsza od starych adapterów iter25
(błędne wnioski ~7 vs ~10-11, poważne błędy podstawy 5 vs 11) -- spójne z
Krokiem 61 (4 dziedziny) i 68/72 (`konsumenckie`). Per dziedzina n=3-6 to za
mało, żeby rozstrzygać pojedynczo; wyraźnie gorzej z adapterem w `budowlane`,
`wlasnosc_przemyslowa`, `ruch_drogowy`; lepiej z adapterem w `podatkowe` i
`kadry_i_place` (adapter `kadry` to inny przepis -- hedge-iter50). Produkcyjnie
(za zgodą użytkownika) `chat.py::DEFAULT_ADAPTER_PATHS` ograniczone do
`kadry_i_place` i `konsumenckie`; pozostałe 20 dziedzin na bazie (także
`podatkowe` -- jedna wygrana adaptera na 5 pytań to za mało), stare adaptery
w `LEGACY_ADAPTER_PATHS`.

**Konwerter LoRA mlx_lm <-> PEFT** (`scripts/convert_lora.py`, commit 7890a0e).
mlx: `y += scale·(x@a)@b`, a (in, r), b (r, out); PEFT: `y += alpha/r·x@Aᵀ@Bᵀ`
=> A = aᵀ, B = bᵀ, alpha = scale·r (rsLoRA: scale·√r). Klucze `model.layers.N...lora_a`
<-> `base_model.model.model.layers.N...lora_A.weight`. mlx_lm nakłada LoRA na
ostatnie `num_layers` warstw, więc przy PEFT z lukami/innym zakresem warstwy bez
wag dostają zerowe lora_a/lora_b (wkład 0, zamiast losowej inicjalizacji
lora_a). Odrzuca błędem DoRA, rank/alpha_pattern, bias, modules_to_save.
Walidacja `scripts/check_lora_converter.py` (wymaga `peft`, którego nie ma we
wspólnym venv -- pip --target + PYTHONPATH): mały losowy model Llama (GQA) w
transformers i mlx_lm z tymi samymi wagami -- logity PEFT vs mlx_lm po konwersji
zgodne do ~1e-6 w obu kierunkach, także rsLoRA i luka w warstwach; prawdziwy
adapter `konsumenckie-distill-iter200` MLX -> PEFT -> MLX bit w bit. Testy
jednostkowe `tests/test_convert_lora.py` (10). **`train_lora_cuda.py`
wyrównany z mlx_lm** (f3d5eee): 7 projekcji zamiast q/v, ostatnie 16 warstw
(`--num-layers`), alpha 160 (== scale 20 przy r=8), stały LR. Sprawdzone na
modelu z 50 warstwami: PEFT tworzy 112 modułów LoRA w warstwach 34-49, jak
mlx_lm. Na prawdziwym GPU jeszcze nie uruchamiane.

## Krok 74 (2026-10-13): dwudziesta trzecia dziedzina -- kredyt konsumencki (`kredyt_konsumencki`)

**Akt** (zweryfikowany przez ELI API): `ustawa_o_kredycie_konsumenckim` --
DU/2011/715, "Ustawa z dnia 12 maja 2011 r. o kredycie konsumenckim"
(tekst jednolity, IN_FORCE, 20 aktów zmieniających). Osobna dziedzina (jak
`konsumenckie` w Kroku 68). Ryzyko na przyszłość: dyrektywa (UE) 2023/2225
(CCD2) ma być stosowana od 20.11.2026 -- możliwa nowa ustawa zastępująca ten
akt (`check_acts_freshness.py`).

**Parser bez zmian:** 111 artykułów + 4 załączniki (115 fragmentów,
najdłuższy 7,9 tys. znaków -- załącznik nr 1), art. 60-65 brak (przepisy
zmieniające), zero śmieci `<>[]`/stopek. Historia: 7 okien (2014-2024), 640
artykułów, bez duplikatów i nietypowych etykiet (korekty indeksów górnych
przez lokalne dopasowanie -- istniejący mechanizm). Indeks 1358 fragmentów.
**Znalezisko (nienaprawione):** fałszywe spacje pypdf w polskich znakach
("zobowi ązany", "po średnik", "b ędących") -- 64 wystąpienia nieprawidłowych
początków słów (ą/ę/ń) w 21 z 115 fragmentów, najwyższy wskaźnik w korpusie
(0,55/1000 znaków; kodeks pracy 0,27, pozostałe akty łącznie 300). Ten sam
mechanizm co "Zał ącznik" z Kroku 69. Retrieval i odpowiedzi działają mimo
to; patrz "Co dalej".

**Retrieval (z `expand_query`, 12 pytań):** 10/12 w top-8; art. 45 (sankcja
kredytu darmowego, "nie podano RRSO") #23, art. 57 (kredyt wiązany) #12 --
przeformułowanie pomyliło odstąpienie od zakupu z odstąpieniem od kredytu.
**Hint `kredyt_konsumencki`** (wadliwa umowa -> zwrot bez odsetek i kosztów,
sankcja kredytu darmowego; zwrot rzeczy kupionej na raty -> skuteczność wobec
kredytu wiązanego): **12/12 w top-8** (45 #1, 57 #1). Kontrola 4 pytaniami
spoza strojenia: 57 #1, 45 #2, 50 #1, 13 #2. Efekt uboczny: przy pytaniu o
koszty odstąpienia przeformułowanie doklejało "sankcja kredytu darmowego"
(art. 54 i tak #1). Hint działa tylko w tej dziedzinie -- inne bez zmian.

**End-to-end (baza bez adaptera, zgodnie z Krokiem 73):** RRSO -- poprawnie
art. 45 przez art. 30 ust. 1 pkt 7; wcześniejsza spłata -- poprawnie art. 49
ust. 1; laptop na raty -- poprawnie art. 57, ale otwarcie "Tak" na pytanie
"czy muszę dalej spłacać" i art. 56 ust. 4 (odsetki przy odstąpieniu od
KREDYTU) przeniesiony na odstąpienie od zakupu.

**Baterie:** `expand_query` +4 (`kk_45`, `kk_57` -- pytanie kontrolne,
`kk_49` #3, `kk_44` #4), wszystkie OK; bateria Qwena +2 (`kk_45`, `kk_49`) --
**przebieg przerwany wymuszonym restartem komputera** (watchdog zabił
WindowServer: główny wątek >40 s w wywołaniu jądra; LM Studio z Qwenem 31,7 GB,
4 równoległe zapytania, + proces baterii 3,9 GB z embeddingami/rerankerem na
MPS; RAM ~36/64 GB, więc raczej rywalizacja o GPU niż brak pamięci -- pewności
brak). Zasada od teraz: żadnych dwóch zadań na GPU naraz, modele LM Studio
zwalniane po użyciu. **Drugie podejście** (Qwen załadowany z `--parallel 1
-c 32768 --ttl 600`, przypadki pojedynczo, pamięć logowana co 10 s): oba OK,
odpowiedzi wzorcowe (art. 30 ust. 1 pkt 7 + art. 45 ust. 1-5; art. 49 ust. 1
+ odróżnienie od prowizji z art. 50); minimum 45% wolnej pamięci, swap 0.

## Krok 75 (2026-10-13): adnotacje "... wejdzie w życie z dn. ..." usunięte z bieżących tekstów

**Problem (Krok 67, "Co dalej"):** notatki z marginesu tekstów ujednoliconych
Kancelarii Sejmu ("Nowe brzmienie § 2 w art. 18³a wejdzie w życie z dn.
5.11.2026 r. (Dz. U. z 2026 r. poz. 1046).") trafiały do treści artykułów --
często w ŚRODEK zdania ("…zaniżoną ocenę Nowe brzmienie art. 94³ wejdzie w
życie … przydatności zawodowej"). Zamiast szacowanych ~55 było już **126**
(Kancelaria dopisała nowe przy kolejnych aktualizacjach): KSH 22, Prawo
wodne 16, lotnicze 14, PoRD 11, PIT 10 itd.

**Naprawa:** `download_acts.strip_entry_into_force_annotations()` --
wołana na każdym artykule po pocięciu (tekst już jednoliniowy), więc działa
w bieżących tekstach i w historii. Dla każdego końca adnotacji ("wejdzie w
życie z dn. D r. (Dz. U. …)." -- także z datą rozbitą spacją przez pypdf:
"1.01.20 30 r.", oraz "z dniem określonym w komunikacie, o którym mowa w art.
N ustawy …") szuka najwcześniejszego słowa-klucza w oknie 200 znaków ("Nowe
brzmienie", "Dodany", "Przepis uchylający", "Zmiana", "Ust.", "Art.", "§"…),
od którego opis składa się wyłącznie z małych liter, numerów i skrótów
jednostek redakcyjnych i kończy się odwołaniem "art./ust. N". Bez takiego
startu nic nie jest wycinane (lepiej szum niż ucięty przepis). Wszystkie
126+5 wycięć przejrzane ręcznie -- zero treści przepisów. Zostają tylko
prawdziwe przepisy o wejściu w życie ("wejdzie w życie z dniem 1 stycznia
2027 r." w Prawie energetycznym) i stare przypisy w historii.

**Wdrożenie bez ponownego pobierania:** ta sama funkcja zastosowana
jednorazowo do `data/processed/*.json` (17 plików, 109 artykułów) -- ponowne
pobranie wciągnęłoby też nowe aktualizacje Kancelarii i pomieszało dwa
efekty. Indeksy 10 dziedzin przebudowane (cache embeddingów: przeliczono
0-229 fragmentów na dziedzinę; w `kadry_i_place` 229, bo wycięcie w długim
art. 21 ustawy o PIT przesuwa granice kolejnych fragmentów). Testy
jednostkowe +10 (`test_strip_entry_into_force_annotations*`).

**Weryfikacja:** pełna bateria `expand_query` (45 przypadków, ~9 min): 41 OK +
4 znane problemy (konsumenckie_28 #9, cywilne_tv #22, kc_118 #20, ksh_61 #13)
-- **zero nowych regresji**.

## Krok 76 (2026-10-13 -- 2026-10-17): destylacja `budowlane` + `cywilne` w całości na RunPodzie (CUDA-first)

**Pierwsza próba (13.10) przepadła.** Pod L40S (1,09 $/h) bez trwałego
dysku policzył pytania, konteksty i odpowiedzi `budowlane`; w trakcie
odpowiedzi `cywilne` zerwał się internet po stronie Maca. Pod stał dalej
(~5 h bezczynnie po skończeniu skryptu), saldo spadło do 0 $, RunPod
zatrzymał i -- bez network volume -- **usunął** poda razem z danymi (~9 $
łącznie z próbami podów). Wnioski wdrożone w drugiej próbie: wyniki na
trwałym `/workspace` (persistent mount), kopia na Maca co 10 min, strażnik
na podzie (`scripts/runpod/watchdog.sh`: własne `RUNPOD_API_KEY`/
`RUNPOD_POD_ID` poda -> `stop` 60 min po `ALL_DONE`/`FAILED` albo po 10 h).

**Druga próba (17.10): RTX PRO 6000 96 GB (Secure, 2,49 $/h), 09:41-11:27 UTC,
~1 h 46 min, ~4,4 $.** Wąskim gardłem na L40S były odpowiedzi Qwena (po
załadowaniu wag ~11 GB na KV, kilka promptów po 11 tys. tokenów naraz,
7-10 odp./min). Na 96 GB: faza 1 -- Qwen (0.58) i Bielik (0.27) naraz,
pytania `cywilne` równolegle z kontekstami `budowlane`, odpowiedzi
`budowlane` równolegle z kontekstami `cywilne`; faza 2 -- sam Qwen z 0.92
pamięci (**33-43 odp./min**, 48 zapytań naraz); faza 3 -- zbiory + trening
bf16. Ewaluacja `budowlane` (vLLM + `--enable-lora`) równolegle z treningiem
`cywilne`. Skrypty: `scripts/runpod/` (`pipeline.sh` -- faza 1, w trakcie
zastąpiony przez `pipeline2.sh` od fazy 2, patrz niżej; `eval.sh`).

| etap | `budowlane` | `cywilne` |
|---|---|---|
| pytania (Qwen) | 825 (4/artykuł), 70 s | 2173 (2/artykuł), ~8 min |
| konteksty (Bielik vLLM + RAG) | 807, ~5 min | 1400 (losowo), ~9 min |
| odpowiedzi (Qwen) | 600 (losowo) | 900 (losowo) |
| zbiór (train/valid) | 487/58 | 709/75 |
| odrzucone | za długa 20, cytat spoza kontekstu 17, za długi przykład 18 | cytat spoza kontekstu **94** (12%), za długa 22 |
| trening | 584 kroki (~1,2 epoki), ~2 s/krok | 850 kroków, ~2 s/krok |
| val loss min | 0,4171 @450 (0,4207 @584) | 0,4536 @650 |

Lokalnie (MLX) ten sam etap kontekstów zajmowałby ~1,5 h + ~3 h, odpowiedzi
Qwena w LM Studio > 3 doby, trening ~77 s/krok (~28 h).

**Dłuższe przykłady (zmiana w trakcie).** 16 384 tokenów kontekstu Qwena
odrzucało prompty z najdłuższym kontekstem (HTTPError: prompt + 4000 > limit),
a filtr `MAX_EXAMPLE_TOKENS` (11 264, limit MLX z Kroku 72) w pierwszej
próbie wyciął 101/482 przykładów `budowlane` (21%) -- adapter uczyłby się
tylko na krótszych kontekstach. Trening jest teraz na CUDA, więc: Qwen 24 576,
`distill_qwen.py dataset --max-example-tokens 16384`, `train_lora_cuda.py
--max-length 16384`. Odpadło 18 przykładów zamiast ~20%. Przełączenie w
locie: `pipeline2.sh` przejął pracę po fazie 1 (zabicie starego skryptu
przez `pkill -f "bash pipeline.sh"` dopasowało też własną sesję SSH --
na przyszłość pgrep z `[b]ash`).

**Zmiany w kodzie:** `distill_qwen.py` -- `contexts --expand-url`
(ekspansja Bielikiem przez vLLM), `--sample` w `contexts`/`answers`
(losowy podzbiór, stałe ziarno), `dataset --tokenizer/--max-example-tokens`,
filtr "za długi przykład", `eval_questions` + pytania z
`adapter_vs_base_questions.json` (trzymane z dala od treningu),
`read_jsonl` toleruje uciętą ostatnią linię; `train_lora_cuda.py` --
`--max-length` (SFTConfig domyślnie 1024 i obcina od końca = bez odpowiedzi),
`--quantization none` (bf16), `--max-eval-samples`, gradient checkpointing;
nowy `scripts/eval_distill_vllm.py` (odpowiednik `eval_distill.py` na vLLM).

**Konwerter potwierdzony na prawdziwym przebiegu:** `convert_lora.py
peft-to-mlx` na podzie, adapter iter584 w MLX na Macu odpowiada na 2
pytaniach (ten sam kontekst) praktycznie tak samo jak w vLLM -- te same
artykuły i wnioski, różnice w pojedynczych sformułowaniach (bf16 vs MLX).

**Ewaluacja (`data/distill/<d>/eval_vllm.json`, vLLM, zachłannie, 500
tokenów jak `chat.py`):** baza + 3 checkpointy (najlepszy val loss, ostatni,
~60% najlepszego); 8 pytań `budowlane`, 17 `cywilne`; wszystkie odpowiedzi
przeczytane ręcznie (z notatkami `expect`).

- `budowlane`: forma adapterów wyraźnie lepsza (konkretna odpowiedź na
  początku, bez markdownu; ucięte 1/8 vs baza 3/8). Treść: iter584
  najlepszy (legalizacja wiaty -- art. 49f tylko po 20 latach, poza tym art.
  48/48b; okna -- bez zmyślonej podstawy), iter250/450 powtarzają błąd z art.
  49f i zmyśloną podstawę przy oknach. Garaż (21 dni z art. 30 ust. 5): błąd
  wszystkich wariantów -- znane ucinanie art. 30 > 6000 znaków (Krok 73).
  **Domyślnie: iter584.**
- `cywilne`: remis z bazą. iter650: dobrze zalanie, rower, ciężar dowodu,
  spadek, szkoda na parkingu, maszyna; słabiej niż baza laptop "sklep
  odsyła do producenta" (asekuracja zamiast "sprzedawca odpowiada 2 lata");
  pożyczka -- dobry wniosek ("nie przedawniło się"), ale z art. 722
  (art. 118 poza kontekstem, znana luka); zaliczka -- wszystkie warianty
  mylą z zadatkiem (znane z Kroku 73); VAT -- "mogę poszukać w innych
  źródłach" (obietnica bez pokrycia). iter850 częściej odmawia mimo
  wystarczającego kontekstu (używany samochód od osoby prywatnej: "skonsultuj
  się z prawnikiem" zamiast rękojmi z art. 559/564/566 -- baza odpowiada
  dobrze). Ucięte: baza 7/17, iter650 6/17, iter850 2/17. **Domyślnie:
  iter650** (reguła z Kroku 73: nie gorszy od bazy; przewaga tylko w formie).
- Sanity `chat.py --domain cywilne` (spadek po ojcu, pytanie spoza
  treningu): poprawnie, zwięźle.

Adaptery: `adapters/bielik11b-{budowlane,cywilne}-distill-lora-iter*` (MLX,
symlinki do wspólnego katalogu), PEFT w `bielik-lora-adapters/cuda/` (bez
optimizer.pt). Dane: `data/distill/{budowlane,cywilne}/` (+ `pod_logs/`),
`data/finetune/{budowlane,cywilne}_distill/`.

## Krok 77 (2026-10-17): długie artykuły -- kontekst z ustępów trafionych przez wyszukiwanie

**Problem (Krok 73, "wysoki priorytet").** 261 z ~12,4 tys. bieżących
artykułów ma > 6000 znaków (`MAX_ARTICLE_CHARS`), a `build_context` dawał
modelowi ich pierwsze 6000 znaków. Indeks RAG od dawna tnie długie artykuły
na okna po ~900 znaków, więc wyszukiwanie TRAFIA we właściwy ustęp -- ale
ten ustęp nie docierał do modelu. Przykład: art. 30 Pr. bud. (12,4 tys.),
ust. 5 z "21 dni" od znaku ~7600 -> Bielik zgadywał 14 dni z art. 54, Qwen
uczciwie odmawiał. Destylacja uczyła ucznia na tych samych uciętych
kontekstach.

**Naprawa (`scripts/article_excerpt.py`, `prompt.build_context`).** Dla
artykułu > 6000 znaków: początek (600 znaków, `ARTICLE_HEAD_CHARS`) + " [...] "
+ ustępy/paragrafy wokół fragmentu trafionego na etapie kosinusowym
(`chunk_text` z `rank_chunks`), poszerzane o całe sąsiednie jednostki, póki
mieszczą się w 6000 znaków. Granica jednostki: "2a. Do" / "§ 3. Jeżeli" po
końcu zdania/punktu (nie "pkt 2. W"). Jeden ustęp dłuższy niż budżet (art. 21
ust. 1 PIT, ~150 punktów) -- okno znakowe wyśrodkowane na fragmencie.
Sprawdzone na wszystkich fragmentach wszystkich 261 długich artykułów:
trafiony fragment zawsze w wycinku, nigdy > 6000 znaków. Indeksy bez zmian
(fragmenty do embeddingów te same), testy +7 (`tests/test_article_excerpt.py`).

**Odrzucone: to samo w rerankerze.** Reranker ocenia pierwsze 1500 znaków
artykułu (`RERANK_TEXT_CHARS`); próba (a) oceniania okna wokół trafionego
fragmentu i (b) maksimum z początku i okna -- obie zepsuły
`vat_86_zakupy_prywatne` w baterii `expand_query` (ranga 8 -> 12 / 11, próg
top-8): ogólna zasada odliczenia stoi w ust. 1, a inne długie artykuły VAT
zyskały na swoich oknach. Cofnięte -- ranking wyszukiwania jest bitowo taki
jak przed Krokiem 77 (bateria `expand_query`: 41 OK + 4 znane, art. 86
ranga 8). Wpływ zmiany kontekstu na odpowiedzi sprawdza bateria Qwena
(przebieg na podzie, Krok 78) -- `budowlane_30_sprzeciw_od_zgloszenia`
powinien przestać być znanym problemem.

**Przy okazji (pod destylację, Krok 78):** `distill_qwen.py holdout`
(odłożone artykuły do ewaluacji, ich pytania wypadają z treningu), w
instrukcji dla nauczyciela "cytuj wyłącznie przepisy z fragmentów",
`scripts/judge_distill.py` (Qwen jako sędzia par baza vs adapter),
`scripts/runpod/distill_multi.sh` (8 dziedzin w jednym przebiegu).

## Krok 78 (2026-10-17): destylacja 8 dziedzin naraz + sędzia Qwen -- adaptery w większości NIE lepsze od bazy

**Przebieg.** RTX PRO 6000 Secure (2,49 $/h), 12:14-17:51 UTC, ~14 $
`budowlane` i `cywilne` od nowa (te same pytania co
w Kroku 76, nowe konteksty z wycinkami Kroku 77), nowe: `kadry_i_place`,
`podatkowe`, `rodzinne`, `mieszkaniowe`, `ruch_drogowy`,
`wlasnosc_przemyslowa`. Na dziedzinę 12 odłożonych artykułów (`holdout`) ->
20-30 pytań ewaluacyjnych (baterie + porównanie z Kroku 73 + odłożone).

| dziedzina | odpowiedzi | train/valid | cytat spoza kontekstu | val loss koniec |
|---|---|---|---|---|
| budowlane | 500 | 407/50 | 8 (1,6%; Krok 76: 2,8%) | 0,435 |
| cywilne | 700 | 566/62 | 54 (7,7%; Krok 76: 10,4%) | 0,447 (min 0,4474 @550) |
| kadry_i_place | 700 | 580/64 | 8 | 0,396 |
| podatkowe | 700 | 543/62 | 23 | 0,444 |
| rodzinne | 500 | 438/45 | 9 | 0,397 |
| mieszkaniowe | 497 | 413/52 | 3 | 0,404 |
| ruch_drogowy | 500 | 405/48 | 13 | 0,448 (min 0,4484 @400) |
| wlasnosc_przemyslowa | 500 | 426/49 | 5 | 0,402 |

Instrukcja "cytuj wyłącznie z fragmentów" + wycinki: cytaty spoza kontekstu
w dół o ~1/4-1/2. W 6/8 dziedzin val loss spada do ostatniego kroku (1 epoka)
-- niedotrenowane; w Kroku 76 minimum wypadało po 0,8-0,9 epoki.

**Sędzia Qwen (`judge.json`, pary baza vs adapter, kolejność losowa, z
kontekstem i notatką `expect`).** Werdykty dla `budowlane` przeczytane
ręcznie -- spójne z uzasadnieniami, wskazane błędy prawdziwe.

| dziedzina | wariant | adapter lepszy / remis / baza lepsza | poważne błędy adapter / baza |
|---|---|---|---|
| budowlane | **Krok 76 iter584 (produkcja)** | **9 / 4 / 6** | 8 / 10 |
| budowlane | iter407 | 5 / 4 / 11 | 9 / 6 |
| cywilne | **iter550** | **12 / 4 / 11** | 10 / 11 |
| cywilne | iter566 | 11 / 7 / 9 | 11 / 10 |
| cywilne | Krok 76 iter650 (produkcja) | 9 / 8 / 10 | 10 / 7 |
| kadry_i_place | iter580 | 8 / 6 / 6 | 7 / 6 |
| podatkowe | iter543 | 8 / 3 / 10 | 7 / 7 |
| rodzinne | iter438 | 6 / 6 / 7 | 7 / 6 |
| mieszkaniowe | **iter413** | **8 / 7 / 4** | 5 / 7 |
| ruch_drogowy | iter400 | 6 / 5 / 8 | 5 / 6 |
| wlasnosc_przemyslowa | iter426 | 4 / 1 / 11 | 11 / 8 |

(~20 pytań na dziedzinę -- różnica 2-3 werdyktów to szum.) **Wniosek:** po
Kroku 77 sam Bielik z pełnymi ustępami jest mocniejszy (np. garaż -- 21 dni z
art. 30 ust. 5 -- teraz dobrze i w bazie, i w adapterach) i destylacja przy 1
epoce, lr 1e-5, rank 8 nie daje stałej przewagi -- poprawia formę, nie
merytorykę. **Produkcja (`DEFAULT_ADAPTER_PATHS`):** `budowlane` zostaje
Krok 76 iter584, `cywilne` -> k78 iter550, `mieszkaniowe` -> k78 iter413
(nowe), `kadry_i_place` zostaje hedge-iter50 (nie był w tym porównaniu),
reszta na bazie. Pozostałe adaptery k78 zostają w `adapters/` (MLX) i
`bielik-lora-adapters/cuda/krok78/` (PEFT) do dalszych porównań.

**Bateria Qwena (na podzie, vLLM FP8):** 29 OK, 1 znany (`ochrona_danych_101`),
1 "nowy": `kk_45_brak_rrso_sankcja_odpowiedz` -- art. 45 poza kontekstem,
art. 45 ma 960 znaków (Krok 77 go nie dotyczy, wybór artykułów bitowo bez
zmian), przypadek niestabilny już w Kroku 74 -> do ponownego sprawdzenia
lokalnie (LM Studio). `budowlane_30_sprzeciw_od_zgloszenia` -- **OK**, zdjęty
`known_issue` (efekt Kroku 77).

**Efektywność (lekcje na kolejne przebiegi, wdrożone w `scripts/runpod/`):**
- trening bez gradient checkpointingu (`train_lora_cuda.py
  --gradient-checkpointing off`): test A/B 10 kroków po ~8 tys. tokenów:
  58,4 s / 27 GiB -> **29,0 s / 48,7 GiB**; w przebiegu krok 2,4-3,0 s -> 0,45-1,8 s
  (bez checkpointingu wejścia nie wymagają gradientu, więc wstecz tylko przez
  16 warstw z LoRA); szczyt ~77 GiB przy przykładach do 16 tys. tokenów.
  Gradienty adaptera matematycznie te same.
- vLLM startujący RÓWNOLEGLE z treningiem nie dostaje pamięci na KV
  (profilowanie liczy mu przyrost pamięci treningu) -- ewaluacja po treningu
  (`distill_eval.sh`); dwa statyczne serwery naraz działają.
- odpowiedzi Qwena: 96 sekwencji naraz -> KV 78%, generowanie 612 -> 910
  tok/s, ale odpowiedzi 40 -> 43/min (limit: prefill ~10 tys. tokenów na
  pytanie); stałe tory dziedzin zostawiały kartę w połowie bezczynną pod koniec
  -> wspólna kolejka (`claim`).
- wyszukiwanie w 14 procesach na CPU naraz = przeciążenie (każdy torch na
  wszystkich rdzeniach, load 50-67/32) -- ewaluacja z wyszukiwaniem na GPU po
  kolei; bateria równolegle (8 procesów) przeszła 27 przypadków w ~60 min.
- błędy po drodze: `holdout` wybierał artykuły bez pytań (None), `pkill -f`
  z nazwą skryptu dopasował własną sesję SSH (dwa razy), monitor brał kod
  wyjścia `ls FAILED` za zerwane SSH, `wait` bez argumentów czekałby na
  serwery vLLM. Wszystkie poprawione.

## Krok 79 (2026-10-18): dłuższy trening (2-3 epoki) na cywilne + mieszkaniowe -- bez stałej przewagi

**Pytanie (z Kroku 78):** val loss spadał do końca 1 epoki -- czy 2-3 epoki poprawiają adaptery?
**Przebieg.** RTX PRO 6000 Secure (2,49 $/h; Community bez wolnych maszyn, 3 próby),
07:54-ok. 09:15 UTC, ~3,4 $. Dane destylacji z Kroku 78 użyte ponownie (bez
generowania Qwenem): `scripts/runpod/longtrain.sh` = trening 3 epok (`cywilne`
566 przykładów / 1698 kroków, `mieszkaniowe` 413 / 1239; bez gradient
checkpointingu, ~1,65 kroku/s, GPU 85-100%, 59 GiB), checkpoint co pół epoki,
potem Qwen (sędzia) + Bielik z adapterami: baza vs k78 (1 epoka) vs ~2 epoki vs
3 epoki.

| dziedzina | wariant | adapter lepszy / remis / baza lepsza | poważne błędy adapter / baza | cytat spoza kontekstu |
|---|---|---|---|---|
| cywilne | k78 iter550 (1 ep., produkcja) | 15 / 6 / 8 | 11 / 10 | 0 |
| cywilne | iter1132 (2 ep.) | 13 / 6 / 10 | 11 / 11 | 0 |
| cywilne | iter1698 (3 ep.) | 11 / 5 / 8 | 8 / 10 | 2 |
| mieszkaniowe | k78 iter413 (1 ep., produkcja) | 9 / 5 / 6 | 6 / 7 | 0 |
| mieszkaniowe | iter824 (2 ep.) | 11 / 5 / 4 | 5 / 7 | 0 |
| mieszkaniowe | iter1239 (3 ep.) | 7 / 5 / 7 | 7 / 8 | 1 |

Val loss: minimum po ~2 epokach (cywilne 0,4479 @849-1132 -- praktycznie
tyle samo co 0,4474 z k78 po 1 epoce; mieszkaniowe 0,3973 @824 vs 0,404 z k78),
od 2,5 epoki skok w górę (przeuczenie: 0,49 / 0,46).

**Werdykty `mieszkaniowe` iter824 przeczytane wszystkie 20.** 11/5/4 wygląda
lepiej niż jest: z 11 wygranych adaptera tylko 3 mają poważny błąd bazy
(odszkodowanie z art. 18 ust. 2, art. 13 ust. 2 do sąsiada, zakładany najem
socjalny), reszta to "precyzyjniejsza/pełniejsza". 3 z 4 wygranych bazy to
poważne błędy adaptera w sprawach eksmisyjnych (pokoik u znajomych: odwrócony
wniosek "nie musisz mieć wyroku"; komornik/pomieszczenie tymczasowe: kategoryczne
"Nie" i samozaprzeczenie; wymiana drzwi jako "zmiana przeznaczenia" art. 22
ust. 3 pkt 4). Jedna pomyłka wspólna (art. 35 przejściowy).

**Wniosek.** 3 epoki szkodzą, 2 epoki = maksimum, a zysk w granicach szumu
(~20 pytań). **Produkcja bez zmian** (`cywilne` iter550, `mieszkaniowe` iter413).
Destylacji na kolejne dziedziny na tej recepturze (rank 8, lr 1e-5) NIE skalować;
sensowne byłoby co najwyżej zmienić recepturę (rank/lr/dane), nie liczbę epok.
Adaptery PEFT: `../bielik-lora-adapters/cuda/krok79/`.

**Narzędzia/lekcje.**
- `scripts/runpod/longtrain.sh`, `make_bundle.sh` (paczka tylko dla wybranych
  dziedzin, ~135 MB), skrypt wysyłający token HF przez stdin SSH (usunięty w Kroku 125),
  `watchdog.sh` teraz w minutach (`MAX_MIN`; bash nie liczy `2.3`).
- Klucz `~/.ssh/id_ed25519` ma hasło, agent pusty: tymczasowy klucz bez hasła +
  `update-ssh-keys`; dla działającego poda `update-pod` (env `PUBLIC_KEY`) +
  restart -- **po restarcie zmienia się publiczny port SSH**. Przepis w pamięci
  (`reference_runpod_ssh_temp_key`). Klucz tymczasowy zdjęty z konta po przebiegu.
- `pkill -f` z wzorcem w tej samej linii SSH znów zabił własną sesję (jak w
  Kroku 78) -- logikę zabijania trzymać w pliku skryptu.
- `eval_vllm.json`/`judge.json` nadpisane wynikami Kroku 79 (poprzednie w git).

## Krok 80 (2026-10-18): dwudziesta czwarta dziedzina -- wykroczenia (Kodeks wykroczeń)

**Po co.** Krok 43: pytanie o kradzież roweru za 300 zł dostawało odpowiedź
z Kodeksu karnego (art. 278), bo wykroczenie (KW art. 119, do 800 zł) nie było
w żadnym indeksie. Pilot autonomiczny (zgoda z Kroku 45), bez GPU w chmurze.

**Akt (ELI API zweryfikowany przed pobraniem).** Dz.U. 1971 poz. 114, "Ustawa z
dnia 20 maja 1971 r. Kodeks wykroczeń." (tekst jednolity: Dz.U. 2025 poz. 734,
IN_FORCE; nowelizacje po nim m.in. 2025 poz. 1676, 1814, 1818, 1872). Pobranie:
207 artykułów, korekta indeksów górnych w 3, **0 śladów nawiasów `<>[]`**
(bieżący tekst i historia). Historia: 1731 artykułów z 8 wersji. Indeks:
2204 fragmenty. Artykuły sprawdzone po TREŚCI przed napisaniem przypadków
(lekcja Kroków 28/40): 119, 51, 92a, 87, 77, 124, 66.

**Retrieval (bez hintu `expand_query`).** Surowe pytania: 12/12 w top-5, 8
na pozycji 1; najsłabsze: art. 92a (prędkość) ranga 4 -- sąsiednie przepisy
drogowe 97/94/98 -- i art. 77 (pies) ranga 5. Z prawdziwym `expand_query`:
**8/8 w top-8, 7 na pozycji 1** (92a: 4; art. 77 poprawił się z 5 do 1).
Baterię dodano do `data/expand_query_reference_cases.json` (+8 przypadków);
bez zmian w `QUERY_EXPANSION_SYSTEM_PROMPT` i bez hintu domenowego, więc inne
dziedziny nie mogły się zmienić. `check_expand_query_regressions.py` dostał
`--domain`.

**Test końcowy (`chat.py --domain wykroczenia`, baza).** Pytanie z Kroku 43:
odpowiedź wskazuje KW art. 119 § 1 (wykroczenie, próg 800 zł), § 3 (osoba
najbliższa -- na żądanie) i § 4 (zapłata równowartości); sprawdzone z tekstem
artykułu -- zgodne. Luka z Kroku 43 zamknięta dla pytań o drobne czyny.

**Nie zrobiono / otwarte.** Bateria Qwena (LM Studio) bez nowych przypadków;
dziedzina bez adaptera (zgodnie z Krokiem 73/78: baza wystarcza). Drugi akt
(procedura) dodany w Kroku 81.

## Krok 81 (2026-10-18): wykroczenia -- drugi akt, Kodeks postępowania w sprawach o wykroczenia

**Akt (ELI zweryfikowany; uwaga na sąsiedni numer).** Dz.U. 2001 poz. 1148,
"Ustawa z dnia 24 sierpnia 2001 r - Kodeks postępowania w sprawach o
wykroczenia." (IN_FORCE, z tekstem jednolitym). **Dz.U. 2001 poz. 1149 to INNY
akt ("Przepisy wprowadzające")** -- nie pobrany, ostrzeżenie w `_comment`
konfiguracji. 133 artykuły, 0 śladów nawiasów (bieżący tekst; historia 1306
wpisów z 4 wersji, też 0). Indeks `wykroczenia` przebudowany: 4460 fragmentów.
Zauważone, nienaprawiane: znacznik przypisu "1)2)" w art. 45 § 1 (klasa
przypisów z Kroku 49, kat. 3, kosmetyczne).

**Retrieval.** Numery artykułów kolidują między aktami (oba mają np. art. 51,
97, 99), więc test sprawdza parę (akt, artykuł). Po dodaniu drugiego aktu
wszystkie 12 trafień Kodeksu wykroczeń bez zmian (surowe pytania; jedna
sąsiednia pozycja KPSW wskoczyła na 5. miejsce przy pytaniu o alkohol, art. 87
nadal #1). Pytania proceduralne (surowe): 4/5 na pozycji 1; apelacja (art. 105)
ranga 6 -- 110/61/83 przed nim. Z prawdziwym `expand_query`: **12/12 w top-8, 10
na pozycji 1** (art. 92a: 5, art. 105: 4); bateria +4 (`kpsw_*`), razem 20
przypadków dziedziny w baterii (57 łącznie), bez hintu i bez zmian w prompcie.

**Test końcowy** (`chat.py --domain wykroczenia`, baza): odmowa przyjęcia
mandatu -> art. 99 KPSW, wniosek o ukaranie do sądu z adnotacją o odmowie i
przyczynach -- zgodne z tekstem art. 99.


## Krok 82 (2026-10-18): `kk_45` zweryfikowany lokalnie + przypadek Qwena dla wykroczeń

**`kk_45_brak_rrso_sankcja_odpowiedz`** (niestabilny od Kroku 74, "nowa
regresja" na podzie w Kroku 78): w LM Studio (Qwen3.8-27B 8-bit, kontekst
24576, `reasoning_effort=low`) **przechodzi**, a odpowiedź przeczytana i
sprawdzona z tekstem art. 45 -- ust. 1 (pisemne oświadczenie, zwrot kredytu
bez odsetek i kosztów), ust. 2 (równe raty miesięczne), ust. 4 (koszty
zabezpieczenia), ust. 5 (wygasa po roku) -- wszystko zgodne. Porażka z Kroku 78
to artefakt vLLM FP8 (inne rozszerzenie zapytania), nie błąd danych ani
retrievalu. Notatka w przypadku zaktualizowana; ostrzeżenie zdjęte.
LM Studio domyślnie ładuje model z kontekstem 8192 -- do tej baterii trzeba
`lms load qwen/qwen3.8-27b --context-length 24576`.

**Nowy przypadek:** `wykroczenia_119_kradziez_300zl_odpowiedz` (pytanie z
Kroku 43) -- przechodzi; odpowiedź Qwena: wykroczenie, art. 119 § 1, areszt /
ograniczenie wolności / grzywna do 5000 zł (art. 1 § 1 KW), § 2-4 poprawnie.

## Krok 83 (2026-10-18): `kadry_i_place` -- hedge-iter50 (produkcja) vs k78 iter580

Pytanie z Kroku 78: czy adapter z destylacji (k78 iter580) bije produkcyjny
`hedge-iter50`? Odpowiedzi trzech wariantów (baza, hedge-iter50, k78 iter580)
wygenerowane lokalnie na MLX (`eval_distill.py`, 20 pytań, ten sam kontekst
RAG; ~20 min na wariant), sędzia Qwen3.8-27B 8-bit w LM Studio (kontekst
24576, współbieżność 4; ~2,6 h -- thinking do limitu, sędzia nie zapisuje
wyników cząstkowych) porównał każdy wariant z bazą. `judge_distill.py` dostał
`--eval-file` / `--out-file`; wyniki w `eval.json` / `judge_local.json`
(`eval_vllm.json` / `judge.json` z Kroku 78 bez zmian).

| wariant | adapter lepszy / remis / baza lepsza | poważne błędy adapter / baza | werdyktów |
|---|---|---|---|
| hedge-iter50 (produkcja) | 8 / 0 / 8 | 5 / 4 | 16 z 20 |
| k78 iter580 | 4 / 0 / 9 | 7 / 4 | 13 z 20 |

11 z 40 par bez werdyktu (odpowiedź sędziego nie zmieściła się w limicie
`max_tokens=6000`), więc wynik jest orientacyjny. Przeczytane wszystkie
uzasadnienia: k78 iter580 zmyśla wymagania wniosków (spółdzielnia socjalna),
myli warunki ustawowe (okres próbny a ciąża, korekta zeznania po zwrocie ZUS),
w pytaniu o urlop na żądanie twierdzi, że kontekst nie ma limitu dni, choć
art. 167² go zawiera. hedge-iter50 błędy ma o podobnym charakterze (art. 151⁷
do pory nocnej, zmyślony art. 21 ust. 1 pkt 63a PIT), ale w parze z bazą daje
remis.

**Wniosek.** Produkcja bez zmian (`hedge-iter50`): k78 iter580 jest gorszy od
bazy i nie ma powodu go wdrażać. Zarazem hedge-iter50 nie wykazuje przewagi
nad bazą (8/0/8) -- otwarte pytanie, czy `kadry_i_place` też nie powinno
działać na samej bazie (jak pozostałe dziedziny po Kroku 73); wymaga decyzji
użytkownika i większej próby (20 pytań to za mało na rozstrzygnięcie).

## Krok 84 (2026-10-18): dwudziesta piąta dziedzina -- postępowanie cywilne (KPC) + poprawka spłaszczonych etykiet

**Akt (ELI zweryfikowany; sąsiedni numer to inny akt).** Dz.U. 1964 poz. 296,
"Ustawa z dnia 17 listopada 1964 r. - Kodeks postępowania cywilnego." (IN_FORCE,
z tekstem jednolitym); **poz. 297 = "Przepisy wprowadzające" -- nie pobrany.**
2026 artykułów, historia 16 672 wpisy z 9 wersji (0 śladów nawiasów, 0
duplikatów etykiet), indeks 22 436 fragmentów -- największy w projekcie.

**Problem znaleziony po drodze i naprawiony ogólnie (kod wspólny, za zgodą
użytkownika): spłaszczone etykiety artykułów.** W bloku 477^14a-16 i 479^28-88
`pdfplumber` w ogóle nie wykrył nagłówków (1950 vs 2026), więc
`align_superscript_corrections` nie miało czego dopasować -- etykiety zostawały
"47928" zamiast "479²⁸" (74 artykuły; model cytowałby nieistniejący art. 47928).
Ta sama klasa co Kroki 18/48. Nowa funkcja `repair_flattened_superscript_labels`
(`download_acts.py`, wołana po dopasowaniu nagłówków): etykieta z samych cyfr,
której początek to numer ostatniego zaufanego artykułu, a PÓŹNIEJ numeracja cofa
się poniżej niej, dostaje indeks górny (obie przesłanki naraz -- bez nich
numeracja monotoniczna nie jest ruszana). 2 testy jednostkowe.
**Kontrola przed wdrożeniem na wszystkich zapisanych danych (bieżące + historia,
25 dziedzin):** 86 trafień, wszystkie PRAWDZIWE błędy lub w oknie śmieciowym:
KPC 74; **Prawo własności przemysłowej art. 2531 -> 253¹** (bieżący, dotąd
niezauważony); Kodeks pracy historia przed 2014: 8 (25¹, 63², 68², 77³, 92¹,
129⁵, 129⁶, 298¹); KPA historia sprzed 2000: 3 w oknie ze skanem OCR o treści
podatkowej (osobny, wcześniejszy problem danych, nie naprawiany). Zero fałszywych
trafień w zdrowych danych. Zastosowane: KPC, PWP (przebudowany indeks `wlasnosc_przemyslowa`; różnice w
196 artykułach to głównie szum odstępów z PDF-a w obie strony + 2 ulepszone cytaty,
netto neutralne/lepsze) i Kodeks pracy (historia, 8 etykiet; przebudowany indeks
`kadry_i_place` 31 001 fragmentów, bateria bez regresji).

**Znane ograniczenia KPC (nienaprawiane).** (1) Art. 477¹⁴: dwa fragmenty
przyszłego prawa (`<§ 2¹` i drugi `<§ 4`) bez zamykającego `>` w PDF-ie -- zostają
w tekście, § 4 jest zdublowany (znany kompromis z Kroku 65: niesparowany `<` na
poziomie paragrafu zostawia tekst, "lepiej za dużo niż wyciąć obowiązujący").
(2) Spłaszczone odwołania do ustępów i cytaty w prozie ("§ 11" zamiast "§ 1¹",
"art. 20512", "art. 4801") -- klasa z Kroków 48/49. (3) Znacznik przypisu "3)" na
początku art. 1.

**Retrieval (bez hintu `expand_query`).** Surowe pytania: 9/9 w top-3, 6 na
pozycji 1 (najsłabsze: art. 139 awizo -- ranga 3; art. 505 i 730 -- ranga 2).
Z prawdziwym `expand_query`: **8/8 w top-8, 5 na pozycji 1** (art. 139: 4).
Bateria +8 (`kpc_*`), razem 65 przypadków (7 dla pozostałych: PWP bez zmian).

**Test końcowy** (`chat.py --domain postepowanie_cywilne`, baza): wyrok zaoczny
-> art. 344 § 1-3 (2 tygodnie, treść sprzeciwu, odrzucenie spóźnionego) i art.
343 § 1 (doręczenie z pouczeniem) -- sprawdzone z tekstem artykułów, zgodne.

## Krok 85 (2026-10-18): dwudziesta szósta dziedzina -- postępowanie karne (KPK)

**Akt (ELI zweryfikowany; sąsiedni numer to inny akt).** Dz.U. 1997 poz. 555,
"Ustawa z dnia 6 czerwca 1997 r. - Kodeks postępowania karnego." (IN_FORCE, z
tekstem jednolitym); **poz. 556 = "Przepisy wprowadzające" -- nie pobrany.**
958 artykułów, historia 7417 wpisów z 8 wersji (0 śladów nawiasów, 0 spłaszczonych
etykiet po nowej kontroli z Kroku 84, 0 duplikatów). Indeks `postepowanie_karne`.
Ostrzeżenie o niesparowanych nawiasach: jeden zbłąkany `>` w środku zdania (art.
o doręczaniu wyroku), nic nie wchłonął -- bezpieczne (klasa Kroku 45).

**Retrieval (bez hintu `expand_query`).** Surowe pytania: 9/9 w top-3, 6 na pozycji
1 (najsłabsze: art. 182 ranga 3 -- 384/185 przed nim; art. 258 i 445 ranga 2).
Z prawdziwym `expand_query`: **8/8 w top-8, 7 na pozycji 1** (art. 445: 2). Bateria
+8 (`kpk_*`), razem 73 przypadki, bez zmian w prompcie i bez hintu.

**Test końcowy** (`chat.py --domain postepowanie_karne`, baza): żona oskarżonego
-> art. 182 § 1 (osoba najbliższa może odmówić zeznań) -- zgodne z tekstem.
Dwie uwagi (nie błąd danych, klasa Reguły 6): model dopowiedział od siebie, że
odmowę trzeba zgłosić przed przesłuchaniem i że "nie wymaga uzasadnienia" (tego
nie ma w art. 182 -- właściwy byłby art. 186), oraz założył płeć użytkownika
("powinnaś").

## Krok 86 (2026-10-18): drobne naprawy danych historii (mojibake czcionki, zepsute skany)

Zadanie "3" z listy po Kroku 84. Skan jakości wszystkich okien historii (wskaźnik
dziwnych znaków na 1000) znalazł dwa osobne problemy.

**1. Mojibake czcionki w 7 oknach historii sprzed ~2014 (7 dziedzin).** PDF-y
obwieszczeń mają polskie litery zakodowane jako Mac Central European odczytane
jako Mac Roman: "dzia∏ajà" = "działają", "okreÊla" = "określa", "wzgl´du" =
"względu", "U˝yte" = "Użyte", "DZIA¸" = "DZIAŁ", "URZ¢DOWE", "KO¡COWE"
(~45 tys. wystąpień; 35-42 dziwnych znaków/1000 wobec <= 0,9 w zdrowych oknach).
Mapowanie stałe (∏→ł à→ą Ê→ś ´→ę ˝→ż ç→ć ƒ→ń ê→ź ¸→Ł Â→Ś ˚→Ż ¢→Ę ¡→Ń),
`fix_font_mojibake` w `pdf_to_clean_text` (tylko gdy dokument ma >= 30 znaków z
sygnatury "∏à´˝", więc zdrowe teksty nietknięte); 2 testy. Okna: USUS
(<=2013-12-04), KPA (<=2013-02-27), ustawa o ochronie praw lokatorów
(<=2014-01-31), Prawo własności przemysłowej (<=2013-11-29), Prawo ochrony
środowiska (<=2013-10-23), Prawo autorskie (2006-2016), Prawo budowlane
(<=2003-11-21). Zastosowane do istniejących plików historii (kopia zapasowa przed
zmianą), indeksy 7 dziedzin przebudowane (1-2 min, głównie z cache). Sprawdzone na
żywo: zapytania `--as-of` z tych okien zwracają czytelny tekst ("niezdolności do
pracy", "określenie przedmiotu zgłoszenia").

**2. Zepsuty skan OCR: KPA przed 2000-11-17** (94 z 306 artykułów, "Orgpn", "podle~
ga", pomieszana treść) -- inny problem niż mojibake. Bramka jakości w
`download_acts_history.py` (`odd_char_rate_per_1000`, próg 4,0: zdrowe <= 0,9, ten
skan 7,7; 1 test): okno powyżej progu jest pomijane z ostrzeżeniem. Okno usunięte z
`kpa_history.json` i z indeksu `administracyjne`; zapytania `--as-of` sprzed
2000-11-17 trafiają teraz w następne okno (do 2013-02-27), które dla tak dawnych
dat nie jest dokładne -- lepsze niż nieczytelny tekst.

**Kontrola po zmianach.** Pełna bateria `check_expand_query_regressions.py` (73
przypadki, po przebudowie 7 indeksów i dodaniu KPC/KPK/wykroczeń): 69 OK, 4 znane
problemy, **0 nowych regresji**; 211 testów jednostkowych przechodzi.

**Świadomie NIE zrobione.** (a) Zdublowany § 4 / przyszły § 2¹ w KPC art. 477¹⁴
(niesparowane `<` na poziomie paragrafu): objaw nie da się tanio zmierzyć na
pozostałych dziedzinach (zdublowane "§ N." to głównie legalne odwołania "art. X §
1."), a zmiana wspólnego `strip_not_yet_in_force_text` wcześniej była świadomie
odrzucona (Krok 65: lepiej za dużo niż wyciąć obowiązujący tekst). (b)
Spłaszczone odwołania do ustępów w prozie ("§ 11" zamiast "§ 1¹") -- klasa z Kroków
48-49, ok. 15% resztek bez bezpiecznego kontekstu.

## Krok 87 (2026-10-18): dwa z czterech znanych problemów baterii naprawione (`kc_118`, `ksh_61`)

Bateria po Kroku 86 miała 4 znane problemy: `kc_118` (ranga 20), `ksh_61` (13),
`konsumenckie_28` (9), `cywilne_konsument_tv_bez_slowa_konsument` (22). Dwa
ostatnie świadomie zostawione (rangę 9 vs próg 8 nie stroimy pod jedno pytanie;
telewizor to niejednoznaczne pytanie bez słowa "konsument").

**`ksh_61_wystapienie_ze_spolki_jawnej`: ranga 13 -> 1.** Para w podpowiedzi
domeny `gospodarcze` (metoda z Kroków 28/41): "jak wystąpić ze spółki jawnej, gdy
wspólnicy się nie zgadzają" -> "wypowiedzenie przez wspólnika umowy spółki jawnej
zawartej na czas nieoznaczony". Bateria `gospodarcze`: 4/4 OK. Wcześniej model
podawał zmyślony termin 3 miesięcy (art. 61: 6 miesięcy przed końcem roku
obrotowego).

**`kc_118_przedawnienie_pozyczki`: ranga 20 -> 3**, ale nie przez dopisanie pary
do podpowiedzi `cywilne`. Trzy próby (para na początku / na końcu / w środku)
dawały trzy RÓŻNE regresje przypadków konsumenckich (`wina_uzytkownika` 1 -> 36-41,
`laptop_odsyla_do_producenta` 1 -> 23, `kc_118` 3 -> 12 -> 31): domena `cywilne`
miesza KC z ustawą o prawach konsumenta, a model reaguje na długość i układ
podpowiedzi -- ten sam wzorzec "ścigania własnego ogona" co Kroki 25/26/29. Przy
okazji pierwsza próba poprawiła telewizor 22 -> 3, ale niestabilnie (8, potem 22).
**Rozwiązanie: podpowiedź włączana słowem kluczowym**
(`DOMAIN_QUERY_EXPANSION_KEYWORD_HINTS`, `prompt.py`): para o przedawnieniu jest
dopisywana tylko gdy pytanie pasuje do `\bprzedaw` -- pytania bez tego słowa
dostają prompt bajt w bajt jak przed zmianą, więc regresja jest niemożliwa z
konstrukcji. Pułapka: "s**przedaw**ca" zawiera "przedaw", dlatego granica słowa
`\b` (bez niej para trafiałaby do prawie wszystkich pytań konsumenckich). Test
jednostkowy. Bateria `cywilne`: 8/9 OK + telewizor jako znany (22, jak przed
zmianą), `wina_uzytkownika` 1, wszystkie konsumenckie bez zmian.

**Wniosek metodyczny:** domenowa podpowiedź jest bezpieczna, dopóki domena jest
jednolita (`gospodarcze`); w domenie mieszanej (`cywilne`) kolejne pary
destabilizują -- tam wyzwalacz słowem kluczowym zamiast dopisywania do wspólnej listy.

## Krok 88 (2026-10-18): `kadry_i_place` -- hedge-iter50 vs baza na 76 pytaniach (RunPod, ~1,1 $)

Opcja "3" z listy po Kroku 84, z rozwiązaniem problemu czasu z Kroku 83 (lokalnie ~3 h
na 40 par): przebieg na RunPodzie, RTX PRO 6000 Secure 2,49 $/h (Community bez wolnych
maszyn), pod 16:15-ok. 16:41 UTC (~26 min, ~1,1 $; zostaje ok. 2 $). Qwen3.8-27B-FP8
(sędzia) + Bielik z adapterem `hedge-iter50` (produkcyjny, konwersja `mlx-to-peft`)
naraz w vLLM. `scripts/runpod/compare_kadry.sh` + `make_bundle_compare.sh`;
`eval_distill_vllm.py --questions-file/--out-file`; sędzia `--max-tokens 9000`,
`--context-window` (limit myślenia przycinany do wolnego miejsca w oknie 24576, błąd
jednej pary nie przerywa całości).

**Próba.** 76 pytań: 20 dotychczasowych (holdout, baterie, porównanie z Kroku 73) + 56
wylosowanych z wygenerowanych (po jednym na artykuł, `random.Random(0)`;
`data/distill/kadry_i_place/compare_questions.json`). Hedge-iter50 uczył się na ręcznie
pisanym zbiorze, więc pytania destylacyjne go nie zanieczyszczają (k78 pominięty --
Krok 83: gorszy od bazy).

| wariant | adapter lepszy / remis / baza lepsza | poważne błędy adapter / baza | cytat spoza kontekstu | mediana długości |
|---|---|---|---|---|
| hedge-iter50 vs baza | **33 / 11 / 25** (69 z 76 par) | 26 / 25 | 5 / 5 | 721 vs 1145 zn. |

Bez remisów adapter 57% vs baza 43%, test znaków p ~ 0,36 -- **różnica nieistotna**.
Poważne błędy po równo (26 vs 25), cytaty spoza kontekstu po równo (5/5). Wygrane
adaptera: 19 z 33 z poważnym błędem bazy; wygrane bazy: 21 z 25 z poważnym błędem
adaptera. Stare 20 pytań: 8/6/6 (jak w Kroku 83), nowe 56: 25/5/19. Odsetek odpowiedzi z
asekuracją/odesłaniem do prawnika: 24% adapter vs 26% baza -- **efekt "hedge" nie jest
widoczny**; adapter pisze krócej i w próbce przegranych 3 z 8 to "odmowa odpowiedzi
mimo wystarczającego kontekstu" (nielegalne zatrudnienie, kurs spawania z KFS, dłuższe
zatrudnienie art. 36).

**Wniosek.** Na 76 pytaniach produkcyjny adapter kadr NIE daje mierzalnej przewagi nad
bazą (ani poprawności, ani asekuracji), jest za to krótszy i bywa zbyt skłonny do
odmowy. Spójne z Krokiem 83 (20 pytań) i Krokiem 73 (stare adaptery gorsze od bazy). 7
par bez werdyktu (obcięty limit myślenia także przy 9000). **Decyzja do użytkownika:**
przełączyć `kadry_i_place` na samą bazę (jak pozostałe dziedziny)? Produkcja na razie
bez zmian.

**Błąd po drodze (naprawiony, wznowione bez przeładowania modeli).** Pierwszy plik pytań
zawierał `null` (jeden wiersz `questions.jsonl` bez pola `question`) -> `AttributeError`
w ewaluacji; plik odfiltrowany. Pętla czekająca na SSH przez serwer pośredni RunPoda
nie przeszła (klucz tymczasowy działa tylko na adresie bezpośrednim).

## Krok 89 (2026-10-18): `kadry_i_place` przełączone na samą bazę

Za zgodą użytkownika, po Kroku 88 (hedge-iter50 vs baza na 76 pytaniach: 33/11/25,
błędy 26/25, p ~ 0,36; efekt "hedge" niewidoczny, adapter krótszy i czasem zbyt skłonny
do odmowy). `chat.py::DEFAULT_ADAPTER_PATHS` bez `kadry_i_place`; adapter zostaje w
`LEGACY_ADAPTER_PATHS` i na dysku (`--adapter-path adapters/bielik11b-kadry-lora-hedge-iter50`).
**Adaptery domyślne teraz: `konsumenckie` (iter200), `budowlane` (k76 iter584), `cywilne`
(k78 iter550), `mieszkaniowe` (k78 iter413); pozostałe 22 dziedziny na bazie.** README
zaktualizowane; 212 testów przechodzi. Odwrócenie: jedna linijka w słowniku.

## Krok 90 (2026-10-18): kontrola świeżości aktów + dwa błędy parsera znalezione przy odświeżaniu

`check_acts_freshness.py` (przed pilotem nowej dziedziny): **12 aktów ze zmienionymi
metadanymi** (nowe teksty ujednolicone z września/października: Kodeks pracy, USUS, ustawa
zasiłkowa, ustawa o rynku pracy, VAT, Kodeks cywilny, prawo autorskie, ruch drogowy, prawo
lotnicze, geologiczne i górnicze, wodne, komunikacja elektroniczna) i jeden uchylony
(Prawo telekomunikacyjne -- znany, zostaje wyłącznie w historii). Kredyt konsumencki (CCD2
od 20.11.2026 nadal bez następcy), wykroczenia, KPC i KPK aktualne.

Odświeżenie ujawniło **dwie regresje parsera** (nowy format PDF Kancelarii Sejmu):

**1. Indeksy górne w treści jako "[n]".** Nowsze PDF-y zapisują indeks górny jako cyfry w
nawiasie przyklejone do liczby: `§ 2[1]`, `art. 22[1]`, `pkt 5[2])`, `ust. 3[3d]`. Nagłówki
artykułów ("Art. 18[4].") miały już obsługę (geometria PDF), ale w treści nawiasy zdejmował
`strip_not_yet_in_force_text` jak adnotację redakcyjną, więc cyfra się spłaszczała ("§ 21",
"art. 221"). Po odświeżeniu **Kodeks pracy stracił 262 z 498 indeksów górnych**. To samo
tłumaczy resztki w KPC z Kroku 84 ("§ 11", "art. 20512"). Nowa funkcja
`convert_bracket_superscripts` (w `pdf_to_clean_text`): `cyfra[1-3 cyfry + litery]` ->
indeks górny; nawias redakcyjny (po spacji, "[tekst]", "[1. ...]") nie pasuje; NAGŁÓWKI
("Art." wielką literą) pominięte -- pierwsza wersja zamieniała też je i wchłonęła 28
artykułów Kodeksu pracy (480 -> 452; "Art. 18³c." nie pasuje do wzorca podziału).
Wynik: Kodeks pracy 480 artykułów (te same etykiety), indeksy 498 -> 560; pełne
pobranie wszystkich aktów: **tylko wzrosty** indeksów górnych w 11 aktach (KPC **+902**,
upadłościowe +141, KC +66, ordynacja +16, restrukturyzacyjne +11, zasiłkowa +6, PIT +5,
lotnicze +5, rodzinne +3, zamówienia +3, energetyczne +1), zero spadków, zmian liczby
artykułów ani etykiet, zero śladów nawiasów.

**2. Prawo lotnicze: 97 tys. znaków załączników w art. 232.** Nowy PDF nie ma nagłówka
"Załączniki do ustawy z dnia...", tylko od razu "Załącznik nr 1" po ostatnim artykule, więc
`split_annexes` niczego nie wydzielał (519 -> 510 wpisów, 9 załączników wchłonięte przez art.
232 "wchodzi w życie"). Zapasowy wzorzec `ANNEX_LINE_START_RE`: "Załącznik nr N" po końcu
zdania i przed wielką literą/nawiasem (tekst artykułu ma już sklejone linie, więc pierwsza
wersja "cała linia" nie zadziałała). Wynik: art. 232 = 173 znaki, 9 załączników, 519 wpisów;
żaden inny akt się nie zmienił. Testy: 2 (nawiasy), 1 (załączniki).

**Odświeżone i przebudowane:** bieżące teksty 12 aktów + historia 12 aktów (bez strat, kilka
przyrostów, lotnicze/PIT +46/+21 wpisów z podzielonych załączników), indeksy dziedzin:
`kadry_i_place`, `podatkowe`, `cywilne`, `postepowanie_cywilne`, `rodzinne`, `energetyczne`,
`lotnicze`, `upadlosciowe`, `zamowienia_publiczne`, `autorskie`, `ruch_drogowy`,
`geologia_i_gornictwo`, `wodne`, `telekomunikacyjne`. **Kontrola końcowa:** pełna bateria `check_expand_query_regressions.py` (73 przypadki): 71 OK, 2 znane
(celowo zostawione: `konsumenckie_28`, telewizor), **0 nowych regresji**; 214 testów jednostkowych.
Zmiany treści w wielu artykułach to
w większości szum odstępów z PDF-a; merytoryczne nowelizacje m.in. KP art. 97 (Prawo
pocztowe, Dz.U. 2026 poz. 558).

## Krok 91 (2026-10-18): dwudziesta siódma dziedzina -- prawa pacjenta

**Akt (ELI zweryfikowany).** Dz.U. 2009 poz. 417, "Ustawa z dnia 6 listopada 2008 r. o prawach
pacjenta i Rzeczniku Praw Pacjenta" (IN_FORCE, z tekstem jednolitym). 99 artykułów (w tym
67q-67zi: świadczenie kompensacyjne), historia 641 wpisów z 7 wersji (0 śladów nawiasów, 0
spłaszczonych etykiet, 0 duplikatów, szum skanów 0,0/1000), indeks 1314 fragmentów. Pilot po
Kroku 90 (nowy parser: nawiasy `[n]`, załączniki bez nagłówka) -- bez problemów danych.
Artykuły sprawdzone po TREŚCI przed napisaniem przypadków; zgoda na zabieg to art. **17**
(nie 32: tam Komisja Lekarska -- zgoda z art. 32 pochodzi z ustawy o zawodach lekarza).

**Retrieval (bez hintu `expand_query`).** Surowe pytania: 9/9 w top-12, 7 na pozycji 1
(najsłabsze: art. 50 skarga do Rzecznika -- ranga 6; klaster świadczenia kompensacyjnego
67r-67zc: art. 67r ranga 12, procedura 67t ranga 3). Z prawdziwym `expand_query`: **9/9 w
top-8, 7 na pozycji 1** (art. 67t: 5; **art. 50: 8 -- dokładnie na progu**, do obserwacji).
Przypadek kompensacyjny oczekuje art. 67t (odpowiedź na "jak uzyskać"), nie definicji
uprawnienia 67r. Bateria +9 (`pac_*`), razem 82 przypadki; bez zmian w prompcie i bez
hintu.

**Test końcowy** (`chat.py --domain prawa_pacjenta`, baza): dokumentacja zmarłego ojca ->
art. 26 ust. 2 (osoba upoważniona / przedstawiciel / osoba bliska, o ile nikt się nie
sprzeciwi) -- zgodne. **Błąd merytoryczny w drugiej części:** model napisał, że osoba bliska
dostanie dokumentację *bezpłatnie* "przy pierwszym wniosku" (art. 28 ust. 2a pkt 1), a ten
przepis zwalnia z opłaty tylko *pacjenta albo jego przedstawiciela ustawowego*. Klasa
Reguły 6 (cytat przepisu, wniosek poza jego zakresem, Kroki 15/16/30/32/43) -- nie błąd
danych ani wyszukiwania, nie naprawiany. Wniosek praktyczny dla tej dziedziny:
odpowiedzi o opłatach i kręgu uprawnionych wymagają ostrożności.

## Krok 92 (2026-10-18): README -- sekcja "Stan projektu"

Zadanie "2" z listy po Kroku 91. README rosło przyrostowo przez 90 kroków i zawierało
zdania sprzeczne ze stanem faktycznym ("każda dziedzina ma własny adapter LoRA", LoRA jako
źródło stylu/asekuracji). Dodana na górze sekcja **"Stan projektu (2026-10-18)"**: 27
dziedzin / 38 aktów, tabela dziedzina -> akty -> adapter/baza (4 adaptery, 23 bazy), lista
ograniczeń ważnych przed użyciem (nie porada prawna; błąd "cytat vs wniosek" Reguły 6 i
opcjonalny tryb Qwena 27B; znane luki wyszukiwania i bateria 82 przypadków; zależność od
formatu PDF-ów Kancelarii), recepta na nową dziedzinę (~1 h). Poprawione dwa sprzeczne
zdania (opis roli LoRA, "własny adapter" w sekcji dziedzin). Tylko dokumentacja.

## Krok 93 (2026-10-20): reranker a długie artykuły -- pomiar, bez zmiany; pula baterii = produkcja

Zadanie "1" z listy po Kroku 92: reranker ocenia pierwsze `RERANK_TEXT_CHARS`=1500 znaków
artykułu, więc ustęp trafiony głęboko w długim artykule (Krok 77 naprawił to tylko w
kontekście modelu) jest dla niego niewidoczny. Krok 77 odrzucił okno wokół fragmentu, bo
psuło `vat_86`, ale na baterii, w której żaden cel nie leżał głęboko.

**Pomiar.** +9 przypadków `deep_*` (odpowiedź za znakiem 1500: art. 30 ust. 5 Pr. bud.,
art. 21 ust. 1 pkt 131 i 148 PIT, art. 86 ust. 10 i 113 ust. 9 VAT, art. 135 PoRD, art. 70
§ 6 OP, art. 38 pkt 13 UPK, art. 26 PIT). Rozszerzenia policzone raz, potem 5 wariantów
tekstu dla rerankera na tych samych rozszerzeniach (pula produkcyjna 30, 91 przypadków):

| wariant | OK | deep OK | suma rang |
|---|---|---|---|
| początek 1500 (obecny) | 88 | 8/9 | 203 |
| okno wokół fragmentu | 88 | 8/9 | 220 |
| okno + 200 zn. początku | 88 | 8/9 | 206 |
| max(początek, okno) | 87 | 8/9 | 206 |
| max(początek, okno+200) | 89 | 9/9 | 205 |

Różnice w szumie (±1-2 pozycje, część przypadków w górę, część w dół). Powód: etap
kosinusowy trafia we właściwy fragment (indeks tnie długie artykuły na okna), a fuzja RRF
trzyma artykuł wysoko nawet przy słabej ocenie rerankera -- art. 30 Pr. bud. jest #1 we
wszystkich wariantach. **Reranker bez zmian.** Przypadki `deep_*` zostają w baterii.

**Pula baterii = produkcja.** `check_expand_query_regressions.py` wołał `search(top_k=300)`,
więc pula kandydatów na wariant zapytania wynosiła 300, a w produkcji (`top_k=8`) 30. Fuzja
RRF liczy pozycje w puli, więc rangi w baterii różniły się od produkcyjnych (np. `vat_86`:
#8 w baterii, #6 przy puli 30). Nowy parametr `RagIndex.search(candidate_pool=...)`; bateria
używa puli 30 i zwraca całą scaloną listę (do 60). Dawne rangi z PROGRESS.md (pula 300) nie
są wprost porównywalne z nowymi.

**Zadanie "2" (znane problemy).** Jedyny przypadek, gdzie ucięcie w rerankerze realnie
szkodzi: `deep_pit_21_pkt131` (#9) -- art. 21 PIT kosinus 0,831, reranker 0,268 (widzi
początek listy zwolnień), wygrywa art. 30e. Naprawa wzorcem z Kroku 87 (podpowiedź
włączana słowem kluczowym, `podatkowe`: sprzedaż + mieszkanie/dom/nieruchomość/działka/
lokal) -> **#2**. Przypadek kontrolny `pit_30e_sprzedaz_dzialki` (sprzedaż działki bez
wątku mieszkaniowego, cel art. 30e) -> #1. Pozostałe pytania nie pasują do wzorca, więc
dostają identyczny prompt. `konsumenckie_28` (#9) zostaje znanym problemem: reranker daje
mu 0,949, ale ~10 artykułów o odstąpieniu ma 0,94-0,99 (zatłoczony temat; poprzednie
podpowiedzi już go rozregulowały). `cywilne_tv` -- celowo niedostrajany (Krok 71).

Bateria: 92 przypadki, 90 OK + 2 znane, 0 regresji (pełny przebieg + `--domain podatkowe`
po podpowiedzi). pytest 214/214.

## Krok 94 (2026-10-22): bateria Qwena; spadki i darowizny w `podatkowe`

**Bateria odpowiedzi Qwena (LM Studio, qwen3.8-27b, pierwsza od Kroku 91):** 32 przypadki --
29 OK, 2 znane, 1 fałszywy alarm: `wykroczenia_119_kradziez_300zl_odpowiedz` -- odpowiedź
poprawna (wykroczenie, KW art. 119 § 1 i § 4), ale zakaz samego napisu `art. 278` łapał
zastrzeżenie "w fragmentach nie ma art. 278 KK". Kryterium zmienione na wniosek
("stanowi przestępstwo", "jest przestępstwem", ...) -- sprawdzone na tej odpowiedzi (OK) i na
błędnej odpowiedzi testowej (łapie). Uwaga: przed baterią trzeba `lms server start` --
`lms load` sam nie uruchamia serwera API.

**Akt (ELI zweryfikowany).** Dz.U. 1983 poz. 207, "Ustawa z dnia 28 lipca 1983 r. o podatku od
spadków i darowizn." (tekst jednolity), dołączony do `podatkowe` (nie osobna dziedzina --
pytania mieszają się z PIT, np. sprzedaż odziedziczonego mieszkania). 28 artykułów (1-21 z
1a, 4a-4c, 17a, 19a, 19b), historia 249 wpisów z 11 okien, indeks `podatkowe` 36 813
fragmentów. **Znana usterka danych:** najstarsze okno (do 2004-06-22, bez `valid_from`) to
zły skan -- tylko 6 artykułów, litery rozstrzelone spacjami ("Zap ł a t a spad kobiercy");
bramka z Kroku 86 go nie łapie (0,42/1000 < 4,0 -- znaki są poprawne, psuje się tylko
odstęp). Dotyczy wyłącznie pytań "na dzień" sprzed 2004-06-22 (filtr dat w wyszukiwaniu);
nie naprawiane -- wykrywanie rozstrzelonych liter to osobna zmiana parsera dla wszystkich
aktów.

**Retrieval.** Bateria +9 (8 `sd_*`: art. 3, 4a x2, 6, 9, 14, 15, 16 -- sprawdzone po treści;
`pit_10_sprzedaz_odziedziczonego` -- strażnik granicy PIT/spadki, cel PIT art. 10). A/B na tych
samych pytaniach, indeks z nowym aktem vs bez (odfiltrowane fragmenty i tytuł):

| przypadek | bez aktu | z aktem |
|---|---|---|
| vat_86_zakupy_prywatne | 6 | 6 |
| deep_pit_21_pkt131 | 2 | 3 |
| **deep_pit_21_pkt148 (ulga dla młodych)** | **2** | **8** |
| deep_vat_86_ust10 | 7 | 7 |
| deep_vat_113_ust9 | 1 | 2 |
| deep_op_70_par6 | 2 | 2 |
| deep_pit_26 | 1 | 1 |
| pit_30e_sprzedaz_dzialki | 1 | 1 |
| pit_10_sprzedaz_odziedziczonego | 7 | 5 |

`pkt148` -- przyczyną NIE są nowe fragmenty (stare rozszerzenie na nowym indeksie daje
identyczny ranking), tylko samo rozszerzenie: czwarty tytuł ustawy w kontekście
`expand_query` przestawił Bielika na styl "Pytanie dotyczy obowiązku podatkowego ..." i
zgubił "24 lata"/"wynagrodzenie". Mieści się w progu (top-8), na granicy -- do obserwacji,
nie strojone (kruchość promptu na listę aktów, nie problem tego aktu).
Nowe: 7/8 w top-4 (6 na #1); `sd_4a_darowizna_od_rodzicow` poza pulą 60 -- art. 4a mówi
"zstępni, wstępni", nie "rodzice", wygrywał art. 16 (ulga mieszkaniowa). Naprawa wzorcem z
Kroków 87/93: druga podpowiedź słowem kluczowym w `podatkowe` (osoba z rodziny + darowizna/
spadek -> "zwolnienie ... małżonka, zstępnych, wstępnych lub rodzeństwo pod warunkiem
zgłoszenia") -> **#1**; "mam" jako czasownik celowo nie wyzwala. Pozostałe rangi bez zmian.
`--domain podatkowe` 17/17; pełna bateria 101: 99 OK + 2 znane (`konsumenckie_28` #9, `cywilne_tv` #17), 0 regresji; pytest 214/214.

**Test końcowy** (`chat.py --domain podatkowe`, baza): 200 tys. zł od rodziców przelewem ->
art. 4a ust. 1 (zstępni, zgłoszenie w 6 miesięcy, udokumentowanie przelewem) -- zgodne.
Luka: kwoty z art. 9 ust. 1 pkt 1 model nie podał ("nie jest podana we fragmencie") -- art. 9
nie trafił do kontekstu; nie błąd.

## Krok 95 (2026-10-24): dwudziesta ósma dziedzina -- `pcc`

**Akt (ELI zweryfikowany).** Dz.U. 2000 poz. 959, "Ustawa z dnia 9 września 2000 r. o podatku od
czynności cywilnoprawnych." (tekst jednolity): 19 artykułów, historia 199 wpisów z 11 okien.

**Najpierw próba wg planu z Kroku 94 -- dołączenie do `podatkowe`: regresja.** Bateria
`podatkowe` (24 przypadki, +7 `pcc_*` sprawdzonych po treści): `pit_10_sprzedaz_odziedziczonego`
#5 -> #11 (wypierają go art. 3, 6, 7 PCC -- też o sprzedaży nieruchomości), `sd_9_kwota_wolna_ciocia`
OK -> poza pulą 60 (przyczyna niezdiagnozowana: w czołówce nie ma fragmentów PCC), a 3 z 7 nowych
(`pcc_4`, `pcc_2` poza pulą, `pcc_9` #13) poza top-8. Zgodnie z regułą z Kroku 94 (osobna
dziedzina przy regresji) -- na decyzję użytkownika PCC jest własną dziedziną `pcc`.

**Wynik:** `podatkowe` wraca do 36 813 fragmentów, `pcc` 651. Bateria 108 przypadków: tylko 2 znane
(`konsumenckie_28` #9, `cywilne_tv` #17), 0 regresji; pytest 214/214. Test końcowy
(`chat.py --domain pcc --no-adapter`): auto za 30 tys. zł -> 2% = 600 zł, art. 7 ust. 1 pkt 1
lit. b, 14 dni z art. 10 -- zgodne. Dziedzina na bazie, bez adaptera.

Wniosek do CIT i podatków lokalnych: spodziewać się tego samego (nakładanie z PIT/VAT) --
robić od razu jako osobne dziedziny.

## Krok 96 (2026-10-24): dwudziesta dziewiąta dziedzina -- `cit`

**Akt (ELI zweryfikowany).** Dz.U. 1992 poz. 86, "Ustawa z dnia 15 lutego 1992 r. o podatku
dochodowym od osób prawnych" (tekst jednolity), od razu jako osobna dziedzina (wniosek z Kroku 95).
219 artykułów w bieżącym tekście, historia 1665 wpisów z 13 okien, indeks 10 338 fragmentów.
Kontrola danych: 0 duplikatów etykiet, 0 zbłąkanych `<>[]`.
**Znana usterka danych:** trzy najstarsze okna (do 2000-07-13: 14 art.; do 2011-03-09: 59 art.;
do 2017-11-21: 94 art.) wyglądają na niepełne -- stare układy PDF, miękkie dywizy w tekście
("obowiąz­ kowi"). Dotyczy tylko pytań `--as-of` sprzed 2011-03-09; nie naprawiane (parser wspólny
dla wszystkich aktów).

**Retrieval.** +10 przypadków `cit_*` (sprawdzone po treści artykułów): 9/10 w top-8 (cztery na
#1-2: art. 15, 24b, 28j...). `cit_19_stawka_9` #8 (na granicy). **`cit_16_kary_nie_koszt` #9**:
art. 16 ma ponad 50 punktów (grzywny to pkt 18), wypierają go art. 15d, 26c, 15cb. Próba: podpowiedź
słowem kluczowym dla `cit` -- Bielik ją zignorował, ranga bez zmiany, **cofnięta**; przypadek
oznaczony jako znany. Bateria 118: 115 OK + 3 znane (`konsumenckie_28`, `cywilne_tv`,
`cit_16`), 0 regresji; pytest 214/214.

**Test końcowy** (`chat.py --domain cit --no-adapter`, grzywna w kosztach): wniosek poprawny
(nie jest kosztem, art. 15 ust. 1 i art. 16 ust. 1), ale numer punktu błędny -- model podał pkt 19,
a grzywny to pkt 18 (pkt 19 to kary umowne i odszkodowania). Do obserwacji, nie błąd wniosku.

## Krok 97 (2026-10-24): trzydziesta dziedzina -- `podatki_lokalne`

**Akt (ELI zweryfikowany).** Dz.U. 1991 poz. 31, "Ustawa z dnia 12 stycznia 1991 r. o podatkach i
opłatach lokalnych" (tekst jednolity), od razu jako osobna dziedzina (wniosek z Kroku 95).
42 pozycje w bieżącym tekście (artykuły + 4 załączniki), historia 335 wpisów, indeks 954 fragmenty.
Kontrola danych: 0 zbłąkanych `<>[]`.

**Retrieval.** +11 przypadków `pl_*` (sprawdzone po treści artykułów): 11/11 OK, większość na
#1-2. Bateria 129: 126 OK + 3 znane (`konsumenckie_28`, `cywilne_tv`, `cit_16`), 0 regresji;
pytest 214/214. Nie wykonano testu końcowego `chat.py` ani baterii Qwena.

## Krok 98 (2026-10-24): bateria Qwena -- `podatki_lokalne`

+2 przypadki w `data/qwen_answer_reference_cases.json` (34 przypadki): `pl_3_podatnik_najem_odpowiedz`
(art. 3) i `pl_6_poczatek_obowiazku_odpowiedz` (art. 6). Qwen3.8-27B 8-bit,
`lms load ... --context-length 24576`, `reasoning_effort=low`. Oba OK, odpowiedzi przeczytane ręcznie
i sprawdzone z tekstem ustawy (art. 3 ust. 1 pkt 1 i 4, ust. 3, art. 1c, art. 6 ust. 1, 5, 6, 7) --
wnioski poprawne (płaci właściciel; obowiązek od 1 kwietnia).

**Pełna bateria Qwena po Krokach 95-98:** 34 przypadki, 33 OK + 1 znany (`ochrona_danych_101`,
zaakceptowany kompromis z Kroków 25/28/29), 0 nowych regresji. Pozostałe przypadki ze `known_issue`
(`cywilne_obraz_kopia_zasada6`, `ksh_210`, `wodne_394`) tym razem przeszły. Nie czytano ręcznie
wszystkich 34 odpowiedzi -- tylko dwa nowe przypadki `podatki_lokalne`.

## Krok 99 (2026-10-28): router dziedzin (Basal) -- wariant minimalny, `--domain auto`

`scripts/router.py` (klient HTTP do `basal-serve`, bez torcha/mlx) + `tests/test_router.py` (16 testów).
Basal `basal-2.0-4.5B-MLX-8bit`, opisy dziedzin = nazwy ustaw z `acts_config.json`, tylko dziedziny
z zbudowanym indeksem (30). `--domain auto` w `rag_search.py`, oraz w `chat.py`/`chat_lmstudio.py`
(tylko z `--prompt`: dziedzina i adapter wybierane raz na starcie). Używany jest tylko `primary`;
`secondary` (pewność < 0,9) jest zwracane, ale jeszcze nieużywane. `basal-serve` to drugi proces MLX --
przed ładowaniem Bielika zatrzymać (jeden proces MLX naraz).

**Pomiar na 129 pytaniach z `expand_query_reference_cases.json`** (sam router, bez generowania):
`primary` zgodne z dziedziną przypadku 99/129 (76,7%), 105/129 (81,4%) gdy akt należy też do wskazanej
dziedziny, właściwa dziedzina w top-2 121/129 (93,8%). Próg 0,9: pokrycie 77, trafność 75 (97,4%);
próg 0,95: 61 pytań, 60 trafnych. Typowe pomyłki to sąsiednie dziedziny (postępowanie karne -> karne,
PCC/CIT -> podatkowe, wykroczenia -> ruch drogowy); 8 pytań poza top-2, m.in. `kpc_882` (-> kadry_i_place, 0,94)
i `kpk_244` (-> ruch_drogowy, 0,9). Wariant minimalny kieruje do złej dziedziny ok. 19-23% pytań;
następny krok: pełny wariant z `secondary` (wyszukiwanie w dwóch indeksach i scalanie).

## Krok 100 (2026-10-28): router dziedzin -- pełny wariant z `secondary` (sklejone indeksy)

Przy pewności routera < 0,9 (52 z 129 pytań) szukamy w DWÓCH dziedzinach: `RagIndex(extra_domains=[...])`
skleja `embeddings`/`meta` obu indeksów (ten sam model embeddingowy, więc kosinusy porównywalne; reranker
i fuzja bez zmian; wspólna ustawa zwija się w `rank_chunks` per (act_short, article)).
`build_system_prompt(domain, extra_domain)` opisuje obie dziedziny ("oraz"). `chat.py` przy dwóch
dziedzinach NIE ładuje adaptera (adapter jednej dziedziny mógłby zniekształcić odpowiedź z drugiej).
`router.py --battery WYNIK.json` routuje bazę 129 pytań; `check_expand_query_regressions.py
--routing-file WYNIK.json [--primary-only]` mierzy retrieval na dziedzinach z routera.
Testy: 91 (router, rag_search, prompt) przechodzi.

**Pomiar: bateria 129 (rozszerzanie zapytania Bielik 11B + retrieval), przypadek OK = ranga w progu:**

| Dziedziny użyte do wyszukiwania | OK | nowe regresje |
|---|---|---|
| z przypadków referencyjnych (stan odniesienia) | 126 (+3 znane) | 0 |
| router, tylko `primary` (Krok 99) | 104 | 23 |
| router, `primary` + `secondary` | 113 | 15 |

`secondary` naprawia 15 przypadków, ale psuje 6, które `primary` miał dobrze (m.in. `konsumenckie_43c`,
`cywilne_konsument_rower`, `pcc_4`, `cit_11c`) -- większa pula kandydatów z dwóch indeksów spycha właściwy
artykuł poza próg. Nadal źle: 15 przypadków, z czego ok. 8 to pomyłki routera poza top-2 (m.in. `kpc_882`
-> kadry_i_place 0,94, `kpk_244` -> ruch_drogowy 0,9, `pac_31`, `telekomunikacyjne_399`, `kk_44`, `kk_57`).
Router sam w sobie nie zastępuje ręcznego `--domain` (87,6% vs 97,7% przy ręcznie podanej dziedzinie).

Ograniczenie: bateria to przypadki referencyjne, nie pytania prawdziwych użytkowników; odpowiedzi
generowane (chat.py/Qwen) z dwoma dziedzinami NIE były mierzone -- tylko retrieval.

Możliwe dalsze kroki: węższy warunek dodawania `secondary` (np. próg 0,8 zamiast 0,9 albo
`secondary` tylko gdy jego prawdopodobieństwo jest bliskie `primary`), żeby ograniczyć 6 psujących
przypadków; ocena na większej bazie pytań poza zbiorem referencyjnym.

**Krok 100b -- próg dla `secondary`.** Wyniki per przypadek z dwóch przebiegów (tylko `primary` i
z `secondary`) pozwalają policzyć dowolny próg ≤ 0,9 bez nowych przebiegów: przypadek bierze wynik
z przebiegu `secondary`, gdy pewność < progu, inaczej z `primary`. OK na 129: próg 0 (brak `secondary`)
104; 0,5 -> 105; 0,6 -> 108; 0,7 -> 111; 0,75 -> 111; 0,8 -> 113 (32 pytania z `secondary`);
0,85 -> 116 (42); 0,9 -> 113 (52). `DEFAULT_SINGLE_THRESHOLD` 0,9 -> 0,85. Uwaga: wynik jest dobrany
na tej samej baterii (przewaga 3 przypadków nad 0,9), więc to słaby dowód; liczby są złożeniem
dwóch zmierzonych przebiegów, nie osobnym przebiegiem z progiem 0,85. Pliki wyników (nieśledzone):
`data/processed/routing_eval/` (routing.json, bat_gold/primary/secondary.txt).

## Krok 101 (2026-10-28): zbiór kontrolny routera (poza baterią 129) -- potwierdzenie `secondary` i progu

`data/routing_holdout_cases.json`: 36 nowych pytań z 22 dziedzin (te same pola co
`expand_query_reference_cases.json`), napisanych po fakcie z naciskiem na pary sąsiednich dziedzin, w których
router się mylił. Każdy artykuł sprawdzony z tekstem w indeksie; odrzucone 15 artykułów już obecnych w baterii
129 i poprawione 8 błędnych etykiet z pierwszych szkiców. `router.py --battery ... --cases ... --single-threshold`
(zapisuje też prawdopodobieństwa), `check_expand_query_regressions.py --cases`.

**Sam router:** `primary` zgodne z dziedziną przypadku 31/36 (86,1%), właściwa dziedzina w top-2 36/36; pomyłki
to te same pary co w Kroku 99 (postępowanie karne -> karne, wykroczenia -> ruch drogowy, cywilne -> postępowanie
cywilne), wszystkie przy pewności 0,56-0,65. Wynik nie jest dowodem lepszej trafności niż 76,7% na baterii 129 --
zbiór jest mały i pisany pod znane pomyłki.

**Retrieval (Bielik 11B rozszerza zapytanie), OK = ranga w progu:** dziedziny z przypadków 36/36; router tylko
`primary` 31/36; `secondary` zawsze 34/36; `secondary` gdy pewność < 0,7 / 0,8 / 0,85 / 0,9 -- 36/36 (6 / 8 / 8 / 8
pytań z drugą dziedziną). `secondary` naprawia 5 przypadków (m.in. `kw_86`, `kpk_23a`, `kc_6`, `kpk_78`, `kpk_303`),
a zawsze-włączone psuje 2 pewne (`kpc_796` 0,93 i `kk_286` 0,95 -- ranga 11 i 19), czyli próg pewności robi to, co
miał robić. Na tym zbiorze progi 0,7-0,9 się nie różnią, więc 0,85 (Krok 100b) pozostaje bez niezależnego
potwierdzenia ponad 0,9; nie ma też sygnału przeciw. Pliki wyników: `data/processed/routing_eval/` (nieśledzone).

## Krok 102 (2026-10-31): odpowiedzi Qwena z routerem (2 dziedziny) -- 7/8 OK, 1 regresja z błędu routera

Bateria odpowiedzi Qwena (34 przypadki): router zmienia dziedzinę w 8. `check_qwen_answer_regressions.py`
dostał `--routing-file`, `--secondary-below`, `--only-changed`; `router.py` -- `effective_domains()`. Zmierzone
tylko te 8 przypadków (Qwen 3.8 27B, kontekst 24576), odpowiedzi przeczytane ręcznie.

**7 OK:** `cywilne_obraz_kopia_zasada6` (konsumenckie+cywilne), `pos_188`, `kro_37`, `wodne_394` (cytuje poprawnie
art. 394 ust. 1 pkt 9), `kk_49`, `wykroczenia_119`, oraz `cywilne_refusal_vat_w_domenie` -- tu router kieruje do
podatkowych (0,99) i Qwen odpowiada merytorycznie zamiast odmówić; to zamierzone działanie routera, nie regresja.

**1 regresja:** `telekomunikacyjne_304_rezygnacja_telefon_promocyjny` -- router wybrał `konsumenckie`+`cywilne`
zamiast `telekomunikacyjne`, art. 304 nie trafił do kontekstu i Qwen odmówił pełnej odpowiedzi (brak przepisów).
Przyczyna po stronie routera (pomyłka sąsiednich dziedzin), nie generacji. Nie naprawiane w tym kroku.

## Krok 103 (2026-10-31): router chował telekomunikację za `konsumenckie` -- dopisek do opisu dziedziny (WYCOFANY w 103c)

**Diagnoza** (przypadek `telekomunikacyjne_304`: konsumenckie 0,36, cywilne 0,27, kredyt_konsumencki 0,19,
telekomunikacyjne 0,09 -- 4. miejsce, więc próg `secondary` nie pomoże). 12 nowych pytań o telekomunikację
(`data/routing_telecom_cases.json`): 9/12 trafia (primary albo secondary). `tele_07` (odstąpienie od umowy u kuriera)
to NIE pudło -- złoty art. 27 ustawy o prawach konsumenta, dziedzina konsumenckie jest poprawna. Prawdziwe pomyłki:
`tele_10` (automatyczne przedłużenie umowy, art. 302) i `tele_11` (zmiana pakietu, art. 307) -- router pewny
konsumenckie 0,89-0,90, retrieval przez router: poza pulą 60; przy złotej dziedzinie ranga 1 i 3. Wzorzec:
pytania o warunki umowy abonamentowej (wypowiedzenie, przedłużenie, zmiana warunków).

**Próba (Krok 103, wycofana w 103c):** `DOMAIN_HINTS` w `router.py` -- dopisek do opisu `telekomunikacyjne` (abonament,
operator, wypowiedzenie, przedłużenie, zmiana warunków/pakietu, roaming, reklamacja, przeniesienie numeru).
Pomiar samego routera (efektywne dziedziny = primary + secondary przy pewności < 0,85) wyszedł optymistycznie: bateria
129 -- 116 -> 122 trafień dziedziny, zbiór kontrolny 36/36, 12 pytań + 304 -- 10 -> 12 z 13 (pogorszone tele_09).

**Pomiar rzeczywisty (Krok 103c, retrieval Bielika na całej baterii 129 z nowym routingiem): 114 OK wobec 116 (-2).**
Baza odtworzona z `bat_primary/bat_secondary/routing.json` (dała dokładnie 116). Różnią się 4 przypadki: gorzej
`cywilne_konsument_rower_zwrot_pieniedzy` (ranga 9), `cywilne_konsument_wina_uzytkownika` (poza pulą 60),
`cywilne_konsument_laptop_odsyla_do_producenta` (ranga 12) -- pewność routera spadła z 0,86-0,88 do 0,81-0,83, więc
przekroczył próg 0,85 i dołożył `cywilne` jako drugą dziedzinę, co wypchnęło art. 43c/43e z top-8; lepiej
`kk_57_kredyt_wiazany_rower`. Żaden przypadek telekomunikacyjny z baterii nie zyskał (`telekomunikacyjne_399` poza pulą
60 przed i po). **Dlaczego proxy mylił:** liczy dziedzinę z etykiety przypadku, a te trzy mają etykietę `cywilne`, choć
artykuł leży w ustawie o prawach konsumenta (dziedzina `konsumenckie`) -- "zysk" był pozorny.

**Decyzja:** wycofane (reguła ustalona przed pomiarem: bateria nie może stracić trafień). Zostaje diagnoza i
`data/routing_telecom_cases.json`; pomyłki telekomunikacyjne -> konsumenckie (304, tele_10, tele_11) to **znane
ograniczenie routera** -- wzorzec: pytania o warunki umowy abonamentowej, router pewny konsumenckie 0,89-0,90.
**Wniosek metodyczny:** zmiany routera oceniać retrievalem (`check_expand_query_regressions.py --routing-file`), nie
zgodnością etykiety dziedziny; każda zmiana promptu Basala przesuwa pewność wokół progu 0,85 i przerzuca kilka
przypadków. Pomocnicze: jeśli telekomunikacja ma być naprawiona, wymaga innego mechanizmu niż opis etykiety.

**Wolne testy (wyjaśnione):** pełne `pytest tests` trwa kilka minut (99% CPU), bo `tests/test_download_acts.py` ma 8
testów end-to-end na PDF-ach przez pdfplumber (5-20+ s każdy; `test_process_act_version_round_trip_on_real_announcement`
dodatkowo odpytuje żywe ELI API) -- mają `skipif` na brak lokalnych danych, a tu dane są, więc się uruchamiają. Reszta
(12 plików, ok. 120 testów) kończy się w 0,1-3,2 s. Szybki zestaw:
`.venv/bin/python -m pytest tests --deselect tests/test_download_acts.py` (sekundy); wolne testy osobno, ręcznie:
`.venv/bin/python -m pytest tests/test_download_acts.py`.

## Krok 104 (2026-10-31): ocena "agentic RAG" (wersja minimalna) -- pętla "słaby wynik -> poszerz dziedziny" NIE pomaga

**Hipoteza:** retrieval z pętlą "jeśli wynik wygląda słabo, poszerz wyszukiwanie" naprawi pomyłki routera (304, tele_10,
tele_11) bez utraty trafień. **Bramka** (ustalona przed pomiarem): naprawić te 3 bez strat na baterii 129.

**Metoda:** `scripts/eval_agentic_fallback.py` -- dla 169 przypadków (bateria 129, zbiór kontrolny 36, 3 pytania
telekomunikacyjne z artykułami + 304) jedno rozszerzenie Bielika, potem wyszukiwanie w konfiguracjach dziedzin:
P (primary), S (primary + drugi), T3 (trzy najbardziej prawdopodobne), ALL (wszystkie dziedziny), N2 (sama druga,
z własnym rozszerzeniem), G (dziedzina przypadku, własne rozszerzenie -- odniesienie). Dla każdej ranga artykułu
i sygnały pewności (rerank top1 / max top-8 / średnia top-3). Polityki liczone offline. Wyniki (nieśledzone):
`data/processed/routing_eval/agentic_out.jsonl`, `agentic_jobs.json`. Ograniczenie: w P/S/T3/ALL rozszerzenie jest
generowane raz (dla primary), nie per konfiguracja.

**Wynik** (OK = ranga <= próg, z 169): P 136, S 149, T3 150, ALL 149, N2 33, **G 166**, oracle (najlepsza z konfiguracji)
167, **baza routera (S gdy pewność < 0,85) 154** (bateria 117, kontrolny 36/36, telekomunikacja 1/4).
- Polityka "P, a gdy sygnał rerankera < τ -> S/T3/ALL": najlepsza 154 (max_rerank, ALL, τ=0,98, eskalacja w 80 z 169
  przypadków) -- tyle co baza, kosztem eskalacji prawie połowy zapytań; przy rozsądnych progach 138-152.
- "Baza, a gdy sygnał < τ -> T3/ALL": netto -5..+1 (najlepsze +1: top1_rerank, ALL, τ=0,8, +2/-1, 12 eskalacji) --
  w granicach szumu.
- "ALL zamiast S gdy pewność routera < 0,85": 156 (+2) -- jedyna wariacja nieco powyżej bazy, ten sam zbiór, słaby dowód.
- 15 błędów bazy: 5 przy pewności >= 0,85 (m.in. telekomunikacyjne_399, tele_10, tele_11, kpk_244, kpc_882).
  Reranker jest pewny także złych trafień, więc jego wynik nie odróżnia "źle zrutowane" od "dobrze".

**Bramka niespełniona:** 304 (ALL: ranga 12 > 8) i tele_10 (pewność 0,90, bez eskalacji) nie naprawione.
**Wniosek:** wąskim gardłem jest routing, nie brak pętli -- dziedzina przypadku (G) daje 166/169, router 154. Sygnał
wyniku wyszukiwania nie wskazuje, kiedy poszerzać; pętla oparta na rerankerze nie jest warta złożoności. Co mogłoby
pomóc, a czego nie mierzono: sygnał z innego źródła niż reranker (np. Basal jako ocena trafności fragmentów albo
weryfikacja odpowiedzi względem kontekstu) i ponowne rozszerzenie zapytania po zmianie dziedziny (G z własnym
rozszerzeniem naprawia 304: ranga 7 zamiast 12).

## Co dalej

(Zaktualizowano 2026-10-18, po Kroku 79. Poniżej stan AKTUALNY; starsze
punkty niżej zostają jako historia i bywają już nieaktualne.)

**Stan po Kroku 79.**
- Produkcja: adaptery tylko `kadry_i_place` (hedge-iter50), `konsumenckie`
  (iter200), `budowlane` (k76 iter584), `cywilne` (k78 iter550),
  `mieszkaniowe` (k78 iter413); pozostałe dziedziny na samej bazie (po Kroku 77
  baza jest zwykle równie dobra).
- Destylacja: dłuższy trening (Krok 79) NIE pomaga -- 2 epoki maksimum, zysk w
  szumie; **nie skalować tej receptury na kolejne dziedziny**. Sensowna byłaby
  tylko zmiana receptury (rank/lr/dane, filtr przykładów z cytatami spoza
  kontekstu -- 7,7% w `cywilne`); wymaga pomysłu i ~6 $ na RunPodzie. Pod
  RunPoda: `scripts/runpod/` (przepis SSH i poda w pamięci
  `reference_runpod_ssh_temp_key`).
- Otwarte, tanie, bez GPU: (1) ~~`kk_45` lokalnie w LM Studio~~ (zrobione, Krok 82); (2) `kadry_i_place`: k78 iter580 vs hedge-iter50; (3) kolejne
  dziedziny na samej bazie (pilot: pobranie aktów, indeks, bateria, bez adaptera);
  Kroki 80-81 zrobiły `wykroczenia` (24 dziedziny, 2 akty), Krok 84 `postepowanie_cywilne`, Krok 85 `postepowanie_karne`, Krok 91 `prawa_pacjenta` (27 dziedzin).
- Krok 93: reranker a długie artykuły zmierzony -- bez zmiany (różnice w szumie); bateria liczy
  rangi na puli produkcyjnej (30); zostały 2 znane problemy (`konsumenckie_28`, `cywilne_tv`).
- Krok 94: bateria Qwena 29 OK + 2 znane (+1 fałszywy alarm poprawiony); spadki i darowizny w
  `podatkowe` (39 aktów). Dalej: PCC, CIT, podatki lokalne -- tym samym schematem (do
  `podatkowe`, A/B starych przypadków, osobna dziedzina tylko przy regresji); obserwować
  `deep_pit_21_pkt148` (#8 na progu).
- Krok 95: PCC jako własna dziedzina `pcc` (28 dziedzin, 40 aktów) -- w `podatkowe` regresowały
  stare przypadki. Dalej: CIT i podatki lokalne, od razu jako osobne dziedziny.
- Krok 96: CIT jako dziedzina `cit` (29 dziedzin, 41 aktów), bateria 118: 115 OK + 3 znane.
  Dalej: podatki i opłaty lokalne (osobna dziedzina).
- Krok 97: `podatki_lokalne` (30 dziedzin, 42 akty), bateria 129: 126 OK + 3 znane.

**Ustalona kolejność (z użytkownikiem, 2026-10-10).** Destylacja Qwen ->
Bielik może iść RÓWNOLEGLE z pozostałymi punktami: generowanie odpowiedzi
Qwena działa na RunPodzie (zdalne GPU), więc nie zajmuje lokalnej maszyny
(zasada: jeden proces MLX naraz dotyczy tylko lokalnego Maca). Lokalnie po
kolei: ~~(1) `cywilne` vs `konsumenckie`~~ (zrobione, Krok 71), ~~(2) 23.
dziedzina -- kredyt konsumencki~~ (zrobione, Krok 74), ~~(3) adnotacje w
bieżących tekstach~~ (zrobione, Krok 75).

- **Destylacja -- decyzje po pilocie (Krok 72).** ~~(a) podmiana
  adaptera `konsumenckie` na distill iter200~~ (zrobione, Krok 72). Pytanie
  ~~przełączenie produkcji na bazę~~ (zrobione, Krok 73, za zgodą
  użytkownika): `DEFAULT_ADAPTER_PATHS` = tylko `kadry_i_place`
  (hedge-iter50) i `konsumenckie` (distill iter200); stare adaptery w
  `LEGACY_ADAPTER_PATHS`. Kolejne dziedziny dostają adapter dopiero z
  destylacji i po porównaniu z bazą. Luki z Kroku 73
  dodane do baterii (NIEURUCHAMIANYCH po dodaniu -- rang nie zmierzono):
  `expand_query` +2 jako `known_issue` (`kc_118_przedawnienie_pozyczki`,
  `ksh_61_wystapienie_ze_spolki_jawnej` -- artykuł poza top-8 kontekstu);
  bateria Qwena +1 (`budowlane_30_sprzeciw_od_zgloszenia` -- art. 30 ust. 5
  był w kontekście, Bielik i tak podał 14 dni z art. 54; błąd odpowiedzi,
  nie retrievalu). Nienaprawione. **Pomiar (Krok 73):** art. 118 KC ranga
  20, art. 61 KSH ranga 13 (próg top-8). Przypadek budowlany NIE jest błędem
  czytania: `prompt.build_context` ucina artykuły > 6000 znaków
  (`MAX_ARTICLE_CHARS`, Krok 10), art. 30 Pr. bud. ma 12,4 tys., a ust. 5
  zaczyna się na znaku ~7575 -- Qwen uczciwie odmówił podania terminu
  ("treść skrócona"), Bielik zgadł 14 dni z art. 54. Oznaczony `known_issue`.
- ~~**NOWE, wysoki priorytet: ucinanie długich artykułów (Krok 73).**~~ (zrobione w Kroku 77 -- kontekst z ustępów trafionych przez wyszukiwanie, bez zmiany indeksów; otwarte: to samo w rerankerze psuje art. 86 VAT). 259 z
  12 382 bieżących artykułów (2%) ma > 6000 znaków i model widzi tylko ich
  początek -- m.in. art. 21 ustawy o PIT (zwolnienia, 118 tys. znaków; ulga
  mieszkaniowa pkt 131 daleko poza progiem), 32 artykuły Prawa
  energetycznego, 31 ustawy o VAT, art. 29/30 Prawa budowlanego. Kierunek:
  dzielić długie artykuły na fragmenty po ustępach przy indeksowaniu (jak
  załączniki w Kroku 70), żeby retrieval trafiał we właściwy ustęp, zamiast
  podnosić próg (art. 50 ustawy systemowej -- 51 tys. znaków -- zapchałby
  okno). Wymaga przebudowy indeksów i pełnych baterii.
  **Krok 78: destylacja 8 dziedzin -- adaptery w większości nie lepsze od
  bazy po Kroku 77** (sędzia Qwen; produkcja: budowlane k76, cywilne k78,
  mieszkaniowe k78). Zanim kolejne dziedziny: (1) czy dłuższy trening (val
  loss spadał do końca 1 epoki) albo wyższy lr/rank daje przewagę -- tani test
  na 2-3 dziedzinach (trening ~5-15 min/dziedzinę bez checkpointingu);
  (2) sędzia zgodny z ręczną oceną? (budowlane przeczytane -- tak);
  (3) `kk_45` lokalnie w LM Studio; (4) `kadry_i_place`: k78 iter580 vs
  hedge-iter50.
  (b) **Krok 76: `budowlane` + `cywilne` zrobione na RunPodzie** (cały
  pipeline + ewaluacja na podzie, ~4,4 $). Następne dziedziny tym samym
  `scripts/runpod/` -- jeden pod obsłuży kilka dziedzin naraz (koszt
  zdominowany przez odpowiedzi Qwena, ~40/min na RTX PRO 6000). Do
  zbadania: 12% odpowiedzi Qwena w `cywilne` cytuje spoza kontekstu;
  adapter `cywilne` uczy się asekuracji ("nie jestem w stanie
  rozstrzygnąć") tam, gdzie kontekst wystarcza.
  Skalowanie (konwerter + `train_lora_cuda.py` gotowe, Krok 73): więcej pytań na
  dziedzinę (mylenie sąsiednich przepisów nie zniknęło przy 215
  przykładach) i kolejne dziedziny; trening na CUDA (RunPod, PEFT ->
  konwersja do MLX), bo lokalnie ~77 s/iter przy 8-11 tys. tokenów.
  (c) Ewaluacja: 16 pytań to za mało do rozróżnienia checkpointów --
  przy skalowaniu ~40-50 pytań na dziedzinę i ocena ręczna/sędzią.
- **Destylacja Qwen -> Bielik na RunPodzie** (szczegóły w punkcie niżej).
  Licencja OK (Apache 2.0, Krok 67). Serwer MCP RunPoda dodany (zakres
  użytkownika) -- narzędzia widoczne dopiero w NOWEJ sesji Claude Code.
  Budżet na koncie ~14,26 $; karta 48 GB VRAM (A40 / RTX A6000 / L40S,
  ~0,4-0,5 $/h) mieści Qwen3.8-27B w 8 bitach -- sprawdzić aktualne ceny
  przez MCP przed uruchomieniem. vLLM zamiast LM Studio (szybciej niż
  lokalne ~200 s/odpowiedź). Adaptery CUDA same w sobie -- NIE teraz
  (brak wdrożenia na NVIDIA; Krok 68: adapter bywa gorszy od bazy).
- **Fałszywe spacje pypdf w polskich znakach (Krok 74)** -- "zobowi ązany",
  "b ędących"; najczęściej w ustawie o kredycie konsumenckim (0,55/1000 zn.),
  kodeks pracy 0,27, reszta ~300 wystąpień. Bezpieczna część poprawki:
  sklejanie przed ą/ę/ń (polskie słowo od nich się nie zaczyna); ś/ć/ź/ż/ó
  wymagają słownika. Przebudowa wszystkich indeksów + pełne baterie.
- **24. dziedzina (opcje):** ustawa o ochronie konkurencji i konsumentów,
  prawo spółdzielcze.
- **Resztki historii po Kroku 69 (niepilne):** okno Prawa lotniczego z 2006
  r. (~110 fragmentów przypisów), 12 zdublowanych etykiet w najstarszych
  oknach (PIT z obwieszczeń z 1993/2000 -- słaba ekstrakcja tych plików).
- **Mylenie sąsiednich przepisów -- nowy punkt danych (Krok 66, art. 309
  vs 304 w `telekomunikacyjne`).** Oba warianty Bielika (z adapterem i bez)
  stosują przepis o węższym zakresie, bo leksykalnie pasuje ("promocyjne
  urządzenie końcowe"), choć jego przesłanka (wypowiedzenie z art. 307/380)
  nie jest spełniona; Qwen nie. Kolejny argument za destylacją (niżej),
  nie za kolejną rundą danych fine-tune (Zasada 6). **Krok 68: jeszcze
  mocniejszy punkt danych** -- w `konsumenckie` adapter odpowiada "Nie"
  wbrew art. 43e ust. 1 pkt 3 będącemu w kontekście, baza i Qwen "Tak".
- **Adapter a pewność werdyktów (Krok 61).** Porównanie z/bez adaptera
  (24 pytania, 4 dziedziny) NIE potwierdziło mocno hipotezy o dopiskach z
  sąsiednimi przepisami (adapter ~5 błędów cytowania, baza ~3, różne
  mechanizmy), ale pokazało wyraźniej, że adapter wydaje pewne, binarne
  werdykty ("Tak/Nie -- zgodnie z art."), także błędne, gdy kontekst jest
  niepełny. Kandydat na zmianę przepisu danych fine-tune: przykłady, w
  których poprawna odpowiedź to "to zależy od..."/"dostarczone przepisy nie
  rozstrzygają". UWAGA: to blisko Zasady 6 (4 nieudane rundy) -- nie
  zaczynać bez nowego pomysłu na metodę; lepiej połączyć z destylacją
  (niżej), bo Qwen zastrzega się trafniej. `scripts/compare_adapter_vs_base.py`
  gotowy do ponownego użycia (np. do oceny adaptera po destylacji).
- **Pytanie "napojenie krów z rzeki" (art. 32 `wodne`) nadal ranga 21** po
  hincie z Kroku 61 -- gospodarstwo rolne zamiast domowego. Jeden punkt
  danych; jeśli wróci, rozważyć dopisanie "rolnego" do przykładu hintu.
- **Resztki w starych oknach historycznych (po Kroku 65, niepilne):** 2
  duplikaty etykiet bez bezpiecznej reguły (Prawo budowlane 2000 art. 54,
  Ordynacja 2018 art. 97a), 46 przypisów w historii (np. przypis
  przechodzący na następną stronę). Kadry: filtr "< >" wciąż regexowy
  (sprzed Kroku 47) -- przenieść, jeśli kiedyś pojawi się tam realny
  przypadek.
- **`vat_86_zakupy_prywatne` na granicy progu (#8/top-8) po Krokach 63-64** --
  obserwować; jeśli wypadnie, sprawdzić, czy to efekt nowej wersji
  historycznej ustawy o PIT w `podatkowe`.
- **Destylacja Qwen -> Bielik -- szczegóły (wybrana przez użytkownika
  2026-10-10, patrz wyżej).** Sekwencyjna (różne tokenizery wykluczają
  destylację na logitach): dużo pytań -> kontekst RAG -> odpowiedź Qwena ->
  automatyczny filtr (cytowany artykuł musi być w kontekście) + ręczna
  kontrola próbki -> LoRA Bielika. Celuje dokładnie w słabości, gdzie Qwen
  wygrywa (Zasada 6, mylenie sąsiednich przepisów) i jest INNĄ metodą/skalą
  niż odrzucone rundy. Start: 1 dziedzina, ~200 przykładów, porównanie z
  obecnym adapterem na obu bateriach. Zastrzeżenia: "identyczny tok
  myślenia" nierealny przy 11B; ~150-250 s/odpowiedź Qwena (~50-70 h na
  1000 przykładów); sprawdzić licencję Qwena co do trenowania na jego
  wyjściach (SPRAWDZONE w Kroku 67: Apache 2.0, dozwolone); przepis
  100 iter/iter25 nie pasuje do dużego zbioru.
- **Nowy, precyzyjnie zdiagnozowany wzorzec błędu cytowania z Kroku 56
  (`wodne_394_staw_deszczowka_zgloszenie`, known_issue) -- jeden
  potwierdzony przypadek, czeka na drugi.** Bielik przypisuje poprawną,
  faktycznie ugruntowaną treść (art. 394 BYŁ w kontekście RAG) numerowi
  SĄSIADUJĄCEGO artykułu o przeciwstawnym znaczeniu (art. 395) -- inny
  mechanizm niż Pattern 1 z Kroku 42/43 (tam artykuł w ogóle nie
  istniał). Qwen na tym samym kontekście cytuje poprawnie. Świadomie NIE
  naprawiane teraz (ten sam rygor co `ksh_210`/Rule 6 niżej) -- jeśli
  ten sam wzorzec (mylenie dwóch sąsiadujących, strukturalnie podobnych,
  znaczeniowo przeciwstawnych przepisów) pojawi się w kolejnej domenie,
  to sygnał, żeby zacząć szukać ogólnej naprawy.
  **Krok 58: drugi, pokrewny punkt danych** -- adapter `mieszkaniowe`
  przypisuje poprawną treść numerowi sąsiedniego/podobnego przepisu (art.
  3 ust. 2 zamiast ust. 3; termin z art. 19i przypisany art. 19d), ale
  tylko w pobocznych dopiskach i tylko z adapterem (baza czysto) -- więc
  nie identyczny mechanizm. Nadal za mało, by naprawiać; jeśli pojawi
  się trzeci przypadek, porównać systematycznie z/bez adaptera.
- **Rozszerzaj `data/expand_query_reference_cases.json` (Krok 27) i
  `data/qwen_answer_reference_cases.json` (Krok 38) przy każdej kolejnej
  dziedzinie i przy każdym nowo znalezionym przypadku gęstości
  tematycznej/luki w recall** -- oba zestawy są celowo małe i tracą
  czułość na regresje w nieobjętych zakątkach dziedzin, jeśli nikt ich
  nie uzupełnia. Krok 26/28 pokazały wprost, że ad-hoc ręczne
  spot-checki tego nie zastępują. Krok 51, 53, 55 i 56 domknęły ten dług
  za każdym razem szybko (Qwen-owa bateria: 6 -> 10 -> 17 -> 18 -> 20
  przypadków, pełne pokrycie 20 dziedzin po Kroku 62) -- pilnować przy 21. dziedzinie.
- **Wspólny `.venv` (`bielik-venv-shared`) miał zepsute wrappery
  konsolowe (Krok 54)** -- naprawione tej samej sesji (reinstalacja
  `mlx-lm`/`pytest`/`huggingface_hub` bez zmiany wersji), ale warto
  pamiętać na przyszłość: jeśli ponownie zmieni się nazwa/lokalizacja
  któregokolwiek z dwóch projektów na dysku, sprawdzić od razu, czy
  console-scripty w współdzielonym venv (shebang) nadal wskazują żywą
  ścieżkę -- ten sam mechanizm może się powtórzyć.
- **Rezydualna zawodność `ksh_210` na Qwenie po Kroku 53** -- root cause
  (pusty `expand_query()` przez zbyt mały `max_tokens`) naprawiony, ale
  ~2/3 skuteczności (nie 100%) przy niezerowej temperaturze na tej
  historycznie najtrudniejszej parze pytanie/przepis w projekcie.
  Świadomie nie strojone dalej -- czekać na więcej danych, zanim próbować
  czegokolwiek (ten sam rygor co pozostałe pozycje niżej).
- **Krótkie formy cytatów do innej ustawy bez daty** (Krok 50, np. samo
  "Kodeksu pracy" gdy pełna "ustawy z dnia..." padła wcześniej gdzie
  indziej w dokumencie) -- 1 potwierdzony, nienaprawiony przypadek
  (`prawo_ochrony_srodowiska`, art. 77⁵). Świadomie NIE budować tabeli
  aliasów nazw z odmianą przez przypadki dla jednego przypadku -- czekać
  na więcej potwierdzonych danych (ten sam rygor co
  `NON_SENTENCE_ENDING_ABBREVIATIONS` z Kroku 49).
- **Zasada 6 (cytat vs wniosek)** -- cztery próby, cztery różne metody
  (Krok 15/16: fine-tune; Krok 30: prompt/szablon), wszystkie
  niejednoznaczne, wszystkie odrzucone. Wiodąca hipoteza: to ograniczenie
  SAMEGO MODELU BAZOWEGO (11B, cały szczyt rodziny SpeakLeash/Bielik),
  nie metody naprawy -- niezależnie potwierdzona na Qwen3.8-27B (Krok
  31/32), który stosuje to rozróżnienie znacznie lepiej bez żadnego
  fine-tune'u. **NIE próbować piątej rundy tą samą metodologią** (mały
  fine-tune ALBO sam prompt na Bieliku) -- nie zadziała niezależnie od
  dopracowania. Niepilne -- brak dowodów na realną szkodę w praktyce
  (patrz Krok 43 dla najnowszego potwierdzenia tego samego wzorca w
  `karne`).
- **Qwen3.8-27B jako opcjonalna, cięższa alternatywa dla Bielika**
  (Krok 32-38) -- wynik konsekwentnie mocny (12 sond, 8 dziedzin, w tym
  `--as-of` i wieloturowa rozmowa), już wpięty do `chat_lmstudio.py`
  (`--reasoning-effort`/`--max-tokens`, Krok 33) i ma własną baterię
  regresyjną (Krok 38). Świadomie NIE zamiennik Bielika -- Bielik
  zostaje domyślną, lżejszą opcją (MLX, `chat.py`, 15 adapterów LoRA).
  8-bit pozostaje rekomendowanym wariantem (4-bit: porównywalna
  szybkość, ale jeden potwierdzony błąd cytatu, Krok 34; bf16 świadomie
  niesprawdzone -- zbyt ryzykowne przy historii freeze'ów systemu, patrz
  reguła: jeden proces MLX naraz). Dalszy krok, jeśli kontynuować:
  wielodziedzinowy adapter LoRA na Qwen (patrz niżej) -- ale to ma sens
  dopiero po tym punkcie, nie samo w sobie.
- **Wspólny multi-domenowy adapter LoRA** -- na Bieliku świadomie
  odłożone (Krok 32 pokazał, że 11B ma trudność z JEDNĄ skoncentrowaną
  regułą, więc rozcieńczenie sygnału na 15 dziedzin przy tej samej
  pojemności prawdopodobnie tylko powtórzy to ograniczenie bez nowej
  informacji). Warty rozważenia dopiero na modelu klasy Qwen3.8-27B --
  ciekawsze pytanie wtedy: czy model wystarczająco zdolny, by dobrze
  stosować Zasadę 6 bez fine-tune'u, również generalizuje styl cytowania
  między dziedzinami bez dedykowanego adaptera per dziedzina. Sensowne
  dopiero PO ewentualnej głębszej integracji Qwen3.8 (punkt wyżej).
- **Realne adaptery CUDA** -- `chat_cuda.py`/`DEFAULT_ADAPTER_PATHS`
  zawierają tylko placeholdery ścieżek (`adapters-cuda/...`), żadna
  dziedzina nie ma faktycznie wytrenowanego adaptera na CUDA, tylko MLX
  (Mac). Potrzebne dopiero, gdy pojawi się maszyna z GPU NVIDIA do
  trenowania/inferencji.

## Krok 105 (2026-11-08): router -- dodawanie `secondary` także przy wysokiej pewności NIE pomaga

Pomysł: mechanizm inny niż tekst etykiety (Krok 103) dla telecom->konsumenckie: `secondary` także przy pewności >= 0,85,
gdy jego prawdopodobieństwo >= 0,05 (Basal 4.5B, prawdopodobieństwa zapisane dla baterii 129 i 12 pytań telecom).
Proxy offline ("gold w zbiorze dziedzin"): 116 -> 121 / 129, telecom 10 -> 11 / 12; drugą dziedzinę dostaje 56 pytań zamiast 42.
**Prawdziwy retrieval (Bielik, bateria 129): 107 OK + 1 znany problem, 21 nowych regresji -- wobec 116 OK dla obecnego routera.**
Proxy znów mylące (jak w Kroku 103). Odrzucone, kod routera bez zmian. Wniosek: poprawa routingu telecom wymaga mechanizmu
poza wyborem dziedziny po prawdopodobieństwach (np. routing na poziomie ustawy) albo zaakceptowania znanego ograniczenia.

## Krok 106 (2026-11-08): Bielik v3.0 vs Qwen3.8-27B-FP8 na 76 pytaniach kadry (RunPod A100) -- Qwen wyraźnie lepszy

Pierwsze bezpośrednie porównanie jakości odpowiedzi (te same konteksty RAG, ekspansja Bielikiem v3.0). Kod: `scripts/runpod/compare_models.sh`,
`eval_distill_vllm.py --max-tokens/--reasoning-effort`, `make_bundle_compare.sh` z adapterem `-`. Pod: A100 80GB PCIe, 1,79 $/h, ok. 27 min pracy
(Bielik 09:37-09:40, Qwen 09:43-09:49, sędzia do 09:56); pod usunięty po pobraniu wyników. Wyniki: `data/distill/kadry_i_place/{eval_models,judge_models}.json`.
- Sędzia (Qwen, parami; "adapter" w pliku = Qwen, "base" = Bielik): **Qwen lepszy 65, remis 7, Bielik lepszy 0**; poważne błędy: Qwen 0, Bielik 40;
  4 pytania bez werdyktu (ucięte myślenie sędziego).
- Cytaty spoza kontekstu: Bielik 2/76, Qwen 7/76 (regex `cited_outside_context`).
- UWAGA: Qwen sądzi samego siebie -- wynik zawyżony. Przeczytano próbkę 4 "poważnych błędów" Bielika: wiarygodne (odwrócona perspektywa najemcy,
  art. 22 ust. 6d do ruchomości, urwana odpowiedź). Pełna weryfikacja merytoryczna nie wykonana (por. brak prawnika).
- Router (Krok 105): bez zmian.

**Krok 106b -- ręczna weryfikacja próbki (Claude, 2026-11-08, bez prawnika).** Przeczytano: 7 przypadków "cytat spoza kontekstu" Qwena + 5 z 40 "poważnych błędów" Bielika.
- Qwen "spoza kontekstu" 7/76 to głównie fałszywe alarmy regexu: 5 razy Qwen sam pisze, że artykułu NIE dostał (art. 152, 92, 188 KP, art. 27-29 PPK, przepisy przejściowe),
  1 raz art. 54 jest w kontekście (artefakt regexu), 1 raz realny wyciek wiedzy własnej (art. 430, 471 KC) -- realnie ok. 1/76, tyle co Bielik (2/76).
- 5/5 przeczytanych błędów Bielika potwierdzone: urlop bezpłatny (sprzeczność art. 174 § 2 vs 155² § 2), najem OD firmy (odwrócona perspektywa),
  art. 41b PIT (Bielik: "brak odpowiedzi", Qwen stosuje wprost), nieistniejący art. 23 ust. 1 pkt 151, zasiłek chorobowy (zły wniosek wbrew art. 14).
- Zastrzeżenia: Qwen pisze 2,4x dłużej (339 vs 142 słów) -- sędzia może premiować długość; w jednym przypadku Qwen uczciwie odmawia odpowiedzi (mniej użyteczne, ale bezpieczne);
  4 pytania bez werdyktu nieprzeczytane. Wniosek: przewaga Qwena na kadrach wiarygodna, nie jest artefaktem samooceny.

## Krok 107 (2026-11-08): Basal 4.5B + Qwen3.8-27B (LM Studio) jednocześnie -- bez problemów z pamięcią

Kontrolowany test (za zgodą użytkownika, wyjątek od zasady "Basal nigdy razem z Qwenem"): `basal-serve` (4.5B) + Qwen 27B w LM Studio (27,5 GiB, kontekst 8192;
użytkownik dodatkowo miał załadowane Bielik 11B i nomic-embed) + `chat_lmstudio.py --domain auto --prompt` dla 4 pytań, sekwencyjnie (11:15-11:42), monitor `mem_guard.sh`
(próbka co 5 s, bezpiecznik < 15% wolnej pamięci -> stop Basala i `lms unload --all`).
- **Wynik: bezpiecznik nie zadziałał, minimum wolnej pamięci 22%, swap stały 3,38 MB, brak zawieszeń.** Maszyna 64 GB: użyte ~54,6 GB (pamięć układowa 31,8, skompresowana 12,5).
- Router: kadry (1,00), telekomunikacyjne (0,82, druga: konsumenckie), podatki_lokalne (0,91), karne (0,95) -- wszystkie poprawne.
- Pułapka: domyślne `--max-tokens 500` bez `--reasoning-effort` daje PUSTE odpowiedzi Qwena (myślenie zjada budżet; Krok 32) -- pierwsze podejście pytań 2-4 puste.
  Z `--max-tokens 3000 --reasoning-effort low` odpowiedzi poprawne i oparte na artykułach (telecom art. 306/307/380, opłata reklamowa art. 17a ust. 5 pkt 2), ok. 4-6 min/pytanie.
- Znalezisko poboczne: pytanie o oszustwo (karne) -- retrieval NIE dostarczył art. 286 KK (Qwen uczciwie odmówił, podał art. 287); `kk_286` jest w baterii -- do sprawdzenia
  osobno (czy to ekspansja zapytania Qwena, czy wpływ `--domain auto`). Wniosek: Basal obok Qwena działa; warunek "razem nie" można zdjąć przy monitorowaniu pamięci i bez dodatkowych modeli.

## Krok 108 (2026-11-08): ekspansja zapytań -- Qwen (LM Studio) vs Bielik na baterii 129; przypadek art. 286 KK

Powód: w Kroku 107 pytanie o oszustwo (`--domain auto`, Qwen) nie dostało art. 286. Diagnostyka (`karne`, 3 próby na model): ekspansja Bielika zawsze zawiera "art. 286 Kodeksu karnego"
(ranga 1, 3/3), ekspansja Qwena (low) nie podaje numeru: ranga 3, 6 i BRAK. Bateria 129 w dziedzinie złotej, ekspansja przez LM Studio (`scripts/eval_expansion_lmstudio.py`, fazy `expand A|BC`, `rank`):
- A Bielik: **124/129** OK (0 pustych) | B Qwen low (800 tok.): **122/129** (7 pustych) | C Qwen low + dopisek "wymień numery przepisów": 116/129 (43 puste -- dłuższe myślenie ucina ekspansję) -- **C odrzucony**.
- Różnica Bielik-Qwen to 2 przypadki: art. 286 to głównie szum losowy, nie systematyczna wada. Z 7 pustych ekspansji Qwena realną stratę dała 1 (kc_415: ranga 11 zamiast 1).
- Zmiana: `expand_query` w `chat_lmstudio.py` ponawia JEDNORAZOWO z `max_tokens=2000`, gdy wynik jest pusty (tylko z `reasoning_effort`); 3 nowe testy.
- Uwaga: pierwszy skrypt (indeksy wszystkich dziedzin naraz + Qwen + Bielik w LM Studio) wywołał swap 15 GB i zadziałał bezpiecznik <15%; poprawka: jedna dziedzina w pamięci, modele LM Studio ładowane fazami.

## Krok 109 (2026-11-08): Qwen3.8-27B vs Bielik v3.0 na 7 innych dziedzinach (RunPod A100) -- przewaga Qwena nie jest specyficzna dla kadr

Rozszerzenie Kroku 106. Użyto istniejących kontekstów RAG i odpowiedzi Bielika `base` z `eval_vllm.json` (pytania wstrzymane z destylacji), na podzie tylko Qwen (odpowiedzi low/3000 tok.) + sędzia Qwen.
Kod: `scripts/runpod/compare_qwen_domains.sh`, `make_bundle_qwen_domains.sh`. Pod: A100 80GB PCIe 1,79 $/h, ok. 42 min (ok. 1,3 $), usunięty po pobraniu wyników.
Wyniki w `data/distill/<d>/{eval_vllm,judge_qwen}.json` ("adapter" w pliku = Qwen, "base" = Bielik):

| dziedzina | Qwen lepszy | remis | Bielik lepszy | poważne błędy Qwen / Bielik |
|---|---|---|---|---|
| budowlane | 17 | 2 | 1 | 1 / 8 |
| cywilne | 24 | 1 | 0 | 0 / 13 |
| mieszkaniowe | 19 | 2 | 0 | 0 / 10 |
| podatkowe | 18 | 2 | 0 | 0 / 9 |
| rodzinne | 15 | 3 | 0 | 0 / 10 |
| ruch_drogowy | 14 | 3 | 0 | 0 / 8 |
| własność przemysłowa | 17 | 0 | 1 | 0 / 9 |
| **razem (139 werdyktów z 149)** | **124** | **13** | **2** | **1 / 67** |

- Razem z kadrami (Krok 106: 65/7/0): Qwen wygrywa w ok. 88% par, Bielik w ok. 1%; 10 pytań bez werdyktu (ucięte myślenie sędziego).
- Dwa zwycięstwa Bielika: Qwen zbyt zachowawczy (zbędne zastrzeżenie interpretacyjne: licencja dla uczelni, art. 69 PWP; książka obiektu budowlanego, bliźniak) -- nie błąd merytoryczny.
  Jedyny poważny błąd Qwena (rozbiórka szopy, art. 31 ust. 1a) dotyczył pytania, na którym oba modele się pomyliły.
- Cytaty spoza kontekstu (regex) Qwen: 11/149 (z czego część to jawne "tego artykułu nie dostałem", jak w Kroku 106b).
- Zastrzeżenia bez zmian: sędzia = Qwen (samoocena, premia za długość), brak weryfikacji prawnika. Porównanie dotyczy Bielika BEZ adaptera (base); produkcyjne adaptery (cywilne, mieszkaniowe, budowlane) nie były porównywane z Qwenem.

## Krok 110 (2026-11-08): `chat_lmstudio.py` -- sensowne domyślne dla Qwena (3000 tokenów, reasoning_effort=low)

Powód: pułapka z Kroku 107 -- domyślne `--max-tokens 500` bez `--reasoning-effort` daje puste odpowiedzi Qwena. `resolve_generation_defaults(model_id, ...)`: dla modeli "qwen" 3000 + `low`,
dla pozostałych (Bielik) bez zmian (500, brak). Jawne argumenty zawsze wygrywają. Test jednostkowy; 156 testów przechodzi.

## Krok 111 (2026-11-08): trzydziesta pierwsza dziedzina -- `ubezpieczenia_obowiazkowe`

**Akt (ELI zweryfikowany).** Dz.U. 2003 poz. 1152, "Ustawa z dnia 22 maja 2003 r. o ubezpieczeniach obowiązkowych, Ubezpieczeniowym Funduszu Gwarancyjnym i Polskim Biurze Ubezpieczycieli
Komunikacyjnych" (tekst jednolity), od razu jako osobna dziedzina. 191 artykułów, historia 1571 wpisów, indeks 3144 fragmenty. Kontrola danych: 0 zbłąkanych `<>[]`.
Pilot prowadzony autonomicznie (zgoda użytkownika: "jak nic z tego nie będzie to kolejne dziedziny z automatu"). `acts_config.json` dopisany tekstowo (bez przeformatowania całego pliku).

**Retrieval.** +10 przypadków `ubez_*` (sprawdzone po treści artykułów 4, 14, 16, 19, 28, 31, 43, 47, 88, 98): 10/10 OK, 8 na #1. Bateria 139 (z dziedziny złotej): **136 OK + 3 znane, 0 nowych regresji**.
**Router (Basal 4.5B, 31 dziedzin):** 9/10 nowych pytań -> `ubezpieczenia_obowiazkowe` jako primary (`ubez_16` -> ruch_drogowy + ubezpieczenia_obowiazkowe jako secondary); na starych 129 pytaniach zmieniły się
tylko 3 decyzje (na granicy pewności). Retrieval przez router na 139: **126 OK + 1 znany; stare 129: 116 OK (bez zmiany względem Kroków 100-103), nowe 10/10 OK**.
pytest 156/156. Nie wykonano baterii Qwena ani testu końcowego `chat.py` dla tej dziedziny.

## Krok 112 (2026-11-10): Qwen3.8-27B vs PRODUKCYJNE adaptery (budowlane k76, cywilne k78-iter550, mieszkaniowe k78-iter413) -- Qwen wygrywa także tu

Luka z Kroków 106/109: tam Qwen był porównywany z Bielikiem BEZ adaptera. Teraz sędzia (Qwen) ocenia pary "adapter produkcyjny vs odpowiedź Qwena" na tych samych kontekstach RAG
(`data/distill/<d>/eval_vs_adapter.json`; wyniki `judge_vs_adapter.json`, "adapter" w pliku = Qwen, "base" = adapter produkcyjny). Kod: `scripts/runpod/judge_qwen_vs_adapter.sh`.
Pod: RTX PRO 6000 2,49 $/h (A100 PCIe/SXM niedostępne), ok. 40 min wliczając restart i wolny `pip` (ok. 1,7 $), usunięty po pobraniu wyników.

| dziedzina (adapter) | Qwen lepszy | remis | adapter lepszy | poważne błędy Qwen / adapter |
|---|---|---|---|---|
| budowlane (k76) | 15 | 4 | 0 | 0 / 7 |
| cywilne (k78-iter550) | 25 | 2 | 0 | 0 / 15 |
| mieszkaniowe (k78-iter413) | 12 | 6 | 0 | 0 / 7 |
| **razem (64 werdykty z 70)** | **52** | **12** | **0** | **0 / 29** |

- Adaptery produkcyjne nie dają przewagi nad Qwenem w żadnej z 3 dziedzin. Konsumenckie nie objęte (brak odpowiedzi Qwena w tym formacie).
- Zastrzeżenia: (1) sędzia = Qwen; (2) odpowiedzi adapterów mają limit 500 tokenów (jak chat.py) i część jest urwana -- sędzia karze to jako błąd, więc porównanie premiuje Qwena (3000 tok.);
  (3) Qwen jest ok. 2,4x dłuższy; (4) brak weryfikacji prawnika. Wniosek: adaptery to opcja dla lekkiego trybu (Bielik/MLX), nie dla jakości odpowiedzi.
- Lekcje RunPod: po `restart` poda zmienia się port SSH (trzeba odczytać z `get-pod`); `pip install` z niektórych hostów bywa bardzo wolny (120 kB/s) -- sędzia potrzebuje tylko
  pustego stubu `sentence_transformers` (import przez `distill_qwen` -> `build_rag_index`); scratchpad bywa odtwarzany między sesjami (klucz `pod_key` zniknął).

## Krok 113 (2026-11-10): Qwen3.8-27B (LM Studio) rekomendowanym trybem odpowiedzi -- decyzja użytkownika

Na podstawie Kroków 106, 109 i 112 użytkownik zdecydował ("tak"), że Qwen jest domyślnym, rekomendowanym modelem do odpowiedzi; Bielik (`chat.py`, MLX) zostaje lekkim trybem.
Zmiana tylko dokumentacyjna (README: opis stanu, ostrzeżenia, sekcja "Użycie" z `chat_lmstudio.py --domain auto`); kod bez zmian (domyślne dla Qwena już w Kroku 110).
Zastrzeżenia pomiarów (sędzia = Qwen, długość odpowiedzi, brak prawnika) opisane w Krokach 106-112.

## Krok 114b (2026-11-11): Bielik 11B v3.1 Instruct 8-bit (MLX) vs v3.0 4-bit i vs Qwen -- kadry, 76 pytań (sędzia Qwen na RunPod)

Odpowiedzi v3.1 wygenerowane lokalnie (`scripts/eval_model_mlx.py`, `data/distill/kadry_i_place/eval_v31.json`, limit 500 tokenów). Sędzia: Qwen3.8-27B-FP8 (vLLM) na podzie A100 SXM 1,79 $/h,
ok. 30 min (ok. 0,9 $), `scripts/runpod/judge_v31.sh`; pod usunięty po pobraniu wyników. W plikach par "base" = v3.1, wariant = przeciwnik. Wyniki: `judge_v31_vs_v30.json`, `judge_v31_vs_qwen.json`.
- **v3.1 vs v3.0** (69 werdyktów z 76): v3.1 lepszy 27, remis 21, v3.0 lepszy 21; poważne błędy v3.1 17, v3.0 25. Lekka przewaga v3.1, niewielka przy 21 remisach.
- **v3.1 vs Qwen**: Qwen lepszy 65, remis 8, v3.1 lepszy 1; poważne błędy Qwen 0, v3.1 34 (74 werdykty). Dla porównania v3.0: 65/7/0 (Krok 106) -- v3.1 nie zmienia przewagi Qwena.
- v3.1: 0 cytatów spoza kontekstu (v3.0: 2), ale 48/76 odpowiedzi ucięte limitem 500 tokenów (v3.0: 37).
- Zastrzeżenia: sędzia = Qwen, premia za długość, ucięte odpowiedzi karane jako błąd, brak prawnika. Domyślny model lekkiego trybu bez zmian (v3.0 4-bit).
- Sprawdzono i odrzucono: Bielik-11B-v3.0-DFlash (drafter spekulatywny, tylko SGLang/vLLM, bez MLX/LM Studio) oraz Bielik-11B-v3-Base (model bazowy do własnego fine-tuningu, nie do odpowiedzi).

## Krok 115 (2026-11-11): małe MoE (Gemma-4-26B-A4B, Qwen3.6-35B-A3B) vs Qwen3.8-27B oraz Bielik v3.1 4-bit vs 8-bit -- kadry, 76 pytań (RunPod)

Odpowiedzi MoE i sędzia na jednym podzie (RTX PRO 6000, 2,49 $/h, ok. 35 min, ok. 1,5 $; A100/H100 chwilowo bez dostępności; wolumen 150 GB na cache HF). Kod: `scripts/runpod/moe_test.sh`,
`eval_distill_vllm.py --template-kwargs '{"enable_thinking": false}'`. Konteksty RAG te same co w Kroku 106 (`eval_models.json`), limit 3000 tokenów, myślenie wyłączone. Sędzia: Qwen3.8-27B-FP8 (vLLM).
Wyniki: `eval_moe_*.json`, `judge_moe_*.json`, `judge_v31_4bit_vs_8bit.json`. Uwaga: w plikach par "adapter" = drugi wariant (Qwen 27B / Bielik 8-bit), "base" = testowany (MoE / 4-bit).

| testowany model | Qwen 27B lepszy | remis | testowany lepszy | poważne błędy: Qwen 27B / testowany | śr. słów | cytaty spoza kontekstu |
|---|---|---|---|---|---|---|
| Gemma-4-26B-A4B (4B akt.) | 50 | 17 | 0 | 0 / 12 | 243 | 0 |
| Qwen3.6-35B-A3B-FP8 (3B akt.) | 47 | 17 | 6 | 2 / 27 | 404 | 2 |
| (Bielik v3.0, Krok 106) | 65 | 7 | 0 | 0 / 40 | 142 | 2 |

- **Bielik v3.1 4-bit vs 8-bit** (68 werdyktów z 76; "adapter" = 8-bit): 8-bit lepszy 25, remis 27, 4-bit lepszy 16; poważne błędy 8-bit 21, 4-bit 23. Różnica mała; 4-bit ma 2 cytaty spoza kontekstu (8-bit: 0), podobną długość (159 vs 154 słów). Kwantyzacja nie jest tu głównym czynnikiem jakości.
- **Wniosek:** żaden MoE nie dorównuje Qwenowi 27B (nigdy nie wygrywa częściej niż 6/76), ale Gemma-4-26B-A4B jest wyraźnie lepsza od Bielika (12 vs 40 poważnych błędów) i zajmuje ok. 15 GB w 4-bit -- kandydat na szybszy tryb lokalny.
  Qwen3.6 pisze najdłużej (404 słowa; sędzia premiuje długość) i ma najwięcej błędów. Domyślny model bez zmian (Qwen3.8-27B).
- Zastrzeżenia: sędzia = Qwen 27B (samoocena Qwena 3.6 jako pokrewnego modelu, premia za długość), brak prawnika, czas z RunPod (współbieżność 48) nie przekłada się na prędkość Maca -- NIE zmierzono jeszcze prędkości lokalnej.
- Nie zrobiono: bateria `check_qwen_answer_regressions.py` dla MoE, test w LM Studio na Macu (czas na odpowiedź).

## Krok 116 (2026-11-11): Gemma 4 lokalnie w LM Studio (prędkość 26B-A4B i 31B, 4/8-bit) + jakość 31B (RunPod)

**Prędkość lokalna** (`scripts/eval_gemma_lmstudio_speed.py`; M-series 64 GB, LM Studio, modele lokalne; 10 pytań kadr z `eval_models.json`, kontekst 16384, max 3000 tok., myślenie wyłączone
`chat_template_kwargs.enable_thinking=false`; wyniki `speed_<id>.json`). Zasady po Krokach 34/56: `lms unload --all`, ładowanie po kluczu wariantu (`...@4bit`/`@8bit`) z unikalnym `--identifier`,
weryfikacja przez `lms ps` (nazwa + rozmiar) -- bez pomyłek 4-bit/8-bit. Uwaga: `lms get` utknął przy 87% (1 MB/s), dokończone przez `hf download lmstudio-community/...` do katalogu LM Studio.

| model | rozmiar | śr. czas / odpowiedź | max | tok/s |
|---|---|---|---|---|
| Gemma-4-26B-A4B 4-bit | 15,6 GB | 16,9 s | 24 s | 30,0 |
| Gemma-4-26B-A4B 8-bit | 28,0 GB | 19,5 s | 25 s | 25,2 |
| Gemma-4-31B (gęsty) 4-bit | 18,4 GB | 95,7 s | 147 s | 4,8 |
| Gemma-4-31B (gęsty) 8-bit | 33,8 GB | 128,0 s | 164 s | 3,7 |

W każdym przebiegu 0 cytatów spoza kontekstu i 0 uciętych odpowiedzi (`finish=stop`). Dla porównania Qwen3.8-27B z myśleniem: 4-6 min/odpowiedź (Krok 107). Próbka 10 pytań -- różnice rzędu kilku sekund to szum.

**Jakość 31B** (RunPod A100 SXM 1,79 $/h, ok. 34 min, ok. 1,0 $; `scripts/runpod/gemma31_test.sh`; Gemma-4-31B-it w BF16 w vLLM, sędzia Qwen3.8-27B-FP8; 76 pytań kadr, 67 werdyktów;
w pliku "adapter" = Qwen 27B, "base" = Gemma): **Qwen 27B lepszy 48, remis 17, Gemma 31B lepsza 2**; poważne błędy Qwen 0 / Gemma 31B **5**; 206 słów/odpowiedź; 0 cytatów spoza kontekstu.
Dla porównania (Krok 115): Gemma 26B-A4B 50/17/0, błędy 12; Bielik v3.0 65/7/0, błędy 40.

**Wnioski:** gęsty 31B ma wyraźnie mniej poważnych błędów niż A4B (5 vs 12), ale wciąż przegrywa z Qwenem 27B w ok. 70% par i jest 5-7 razy wolniejszy lokalnie (96-128 s vs 17-20 s).
Dla lekkiego trybu lokalnego najlepszy kompromis to Gemma-4-26B-A4B (4-bit lub 8-bit -- 8-bit tylko ok. 15% wolniejszy); 31B opłaca się tylko przy priorytecie jakości nad szybkością.
Domyślny model (Qwen3.8-27B) bez zmian. Zastrzeżenia: sędzia = Qwen, premia za długość (31B pisze krócej niż A4B), brak prawnika, próbka prędkości 10 pytań; nie uruchomiono baterii regresji dla Gemmy.

## Krok 117 (2026-11-11): Gemma-4-26B-A4B jako drugi lekki tryb + ostrzeżenie o Bieliku w README (decyzja użytkownika: "ścieżka 2")

- **Kod:** `--template-kwargs` (JSON -> `chat_template_kwargs`, np. `{"enable_thinking": false}` dla Gemma 4) w `chat_lmstudio.py` (`expand_query`, `answer`, CLI) i `check_qwen_answer_regressions.py`;
  domyślnie brak -- zachowanie dla Qwena/Bielika bez zmian. 242 testy jednostkowe przechodzą.
- **Bateria odpowiedzi (34 przypadki) dla `gemma-4-26b-a4b-it-mlx@8bit`** (LM Studio, identyfikator `gemma26-8bit`, kontekst 24576, myślenie wyłączone): **33 OK + 1 znany problem
  (`ochrona_danych_101_kara_wyciek_odpowiedz`, ten sam co u Qwena -- zaakceptowany kompromis, Kroki 25/28/29), 0 nowych regresji**. Wynik identyczny jak Qwen (Krok 95-98). Zastrzeżenie: substring-checki
  to tylko "kanarek", nie ocena merytoryczna. Komenda z README sprawdzona na żywo (art. 154 § 1 pkt 2 KP, 22 s).
- **README:** dwa lekkie tryby -- Gemma-4-26B-A4B (LM Studio) i Bielik (MLX) z wyraźnym ostrzeżeniem o słabszej jakości Bielika (40 poważnych błędów na 76 pytań, v3.1 8-bit: 34; urwane odpowiedzi;
  adaptery v3.0 nie pomagają); komenda dla Gemmy z ładowaniem po kluczu wariantu i `--identifier` (lekcja Kroków 34/56: warianty 4-bit/8-bit mają to samo id).
- **Plan użytkownika (pamięć `project_bielik31_target`):** docelowo Bielik v3.1 8-bit, z adapterami/dotrenowaniem jeśli potrzeba. NIE zrobione: przełączenie domyślnego modelu `chat.py` na v3.1 (wymaga
  `--no-adapter`/wyłączenia `DEFAULT_ADAPTER_PATHS` -- adaptery są trenowane na v3.0 -- oraz przeliczenia baterii 129 i podniesienia limitu 500 tokenów) ani nowych adapterów dla v3.1.

## Krok 118 (2026-11-11): `chat.py` -- odpowiedzi Bielika v3.1 8-bit (limit 1000 tokenów), ekspansja zapytań zostaje na v3.0 4-bit; osłona adapterów

**Zmiana w `scripts/chat.py`:** domyślny model odpowiedzi `ANSWER_MODEL_PATH` = `models/Bielik-11B-v3.1-Instruct-mlx-8bit` (`--model`); `MODEL_PATH` (v3.0 4-bit) NIE zmieniony -- importują go inne skrypty
(`compare_adapter_vs_base`, `eval_distill`, `distill_qwen`, `eval_agentic_fallback`, bateria `check_expand_query_regressions.py`) jako bazę adapterów/ekspansji.
`--expand-model` (domyślnie v3.0 4-bit, `same` = ten sam co `--model`): ekspansję robi osobny model, ładowany w tym samym procesie (szczyt pamięci ok. 20 GB, żywy test: poprawna odpowiedź kadry, ok. 1 min z ładowaniem).
`resolve_adapter_path`: domyślne adaptery (trenowane na v3.0) ładują się tylko z modelem `Bielik-11B-v3.0*`; jawny `--adapter-path` zawsze wygrywa. `--max-tokens` (domyślnie 1000, było stałe 500). 250 testów przechodzi (8 nowych: `tests/test_chat_adapter_guard.py`).

**Dlaczego ekspansja nie na v3.1:** bateria `check_expand_query_regressions.py` (139 przypadków) z v3.1 8-bit: **129 OK + 1 znany + 9 NOWYCH regresji** (kc_415 ranga 12, vat_86 i ochrona_danych_101 i mieszkaniowe_19 poza pulą 60, kc_118, deep_pit_21, deep_vat_86, pit_10, cit_11c)
wobec 136 OK + 3 znane (v3.0 4-bit, Krok 111). Retrieval pozostaje więc bez zmian (ekspansja jak dotąd), a odpowiedzi generuje v3.1.

**Czy limit 500 tokenów tłumaczy lukę Bielika?** (próbka 30 pytań kadr, RunPod A100 1,79 $/h, ok. 17 min, ok. 0,6 $, `scripts/runpod/judge_pairs.sh`; sędzia Qwen3.8-27B; pliki `eval_v31_1000tok.json`, `judge_v31_1000_vs_*.json`):
- Długość odpowiedzi v3.1 8-bit: limit 500: mediana 162, maks. 182 słów (ucięte); limit 1000: mediana 171, maks. 334, średnia 188 vs 160.
- v3.1@1000 vs Qwen 27B: Qwen lepszy 25, remis 3, Bielik 0; poważne błędy Bielika 14/28. Te same 30 pytań przy 500: Qwen 28, remis 2, Bielik 0; błędy 15/30.
- v3.1@1000 vs v3.1@500 (bezpośrednio): 1000 lepszy 10, remis 17, 500 lepszy 2; poważne błędy 11 vs 12.
- **Wniosek:** dłuższy limit lekko poprawia odbiór (10:2), ale NIE zamyka luki wobec Qwena -- połowa odpowiedzi Bielika nadal ma poważny błąd wg sędziego. Luka jest merytoryczna, nie tylko efekt ucinania.
- Zastrzeżenia: próbka 30 pytań, jedna dziedzina, sędzia = Qwen (premia za długość), brak prawnika.
- **Adaptery/finetuning dla v3.1 -- NIE rozpoczęte.** Dotychczasowy przepis (destylacja Qwena do LoRA) dawał tylko skromny zysk (Krok 112: adaptery 29 poważnych błędów vs Qwen 0; Krok 79: dłuższy trening bez zysku) i wymaga budżetu RunPod (saldo ok. 4,4 $).

## Krok 119 (2026-11-13): Bielik-One 11B v3.1 jako router dziedzin -- zmierzony, NIE wdrażamy (zostaje Basal)

**Co to jest:** Bielik-One (SpeakLeash, 2026-11-12, Apache 2.0, bez bramki HF) -- 4 modele "decyzyjne" (11B v3.1, 11B v3.0, 7B v3.0 -- na Bieliku; 4B -- na Qwen3.5-4B), każdy scalony
albo jako adapter LoRA + głowice. Zwraca typowaną decyzję (`choice`/`noul`/`score`/`labels`) z rozkładem prawdopodobieństw (skalibrowanym tylko dla bf16). API `/v1/systemone`
jest zgodne z naszym `router.py` (to samo `choice`), dwie różnice: w zapytaniu trzeba `"model":"decision"`, a proxy RunPod zwraca 403 dla domyślnego User-Agenta urllib.
**Wymaga GPU z CUDA** (vLLM 0.32.0 + pakiet `bielik-one`); na Macu nie ruszy (są cudze konwersje MLX `agentGreg/Bielik-One-*-MLX`, NIEsprawdzone).

**Pomiar:** pod RunPod RTX 6000 Ada (secure, 1,09 $/h, ok. 14 min, ok. 0,25 $), obraz `vllm/vllm-openai:v0.32.0`, `pip install "bielik-one[vllm] @ git+https://github.com/speakleash/bielik-one"`,
`hf download speakleash/Bielik-One-11B-v3.1-r1`, `bielik-one backend` + `frontend --host 0.0.0.0` (port 8026, dostęp przez proxy RunPod). Klient: nasz `router.py` lokalnie
(`route_battery`, `single_threshold=2`, własny User-Agent, `model: decision`). Potem retrieval lokalnie (`check_expand_query_regressions.py --routing-file`, jeden proces MLX).

**Trafność samego routingu (top-1 / top-2):** bateria 139: 119 (85,6%) / 137 (98,6%); zbiór kontrolny 36: 33 (91,7%) / 36; telekomunikacja 12: 10 / 12. Główne pomyłki: `cywilne`->`konsumenckie` (6),
`cit`->`podatkowe` (5), `wykroczenia`->`karne`/`ruch_drogowy`.

**Retrieval (OK = ranga w progu), bateria, 129 wspólnych przypadków (10 nowych `ubez_*` Basal nie widział):**

| wariant | Basal | Bielik-One 11B |
|---|---|---|
| tylko `primary` | 104 | 114 |
| `secondary` zawsze | 113 | 112 |
| najlepszy próg `secondary` | 116 (0,85; 42 pytania) | 120 (0,6; 47 pytań) |
| dziedzina ręcznie | 126 | 126 |

Zbiór kontrolny 36 (ręcznie 36/36): `primary` 31 vs 33; próg 0,6 -- Basal 33 vs One 36 (7 pytań z `secondary`); Basal przy 0,7-0,85 też 36. Na całej baterii 139 z `ubez_*`: One `primary` 124, próg 0,6: 130.
Progi dla One: najlepiej 0,4-0,6 (przy 0,85 `secondary` dostaje 118 z 129 pytań, bo pewność `choice` jest niska przy 32 dziedzinach). Zastrzeżenia: próg dobrany na tej samej baterii, zbiór kontrolny mały (36),
retrieval telekomunikacji nie do zmierzenia (przypadki bez `act_short`/`article`).

**Decyzja (użytkownik, 2026-11-13): zostajemy przy Basalu** -- zysk jest mały (+4 na baterii, +0..3 na zbiorze kontrolnym), a Basal działa lokalnie na Macu. Bielik-One zapisany jako
**alternatywa dla środowiska CUDA** (np. stały serwer GPU) albo gdy pojawi się sprawdzona konwersja MLX; wtedy domyślny próg `secondary` dla niego ok. 0,6 (nie 0,85). Wariant 4B (Qwen3.5) NIE mierzony.
Zmiany w `router.py`/repo: brak (przypomnienie: `model: decision` trzeba dodać do body przy wdrożeniu).

## Krok 120 (2026-11-13): `prawo_bankowe` -- 32. dziedzina (ustawa Prawo bankowe, Dz.U. 1997 poz. 939)

Dokończony pilot zaczęty 2026-11-10 (`acts_config.json`, pobranie 497 artykułów, historia, indeks 10080 chunków -- patrz wyżej). Dopisane: 12 przypadków do `data/expand_query_reference_cases.json`
(`pb_*`: art. 50, 51, 52, 54, 55, 69, 70, 75, 75a, 75c, 104, 111; każdy artykuł sprawdzony po treści w indeksie), opis dziedziny w `scripts/prompt.py`, README (32 dziedziny, 44 akty, wiersz w tabeli; adapter: baza).
BFG i ustawa o BFG NIE wchodzą (osobny akt, w opisie dziedziny tylko Prawo bankowe).

**Pomiary (bateria 151 przypadków = 139 + 12 nowych):**
- Testy jednostkowe: 250 przechodzi.
- Bateria `check_expand_query_regressions.py` z dziedzinami ręcznie: **148 OK + 3 znane problemy, 0 nowych regresji**; wszystkie 12 `pb_*` w randze 1-2 (`pb_75a` ranga 2).
- Router Basal (4.5B 8-bit, opisy = nazwy ustaw): top-1 120/151 (79,5%), top-2 144/151 (95,4%); wszystkie 12 `pb_*` routowane poprawnie, ŻADEN inny przypadek nie trafił do `prawo_bankowe`
  (nowa dziedzina nie przejęła cudzych pytań). Zbiór kontrolny 36: 30 top-1, 36 top-2 (bez zmian względem 100%/83% z Kroku 101).
- Retrieval przez router (próg `secondary` 0,85, składany offline z `secondary` nulowanym przy pewności >= 0,85): **137/151 OK**; `pb_*` 12/12, `ubez_*` 10/10; na starych 129 przypadkach 115 (Krok 100b: 116) --
  różnica 1 mieści się w szumie (inny przebieg niż złożenie z dwóch), a niepowodzenia to te same znane pomyłki routera (cywilne->konsumenckie, pcc/cit->podatkowe, kpk->ruch_drogowy), nie `prawo_bankowe`.

Zastrzeżenie: treść dziedziny bankowej NIE została zweryfikowana przez prawnika (disclaimer + RAG z cytowaniem artykułów jako zabezpieczenie); przypadki napisane z treści artykułów, nie z pytań prawdziwych użytkowników.
Adaptery: brak (pytania/odpowiedzi do destylacji nie istnieją; decyzja 2026-11-13: najpierw eksperyment adapterów Bielik v3.1 dla kadr).

## Krok 121 (2026-11-13): adapter LoRA dla Bielika v3.1 (kadry_i_place) -- BEZ zysku, nie skalujemy

**Pytanie:** czy LoRA z destylacji Qwena (przepis z Kroków 76/78) domyka lukę Bielika v3.1 wobec Qwena (Krok 118: Qwen 25 / remis 3 / Bielik 0, poważne błędy Bielika 14/28 na 30 pytaniach kadr).
**Odpowiedź: nie.** Adapter nie poprawia merytoryki względem samej bazy v3.1, a wobec Qwena wypada tak samo albo gorzej.

**Pipeline (jeden pod RunPod, RTX PRO 6000 96 GB secure 2,69 $/h, ok. 2 h 20 min, ok. 6,3 $; skrypty `scripts/runpod/{make_bundle_k121,setup_k121,kadry_v31}.sh`):**
konteksty RAG dla 1500 pytań (ekspansja Bielik v3.0, jak produkcyjny chat.py) -> 750 NOWYCH odpowiedzi Qwena3.8-27B (łącznie 1450, `answers.jsonl` rozszerzony z 700) -> zbiór -> trening LoRA na `speakleash/Bielik-11B-v3.1-Instruct` BF16
(rank 8, alpha 160, lr 1e-5, batch 1, 1122 kroków = 1,2 epoki, max-length 12288, 2,3 s/krok, ok. 45 min; walidacja 0,448 -> 0,376, minimum przy kroku 1000, od ok. 700 prawie płasko) -> odpowiedzi bazy i adaptera (ckpt 1000) na 76 pytaniach
(`compare_questions.json`, konteksty i odpowiedzi Qwena z `eval_models.json`, max 1000 tokenów) -> sędzia Qwen3.8-27B-FP8, **każda para w obu kolejnościach A/B** (nowa opcja `judge_distill.py --both-orders`: werdykt liczy się tylko gdy zgodny, niespójne = remis, błąd tylko gdy zgłoszony w obu).
Adapter: `../bielik-lora-adapters/cuda/krok121/checkpoint-1000` (tylko lokalnie, NIE publikować). Wyniki: `data/distill/kadry_i_place/{eval_k121,judge_k121_ad_vs_base,judge_k121_ad_vs_qwen,judge_k121_base_vs_qwen}.json`, `pod_logs/train_k121.log`.

**Zbiór:** z 1450 odpowiedzi zostało 1040 (train 935 / valid 105). Odrzucone: 264 za długie przykłady (> 12288 tokenów, czyli stronniczość ku krótszym kontekstom), 77 za długich odpowiedzi, **45 bliskich pytaniom ewaluacyjnym**, 23 z cytatem spoza kontekstu, 1 obce pismo.

**Wyniki sędziego (76 pytań; "z werdyktem" = para z werdyktem w OBU kolejnościach):**

| para | z werdyktem | lepszy A / remis / lepszy B | poważne błędy |
|---|---|---|---|
| adapter vs baza v3.1 | 69 | adapter 22 / remis 29 / baza 18 (zgodne werdykty: 22/14/18 z 54) | adapter 21, baza 20 |
| baza v3.1 vs Qwen | 61 | baza 0 / remis 12 / Qwen 49 | baza 23, Qwen 0 |
| adapter vs Qwen | 64 | adapter 0 / remis 15 / Qwen 49 | adapter 30, Qwen 0 |

- adapter vs baza: 22 vs 18 nie jest istotne (test znaku p = 0,32); pary o podobnej długości (±25%, n = 28): adapter 9 / remis 7 / baza 12; adapter krótszy >20% (n = 38): 12 / 20 / 6.
- Metryki obiektywne: długość odpowiedzi (mediana słów) Qwen 348, baza 183, adapter 136; bez zakończenia zdania: baza 8, adapter 0; cytat spoza kontekstu: baza 0, adapter 2 (art. 86 i art. 188), Qwen 7.
- Punkt odniesienia z Kroku 118 (MLX 8-bit, 30 pytań, jedna kolejność): 25/3/0, błędy 14/28 -- tu baza v3.1 BF16 23/61, więc luka wobec Qwena ta sama.
- Zastrzeżenia: sędzia = Qwen (samopreferencja wobec odpowiedzi Qwena, na które adapter był uczony -- gdyby adapter naprawdę był bliższy Qwenowi, sędzia by go premiował, a nie premiuje); 8 par bez werdyktu (limit myślenia); jedna dziedzina; brak weryfikacji prawnika.

**Wnioski:** (1) dodanie danych (700 -> 1040 przykładów) nie zmieniło obrazu z Kroków 78/79/112; to przepis (destylacja Qwena do LoRA), nie liczba przykładów, jest ograniczeniem -- NIE skalować na inne dziedziny.
(2) Adapter v3.1 kadr NIE wchodzi do produkcji; produkcyjnie zostaje Qwen3.8-27B (Krok 113), Bielik v3.1 to tryb "lekki" z ostrzeżeniem w README.
(3) **Znaleziony przeciek:** 13 z 76 pytań testowych `compare_questions.json` było dosłownie w `answers.jsonl`, a razem z bliskimi (kosinus >= 0,85) 45 z 700 odpowiedzi było zbyt podobnych do pytań ewaluacyjnych (w tym stary adapter kadr iter580 z Kroku 78). Naprawa: `eval_questions()` uwzględnia `compare_questions.json`, a `cmd_dataset` odrzuca takie przykłady (`near_eval_duplicates`).
(4) Lekcje z poda: pip na podzie utknął (ok. 100 kB/s), a `uv pip` w venv z `--system-site-packages` NIE widzi pakietów systemowych i ściąga cały torch (3 GB) -- potem trzeba `uv pip install --no-deps -U torchvision torchaudio` (inaczej `operator torchvision::nms does not exist`); `pkill -f` zabija własną powłokę (użyj `[s]etup`); łącze Mac->pod ok. 200 kB/s, więc indeks RAG (127 MB) budowany na podzie (`build_rag_index.py --domain --include-history`, 31009 wektorów, 1 min) zamiast wysyłany; zsh nie dzieli słów w `$K` dla ssh (funkcja/`${=K}`).
(5) CI: workflow "Testy" padał od Kroku 118 (test osłony adapterów importował `chat.py` z MLX na Linuksie); naprawione w commitach `a7fbd19`, `01be66f` (atrapy MLX, test biegnie też w CI).

## Krok 122 (2026-11-13): adaptery PRODUKCYJNE (v3.0 + LoRA) vs Bielik v3.1 bazowy vs Gemma-4-26B-A4B vs Qwen3.8-27B -- budowlane, cywilne, mieszkaniowe

**Pytanie:** czy przejście na Bielika v3.1 bez adaptera (Krok 118: adaptery v3.0 nie ładują się z v3.1) coś kosztuje w dziedzinach, w których miały adaptery produkcyjne (budowlane k76 iter584, cywilne k78 iter550, mieszkaniowe k78 iter413; konsumenckie poza zestawem -- brak odpowiedzi Qwena). Dodatkowo Gemma-26B-A4B jako punkt odniesienia.
**Metoda:** ten sam kontekst RAG i te same 70 pytań co w Kroku 112 (budowlane 20, cywilne 29, mieszkaniowe 21); odpowiedzi WSZYSTKICH modeli wygenerowane od nowa na jednym podzie (RTX PRO 6000 w EU-RO-1, 2,69 $/h, ok. 1 h 5 min, ok. 3,0 $): Bielik v3.0 BF16 + 3 adaptery (skonwertowane MLX->PEFT, vLLM LoRA), Bielik v3.1 BF16, Gemma-4-26B-A4B BF16 (bez myślenia, 3000 tok.), limit 1000 tok. dla Bielików (jak chat.py od Kroku 118; adaptery NIE są już urywane jak w Kroku 112 przy 500); odpowiedzi Qwena z Kroku 112. Sędzia Qwen3.8-27B-FP8, obie kolejności A/B (`--both-orders`).
Skrypty: `scripts/runpod/{make_bundle_k122,prod_vs_v31}.sh`; wyniki: `data/distill/<d>/{eval_k122,judge_k122_<para>}.json`. W plikach par "adapter" = drugi model w parze.

| para (A vs B) | werdykty | A lepszy / remis / B lepszy | poważne błędy A / B | niespójne |
|---|---|---|---|---|
| adapter prod vs **v3.1 baza** | 58 | 18 / 20 / 20 | 20 / 18 | 15 |
| adapter prod vs Gemma-26B-A4B | 59 | 5 / 27 / 27 | 23 / 3 | 18 |
| v3.1 baza vs Gemma-26B-A4B | 59 | 7 / 22 / 30 | 24 / 7 | 11 |
| v3.1 baza vs Qwen 27B | 55 | 0 / 10 / 45 | 21 / 0 | 6 |
| Gemma-26B-A4B vs Qwen 27B | 59 | 3 / 19 / 37 | 7 / 0 | 8 |

- **adapter prod vs v3.1 baza: brak różnicy** (18:20, test znaku p = 0,87; pary o podobnej długości ±25%, n = 23: 5 / 10 / 8; per dziedzina: budowlane 4 / 5 / 7, cywilne 9 / 9 / 8, mieszkaniowe 5 / 6 / 5; poważne błędy 20 vs 18). **Przejście na v3.1 bez adaptera nic nie kosztuje w tych dziedzinach** -- adaptery produkcyjne są teraz zbędne.
- Gemma-26B-A4B wygrywa wyraźnie z obydwoma Bielikami (z adapterem 27 : 5, z v3.1 30 : 7; błędy 3-7 vs 18-24), ale przegrywa z Qwenem 37 : 3.
- Metryki obiektywne (70 pytań), mediana słów / cytat spoza kontekstu: adapter 128 / 0, v3.1 156 / 0, Gemma 252 / 0, Qwen 304 / 7. Gemma pisze ok. 1,6x dłużej niż v3.1 (premia sędziego za długość -- część przewagi); heurystyka "bez zakończenia zdania" nie nadaje się dla Gemmy (markdown/listy: 48 z 70) -- nie traktować jako ucięcia.
- Zastrzeżenia: sędzia = Qwen (samopreferencja i premia za długość działa na korzyść Gemmy i Qwena), próba mała (70 pytań, 3 dziedziny), 11-18 niespójnych werdyktów na parę liczonych jako remis, brak prawnika; `konsumenckie` (adapter iter200) NIE sprawdzone.

**Decyzja:** zostajemy przy Bieliku v3.1 bez adapterów (zgodnie z Krokiem 118); w README można usunąć wzmiankę o 4 adapterach jako elemencie produkcyjnym przy v3.1. Lekki tryb: Gemma-26B-A4B jest jakościowo lepsza niż Bielik (wcześniej Kroki 115-117), Bielik zostaje dla polskiego pochodzenia (cel użytkownika).
Lekcje: EU-RO-1 daje wielokrotnie szybszy upload z Maca niż US (117 MB w kilkanaście sekund vs ok. 200 kB/s) i pobieranie modeli rzędu 100+ GB w ok. 45 s; `minCudaVersion: "13.2"` w `create-pod` pomogło utworzyć pod, gdy zwykłe próby zwracały brak mocy; skrypt na pod uruchamiać z KOPII (`cp`), bo ponowne rozpakowanie paczki nadpisuje działający skrypt.

**Krok 122b (2026-11-13): to samo dla `konsumenckie`** (adapter iter200, Krok 72; 16 pytań ewaluacyjnych z gotowymi kontekstami z `data/distill/konsumenckie/eval.json`, odpowiedzi Qwena wygenerowane na podzie: 3000 tok., reasoning-effort low).
Pod RTX PRO 6000 w US-NC-1 (w Europie brak mocy), 2,69 $/h, ok. 33 min, ok. 1,5 $; skrypty te same (`make_bundle_k122.sh` z `DOMAINS=konsumenckie`, `prod_vs_v31.sh` z `DOMAINS=konsumenckie` -- generuje brakujące odpowiedzi Qwena). Wyniki: `data/distill/konsumenckie/{eval_k122,judge_k122_<para>}.json`.

| para (A vs B) | werdykty z 16 | A lepszy / remis / B lepszy | poważne błędy A / B | niespójne |
|---|---|---|---|---|
| adapter prod vs v3.1 baza | 12 | 2 / 7 / 3 | 3 / 3 | 4 |
| adapter prod vs Gemma-26B-A4B | 13 | 3 / 6 / 4 | 1 / 2 | 3 |
| v3.1 baza vs Gemma-26B-A4B | 15 | 1 / 7 / 7 | 5 / 1 | 2 |
| v3.1 baza vs Qwen 27B | 12 | 0 / 4 / 8 | 4 / 0 | 2 |
| Gemma-26B-A4B vs Qwen 27B | 13 | 1 / 6 / 6 | 2 / 1 | 3 |

Mediana słów: adapter 146, v3.1 168, Gemma 240, Qwen 302; cytaty spoza kontekstu: 0 / 0 / 0 / 1.
**Wniosek:** także w `konsumenckie` adapter produkcyjny i v3.1 bez adaptera nie różnią się (2:3, błędy 3:3) -- czyli we WSZYSTKICH czterech dziedzinach z adapterem v3.1 bez adaptera niczego nie traci. Zastrzeżenie: tylko 16 pytań (12-15 werdyktów w obu kolejnościach), więc to test kontrolny "bez katastrofalnej straty", nie dowód równości. Tu Gemma nie wygrywa z adapterem istotnie (4:3), ale wygrywa z v3.1 (7:1), a Qwen z oboma.

## Krok 123 (2026-11-14): `cudzoziemcy` -- 33. dziedzina (ustawa o cudzoziemcach, Dz.U. 2013 poz. 1650)

Pilot wykonany autonomicznie (zgoda użytkownika na piloty nowych dziedzin). Wybór: ustawa o cudzoziemcach -- praktyczna, odrębna tematycznie, jednoznaczny akt; sprawdzona przez ELI API (tekst jednolity). Dziedziny już istniejące pominięte (m.in. `ochrona_danych`, `administracyjne`).
Wykonane: wpis w `data/acts_config.json` (dopisany tekstowo -- zapis przez `json.dump` przeformatowałby cały plik), `download_acts.py` (650 artykułów), `download_acts_history.py` (3856 artykułów historycznych, wersje do 2024-03-20), indeks RAG 1419 chunków, 13 przypadków `cud_*` w `data/expand_query_reference_cases.json` (art. 25, 28, 60, 100, 108, 142, 159, 187, 195, 230, 238, 302, 401; każdy artykuł sprawdzony po treści), opis w `prompt.DOMAIN_DESCRIPTIONS`, README (33 dziedziny, 45 aktów, wiersz w tabeli; adapter: baza).
Pomiary: testy jednostkowe 250 przechodzi; bateria `check_expand_query_regressions.py --domain cudzoziemcy`: 13/13 OK w progu top-8 (rangi: 1 3 1 1 1 1 1 6 2 1 2 1 1; najsłabsze: art. 187 nauka -- ranga 6, art. 28 odmowa wjazdu -- ranga 3); router Basal (4.5B 8-bit) na 164 przypadkach: wszystkie 13 `cud_*` top-1 poprawnie, żaden z 151 starych przypadków nie trafił do `cudzoziemcy`, top-1 na starych 120/151 (bez zmian względem Kroku 120).
**Pełna bateria 164 przypadków (151 + 13), uzupełnienie tego samego dnia:**
- dziedziny ręcznie: **161 OK + 3 znane problemy (konsumenckie_28, cywilne_konsument_tv_bez_slowa_konsument, cit_16), 0 nowych regresji**; wszystkie 13 `cud_*` OK;
- retrieval przez router (próg `secondary` 0,85 jak w Kroku 120): **149/164 OK**, 15 niepowodzeń (na starych 151: 136, w Kroku 120: 137 -- różnica 1, w szumie, ale konkretnego przypadku nie wskazano). Wszystkie to pomyłki routera znanych typów (konsumenckie<->cywilne, pcc/cit->podatkowe, kpk/wykroczenia->ruch_drogowy, kadry->kpc/pacjenta, telekom->ochrona_danych); **żadne nie dotyczy `cudzoziemcy`**, wszystkie 13 `cud_*` przez router OK.
- Logi tylko w scratchpadzie sesji, nie w repo.
Zastrzeżenie: treść NIE zweryfikowana przez prawnika (disclaimer + RAG z cytowaniem); przypadki napisane z treści artykułów. Ustawa była często nowelizowana (ostatnia wersja w historii 2024-03-20, sprawdzaj `check_acts_freshness.py`). Adaptery: brak (przepis destylacji zamknięty w Krokach 121-122).

## Krok 124 (2026-11-14): `chat_cuda.py` -- Bielik v3.1 bez adaptera na realnej karcie NVIDIA + przegląd przed publikacją + testy `slow`

**Zmiana w `scripts/chat_cuda.py`:** domyślny model `speakleash/Bielik-11B-v3.1-Instruct` (4-bit nf4), BEZ adaptera; stare adaptery v3.0 tylko jawnie (`--legacy-adapter` z `--model speakleash/Bielik-11B-v3.0-Instruct`, albo `--adapter-path`), słownik `DEFAULT_ADAPTER_PATHS` -> `LEGACY_ADAPTER_PATHS` (jak w `chat.py`, Krok 118).
**Test na realnej karcie** (RunPod, RTX 4090 24 GB, EUR-IS-1, SECURE, **0,89 $/h** -- katalog pokazywał 0,34 $/h dla najtańszej oferty, nie tej dostanej; obraz `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, torch 2.8.0+cu128, sterownik 595.91.07, ok. 1 h, koszt rzędu 1 $): `chat_cuda.py --domain cudzoziemcy --prompt ...` z indeksem RAG `cudzoziemcy`, 3 pytania, każde jako osobny proces:
- "Kiedy cudzoziemiec może dostać zezwolenie na pobyt stały" -> poprawna odpowiedź z art. 195 (warunki w tym zgodne z treścią artykułu); 93 s;
- "Czy mogę legalnie przebywać w Polsce, gdy czekam na decyzję..." -> art. 108 ust. 1 i wyjątek z ust. 2 (zawieszenie na wniosek strony); 56 s;
- pytanie spoza dziedziny (czerwone światło) -> uprzejma odmowa "nie znalazłem w dostarczonych fragmentach"; 42 s.
Czasy zawierają ładowanie modelu (4-bit z cache) i indeksu, więc nie są prędkością generacji. Szczytowe VRAM NIE zmierzone (skrypt pytał nvidia-smi PO zakończeniu procesu, 1 MiB) -- 4-bit 11B mieści się w 24 GB (załadował się i odpowiedział). Odpowiedzi czytane przeze mnie, nie sędziowane i bez prawnika.
**Lekcje:** (1) Bielik v3.1 ma `gated: auto` na HF -- wymaga tokenu (pierwsza próba bez tokenu: "Access ... is restricted"); mój pierwszy odczyt "nie jest gated" był błędny. (2) Pierwszy "sukces" skryptu był FAŁSZYWY (`/usr/bin/time` nie istnieje na podzie, pytania w ogóle nie poszły) -- dodany warunek ALL_DONE: 3 pytania i 3 znaczniki czasu w logu. (3) Obraz ubuntu2404 ma Pythona "externally managed": `pip --break-system-packages`. (4) Sesja SSH NIE dziedziczy `RUNPOD_*` -- watchdog trzeba startować ze zmiennymi z `/proc/1/environ`. (5) `pkill -f <wzorzec>` w komendzie ssh zabija własną sesję (jak w notatce z Kroku 78). (6) Klucz tymczasowy przekazany przez `env.PUBLIC_KEY` przy create-pod -- bez ruszania kluczy konta (`update-ssh-keys`). Pod usunięty po teście (`list-pods`: pusto).
Skrypty: `scripts/runpod/{make_bundle_cuda_smoke,cuda_smoke}.sh` (skrypt wysyłający token HF na poda usunięty w Kroku 125 -- token HF wysyła się ręcznie przez stdin SSH).

**Aktualność aktów (`check_acts_freshness.py`, 2026-11-14):** żaden z 45 aktów nie ma nowszego tekstu ujednoliconego (brak komunikatu "nowszy tekst"); 27 ma tylko późniejszą `changeDate` w metadanych; `prawo_telekomunikacyjne` uchylone celowo (akt historyczny, Krok 67). Ponowne pobieranie niepotrzebne.
**Przegląd przed publikacją repo** (repo PRYWATNE; bez zmian): brak sekretów w kodzie i w historii git; ścieżki lokalne w logach `data/distill`, PROGRESS.md (7), dwóch docstringach i skrypcie wysyłającym token; `data/distill` (191 MB) i `data/finetune` (73 MB) z odpowiedziami Qwena i logami podów -- do decyzji, co publikować; adres e-mail w 148 commitach; LICENSE Apache-2.0 vs notatka o sprzedaży licencji niewyłącznej (Krok ~"Na kiedyś") -- wykluczają się; README bez sekcji o licencjach składników (Bielik, Qwen, `sdadas/*`, dane ELI).
**Testy:** 8 testów na prawdziwych PDF-ach z `data/raw` oznaczone `@pytest.mark.slow` (`pytest.ini`); `pytest -m "not slow"` 3 s (242 testy) zamiast 189 s; CI ich i tak nie uruchamia (brak data/raw).

## Krok 125 (2026-11-14): `scripts/export_public.py` -- eksport czystego drzewa do publikacji (bez pushowania)

Decyzja (rozmowa 2026-11-14): `data/distill` (164 MB w HEAD: odpowiedzi Qwena + ok. 170 logów z podów) i `data/finetune` (76 MB: stare zbiory iter25 + zbiory `*_distill` z kontekstem RAG) NIE idą do publicznego repo -- przepis destylacji jest zamknięty (Kroki 79, 112, 121, 122), odpowiedzi to nieweryfikowane przez prawnika wyniki AI, logi mają ścieżki lokalne, a publikacja danych zamknęłaby opcję sprzedaży licencji. Obecne repo zostaje PRYWATNYM archiwum; publiczne powstaje z wyselekcjonowanego drzewa i świeżej historii (to samo załatwia adres e-mail w 148 commitach i ścieżki w historii).
Skrypt: bierze tylko pliki śledzone w HEAD (`git archive`, pathspec exclude), pomija wybrane katalogi, czyści ścieżki lokalne (katalogi użytkownika, scratchpad Claude Code), skanuje wynik (sekrety i prywatne e-maile => kod 1; ścieżki/IP => uwagi), opcjonalnie `--run-tests`, `--init-git --author <noreply>` (jeden commit, bez zdalnego repo), `--no-distill-scripts` (pomija też `scripts/runpod`, skrypty destylacji/adapterów i ich testy), `--dry-run`. Nigdy nie zapisuje do repo źródłowego, nie nadpisuje niepustego katalogu, nie pushuje.
Pomiary: pełny eksport 140 plików, 2,7 MB (z `--no-distill-scripts`: 98 plików, 2,2 MB); testy w wyeksportowanym drzewie `pytest -m "not slow"`: 235 przeszło, 7 pominiętych (brak data/raw, jak w CI); skan na realnym drzewie czysty; test negatywny skanu (podstawiony token HF i prywatny e-mail) wyłapany, adresy z listy dozwolonych (`@example.`, `noreply@`) nie.
Stan przygotowań do publikacji: patrz Krok 126.

## Krok 126 (2026-11-14): przygotowania do upublicznienia repo (PROGRESS, README, CI, licencja)

Decyzje użytkownika (2026-11-14): z PROGRESS.md usunąć wzmianki o dysku zewnętrznym i zdalnym repo; skrypty wysyłające token HF na poda (`send_hf_token*.sh`) usunięte; reszta porządków "pod publiczne" na mojej głowie. Repo źródłowe zostaje PRYWATNYM archiwum; publiczne powstaje przez `scripts/export_public.py` (Krok 125).
Zrobione:
- **PROGRESS.md:** usunięte nazwy prywatnych projektów siostrzanych (zostaje "projekt źródłowy"), ścieżki lokalne, linki `[[...]]` do notatek asystenta, saldo konta RunPod w jednym miejscu oraz punkt "Na kiedyś: sprzedaż licencji niewyłącznej" (pomysł biznesowy; zachowany poza repo, a w historii gita).
- **README.md:** angielskie streszczenie na górze; intro bez nazw prywatnych projektów; sekcja "Licencje składników i źródła danych" (Apache-2.0 wg kart HF, odczyt 2026-11-14); bez odwołań do `data/finetune`.
- **LICENSE:** Apache-2.0 bez zmian (kod); modele, wagi, adaptery i dane treningowe NIE są dystrybuowane. Imię i nazwisko w nagłówku praw autorskich zostaje decyzją właściciela.
- **CI `acts-freshness.yml`:** wcześniej każdy tydzień kończył się czerwonym wynikiem, bo `check_acts_freshness.py` traktował jako problem samą zmianę `changeDate` (25 z 45 aktów) i celowo uchylone Prawo telekomunikacyjne. Teraz: zmiana samej daty = INFO (nie psuje wyniku); uchylony akt z `"historical": true` w `acts_config.json` (`prawo_telekomunikacyjne`) nie jest problemem; problemem zostaje uchylenie nieoznaczonego aktu, nowy tekst ujednolicony i brak metadanych. `check_act` zwraca teraz `(problemy, informacje)`; 6 nowych testów (`tests/test_check_acts_freshness.py`, odpowiedzi API podstawione). Prawdziwy przebieg: kod 0, 25 INFO, 0 UWAG.

## Krok 127 (2026-11-14): publiczne repo utworzone + `export_public.py --update`

**Repo publiczne** (na razie PRYWATNE): `marduk112/prawo-polskie-rag`, lokalnie `~/Projekty/prawo-polskie-rag`; jeden commit z drzewa HEAD archiwum (140 plików, 2,7 MB, skrypty destylacji wchodzą jako historyczne), autor z adresem noreply GitHuba; CI "Testy" na GitHubie zielone. Archiwum (to repo) zostaje prywatne i jest źródłem prawdy. Do publicznego przełącza właściciel sam (Settings -> Danger Zone), po przeglądzie README/LICENSE.
**`--update <klon>`:** buduje i skanuje drzewo jak pełny eksport (w katalogu tymczasowym), potem synchronizuje z klonem publicznego repo i robi ZWYKŁY commit (historia rośnie; nic nie jest wypychane). Do publicznego trafia tylko to, co przeszło filtr -- `data/distill`, `data/finetune` itd. nigdy. Zabezpieczenia: odmawia, gdy klon to archiwum (też po adresie zdalnym) lub leży w nim, gdy ma niezacommitowane zmiany, przy `--keep-data`, bez `--author`; błąd skanu/testów = klon nietknięty. Zmiany zrobione bezpośrednio w klonie publicznym zostaną nadpisane (źródłem prawdy jest archiwum; `--dry-run` pokazuje listę).
Test na lokalnym klonie: dry-run (~1 zmieniony: sam skrypt), commit, drugi przebieg = "brak zmian", wszystkie 4 odmowy zadziałały.

## Krok 128 (2026-11-14): publiczne repo projektu kadrowo-płacowego -- przepisanie historii i poprawki

`marduk112/RAG-LoRA---kadry-i-place` jest PUBLICZNE (od 2026-10-10; właściciel podał link na LinkedInie). Audyt tymi samymi kryteriami co Krok 125: brak sekretów, e-maili i wag/adapterów; 5 ścieżek lokalnych; cotygodniowy workflow "Aktualność ustaw" czerwony od 3 tygodni z tej samej przyczyny co w Kroku 126 (sama zmiana `changeDate` traktowana jako problem).
**Przepisanie historii** (za zgodą właściciela, pod warunkiem że link zostaje -- zostaje): 44 z 45 commitów miało prywatny adres Gmail jako autora/committera. `git filter-repo --mailmap` -> adres noreply GitHuba; drzewa plików identyczne we wszystkich 45 commitach, Gmail nie występuje już nigdzie w historii (treść, komunikaty). `push --force-with-lease` (jedna gałąź, brak tagów/forków/gwiazdek); stary HEAD `3a7b49c` -> `4faf570`. Kopia zapasowa starego stanu (bundle, zawiera stary adres): `~/Projekty/backup-kadry-i-place-przed-przepisaniem.bundle`. Lokalny klon `RAG-LoRA - kadry i płace` (czysty, bez niewypchniętych commitów) zresetowany do nowej historii. Nie do cofnięcia: stare SHA mogą przez jakiś czas być w cache GitHuba; kopie zrobione wcześniej przez osoby trzecie zachowują stare commity; ewentualne usunięcie z cache = GitHub Support.
**Poprawki** (zwykły commit `bf02b2d`, bez force): `check_acts_freshness.py` (INFO vs problem, jak w Kroku 126) + 5 testów (całe testy: 120 przeszło, 5 pominiętych); ścieżki lokalne usunięte (PROGRESS.md 3, `download_acts_history.py`, `tests/test_rag_search.py`); angielskie streszczenie w README. Workflow "Aktualność ustaw" po zmianie: sukces (kod 0, 5 INFO).
**Link w nowym repo:** README `prawo-polskie-rag` linkuje do repo kadrowego (zrobione przez `--update`, commit `2d1853a`). Odwrotnie (link z kadrowego do nowego) NIE -- nowe repo jest jeszcze prywatne; dopisać po upublicznieniu.
Uwaga: README kadrowe nadal opisuje LoRA jako źródło stylu odpowiedzi -- późniejsze pomiary (Kroki 79, 112, 121, 122) nie pokazały zysku adapterów; treść zostawiona bez zmian jako opis tamtego etapu.

## Krok 129 (2026-11-14): `prawo_przedsiebiorcow` -- 34. dziedzina (Prawo przedsiębiorców + ustawa o CEIDG)

Pilot autonomiczny (zgoda użytkownika). Dwa akty w jednej dziedzinie, bo Prawo przedsiębiorców (Dz.U. 2018 poz. 646, art. 1-72) nie zawiera rejestracji w CEIDG -- ta jest w osobnej ustawie (Dz.U. 2018 poz. 647, 58 art.), a pytania "jak zawiesić/zgłosić zmianę" dotykają obu. Obie zweryfikowane przez ELI API (tekst jednolity, IN_FORCE). Wpisy w `acts_config.json` dopisane tekstowo; pobranie 81 + 58 artykułów, historia (374 + 205), indeks RAG 174 chunków; 14 przypadków `pp_*` (8 z Prawa przedsiębiorców: art. 3, 4, 22, 25, 34, 46, 54, 59; 6 z CEIDG: art. 6, 13, 14, 15, 17, 29; każdy artykuł sprawdzony po treści), opis w `prompt.DOMAIN_DESCRIPTIONS`, README (34 dziedziny, 47 aktów, wiersz w tabeli).
**Pomiary:**
- testy jednostkowe 248 (bez `slow`) przechodzi;
- bateria nowej dziedziny: **14/14 OK**, rangi 1-2;
- pełna bateria 178 przypadków, dziedziny ręcznie: **175 OK + 3 znane (te same), 0 nowych regresji**;
- router Basal (top-1) na 178: 141; `pp_*` trafia 9/14; **pomyłki: `pp_34` (wyjaśnienia/interpretacje), `pp_46`, `pp_54`, `pp_59` (kontrole przedsiębiorców) -> `administracyjne`**, `pp_25` -> `podatkowe` z `secondary` = `prawo_przedsiebiorcow`. Żadne cudze pytanie nie trafiło do nowej dziedziny; na starych 151 top-1 119 (wcześniej 120: różnica to `ksh_230` przy pewności 0,25, szum);
- retrieval przez router (próg `secondary` 0,85): **157/178 OK**, 21 niepowodzeń; nowe wobec Kroku 123: 4 pomyłki routera `pp_*` powyżej oraz dwa szumy (`ksh_230` jak wyżej; `wyk_92a` -- pewność 0,84 -> 0,86, więc `secondary` zostało wyzerowane progiem 0,85). Wszystkie 8 pozostałych `pp_*`/`cud_*` w routerze OK.
**Znane ograniczenie (nienaprawiane):** pytania o kontrole przedsiębiorców i interpretacje indywidualne router kieruje do `administracyjne` (słownictwo "kontrola", "urząd", "wyjaśnienie" kojarzy się z KPA). Hint słownikowy dla routera w Kroku 105 nie dał zysku w telekomunikacji -- nie próbowano.
Zastrzeżenie: treść NIE zweryfikowana przez prawnika; przypadki napisane z treści artykułów. Adaptery: brak.

## Krok 130 (2026-11-14): router -- opis tematyczny dla `prawo_przedsiebiorcow` (opcjonalne `router_description`)

**Problem (Krok 129):** pytania o kontrole przedsiębiorców i interpretacje routowane do `administracyjne` (4 z 14 `pp_*`); opis etykiety z nazw ustaw wyglądał dla modelu jak "prawo przedsiebiorcow, ustawa o ceidg" (skróty `short` bez polskich znaków).
**Warianty (Basal 4.5B 8-bit, jedna etykieta zmieniana):** A = nazwy ustaw z polskimi znakami ("Prawo przedsiębiorców, ustawa o CEIDG"); C = A + hasła tematyczne ("zawieszenie działalności, kontrole przedsiębiorców, interpretacje").
**Zbiór treningowy (178 przypadków, top-1):** baza 141, A 143, C 147; `pp_*` 9 -> 10 -> 12 z 14; żaden cudzy przypadek nie trafił do `prawo_przedsiebiorcow`.
**Zastrzeżenie metodyczne:** C napisano, widząc dokładnie te pytania, które router pomylił -- wynik na nich jest "w próbie" i zawyżony. Dlatego nowy, niewidziany zbiór `data/routing_holdout_pp_cases.json` (20 pytań: 10 o `prawo_przedsiebiorcow` sformułowanych inaczej + 10 z sąsiednich dziedzin, w tym PIP, kontrola skarbowa, drogowa, nadzór budowlany -- słowa "kontrola"/"interpretacja", które mogłyby być przejmowane): baza 6/10, A 6/10, **C 8/10**; negatywy 10/10 w każdym wariancie, wycieku do `pp` 0. Próba mała (n = 10 + 10), więc to wskazówka, nie dowód.
**Retrieval przez router (próg `secondary` 0,85, 178 przypadków):** baza 157 OK, **C 163 OK**; naprawione: 4x `pp_*` (w próbie), `wyk_92a_predkosc` i `cywilne_konsument_rower_zwrot_pieniedzy` (skutki uboczne zmiany rozkładu prawdopodobieństw, nie celowane -- traktować jak szum); **zepsutych 0**.
**Wdrożone:** w `router.build_criteria` pole `router_description` we wpisie `acts_config.json` zastępuje automatyczny opis całej dziedziny (+ test); użyte tylko dla `prawo_przedsiebiorcow` ("Prawo polskie: Prawo przedsiębiorców (zawieszenie działalności, kontrole przedsiębiorców, interpretacje), ustawa o CEIDG"). Zbiór `routing_holdout_pp_cases.json` jest od teraz "widziany" (użyty do decyzji) -- przy kolejnym eksperymencie pisać nowy.
**Wniosek ogólny:** to jedna etykieta i jeden pomiar; wcześniejsza notatka z Kroku 99 ("tematyczne opisy nie poprawiły wyniku") dotyczyła globalnego opisu wszystkich dziedzin i zostaje w mocy. Nie uogólniać na inne dziedziny bez własnego pomiaru na zbiorze niewidzianym. Znane pomyłki routera (konsumenckie<->cywilne, pcc/cit->podatkowe itd.) bez zmian.

## Krok 131 (2026-11-14): test świeżego klonu publicznego repo + poprawki README

Klon `prawo-polskie-rag` do pustego katalogu, venv Python 3.12, kroki z README (ścieżka "RAG + LM Studio", bez MLX): `pip install -r requirements-common.txt -r requirements-dev.txt` OK; `pytest` 243 przeszło + 14 pominiętych (brak PDF-ów, jak w CI); `download_acts.py --domain cudzoziemcy` (650 art.), `build_rag_index.py --domain cudzoziemcy`, `rag_search.py` -> poprawne artykuły (202, 210, 195...) dla pytania o pobyt stały. Wszystko przeszło bez poprawek kodu.
Luki w README znalezione tym testem i poprawione: (1) sekcja "Wymagania" wskazywała Bielika v3.0 jako domyślny (od Kroku 118 model odpowiedzi to v3.1 8-bit, v3.0 4-bit tylko do ekspansji); (2) krok konwersji modeli opisywał tylko v3.0 -- dodane obie konwersje (`--q-bits 8` sprawdzone w `mlx_lm.convert --help`); (3) **brak instrukcji instalacji Basala**, a `pip install basal` instaluje INNY pakiet (PyPI: 0.0.15) -- właściwy to tag `v2.4.1` z GitHuba `rkinas/basal`; (4) nowa sekcja "Szybki start (bez modelu językowego)" ze sprawdzonymi czterema komendami.
NIE sprawdzone na świeżym klonie: ścieżki MLX i CUDA (konwersja modeli, `chat.py`), instalacja Basala z GitHuba i LM Studio -- opis oparty na tym, co działa na maszynie autora.

## Krok 132 (2026-11-14): jakość PEŁNYCH odpowiedzi Qwena w nowych dziedzinach (`cudzoziemcy`, `prawo_przedsiebiorcow`)

Dotąd (Kroki 123, 129) w nowych dziedzinach mierzono tylko retrieval. Teraz 12 przypadków w `qwen_answer_reference_cases.json` (6 + 6: 4 pytania ze znanym artykułem, 1 "pułapka" -- liczba spoza ustaw, 1 pytanie spoza dziedziny), Qwen3.8-27B w LM Studio (8-bit, kontekst 24576, `reasoning_effort=low`), dziedzina podana ręcznie (bez routera). Nowy filtr `check_qwen_answer_regressions.py --domain <d>`. Ok. 6 minut na przypadek (12 przypadków ~ 1 h 15 min).
**Wynik:** automatyczny check 12/12 OK; ręcznie przeczytałem WSZYSTKIE odpowiedzi i sprawdziłem szczegóły z tekstem artykułów (cudzoziemcy: art. 195 w całości, 196, 202, 108, 237, 238, 302; przedsiębiorcy: art. 18, 25, 34 ust. 6, 54 ust. 1-4 z 16 wyjątkami, CEIDG art. 13 i 15). **Wszystkie przytoczone przepisy, liczby (300%, 7 dni, 40 zł, 6 miesięcy, 3/2/5/10/4 lata) i wyjątki zgodne z tekstem.** Pułapki obsłużone poprawnie (nie zmyślono ani opłaty za pobyt czasowy, ani składki ZUS; odesłanie do rozporządzenia z art. 239 / ustawy o systemie ubezpieczeń społecznych), pytania spoza dziedziny (PIT-37, pozew) -> uprzejma odmowa z odesłaniem.
**Zastrzeżenia (znalezione, nienaprawiane):** (1) przy składce ZUS model napisał, że początkujący "nie płaci składek ZUS" przez 6 miesięcy -- art. 18 ust. 1 dotyczy ubezpieczeń SPOŁECZNYCH, składka zdrowotna pozostaje; to uproszczenie z wiedzy modelu, nie z kontekstu; (2) "studentom" zamiast "nauka w szkole ponadpodstawowej i wyższej" (art. 237 ust. 1 pkt 2); (3) przy CEIDG art. 15 model uczciwie zaznaczył, że w kontekście nie było art. 5 (retrieval go nie dostarczył) -- odpowiedź poprawna, ale niepełna. Próba: 12 przypadków, jeden model, jedno uruchomienie, bez prawnika -- wskazówka, nie dowód. Dodatkowo: indeks `cudzoziemcy` zawiera adnotację o przyszłej zmianie art. 202-203 (ustawa z 21.11.2025, wejdzie w życie z dniem z komunikatu) -- model użył brzmienia obowiązującego.

## Krok 133 (2026-11-14): router -- opisy etykiet z prawdziwymi tytułami ustaw (wariant T) -- NIE wdrażamy

**Pytanie:** `build_criteria` podaje routerowi skróty z `acts_config.json` ("kpa", "ustawa o pit", "prawo przedsiebiorcow" -- bez polskich znaków). Czy tytuły ustaw z metadanych ELI ("Kodeks postępowania administracyjnego", "ustawa o podatku dochodowym od osób fizycznych", z polskimi znakami) poprawiają routing wszystkich dziedzin? (Wariant T; ręczny `router_description` dla `prawo_przedsiebiorcow` zachowany w obu wariantach.)
**Metoda:** NOWY, niewidziany zbiór 68 pytań (po 2 na każdą z 34 dziedzin, tematy inne niż w bateriach; `data/routing_unseen68_cases.json`) + bateria 178 (widziana -- na niej pracowano wcześniej). Basal 4.5B 8-bit, top-1 / top-2 (primary lub secondary).
| zbiór | baza (skróty) | T (tytuły) |
|---|---|---|
| niewidziane 68, top-1 | **61** | 60 |
| niewidziane 68, top-2 | 67 | 67 |
| bateria 178 (widziana), top-1 | 147 | 152 |
| bateria 178 (widziana), top-2 | 163 | 168 |
Na niewidzianych: 0 naprawionych, 1 zepsute (`u_budowlane_1` -> `cywilne`). Na widzianych: 7 naprawionych, 2 zepsute (`wyk_66`, `pp_54`).
**Wniosek:** brak dowodu na uogólnianie (niewidziane 61 -> 60); poprawa na widzianej baterii to dopasowanie do przypadków, na których wcześniej strojono. **Nie wdrożono.** Zgodne z notatką z Kroku 99 (tematyczne/ręczne opisy nie pomagają globalnie) i z wnioskiem Kroku 130 (jedna etykieta, pomiar na niewidzianych).
**Pomyłki bazy na niewidzianych (7/68):** `postepowanie_karne` (tymczasowe aresztowanie -> `karne`), `prawo_przedsiebiorcow` (kontrola urzędu bez słowa "przedsiębiorca" -> `administracyjne`), `telekomunikacyjne` (podwyżka abonamentu -> `konsumenckie`), `wodne` 2/2 (studnia -> `geologia_i_gornictwo`, zanieczyszczenie rzeki -> `ochrona_srodowiska`), `wykroczenia` 2/2 (zaśmiecanie lasu -> `ochrona_srodowiska`, odrzucenie mandatu -> `ruch_drogowy`). W 6 z 7 poprawna dziedzina jest `secondary` (top-2: 67/68), czyli przy progu 0,85 retrieval często to łapie; część pytań jest naprawdę niejednoznaczna (zaśmiecanie lasu to i wykroczenie, i ochrona środowiska).
**Szacunek jakości routera:** top-1 ok. 90% na niewidzianych zwykłych pytaniach (61/68), ok. 82% na bateriach (zawierają znane trudne przypadki), top-2 ok. 98%. Zbiór niewidziany jest od teraz "widziany" -- przy następnym eksperymencie pisać nowy. Najsłabsze dziedziny: `wodne`, `wykroczenia` (oba po 2/2 pomyłki).

## Krok 134 (2026-11-14): router -- opisy tematyczne dla `wodne` i `wykroczenia` -- NIE wdrażamy

**Pytanie:** `wodne` i `wykroczenia` miały po 2/2 pomyłki na niewidzianym zbiorze 68 (Krok 133). Czy opis tematyczny etykiety (jak dla `prawo_przedsiebiorcow`, Krok 130) to naprawia? Warianty: W = Prawo wodne (pozwolenia wodnoprawne, pobór wody, studnie, rzeki i jeziora, melioracje, zanieczyszczenie wód); K = Kodeks wykroczeń (grzywny, mandaty, nagana) + Kodeks postępowania w sprawach o wykroczenia (odmowa przyjęcia mandatu, sprzeciw od wyroku nakazowego); WK = oba.
**Nowy, niewidziany zbiór 41 pytań** (`data/routing_unseen41_cases.json`; 8 `wodne` + 8 `wykroczenia` sformułowanych inaczej niż poprzednio + 25 z sąsiednich dziedzin: środowisko, górnictwo, ruch drogowy, karne, kpk, budowlane, kadry, administracyjne, cywilne, energetyczne, pp -- do wykrywania przejmowania cudzych pytań):
| wariant | wodne | wykroczenia | negatywy poprawne | wyciek do wodne/wykr. |
|---|---|---|---|---|
| baza | 7/8 | 8/8 | 22/25 | 0 |
| W | 6/8 | 8/8 | 22/25 | 0 |
| K | 7/8 | 8/8 | 23/25 | 0 |
| WK | 6/8 | 8/8 | 23/25 | 0 |
Różnice to 1 pytanie w każdą stronę (W psuje `v_00` -> `ochrona_srodowiska`, K naprawia jedno pytanie negatywne); kontrola WK na zbiorach już widzianych: niewidziane68 61 -> 62, bateria 178 147 -> 148 (naprawione 3, zepsute 1: `ubez_98_ufg_sprawca_uciekl` -> `ruch_drogowy`). Wszystko w granicach szumu przy tej wielkości próby.
**Wniosek:** brak dowodu na zysk, wariant W lekko szkodzi -- **nie wdrożono**. Baza trafia w tych dwóch dziedzinach lepiej niż sugerował zbiór 68: razem (68 + 41) `wodne` 7/10, `wykroczenia` 8/10 -- niewielka słabość, której opisy nie usuwają; wcześniejsze "2/2 pomyłki" to artefakt małej próby (n = 2). Zbiór 41 jest od teraz "widziany". Kolejne kroki w routerze mają sens dopiero przy innym pomyśle (np. kalibracja progu `secondary` dla tych dziedzin, albo własny klasyfikator), nie przy dopisywaniu haseł do opisów -- cztery próby (Kroki 99, 105, 133, 134) tego nie wspierają.

**Do sprawdzenia później (pomysł użytkownika, 2026-11-14; nie zrobione):** czy asystent poprawnie odpowiada na pytania z wymiarem czasowym w prawie karnym i wykroczeniach -- sprawdza się ustawę z czasu popełnienia czynu i z czasu orzekania i stosuje względniejszą dla sprawcy (KK art. 1 § 1 i art. 4 § 1-4; KW art. 2 § 1-2 -- przepisy są w indeksie, odczytane z tekstu). Tryb `--as-of` obsługuje jedną datę, a takie pytanie wymaga dwóch (czyn i orzekanie) oraz zasady lex mitior, więc wyszukiwanie może nie wyciągać tych artykułów. Plan: 3-4 pytania w `karne`/`wykroczenia` w zestawach regresyjnych, ręczne porównanie odpowiedzi z KK art. 1, 4 i KW art. 2, ewentualna uwaga w prompcie lub dokumentacji.

**Do zrobienia później (pomysł użytkownika, 2026-11-14; nie zrobione): przeniesienie ulepszeń do publicznego repo kadrowo-płacowego.** Repo `RAG-LoRA---kadry-i-place` ma ten sam rdzeń (ELI + historia + `--as-of`, PDF, MLX/CUDA/LM Studio, prompt, CI aktualności), ale wyszukiwanie oparte wyłącznie na podobieństwie wektorów: bez rerankera, fuzji RRF, rozszerzania zapytań, wycinków długich artykułów i zestawów regresyjnych. Kandydaci do przeniesienia (od największego spodziewanego zysku): reranking + RRF, rozszerzanie zapytania, `article_excerpt.py`, jednodziedzinowa wersja baterii, aktualizacja README (Qwen w LM Studio, Bielik v3.1, link do repo wielodziedzinowego po jego upublicznieniu). Raczej nie warto: router i konfiguracja wielodziedzinowa, skrypty destylacji i adaptery LoRA. Warunek: pomiar przed/po na tych samych pytaniach kadrowych (dziś przewaga to wniosek z różnicy mechanizmów, nie z pomiaru); praca na gałęzi, autor z adresem noreply, bez force-push (repo publiczne).

## Krok 135 (2026-11-14): `rachunkowosc` -- 35. dziedzina (ustawa o rachunkowości, Dz.U. 1994 poz. 591)

Pilot autonomiczny. **Uwaga na identyfikator:** Dz.U. 1994 poz. 694 to INNY akt (rozporządzenie o kontyngencie celnym), a wyszukiwarka ELI po tytule zwraca same nowelizacje -- poprawny jest poz. **591** (tekst jednolity, IN_FORCE; sprawdzone odczytem tytułu). Pobranie 154 artykułów + 6 załączników, historia 1169, indeks RAG, 14 przypadków `rach_*` (art. 2, 3, 10, 13, 25, 26, 28, 46, 52, 53, 64, 69, 71, 77; artykuły sprawdzone po treści; progi jednostki mikro w tekście to 2 mln zł / 4 mln zł / 10 osób), opis w `prompt.DOMAIN_DESCRIPTIONS`, README (35 dziedzin, 48 aktów).
**Pomiary:** testy jednostkowe 249 przechodzi; bateria nowej dziedziny 14/14 OK (rangi 4 1 1 1 1 1 2 1 1 1 1 2 1 1); pełna bateria 192 przypadków, dziedziny ręcznie: **189 OK + 3 znane, 0 nowych regresji**; router Basal top-1 159/192, `rach_*` 13/14 (jedyna pomyłka: `rach_77_kary` -> `wykroczenia`, `rachunkowosc` jako secondary), żadne cudze pytanie nie trafiło do nowej dziedziny; **retrieval przez router (próg 0,85): 177/192 OK**, 15 niepowodzeń (poprzednio 15/178), wszystkie `rach_*`/`pp_*`/`cud_*` OK; różnice wobec poprzedniego pomiaru to dwa przypadki przy granicy pewności routera (`ksh_230` naprawiony, `cywilne_konsument_rower_zwrot_pieniedzy` zepsuty -- znane przerzucanie).
Nie sprawdzano pełnych odpowiedzi modelu w tej dziedzinie (tylko retrieval); treść NIE zweryfikowana przez prawnika. Adaptery: brak.
**Dodatkowo tego dnia:** `JAK_TO_DZIALA.md` (opis projektu, link w README); router zamknięty bez dalszych zmian (Kroki 133-134).

## Krok 136 (2026-11-15): jakość PEŁNYCH odpowiedzi Qwena w dziedzinie `rachunkowosc`

Jak w Kroku 132: 6 przypadków w `qwen_answer_reference_cases.json` (4 pytania ze znanym artykułem: art. 3, 52, 64, 77; 1 "pułapka" -- stawka amortyzacji samochodu, której ustawa nie podaje; 1 pytanie spoza dziedziny -- urlop), Qwen3.8-27B w LM Studio (8-bit, kontekst 24576, `reasoning_effort=low`), dziedzina ręcznie. Lokalnie, bez RunPod (przy ~6 min/przypadek wynajem GPU z konfiguracją stosu nie ma sensu).
**Wynik:** automatyczny check 6/6 OK; ręcznie przeczytałem wszystkie odpowiedzi i sprawdziłem z tekstem artykułów (art. 3, 28a, 32, 52, 53, 63c, 64, 77). **Przytoczone przepisy i liczby zgodne z tekstem** (progi mikro 2 mln zł / 4 mln zł / 10 osób; terminy 3 i 6 miesięcy; progi badania 50 osób / 3 125 000 euro / 6 250 000 euro; kara do lat 2). Pułapka obsłużona (nie zmyślono stawki), pytanie spoza dziedziny -> odmowa z odesłaniem do Kodeksu pracy.
**Zastrzeżenia (nienaprawiane):** (1) przy amortyzacji model napisał, że wykaz stawek jest "załącznikiem do rozporządzenia MF" lub "do ustawy o rachunkowości" -- to wiedza spoza kontekstu i nieprawdziwa (wykaz jest załącznikiem do ustaw o PIT/CIT); pominął też art. 32 ust. 8 (wyłączenia z ust. 7: spółki kapitałowe itd.); (2) przy badaniu (art. 64) pominął skonsolidowane sprawozdania grup kapitałowych, a prostej spółki akcyjnej nie umiał przypisać (uczciwie to zaznaczył; wchodzi pod pkt 4, nie pkt 3). Próba: 6 przypadków, jeden model, jedno uruchomienie, bez prawnika -- wskazówka, nie dowód. Pozostałych 8 pytań `rach_*` sprawdzono tylko retrievalem (Krok 135).

## Krok 137 (2026-11-15): backport ulepszeń wyszukiwania do publicznego repo kadrowo-płacowego

Gałąź `retrieval-rerank` w `RAG-LoRA---kadry-i-place` (autor noreply, bez force-push, NIE scalona do `main` -- do decyzji użytkownika). Przeniesiono reranker + fuzję RRF z `rag_search.py` (wersja jednodziedzinowa, z testami, domyślnie włączone, `--no-rerank` przywraca stare zachowanie) i nową baterię `data/retrieval_battery.json` (26 pytań kadrowych, etykiety sprawdzone po treści) ze skryptem `check_retrieval_regressions.py`.
**Pomiar (ten sam indeks i pytania):** tylko kosinus top-1/3/8 = 17/22/25; reranker + RRF = 18/25/26 (poprawione 7, pogorszone 1). Rozszerzanie zapytań (Qwen): 20/24/26 -- zysk w szumie, kosztowne, NIE przeniesione. Wycinki artykułów: nie mierzono (wymagają oceny pełnych odpowiedzi). Zastrzeżenia: próba 26 pytań ułożonych przez autora, baza już wysoka; `search_with_history` porównuje kosinus pierwszego wyniku po fuzji (tak samo jak tutaj), wpływu na pytania uzupełniające nie mierzono. Pełnych odpowiedzi modelu w repo kadrowym po zmianie nie sprawdzano.
**Uzupełnienie Kroku 137 (2026-11-15):** gałąź `retrieval-rerank` scalona do `main` repo kadrowego i usunięta. Test pytań uzupełniających (6 rozmów) i pełnych odpowiedzi Bielika v3.1 (4 pytania) po zmianie: bez regresji; przytoczone przepisy i liczby zgodne z tekstem (uproszczenie: urlop wychowawczy dla dziecka niepełnosprawnego podany jako wyjątek zamiast dodatkowego wymiaru, art. 186 § 3). Znana, wcześniejsza słabość (nie z backportu): fallback historii (`search_with_history`, próg kosinusa 0,75) nie włącza się dla "A po 15 latach?" (kosinus ok. 0,78) -- pytanie uzupełniające nie trafia w art. 154, ani z rerankerem, ani bez. Dotyczy też tego repo (ten sam próg); nienaprawiane, do ewentualnego osobnego kroku.
