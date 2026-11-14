# Asystent prawa polskiego (Bielik + RAG + LoRA, wielodziedzinowy)

> **English summary.** A local, multi-domain question-answering assistant for Polish law: retrieval-augmented
> generation (RAG) over the official statute texts from the Sejm's ELI API (33 domains, 45 acts, with version
> history so you can ask "as of date X"), a domain router, and Polish/multilingual LLMs run locally (Qwen3.8-27B via
> LM Studio recommended; Bielik 11B on Apple Silicon/MLX or NVIDIA/CUDA, and Gemma as lighter modes). Documentation
> and the development log (`PROGRESS.md`) are in Polish. **Not legal advice**: contents have not been verified
> by a lawyer; always check the cited articles at the source. Code: Apache-2.0; model weights and training data are
> not distributed here (see "Licencje składników i źródła danych" at the end of this file).

Lokalny (offline po pierwszym pobraniu) asystent odpowiadający na pytania
z zakresu polskiego prawa, bez wysyłania niczego do chmury. Natywnie na
Apple Silicon (MLX) — plus alternatywne ścieżki dla Linuksa/Windows z
kartą NVIDIA (CUDA) i dla dowolnego modelu już załadowanego w LM Studio.

Ten projekt to rozwinięcie wcześniejszego, jednodziedzinowego asystenta
kadrowo-płacowego na **wiele dziedzin prawa jednocześnie**, budowanych
i mierzonych osobno. Architektura RAG, parsowanie PDF-ów z ELI API i cały
pipeline pochodzą z tamtego projektu — zmienił się sposób, w jaki wybiera
się, które ustawy wchodzą w skład bazy (patrz "Dziedziny prawa" niżej).

Dziennik decyzji i znalezisk (co, dlaczego i jakie pułapki po drodze)
jest w [PROGRESS.md](PROGRESS.md). Ten plik to skrócona instrukcja "jak
tego używać".

## Stan projektu (2026-11-14)

**33 dziedziny, 45 aktów prawnych** (pobieranych z oficjalnego ELI API Sejmu, z historią wersji
dla odpowiedzi „na dany dzień"). Każda dziedzina ma własny indeks RAG. **Bielik 11B (domyślnie
v3.1 8-bit od Kroku 118) działa na samej bazie, bez adapterów LoRA:** 4 adaptery z destylacji
Qwena (`konsumenckie`, `budowlane`, `cywilne`, `mieszkaniowe`) są trenowane na v3.0 i z v3.1 się
nie ładują, a Krok 122 pokazał, że nic nie tracimy (adapter produkcyjny vs v3.1 bez adaptera: 18:20,
p = 0,87, poważne błędy 20 vs 18; `konsumenckie` osobno, tylko 16 pytań: 2:3, błędy 3:3). Nowy adapter dla v3.1 (kadry, Krok 121)
też nie dał zysku, więc **nie rozwijamy dalej tego przepisu destylacji**. Pomiary
(PROGRESS.md, Kroki 73, 78, 79, 88, 112, 121, 122) pokazują, że adaptery nie dają mierzalnej
przewagi nad bazą z pełnymi ustępami artykułów, więc **o jakości odpowiedzi decyduje przede
wszystkim wyszukiwanie (RAG)**, nie fine-tuning.

| dziedzina | akty | adapter (tylko z Bielikiem v3.0; v3.1 -- baza) |
|---|---|---|
| `kadry_i_place` | Kodeks pracy, ubezpieczenia społeczne, płaca minimalna, zasiłki, PIT, PPK, rynek pracy | baza |
| `podatkowe` | PIT, Ordynacja podatkowa, VAT, podatek od spadków i darowizn | baza |
| `administracyjne` | KPA | baza |
| `cywilne` | Kodeks cywilny, prawa konsumenta | **adapter** |
| `rodzinne` | Kodeks rodzinny i opiekuńczy | baza |
| `karne` | Kodeks karny | baza |
| `gospodarcze` | Kodeks spółek handlowych | baza |
| `budowlane` | Prawo budowlane | **adapter** |
| `autorskie` | Prawo autorskie | baza |
| `wlasnosc_przemyslowa` | Prawo własności przemysłowej | baza |
| `ochrona_danych` | ustawa o ochronie danych osobowych | baza |
| `zamowienia_publiczne` | Prawo zamówień publicznych | baza |
| `ochrona_srodowiska` | Prawo ochrony środowiska | baza |
| `ruch_drogowy` | Prawo o ruchu drogowym | baza |
| `upadlosciowe` | Prawo upadłościowe, Prawo restrukturyzacyjne | baza |
| `lotnicze` | Prawo lotnicze | baza |
| `geologia_i_gornictwo` | Prawo geologiczne i górnicze | baza |
| `wodne` | Prawo wodne | baza |
| `mieszkaniowe` | ochrona praw lokatorów, własność lokali | **adapter** |
| `energetyczne` | Prawo energetyczne | baza |
| `telekomunikacyjne` | Prawo komunikacji elektronicznej (+ uchylone Prawo telekomunikacyjne w historii) | baza |
| `konsumenckie` | ustawa o prawach konsumenta | **adapter** |
| `kredyt_konsumencki` | ustawa o kredycie konsumenckim | baza |
| `wykroczenia` | Kodeks wykroczeń, Kodeks postępowania w sprawach o wykroczenia | baza |
| `postepowanie_cywilne` | Kodeks postępowania cywilnego | baza |
| `postepowanie_karne` | Kodeks postępowania karnego | baza |
| `prawa_pacjenta` | ustawa o prawach pacjenta i Rzeczniku Praw Pacjenta | baza |
| `pcc` | ustawa o podatku od czynności cywilnoprawnych | baza |
| `cit` | ustawa o podatku dochodowym od osób prawnych | baza |
| `podatki_lokalne` | ustawa o podatkach i opłatach lokalnych | baza |
| `ubezpieczenia_obowiazkowe` | ustawa o ubezpieczeniach obowiązkowych, UFG i PBUK | baza |
| `prawo_bankowe` | ustawa Prawo bankowe | baza |
| `cudzoziemcy` | ustawa o cudzoziemcach | baza |

**Czego ten projekt NIE gwarantuje (ważne przed użyciem):**
- To narzędzie pomocnicze, **nie porada prawna**. Dane nie zostały zweryfikowane przez
  specjalistę (patrz zastrzeżenie niżej).
- **Model 11B potrafi popełnić błąd „cytat przepisu, wniosek poza jego zakresem"** (np.
  rozciągnąć zwolnienie z opłaty z pacjenta na jego bliskich -- Krok 91). Wygląda to na
  ograniczenie pojemności modelu: nie ustąpiło po fine-tuningu (Kroki 15-16) ani po zmianie
  promptu (Krok 30), a Qwen 27B radzi sobie z tym wyraźnie lepiej bez żadnego dostrajania
  (Kroki 32, 38). **Rekomendowanym trybem
  odpowiedzi jest `scripts/chat_lmstudio.py` z Qwen3.8-27B w LM Studio** (Krok 113): wolniejszy
  (kilka minut na odpowiedź, ok. 30 GB pamięci), ale w pomiarach na 8 dziedzinach (Kroki 106, 109, 112;
  sędzia = Qwen, więc wynik zawyżony) wygrywał z Bielikiem bez adaptera w ok. 88% par (Bielik w ok. 1%)
  i z produkcyjnymi adapterami 52:0 (12 remisów); poważne błędy merytoryczne: Qwen 1, Bielik ok. 67.
  Dwa lekkie tryby (Krok 117):
  - **Gemma-4-26B-A4B** w LM Studio (`chat_lmstudio.py`, MoE, 3,8B aktywnych parametrów; ok. 17-20 s na odpowiedź
    w 4/8-bit, 16-28 GB): wobec Qwena 27B 50:0 (17 remisów), poważne błędy 12 (Qwen 0; Bielik v3.0: 40);
    bateria odpowiedzi 33/34 OK + ten sam znany problem co Qwen (Kroki 115-117). Gęsty wariant 31B ma 5 poważnych
    błędów, ale jest 5-7 razy wolniejszy (96-128 s).
  - **Bielik** (`chat.py`, MLX): lekki, w pełni polski tryb. **Ostrzeżenie: jego odpowiedzi są wyraźnie słabsze** niż
    Qwena i Gemmy -- w naszych pomiarach (sędzia = Qwen, bez weryfikacji prawnika) 40 poważnych błędów na 76 pytań
    (v3.1 8-bit: 34), dużo odpowiedzi urwanych przy limicie 500 tokenów (od Kroku 118 domyślnie 1000), a adaptery
    nie poprawiają tego względem bazy ani Qwena (Kroki 112, 121, 122; Gemma-26B-A4B wygrywa z Bielikiem 30:7). Traktuj jego odpowiedzi jako wstępne i sprawdzaj cytowane przepisy.
- **Wyszukiwanie ma znane luki** -- np. bardzo potoczne pytania mogą trafić poza właściwy
  przepis; bateria regresyjna (`scripts/check_expand_query_regressions.py`, 101 przypadków:
  99 OK i 2 znane problemy zostawione świadomie, 0 regresji) pilnuje stanu, ale nie
  gwarantuje pełnego pokrycia.
- **Jakość danych źródłowych** zależy od PDF-ów Kancelarii Sejmu (parser koryguje indeksy
  górne, załączniki, nawiasy zmian, mojibake starych czcionek; okna historii z zepsutym
  skanem OCR są pomijane przez bramkę jakości). Nowe wersje PDF-ów potrafią zmienić format -- po każdym odświeżeniu danych
  uruchom `scripts/check_acts_freshness.py` i baterię regresyjną.

**Jak dodać nową dziedzinę (sprawdzona recepta, ok. 1 h):** zweryfikuj akt w ELI (uwaga na
sąsiedni numer „Przepisy wprowadzające"), dopisz wpis do `data/acts_config.json` i opis do
`prompt.DOMAIN_DESCRIPTIONS`, pobierz (`download_acts.py`, `download_acts_history.py`),
zbuduj indeks (`build_rag_index.py --domain D --include-history`), dopisz przypadki do
`data/expand_query_reference_cases.json` (artykuły sprawdzone po treści) i uruchom
`check_expand_query_regressions.py --domain D`. Adaptera nie trzeba -- baza wystarcza.

## Jak to działa

Dwa niezależne mechanizmy działają razem, celowo:

1. **RAG (retrieval-augmented generation)** — pytanie użytkownika jest
   embedowane i porównywane z bazą artykułów pobranych ustaw. Najtrafniejsze
   fragmenty trafiają jako kontekst do modelu. To źródło **faktów** —
   aktualnych, źródłowych tekstów przepisów.
2. **Model Bielik** (polski LLM od SpeakLeash) z opcjonalnym adapterem **LoRA**.
   Pierwotnie adapter miał uczyć **stylu** (zwięzłe odpowiedzi z cytowaniem
   artykułu, przyznawanie się do niewiedzy). Pomiary pokazały jednak, że poza 4
   dziedzinami (patrz „Stan projektu") adapter nie poprawia wyników względem
   samej bazy, więc domyślnie działa baza + RAG; tę część opisu zachowano jako
   kontekst historyczny.

Testy w tym projekcie (pierwsza dziedzina: kadry/płace, patrz PROGRESS.md) pokazały,
że sam doduczony model bez kontekstu RAG nie jest wystarczająco
niezawodny w precyzyjnych progach liczbowych i przy pytaniach spoza
wytrenowanego zakresu potrafi pewnie zmyślać fakty — stąd oba mechanizmy
naraz, nie tylko fine-tuning.

**Wyszukiwanie RAG ma trzy dodatkowe etapy** (dodane przy kolejnych
pilotażach, gdzie samo podobieństwo kosinusowe nie wystarczało):
1. **Reranking** (`rag_search.py`, domyślnie włączony, `--no-rerank` do
   wyłączenia) — cross-encoder (`sdadas/polish-reranker-base-ranknet`)
   ocenia precyzyjniej top-30 kandydatów z etapu kosinusowego, bo samo
   podobieństwo bi-encodera potrafi przegrać ogólny, trafny przepis
   niszowym, leksykalnie bliższym artykułem z innej ustawy.
2. **Rozszerzanie zapytania** (`chat*.py`, domyślnie włączone,
   `--no-expand-query` do wyłączenia) — jedno dodatkowe, krótkie
   wywołanie modelu przeformułowuje potoczne pytanie na język formalny,
   zbliżony do ustaw (np. "zakupy prywatne" -> "wydatki niezwiązane z
   prowadzoną działalnością gospodarczą"), zanim trafi do wyszukiwania —
   naprawia przypadki, w których potoczne sformułowanie w ogóle nie
   trafiało we właściwy artykuł, nawet po rerankingu. Od Kroku 21 to
   wywołanie zawsze używa BAZOWEGO modelu (LoRA wyłączana na czas tego
   jednego wywołania, `chat.py::set_lora_enabled`) -- domenowy adapter,
   mimo odmiennego promptu systemowego każącego mu nie cytować przepisów,
   i tak "przeciekał" swój wyuczony nawyk cytowania do tego wywołania,
   czasem podstawiając pod poprawny numer artykułu treść zupełnie innego
   przepisu (patrz PROGRESS.md, Krok 21).
3. **Fuzja rang (RRF)** (`rag_search.py::fuse_rankings`, domyślnie
   włączona, `--no-fuse` do wyłączenia) — reranker sam w sobie ma
   odwrotną wadę do bi-encodera: potrafi spychać krótki, ogólny/
   fundamentalny przepis mimo bardzo wysokiego podobieństwa
   kosinusowego, gdy w tym samym temacie jest gęsty klaster podobnie
   sformułowanych przepisów szczegółowych (patrz "Dziedziny prawa" niżej
   -- `cywilne`/`karne`). RRF łączy pozycję kandydata w obu rankingach
   (kosinus + reranker) zamiast ufać tylko jednemu z nich — naprawia to
   bez arbitralnej wagi ani heurystyki opartej na długości tekstu.

Reranking i fuzja pomagają na etapie OCENIANIA kandydatów; rozszerzanie
zapytania pomaga na etapie ZNAJDOWANIA ich — to inny problem (luka
słownikowa potoczny/formalny język), więc obie naprawy działają
niezależnie i czasem wchodzą w interakcję (patrz "Dziedziny prawa").

## Dziedziny prawa

Zamiast jednego indeksu na całe polskie prawo, każda dziedzina ma **własny
indeks RAG** (adaptery LoRA, jeśli w ogóle, to tylko z Bielikiem v3.0 -- patrz wyżej), budowane i oceniane osobno — dopiero
gdy dana dziedzina jest sprawdzona w izolacji, ma sens rozważać łączenie
indeksów.

**Uwaga o adapterach (PROGRESS.md, Krok 73):** pierwsze adaptery per
dziedzina (iter25, mały ręczny zbiór bez kontekstu RAG) wypadły na 89
pytaniach spoza treningu średnio gorzej od samego modelu bazowego, więc
`chat.py` domyślnie używa adaptera tylko w dziedzinach z adapterem z
destylacji Qwena: `konsumenckie` (Krok 72),
`budowlane` (Krok 76), `cywilne` i `mieszkaniowe` (Krok 78; trening na
CUDA/RunPodzie, konwersja `convert_lora.py peft-to-mlx`); pozostałe dziedziny
(w tym `kadry_i_place`: hedge-iter50 nie wykazał przewagi nad bazą na 76 pytaniach,
Krok 88, odpięty w Kroku 89) działają na bazie -- w Kroku 78 sędzia Qwen nie wskazał w nich przewagi
adaptera nad samym Bielikiem z pełnymi ustępami długich artykułów (Krok 77).
Stare adaptery są w `chat.py::LEGACY_ADAPTER_PATHS` (`--adapter-path`). Kroki 121-122: adapter
destylacyjny dla Bielika v3.1 (kadry) nie pobił bazy ani nie zbliżył się do Qwena, a adaptery produkcyjne
v3.0 są od bazy v3.1 nieodróżnialne, więc **kolejnych adapterów z tego przepisu nie robimy**; skrypty poda
w `scripts/runpod/` (pipeline, ewaluacja `scripts/eval_distill_vllm.py`, sędzia `judge_distill.py --both-orders`,
strażnik kosztów) zostają do ewentualnych innych eksperymentów.

Lista aktów do pobrania (z podziałem na dziedziny) jest w
[`data/acts_config.json`](data/acts_config.json) — płaska lista wpisów
`{short, publisher, year, position, domains}`. Pole `domains` to lista,
bo jeden akt może należeć do więcej niż jednej dziedziny (np. ustawa o
PIT jest jednocześnie w `kadry_i_place` i `podatkowe`).

**Zastrzeżenie dotyczące WSZYSTKICH dziedzin:** żaden z zestawów
zbiorów treningowych używanych w eksperymentach z adapterami (nie są częścią tego repo) nie zweryfikował specjalista
z danej branży (kadrowego, doradcę podatkowego, prawnika) -- napisał je
autor tego repo na podstawie wiedzy ogólnej i testów działania kodu
(czy RAG trafia w dobry artykuł, czy model poprawnie cytuje i przyznaje
się do niewiedzy), nie merytorycznej poprawności prawnej każdego
szczegółu. Traktuj te przykłady jako materiał do nauki STYLU odpowiedzi,
nie jako zweryfikowane źródło faktów. Fakty i tak mają pochodzić z
kontekstu RAG (prawdziwy tekst ustawy z ELI API, `source_url` w każdym
wyniku pozwala zweryfikować źródło) -- ale zanim ktokolwiek oprze się na
odpowiedziach tego asystenta w realnej sprawie, powinien je zweryfikować
z odpowiednim specjalistą lub bezpośrednio z tekstem ustawy. To projekt
edukacyjny/eksperymentalny, nie źródło porady prawnej czy podatkowej.

Obecne dziedziny (wszystkie mają zbudowany indeks z historią
wersji, `--include-history` -- działa też `--as-of`/`/data`, patrz
"Użycie" niżej; każde okno to tekst jednolity z dnia ogłoszenia, patrz
PROGRESS.md Krok 69):
- **`kadry_i_place`** — prawo pracy i ubezpieczeń społecznych (7 ustaw).
- **`podatkowe`** — Ordynacja podatkowa + ustawa o VAT (ustawa o PIT
  współdzielona z `kadry_i_place`). Pilotażowa dziedzina -- mniej
  przetestowana niż `kadry_i_place` (patrz zastrzeżenie wyżej).
- **`administracyjne`** — Kodeks postępowania administracyjnego (KPA,
  jedna ustawa). Też pilotażowa (patrz zastrzeżenie wyżej).
- **`cywilne`** — Kodeks cywilny (jedna ustawa, ale duża i wewnętrznie
  zróżnicowana -- 1296 artykułów). Świadomie wybrana jako najtrudniejszy
  dotąd test: realnie nakłada się tematycznie na `kadry_i_place` (umowa
  o pracę odwołuje się do KC). Tu po raz pierwszy znaleziony problem
  gęstości tematycznej rerankingu (patrz niżej) -- naprawiony fuzją rang
  w pełni, także z domyślnie włączonym rozszerzaniem zapytania (Krok
  12; art. 415 wraca do top-8). Od Kroku 71 indeks zawiera też ustawę
  o prawach konsumenta (wspólną z `konsumenckie`) + hint słownikowy dla
  `expand_query` -- pytanie konsumenta o wadliwy towar trafia w art.
  43a-43q zamiast w rękojmię z KC, która od 2023 r. go nie dotyczy.
- **`rodzinne`** — Kodeks rodzinny i opiekuńczy (jedna ustawa, 247
  artykułów). Wybrana celowo jako kolejny "czysty" test (pojedynczy
  akt, słownictwo odrębne) -- drugi punkt danych dla hipotezy o
  gęstości tematycznej z pilotażu `cywilne`. Problem tu był od początku
  praktycznie nieobecny (ogólne przepisy w gęstych klastrach, np. art.
  133 wśród ~15 przepisów o alimentach, trafiają na #1) -- jedyny
  sprawdzony przypadek z łagodnym odchyleniem to art. 56 (ogólna
  przesłanka rozwodu, #2 zamiast #1).
- **`karne`** — Kodeks karny (jedna ustawa, 461 artykułów). Trzeci punkt
  danych -- problem wystąpił tu poważnie (art. 148, podstawowa kara za
  zabójstwo: #1 kosinusowo, ale sam reranker spychał go na #6), naprawiony
  fuzją rang w pełni (art. 148 zostaje #1 z rozszerzaniem zapytania).
- **`gospodarcze`** — Kodeks spółek handlowych (jedna ustawa, ale duża --
  941 artykułów, 7463 fragmentów po podziale). Siódma dziedzina, wybrana
  jako naturalne rozszerzenie w stronę prawa gospodarczego (patrz
  PROGRESS.md, Krok 17). Wewnętrznie mocno powtarzalna -- KSH duplikuje
  niemal identyczne przepisy dla różnych typów spółek (np. zakaz łączenia
  funkcji zarządu i rady nadzorczej: art. 214 dla sp. z o.o., art. 387 dla
  S.A., art. 300⁷² dla prostej spółki akcyjnej) -- w praktyce to jeszcze
  ostrzejszy przypadek problemu gęstości tematycznej niż `cywilne`/`karne`.
  Znaleziony też przypadek, w którym rozszerzanie zapytania (Krok 14) nie
  trafiało w konkretny mechanizm ustawowy przy luźno sformułowanym
  pytaniu ("zwiększyć zadłużenie ponad dwukrotność kapitału" nie trafiało
  w art. 230) -- **naprawiony w Kroku 22**, rozszerzeniem
  `QUERY_EXPANSION_SYSTEM_PROMPT` o tłumaczenie potocznego słownictwa
  ekonomicznego (np. "zadłużenie") na jego ścisły cywilnoprawny
  odpowiednik ("zaciągnięcie zobowiązania").
- **`budowlane`** — Prawo budowlane (jedna ustawa, 235 artykułów, 545
  fragmentów po podziale; z historią -- 2696 artykułów, 5892 fragmenty).
  Ósma dziedzina (patrz PROGRESS.md, Krok 19). Jeden z 14 historycznych
  PDF-ów tej ustawy (obwieszczenie DU/2000/1126) miał szyfrowanie RC4 o
  długości klucza, której nie obsługiwał backend deszyfrujący
  `pypdf`/`cryptography` (`Invalid key size (104) for RC4`) -- naprawione
  w `download_acts.py` przez podmianę wyłącznie implementacji RC4 na
  czysto pythonowy fallback z `pypdf._crypt_providers._fallback` (ten
  sam algorytm, bez nadgorliwej walidacji długości klucza; szyfrowanie
  AES nadal przez `cryptography`, nienaruszone) -- bez zmian w
  zależnościach współdzielonego `.venv`. Znaleziony też przypadek podobny
  do luki `gospodarcze` wyżej -- pytanie o obowiązek ustanowienia
  kierownika budowy (art. 42) sformułowane przez pryzmat doświadczenia
  inwestora myliło reformulację zapytania z zupełnie innym przepisem
  (art. 12/13, uprawnienia budowlane), co dawało błędną podstawę prawną
  w cytacie -- **naprawione w Kroku 21** (przyczyną nie był za słaby
  prompt ekspansji, tylko wyuczony nawyk cytowania samego adaptera
  domenowego, "przeciekający" do wywołania ekspansji mimo odmiennego
  promptu systemowego dla tego wywołania -- naprawa: LoRA jest teraz
  wyłączana na czas `expand_query()` we wszystkich dziedzinach).
- **`autorskie`** — Ustawa o prawie autorskim i prawach pokrewnych (jedna
  ustawa, 215 artykułów, 2211 fragmentów z historią -- 7 wersji, 1495
  artykułów łącznie). Dziewiąta dziedzina (patrz PROGRESS.md, Krok 20).
  Najgęstsza notacja indeksu górnego ze wszystkich dziewięciu dziedzin
  (92 poprawki w bieżącym tekście -- art. 6¹, art. 33³, art. 47¹ itd.).
  Znaleziony też przypadek indeksu górnego na poziomie USTĘPU, nie
  artykułu (art. 1 ust. 2¹ spłaszczony w danych źródłowych) -- osobna
  kategoria od Kroku 18, **naprawiona w Kroku 48** po tym, jak sonda
  przez wszystkie 15 dziedzin ujawniła 201 potwierdzonych przypadków w
  7 dziedzinach (nie tylko tu), a **rozszerzona w Kroku 49** o spłaszczone
  cytaty w prozie do numeru artykułu tej samej ustawy (642 przypadki
  ogółem w 13 aktach, po odsianiu cytatów do innej ustawy i znaczników
  przypisów jako świadomie nienaprawionych) -- cytaty do INNEJ, ale też
  zaindeksowanej ustawy (np. Kodeksu pracy) **dokończone w Kroku 50**
  (14 potwierdzonych korekt w 3 aktach: `ustawa_o_rynku_pracy`,
  `ustawa_o_ochronie_danych_osobowych`, `ustawa_o_ppk`), patrz PROGRESS.md.
  Podobna luka
  w recall jak `gospodarcze`/`budowlane` wyżej (pytanie o prawa do logo
  freelancera na umowie o dzieło nie trafia w kluczowe artykuły nawet
  bez rerankingu) -- ale tu, w odróżnieniu od `budowlane`, end-to-end z
  adapterem odpowiedział w pełni poprawnie.
- **`wlasnosc_przemyslowa`** — Prawo własności przemysłowej (jedna
  ustawa, 412 artykułów, 3283 fragmenty z historią -- 5 wersji, 2403
  artykuły łącznie). Dziesiąta dziedzina, naturalne uzupełnienie
  `autorskie` (patrz PROGRESS.md, Krok 23). Druga najgęstsza notacja
  indeksu górnego (92 poprawki -- w tym art. 129¹/132¹ zastępujące
  uchylone art. 129/132 po reformie unijnej znaków towarowych z 2019 r.).
  Pierwsza dziedzina od Kroku 17, w której pełna weryfikacja end-to-end
  nie ujawniła żadnej nowej luki w recall ani problemu z numeracją --
  obie odpowiedzi scenariuszowe (przestrzenny znak towarowy dla kształtu
  opakowania; wynalazek pracowniczy na umowie zlecenie) wypadły w pełni
  poprawnie za pierwszym razem.
- **`ochrona_danych`** — ustawa o ochronie danych osobowych (jedna ustawa,
  131 artykułów rdzenia merytorycznego, 174 fragmenty; 0 wersji
  historycznych -- nigdy nie było osobnego obwieszczenia o tekście
  jednolitym poza bieżącym). Jedenasta dziedzina (patrz PROGRESS.md,
  Krok 24) i pierwsza, w której materialne zasady tematu (zgoda,
  podstawy przetwarzania, prawa osoby, WYSOKOŚĆ kar) reguluje wprost
  unijne RODO, a nie ta krajowa ustawa -- sam indeks (i model) celowo
  NIE obejmuje treści RODO, tylko krajową warstwę instytucjonalną (IOD,
  akredytacja/certyfikacja, Prezes UODO, postępowanie, kontrola, KTO
  nakłada kary). Znaleziona przy tym pilotażu NOWA, cross-domenowa
  usterka (nie specyficzna dla tej dziedziny, tylko dotąd niewidoczna) w
  `expand_query()`: dla pytań o wysokość sankcji model czasem po prostu
  ODPOWIADA (z konkretną, tu błędną kwotą), zamiast zwrócić samo
  przeformułowanie -- w `karne` (art. 148) ten sam błąd był niewidoczny,
  bo akurat cytował prawidłowy artykuł. Trzy próby naprawy przez
  wzmocnienie promptu ODRZUCONE -- każda naprawiała jeden przypadek
  kosztem regresji w innym, niepowiązanym pytaniu (patrz PROGRESS.md,
  Krok 25) -- pozostaje otwarte.
- **`zamowienia_publiczne`** — Prawo zamówień publicznych (jedna ustawa
  z 2019 r., 636 artykułów, 5530 fragmentów z historią -- 4 wersje, 3137
  artykuły łącznie). Dwunasta dziedzina (patrz PROGRESS.md, Krok 42).
  W odróżnieniu od `gospodarcze`, mimo podobnej wewnętrznej powtarzalności
  (analogiczne mechanizmy dla zamówień klasycznych/sektorowych/obronności
  i bezpieczeństwa), nie znaleziono tu problemu gęstości tematycznej --
  sprawdzone kolokwialne pytania trafiały docelowy artykuł w top-4 nawet
  bez rozszerzania zapytania. Weryfikacja end-to-end ujawniła za to dwa
  generacyjne (nie retrievalowe) znaleziska, świadomie nienaprawione
  do czasu potwierdzenia wzorca w kolejnej dziedzinie: model raz
  zmyślił numer artykułu spoza dostarczonego kontekstu RAG (art. 415
  zamiast art. 118 przy pytaniu o poleganie na zasobach innych
  podmiotów), a raz podał liczbę z wąskiego, nietrafionego przepisu
  (próg dla usług społecznych, art. 359) bez zahedgowania przy ogólnym
  pytaniu o progi unijne. Wzorzec drugi (niewłaściwe uogólnienie wąskiego
  przepisu) potwierdzony niezależną sondą w `karne` (Krok 43) -- rozpoznany
  jako kolejny przejaw już zdiagnozowanego ograniczenia Zasady 6, nie nowy
  problem.
- **`ochrona_srodowiska`** — Prawo ochrony środowiska (jedna ustawa z
  2001 r., 559 artykułów, 12069 fragmentów z historią -- 6 wersji, 6180
  artykuły łącznie, największy indeks domenowy w projekcie). Trzynasta
  dziedzina (patrz PROGRESS.md, Krok 44), najbogatsza historia wersji ze
  wszystkich dziedzin (12 obwieszczeń o tekście jednolitym). Retrieval
  wypadł łagodnie -- podobnie do `rodzinne`, nie głęboki problem gęstości
  jak `cywilne`/`karne`. Weryfikacja end-to-end wypadła czysto (lepiej niż
  `zamowienia_publiczne`): dwie trudne, scenariuszowe odpowiedzi poprawnie
  zgeneralizowały do artykułów spoza zbioru treningowego (w tym dokładne
  przywołanie art. 211 ust. 6 pkt 3 o wymaganiach ochrony gleby i wód
  gruntowych w pozwoleniu zintegrowanym) -- żadnej z dwóch generacyjnych
  usterek znalezionych w `zamowienia_publiczne` nie zaobserwowano.
- **`ruch_drogowy`** — Prawo o ruchu drogowym (jedna ustawa z 1997 r.,
  319 artykułów, 8694 fragmenty z historią -- 4 wersje). Czternasta
  dziedzina (patrz PROGRESS.md, Krok 45), pierwszy pilot prowadzony
  samodzielnie po upoważnieniu użytkownika do "automatycznego" trybu.
  Napotkano i naprawiono nowy wariant artefaktu "jeszcze
  nieobowiązujących" nawiasów `< >` -- dwa artykuły (53a, 65l) kończyły
  się osieroconym `>` bez pary, pojedynczy artefakt ekstrakcji PDF, nie
  brakujący fragment tekstu; naprawione ogólnie w
  `strip_not_yet_in_force_text` (bezpieczne dla wszystkich dziedzin,
  poprzednie 13 nigdy nie zgłosiły tego ostrzeżenia). Ustawa nie zawiera
  już przepisów o egzaminach na prawo jazdy ani o systemie punktów
  karnych -- od 2011 r. reguluje je odrębna ustawa o kierujących
  pojazdami, dobra naturalna podstawa dla przykładów hedge. Jedna
  potwierdzona kolokwialna luka recall (zgłoszenie zbycia pojazdu, art.
  78) w klasycznym wzorcu Kroku 7/14/22 -- zweryfikowana jako w pełni
  zamaskowana przez query expansion w produkcyjnym `chat.py`
  (spontanicznie i poprawnie cytuje też art. 140mb ust. 6, karę 250 zł,
  artykuł spoza zbioru treningowego). Weryfikacja end-to-end czysta,
  bez znalezisk z `zamowienia_publiczne`.
- **`upadlosciowe`** — Prawo upadłościowe (2003 r.) + Prawo
  restrukturyzacyjne (2015 r.), dwa powiązane akty (wzorzec
  `podatkowe`), 602 + 406 artykułów, 13614 fragmentów z historią --
  największy indeks domenowy w projekcie. Piętnasta dziedzina (patrz
  PROGRESS.md, Krok 46). Znaleziony nowy, głębszy wariant problemu z
  nawiasami "jeszcze nieobowiązujące"/"stare ale ważne" -- zagnieżdżone
  lub sąsiadujące bloki `[...]`/`<...>` w dwóch historycznych wersjach
  artykułów myliły jednoprzebiegowe podstawienie regex (w jednym
  przypadku art. 242 był w całości wchłonięty jako "treść" sąsiedniego
  art. 241 dla jednej wersji historycznej) -- **naprawiony w Kroku 47**
  (parser dopasowujący nawiasy pozycyjnie zamiast dwóch niezależnych
  podstawień regex), patrz PROGRESS.md. Dwie potwierdzone kolokwialne
  luki recall (rola syndyka; skutki układu dla egzekucji), obie w pełni
  zamaskowane przez query expansion -- NIE potwierdzono głębszej
  kolizji między dwoma aktami tej samej dziedziny. Weryfikacja
  end-to-end czysta, w tym spontaniczne, poprawne cytowanie art. 491¹¹a
  spoza zbioru treningowego.
- **`lotnicze`** — Prawo lotnicze (2002 r.), jeden akt, 510 artykułów,
  1393 fragmenty, 4157 artykułów historycznych z 5 wersji. Szesnasta
  dziedzina (patrz PROGRESS.md, Krok 52), wybrana jako "czysty" pilotaż
  (słownictwo lotnicze praktycznie nie nakłada się z żadną z pozostałych
  15 dziedzin) spośród trzech kandydatów bez oczywistego faworyta (prawo
  geologiczne i górnicze, prawo wodne -- oba tematycznie bliższe
  `ochrona_srodowiska`). Weryfikacja end-to-end czysta, w tym
  spontaniczne, poprawne cytowanie art. 78a ust. 5 spoza zbioru
  treningowego (przywrócenie dostępu do lotniska po uregulowaniu
  zaległych opłat). Uwaga: do Kroku 63 indeks tej dziedziny był (mimo
  pobranej historii) zbudowany BEZ `--include-history` -- 1393 fragmenty
  samego bieżącego tekstu, więc `--as-of` po cichu nie działał; od Kroku
  63 indeks zawiera historię (~11,6 tys. fragmentów).
- **`geologia_i_gornictwo`** — Prawo geologiczne i górnicze (2011 r.),
  jeden akt, 352 artykuły, 8483 fragmenty, 3261 artykułów historycznych
  z 5 wersji. Siedemnasta dziedzina (patrz PROGRESS.md, Krok 54),
  wybrana nad prawem wodnym (ostatnim z trzech kandydatów odłożonych w
  Kroku 52) jako mniejsza i węziej skoncentrowana tematycznie (koncesje
  górnicze, kwalifikacje geologa, dokumentacja geologiczna) niż prawo
  wodne, którego słownictwo pozwoleń wodnoprawnych mocniej pokrywa się z
  `ochrona_srodowiska`. Weryfikacja end-to-end czysta, w tym spontaniczne,
  poprawne cytowanie art. 4 (zwolnienie z koncesji dla wydobycia żwiru/
  piasku na potrzeby własne, limit 10 m³/rok) spoza zbioru treningowego,
  identyczne z i bez adaptera.
- **`wodne`** — Prawo wodne (2017 r.), jeden akt, 585 artykułów, 1379
  fragmentów, 3913 artykułów historycznych z 7 wersji (indeks łącznie
  4498 artykułów/10464 fragmenty). Osiemnasta dziedzina (patrz
  PROGRESS.md, Krok 56), ostatni z trzech kandydatów bez oczywistego
  faworyta z Kroku 52. Znaleziona i naprawiona luka recall dla flagowego
  pytania o własność rzeki (art. 211 -- śródlądowe wody płynące = własność
  Skarbu Państwa) -- kosinusowo poza pulą 30 kandydatów do rerankingu,
  naprawione domenowym hintem słownikowym (metoda Kroku 28/41). Znaleziony
  też nowy, precyzyjnie zdiagnozowany wzorzec błędnego cytowania: Bielik
  przypisuje poprawną, faktycznie ugruntowaną treść (art. 394 BYŁ w
  kontekście RAG) numerowi sąsiadującego artykułu o przeciwstawnym
  znaczeniu (art. 395) -- inny mechanizm niż znana luka z Kroku 42/43,
  gdzie cytowany artykuł w ogóle nie istniał. Qwen3.8-27B na tym samym
  kontekście cytuje poprawnie. Jeden potwierdzony przypadek, świadomie
  NIE naprawiony -- udokumentowany jako `known_issue` w
  `data/qwen_answer_reference_cases.json`, czeka na drugi potwierdzony
  przypadek przed próbą ogólnej naprawy.
- **`mieszkaniowe`** — ustawa o ochronie praw lokatorów (2001 r., 78
  artykułów) + ustawa o własności lokali (1994 r., 39 artykułów), 773
  artykuły historyczne z 7 wersji każdej ustawy (indeks łącznie 890
  artykułów/1581 fragmentów). Dziewiętnasta dziedzina (patrz PROGRESS.md,
  Krok 58), wybrana jako pierwsza "praktyczna, codzienna" dziedzina po
  serii czysto branżowych (najem, kaucja, podwyżki czynszu, eksmisja,
  wspólnoty mieszkaniowe). Retrieval bardzo pewny (6/8 potocznych pytań
  z właściwym artykułem na #1; art. 19 -- mała wspólnota -- trafia na #3
  dzięki samej ekspansji zapytania, bez hintu domenowego). Weryfikacja
  end-to-end: poprawne cytowanie art. 13, 18, 12 i `--as-of`
  (art. 19a ust. 4 w oknie 2016-2018), ale adapter w 2/3 odpowiedzi
  dorzuca poboczną uwagę z błędnym numerem przepisu (art. 3 ust. 2 zamiast
  ust. 3; termin 14 dni z art. 19i przypisany art. 19d) -- baza bez
  adaptera tych dopisków nie robi. Udokumentowane, nienaprawiane.
- **`energetyczne`** — Prawo energetyczne (1997 r., 332 artykuły, w tym
  17 z indeksem górnym po literze, np. art. 5a¹, 5b¹-5b⁴), 2150 artykułów
  historycznych z 10 wersji (indeks łącznie 2482 artykuły/9386
  fragmentów). Dwudziesta dziedzina (patrz PROGRESS.md, Krok 60):
  odcinanie prądu/gazu za zaległości, reklamacje, zmiana sprzedawcy,
  dodatek energetyczny, liczniki przedpłatowe, koncesje, taryfy. Pilotaż
  wykrył i naprawił błąd parsera (artykuły typu "Art. 5b¹." były
  wchłaniane przez poprzedni artykuł). Retrieval bardzo pewny (9/11
  potocznych pytań z właściwym artykułem na #1, pozostałe na #3).
  Weryfikacja end-to-end: poprawne cytowanie art. 57, 6a/6f, 6d i
  `--as-of` (art. 5c w oknie 2019-2020), ale adapter raz dorzucił
  poboczny opis przepisu pod błędnym numerem (art. 6b ust. 4 opisany jako
  "nielegalny pobór") -- baza bez adaptera tego nie robi.
- **`telekomunikacyjne`** — Prawo komunikacji elektronicznej (2024 r.,
  450 artykułów; ustawa obowiązuje od 2024-11-10, a historię sprzed tej
  daty daje uchylone Prawo telekomunikacyjne z 2004 r. -- 338 artykułów +
  2278 historycznych z 7 okien, Krok 67; indeks 7374 fragmentów).
  Dwudziesta pierwsza dziedzina (patrz PROGRESS.md, Krok 66): umowy z
  operatorem (maks. 24 miesiące, automatyczne przedłużenie, jednostronne
  zmiany warunków), odszkodowanie za wcześniejsze wypowiedzenie,
  przeniesienie numeru, reklamacje, cookies, telemarketing. Jedyna
  dziedzina poza `wodne` i `gospodarcze` z hintem słownikowym dla
  `expand_query` (ustawa nie mówi "cookies"). Weryfikacja end-to-end:
  3/4 poprawne; przy rezygnacji z umowy z telefonem promocyjnym Bielik (z
  adapterem i bez) stosuje art. 309 zamiast art. 304 -- Qwen nie.
- **`konsumenckie`** — ustawa o prawach konsumenta (2014 r., 74 artykuły,
  321 historycznych z 5 wersji, 764 fragmenty). Dwudziesta druga
  dziedzina (patrz PROGRESS.md, Krok 68): odstąpienie od umowy zawartej
  na odległość/poza lokalem, koszty i terminy zwrotu, wyjątki (towar na
  wymiar, zapieczętowany, bilety), niezgodność towaru z umową (rozdział
  5a -- od 2023 r. zamiast rękojmi z KC dla konsumentów), reklamacje.
  Hint słownikowy dla `expand_query`. Weryfikacja end-to-end: 3/4
  poprawne; przy wadzie powtarzającej się po naprawie adapter błędnie
  odmawia odstąpienia (art. 43e), baza bez adaptera i Qwen -- poprawnie.
  Od Kroku 72 domyślny adapter z destylacji Qwena odpowiada poprawnie.
  Ustawa jest też w indeksie `cywilne` (Krok 71).
- **`kredyt_konsumencki`** — ustawa o kredycie konsumenckim (2011 r., 111
  artykułów + 4 załączniki, 640 historycznych z 7 wersji, 1358 fragmentów).
  Dwudziesta trzecia dziedzina (patrz PROGRESS.md, Krok 74): odstąpienie od
  umowy kredytu (14 dni), wcześniejsza spłata i obniżenie kosztów, limit
  pozaodsetkowych kosztów, sankcja kredytu darmowego, kredyt wiązany,
  przelew wierzytelności. Bez adaptera (model bazowy, Krok 73). Hint
  słownikowy dla `expand_query`: 12/12 pytań w top-8 (bez hintu 10/12).
- **`wykroczenia`** — Kodeks wykroczeń (1971 r., 207 artykułów) i Kodeks
  postępowania w sprawach o wykroczenia (2001 r., 133 artykuły); historia
  3037 wpisów, 4460 fragmentów w indeksie. Dwudziesta czwarta dziedzina
  (patrz PROGRESS.md, Kroki 80-81): kradzież i zniszczenie mienia do 800 zł,
  zakłócanie spoczynku nocnego, wykroczenia drogowe (prędkość, alkohol),
  zwierzęta, fałszywy alarm oraz tryb mandatowy (odmowa przyjęcia mandatu,
  limit mandatu, uchylenie), apelacja w 7 dni. Zamyka lukę z Kroku 43
  (kradzież 300 zł była kwalifikowana jako przestępstwo z KK). Bez adaptera
  (model bazowy), bez hintu `expand_query` (nie był potrzebny): 12/12 pytań w
  top-8, 10 na pozycji 1.
- **`postepowanie_cywilne`** — Kodeks postępowania cywilnego (1964 r., 2026
  artykułów, 16 672 historycznych z 9 wersji, 22 436 fragmentów w indeksie --
  największa dziedzina). Dwudziesta piąta dziedzina (patrz PROGRESS.md, Krok
  84): właściwość sądu, pozew, terminy i przywrócenie terminu, doręczenia
  (awizo), wyrok zaoczny i sprzeciw, apelacja, nakaz zapłaty, zabezpieczenie,
  egzekucja (komornik, zajęcie wynagrodzenia). Bez adaptera (model bazowy),
  bez hintu `expand_query`: 8/8 pytań w top-8, 5 na pozycji 1.
- **`postepowanie_karne`** — Kodeks postępowania karnego (1997 r., 958
  artykułów, 7417 historycznych z 8 wersji, 5013 unikalnych tekstów w indeksie).
  Dwudziesta szósta dziedzina (patrz PROGRESS.md, Krok 85): zawiadomienie o
  przestępstwie, pokrzywdzony i oskarżyciel posiłkowy, świadek (odmowa zeznań
  osoby najbliższej), zatrzymanie i tymczasowe aresztowanie, pouczenie
  podejrzanego, apelacja w 14 dni. Dopełnienie Kodeksu karnego i wykroczeń.
  Bez adaptera (model bazowy), bez hintu `expand_query`: 8/8 pytań w top-8,
  7 na pozycji 1.
- **`prawa_pacjenta`** — ustawa o prawach pacjenta i Rzeczniku Praw Pacjenta (2008 r., 99
  artykułów, 641 historycznych z 7 wersji, 1314 fragmentów). Dwudziesta siódma dziedzina (patrz
  PROGRESS.md, Krok 91): dokumentacja medyczna (wgląd, kopie, opłaty, po śmierci pacjenta),
  zgoda na świadczenia (także małoletni), tajemnica i jej wyjątki, osoba bliska, sprzeciw wobec
  opinii/orzeczenia, zadośćuczynienie, świadczenie kompensacyjne za zdarzenia medyczne, skargi do
  Rzecznika. Bez adaptera (model bazowy), bez hintu `expand_query`: 9/9 pytań w top-8, 7 na
  pozycji 1.

**Problem gęstości tematycznej rerankingu i jego naprawa (Krok 8-12):**
cross-encoder reranker (`sdadas/polish-reranker-base-ranknet`) ma wadę
odwrotną do bi-encodera -- w gęsto obsadzonym temacie (dużo podobnie
sformułowanych przepisów szczegółowych, np. "kto robi X, podlega
karze/odpowiedzialności Y") potrafi spychać krótki, ogólny/
fundamentalny przepis mimo bardzo wysokiego podobieństwa kosinusowego
(sprawdzone na 3 dziedzinach o różnym nasileniu: `rodzinne` -- brak/
łagodny, `karne` -- poważny, `cywilne` -- poważny). Naprawione fuzją
rang (RRF, `rag_search.py::fuse_rankings`/`fuse_rankings_multi_query`,
domyślnie włączona, `--no-fuse` do wyłączenia) -- łączy pozycję
kandydata w rankingu kosinusowym i w rankingu rerankera zamiast ufać
tylko jednemu, bez arbitralnej wagi czy heurystyki opartej na długości
tekstu. Z więcej niż jednym wariantem zapytania (rozszerzanie zapytania
aktywne) fuzja liczona jest OSOBNO w obrębie każdego wariantu, dopiero
potem bierze się maksimum wyniku fuzji per kandydat (Krok 12) -- inaczej
kandydat trafiony słabo przez oryginalne pytanie, ale bardzo dobrze
kosinusowo przez samą parafrazę, mógłby wygrać ranking z kandydatem
faktycznie najlepiej pasującym do prawdziwej intencji użytkownika.
Zweryfikowane end-to-end z PRAWDZIWYM `expand_query()` na 10 przypadkach
referencyjnych z 6 dziedzin: 9/10 w top-8, w tym oba poważne przypadki
(KC art. 415: #9 -> #4, KK art. 148: #6 -> #1) bez psucia żadnego z
wcześniej pewnych wyników. Ostatni wyjątek, VAT art. 86 (Krok 7) --
osobna, niezwiązana z gęstością tematyczną krucha zależność query
expansion od dokładnego sformułowania parafrazy -- naprawiony w Kroku
14: `QUERY_EXPANSION_SYSTEM_PROMPT` dostał instrukcję, żeby zamiast
ogólnego, "ładnie brzmiącego" prawniczo sformułowania, nazywać
KONKRETNY mechanizm prawny tak, jak robi to sama ustawa (bez
domenowo-specyficznego przykładu, żeby nie faworyzować jednej
dziedziny) -- podniosło art. 86 z rangi ~35 na #6, bez regresji na
pozostałych 9 przypadkach. Wszystkie 10/10 referencyjnych przypadków
były wtedy w top-8 -- patrz PROGRESS.md, Krok 14.

**Aktualizacja (Krok 26):** ten stan NIE utrzymał się -- VAT art. 86
cicho zregresował do rangi ~27 przez kolejne, niepowiązane zmiany tego
samego promptu (Krok 21, Krok 22), niezłapane przez ręczną, wybiórczą
weryfikację. Od Kroku 27 istnieje trwały zestaw testów regresyjnych,
`python scripts/check_expand_query_regressions.py`
(`data/expand_query_reference_cases.json`) -- uruchom go PRZED i PO
każdej zmianie `QUERY_EXPANSION_SYSTEM_PROMPT`/
`build_query_expansion_messages()`, nie tylko ręcznie sprawdzaj kilka
wybranych przypadków.

**Aktualizacja (Krok 28):** oba znane problemy naprawione -- ale INNĄ
metodą niż zmiana `QUERY_EXPANSION_SYSTEM_PROMPT` (cztery wcześniejsze
próby tej metody, Krok 25-26, odrzucone). `build_query_expansion_messages()`
dostaje teraz opcjonalny `act_titles` (z nowego `RagIndex.act_titles()`)
-- krótką, czysto faktyczną listę nazw ustaw BIEŻĄCEJ dziedziny,
dopisywaną do wiadomości UŻYTKOWNIKA (system prompt nietknięty).
Domenowo-specyficzne, więc naprawa jednej dziedziny nie może
strukturalnie zepsuć innej. VAT art. 86: ranga 27 -> 2. `ochrona_danych`
art. 101 (Krok 25): ranga 46 -> 6, zweryfikowane też end-to-end przez
`chat.py`. Ograniczenie: to nadal "szczęśliwe maskowanie" widocznego
skutku, nie naprawa źródłowego nieposłuszeństwa modelu przy pytaniach o
wysokość sankcji -- patrz PROGRESS.md, Krok 28.

Żeby dodać kolejną dziedzinę: znajdź identyfikatory ELI nowych aktów
(`/eli/acts/search?title=...` na `https://api.sejm.gov.pl/eli`), dopisz
je do `data/acts_config.json` z odpowiednim polem `domains`, uruchom
`download_acts.py --domain <nazwa>` i `build_rag_index.py --domain
<nazwa>` i dopisz przypadki do baterii regresyjnej
(`data/expand_query_reference_cases.json`). Adaptery LoRA nie są zalecane
(patrz "Stan projektu" wyżej).

## Wymagania

Wspólne dla wszystkich wariantów:
- Python 3.12

Zależnie od wybranej ścieżki:

| Wariant | Platforma | Dodatkowe wymagania |
|---|---|---|
| `chat.py` (MLX) | macOS + Apple Silicon | Konto HF z zaakceptowaną licencją Bielika |
| `chat_cuda.py` (CUDA) | Linux/Windows + karta NVIDIA | Sterowniki CUDA, `torch` z obsługą CUDA, konto HF z licencją Bielika (v3.1 jest "gated": wymaga zalogowania i akceptacji); domyślnie Bielik v3.1 4-bit bez adaptera (Krok 124) |
| `chat_lmstudio.py` (LM Studio) | dowolna | [LM Studio](https://lmstudio.ai/) z załadowanym modelem -- nie wymaga konta HF (bez LoRA -- format GGUF nie wspiera adapterów) |

Reszta projektu (`download_acts.py`, `build_rag_index.py`, `rag_search.py`,
`check_acts_freshness.py`) to czysty Python + `sentence-transformers`/
`requests`/`pypdf` — platformowo niezależne.

Konto Hugging Face z zaakceptowaną licencją modelu (dotyczy wariantów
MLX i CUDA):
- [`speakleash/Bielik-11B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) (domyślny)
- [`speakleash/Bielik-4.5B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) (szybszy wariant, opcjonalny)

## Instalacja

```bash
python3.12 -m venv .venv

# macOS + Apple Silicon (MLX) -- wariant domyślny:
.venv/bin/pip install -r requirements.txt

# Linux/Windows + NVIDIA (CUDA):
# .venv/bin/pip install -r requirements-common.txt -r requirements-cuda.txt

# Tylko RAG + LM Studio (dowolna platforma, bez MLX/CUDA):
# .venv/bin/pip install -r requirements-common.txt

.venv/bin/hf auth login  # pomiń, jeśli używasz tylko chat_lmstudio.py
```

## Odtworzenie pipeline'u od zera

`models/`, `adapters/`, `data/raw/` i `data/processed/` są celowo
niecommitowane (w pełni odtwarzalne, patrz `.gitignore`; wyjątek:
`data/raw/*_meta.json`, potrzebne przez `check_acts_freshness.py` w CI).

```bash
# 1. Pobranie aktów prawnych z ELI API Sejmu -> data/raw/, data/processed/
.venv/bin/python scripts/download_acts.py                    # wszystkie dziedziny
.venv/bin/python scripts/download_acts.py --domain podatkowe  # tylko jedna dziedzina

# 2. Zbudowanie indeksu RAG -- OSOBNO na dziedzinę
.venv/bin/python scripts/build_rag_index.py --domain kadry_i_place
.venv/bin/python scripts/build_rag_index.py --domain podatkowe

# 2b. [opcjonalnie] Historyczne wersje ustaw -> odpowiadanie "na dany dzień
# w przeszłości" (--as-of / /data w chat.py)
.venv/bin/python scripts/download_acts_history.py --domain kadry_i_place
.venv/bin/python scripts/build_rag_index.py --domain kadry_i_place --include-history

# Wydajność (Krok 63): download_acts*.py parsują PDF-y równolegle w osobnych
# procesach -- liczba dobierana automatycznie do rdzeni i WOLNEJ pamięci
# (~3 GB na proces), nadpisywana przez --workers N (1 = sekwencyjnie);
# pobieranie z API Sejmu zostaje sekwencyjne. build_rag_index.py trzyma
# trwały cache embeddingów (data/processed/embedding_cache.sqlite) i liczy
# tylko fragmenty nowe albo o zmienionej treści (wynik identyczny z pełnym
# przeliczeniem); --no-cache wymusza liczenie wszystkiego od nowa,
# --seed-cache-from-index zasila cache z już zbudowanego indeksu.

# 2c. [opcjonalnie, tylko kadry_i_place] Coroczne rozporządzenia z konkretnymi
# kwotami (minimalne wynagrodzenie, limit ZUS, przeciętne wynagrodzenie GUS)
.venv/bin/python scripts/download_wage_regulations.py
.venv/bin/python scripts/download_zus_limit_regulations.py
.venv/bin/python scripts/download_avg_wage_regulations.py

# 3. [macOS/MLX] Pobranie i konwersja Bielika do MLX (kwantyzacja 4-bit) -> models/
.venv/bin/hf download speakleash/Bielik-11B-v3.0-Instruct
.venv/bin/mlx_lm.convert \
  --hf-path speakleash/Bielik-11B-v3.0-Instruct \
  --mlx-path models/Bielik-11B-v3.0-Instruct-mlx \
  -q

# 4. [opcjonalne, historyczne] Fine-tuning LoRA na własnym zbiorze {train,valid}.jsonl (format `messages`);
#    zbiory treningowe z eksperymentów nie są częścią tego repo, a adaptery nie dały zysku (patrz "Stan projektu")
.venv/bin/mlx_lm.lora \
  --model models/Bielik-11B-v3.0-Instruct-mlx \
  --train \
  --data <katalog_z_train_i_valid_jsonl> \
  --batch-size 2 \
  --iters 100 \
  --mask-prompt \
  --adapter-path adapters/bielik11b-podatkowe-lora
```

## Użycie

```bash
# REKOMENDOWANE (Krok 113): Qwen3.8-27B w LM Studio + automatyczny wybór dziedziny (router Basal).
# `chat_lmstudio.py` dla modeli Qwen sam ustawia --max-tokens 3000 i --reasoning-effort low (Krok 110).
lms load qwen/qwen3.8-27b --context-length 24576 -y && lms server start      # `lms load` NIE startuje serwera API
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000     # router dziedzin (Krok 99-103)
.venv/bin/python scripts/chat_lmstudio.py --domain auto --prompt "Czy gmina może pobrać opłatę reklamową za szyld?"
# `--domain auto` działa tylko z --prompt; w trybie interaktywnym podaj --domain <dziedzina>.
# Uwaga: razem z Qwenem w LM Studio trzymaj tylko Basala (nie ładuj dodatkowo Bielika) i patrz na wolną pamięć (Krok 107).

# Lekki tryb 1 (Krok 117): Gemma-4-26B-A4B w LM Studio -- szybka (ok. 20 s), myślenie wyłączone
# (warianty 4-bit/8-bit mają to samo id modelu: ładuj po kluczu z własnym --identifier i sprawdź `lms ps`):
lms unload --all && lms load gemma-4-26b-a4b-it-mlx@8bit --identifier gemma26-8bit --context-length 24576 -y && lms server start
.venv/bin/python scripts/chat_lmstudio.py --model gemma26-8bit --domain kadry_i_place --max-tokens 3000 \
  --template-kwargs '{"enable_thinking": false}' --prompt "Ile dni urlopu przysługuje po 10 latach pracy?"

# Lekki tryb 2: Bielik (MLX) -- słabsza jakość (patrz ostrzeżenie wyżej), tryb interaktywny, dziedzina domyślna (kadry_i_place)
.venv/bin/python scripts/chat.py

# Wybór dziedziny -- dobiera indeks RAG i domyślny adapter LoRA
.venv/bin/python scripts/chat.py --domain podatkowe

# Pojedyncze pytanie
.venv/bin/python scripts/chat.py --domain podatkowe --prompt "Kiedy przedawnia się zobowiązanie podatkowe?"

# Dla porównania: bez LoRA (sam bazowy model + RAG)
.venv/bin/python scripts/chat.py --domain podatkowe --no-adapter
```

Reszta zachowań: kontekst rozmowy (`/nowy`), upload PDF (`/plik`), stan
prawny na dany dzień (`/data`, `--as-of`). Opis mechanizmów jest w sekcji
„Jak to działa" i w PROGRESS.md.

## Licencje składników i źródła danych

Kod tego repo: Apache-2.0 (plik `LICENSE`). Modeli i wag tu **nie dystrybuujemy** -- pobierasz je sam
z Hugging Face. Licencje poniżej wg kart modeli na HF (odczyt 2026-11-14); to nie jest porada prawna,
przed użyciem komercyjnym sprawdź aktualne warunki.

| składnik | rola | licencja wg HF |
|---|---|---|
| `speakleash/Bielik-11B-v3.1-Instruct` (także v3.0) | tryb "lekki" (lokalnie, MLX/CUDA) | Apache-2.0; repo "gated" -- wymaga zalogowania i akceptacji na HF |
| `Qwen/Qwen3.8-27B` | zalecany model odpowiedzi (LM Studio) | Apache-2.0 |
| `google/gemma-4-26B-A4B-it` | drugi tryb "lekki" | Apache-2.0 |
| `sdadas/mmlw-retrieval-roberta-large` | embeddingi RAG | Apache-2.0 |
| `sdadas/polish-reranker-base-ranknet` | reranker | Apache-2.0 |
| `Remek/basal-2.0-4.5B-MLX-8bit` | router dziedzin | Apache-2.0 |

**Dane prawne:** teksty aktów pochodzą z publicznego API ELI Sejmu (`https://api.sejm.gov.pl/eli`). Akty
normatywne nie są przedmiotem prawa autorskiego (art. 4 pkt 1 ustawy o prawie autorskim i prawach
pokrewnych), ale samo API może mieć własne warunki korzystania -- sprawdź je przy większym użyciu.

**Odpowiedzi systemu nie są poradą prawną.** Treść dziedzin nie została zweryfikowana przez prawnika;
zawsze sprawdzaj cytowane artykuły w źródle (`source_url`).
