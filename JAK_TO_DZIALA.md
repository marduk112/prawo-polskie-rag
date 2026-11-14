# Co robi ten projekt i jak

Krótki opis dla kogoś, kto widzi to repo pierwszy raz. Szczegóły instalacji i użycia są w [README.md](README.md),
a pełny dziennik decyzji i pomiarów w [PROGRESS.md](PROGRESS.md).

## 1. Co to jest

**Lokalny asystent, który odpowiada na pytania o polskie prawo, opierając się na prawdziwych tekstach ustaw.**
Zadajesz pytanie zwykłym językiem („Czy mogę legalnie przebywać w Polsce, gdy czekam na decyzję w sprawie pobytu
czasowego?”), a system znajduje odpowiednie artykuły w ustawach i prosi model językowy, żeby odpowiedział
**wyłącznie na ich podstawie**, z podaniem numeru artykułu i nazwy aktu.

Przykład (prawdziwa odpowiedź z testów, skrócona):

> **Pytanie:** Czy mogę legalnie przebywać w Polsce, gdy czekam na decyzję w sprawie pobytu czasowego?
> **Odpowiedź:** Tak – pobyt cudzoziemca jest uznawany za legalny od dnia złożenia wniosku do dnia, w którym decyzja
> stanie się ostateczna (art. 108 ust. 1 ustawy o cudzoziemcach), pod warunkiem że wniosek złożono w terminie i bez
> braków formalnych. Wyjątek: zawieszenie postępowania na wniosek strony (art. 108 ust. 2).

Co system **robi**:
- obejmuje **35 dziedzin** (kadry i płace, podatki, prawo cywilne, karne, budowlane, cudzoziemcy, przedsiębiorcy,
  rachunkowość i inne; razem 48 aktów prawnych z oficjalnego API Sejmu),
- potrafi odpowiedzieć **„na dany dzień”** (stan prawny z przeszłości), bo trzyma historię wersji ustaw,
- potrafi uwzględnić **wgrany przez Ciebie PDF** (np. umowę) jako osobne źródło, wyraźnie odróżnione od prawa,
- działa **lokalnie** (nic nie wysyła do chmury; modele i indeksy są na Twoim komputerze).

Czego **nie robi**:
- **nie jest poradą prawną** i nie zastępuje prawnika; treść dziedzin nie była weryfikowana przez prawnika,
- nie zna orzecznictwa, interpretacji organów ani większości rozporządzeń wykonawczych (tylko wybrane akty),
- nie wymyśla odpowiedzi spoza przepisów: gdy w znalezionych fragmentach czegoś nie ma, ma to powiedzieć wprost.

## 2. Idea: model nie „wie” prawa, tylko czyta przepisy

Modele językowe potrafią pewnie zmyślać liczby, terminy i numery artykułów. Dlatego system nie polega na pamięci
modelu. Przy każdym pytaniu:

1. **wyszukuje** w bazie prawdziwe fragmenty ustaw (to źródło **faktów**),
2. daje je modelowi jako kontekst z regułami: odpowiadaj tylko na tej podstawie, cytuj artykuły, przyznaj się do
   niewiedzy, rozróżniaj dosłowny zapis od własnego wniosku,
3. model **formułuje** odpowiedź (to jego rola: język i uporządkowanie, nie źródło faktów).

To technika RAG (*retrieval-augmented generation*, generowanie wspomagane wyszukiwaniem).

## 3. Przygotowanie danych (robione raz, offline)

```
API Sejmu (ELI) ──► PDF ustawy ──► tekst ──► podział na artykuły ──► fragmenty ──► wektory ──► indeks dziedziny
 download_acts.py                  oczyszczanie   (z numeracją, np.     ~900 zn.      embeddingi   rag_index_<dziedzina>
 download_acts_history.py          i naprawa      art. 67¹⁸, 4a)        z zakładką    (sdadas/mmlw)   .npy + metadane
                                   numeracji                            150 zn.
```

- **Źródło:** oficjalne API Sejmu (`api.sejm.gov.pl/eli`). Lista aktów i ich podział na dziedziny jest w
  [`data/acts_config.json`](data/acts_config.json) (jeden akt może należeć do kilku dziedzin).
- **Artykuły:** tekst jednolity z PDF jest dzielony na artykuły; skrypt naprawia typowe błędy PDF (indeksy górne w
  numerach, załączniki jako osobne fragmenty).
- **Historia:** dla wersji ustaw sprzed lat pobierane są wcześniejsze teksty jednolite, żeby dało się pytać „na dzień X”
  (`download_acts_history.py`, indeks z `--include-history`).
- **Fragmenty:** długie artykuły są cięte na okna po ok. 900 znaków z zakładką 150 znaków, żeby trafić we właściwy ustęp.
- **Wektory:** każdy fragment zamieniany jest na wektor modelem `sdadas/mmlw-retrieval-roberta-large`.
  Każda dziedzina ma **własny indeks** (`data/processed/rag_index_<dziedzina>.npy`), budowany i mierzony osobno.
- **Aktualność:** `check_acts_freshness.py` (cotygodniowy workflow w CI) sprawdza w API, czy ustawa nie została
  uchylona i czy nie pojawił się nowszy tekst jednolity.

## 4. Co się dzieje przy pytaniu (online)

```
pytanie
  │
  ├─ (0) dziedzina ........ podana ręcznie (--domain) albo wybrana przez router (Basal, --domain auto);
  │                         przy niskiej pewności routera (< 0,85) przeszukiwane są dwie dziedziny
  │
  ├─ (1) rozszerzenie ..... krótkie wywołanie modelu zamienia potoczne pytanie na język zbliżony do ustaw
  │                         („zakupy prywatne” → „wydatki niezwiązane z działalnością gospodarczą”)
  │
  ├─ (2) wyszukiwanie ..... a) podobieństwo wektorów → ok. 30 kandydatów
  │                         b) reranker (cross-encoder sdadas/polish-reranker-base-ranknet) ocenia je precyzyjniej
  │                         c) fuzja rang RRF łączy oba rankingi, żeby żaden nie wypychał ważnego przepisu
  │                         → domyślnie 8 najlepszych fragmentów
  │
  ├─ (3) kontekst ......... dla trafionych artykułów wycinek wokół właściwego ustępu (długie artykuły nie są ucinane
  │                         „od początku”); przy pytaniu o datę adnotacje „stan prawny”
  │
  ├─ (4) prompt ........... reguły systemowe (patrz niżej) + kontekst + pytanie (+ historia rozmowy, + fragmenty
  │                         wgranego PDF jako osobne źródło)
  │
  └─ (5) odpowiedź ........ model językowy (zalecany Qwen3.8-27B w LM Studio) → odpowiedź z cytowaniem artykułów
```

**Reguły z promptu systemowego** ([`scripts/prompt.py`](scripts/prompt.py)): odpowiadaj wyłącznie na podstawie
dostarczonych fragmentów; jeśli ich brak, napisz to wprost i nie zgaduj; cytuj numer artykułu i nazwę aktu; nie jesteś
poradą prawną; przy pytaniu o datę z przeszłości odpowiadaj wersją z tej daty; odróżniaj dosłowny zapis od własnego wniosku.

**Dlaczego tyle etapów wyszukiwania?** Każdy został dodany, bo samo podobieństwo wektorów zawodziło w konkretnych
przypadkach (potoczny język nie trafiał w ustawę; reranker sam spychał krótkie, ogólne przepisy). Uzasadnienia i
pomiary są w PROGRESS.md (reranking i fuzja rang: Kroki 11 i 12, rozszerzanie zapytania: Krok 21 i kolejne).

## 5. Modele

| rola | model | uwagi |
|---|---|---|
| **odpowiedzi (zalecany)** | Qwen3.8-27B w LM Studio (`chat_lmstudio.py`) | najlepsze wyniki w pomiarach, wymaga komputera z dużą pamięcią |
| odpowiedzi, tryb lekki | Bielik 11B v3.1 (MLX na Apple Silicon: `chat.py`; CUDA: `chat_cuda.py`) | słabsza jakość, ale polski model i mniejsze wymagania |
| odpowiedzi, tryb lekki 2 | Gemma-4-26B-A4B w LM Studio | szybsza, lepsza od Bielika w pomiarach |
| embeddingi i reranker | `sdadas/mmlw-retrieval-roberta-large`, `sdadas/polish-reranker-base-ranknet` | liczone lokalnie |
| router dziedzin | Basal 4.5B (`router.py`) | wybiera dziedzinę, gdy nie podasz jej ręcznie |

Adaptery **LoRA** (douczanie Bielika) były wcześniejszym pomysłem na „styl” odpowiedzi. Pomiary (PROGRESS.md, Kroki
79, 112, 121, 122) nie wykazały zysku względem samej bazy z RAG, więc domyślnie nie są używane; zostały w repo jako
kontekst historyczny.

## 6. Jak mierzona jest jakość

Każda zmiana w wyszukiwaniu czy routerze jest sprawdzana na zestawach regresyjnych, a nie „na oko”:

- **Wyszukiwanie** (`check_expand_query_regressions.py`, [`data/expand_query_reference_cases.json`](data/expand_query_reference_cases.json)):
  192 pytania, każde z oczekiwanym artykułem; sprawdza, czy ten artykuł jest w pierwszych 8 wynikach.
- **Router** (`router.py --battery`, pliki `data/routing_*`): czy pytanie trafia do właściwej dziedziny; zbiory
  „niewidziane” służą do wykrywania przeuczenia na własnych przypadkach.
- **Pełne odpowiedzi** (`check_qwen_answer_regressions.py`, [`data/qwen_answer_reference_cases.json`](data/qwen_answer_reference_cases.json)):
  46 przypadków; skrypt sprawdza proste kryteria, a odpowiedzi czyta się i porównuje z tekstem artykułów ręcznie.
- **Testy jednostkowe** (`pytest`) dla parsowania, indeksowania, routera i skryptów pomocniczych.

Wynik pomiarów na dzień 2026-11-14 (szczegóły i zastrzeżenia w PROGRESS.md): wyszukiwanie z ręcznie podaną dziedziną
trafia oczekiwany artykuł w prawie wszystkich przypadkach (kilka znanych, opisanych wyjątków); router trafia właściwą
dziedzinę jako pierwszą w ok. 90% zwykłych pytań i w ok. 98%, gdy liczy się także druga sugestia.

## 7. Znane ograniczenia

- **Brak weryfikacji prawnika.** Przypadki testowe pisał autor na podstawie treści artykułów, nie prawnik.
- **Model potrafi się mylić**, zwłaszcza lekkie modele; zdarza się upraszczanie (np. pominięcie wyjątku). Zawsze
  sprawdzaj cytowany artykuł w źródle (`source_url` przy każdym wyniku).
- **Zakres:** tylko wybrane ustawy (nie wszystkie rozporządzenia, brak orzecznictwa). Kwoty i stawki bywają w
  rozporządzeniach spoza bazy; system ma to wtedy przyznać zamiast zgadywać.
- **Router myli dziedziny** w ok. 10% pytań (zwłaszcza na styku cywilne/konsumenckie, podatki); część pytań jest
  naprawdę niejednoznaczna.
- **Aktualność** zależy od ostatniego pobrania ustaw; nowelizacje trzeba doładować (`download_acts.py`).

## 8. Mapa repo

| ścieżka | po co |
|---|---|
| `scripts/download_acts*.py`, `build_rag_index.py` | pobranie ustaw i budowa indeksów |
| `scripts/rag_search.py`, `prompt.py`, `article_excerpt.py` | wyszukiwanie, prompty, kontekst |
| `scripts/router.py` | router dziedzin |
| `scripts/chat_lmstudio.py`, `chat.py`, `chat_cuda.py` | interfejsy rozmowy (LM Studio, MLX, CUDA) |
| `scripts/check_*.py` | zestawy regresyjne i kontrola aktualności |
| `scripts/export_public.py` | eksport czystego drzewa do publikacji |
| `data/acts_config.json` | lista aktów i dziedzin |
| `data/expand_query_reference_cases.json`, `data/routing_*`, `data/qwen_answer_reference_cases.json` | zbiory testowe |
| `tests/` | testy jednostkowe |
| `PROGRESS.md` | dziennik decyzji, pomiarów i pułapek |

## 9. Jak to uruchomić

Najkrótsza droga (bez modelu językowego, tylko wyszukiwanie w ustawach) i pełna instalacja są w
[README.md](README.md): sekcje „Szybki start” i „Instalacja”.
