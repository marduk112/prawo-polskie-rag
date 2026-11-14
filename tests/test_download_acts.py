"""Testy dla scripts/download_acts.py -- ekstrakcji i czyszczenia
tekstu artykułów z PDF-ów. Górna część pliku to testy jednostkowe na
małych, ręcznie skonstruowanych fragmentach tekstu (bez zależności od
pobranych danych). Testy na dole (oznaczone `skipif`) to regresje
konkretnych błędów znalezionych w audycie (PROGRESS.md, krok 12/13) --
wymagają lokalnie pobranej bazy (`data/raw/`, `data/processed/`),
patrz README.md, sekcja "Odtworzenie pipeline'u".
"""

import json
from pathlib import Path

import pytest

import download_acts as da

from download_acts import (
    ARTICLE_SPLIT_RE,
    HEADER_FOOTER_PATTERNS,
    apply_flattened_superscript_corrections,
    ParsePool,
    align_superscript_corrections,
    build_version_articles,
    compute_version_windows,
    default_parse_workers,
    find_ustep_superscript_corrections_pdfplumber,
    convert_bracket_superscripts,
    split_annexes,
    fix_font_mojibake,
    odd_char_rate_per_1000,
    SCAN_QUALITY_MAX_ODD_PER_1000,
    fix_stray_space_before_period,
    parse_articles,
    process_act_version,
    recover_letter_superscript_headers,
    repair_flattened_superscript_labels,
    recover_midtext_superscript_headers,
    strip_announcement_preamble,
    strip_not_yet_in_force_text,
    strip_page_footnotes,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# --- ARTICLE_SPLIT_RE / parse_articles ---


def test_article_split_re_handles_polish_letter_suffix():
    """Regresja: '[a-z]{0,3}' był tylko ASCII, więc np. 'Art. 22ł.'
    (litera 'ł') nigdy nie pasował -- patrz PROGRESS.md, krok 12,
    znalezisko 4."""
    m = ARTICLE_SPLIT_RE.match("Art. 22ł. Treść artykułu.")
    assert m is not None
    assert m.group(1) == "22ł"


def test_recover_letter_superscript_headers_splits_glued_article():
    """Regresja Krok 60: "Art. 5b¹." (pypdf: "Art. 5b1." albo "Art. 5a 1.")
    nie trafiał na ARTICLE_SPLIT_RE, więc cały artykuł był wchłaniany przez
    poprzedni (Prawo energetyczne, 17 nagłówków)."""
    text = (
        "Art. 5a. 1. Sprzedawca z urzędu. Art. 5a 1. 1. Agregacja odbywa się.\n"
        "Art. 5b1. 1. Umowa. Zob. art. 5b ust. 1.\nArt. 5c. 1. Dodatek."
    )
    articles = parse_articles(recover_letter_superscript_headers(text))
    assert [a["article"] for a in articles] == ["5a", "5a¹", "5b¹", "5c"]
    assert "Agregacja" not in articles[0]["text"]
    assert articles[1]["text"].startswith("Art. 5a¹. 1. Agregacja")


def test_recover_letter_superscript_headers_ignores_normal_headers_and_citations():
    text = "Art. 5a. 1. Treść. Zgodnie z art. 5a ust. 1 oraz Art. 111. 2. Inny."
    assert recover_letter_superscript_headers(text) == text


def test_strip_page_footnotes_cuts_from_last_marker_line():
    """Regresja Krok 63: przypis dolny ("15) Zmiany tekstu jednolitego...")
    trafiał do środka artykułu, a jego numer wyglądał jak punkt wyliczenia
    (adapter zacytował nieistniejący "art. 389 pkt 15" Prawa wodnego).
    Obcinamy od OSTATNIEJ linii z markerem -- zwykły punkt "15)" wyżej na
    stronie zostaje."""
    lines = [
        "Art. 389. Pozwolenie wodnoprawne jest wymagane na:",
        "15) coś w treści, zwykły punkt;",
        "5) wprowadzanie substancji;",
        "15) Zmiany tekstu jednolitego wymienionej ustawy zostały ogłoszone w Dz. U. z 2024 r. poz. 731 i",
        "1222 oraz z 2025 r. poz. 31.",
    ]
    assert strip_page_footnotes(lines, "15") == lines[:3]


def test_strip_page_footnotes_footnote_with_own_enumeration():
    """Regresja Krok 63: przypis z 1. strony sam jest wyliczeniem ("1)
    Niniejsza ustawa wdraża: 1) dyrektywę..."), więc cięcie od ostatniej
    linii "1)" zostawiało jego nagłówek w art. 1 -- dopasowanie po początku
    treści przypisu (z pdfplumber) tnie od właściwej linii."""
    lines = [
        "Art. 1. Ustawa reguluje gospodarowanie wodami.",
        "1) Niniejsza ustawa w zakresie swojej regulacji wdraża:",
        "1) dyrektywę Rady 91/271/EWG;",
        "2) dyrektywę Rady 91/676/EWG.",
    ]
    assert strip_page_footnotes(lines, ("1", "Niniejszaustawa")) == lines[:1]
    # sam numer (bez treści) -- dawne zachowanie: od ostatniej linii "1)"
    assert strip_page_footnotes(lines, "1") == lines[:2]


def test_strip_page_footnotes_without_marker_or_match_is_noop():
    lines = ["Art. 1. Treść.", "2) punkt"]
    assert strip_page_footnotes(lines, None) == lines
    assert strip_page_footnotes(lines, "7") == lines


def test_default_parse_workers_is_bounded():
    assert default_parse_workers(1) == 1
    assert 1 <= default_parse_workers() <= (__import__("os").cpu_count() or 1)


def test_parse_pool_single_worker_runs_inline():
    """workers=1 -- bez puli procesów, wynik identyczny ze ścieżką sekwencyjną
    (tu na pustym PDF-ie z pypdf: zero artykułów, bez wyjątku)."""
    import io

    from pypdf import PdfWriter

    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    with ParsePool(1) as pool:
        assert pool.submit(buf.getvalue(), "pusty").result() == []


def test_align_superscript_corrections_with_count_mismatch():
    """Regresja Krok 64: przy niezgodnej liczbie nagłówków korekta była
    pomijana w całości, więc 22¹ zostawał jako "221" obok prawdziwego art.
    221. Dopasowanie lokalne poprawia zgodne pary mimo rozjazdu w innym
    miejscu (tu: pypdf ma artykuł "30", którego pdfplumber nie wykrył)."""
    pypdf = ["22", "221", "222", "23", "30", "221"]
    detected = ["22", "22¹", "22²", "23", "221"]
    assert align_superscript_corrections(pypdf, detected) == {1: "22¹", 2: "22²"}


def test_align_superscript_corrections_no_superscripts_no_fixes():
    assert align_superscript_corrections(["1", "2", "3"], ["1", "3"]) == {}


def test_align_superscript_corrections_bracket_letter_suffix():
    """Notacja nawiasowa z literą (KP "Art. 18[3d]." -> "18³d"), spłaszczona
    przez pypdf do "183d"."""
    assert align_superscript_corrections(["18", "183d", "19", "20"], ["18", "18³d", "20"]) == {1: "18³d"}


def test_parse_articles_basic_split():
    text = "Preambuła.\nArt. 1. Treść pierwszego.\nArt. 2. Treść drugiego."
    articles = parse_articles(text)
    assert [a["article"] for a in articles] == ["1", "2"]
    assert articles[0]["text"].startswith("Art. 1.")


# --- strip_not_yet_in_force_text ---


def test_strip_not_yet_in_force_removes_angle_bracket_span():
    text = "Przed. <Art. 5. Nieobowiązujący tekst.> Po."
    result = strip_not_yet_in_force_text(text)
    assert "Nieobowiązujący" not in result
    assert "Przed." in result and "Po." in result


def test_strip_not_yet_in_force_keeps_square_bracket_content():
    text = "Wciąż obowiązuje: [stary ale ważny tekst]."
    result = strip_not_yet_in_force_text(text)
    assert "stary ale ważny tekst" in result
    assert "[" not in result and "]" not in result


def test_strip_not_yet_in_force_unmatched_bracket_left_untouched(capsys):
    """Bezpiecznik: niesparowany '<' (bez ryzyka pochłonięcia
    poprawnego tekstu) zostaje, zamiast go zjeść -- z ostrzeżeniem."""
    text = "Tekst z niesparowanym < nawiasem bez zamknięcia."
    result = strip_not_yet_in_force_text(text)
    assert "niesparowanym" in result
    assert "UWAGA" in capsys.readouterr().out


def test_strip_not_yet_in_force_removes_orphan_angle_bracket(capsys):
    """Prawo o ruchu drogowym (Krok 45): art. 53a/65l kończyły się
    osieroconym '>' bez odpowiadającego '<' -- pojedynczy znak-artefakt
    ekstrakcji PDF, nie brakujący fragment tekstu. Cokolwiek zostaje po
    podstawieniu właściwych par '< >' jest z definicji takim artefaktem
    i ma zostać usunięte, żeby nie zostawiać widocznego '>' w treści
    artykułu."""
    text = "Art. 53a. Dopuszcza się coś tam.>\nArt. 54. Inny przepis."
    result = strip_not_yet_in_force_text(text)
    assert "<" not in result and ">" not in result
    assert "Dopuszcza się coś tam." in result


def test_strip_not_yet_in_force_handles_nested_square_and_angle_brackets():
    """Regresja Krok 46/47 (`prawo_upadlosciowe`, art. 216a@2017-11-22):
    zagnieżdżone '[...[...] <...>...]' łamały dwa niezależne, niezachłanne
    podstawienia regex -- `\\[(.*?)\\]` dopasowywało od zewnętrznego '['
    do PIERWSZEGO napotkanego ']' (czyli wewnętrznego), zostawiając
    osierocone '[' i ']' dosłownie w tekście: `[albo podpisem zaufanym .]`
    zamiast czystego `albo podpisem zaufanym .` -- potwierdzone wprost w
    przetworzonych danych tego kroku."""
    text = (
        "Przed. [1. Tekst ważny [stare słowo] <nowe słowo, jeszcze "
        "nieobowiązujące>.] Po."
    )
    result = strip_not_yet_in_force_text(text)
    assert "[" not in result and "]" not in result
    assert "<" not in result and ">" not in result
    assert "stare słowo" in result
    assert "nowe słowo" not in result
    assert "Przed." in result and "Po." in result


def test_strip_not_yet_in_force_handles_adjacent_articles_across_bracket_boundary():
    """Regresja Krok 46/47 (`prawo_upadlosciowe`, art. 241/242@2019-02-21):
    gdy jeden zagnieżdżony nawias w art. 241 zostawiał osierocone '[' tuż
    przed nagłówkiem SĄSIEDNIEGO art. 242, drugie podstawienie regex
    (`\\[(.*?)\\]`) parowało ten osierocony '[' z pierwszym napotkanym ']'
    GDZIEKOLWIEK dalej w tekście -- wchłaniając przy okazji cały nagłówek
    i treść art. 242, który znikał jako osobny wpis dla tej wersji
    historycznej. Nowy parser dopasowuje pary '< >' pozycyjnie (nie
    regexem), a '[ ]' usuwa znak-po-znaku bez dopasowywania par w ogóle --
    więc taki efekt domina nie może wystąpić."""
    text = (
        "Art. 241. [1. Tekst art. 241 [stara wersja] <nowa wersja>.]\n"
        "Art. 242. Zupełnie inny, niepowiązany przepis."
    )
    result = strip_not_yet_in_force_text(text)
    assert "Art. 242. Zupełnie inny, niepowiązany przepis." in result
    assert "[" not in result and "]" not in result
    assert "nowa wersja" not in result
    assert "stara wersja" in result


def test_strip_not_yet_in_force_closes_unclosed_future_article_at_next_header(capsys):
    """Regresja Krok 65 (POŚ 2016 art. 286, ruch drogowy 2017 art. 80b):
    Kancelaria zgubiła w PDF-ie zamykające '>' przyszłej wersji artykułu,
    więc zostawała ona obok obowiązującej jako duplikat etykiety. Blok
    zamykamy na najbliższym nagłówku -- następny artykuł zostaje."""
    text = (
        "[Art. 286. 1. Obowiązująca wersja.]\n"
        "<Art. 286. 1. Przyszła wersja bez zamknięcia.\n"
        "Art. 286a. Obowiązujący sąsiad.\n"
        "Art. 287. Coś <nowe> dalej."
    )
    result = strip_not_yet_in_force_text(text)
    assert "Przyszła wersja" not in result
    assert "Obowiązująca wersja." in result
    assert "Art. 286a. Obowiązujący sąsiad." in result
    assert "nowe" not in result and "Coś  dalej." in result
    assert [a["article"] for a in parse_articles(result)] == ["286", "286a", "287"]


def test_strip_not_yet_in_force_keeps_article_opened_with_angle_but_closed_with_square():
    """Prawo energetyczne 2024, art. 5ab: obowiązująca wersja otwarta
    literówką "<" i zamknięta "]", dalej osobna przyszła wersja w "< >".
    Obowiązująca ma zostać, przyszła zniknąć."""
    text = (
        "<Art. 5ab. 1. Obowiązujący tekst (gaz i prąd).]\n"
        "<Art. 5ab. 1. Przyszły tekst (tylko gaz).>\n"
        "Art. 5ac. Sąsiad."
    )
    result = strip_not_yet_in_force_text(text)
    assert [a["article"] for a in parse_articles(result)] == ["5ab", "5ac"]
    assert "Obowiązujący tekst" in result and "Przyszły tekst" not in result


def test_strip_not_yet_in_force_unclosed_bracket_midtext_not_extended():
    """Niesparowany '<' w środku tekstu (nie na nagłówku artykułu) nadal
    nie wycina niczego -- bezpiecznik z Kroku 45 bez zmian."""
    text = "Art. 5. Tekst <niesparowany ciąg dalszy.\nArt. 6. Dalej."
    result = strip_not_yet_in_force_text(text)
    assert "niesparowany ciąg dalszy." in result


def test_header_footer_patterns_drop_dotted_date_stamp_only_as_whole_line():
    """Regresja Krok 65: stempel daty wygenerowania tekstu w formacie
    "09.05.2019" (co stronę, teksty ~2017-2021) trafiał do treści
    artykułów. Data wewnątrz zdania (np. dyrektywa UE) zostaje."""
    def dropped(line):
        return any(p.match(line) for p in HEADER_FOOTER_PATTERNS)

    assert dropped("09.05.2019")
    assert not dropped("(Dz. Urz. UE L 189 z 18.07.2002, str. 12)")
    assert not dropped("31.12.2020 r.")


def test_build_version_articles_id_suffix_for_window_without_start():
    """Regresja Krok 65: okno bez znanego początku dostawało id "@current",
    powtarzające się między kilkoma najstarszymi oknami tej samej ustawy."""
    def ids(valid_from, valid_to):
        recs = build_version_articles(
            {"publisher": "DU", "year": 2000, "position": 1}, "pb", "DU/1994/414", "Prawo budowlane",
            valid_from, valid_to, "DU/2000/1126", ["budowlane"], [{"article": "54", "text": "Art. 54. X"}],
            "U", "f.pdf",
        )
        return recs[0]["id"]

    assert ids(None, "2003-11-21") == "pb_art_54@do_2003-11-21"
    assert ids("2003-11-21", "2006-08-18") == "pb_art_54@2003-11-21"
    assert ids(None, None) == "pb_art_54@current"


# --- strip_announcement_preamble ---


ANNOUNCEMENT_TEXT = (
    "OBWIESZCZENIE\n"
    "2. Podany w załączniku tekst jednolity ustawy nie obejmuje: art. 10 ..., które stanowią:\n"
    "„Art. 10. W ustawie ... wprowadza się zmiany.”\n"
    "Marszałek Sejmu: M. Kidawa-Błońska\n"
    "j) Zmiany wymienionej ustawy zostały ogłoszone w ...\n"
    "Załącznik do obwieszczenia Marszałka Sejmu Rzeczypospolitej\n"
    "Polskiej z dnia 10 listopada 2015 r. (poz. 2008)\n"
    "USTAWA\n"
    "z dnia 10 października 2002 r.\n"
    "Art. 1. Pierwszy.\n"
    "Art. 10. (pominięty)"
)


def test_strip_announcement_preamble_drops_quoted_provisions():
    """Regresja Krok 65 (ustawa o min. wynagrodzeniu, okno 2015): przepisy
    przytoczone w obwieszczeniu jako nieobjęte tekstem jednolitym były
    brane za artykuły ustawy -- podwojone art. 10-24."""
    articles = parse_articles(strip_announcement_preamble(ANNOUNCEMENT_TEXT))
    assert [a["article"] for a in articles] == ["1", "10"]
    assert articles[1]["text"] == "Art. 10. (pominięty)"


def test_strip_announcement_preamble_keeps_article_emitted_before_annex_header():
    """Prawo lotnicze 2006: na dwukolumnowej stronie pypdf wypisuje art. 1
    PRZED nagłówkiem załącznika, a zepsute kodowanie daje "Za∏àcznik" /
    "Marsza∏ek" -- tniemy za podpisem, nie na nagłówku."""
    text = (
        "Art. 3. Cytat z obwieszczenia.\n"
        "Marsza∏ek Sejmu: M. Jurek\n"
        "Art. 1. 1. Przepisy Prawa lotniczego.\n"
        "Za∏àcznik do obwieszczenia Marsza∏ka Sejmu\n"
        "(poz. 696)\n"
        "USTAWA\n"
        "Art. 2. Definicje."
    )
    assert [a["article"] for a in parse_articles(strip_announcement_preamble(text))] == ["1", "2"]


def test_strip_announcement_preamble_noop_without_annex_header():
    text = "Marszałek Sejmu: X\nArt. 1. Zwykły tekst ujednolicony."
    assert strip_announcement_preamble(text) == text


# --- apply_flattened_superscript_corrections ---


def test_apply_flattened_superscript_corrections_fixes_matching_context():
    clean_text = "audiowizualne (w tym filmowe).\n21. Ochroną objęty może być wyłącznie sposób wyrażenia."
    corrections = [(["(w", "tym", "filmowe)."], "21", "2¹")]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "2¹. Ochroną" in result
    assert "21. Ochroną" not in result


def test_apply_flattened_superscript_corrections_tolerates_line_break_whitespace():
    """Kluczowy przypadek Kroku 48: `pypdf` wstawia '\\n' dokładnie tam,
    gdzie `pdfplumber` (źródło kontekstu) nie widzi żadnego jawnego
    znaku -- dopasowanie musi przetrwać tę różnicę w białych znakach."""
    clean_text = "za wynagrodzeniem.\n§ 11. Zatrudnienie w warunkach określonych."
    corrections = [(["za", "wynagrodzeniem.", "§"], "11", "1¹")]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "§ 1¹. Zatrudnienie" in result


def test_apply_flattened_superscript_corrections_applies_in_document_order():
    """Dwie korekty z identycznym krótkim kontekstem trafiają we
    właściwej kolejności, bo każde zastosowanie 'zużywa' jedno
    dopasowanie (pierwsze pozostałe), tak jak w kolejności występowania
    w dokumencie."""
    clean_text = "§ 1. Pierwsze. § 11. Drugie. § 11. Trzecie."
    corrections = [(["Pierwsze.", "§"], "11", "1¹"), (["Drugie.", "§"], "11", "2¹")]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "Pierwsze. § 1¹. Drugie. § 2¹. Trzecie." in result


def test_apply_flattened_superscript_corrections_skips_missing_context_safely(capsys):
    clean_text = "Zupełnie inny tekst bez żadnego dopasowania."
    corrections = [(["nigdy", "nieobecne", "słowa"], "21", "2¹")]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert result == clean_text
    assert "UWAGA" in capsys.readouterr().out


def test_apply_flattened_superscript_corrections_handles_terminator_other_than_period():
    """Krok 49: cytaty w prozie nie zawsze kończą się kropką (w
    przeciwieństwie do nagłówków ustępów) -- przecinek, myślnik zakresu
    czy koniec zdania przed kolejnym słowem też muszą działać, dzięki
    negatywnemu lookahead `(?!\\d)` zamiast wymogu konkretnego znaku."""
    clean_text = "Przepisy art. 1291, art. 1361 i art. 1363 stosuje się odpowiednio."
    corrections = [
        (["Przepisy", "art."], "1291", "129¹"),
        (["art.", "1291,", "art."], "1361", "136¹"),
        (["i", "art."], "1363", "136³"),
    ]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "art. 129¹, art. 136¹ i art. 136³ stosuje się" in result


def test_apply_flattened_superscript_corrections_neighbor_context_not_invalidated():
    """Regresja Kroku 49 (znaleziona przy okazji naprawy cytatów, ale
    dotyczyła też Kroku 48): naprawienie WCZEŚNIEJSZEJ korekty nie może
    zepsuć "odcisku palca" NASTĘPNEJ, jeśli ten odcisk akurat cytuje
    spłaszczony numer, który właśnie poprawiliśmy. Potwierdzony
    rzeczywisty przypadek: `prawo_wlasnosci_przemyslowej`, "art. 1291,
    art. 1361" -- kontekst drugiej korekty to dosłownie '1291,' (jeszcze
    niepoprawione w momencie WYKRYCIA), więc dopasowanie musi być zawsze
    liczone względem ORYGINALNEGO tekstu, nie względem tekstu już
    zmodyfikowanego przez poprzednią korektę."""
    clean_text = "w art. 1291, art. 1361 wydaje decyzję."
    corrections = [
        (["w", "art."], "1291", "129¹"),
        (["w", "art.", "1291,", "art."], "1361", "136¹"),
    ]
    result = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "w art. 129¹, art. 136¹ wydaje decyzję." in result


# --- recover_midtext_superscript_headers ---


def test_recover_midtext_superscript_headers_merges_split_number():
    text = "koniec zdania. Art. 22 3. Treść nowego artykułu."
    result = recover_midtext_superscript_headers(text)
    assert "Art. 223." in result
    assert "Art. 22 3." not in result


# --- fix_stray_space_before_period ---


def test_fix_stray_space_before_period():
    text = "Art. 52zb . Treść artykułu."
    result = fix_stray_space_before_period(text)
    assert "Art. 52zb." in result
    assert "Art. 52zb ." not in result


# --- compute_version_windows (wersjonowanie czasowe, "as-of", Krok 18) ---
# Testy czystej funkcji (bez HTTP) na ręcznie skonstruowanych metadanych --
# atrapy fragmentów prawdziwych odpowiedzi ELI API zebranych w spike'u
# walidacyjnym (PROGRESS.md, Krok 18).


def _fake_announcement_meta(publisher, year, position, legal_status_date, expiration_date, announcement_date=None):
    return {
        "publisher": publisher,
        "year": year,
        "pos": position,
        "legalStatusDate": legal_status_date,
        "expirationDate": expiration_date,
        "announcementDate": announcement_date,
        "ELI": f"{publisher}/{year}/{position}",
    }


def test_compute_version_windows_normal_chronological_chain():
    announcements = [
        _fake_announcement_meta("DU", 2020, 2207, "2020-11-12", "2024-12-03"),
        _fake_announcement_meta("DU", 2018, 2177, "2018-11-08", "2020-12-10"),
        _fake_announcement_meta("DU", 2024, 1773, "2024-11-27", None),
    ]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    # DU/2024/1773 (expirationDate=None, wciąż oficjalnie obowiązujące) NIE
    # trafia do "past" -- opisywałoby ten sam, wciąż otwarty okres co wpis
    # "is_current" (regresja realnego przypadku znalezionego przy smoke
    # teście na ustawie o PPK, patrz docstring compute_version_windows).
    assert [w["valid_from"] for w in past] == ["2018-11-08", "2020-11-12"]
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-11-27"  # najnowsze wciąż otwarte (expirationDate=None) -> legalStatusDate
    assert current["valid_to"] is None


def test_compute_version_windows_sorts_by_announcement_date_when_legal_status_missing():
    """Regresja znaleziska 1 z Kroku 18: dwa najstarsze wpisy bez
    legalStatusDate, w API zwrócone w BŁĘDNEJ (odwrotnej) kolejności
    chronologicznej -- musimy sięgnąć po announcementDate (a NIE
    expirationDate, który dla starego, długo obowiązującego wpisu potrafi
    wypaść PO legalStatusDate kolejnego wpisu i fałszywie go wyprzedzić --
    sprawdzone na tych samych, realnych wartościach co w API: DU/2009/1585
    ma expirationDate 2013-12-04, czyli PO legalStatusDate DU/2013/1442
    (2013-10-16), mimo że DU/2009/1585 jest wcześniejszy). Dane -- dokładnie
    to, co zwróciło żywe ELI API dla ustawy systemowej, zweryfikowane przy
    tej poprawce."""
    announcements = [
        _fake_announcement_meta("DU", 2009, 1585, None, "2013-12-04", announcement_date="2009-11-10"),
        _fake_announcement_meta("DU", 2007, 74, None, "2009-11-10", announcement_date="2007-01-08"),
        _fake_announcement_meta("DU", 2013, 1442, "2013-10-16", "2015-01-22", announcement_date="2013-10-24"),
    ]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert [w["position"] for w in past] == [74, 1585, 1442]


def test_compute_version_windows_latest_still_open():
    """Jedyne obwieszczenie wciąż otwarte (expirationDate=None) -> zero
    wpisów w 'past' (nic nie trzeba pobierać jako historię), a granica
    bieżącej wersji to jego legalStatusDate."""
    announcements = [_fake_announcement_meta("DU", 2024, 1773, "2024-11-27", None)]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert past == []
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-11-27"


def test_compute_version_windows_latest_already_expired_uses_expiration_date():
    """Najnowsze obwieszczenie ma już swój expirationDate (zostało
    wyprzedzone przez nowelizacje widoczne tylko w żywym tekście 'U') ->
    granica bieżącej wersji to expirationDate, nie legalStatusDate."""
    announcements = [_fake_announcement_meta("DU", 2020, 2207, "2020-11-12", "2024-12-03")]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert len(past) == 1  # ma expirationDate ustawione -> to prawdziwa, zamknięta przeszła wersja
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-12-03"


def test_compute_version_windows_no_announcements():
    windows = compute_version_windows({}, [])
    assert [w for w in windows if not w.get("is_current")] == []
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] is None
    assert current["valid_to"] is None


@pytest.mark.skipif(
    not (RAW_DIR / "ustawa_o_ppk_meta.json").exists(),
    reason="wymaga sieci (żywe zapytanie do ELI API) i lokalnej bazy dla porównania -- pomijane offline",
)
@pytest.mark.slow
def test_process_act_version_round_trip_on_real_announcement():
    """Smoke test end-to-end na jednym, realnym, małym obwieszczeniu (ustawa
    o PPK ma tylko 4 wersje w historii -- najlżejsza z 7 ustaw)."""
    source = {"publisher": "DU", "year": 2020, "position": 1342}
    articles = process_act_version(
        source,
        "ustawa_o_ppk",
        "DU/2018/2215",
        "Ustawa z dnia 4 października 2018 r. o pracowniczych planach kapitałowych",
        "2020-07-06",
        "2023-01-08",
        "DU/2020/1342",
    )
    assert len(articles) > 0
    a = articles[0]
    assert a["act_short"] == "ustawa_o_ppk"
    assert a["eli"] == "DU/2018/2215"  # akt bazowy, NIE obwieszczenie
    assert a["act_title"].startswith("Ustawa z dnia 4 października 2018")  # akt bazowy, NIE tytuł obwieszczenia
    assert a["announcement_eli"] == "DU/2020/1342"
    assert a["valid_from"] == "2020-07-06"
    assert a["valid_to"] == "2023-01-08"
    assert a["id"].endswith("@2020-07-06")


# --- Regresje realnych błędów z audytu (PROGRESS.md, krok 12/13) ---
# Wymagają lokalnej bazy (data/raw/, data/processed/) -- pomijane, jeśli
# jej nie ma (np. świeży checkout bez uruchomionego download_acts.py).


@pytest.mark.skipif(
    not (RAW_DIR / "kodeks_pracy.pdf").exists(),
    reason="wymaga data/raw/kodeks_pracy.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
@pytest.mark.slow
def test_find_article_numbers_matches_parsed_count_for_kodeks_pracy():
    from download_acts import find_article_numbers_pdfplumber, pdf_to_clean_text

    pdf_bytes = (RAW_DIR / "kodeks_pracy.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    articles = parse_articles(clean_text)
    detected = find_article_numbers_pdfplumber(pdf_bytes)
    assert len(detected) == len(articles)
    assert "11¹" in detected


@pytest.mark.skipif(
    not (RAW_DIR / "prawo_budowlane.pdf").exists(),
    reason="wymaga data/raw/prawo_budowlane.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
@pytest.mark.slow
def test_find_article_numbers_detects_bracket_superscript_notation():
    """Regresja: część tekst-jednolitych PDF-ów koduje indeks górny
    zwykłymi nawiasami kwadratowymi w warstwie tekstowej ("Art. 5[1]."
    zamiast mniejszego rozmiaru czcionki) -- inna konwencja niż ta,
    którą `find_article_numbers_pdfplumber` wykrywał dotychczas.
    Potwierdzony przypadek: Prawo budowlane art. 5[1]/5[2] kolidowały
    (po spłaszczeniu przez `strip_not_yet_in_force_text`) z zupełnie
    innymi, prawdziwymi art. 51/52 (patrz PROGRESS.md, krok 18)."""
    from download_acts import find_article_numbers_pdfplumber, pdf_to_clean_text

    pdf_bytes = (RAW_DIR / "prawo_budowlane.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    articles = parse_articles(clean_text)
    detected = find_article_numbers_pdfplumber(pdf_bytes)
    assert len(detected) == len(articles)
    assert "5¹" in detected
    assert "5²" in detected


@pytest.mark.skipif(
    not (RAW_DIR / "prawo_autorskie.pdf").exists(),
    reason="wymaga data/raw/prawo_autorskie.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
@pytest.mark.slow
def test_find_ustep_superscript_corrections_fixes_autorskie_art_1():
    """Regresja end-to-end Kroku 20/48: art. 1 ust. 2¹ Prawa autorskiego
    (ustęp wstawiony między ust. 2 i ust. 3) był spłaszczany do zwykłego
    '21.' -- potwierdzony pierwszy odkryty przypadek problemu superskryptu
    NA POZIOMIE USTĘPU (osobna, węższa kategoria od Kroku 18, który
    dotyczył wyłącznie numerów artykułów). Sonda w Kroku 48 potwierdziła
    to jako systemowy problem w 7 z 15 dziedzin (201 przypadków), nie
    tylko tu -- ten test pilnuje jednego, dokładnie zweryfikowanego
    przypadku źródłowego."""
    from download_acts import pdf_to_clean_text

    pdf_bytes = (RAW_DIR / "prawo_autorskie.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    corrections = find_ustep_superscript_corrections_pdfplumber(pdf_bytes)
    clean_text = apply_flattened_superscript_corrections(clean_text, corrections)
    articles = parse_articles(clean_text)
    art_1 = next(a for a in articles if a["article"] == "1")
    assert "2¹. Ochroną objęty może być wyłącznie sposób wyrażenia" in art_1["text"]
    assert "21. Ochroną objęty" not in art_1["text"]


@pytest.mark.skipif(
    not (RAW_DIR / "prawo_wlasnosci_przemyslowej.pdf").exists(),
    reason="wymaga data/raw/prawo_wlasnosci_przemyslowej.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
@pytest.mark.slow
def test_find_citation_superscript_corrections_fixes_same_act_citation_list():
    """Regresja end-to-end Kroku 49: cytat w prozie 'art. 1291, art. 1361
    i art. 1363' (odsyłający do trzech artykułów TEJ SAMEJ ustawy) był
    spłaszczony w trzech miejscach naraz -- potwierdzony rzeczywisty
    przypadek, który przy okazji ujawnił błąd we wspólnym mechanizmie
    zamiany (patrz test_apply_flattened_superscript_corrections_neighbor_
    context_not_invalidated): środkowy człon "1361" cytuje w swoim
    kontekście spłaszczony "1291," poprzedniego członu, więc naiwna
    kolejna zamiana w miejscu psuła to dopasowanie."""
    from download_acts import find_citation_superscript_corrections_pdfplumber, pdf_to_clean_text

    pdf_bytes = (RAW_DIR / "prawo_wlasnosci_przemyslowej.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    corrections = find_citation_superscript_corrections_pdfplumber(pdf_bytes)
    clean_text = apply_flattened_superscript_corrections(clean_text, corrections)
    assert "art. 129¹, art. 136¹ i art. 136³" in clean_text


@pytest.mark.skipif(
    not (RAW_DIR / "prawo_wlasnosci_przemyslowej.pdf").exists(),
    reason="wymaga data/raw/prawo_wlasnosci_przemyslowej.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
@pytest.mark.slow
def test_find_citation_superscript_corrections_excludes_different_act_citation():
    """Regresja Kroku 49: 'na podstawie art. 479128 § 1 ustawy z dnia 17
    listopada 1964 r.' cytuje artykuł Kodeksu postępowania cywilnego, NIE
    Prawa własności przemysłowej -- naprawienie tego wymagałoby znajomości
    prawdziwej numeracji INNEJ ustawy, poza zakresem tej funkcji, więc ma
    zostać świadomie pominięty (numeracja jak z pypdf, niepoprawiona)."""
    from download_acts import find_citation_superscript_corrections_pdfplumber

    pdf_bytes = (RAW_DIR / "prawo_wlasnosci_przemyslowej.pdf").read_bytes()
    corrections = find_citation_superscript_corrections_pdfplumber(pdf_bytes)
    flattened_numbers = {flattened for _, flattened, _ in corrections}
    assert "479128" not in flattened_numbers


@pytest.mark.skipif(
    not (RAW_DIR / "ustawa_o_rynku_pracy.pdf").exists() or not (RAW_DIR / "kodeks_pracy.pdf").exists(),
    reason="wymaga data/raw/ustawa_o_rynku_pracy.pdf i kodeks_pracy_meta.json (README, krok 2)",
)
@pytest.mark.slow
def test_find_cross_act_citation_corrections_fixes_kodeks_pracy_and_ksh_citations():
    """Regresja end-to-end Kroku 50 (dokończenie Kroku 49): ustawa o rynku
    pracy cytuje kilkanaście artykułów Kodeksu pracy dotyczących pracy
    zdalnej/stażu (art. 67¹⁸-67³¹) oraz jeden artykuł KSH (art. 300⁷³,
    prosta spółka akcyjna) pełną formą "ustawy z dnia D MIESIĄC RRRR r." --
    to jednoznacznie identyfikuje CYTOWANY akt, a jego prawdziwa numeracja
    (z własnego PDF-u) potwierdza, że naprawiony numer faktycznie istnieje."""
    from download_acts import (
        find_citation_superscript_corrections_pdfplumber,
        find_cross_act_citation_corrections_pdfplumber,
        find_ustep_superscript_corrections_pdfplumber,
        pdf_to_clean_text,
    )

    pdf_bytes = (RAW_DIR / "ustawa_o_rynku_pracy.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    ustep_corrections = find_ustep_superscript_corrections_pdfplumber(pdf_bytes)
    if ustep_corrections:
        clean_text = apply_flattened_superscript_corrections(clean_text, ustep_corrections)
    citation_corrections = find_citation_superscript_corrections_pdfplumber(pdf_bytes)
    if citation_corrections:
        clean_text = apply_flattened_superscript_corrections(clean_text, citation_corrections)
    cross_act_corrections = find_cross_act_citation_corrections_pdfplumber(pdf_bytes, "ustawa_o_rynku_pracy")
    assert len(cross_act_corrections) == 10
    clean_text = apply_flattened_superscript_corrections(clean_text, cross_act_corrections)
    assert "art.  67¹⁸,\nart. 67¹⁹ § 3–5, art. 67²⁴ § 1 pkt 1, 2 i 4, § 2–5, art. 67²⁵, art. 67²⁷, art. 67³¹" in clean_text
    assert "art. 300⁷³\nustawy z dnia 15 września 2000 r. – Kodeks spółek handlowych" in clean_text


@pytest.mark.skipif(
    not (RAW_DIR / "prawo_autorskie.pdf").exists() or not (RAW_DIR / "kpa.pdf").exists(),
    reason="wymaga data/raw/prawo_autorskie.pdf i kpa.pdf (README, krok 2)",
)
@pytest.mark.slow
def test_find_cross_act_citation_corrections_rejects_unvalidated_number():
    """Regresja Kroku 50: 'art. 9915' w Prawie autorskim wspomina KPA
    ('ustawy z dnia 14 czerwca 1960 r.') kilka słów dalej w tym samym
    zdaniu, ale z zupełnie INNEGO powodu proceduralnego (wyłączenie
    pracownika) -- to w rzeczywistości własny artykuł 99¹⁵ Prawa
    autorskiego, nie cytat do KPA. Bezpiecznik warunku 3 (naprawiony numer
    musi istnieć w prawdziwej numeracji CYTOWANEGO aktu) poprawnie odrzuca
    tę korektę, bo KPA nie ma art. 99¹⁵."""
    from download_acts import find_cross_act_citation_corrections_pdfplumber

    pdf_bytes = (RAW_DIR / "prawo_autorskie.pdf").read_bytes()
    corrections = find_cross_act_citation_corrections_pdfplumber(pdf_bytes, "prawo_autorskie")
    flattened_numbers = {flattened for _, flattened, _ in corrections}
    assert "9915" not in flattened_numbers


@pytest.mark.skipif(
    not (PROCESSED_DIR / "kodeks_cywilny.json").exists()
    or not (PROCESSED_DIR / "prawo_budowlane.json").exists(),
    reason="wymaga data/processed/kodeks_cywilny.json i prawo_budowlane.json",
)
class TestBracketSuperscriptCorrection:
    """Regresja end-to-end (PROGRESS.md, krok 18): po korekcie numeracji
    żaden z artykułów kodowanych nawiasami nie koliduje już z innym,
    prawdziwym artykułem o tym samym spłaszczonym numerze."""

    @staticmethod
    def _load(short: str) -> list[dict]:
        return json.loads((PROCESSED_DIR / f"{short}.json").read_text(encoding="utf-8"))

    def test_no_duplicate_article_labels_in_kodeks_cywilny(self):
        data = self._load("kodeks_cywilny")
        labels = [a["article"] for a in data]
        assert len(labels) == len(set(labels))

    def test_konsument_article_correctly_labeled(self):
        data = self._load("kodeks_cywilny")
        by_num = {a["article"]: a for a in data}
        assert "22¹" in by_num
        assert "konsument" in by_num["22¹"]["text"].lower()

    def test_no_duplicate_article_labels_in_prawo_budowlane(self):
        data = self._load("prawo_budowlane")
        labels = [a["article"] for a in data]
        assert len(labels) == len(set(labels))


@pytest.mark.skipif(
    not (PROCESSED_DIR / "all_articles.json").exists(),
    reason="wymaga data/processed/all_articles.json (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
class TestProcessedArticles:
    @staticmethod
    def _load():
        return json.loads((PROCESSED_DIR / "all_articles.json").read_text(encoding="utf-8"))

    def test_no_duplicate_article_labels_in_kodeks_pracy(self):
        data = self._load()
        kp = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 czerwca 1974")]
        labels = [a["article"] for a in kp]
        assert len(labels) == len(set(labels)), "zdublowane numery artykułów -- regresja naprawy indeksu górnego"

    def test_art_223_present_not_merged_into_222(self):
        data = self._load()
        kp = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 czerwca 1974")]
        assert "22³" in {a["article"] for a in kp}

    def test_not_yet_in_force_articles_excluded_from_social_insurance_act(self):
        data = self._load()
        act = [a for a in data if a["act_title"].startswith("Ustawa z dnia 13 października 1998")]
        numbers = {a["article"] for a in act}
        for n in ("85c", "85d", "85e", "85f", "85g", "85h", "85i", "85j"):
            assert n not in numbers, f"art. {n} (jeszcze nieobowiązujący) nie powinien być w bazie"

    def test_art_52zb_present_in_pit(self):
        data = self._load()
        pit = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 lipca 1991")]
        assert "52zb" in {a["article"] for a in pit}


def test_pick_text_file_history_prefers_snapshot_over_updated_unified_text():
    """Krok 69: dla obwieszczenia plik 'U' jest aktualizowany o nowelizacje
    aż do wygaśnięcia okna (stan z KOŃCA okna) -- historia ma brać migawkę
    z dnia ogłoszenia: 'T', potem 'O', a 'U' tylko w ostateczności.
    Bieżący tekst ustawy nadal woli 'U'."""
    meta = {"texts": [
        {"type": "O", "fileName": "D20200287.pdf"},
        {"type": "T", "fileName": "D20200287L.pdf"},
        {"type": "U", "fileName": "D20200287Lj.pdf"},
    ]}
    assert da.pick_text_file(meta, da.HISTORY_TEXT_KINDS) == ("T", "D20200287L.pdf")
    assert da.pick_text_file(meta) == ("U", "D20200287Lj.pdf")
    no_t = {"texts": [{"type": "O", "fileName": "a.pdf"}, {"type": "U", "fileName": "b.pdf"}]}
    assert da.pick_text_file(no_t, da.HISTORY_TEXT_KINDS) == ("O", "a.pdf")
    only_u = {"texts": [{"type": "U", "fileName": "b.pdf"}]}
    assert da.pick_text_file(only_u, da.HISTORY_TEXT_KINDS) == ("U", "b.pdf")


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise da.requests.HTTPError(str(self.status_code), response=self)


def test_get_with_retry_retries_5xx_then_succeeds(monkeypatch):
    """Krok 69: chwilowy 502 z API Sejmu nie może zabić całego przebiegu historii."""
    responses = iter([_FakeResponse(502), _FakeResponse(503), _FakeResponse(200)])
    monkeypatch.setattr(da.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    assert da._get_with_retry("http://x", timeout=1).status_code == 200


def test_get_with_retry_does_not_retry_4xx(monkeypatch):
    calls = []
    monkeypatch.setattr(da.requests, "get", lambda *a, **k: calls.append(1) or _FakeResponse(404))
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    with pytest.raises(da.requests.HTTPError):
        da._get_with_retry("http://x", timeout=1)
    assert len(calls) == 1


def test_get_with_retry_gives_up_after_all_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(da.requests, "get", lambda *a, **k: calls.append(1) or _FakeResponse(502))
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    with pytest.raises(da.requests.HTTPError):
        da._get_with_retry("http://x", timeout=1)
    assert len(calls) == len(da.RETRY_DELAYS) + 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("§ 2.2) Kto", "§ 2. Kto"),
        ("Art. 75a.46) (uchylony)", "Art. 75a. (uchylony)"),
        ("(uchylony)46) dalej", "(uchylony) dalej"),
        ("pojazdów. 3. (uchylony)14) 4.13) Minister", "pojazdów. 3. (uchylony) 4. Minister"),
        # fałszywe alarmy, które NIE mogą się zmienić
        ("(PKWiU ex 22.11) – z wyłączeniem", "(PKWiU ex 22.11) – z wyłączeniem"),
        ("1) pierwszy; 2) drugi", "1) pierwszy; 2) drugi"),
        ("zgodnie z art. 5 ust. 2.", "zgodnie z art. 5 ust. 2."),
    ],
)
def test_strip_inline_footnote_refs(raw, expected):
    """Krok 69: odsyłacze do przypisów sklejone z numerem jednostki redakcyjnej."""
    assert da.strip_inline_footnote_refs(raw) == expected


def test_dziennik_ustaw_page_header_is_header_footer():
    """Krok 69: nagłówek strony w układzie Dz.U. (pliki "T" obwieszczeń)."""
    assert any(p.match("Dziennik Ustaw – 23 – Poz. 17") for p in da.HEADER_FOOTER_PATTERNS)
    assert not any(p.match("Dziennik Ustaw – 23 – Poz. 17 i inne") for p in da.HEADER_FOOTER_PATTERNS)


def test_drop_future_wording_duplicates_keeps_first_of_consecutive_pair():
    """Krok 69: plik "T" podaje brzmienie obowiązujące, a zaraz po nim przyszłe."""
    arts = [
        {"article": "41", "text": "a"},
        {"article": "42", "text": "obowiązuje"},
        {"article": "42", "text": "przyszłe"},
        {"article": "43", "text": "b"},
        {"article": "42", "text": "nie-kolejny duplikat zostaje"},
    ]
    assert [a["text"] for a in da.drop_future_wording_duplicates(arts)] == [
        "a", "obowiązuje", "b", "nie-kolejny duplikat zostaje"
    ]


def test_split_annexes_numbered_with_glued_footnote_ref():
    """Krok 70: załączniki odcięte od ostatniego artykułu; "Załącznik nr 617)"
    w VAT to załącznik nr 6 ze sklejonym odsyłaczem "17)"."""
    arts = [
        {"article": "1", "text": "Art. 1. Coś."},
        {"article": "55", "text": "Art. 55. Ustawa wchodzi w życie po upływie 6 miesięcy. Załączniki do ustawy z dnia "
                                  "30 maja 2014 r. (Dz. U. z 2026 r. poz. 1244) Załącznik nr 1 WZÓR POUCZENIA treść "
                                  "Załącznik nr 2 FORMULARZ treść Załącznik nr 317) TABELA"},
    ]
    out = da.split_annexes(arts)
    assert [a["article"] for a in out] == ["1", "55", "załącznik nr 1", "załącznik nr 2", "załącznik nr 3"]
    assert out[1]["text"] == "Art. 55. Ustawa wchodzi w życie po upływie 6 miesięcy."
    assert out[2]["text"].startswith("Załącznik nr 1 WZÓR POUCZENIA")


def test_split_annexes_single_unnumbered_and_none():
    arts = [{"article": "108", "text": "Art. 108. Ustawa wchodzi w życie. Załącznik do ustawy z dnia 7 lipca "
                                       "1994 r. (Dz. U. z 2026 r. poz. 524) Kategorie obiektów budowlanych"}]
    out = da.split_annexes(arts)
    assert [a["article"] for a in out] == ["108", "załącznik"]
    assert out[1]["text"] == "Załącznik. Kategorie obiektów budowlanych"
    plain = [{"article": "5", "text": "Art. 5. zgodnie z załącznikiem do ustawy."}]
    assert da.split_annexes(plain) == plain


@pytest.mark.parametrize(
    "raw,expected",
    [
        # adnotacja w środku zdania -- zdanie zostaje sklejone (Krok 74, art. 94³ KP)
        (
            "wywołujące u niego zaniżoną ocenę Nowe brzmienie art. 94 3 wejdzie w życie z dn. 5.11.2026 r. "
            "(Dz. U. z 2026 r. poz. 1046). przydatności zawodowej",
            "wywołujące u niego zaniżoną ocenę przydatności zawodowej",
        ),
        # dwie adnotacje pod rząd na końcu artykułu
        (
            "na podstawie odrębnych przepisów. Nowe brzmienie art. 18 3d wejdzie w życie z dn. 5.11.2026 r. "
            "(Dz. U. z 2026 r. poz. 1046). Nowe brzmienie art. 18 3e wejdzie w życie z dn. 5.11.2026 r. "
            "(Dz. U. z 2026 r. poz. 1046).",
            "na podstawie odrębnych przepisów.",
        ),
        # opis z nawiasem, skrótami i wielokrotnym "w ..."
        (
            "przychód w kwocie Zmiana (dodanie wyrazów) w lit. b w pkt 3 w ust. 4 w art. 25b wejdzie w życie "
            "z dn. 1.10.2026 r. (Dz. U. z 2026 r. poz. 846). przekraczającej",
            "przychód w kwocie przekraczającej",
        ),
        (
            "ubezpieczeń społecznych; Art. 85 j wejdzie w życie z dn. 1.01.2027 r. (Dz. U. z 2026 r. poz. 26). "
            "2) prawidłowość",
            "ubezpieczeń społecznych; 2) prawidłowość",
        ),
        (
            "toczącym się postępowaniem; Nowe brzmienie wprowadz. do wylicz. w art. 298 wejdzie w życie z dn. "
            "1.01.2027 r. (Dz. U. z 2026 r. poz. 1098).",
            "toczącym się postępowaniem;",
        ),
        # bez rozpoznawalnego opisu -- nic nie ruszamy (lepiej szum niż ucięty przepis)
        (
            "Ustawa wejdzie w życie z dn. 1.01.2027 r. (Dz. U. z 2026 r. poz. 1).",
            "Ustawa wejdzie w życie z dn. 1.01.2027 r. (Dz. U. z 2026 r. poz. 1).",
        ),
        ("Art. 5. Zwykły przepis bez adnotacji.", "Art. 5. Zwykły przepis bez adnotacji."),
    ],
)
def test_strip_entry_into_force_annotations(raw, expected):
    assert da.strip_entry_into_force_annotations(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "o których mowa w lit. a; Zmiana we wprowadzeniu do wyliczenia w ust. 9c w art. 401c wejdzie w życie "
            "z dn. 1.01.20 30 r. (Dz. U. z 2025 r. poz. 303). 5) dofinansowanie",
            "o których mowa w lit. a; 5) dofinansowanie",
        ),
        (
            "na kierowcę. Dodany ust. 7c w art. 100aa wejdzie w życie z dn. określonym w komunikacie, o którym mowa "
            "w art. 21 ustawy z dnia 14 października 2021 r. o zmianie ustawy o transporcie drogowym oraz niektórych "
            "innych ustaw (Dz. U. poz. 1997, z 2022 r. poz. 1002). b) pkt 11",
            "na kierowcę. b) pkt 11",
        ),
        # prawdziwa treść przepisu o wejściu w życie -- zostaje
        (
            "w zakresie, w jakim dotyczy planu, wejdzie w życie z dniem 1 stycznia 2027 r. 1a. W planie",
            "w zakresie, w jakim dotyczy planu, wejdzie w życie z dniem 1 stycznia 2027 r. 1a. W planie",
        ),
    ],
)
def test_strip_entry_into_force_annotations_variants(raw, expected):
    assert da.strip_entry_into_force_annotations(raw) == expected


def test_repair_flattened_superscript_labels_kpc_block():
    # Krok 84: KPC art. 479^28-479^35 -- pdfplumber nie wykrył nagłówków
    labels = ["477", "477¹⁴", "47714a", "47715", "47716", "478", "479", "47928", "47929", "47929a", "47930", "479⁸⁹", "480"]
    assert repair_flattened_superscript_labels(labels) == {
        2: "477¹⁴a", 3: "477¹⁵", 4: "477¹⁶", 7: "479²⁸", 8: "479²⁹", 9: "479²⁹a", 10: "479³⁰",
    }


def test_repair_flattened_superscript_labels_leaves_normal_numbering_alone():
    # Poprawna, rosnąca numeracja (także z artykułami o dłuższych numerach, które
    # zaczynają się od poprzedniego) nie może być ruszana -- brak późniejszego spadku
    assert repair_flattened_superscript_labels(["1", "12", "121", "1217", "1217a", "1218"]) == {}
    assert repair_flattened_superscript_labels(["1", "2", "3", "10", "11", "100", "101", "1000"]) == {}
    assert repair_flattened_superscript_labels(["4", "4¹", "5", "50", "500", "5000"]) == {}


def test_fix_font_mojibake_repairs_mac_font_text():
    # Krok 86: realne fragmenty z okien historii sprzed 2014 (USUS, KPA, POŚ...)
    garbled = ("Przepisy okreÊlajà, ˝e dzia∏ajà organy i uwa˝a si´ wzgl´du; "
               "DZIA¸ I PRZEJÂCIOWE IKO¡COWE, U˚YTKOWE, URZ¢DOWE; zg∏oszeƒ, paêdziern, chodziç. ") * 4
    fixed = fix_font_mojibake(garbled)
    assert "określają, że działają organy i uważa się względu" in fixed
    assert "DZIAŁ I PRZEJŚCIOWE IKOŃCOWE, UŻYTKOWE, URZĘDOWE; zgłoszeń, październ, chodzić." in fixed
    assert not any(c in fixed for c in "∏à´˝ƒ")


def test_fix_font_mojibake_leaves_healthy_text_alone():
    healthy = "Art. 1. § 1. Przepisy ustawy określają zasady działania; zob. też art. 5 (fr. café, à la carte)."
    assert fix_font_mojibake(healthy) == healthy


def test_odd_char_rate_separates_healthy_text_from_broken_scans():
    # Krok 86: zdrowy tekst ustawy vs. zepsuty skan OCR (KPA przed 2000: "Orgpn", "podle~ ga")
    healthy = ["Art. 1. § 1. Kodeks postępowania administracyjnego normuje postępowanie przed organami (zob. też art. 5)."] * 20
    broken = ["Art. 25. §, 1. Orgpn administracji państwowej podle~ ga wyłączeniu od . załatwienia sprawy ~ ~ ' \u2022 \u2022 ~"] * 20
    assert odd_char_rate_per_1000(healthy) < 1.0
    assert odd_char_rate_per_1000(broken) > SCAN_QUALITY_MAX_ODD_PER_1000
    assert odd_char_rate_per_1000([]) == 0.0


def test_convert_bracket_superscripts_converts_attached_digits_only():
    # Krok 90: "[n]" przyklejone do liczby = indeks górny; nawias redakcyjny zostaje
    # nagłówek "Art. 18[4]." (wielka litera) zostaje -- obsługuje go geometria PDF
    assert convert_bracket_superscripts("Art. 18[4]. § 2[1]. art. 22[1] pkt 5[2]) ust. 3[3d]") == (
        "Art. 18[4]. § 2¹. art. 22¹ pkt 5²) ust. 3³d"
    )
    editorial = "[stary tekst] oraz [1. Treść] i ust. [2] albo [nowy wyraz]."
    assert convert_bracket_superscripts(editorial) == editorial


def test_split_annexes_without_annexes_header_uses_standalone_line():
    # Krok 90: Prawo lotnicze 2026 -- po ostatnim artykule od razu "Załącznik nr 1"
    arts = [{"article": "231", "text": "Art. 231. Traci moc ustawa."},
            {"article": "232", "text": "Art. 232. Ustawa wchodzi w życie po 3 miesiącach.\nZałącznik nr 1\nLp. Wyszczególnienie\n1. Naruszenie A\nZałącznik nr 2\n(uchylony)"}]
    out = split_annexes(arts)
    assert [a["article"] for a in out] == ["231", "232", "załącznik nr 1", "załącznik nr 2"]
    assert out[1]["text"] == "Art. 232. Ustawa wchodzi w życie po 3 miesiącach."
    assert "Naruszenie A" in out[2]["text"]
    # zwykła proza z odwołaniem w środku linii nie uruchamia podziału
    prose = [{"article": "9", "text": "Art. 9. Zgodnie z\nZałącznik nr 1 do ustawy stosuje się wzór."}]
    assert split_annexes(prose) == prose
