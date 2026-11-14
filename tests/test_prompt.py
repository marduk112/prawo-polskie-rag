"""Testy dla scripts/prompt.py -- logiki wspólnej dla wszystkich
wariantów czatu (kontekst rozmowy, wykrywanie pytań meta, fallback
wyszukiwania RAG dla pytań eliptycznych, ucinanie długich artykułów).
Używają lekkiego fałszywego RagIndex zamiast prawdziwego (bez ładowania
modelu embeddingowego) -- patrz PROGRESS.md, krok 10/11.
"""

import pytest

from prompt import (
    DEFAULT_DOMAIN_DESCRIPTION,
    DOMAIN_DESCRIPTIONS,
    MAX_ARTICLE_CHARS,
    QUERY_EXPANSION_SYSTEM_PROMPT,
    build_context,
    build_query_expansion_messages,
    build_system_prompt,
    build_user_message,
    looks_like_meta_question,
    search_with_history,
    trim_history,
)


class FakeRag:
    """Podmiennik RagIndex: zwraca z góry zdefiniowane wyniki per zapytanie."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def search(self, query: str, top_k: int, as_of: str | None = None, extra_queries: list[str] | None = None):
        self.calls.append((query, as_of))
        return self.responses.get(query, [])[:top_k]


def make_result(
    article,
    score,
    text="treść",
    act_title="Testowa ustawa",
    source_url="http://example.test",
    valid_from=None,
    valid_to=None,
):
    result = {"article": article, "score": score, "text": text, "act_title": act_title, "source_url": source_url}
    result["valid_from"] = valid_from
    result["valid_to"] = valid_to
    return result


# --- trim_history ---


def test_trim_history_keeps_last_n_pairs():
    history: list[dict] = []
    for i in range(10):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
        history = trim_history(history, 3)
    assert len(history) == 6
    assert history[0]["content"] == "q7"
    assert history[-1]["content"] == "a9"


def test_trim_history_noop_when_under_limit():
    history = [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]
    result = trim_history(list(history), 6)
    assert result == history


# --- looks_like_meta_question ---


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Zacytuj dokładnie przepis o którym mówiliśmy na samym początku, w pierwszym pytaniu.", True),
        ("Przypomnij, co powiedziałeś wcześniej.", True),
        ("O czym rozmawialiśmy na początku?", True),
        ("Podsumuj naszą rozmowę.", True),
        ("Ile dni urlopu przysługuje po 10 latach pracy?", False),
        ("A po 15 latach?", False),
        ("A ile wynosi minimalne wynagrodzenie w 2025 roku?", False),
        ("Jaki jest termin wypowiedzenia umowy o pracę?", False),
    ],
)
def test_looks_like_meta_question(question, expected):
    assert looks_like_meta_question(question) is expected


# --- build_context (obcinanie długich artykułów) ---


def test_build_context_truncates_long_articles():
    long_text = "x" * (MAX_ARTICLE_CHARS + 500)
    context = build_context([make_result("50", 0.9, text=long_text)])
    assert len(context) < len(long_text)
    assert "http://example.test" in context


def test_build_context_leaves_short_articles_untouched():
    short_text = "krótki tekst artykułu"
    context = build_context([make_result("1", 0.9, text=short_text)])
    assert short_text in context
    assert "skrócona" not in context


def test_build_context_header_unchanged_when_no_version_info():
    """Regresja: format nagłówka bez wymiaru czasowego (valid_from/valid_to
    oba None -- domyślny przypadek, indeks bez --include-history) musi
    zostać DOKŁADNIE taki jak przed dodaniem wersjonowania czasowego."""
    context = build_context([make_result("154", 0.9, text="treść", act_title="Kodeks pracy")])
    assert context == "### Kodeks pracy -- art. 154\ntreść"


def test_build_context_adds_validity_annotation_when_present():
    context = build_context(
        [make_result("25", 0.9, text="treść", act_title="Ustawa X", valid_from="2015-11-09", valid_to="2017-04-26")]
    )
    assert "### Ustawa X -- art. 25 (stan prawny: 2015-11-09 – 2017-04-26)" in context


def test_build_context_annotation_handles_open_ended_bounds():
    context = build_context([make_result("25", 0.9, act_title="Ustawa X", valid_from="2024-11-27", valid_to=None)])
    assert "(stan prawny: 2024-11-27 – nadal)" in context


# --- search_with_history (fallback dla pytań eliptycznych, patrz PROGRESS.md krok 11) ---


def test_search_with_history_elliptical_question_uses_prior_question():
    rag = FakeRag(
        {
            "A po 15 latach?": [make_result("15", 0.5)],  # słabe dopasowanie, poniżej progu
            "Ile dni urlopu przysługuje po 10 latach pracy? A po 15 latach?": [make_result("154", 0.9)],
        }
    )
    history = [
        {"role": "user", "content": "Ile dni urlopu przysługuje po 10 latach pracy?"},
        {"role": "assistant", "content": "26 dni"},
    ]
    results = search_with_history(rag, history, "A po 15 latach?", top_k=5)
    assert results[0]["article"] == "154"


def test_search_with_history_keeps_strong_match_when_lookback_finds_nothing_better():
    """Od Kroku 36 wariant z historią jest próbowany ZAWSZE (nie tylko
    gdy wynik dla samego pytania jest słaby) -- ale gdy nic lepszego nie
    znajdzie, wynik dla samego pytania i tak wygrywa."""
    rag = FakeRag({"Ile wynosi minimalne wynagrodzenie?": [make_result("5", 0.8)]})
    history = [{"role": "user", "content": "cokolwiek innego"}, {"role": "assistant", "content": "..."}]
    results = search_with_history(rag, history, "Ile wynosi minimalne wynagrodzenie?", top_k=5)
    assert results[0]["article"] == "5"


def test_search_with_history_overrides_coincidentally_high_scoring_wrong_match():
    """Reprodukcja Kroku 36: pytanie eliptyczne trafia PRZYPADKOWO w
    wysoko punktowany (0.779), ale tematycznie błędny fragment -- próg
    sam w sobie by to przepuścił (0.779 > dawny próg 0.75), więc wariant
    z historią MUSI być sprawdzony zawsze, nie tylko poniżej progu."""
    rag = FakeRag(
        {
            "a jeśli jest zatrudniony dłużej?": [make_result("25indeks1", 0.779)],
            "ile dni urlopu wypoczynkowego przysługuje pracownikowi zatrudnionemu krócej niż 10 lat a jeśli jest zatrudniony dłużej?": [
                make_result("154", 0.9)
            ],
        }
    )
    history = [
        {"role": "user", "content": "ile dni urlopu wypoczynkowego przysługuje pracownikowi zatrudnionemu krócej niż 10 lat"},
        {"role": "assistant", "content": "20 dni"},
    ]
    results = search_with_history(rag, history, "a jeśli jest zatrudniony dłużej?", top_k=5)
    assert results[0]["article"] == "154"


def test_search_with_history_skips_unrelated_immediately_prior_turn():
    """Reprodukcja realnego przypadku: poprzednia tura (Q2) to inny,
    zakończony odmową temat -- fallback musi sięgnąć do wcześniejszego,
    faktycznie powiązanego pytania (Q1), nie tylko bezpośrednio
    poprzedniego."""
    rag = FakeRag(
        {
            "A po 15 latach?": [make_result("x", 0.5)],
            "A ile wynosi minimalne wynagrodzenie w 2025 roku? A po 15 latach?": [make_result("wrong", 0.6)],
            "Ile dni urlopu przysługuje po 10 latach pracy? A po 15 latach?": [make_result("154", 0.9)],
        }
    )
    history = [
        {"role": "user", "content": "Ile dni urlopu przysługuje po 10 latach pracy?"},
        {"role": "assistant", "content": "26 dni"},
        {"role": "user", "content": "A ile wynosi minimalne wynagrodzenie w 2025 roku?"},
        {"role": "assistant", "content": "Nie znalazłem."},
    ]
    results = search_with_history(rag, history, "A po 15 latach?", top_k=5)
    assert results[0]["article"] == "154"


def test_search_with_history_empty_history_uses_plain_search():
    rag = FakeRag({"pytanie": [make_result("1", 0.5)]})
    results = search_with_history(rag, [], "pytanie", top_k=5)
    assert results[0]["article"] == "1"
    assert rag.calls == [("pytanie", None)]


def test_search_with_history_threads_as_of_through_every_call():
    """as_of musi trafić zarówno do pierwszego wyszukania, jak i do KAŻDEJ
    próby fallbacku w pętli lookback (patrz PROGRESS.md, plan implementacji
    "as-of", Faza 3)."""
    rag = FakeRag(
        {
            "A po 15 latach?": [make_result("15", 0.5)],
            "Ile dni urlopu przysługuje po 10 latach pracy? A po 15 latach?": [make_result("154", 0.9)],
        }
    )
    history = [
        {"role": "user", "content": "Ile dni urlopu przysługuje po 10 latach pracy?"},
        {"role": "assistant", "content": "26 dni"},
    ]
    search_with_history(rag, history, "A po 15 latach?", top_k=5, as_of="2015-11-09")
    assert rag.calls == [
        ("A po 15 latach?", "2015-11-09"),
        ("Ile dni urlopu przysługuje po 10 latach pracy? A po 15 latach?", "2015-11-09"),
    ]


# --- build_user_message (patrz też scripts/file_index.py -- upload plików) ---


def test_build_user_message_without_file_results_has_no_file_section():
    msg = build_user_message("pytanie", [make_result("1", 0.9)])
    assert "Fragmenty aktów prawnych" in msg
    assert "wgranego przez użytkownika pliku" not in msg


def test_build_user_message_includes_file_section_when_present():
    file_results = [{"source_file": "umowa.pdf", "text": "treść umowy", "score": 0.8}]
    msg = build_user_message("pytanie", [make_result("1", 0.9)], file_results)
    assert "Fragmenty aktów prawnych" in msg
    assert "wgranego przez użytkownika pliku" in msg
    assert "umowa.pdf" in msg
    assert "treść umowy" in msg


def test_build_user_message_omits_file_section_when_empty_list():
    msg = build_user_message("pytanie", [make_result("1", 0.9)], [])
    assert "wgranego przez użytkownika pliku" not in msg


def test_build_user_message_includes_as_of_note_when_given():
    msg = build_user_message("pytanie", [make_result("1", 0.9)], as_of="2019-06-01")
    assert "[Data, na którą ma obowiązywać odpowiedź: 2019-06-01]" in msg


def test_build_user_message_omits_as_of_note_when_none():
    msg = build_user_message("pytanie", [make_result("1", 0.9)])
    assert "Data, na którą ma obowiązywać odpowiedź" not in msg


def test_build_user_message_warns_when_best_score_below_threshold():
    msg = build_user_message("pytanie", [make_result("1", 0.5)])
    assert "[Uwaga: żaden odnaleziony fragment nie jest wysoko dopasowany" in msg
    assert "0.50" in msg


def test_build_user_message_warns_when_no_results():
    msg = build_user_message("pytanie", [])
    assert "[Uwaga: żaden odnaleziony fragment nie jest wysoko dopasowany" in msg
    assert "brak wyników" in msg


def test_build_user_message_no_warning_when_best_score_above_threshold():
    msg = build_user_message("pytanie", [make_result("1", 0.9)])
    assert "[Uwaga: żaden odnaleziony fragment" not in msg


class TestBuildSystemPrompt:
    def test_known_domain_gets_its_description(self):
        prompt = build_system_prompt("podatkowe")
        assert DOMAIN_DESCRIPTIONS["podatkowe"] in prompt
        assert DOMAIN_DESCRIPTIONS["kadry_i_place"] not in prompt

    def test_unknown_domain_falls_back_to_default_description(self):
        prompt = build_system_prompt("cos_nieznanego")
        assert DEFAULT_DOMAIN_DESCRIPTION in prompt

    def test_none_domain_falls_back_to_default_description(self):
        prompt = build_system_prompt(None)
        assert DEFAULT_DOMAIN_DESCRIPTION in prompt

    def test_extra_domain_describes_both_domains(self):
        prompt = build_system_prompt("pcc", "podatkowe")
        assert DOMAIN_DESCRIPTIONS["pcc"] + " oraz " + DOMAIN_DESCRIPTIONS["podatkowe"] in prompt

    def test_extra_domain_same_as_primary_or_none_is_ignored(self):
        assert build_system_prompt("pcc", "pcc") == build_system_prompt("pcc") == build_system_prompt("pcc", None)

    def test_every_domain_shares_the_same_rules_section(self):
        """Tylko opis dziedziny się zmienia -- reszta instrukcji (zasady
        1-5) musi być identyczna niezależnie od dziedziny."""
        a = build_system_prompt("kadry_i_place")
        b = build_system_prompt("administracyjne")
        assert a.split("\n\n", 1)[1] == b.split("\n\n", 1)[1]


class TestBuildQueryExpansionMessages:
    def test_returns_system_and_user_message(self):
        messages = build_query_expansion_messages("Czy mogę odliczyć VAT od zakupów prywatnych?")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "Czy mogę odliczyć VAT od zakupów prywatnych?"
        assert "przeformułowanie" in messages[0]["content"]

    def test_without_act_titles_leaves_question_unchanged(self):
        messages = build_query_expansion_messages("pytanie testowe", act_titles=None)
        assert messages[1]["content"] == "pytanie testowe"

    def test_with_act_titles_prepends_context_line(self):
        messages = build_query_expansion_messages("pytanie testowe", act_titles=["Ustawa A", "Ustawa B"])
        assert messages[1]["content"].startswith("Kontekst: pytanie dotyczy jednej z następujących ustaw: ")
        assert "Ustawa A; Ustawa B" in messages[1]["content"]
        assert messages[1]["content"].endswith("\n\npytanie testowe")
        # system prompt sam w sobie musi zostać NIETKNIĘTY (Krok 28 -- cała
        # naprawa jest celowo w wiadomości użytkownika, nie systemowej)
        assert messages[0]["content"] == QUERY_EXPANSION_SYSTEM_PROMPT

    def test_empty_act_titles_list_treated_as_no_context(self):
        messages = build_query_expansion_messages("pytanie testowe", act_titles=[])
        assert messages[1]["content"] == "pytanie testowe"


class TestSearchWithHistoryExtraQueries:
    def test_forwards_extra_queries_to_primary_search_only(self):
        rag = FakeRag({"pytanie": [make_result("1", 0.9)]})
        search_with_history(rag, [], "pytanie", top_k=5, extra_queries=["sformalizowane pytanie"])
        assert rag.calls == [("pytanie", None)]  # FakeRag nie loguje extra_queries, ale wywołanie nie wywala się


def test_article_ref_annex_label():
    """Krok 70: załącznik nie jest "art. załącznik nr 2"."""
    from prompt import article_ref
    assert article_ref("12a") == "art. 12a"
    assert article_ref("załącznik nr 2") == "załącznik nr 2"


def test_keyword_hint_added_only_for_matching_questions():
    import prompt

    # Krok 87: para o przedawnieniu tylko dla pytań ze słowem "przedaw..."; pytania
    # konsumenckie ("sprzedawca" zawiera "przedaw") dostają prompt jak przed zmianą
    base = prompt.DOMAIN_QUERY_EXPANSION_HINTS["cywilne"]
    with_kw = prompt.build_query_expansion_messages("Czy moje roszczenie już się przedawniło?", domain="cywilne")[1]["content"]
    assert "ulega przedawnieniu" in with_kw and base in with_kw
    consumer = prompt.build_query_expansion_messages("sprzedawca twierdzi, że to moja wina", domain="cywilne")[1]["content"]
    assert "ulega przedawnieniu" not in consumer and base in consumer
    other = prompt.build_query_expansion_messages("Czy roszczenie się przedawniło?", domain="karne")[1]["content"]
    assert "ulega przedawnieniu" not in other
