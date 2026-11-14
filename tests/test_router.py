"""Testy dla scripts/router.py -- logika routera dziedzin (Basal) bez prawdziwego
serwera: odpowiedź HTTP podmieniana przez wstrzykiwane `post`."""

import pytest

from router import (
    DEFAULT_SINGLE_THRESHOLD,
    RouterError,
    build_criteria,
    decide,
    load_acts_config,
    route,
)

ACTS = [
    {"short": "kodeks_pracy", "domains": ["kadry_i_place"]},
    {"short": "ustawa_o_pit", "domains": ["kadry_i_place", "podatkowe"]},
    {"short": "ustawa_o_vat", "domains": ["podatkowe"]},
]


def test_build_criteria_uses_act_names_sorted():
    criteria = build_criteria(ACTS)
    assert criteria["kadry_i_place"] == "Prawo polskie: kodeks pracy, ustawa o pit"
    assert criteria["podatkowe"] == "Prawo polskie: ustawa o pit, ustawa o vat"


def test_build_criteria_restricts_to_available_domains():
    assert list(build_criteria(ACTS, domains=["podatkowe"])) == ["podatkowe"]


def test_build_criteria_matches_real_config_domains():
    acts = load_acts_config()
    criteria = build_criteria(acts)
    assert {d for a in acts for d in a["domains"]} == set(criteria)


def test_decide_confident_returns_single_domain():
    d = decide({"a": 0.95, "b": 0.04, "c": 0.01}, 0.95)
    assert (d.primary, d.secondary, d.domains) == ("a", None, ["a"])


def test_decide_unconfident_adds_second_best():
    d = decide({"a": 0.5, "b": 0.3, "c": 0.2}, 0.5)
    assert (d.primary, d.secondary, d.domains) == ("a", "b", ["a", "b"])


def test_decide_threshold_is_inclusive():
    probs = {"a": DEFAULT_SINGLE_THRESHOLD, "b": 1 - DEFAULT_SINGLE_THRESHOLD}
    assert decide(probs, DEFAULT_SINGLE_THRESHOLD).secondary is None


def test_decide_tie_broken_by_name():
    d = decide({"b": 0.4, "a": 0.4, "c": 0.2}, 0.4)
    assert d.domains == ["a", "b"]


def test_decide_single_domain_never_has_secondary():
    assert decide({"a": 1.0}, 0.2).secondary is None


def test_decide_empty_probabilities_raises():
    with pytest.raises(RouterError):
        decide({}, 0.5)


def make_post(answer):
    calls = []

    def post(url, body, timeout):
        calls.append((url, body, timeout))
        return {"answers": {"dziedzina": answer}}

    return post, calls


def test_route_sends_choice_question_and_parses_answer():
    post, calls = make_post({"choice": "podatkowe", "probabilities": {"podatkowe": 0.97, "kadry_i_place": 0.03}, "confidence": 0.97})
    criteria = build_criteria(ACTS)
    decision = route("Czy mogę odliczyć VAT?", criteria, url="http://x/v1/systemone", post=post)
    assert decision.domains == ["podatkowe"]
    url, body, _ = calls[0]
    assert url == "http://x/v1/systemone"
    assert body["state"] == "Czy mogę odliczyć VAT?"
    question = body["questions"]["dziedzina"]
    assert question["type"] == "choice" and question["criteria"] == criteria


def test_route_low_confidence_returns_two_domains():
    post, _ = make_post({"choice": "a", "probabilities": {"a": 0.55, "b": 0.4, "c": 0.05}, "confidence": 0.55})
    assert route("pytanie", {"a": "A", "b": "B", "c": "C"}, post=post).domains == ["a", "b"]


@pytest.mark.parametrize("bad", [{}, {"probabilities": {"a": 1.0}}, {"choice": "a", "probabilities": {"a": 1.0}, "confidence": "x"}])
def test_route_malformed_answer_raises_router_error(bad):
    post, _ = make_post(bad)
    with pytest.raises(RouterError):
        route("pytanie", {"a": "A"}, post=post)


def test_route_unreachable_server_gives_actionable_error():
    # port 9 (discard) -- nic tam nie nasłuchuje; błąd ma wskazać komendę startową
    with pytest.raises(RouterError, match="basal-serve"):
        route("pytanie", {"a": "A"}, url="http://127.0.0.1:9/v1/systemone", timeout=2)


def test_available_domains_requires_both_index_files(tmp_path):
    from router import available_domains

    for name in ("rag_index_a_meta.json", "rag_index_a.npy", "rag_index_b_meta.json"):
        (tmp_path / name).write_text("")
    assert available_domains(tmp_path) == ["a"]


def test_route_battery_writes_primary_secondary_confidence(tmp_path, monkeypatch):
    import json

    import router

    monkeypatch.setattr(router, "available_domains", lambda *a, **k: ["kadry_i_place", "podatkowe"])
    probs = {"kadry_i_place": 0.5, "podatkowe": 0.4}
    post = lambda url, body, timeout: {"answers": {"dziedzina": {"probabilities": probs, "confidence": 0.5}}}
    out = tmp_path / "routing.json"
    router.route_battery([{"id": "c1", "question": "q"}], out, post=post)
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "c1": {"primary": "kadry_i_place", "secondary": "podatkowe", "confidence": 0.5, "probabilities": probs}
    }
