"""Testy dla data/acts_config.json -- konfiguracji aktów prawnych z
podziałem na dziedziny, która zastąpiła hardcodowaną listę ACTS.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ACTS_CONFIG_PATH = ROOT / "data" / "acts_config.json"

REQUIRED_FIELDS = {"short", "publisher", "year", "position", "domains"}


def _load():
    return json.loads(ACTS_CONFIG_PATH.read_text(encoding="utf-8"))


def test_acts_config_is_a_list_of_dicts():
    acts = _load()
    assert isinstance(acts, list)
    assert len(acts) > 0
    assert all(isinstance(a, dict) for a in acts)


def test_every_act_has_required_fields():
    for act in _load():
        missing = REQUIRED_FIELDS - act.keys()
        assert not missing, f"{act.get('short')!r} brakuje pól: {missing}"


def test_shorts_are_unique():
    shorts = [a["short"] for a in _load()]
    assert len(shorts) == len(set(shorts)), "zduplikowane 'short' w acts_config.json"


def test_domains_is_a_non_empty_list_of_strings_for_every_act():
    for act in _load():
        domains = act["domains"]
        assert isinstance(domains, list) and len(domains) > 0, f"{act['short']!r}: 'domains' puste lub nie-lista"
        assert all(isinstance(d, str) for d in domains)


def test_year_and_position_are_ints():
    for act in _load():
        assert isinstance(act["year"], int), f"{act['short']!r}: 'year' nie jest int"
        assert isinstance(act["position"], int), f"{act['short']!r}: 'position' nie jest int"


def test_load_acts_config_filters_by_domain():
    from download_acts import load_acts_config

    all_acts = load_acts_config()
    podatkowe = load_acts_config(domain="podatkowe")
    assert 0 < len(podatkowe) < len(all_acts)
    assert all("podatkowe" in a["domains"] for a in podatkowe)


def test_load_acts_config_filters_by_shorts():
    from download_acts import load_acts_config

    result = load_acts_config(shorts={"kodeks_pracy"})
    assert [a["short"] for a in result] == ["kodeks_pracy"]


def test_act_in_multiple_domains_appears_once_not_duplicated():
    acts = _load()
    pit = [a for a in acts if a["short"] == "ustawa_o_pit"]
    assert len(pit) == 1
    assert set(pit[0]["domains"]) >= {"kadry_i_place", "podatkowe"}
