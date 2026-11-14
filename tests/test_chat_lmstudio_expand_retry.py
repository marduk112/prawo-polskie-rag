import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import chat_lmstudio as c  # noqa: E402


def _fake(responses, calls):
    def post_chat(base_url, payload, timeout):
        calls.append(payload["max_tokens"])
        return {"content": responses.pop(0)}
    return post_chat


def test_empty_expansion_is_retried_once_with_larger_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "post_chat", _fake(["", "przeformułowane"], calls))
    assert c.expand_query("u", "m", "pytanie", reasoning_effort="low") == "przeformułowane"
    assert calls == [800, c.EMPTY_EXPANSION_RETRY_MAX_TOKENS]


def test_non_empty_expansion_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "post_chat", _fake(["ok"], calls))
    assert c.expand_query("u", "m", "pytanie", reasoning_effort="low") == "ok"
    assert calls == [800]


def test_no_retry_without_reasoning_effort(monkeypatch):
    calls = []
    monkeypatch.setattr(c, "post_chat", _fake([""], calls))
    assert c.expand_query("u", "m", "pytanie") == ""
    assert calls == [800]


def test_generation_defaults_for_qwen_and_others():
    assert c.resolve_generation_defaults("qwen/qwen3.8-27b", None, None) == (3000, "low")
    assert c.resolve_generation_defaults("bielik-11b-v3.0-instruct", None, None) == (500, None)
    assert c.resolve_generation_defaults("qwen/qwen3.8-27b", 1200, "high") == (1200, "high")
