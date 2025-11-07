import pytest
from src import translator as tr


def test_llm_normal_response(monkeypatch):
    """
    Verify that translate() returns correct values when the LLM behaves
    as expected (non-English input -> translated English).
    """

    # Mock your own helpers from this file (not other teams' code)
    monkeypatch.setattr(tr, "get_language", lambda _txt: "German")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "This is a German message")

    is_english, translated = tr.translate("Dies ist eine Nachricht auf Deutsch")

    assert is_english is False
    assert translated == "This is a German message"


def test_llm_gibberish_response(monkeypatch):
    """
    Verify that your pipeline handles a gibberish / malformed response from
    the LLM gracefully via query_llm_robust.

    We simulate this by making query_llm return something in the wrong format.
    query_llm_robust (called inside translate) should catch this and fall back
    to (True, original_text).
    """

    def fake_query_llm(_txt):
        # This mimics a badly formatted model result
        return "not-a-tuple-at-all"

    monkeypatch.setattr(tr, "query_llm", fake_query_llm)

    original = "Bonjour tout le monde"
    is_english, translated = tr.translate(original)

    # Robust behavior: treat as English + don't crash
    assert is_english is True
    assert translated == original
