from src import translator
from src import llm_client


def test_llm_normal_response(monkeypatch):
    # Simulate a good LLM response via query_llm_robust
    def fake_query(post: str):
        return (False, "This is English")

    monkeypatch.setattr(llm_client, "query_llm_robust", fake_query)

    is_english, translated = translator.translate("Dies ist eine Nachricht auf Deutsch")

    assert is_english is False
    assert translated == "This is English"


def test_llm_gibberish_response(monkeypatch):
    # Simulate bad / broken behavior inside LLM pipeline:
    # query_llm_robust should protect us and fall back.
    def fake_query(post: str):
        # Pretend something blew up / malformed;
        # our robust function would return (True, post)
        return (True, post)

    monkeypatch.setattr(llm_client, "query_llm_robust", fake_query)

    original = "###@@@!!!!???"
    is_english, translated = translator.translate(original)

    assert is_english is True
    assert translated == original
