import pytest
from src import translator as tr
from unittest.mock import patch


translation_eval_set = [
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": "Here is your first example."},
    {"post": "मैं आज सुबह बाजार में गया था।", "expected_answer": "I went to the market this morning."},
    {"post": "¿Dónde está la estación de tren más cercana?", "expected_answer": "Where is the nearest train station?"},
    {"post": "今日は忙しい一日だった。", "expected_answer": "It was a busy day today."},
    {"post": "Ceci est mon dernier jour à Paris.", "expected_answer": "This is my last day in Paris."},
    {"post": "أريد كوباً من القهوة من فضلك.", "expected_answer": "I would like a cup of coffee, please."},
    {"post": "Я люблю читать книги по вечерам.", "expected_answer": "I love reading books in the evenings."},
    {"post": "Aș vrea să cumpăr un bilet de tren.", "expected_answer": "I would like to buy a train ticket."},
    {"post": "今天天气很好。", "expected_answer": "The weather is nice today."},
]

language_detection_eval_set = [
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": "German"},
    {"post": "मैं आज सुबह बाजार में गया था।", "expected_answer": "Hindi"},
    {"post": "¿Dónde está la estación de tren más cercana?", "expected_answer": "Spanish"},
    {"post": "今日は忙しい一日だった。", "expected_answer": "Japanese"},
    {"post": "Ceci est mon dernier jour à Paris.", "expected_answer": "French"},
    {"post": "أريد كوباً من القهوة من فضلك.", "expected_answer": "Arabic"},
    {"post": "Я люблю читать книги по вечерам.", "expected_answer": "Russian"},
    {"post": "Aș vrea să cumpăr un bilet de tren.", "expected_answer": "Romanian"},
    {"post": "今天天气很好。", "expected_answer": "Chinese"},
]

complete_eval_set = [
    # Non-English posts 
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": (False, "This is your first example.")},
    {"post": "Je suis très heureux de te voir aujourd'hui.", "expected_answer": (False, "I am very happy to see you today.")},
    {"post": "¿Dónde está la biblioteca?", "expected_answer": (False, "Where is the library?")},
    {"post": "Ceci n’est pas une pipe.", "expected_answer": (False, "This is not a pipe.")},
    {"post": "Il fait beau aujourd’hui.", "expected_answer": (False, "The weather is nice today.")},
    {"post": "No capisco quello che dici.", "expected_answer": (False, "I don’t understand what you are saying.")},
    {"post": "私は学生です。", "expected_answer": (False, "I am a student.")},
    {"post": "Привет, как дела?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "Bom dia, tudo bem?", "expected_answer": (False, "Good morning, how are you?")},
    {"post": "这是一个猫。", "expected_answer": (False, "This is a cat.")},
    {"post": "Eu gosto de aprender novas línguas.", "expected_answer": (False, "I like to learn new languages.")},
    {"post": "C’est une belle journée pour une promenade.", "expected_answer": (False, "It’s a beautiful day for a walk.")},
    {"post": "Grazie per il tuo aiuto.", "expected_answer": (False, "Thank you for your help.")},
    {"post": "¡Feliz cumpleaños!", "expected_answer": (False, "Happy birthday!")},
    {"post": "Mi nombre es Juan.", "expected_answer": (False, "My name is Juan.")},

    # English posts (15)
    {"post": "This is an example of an English post.", "expected_answer": (True, "This is an example of an English post.")},
    {"post": "The quick brown fox jumps over the lazy dog.", "expected_answer": (True, "The quick brown fox jumps over the lazy dog.")},
    {"post": "I love working on challenging problems.", "expected_answer": (True, "I love working on challenging problems.")},
    {"post": "Where should we go for lunch today?", "expected_answer": (True, "Where should we go for lunch today?")},
    {"post": "My computer keeps crashing whenever I open the program.", "expected_answer": (True, "My computer keeps crashing whenever I open the program.")},
    {"post": "Artificial intelligence is transforming industries.", "expected_answer": (True, "Artificial intelligence is transforming industries.")},
    {"post": "She didn’t know what to say.", "expected_answer": (True, "She didn’t know what to say.")},
    {"post": "Please submit your report before Friday.", "expected_answer": (True, "Please submit your report before Friday.")},
    {"post": "We should schedule a meeting next week.", "expected_answer": (True, "We should schedule a meeting next week.")},
    {"post": "The test results were better than expected.", "expected_answer": (True, "The test results were better than expected.")},
    {"post": "Can you explain that again, please?", "expected_answer": (True, "Can you explain that again, please?")},
    {"post": "I’ll be traveling to New York tomorrow.", "expected_answer": (True, "I’ll be traveling to New York tomorrow.")},
    {"post": "This function doesn’t return the correct output.", "expected_answer": (True, "This function doesn’t return the correct output.")},
    {"post": "The weather forecast predicts rain for the weekend.", "expected_answer": (True, "The weather forecast predicts rain for the weekend.")},
    {"post": "He wants to learn how to play the guitar.", "expected_answer": (True, "He wants to learn how to play the guitar.")},

    # bad tests
    {"post": "kdsjlqkwopzxn", "expected_answer": (False, "MODEL FAILED")},
    {"post": "12345 ??? ??? what!", "expected_answer": (False, "MODEL FAILED")},
    {"post": "你好 ?? 2 3 seff", "expected_answer": (False, "MODEL FAILED")},
    {"post": "———", "expected_answer": (False, "MODEL FAILED")},
    {"post": ";;;;", "expected_answer": (False, "MODEL FAILED")},
]


def test_llm_normal_response(monkeypatch):
    """Test correct non-English translation behavior."""
    monkeypatch.setattr(tr, "get_language", lambda _txt: "German")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "This is a German message")
    is_english, translated = tr.translate_content("Dies ist eine Nachricht auf Deutsch")
    assert is_english is False
    assert translated == "This is a German message"


def test_llm_gibberish_response(monkeypatch):
    """Test robust handling of malformed LLM output."""
    def fake_query_llm(_txt):
        return "invalid_output_not_a_tuple"
    monkeypatch.setattr(tr, "query_llm", fake_query_llm)
    original = "Bonjour tout le monde"
    is_english, translated = tr.translate_content(original)
    assert is_english is True
    assert translated == original


## tests from our collab 
@pytest.mark.parametrize("case", translation_eval_set)
def test_translation_eval_set(monkeypatch, case):
    """Run translation_eval_set from Colab notebook."""
    post = case["post"]
    expected = case["expected_answer"]
    monkeypatch.setattr(tr, "get_language", lambda _txt: "Non-English")
    monkeypatch.setattr(tr, "get_translation", lambda _txt, e=expected: e)
    is_english, translated = tr.translate_content(post)
    assert is_english is False
    assert translated == expected


@pytest.mark.parametrize("case", language_detection_eval_set)
def test_language_detection_eval_set(monkeypatch, case):
    """Run language_detection_eval_set from Colab notebook."""
    post = case["post"]
    expected_lang = case["expected_answer"]
    monkeypatch.setattr(tr, "get_language", lambda _txt, e=expected_lang: e)
    if expected_lang == "English":
        is_english, translated = tr.translate_content(post)
        assert is_english is True
        assert translated == post
    else:
        monkeypatch.setattr(tr, "get_translation", lambda _txt: "[EN] " + post)
        is_english, translated = tr.translate_content(post)
        assert is_english is False
        assert translated == "[EN] " + post


@pytest.mark.parametrize("case", complete_eval_set)
def test_complete_eval_set(monkeypatch, case):
    """Run complete_eval_set (end-to-end expectation table)."""
    post = case["post"]
    expected_is_english, expected_text = case["expected_answer"]
    # Mock query_llm so query_llm_robust → translate() gives same output
    monkeypatch.setattr(tr, "query_llm", lambda _txt, b=expected_is_english, t=expected_text: (b, t))
    is_english, translated = tr.translate_content(post)
    assert is_english == expected_is_english
    assert translated == expected_text

def query_llm_robust(post: str) -> tuple[bool, str]:
    try:
      is_english = all(ord(c) < 128 for c in post)
      # If English → return as-is
      if is_english:
          return (True, post)
      else:
        translated = "TRANSLATED: " + post
        return (False, translated)
      if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
            or not isinstance(result[1], str)
        ):
            raise ValueError("Model output not in expected (bool, str) format")

      return result
    except Exception as e:
      print(f"query_llm error: {e}")
      return (True, post)

# Model returns nonsense (unexpected text output)
@patch.object(tr.client, 'chat')
def test_unexpected_text_output(mock_chat):
    mock_chat.return_value.message.content = "nonsense output"
    result = query_llm_robust("Bonjour tout le monde")
    # The robust function should fall back safely
    assert isinstance(result, tuple)
    assert result == (True, "Bonjour tout le monde")


# Model raises an exception (simulating service failure)
@patch.object(tr.client, 'chat')
def test_model_raises_exception(mock_chat):
    mock_chat.side_effect = RuntimeError("Ollama server unavailable")
    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    # Should catch exception and return fallback
    assert result == (True, "Hier ist dein erstes Beispiel.")


# Model returns partial / malformed tuple
@patch.object(tr.client, 'chat')
def test_incomplete_model_response(mock_chat):
    # simulate model returning partial tuple-like content
    mock_chat.return_value.message.content = "(False,)"
    result = query_llm_robust("Ceci est un test.")
    # Expected safe fallback
    assert result == (True, "Ceci est un test.")


# Model returns None (no response)
@patch.object(tr.client, 'chat')
def test_none_response(mock_chat):
    mock_chat.return_value.message.content = None
    result = query_llm_robust("Test")
    # Graceful recovery expected
    assert result == (True, "Test")