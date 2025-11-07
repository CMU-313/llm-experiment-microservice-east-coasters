import os
from typing import Tuple
from ollama import Client

# Match Colab: talk to Ollama via its host
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")

client = Client(host=OLLAMA_HOST)


def get_translation(post: str) -> str:
    """
    From Colab: pure translation helper.
    Takes any language, returns ONLY English translation text.
    """
    context = """You are a language translator. Read the instructions below:

You are given a text in any language.
Respond only with the English translation.
Do not output ANYTHING else.
Do not talk about the original sentence.
Translate everything provided.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hi, my name is Bob
INPUT: Yo vivo en Pittsburgh
OUTPUT: I live in Pittsburgh

Translate ONLY the text below into English."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": post},
        ],
    )
    # In Colab you printed this; here we just return.
    return response.message.content.strip()


def get_language(post: str) -> str:
    """
    From Colab: language classification helper.
    Returns ONLY the English name of the language.
    """
    context = """You are a language classifier.

You are given a text in any language.
Respond only with the English name of the language that text is written in.
Do not output ANYTHING else.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French
INPUT: Yo vivo en Pittsburgh
OUTPUT: Spanish
INPUT: Bundesrepublik Deutschland
OUTPUT: German

Classify ONLY the below text:"""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": post},
        ],
    )
    return response.message.content.strip()


def query_llm(post: str) -> Tuple[bool, str]:
    """
    Uses get_language + get_translation to produce (is_english, text).

    - If language is English -> (True, original_post)
    - Else -> (False, english_translation)
    """
    lang = get_language(post).strip().lower()

    if lang == "english":
        return True, post

    # Anything else: treat as non-English and translate.
    translated = get_translation(post).strip()
    return False, translated


def query_llm_robust(post: str) -> Tuple[bool, str]:
    """
    Robust version (matches the PDF's intent):
    - Calls query_llm
    - Ensures output is (bool, str)
    - On any error or malformed output, falls back to (True, post)
      so NodeBB / the Flask app never breaks.
    """
    try:
        result = query_llm(post)

        # Validate format is exactly (bool, str)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
            or not isinstance(result[1], str)
        ):
            raise ValueError("Model output not in expected (bool, str) format")

        return result

    except Exception as e:
        # Safe fallback: treat as English and show original text.
        # This is exactly the “graceful degradation” behavior they want.
        print(f"query_llm_robust error: {e}")
        return True, post
