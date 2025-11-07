import os
from typing import Tuple
from ollama import Client

# ==== OLLAMA CLIENT SETUP (matches your Colab) ====

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

client = Client(host=OLLAMA_HOST)


# ==== FUNCTIONS BASED ON YOUR COLAB NOTEBOOK ====


def get_translation(post: str) -> str:
    """
    Use Ollama to translate arbitrary text into English.
    Logic adapted directly from your Project 4 Colab.
    """
    context = """You are a language translator. Read the instructions below for how you should operate:

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
    # In the notebook you printed; here we just return
    return (response.message.content or "").strip()


def get_language(post: str) -> str:
    """
    Use Ollama to classify the language.
    Logic adapted directly from your Project 4 Colab.
    """
    context = """You are a language classifier. Read the instructions below for how you should operate:

You are given a text in any language.
Respond only with the English name of the language that text is written in.
Do not output ANYTHING else.
Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French
INPUT: Yo vivo en Pittsburgh
OUTPUT: Spanish
Input: Bundesrepublik Deutschland
Output: German

Classify ONLY the below text:"""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": post},
        ],
    )
    return (response.message.content or "").strip()


def query_llm(post: str) -> Tuple[bool, str]:
    """
    Core behavior: detect language, translate if needed.
    This is the 'python service' you developed in the Colab.

    - If language is English: (True, original)
    - Else: (False, English translation)
    """
    lang = get_language(post).strip().lower()

    if lang.startswith("english"):
        return True, post

    translated = get_translation(post).strip()
    return False, translated


def query_llm_robust(post: str) -> Tuple[bool, str]:
    """
    Robust wrapper around query_llm, per your Colab design:

    - Ensures output is a (bool, str) tuple.
    - If the LLM / pipeline misbehaves (exception or wrong format),
      fall back to (True, post) so NodeBB / the service never breaks.

    This is exactly the behavior you described in the notebook.
    """
    try:
        result = query_llm(post)

        # Validate format from LLM pipeline
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], bool)
            or not isinstance(result[1], str)
        ):
            raise ValueError("Model output not in expected (bool, str) format")

        return result

    except Exception as e:
        # Log for debugging; fall back safely
        print(f"query_llm_robust error: {e}")
        return True, post


# ==== PUBLIC ENTRYPOINT REQUIRED BY ASSIGNMENT ====


def translate(content: str) -> Tuple[bool, str]:
    """
    This is the function the Flask app and NodeBB integration will call.

    Requirements:
      - Takes a string content.
      - Returns (is_english: bool, translated_content_or_original: str).
      - Uses your LLM-based python service (query_llm_robust).
    """
    if content is None:
        return True, ""

    content = str(content)
    if not content.strip():
        return True, content

    return query_llm_robust(content)
