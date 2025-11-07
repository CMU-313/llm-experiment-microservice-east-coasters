from typing import Tuple
from .llm_client import query_llm_robust


def translate(content: str) -> Tuple[bool, str]:
    """
    Adapter for the Flask app.

    Uses the robust LLM pipeline from the Colab (query_llm_robust)
    to return (is_english, translated_content_if_non_english_or_original).
    """
    if content is None:
        return True, ""

    return query_llm_robust(content)
