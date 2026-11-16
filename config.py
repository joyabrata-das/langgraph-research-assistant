"""Settings and the LLM client. Everything you might want to tune lives here."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads FREELLMAPI_API_KEY from a local .env file

# ---- Tunable knobs -------------------------------------------------------
NUM_QUERIES = 3          # search queries generated per round
RESULTS_PER_QUERY = 3    # URLs requested per query
MAX_PAGE_CHARS = 10_000  # cut each web page to this many characters
MIN_PAGE_CHARS = 200     # pages shorter than this are treated as "unreadable"
MIN_RELEVANCE = 50       # below this score (0-100) we retry with new queries
MAX_ROUNDS = 3           # hard cap on query rounds, so the loop always ends

BASE_URL = os.getenv("FREELLMAPI_BASE_URL", "http://127.0.0.1:31415/v1")
@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Create the LLM client once and reuse it on every call."""
    api_key = os.getenv("FREELLMAPI_API_KEY")
    if not api_key:
        raise ValueError("FREELLMAPI_API_KEY is missing. Add it to your .env file.")

    return ChatOpenAI(
        api_key=api_key,
        base_url=BASE_URL,  # local proxy that speaks the OpenAI API
        model="auto",
    )
if __name__ == "__main__":
    print("NUM_QUERIES =", NUM_QUERIES)
    print("client:", type(get_llm()).__name__)
