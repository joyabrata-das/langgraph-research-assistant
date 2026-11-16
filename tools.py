"""The two things that touch the internet: searching and scraping.

Keeping them here means the rest of the code never imports a network library,
and tests can replace these two functions with fakes.
"""

from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


@lru_cache(maxsize=1)
def _search_client() -> DuckDuckGoSearchAPIWrapper:
    return DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int) -> list[str]:
    """Return up to num_results URLs from DuckDuckGo. Raises on network errors."""
    results = _search_client().results(query, num_results)  # list of dicts
    return [r["link"] for r in results if "link" in r]


def web_scrape(url: str, max_chars: int) -> str:
    """Return the visible text of a page (cut to max_chars), or "" if it fails."""
    try:
        response = requests.get(
            url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (research-assistant)"}
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()  # remove non-content parts
    text = soup.get_text(separator=" ", strip=True)
    return text[:max_chars]
