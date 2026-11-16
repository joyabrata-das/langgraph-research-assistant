"""The six graph nodes.

Every node has the same shape:   state in  ->  dict of CHANGED keys out.
LangGraph merges the returned dict into the shared state.
"""

import logging
import re

from config import (
    MAX_PAGE_CHARS,
    MIN_PAGE_CHARS,
    NUM_QUERIES,
    RESULTS_PER_QUERY,
    get_llm,
)
from prompts import (
    ASSISTANT_PROMPT,
    EVALUATE_PROMPT,
    QUERIES_PROMPT,
    REPORT_PROMPT,
    SUMMARY_PROMPT,
)
from state import ResearchState
from tools import web_scrape, web_search

log = logging.getLogger(__name__)


# ---- Helpers (small, pure, easy to test) --------------------------------

def ask(prompt: str) -> str:
    """Send one prompt to the LLM and return its text. The ONLY place we call the LLM."""
    return get_llm().invoke(prompt).content.strip()


def parse_queries(text: str, n: int) -> list[str]:
    """Turn 'one query per line' text into a clean list of at most n queries."""
    queries = []
    for line in text.splitlines():
        # strip list markers like "1." "2)" "- " "* " but NOT a query that starts with a year
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip(' "\'')
        if line:
            queries.append(line)
    return queries[:n]


def parse_evaluation(reply: str) -> tuple[int, str]:
    """Pull (score, reason) out of 'SCORE: 80 / REASON: ...'. Unparseable -> (0, ...)."""
    score_match = re.search(r"SCORE:\s*(\d+)", reply)
    reason_match = re.search(r"REASON:\s*(.+)", reply)
    score = min(int(score_match.group(1)), 100) if score_match else 0
    reason = reason_match.group(1).strip() if reason_match else "No reason given."
    return score, reason


# ---- Nodes ----------------------------------------------------------------

def select_assistant(state: ResearchState) -> dict:
    """Ask the LLM for an expert persona that suits the question."""
    return {"assistant": ask(ASSISTANT_PROMPT.format(question=state["question"]))}


def generate_queries(state: ResearchState) -> dict:
    """Write search queries. On retries the prompt includes past queries + the judge's feedback."""
    prompt = QUERIES_PROMPT.format(
        assistant=state["assistant"],
        question=state["question"],
        n=NUM_QUERIES,
        past=", ".join(state["past_queries"]) or "none",
        feedback=state["feedback"] or "none",
    )
    queries = parse_queries(ask(prompt), NUM_QUERIES) or [state["question"]]
    return {
        "queries": queries,
        "past_queries": state["past_queries"] + queries,
        "round": state["round"] + 1,  # the counter is updated HERE, inside a node
    }


def search_web(state: ResearchState) -> dict:
    """Run every query, collect unique URLs. One failing query never stops the others."""
    urls: list[str] = []
    for query in state["queries"]:
        try:
            found = web_search(query, RESULTS_PER_QUERY)
        except Exception as e:  # network error, rate limit, ...
            log.warning("Search failed for %r: %s", query, e)
            continue
        urls.extend(u for u in found if u not in urls)
    return {"urls": urls}


def summarize_pages(state: ResearchState) -> dict:
    """Scrape each URL and ask the LLM for a short summary tied to the question."""
    summaries = []
    for url in state["urls"]:
        text = web_scrape(url, MAX_PAGE_CHARS)
        if len(text) < MIN_PAGE_CHARS:
            log.info("Skipping %s (empty or unreadable)", url)
            continue
        try:
            summary = ask(SUMMARY_PROMPT.format(text=text, question=state["question"]))
        except Exception as e:
            log.warning("Summary failed for %s: %s", url, e)
            continue
        summaries.append(f"Source: {url}\n{summary}")
    return {"summaries": summaries}


def evaluate_results(state: ResearchState) -> dict:
    """LLM-as-judge: score how useful the summaries are and say why."""
    if not state["summaries"]:  # nothing to judge, so skip the LLM call
        return {"relevance": 0, "feedback": "No pages could be read. Try different, simpler queries."}

    prompt = EVALUATE_PROMPT.format(
        question=state["question"], summaries="\n\n".join(state["summaries"])
    )
    try:
        score, reason = parse_evaluation(ask(prompt))
    except Exception as e:
        log.warning("Evaluation failed: %s", e)
        score, reason = 0, "The evaluation failed; try different queries."
    log.info("Relevance score: %s (%s)", score, reason)
    return {"relevance": score, "feedback": reason}


def write_report(state: ResearchState) -> dict:
    """Write the final markdown report from the summaries."""
    if not state["summaries"]:  # be honest instead of letting the LLM invent an answer
        return {"report": "No usable information could be found for this question."}

    prompt = REPORT_PROMPT.format(
        assistant=state["assistant"],
        question=state["question"],
        summaries="\n\n".join(state["summaries"]),
    )
    return {"report": ask(prompt)}
