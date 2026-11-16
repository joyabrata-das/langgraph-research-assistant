"""The shared 'notebook' every node reads from and writes to."""

from typing import TypedDict


class ResearchState(TypedDict):
    question: str             # the user's question (never changes)
    assistant: str            # expert persona instructions, e.g. "You are a historian..."
    queries: list[str]        # search queries for the CURRENT round
    past_queries: list[str]   # every query tried so far (so retries don't repeat them)
    urls: list[str]           # pages found this round
    summaries: list[str]      # "Source: <url>\n<summary>" for each readable page
    relevance: int            # judge's score, 0-100
    feedback: str             # judge's one-sentence reason, fed into the next retry
    round: int                # how many query rounds have started
    report: str               # the final answer
