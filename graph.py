"""Wires the nodes into a flowchart and runs it.

    START -> select_assistant -> generate_queries -> search_web
          -> summarize_pages -> evaluate_results --+-- score low and rounds left --> generate_queries
                                                   +-- otherwise ------------------> write_report -> END
"""

import logging
import sys
from typing import Literal

from langgraph.graph import END, START, StateGraph

from config import MAX_ROUNDS, MIN_RELEVANCE
from nodes import (
    evaluate_results,
    generate_queries,
    search_web,
    select_assistant,
    summarize_pages,
    write_report,
)
from state import ResearchState


def route_after_evaluation(state: ResearchState) -> Literal["generate_queries", "write_report"]:
    """Read-only decision: retry with new queries, or write the report?

    A router only READS state and returns a node name. It never changes state
    (changes made here would be lost). The counter lives in generate_queries.
    """
    if state["relevance"] < MIN_RELEVANCE and state["round"] < MAX_ROUNDS:
        return "generate_queries"
    return "write_report"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("select_assistant", select_assistant)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("search_web", search_web)
    graph.add_node("summarize_pages", summarize_pages)
    graph.add_node("evaluate_results", evaluate_results)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "select_assistant")
    graph.add_edge("select_assistant", "generate_queries")
    graph.add_edge("generate_queries", "search_web")
    graph.add_edge("search_web", "summarize_pages")
    graph.add_edge("summarize_pages", "evaluate_results")
    graph.add_conditional_edges("evaluate_results", route_after_evaluation)
    graph.add_edge("write_report", END)

    return graph.compile()


def run_research(question: str) -> str:
    app = build_graph()
    initial_state: ResearchState = {
        "question": question,
        "assistant": "",
        "queries": [],
        "past_queries": [],
        "urls": [],
        "summaries": [],
        "relevance": 0,
        "feedback": "",
        "round": 0,
        "report": "",
    }
    return app.invoke(initial_state)["report"]


def ask_for_question() -> str:
    """Keep asking until the user types a real (non-empty) question."""
    while True:
        question = input("What would you like to research? ").strip()
        if question:
            return question
        print("Please type a question (or press Ctrl+C to quit).")

def save_report(question: str, report: str) -> str:
    """Write the report to a .txt file and return the file name."""
    safe = "".join(c if c.isalnum() else "_" for c in question)[:40].strip("_")
    filename = f"report_{safe}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"QUESTION: {question}\n{'=' * 60}\n\n{report}\n")
    return filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        question = ask_for_question()
    except (KeyboardInterrupt, EOFError):  # Ctrl+C, or no input available at all
        sys.exit("\nNo question given. Bye!")
    report = run_research(question)
    print(report)
    print(f"\nSaved to: {save_report(question, report)}")
