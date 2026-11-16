# Research Assistant: Simplified Rewrite and Learning Guide

## 1. What it does

You ask a question. The program picks an expert persona, writes search queries, searches DuckDuckGo, reads the pages, summarizes them, grades its own results, retries with better queries if the grade is low, and finally writes a markdown report.

```
START
  v
select_assistant        "You are a historian of Roman Spain..."
  v
generate_queries  <--------------------------------+
  v                                                |  score < 50 AND round < 3
search_web                                         |
  v                                                |
summarize_pages                                    |
  v                                                |
evaluate_results ---- route_after_evaluation ------+
                               | otherwise
                               v
                         write_report -> END
```

**Memory sentence:** *Pick expert, write queries, search, read, grade, retry or write.*

## 2. Project layout (read in this order)

| File | One-line job | Lines |
|---|---|---|
| `config.py` | settings + the LLM client | 31 |
| `state.py` | the shared notebook (`ResearchState`) | 16 |
| `prompts.py` | all prompt text | 59 |
| `tools.py` | the only code that touches the internet | 39 |
| `nodes.py` | the six steps + 2 parsing helpers | 138 |
| `graph.py` | wiring, router, `run_research()` | 78 |
| `tests/test_flow.py` | offline tests with a fake LLM | n/a |

Run it:
```bash
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env
python graph.py "What can you tell me about Astorga's Roman spas?"
python -m pytest -q         # tests need no key and no internet
```

## 3. What changed from the original, and why

| Original | Now | Why it is better |
|---|---|---|
| State held lists of dicts, each repeating `user_question` | State holds plain `list[str]` | No dict-inside-list indexing, no repeated data |
| Prompts asked the LLM for **JSON** and parsed it with `find('[')` | Queries come **one per line**, the judge replies `SCORE:` / `REASON:` | Free models often break JSON; plain lines are hard to get wrong |
| Three separate retry prompts, chosen by `iteration_count` | **One** prompt that always includes past queries and the judge's feedback | Less code, and every retry gets feedback |
| `is_fallback` and `used_fallback_search` flags | Removed | `web_search` never had a fallback, so the flags only fired when Wikipedia appeared in normal results |
| Counter incremented **inside the router** | Counter incremented **inside `generate_queries`**, router is read-only | **Fixes a real bug** (section 4) |
| Failed scrape returned the string `"Failed to..."`, checked with `startswith` | Failed scrape returns `""`, checked with `len(text) < 200` | One check covers errors and near-empty pages |
| If nothing was found, the report prompt got "No relevant information found" and the LLM wrote a report anyway | `write_report` returns an honest "no usable information" message | Avoids invented reports |
| Report prompt said "this is very important to my career" and "you MUST have an opinion" | Says "use ONLY the information below, say so if something is missing" | Pushes the model toward grounded answers |
| New `ChatOpenAI` object on every call | `@lru_cache` creates it once | Cheap and simple |
| `print` everywhere | `logging` | Levels, and can be silenced or redirected |

## 4. The bug that was in the original (verified)

In the original `graph.py`, the router did this:

```python
state["iteration_count"] = new_iteration_count   # inside the ROUTER
```

I tested the same pattern in a tiny graph. Result:

```
n seen by node each loop: [0, 0, 0, 0, 0, 0, 0, 0]
GraphRecursionError
```

The write was thrown away every time. In your project that means:
- `iteration_count` stayed 0, so the "be more specific" and "final attempt" prompts never ran
- the `>= 3` cap never fired, so a judge that kept saying "not relevant" would loop until LangGraph's step limit crashed it

**Rule to remember:** *nodes write state, routers only read state and return a node name.*

## 5. File-by-file walkthrough

### `config.py`

```python
load_dotenv()
```
Loads `.env` into environment variables. Runs once on import.

The constants (`NUM_QUERIES`, `MAX_ROUNDS`, ...) are named knobs. If you want 5 queries instead of 3, change one number. Nothing else needs editing.

```python
@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
```
`lru_cache` remembers the return value. The first call builds the client, and every later call returns the same object. The key is read **inside** the function, so tests can run without one until it is actually needed.

`base_url="http://localhost:3001/v1"` points LangChain's OpenAI client at your local proxy. `model="auto"` is a name the proxy understands, not a real OpenAI model.

### `state.py`

A `TypedDict` is a normal dict that editors know the keys of. Every node reads from and writes to this one shared notebook.

| Key | Written by | Read by |
|---|---|---|
| `assistant` | select_assistant | generate_queries, write_report |
| `queries` | generate_queries | search_web |
| `past_queries` | generate_queries | generate_queries (next round) |
| `urls` | search_web | summarize_pages |
| `summaries` | summarize_pages | evaluate_results, write_report |
| `relevance`, `feedback` | evaluate_results | router, generate_queries |
| `round` | generate_queries | router |
| `report` | write_report | `run_research` |

### `prompts.py`

Each prompt is a plain string with `{placeholders}`, filled with `.format(...)`. Because there is no JSON in the prompts, there are no `{{ }}` escapes to remember.

The two format tricks:
- `"one per line"` for queries
- `SCORE: <number>` / `REASON: <sentence>` for the judge

### `tools.py`

```python
return [r["link"] for r in _search_client().results(query, num_results) if "link" in r]
```
`.results()` returns a list of dicts, and the comprehension keeps each dict's `"link"`. The `if "link" in r` guard skips malformed results instead of crashing. Print one raw result to confirm the keys in your installed version.

`web_scrape` downloads the page (10 second timeout), deletes `script/style/nav/footer/header/aside` tags, takes the visible text, and cuts it to `max_chars`. **On any network error it returns `""`**, which the caller treats as "skip this page". I wrote this version myself because I never saw your original `web_scraping.py`, so compare it with yours.

### `nodes.py`

`ask(prompt)` is the single place that calls the LLM. Tests replace this one function to fake the whole LLM.

`parse_queries(text, n)`:
```python
re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line)
```
Removes a leading `1.`, `2)`, `-`, `*` or bullet. It deliberately does **not** strip bare digits, so the query `2026 ev market` keeps its year. The tests check this.

`parse_evaluation(reply)` finds `SCORE:` and `REASON:` with regex. If the reply is garbled the score becomes 0, which triggers a retry (when in doubt, try again).

The six nodes, each in one line:

| Node | What it returns |
|---|---|
| `select_assistant` | `{"assistant": "<persona text>"}` |
| `generate_queries` | `queries`, the extended `past_queries`, and `round + 1` |
| `search_web` | `{"urls": [...]}`, unique, skipping any query that raised an error |
| `summarize_pages` | `{"summaries": ["Source: url\n<summary>", ...]}` |
| `evaluate_results` | `{"relevance": 0-100, "feedback": "<reason>"}` |
| `write_report` | `{"report": "<markdown>"}` |

Three defensive habits repeat through the file:
1. **Narrow `try/except`** around only the risky call, with `continue` so one failure does not stop the rest.
2. **Quality gate** before using data (`len(text) < MIN_PAGE_CHARS`).
3. **Skip the LLM when there is nothing to judge or write**, which saves a call and avoids made-up output.

### `graph.py`

```python
def route_after_evaluation(state) -> Literal["generate_queries", "write_report"]:
    if state["relevance"] < MIN_RELEVANCE and state["round"] < MAX_ROUNDS:
        return "generate_queries"
    return "write_report"
```
Retry only when **both** are true: the score is low **and** rounds remain. The `Literal[...]` return type tells LangGraph the possible destinations, so no mapping dict is needed.

`build_graph()` is three steps: `add_node` for each box, `add_edge` for each arrow, `compile()` to make it runnable. `run_research()` builds the starting state and calls `app.invoke(...)`, which returns the **final state**, then picks out `["report"]`.

Step count: 1 + 3 rounds x 4 nodes + 1 = 14, under LangGraph's default limit of 25. If you raise `MAX_ROUNDS` above 5, also pass `{"recursion_limit": ...}` to `invoke`.

## 6. Trace one run (judge says 30, then 80)

| Step | Node | `round` | `relevance` | Note |
|---|---|---|---|---|
| 1 | select_assistant | 0 | 0 | persona written |
| 2 | generate_queries | 1 | 0 | `past_queries` has 3 queries |
| 3 | search_web | 1 | 0 | up to 9 URLs |
| 4 | summarize_pages | 1 | 0 | unreadable pages skipped |
| 5 | evaluate_results | 1 | 30 | feedback: "pages are about modern spas" |
| 6 | router | | | 30 < 50 and 1 < 3, so back to generate_queries |
| 7 | generate_queries | 2 | 30 | prompt includes past queries and the feedback |
| 8-10 | search, summarize, evaluate | 2 | 80 | |
| 11 | router | | | 80 is not below 50, so write_report |
| 12 | write_report | 2 | 80 | `report` filled, END |

## 7. What the tests prove

`tests/test_flow.py` replaces `ask`, `web_search` and `web_scrape` with fakes. All 9 pass, covering:
- good results finish in 1 round
- bad then good retries once, and the second prompt contains the judge's reason and the first round's queries
- a judge that is never satisfied stops at exactly 3 rounds
- all searches failing does not crash and yields an honest report
- unreadable pages are skipped
- query and score parsing, including a year at the start of a query

**What was not tested:** the real DuckDuckGo search, the real scraper, and your real LLM proxy. Run the program once with your key to check those.

## 8. Known limits (good next exercises)

1. **A retry replaces the first round's summaries.** If round 2 is worse than round 1, you lose round 1. Fix: accumulate summaries across rounds, or keep the best round.
2. **Sequential and slow.** Up to 9 scrapes and 9 summaries run one after another. Use `ThreadPoolExecutor` in `summarize_pages`.
3. **Same URL can come back on a retry.** Track a `seen_urls` list in state.
4. **Prompt injection.** Scraped page text goes straight into prompts. A hostile page could try to instruct your LLM. Production systems sanitize or isolate this.
5. **`langchain-community` is being sunset** (a warning appears on import). The DuckDuckGo wrapper may move to another package. The `ddgs` package can be called directly instead.
6. **The judge is the same model that wrote the summaries.** LLM-as-judge can be lenient, so a different or stronger model is better for evaluation.

## 9. Exercises (type them, do not paste)

1. Change `MAX_ROUNDS` to 2 and update the test that expects 3 query rounds.
2. Add a `seen_urls` key to the state so no URL is scraped twice across rounds.
3. Keep summaries from all rounds (`state["summaries"] + new`) and rerun the tests. Which one fails, and why?
4. Parallelize `summarize_pages` with `ThreadPoolExecutor`.
5. Add a test where the judge's reply is garbage and confirm it retries.
6. Replace the plain-line queries with `llm.with_structured_output(...)` and compare reliability on your proxy.
7. Draw the graph: `print(build_graph().get_graph().draw_mermaid())`, and paste the output into a Mermaid viewer.

## 10. Remember forever

- **State is a shared notebook. Nodes return only what changed.**
- **Routers only read.** Counters belong in nodes.
- **Ask the LLM for the simplest output you can parse.**
- **Every loop needs a hard cap.**
- **Isolate the internet in one file** so tests can fake it.
- **Test the logic offline with fakes.** Test the real services separately.
