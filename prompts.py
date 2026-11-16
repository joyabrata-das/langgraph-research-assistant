"""All prompt text in one place. Plain strings filled in with .format(...).

Design rule: ask the LLM for the *simplest* output we can parse.
- queries  -> one per line (no JSON)
- judge    -> two lines: "SCORE: 80" and "REASON: ..."
Free models often break JSON; plain lines are much harder to get wrong.
"""

ASSISTANT_PROMPT = """Write 2 sentences of instructions for the AI research assistant \
best suited to answer this question. Start with "You are". Return only the instructions.

Question: {question}"""

QUERIES_PROMPT = """{assistant}

Write {n} web search queries that will gather the information needed to answer this question:
{question}

Queries already tried (do NOT repeat or rephrase them): {past}
Feedback on the last attempt: {feedback}

Return exactly {n} queries, one per line. No numbering, no quotes, no extra text."""

SUMMARY_PROMPT = """Read the text below and answer the question in a few sentences. \
If the text cannot answer it, summarize the text instead. Keep all facts, numbers and names.

Text:
{text}

Question: {question}"""

EVALUATE_PROMPT = """You are a strict research evaluator.

Question: {question}

Summaries of the pages found:
{summaries}

What percentage of these summaries contain information that helps answer the question?

Reply in exactly this format:
SCORE: <number from 0 to 100>
REASON: <one sentence saying what worked or what is missing>"""

REPORT_PROMPT = """{assistant}

Use ONLY the information below to answer the question. If something is not covered, \
say so instead of guessing.

Information:
{summaries}

Question: {question}

Write a detailed, well-structured report in markdown:
- start with a short direct answer, then sections with headings
- include facts and numbers when available
- finish with your own reasoned conclusion
- end with a "Sources" section listing each URL once"""
