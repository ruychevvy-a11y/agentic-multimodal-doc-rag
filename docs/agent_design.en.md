# Agent Design

<p>
  <a href="./agent_design.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-DFE0E5"></a>
  <a href="./agent_design.en.md"><img alt="English" src="https://img.shields.io/badge/English-DBEDFA"></a>
</p>

On top of the baseline "retrieve --> view the top 4 page images --> answer once", the model gets three tools so that, when evidence is missing, it can search again and turn pages on its own before giving an answer with cited pages.

## Motivation

Retrieval ranks every gold page into the top 4 for 73.1% of questions. For the rest, the baseline has to answer with incomplete evidence: on a held-out set of 200 questions, questions with incomplete starting evidence score only 0.296 and multi-page questions 0.451, against 0.699 for questions with complete evidence.

Adding pages across the board does not pay off: k=6 beats k=4 by only +0.015 overall (not significant), all of the gain comes from questions with incomplete evidence (+0.134\*), unanswerable questions get answered more often, and cost rises 30%. What is needed is effort on demand — easy questions finish after one answer, and only questions that really lack evidence trigger a search.

## Structure

```
Start: system instruction + user (question + top 4 retrieved page images, each labelled [Page N])
  │
  ├─ Loop, at most 5 model calls:
  │    Did the model return tool_calls?
  │      search_pages(query)      --> search within this document --> "page number + query-relevant excerpt" for the top 5 pages (text only)
  │      view_pages(page_numbers) --> validate page numbers, skip pages already seen, at most 4 pages per call
  │                                   --> the tool message returns text; page images go in the user message right after it
  │      answer(answer, cited_pages) --> record the answer and cited pages, stop
  │    No tool_calls --> the message text is the answer, stop
  │
  └─ 5 calls used up or a single call timed out --> fallback: one more call with tool_choice="none",
                                                   answering only from pages already seen (Not answerable allowed)
```

Every step emits a structured step record — model call, tool call or final answer. Evaluation computes process metrics from these records, and the demo streams them straight to the frontend.

## The three tools

| Tool | Arguments | Returns | Rationale |
|---|---|---|---|
| `search_pages` | `query` | Page numbers of the top 5 pages with a one-line query-relevant excerpt each; pages already viewed are marked | Text only, so it is cheap. The model uses it to decide which pages are worth viewing before loading any image |
| `view_pages` | `page_numbers` | Text confirmation + page images | Accepts any page number, not only search results: when a table says "continued on next page" it can view the next page, and when a table of contents says a section starts on page 45 it can jump there |
| `answer` | `answer`, `cited_pages` | Ends the question | A question only ends through this tool, so the stopping condition is explicit; cited pages come out structured and can be checked against the evidence directly |

Excerpts are taken from the window most relevant to the query rather than from the start of the page: page text usually opens with a sentence continued from the previous page or with a header — every page of a 10-K starts with "Table of Contents" — which does not help choose pages.

## Limits and fallback

| Parameter | Value | Rationale |
|---|---|---|
| Model calls per question | 5 | Bounds the worst-case cost; in practice only 1.59 on average |
| Pages per `view_pages` call | 4 | One page image is about 2500 tokens, and the history is resent every round |
| Total pages viewed per question | 12 | Among questions still missing gold pages outside the top 4, 21.8% miss one page and fewer than 3% miss five or more; 12 pages cover 99.4% |
| Search results returned | 5 | Closed-domain reranked recall@5 is 0.796 and @10 is 0.871, so the top 5 already hold about 90% of the top-10 hits |
| Timeout per call | 45 s | No retry on timeout; go straight to the fallback so one question cannot block for minutes |
| Thinking budget | 1024 tokens | Ties default thinking on accuracy while p90 drops from 68 s to 15.6 s |

Errors are never raised; they are returned to the model as text so it can correct itself: arguments that are not valid JSON, an unknown tool name, a page number out of range, a repeated search for the same query (the previous result is returned without searching again), or viewing a page already shown (the model is told the page is already in the conversation). Content-moderation blocks and rate limits use the shared retry-with-backoff policy.

## Trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Starting point | Give the baseline's top 4 pages first; tools only add to them | Easy questions end after a single call at about the baseline's cost; starting from the question alone and relying fully on the agent's own search would make easy questions take several rounds too |
| Number of agents | A single agent | Single-document QA has no sub-tasks that need dividing up; multiple agents would only multiply calls and add conflict handling |
| Calling style | Function calling | The model natively returns structured tool_calls, so no hand-written `Action:` parser is needed |
| How it ends | Must call `answer` | Explicit stopping condition; cited pages come out structured |
| Memory | Only the conversation within one question | Questions are independent; there is nothing reusable across questions |
| Framework | A hand-written loop of about 100 lines | Needs per-step logging of tokens, time and tool failures, control over limits and fallback, and the same scoring as the baseline |
| MCP | Not used | The tools run in the same process with a single caller, whereas MCP solves integration across many parties |

## Implementation notes

- **Tool messages cannot carry images**: the API returns 400, so page images must go in a user message immediately after the tool message.
- **The retriever can be shared across threads**: one retriever serving 240 questions on 12 threads returns candidates bit-for-bit identical to a sequential run, so evaluation can run concurrently and the demo can call retrieval from worker threads.

## Results

Paired against one-shot answering on 200 held-out questions (same retrieval):

| Route | All | Multi-page | Incomplete starting evidence | Unanswerable |
|---|---|---|---|---|
| One-shot answering | 0.645 | 0.451 | 0.296 | 0.805 |
| Agent | 0.655 | 0.556 | 0.424 | 0.707 |

Overall the two tie; the gains concentrate on multi-page questions and questions with incomplete evidence, at the cost of answering unanswerable questions more often. Only 28% of the 200 questions triggered a search, the fallback forced an answer once, there were no tool errors, and each question cost ¥0.0234 against ¥0.0111 for one-shot answering on the same questions.

Two enhancements were also evaluated: first, having the model verify the evidence after answering and refusing if it is not supported; second, letting the agent search one more round when verification fails. The first tied overall and missed the pre-set criterion; the second gave +0.026\* over 400 pooled questions but raised the cost to ¥0.032 per question and p90 to 54 s. Neither was adopted. Full numbers are in [results_summary.en.md](results_summary.en.md).
