# Results Summary

<p>
  <a href="./results_summary.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-DFE0E5"></a>
  <a href="./results_summary.en.md"><img alt="English" src="https://img.shields.io/badge/English-DBEDFA"></a>
</p>

Every number in the README comes from here; raw results are in `evaluations/<dataset>/<closed|open>/results_*.json`. All comparisons are paired bootstrap 95% confidence intervals over the same questions; **\* means the interval excludes 0**.

## Data

| Item | Count |
|---|---|
| MMLongBench-Doc | 135 PDFs / 6522 pages / 1082 questions; 223 unanswerable; 845 with valid gold pages, used for retrieval evaluation |
| Page source | PDF text layer 4775 pages / OCR 1747 pages |
| Visual descriptions | 5377 pages contain describable charts or photos, 1145 do not |
| MP-DocVQA (parameter selection) | 2000 questions / 2740 pages |

## Retrieval

Closed-domain, 845 questions (retrieval only within the question's own document):

| Route | r@1 | r@10 | MRR | All gold pages in top 4 |
|---|---|---|---|---|
| BM25 | 0.433 | 0.809 | 0.670 | 0.604 |
| Text dense | 0.419 | 0.804 | 0.659 | 0.569 |
| Image dense | 0.258 | 0.695 | 0.477 | 0.408 |
| Three-way convex combination (0.25 / 0.55 / 0.20) | 0.465 | 0.847 | 0.711 | 0.632 |
| **+ reranking (top 30 candidates, current system)** | **0.581** | **0.897** | **0.826** | **0.731** |
| Control: page text without visual descriptions | 0.526 | 0.872 | 0.776 | 0.675 |

- **Reranking gives the largest gain**: convex combination --> reranking r@1 +0.116, far above the +0.032 that three-way fusion adds over the best single route.
- **Visual descriptions come second**: r@1 +0.055\*, all gold pages in top 4 +0.057\*; figure evidence 0.601 --> **0.718 (+0.117\*)**. Describing the whole corpus once costs ¥4.32.
- **Image dense is the weakest route but stays**: removing it lowers "all gold pages in top 30" by 0.009\*.
- **Open-domain is much harder** (before visual descriptions): reranked r@1 0.437 / MRR 0.618, convex 0.358 / 0.531, BM25 0.309 / 0.452.
- **Parsing is not the bottleneck** (before visual descriptions): r@1 0.498 when all gold pages are OCR pages, 0.540 when all are text-layer pages.

## Answering

All 1082 questions, scored with the dataset's official extraction and scoring code. One-shot answering = the top 4 reranked page images go to the vision model; it is the baseline for every comparison below. Both routes use a 1024-token thinking budget.

| Question type | Questions | Acc |
|---|---|---|
| **All** (F1 0.628) | 1082 | **0.643** |
| Unanswerable | 223 | 0.749 |
| Single-page | 494 | 0.711 |
| Layout evidence | 119 | 0.640 |
| Table evidence | 218 | 0.623 |
| Figure evidence | 304 | 0.594 |
| Plain-text evidence | 305 | 0.581 |
| Chart evidence | 178 | 0.568 |
| Multi-page | 372 | 0.488 |

| Evidence type | Single-page Acc | Multi-page Acc |
|---|---|---|
| Table | 0.772 (n=104) | 0.483 (n=113) |
| Layout | 0.726 (n=56) | 0.572 (n=62) |
| Chart | 0.686 (n=98) | 0.423 (n=80) |
| Figure | 0.690 (n=174) | 0.467 (n=127) |
| Plain text | 0.660 (n=160) | 0.482 (n=138) |

- **Losses depend on spanning pages, not on evidence type**: every type drops 0.15–0.29 from single-page to multi-page.
- **Plain text and figures are weaker on single pages**: plain text 0.660, chart 0.686, figure 0.690, all below table 0.772.
- **Visual descriptions add +0.018\* to answering** (paired, measured before the thinking budget was limited).

## Answer routes and routing

### The two routes differ structurally

Same retrieval, same starting 4 pages, paired per question:

| Route | All (1082) | Single-page (501) | Multi-page (358) | Incomplete starting evidence (228) | Unanswerable (223) |
|---|---|---|---|---|---|
| One-shot answering | 0.643 | 0.707 | 0.488 | 0.243 | 0.749 |
| Agent | 0.655 | 0.710 | 0.625 | 0.483 | 0.578 |
| Difference | +0.012 | +0.004 | **+0.136\*** | **+0.240\*** | **−0.170\*** |

- **The overall tie is two strong effects cancelling out** (+0.012 [−0.011, +0.035]).
- **Held-out check**: the agent prompt was chosen on 200 questions; on the other 882, multi-page +0.130\*, incomplete starting evidence +0.228\*, unanswerable −0.198\*, overall −0.003. On the 200 selection questions it was +0.078\*, so judging by the selection set alone overstates the gain by about 8 points.
- **The cost is fewer refusals**: the agent says "not in the document" on only 170 questions against 256 for one-shot answering; it keeps searching and answers even without evidence.
- **Overhead falls only where needed**: 25% of questions trigger a search, 1.59 calls on average, 745 questions end after one call, 0.45 extra pages viewed on average, 16 fallbacks, 2 tool errors.
- **Run-to-run variance**: two runs of the same configuration score 0.661 / 0.642 with 11% of questions flipping, so only differences above about 0.02 are trusted.

### Router results

10 answering-process signals --> logistic regression --> which route to trust:

| Route | All (1082) | Single-page (501) | Multi-page (358) | Unanswerable (223) |
|---|---|---|---|---|
| One-shot answering | 0.643 | 0.707 | 0.488 | 0.749 |
| Agent | 0.655 | 0.710 | 0.625 | 0.578 |
| Trust the agent only if it searched | 0.662 | 0.715 | 0.572 | 0.691 |
| **Classifier router** | **0.677** | 0.709 | 0.618 | 0.700 |

- **Significance**: classifier vs one-shot **+0.034\*** [+0.019, +0.050], vs the hand rule +0.015\* [+0.003, +0.026]; hand rule vs one-shot +0.019\*; on the 882 held-out questions the router gives +0.029\*.
- **Protocol**: 5-fold out-of-fold, threshold fixed at 0.5; trained only on the 190 questions where the two routes score differently (training on all questions drops it to 0.667); over 20 different fold splits the mean is 0.679 (0.675–0.684). Computed offline from the two run files, with no extra API calls.
- **Robustness**: resampling the agent on 199 questions drops the bare agent from 0.720 to 0.686, while the hand rule (0.687) and the classifier (0.702) stay unchanged.
- **Labelling cost**: 10 labels per fold +0.024, 20 labels +0.030, all labels +0.034; a label says which of two answers is better, no gold answer needed.
- **Learned weights**: pages viewed and model calls are largest and positive; a refusal from either side is negative. Without a weights file the router falls back to "trust the agent only if it searched".

### Failure breakdown (classifier routing, all 1082 questions)

Acc 0.677, 349.2 points lost in total (one-shot answering loses 386.2). Unanswerable questions are grouped first, then questions are split by the number of distinct evidence pages, which differs slightly from the official grouping.

| Question type | Questions | Acc | Points lost | Share of losses |
|---|---|---|---|---|
| Multi-page | 370 | 0.619 | 140.9 | 40.3% |
| Single-page | 489 | 0.711 | 141.3 | 40.5% |
| Unanswerable | 223 | 0.700 | 67.0 | 19.2% |

| | Incomplete starting evidence | Complete starting evidence |
|---|---|---|
| Single-page | 36.1 (10.3%) | **105.3 (30.1%)** |
| Multi-page | **89.8 (25.7%)** | 46.6 (13.3%) |
| Unanswerable | — | 67.0 (19.2%), answered when it should refuse |

- **Multi-page is a retrieval problem**: 25.7% of losses come from evidence not fitting into the top 4; routing cuts multi-page losses from 189.5 to 140.9.
- **Single-page is an answering problem**: wrong answers despite complete evidence make up 30.1%, the largest single block.
- **Unanswerable is a refusal problem**: 19.2%; none of the fixes below worked.

| Answer format | Questions | Acc | Share of losses | | Evidence type | Questions | Acc | Share of losses |
|---|---|---|---|---|---|---|---|---|
| Int | 299 | 0.629 | 31.8% | | Pure-text | 305 | 0.633 | 32.1% |
| List | 151 | 0.549 | 19.5% | | Figure | 304 | 0.646 | 30.8% |
| Unanswerable | 223 | 0.700 | 19.2% | | Table | 218 | 0.678 | 20.1% |
| Str | 250 | 0.740 | 18.6% | | Chart | 178 | 0.618 | 19.5% |
| Float | 159 | 0.761 | 10.9% | | Layout | 119 | 0.707 | 10.0% |

- **Int is 0.13 below Float**: integer questions are mostly counting, not reading a number printed on the page.
- **List is lowest at 0.549**: a list of the wrong length scores 0 outright.
- **Plain text accounts for the most losses (32.1%)**: its Acc of 0.633 is close to figures' 0.646, so the remaining bottleneck is no longer only "can it see it" but also "does it understand it".

### Evaluated but not adopted

| Method | What it does | Result |
|---|---|---|
| Stronger prompt | Look at page numbers named in the question first; answer only from viewed pages | Answerable +0.118\*, unanswerable −0.341\* |
| Cascade | Hand over to the agent only when one-shot answering says "not in the document" | 0.637 (−0.006): 167 of the 256 escalated questions are truly unanswerable, unanswerable −0.188\* — the gate sends exactly the wrong questions to the agent |
| Quote the source text | The answer must quote page text verbatim; bounced once if it does not match | 0.705: answers often live in charts, so the text layer cannot match and correct answers get rejected |
| Targeted visual verification | For answers given after a search, look at the cited page images again | 0.684: of 6 overturned answers only 1 was right; only 5 of 19 over-confident answers triggered it |
| Search before answering | Bounce an answer given without any search | 0.678, unanswerable 0.512: searching is a result of doubt, not its cause |
| Calculator tool | Four arithmetic operations and rounding (`ast` whitelist evaluation) | 458 Int/Float questions +0.013 [−0.013, +0.041]: losses come from accounting definitions, not arithmetic |
| Evidence verification | After answering, check whether the cited pages support the answer; refuse if not | Unanswerable +0.122\*, answerable −0.036\*, tie |
| Verification + re-search | When verification fails, return the reason to the agent for one more round of search | +0.026\* over 400 pooled questions, but ¥0.032 per question and p90 54 s |

Source-text quoting, visual verification and search-before-answering are compared with the current agent on the same 200 questions (0.717, unanswerable 0.585); raw results are in `answer_experiments/agent_variants/`.

## Cost and latency (per question)

Measured on the full set, priced at qwen3.8-flash rates, excluding evaluation-time answer extraction (about ¥0.002):

| Setup | Cost | p50 / p90 |
|---|---|---|
| One-shot answering | ¥0.0093 | 8.7 s / 15.0 s |
| Agent | ¥0.0172 | 11.1 s / 30.8 s |
| **Both routes in parallel + router** | **¥0.0265** | 11.9 s / 31.2 s |
| Agent + verification + re-search (not adopted) | ¥0.0318 | 21.0 s / 54.2 s |
| One rerank (30 candidates) | ¥0.0096 | — |
| Follow-up rewriting | < ¥0.001 | about 1 s |
| Visual descriptions (whole corpus, once) | ¥4.32 / 6522 pages | — |

- **Both routes in parallel**: in the demo the router waits at most 10 s for one-shot answering after the agent finishes; in a simulation over the full set only 1% of questions hit the limit. Run sequentially it would be 20.9 s / 42.7 s.
- **The agent's cost is input**: every extra round resends the page images viewed so far, so capping total viewed pages works better than capping the number of calls.

## Parameters and model choices

| Choice | Evidence |
|---|---|
| Convex combination, not RRF | MP-DocVQA, 500 questions, r@1: convex 0.486 > BM25 0.458 > RRF 0.420 |
| Add a cross-encoder reranker | r@1 +0.132\* over convex on MP-DocVQA, +0.116 on MMLongBench |
| Rerank the top 30 candidates | Top 30 and top 50 tie; 30 saves cost |
| Convex weights (0.25 / 0.55 / 0.20) | Re-tuning with 2-fold cross-validation was not significant and the two folds picked different weights, so the original values were kept |
| Answer from 4 pages | k=6 gives +0.015 overall, not significant; only questions with incomplete evidence gain (+0.134\*), unanswerable questions get answered more often, and cost rises 30% |
| 1024-token thinking budget (shared by both routes) | 200 questions: default 0.628 / 1024 0.612 / off 0.613, a tie; p90 68.0 s --> 15.6 s |
| Generation model qwen3.8-flash | 200 questions: 3.8-flash 0.629 > 3.7-flash 0.593 > 3.7-plus 0.588 > qwen3-vl-plus 0.568 |
| OCR with PP-OCRv6_medium | The larger v5_server gives −0.042\* answer hit rate and is 2.4× slower |
| No chunking within a page | Evidence is page-level; chunking, a different embedding model and structured tables were all insignificant |
| No routing at the retrieval stage | Per-question best single route (oracle) MRR 0.786 ≈ reranking 0.779 |
