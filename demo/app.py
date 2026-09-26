import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from itertools import groupby
from operator import itemgetter

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from docrag import config
from docrag.agent.agent_answer import agent_record
from docrag.agent.loop import cited_page_numbers, run_agent
from docrag.agent.routing import load_router, route, routing_features
from docrag.agent.tools import DocumentTools
from docrag.generation.answer import generate_answer
from docrag.generation.rewrite import rewrite_question
from docrag.jsonl import load_jsonl
from docrag.retrieval.rerank import rerank_convex_bm25_text_image_retrieval

# 改写追问时最多带几轮历史
HISTORY_ROUNDS = 3

# 一次作答和 agent 并行跑；agent 答完后最多再等一次作答这么多秒，没等到就直接用 agent 的答案
ROUTE_WAIT_SECONDS = 10

# 后台跑一次作答的线程数
BASELINE_WORKERS = 4

# 启动时加载一次、所有请求共用的资源
resources = {}


# 启动加载
@asynccontextmanager
async def lifespan(app):
    answers = load_jsonl(config.ANSWERS_PATH)
    pages = load_jsonl(config.PAGES_PATH)
    pages_by_id = {page["page_id"]: page for page in pages}
    for answer in answers:
        # 金标页配上页号和页图路径 ['..._p19'] --> [19] / ['data/.../..._p19.jpg']
        gold_pages = [pages_by_id[page_id] for page_id in answer["gold_page_ids"]]
        answer["gold_pages"] = [page["page_number"] for page in gold_pages]
        answer["gold_images"] = [page["image_path"] for page in gold_pages]
    # 题库 {query_id: 题}
    resources["answers"] = {answer["query_id"]: answer for answer in answers}
    resources["pages_by_doc"] = {
        doc_id: list(doc_pages)
        for doc_id, doc_pages in groupby(pages, key=itemgetter("doc_id"))
    }
    resources["retrieval"] = rerank_convex_bm25_text_image_retrieval([])
    resources["baseline_pool"] = ThreadPoolExecutor(max_workers=BASELINE_WORKERS)
    yield
    resources["baseline_pool"].shutdown(wait=False, cancel_futures=True)
    resources.clear()


app = FastAPI(title="混合多模态文档问答", lifespan=lifespan)


# 文档列表
@app.get("/documents")
def list_documents():
    answers = resources["answers"].values()
    question_counts = Counter(answer["doc_id"] for answer in answers)
    doc_types = {answer["doc_id"]: answer["doc_type"] for answer in answers}
    return [
        {
            "doc_id": doc_id,
            "doc_type": doc_types[doc_id],
            "pages": len(doc_pages),
            "questions": question_counts[doc_id],
        }
        for doc_id, doc_pages in resources["pages_by_doc"].items()
    ]


# 某份文档的题目
@app.get("/questions")
def list_questions(doc_id: str):
    questions = [
        answer for answer in resources["answers"].values() if answer["doc_id"] == doc_id
    ]
    if not questions:
        raise HTTPException(status_code=404, detail=f"没有这份文档: {doc_id}")
    return questions


# 随机抽一道题
@app.get("/questions/random")
def random_question():
    return random.choice(list(resources["answers"].values()))


# 一轮问答
class Turn(BaseModel):
    question: str
    answer: str


# 提问的请求体；history 是同一份文档上的历史问答，用来把追问补成完整问题
class AskRequest(BaseModel):
    question: str
    doc_id: str
    history: list[Turn] = []


# 一条 SSE 消息 {"type": "tool", ...} --> 'data: {"type": "tool", ...}\n\n'
def sse_event(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# 给页号配上页图路径，agent 步骤里只有页号没有路径
def with_images(step, pages_by_number):
    if step["type"] == "tool":
        step["shown_images"] = [
            pages_by_number[number]["image_path"] for number in step["shown_pages"]
        ]
    elif step["type"] == "final":
        step["cited_images"] = [
            pages_by_number[number]["image_path"] for number in step["cited_pages"]
        ]
    return step


# 等一次作答：agent 答完后最多再等 ROUTE_WAIT_SECONDS 秒 --> (一次作答结果, 没等到的原因代码, 报错内容)
def wait_baseline(baseline_future):
    try:
        return baseline_future.result(timeout=ROUTE_WAIT_SECONDS), None, None
    except TimeoutError:
        return None, "baseline_timeout", None
    except (RuntimeError, requests.exceptions.ReadTimeout) as error:
        return None, "baseline_error", repr(error)


# 去掉回答里的 Cited pages 行，引用页另外显示
def without_citation_line(text):
    lines = [line for line in text.splitlines() if "cited page" not in line.lower()]
    return "\n".join(lines).strip()


# 两条路的结果 --> route 事件：采信哪条、依据、最终答案和引用页，附两条路各自的回答；一次作答缺席时直接用 agent
# 依据只发代码（classifier / rule / baseline_timeout / baseline_error），界面按语言翻译
def route_event(agent_result, baseline, missing_reason, baseline_error, initial_pages, pages_by_number):
    if baseline is None:
        use_agent, probability, reason = True, None, missing_reason
    else:
        router = load_router()
        use_agent, probability = route(
            routing_features(agent_result, baseline["response"]), router
        )
        reason = "classifier" if router else "rule"
    if use_agent:
        answer = agent_result["response"]
        cited_pages = agent_result["cited_pages"]
    else:
        answer = baseline["response"]
        # 一次作答在最后一行写 Cited pages，只认给它看过的起点页
        cited_pages = cited_page_numbers(
            answer, {page["page_number"] for page in initial_pages}
        )
    return {
        "type": "route",
        "route": "agent" if use_agent else "baseline",
        "agent_probability": None if probability is None else round(probability, 3),
        "reason": reason,
        "wait_seconds": ROUTE_WAIT_SECONDS,
        "answer": without_citation_line(answer),
        "cited_pages": cited_pages,
        "cited_images": [pages_by_number[number]["image_path"] for number in cited_pages],
        "fallback": agent_result["fallback"] if use_agent else None,
        "agent_answer": agent_result["response"],
        "baseline_answer": None if baseline is None else baseline["response"],
        "baseline_error": baseline_error,
    }


# 提问：检索起点页 --> 一次作答放后台线程，agent 每走一步推一条 SSE --> 两条路择一，最后推一条 route 事件
@app.post("/ask")
def ask(request: AskRequest):
    if request.doc_id not in resources["pages_by_doc"]:
        raise HTTPException(status_code=404, detail=f"没有这份文档: {request.doc_id}")
    doc_pages = resources["pages_by_doc"][request.doc_id]

    def event_stream():
        # 流已经开始推了就改不了 HTTP 状态码，出错只能推一条 error 事件
        try:
            question = request.question
            # 有历史就先把追问补成完整问题，之后检索和作答都用它
            if request.history:
                history = [
                    turn.model_dump() for turn in request.history[-HISTORY_ROUNDS:]
                ]
                question = rewrite_question(request.question, history)
                yield sse_event({"type": "rewrite", "question": question})
            search_filter = f'doc_id == "{request.doc_id}"'
            page_ids = resources["retrieval"].retrieve(question, search_filter)
            pages_by_id = {page["page_id"]: page for page in doc_pages}
            initial_pages = [
                pages_by_id[page_id] for page_id in page_ids[: config.ANSWER_PAGES]
            ]
            yield sse_event(
                {
                    "type": "retrieval",
                    "pages": [page["page_number"] for page in initial_pages],
                    "images": [page["image_path"] for page in initial_pages],
                }
            )
            # 一次作答不依赖 agent，丢到后台线程和 agent 并行
            baseline_future = resources["baseline_pool"].submit(
                generate_answer, question, initial_pages
            )
            tools = DocumentTools(
                resources["retrieval"], request.doc_id, doc_pages, initial_pages
            )
            steps = []
            for step in run_agent(question, tools, initial_pages):
                steps.append(step)
                yield sse_event(with_images(step, tools.pages_by_number))
            baseline, missing_reason, baseline_error = wait_baseline(baseline_future)
            yield sse_event(
                route_event(
                    agent_record(steps, tools),
                    baseline,
                    missing_reason,
                    baseline_error,
                    initial_pages,
                    tools.pages_by_number,
                )
            )
        except (RuntimeError, requests.exceptions.ReadTimeout) as error:
            yield sse_event({"type": "error", "message": repr(error)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
