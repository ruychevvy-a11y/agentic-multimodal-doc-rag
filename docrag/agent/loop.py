import re
import time

import requests
from dashscope import MultiModalConversation

from docrag import config
from docrag.agent.tools import TOOL_SCHEMAS, tool_result
from docrag.embeddings import call_with_retry
from docrag.generation.answer import message_text, page_contents

# agent 的系统指令
AGENT_PROMPT = (
    "You answer a question about one document of {page_count} pages. "
    "Some pages are shown to you, each preceded by its label [Page N].\n"
    "Answer only from what the pages show. If the shown pages do not clearly contain the answer, "
    "use search_pages to find other pages and view_pages to look at them before answering.\n"
    "Finish by calling answer with the answer and the viewed pages it is based on. "
    'If the document does not contain the information, answer "Not answerable".'
)

# 兜底时的强制作答指令
FALLBACK_PROMPT = (
    "Answer now using only the pages you have seen. "
    'If they do not contain the information, answer "Not answerable". '
    'End with a line "Cited pages: N, M".'
)

# 没搜过就答 Not answerable 时的提醒
SEARCH_REMINDER = (
    "You have not searched the document yet. "
    'Use search_pages at least once before answering "Not answerable".'
)


# 调 agent 模型（**parameters 只在兜底时传 tool_choice="none"）
def call_agent_model(messages, **parameters):
    started = time.perf_counter()
    resp = call_with_retry(
        lambda: MultiModalConversation.call(
            model=config.GENERATION_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            thinking_budget=config.AGENT_THINKING_BUDGET,
            request_timeout=config.AGENT_REQUEST_TIMEOUT,
            api_key=config.API_KEY,
            **parameters,
        )
    )
    usage = {
        "input_tokens": resp.usage["input_tokens"],
        "cached_tokens": resp.usage["prompt_tokens_details"]["cached_tokens"],
        "output_tokens": resp.usage["output_tokens"],
        "seconds": round(time.perf_counter() - started, 2),
    }
    return resp.output.choices[0].message, usage


# 取出引用页，只留存在且看过的页
def cited_page_numbers(text, viewed):
    lines = [line for line in text.splitlines() if "cited page" in line.lower()]
    if not lines:
        return []
    numbers = [int(number) for number in re.findall(r"\d+", lines[-1])]
    return [number for number in dict.fromkeys(numbers) if number in viewed]


# 结束步骤
# fallback 取三种值:
# None：正常结束;   "max_model_calls"：5 次调用都用完了;   "timeout"：调用超时
def final_step(answer, cited_pages, fallback):
    return {
        "type": "final",
        "answer": answer,
        "cited_pages": cited_pages,
        "fallback": fallback,
    }


# 运行 agent 循环，每走一步交回一条事件
def run_agent(question, tools, initial_pages):
    # 将page_count填入prompt
    system_prompt = AGENT_PROMPT.format(page_count=len(tools.pages_by_number))
    question_content = page_contents(initial_pages)
    question_content.append({"text": f"Question: {question}"})
    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {"role": "user", "content": question_content},
    ]
    reminded = False
    fallback = "max_model_calls"

    for call_index in range(config.AGENT_MAX_MODEL_CALLS):
        try:
            message, usage = call_agent_model(messages)
        except requests.exceptions.ReadTimeout:
            fallback = "timeout"
            break
        text = message_text(message)
        tool_calls = message.get("tool_calls") or []
        yield {
            "type": "model_call",
            "call_index": call_index,
            "text": text,
            "tool_calls": tool_calls,
            **usage,
        }
        # 没调工具，正文就是答案
        if not tool_calls:
            yield final_step(text, cited_page_numbers(text, tools.viewed), None)
            return
        messages.append(dict(message))
        # 执行工具
        shown_pages = []
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            result = tools.call_tool(name, arguments)
            # 没搜过就答 Not answerable，提醒一次
            answer = result["answer"]
            if (
                answer
                and "not answerable" in answer["text"].lower()
                and not tools.search_cache
                and not reminded
            ):
                reminded = True
                result = tool_result(SEARCH_REMINDER)
            yield {
                "type": "tool",
                "name": name,
                "arguments": arguments,
                "text": result["text"],
                "error": result["error"],
                "shown_pages": [page["page_number"] for page in result["pages"]],
            }

            # 回答了则结束返回，否则附上新页图
            if result["answer"]:
                yield final_step(answer["text"], answer["cited_pages"], None)
                return
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": name,
                    "content": [{"text": result["text"]}],
                }
            )
            shown_pages += result["pages"]
        if shown_pages:
            messages.append({"role": "user", "content": page_contents(shown_pages)})
    # 兜底，禁止调工具，用看过的页强制作答
    messages.append({"role": "user", "content": [{"text": FALLBACK_PROMPT}]})
    message, usage = call_agent_model(messages, tool_choice="none")
    text = message_text(message)
    yield {
        "type": "model_call",
        "call_index": call_index + 1,
        "text": text,
        "tool_calls": [],
        **usage,
    }
    yield final_step(text, cited_page_numbers(text, tools.viewed), fallback)
