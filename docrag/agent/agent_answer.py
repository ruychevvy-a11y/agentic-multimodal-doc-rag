from collections import Counter

from docrag import config
from docrag.agent.loop import SEARCH_REMINDER, run_agent
from docrag.agent.tools import DocumentTools


# agent 作答
def generate_agent_answer(answer, pages, retrieval, pages_by_doc):
    tools = DocumentTools(
        retrieval, answer["doc_id"], pages_by_doc[answer["doc_id"]], pages
    )
    steps = list(run_agent(answer["question"], tools, pages))
    model_steps = [step for step in steps if step["type"] == "model_call"]
    tool_steps = [step for step in steps if step["type"] == "tool"]
    final = steps[-1]
    searches = sum(step["name"] == "search_pages" for step in tool_steps)
    return {
        "response": final["answer"],
        "input_tokens": sum(step["input_tokens"] for step in model_steps),
        "output_tokens": sum(step["output_tokens"] for step in model_steps),
        "cached_tokens": sum(step["cached_tokens"] for step in model_steps),
        "cited_pages": final["cited_pages"],
        "fallback": final["fallback"],
        "model_calls": len(model_steps),
        "tool_calls": len(tool_steps),
        "tool_errors": sum(step["error"] for step in tool_steps),
        "searches": searches,
        # 规范化后重复的查询读缓存，不进 search_cache
        "repeated_searches": searches - len(tools.search_cache),
        "reminders": sum(step["text"] == SEARCH_REMINDER for step in tool_steps),
        "viewed_pages": len(tools.viewed),
        "steps": steps,
    }


# 汇总 agent 的过程指标（调用次数、看页数、兜底率等）
def agent_summary(rows):
    # 作答报错的题没有过程统计
    finished = [row for row in rows if "model_calls" in row]
    if not finished:
        return {}

    def mean(field):
        return round(sum(row[field] for row in finished) / len(finished), 2)

    searched = sum(row["searches"] > 0 for row in finished)
    tool_calls = sum(row["tool_calls"] for row in finished)
    tool_errors = sum(row["tool_errors"] for row in finished)
    # 兜底原因 --> 题数；none 是正常结束
    fallbacks = Counter(row["fallback"] or "none" for row in finished)
    return {
        "agent": {
            "thinking_budget": config.AGENT_THINKING_BUDGET,
            "max_model_calls": config.AGENT_MAX_MODEL_CALLS,
            "max_viewed_pages": config.AGENT_MAX_VIEWED_PAGES,
            "finished": len(finished),
            "model_calls_mean": mean("model_calls"),
            "searched_rate": round(searched / len(finished), 4),
            "searches_mean": mean("searches"),
            "repeated_searches_mean": mean("repeated_searches"),
            "viewed_pages_mean": mean("viewed_pages"),
            "tool_error_rate": round(tool_errors / max(tool_calls, 1), 4),
            "reminders": sum(row["reminders"] for row in finished),
            "fallback_counts": dict(fallbacks),
            "cached_tokens_mean": round(mean("cached_tokens")),
        }
    }
