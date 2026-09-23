import json
import math
import os
from functools import lru_cache

from docrag import config
from docrag.agent.agent_answer import generate_agent_answer
from docrag.generation.answer import generate_answer

# 特征顺序，训练和推理必须一致
FEATURE_NAMES = [
    "searches",
    "model_calls",
    "viewed_pages",
    "cited_pages",
    "fallback",
    "agent_answer_length",
    "baseline_answer_length",
    "agent_refused",
    "baseline_refused",
    "agent_output_tokens",
]


# 回答里说了文档中没有
def is_refusal(response):
    return "not answerable" in str(response).lower()


# agent 结果 + 基线回答 --> 路由特征；全部来自作答过程，不依赖标准答案
def routing_features(agent_result, baseline_response):
    return {
        "searches": agent_result["searches"],
        "model_calls": agent_result["model_calls"],
        "viewed_pages": agent_result["viewed_pages"],
        "cited_pages": len(agent_result["cited_pages"]),
        "fallback": float(bool(agent_result["fallback"])),
        "agent_answer_length": len(str(agent_result["response"])),
        "baseline_answer_length": len(str(baseline_response)),
        "agent_refused": float(is_refusal(agent_result["response"])),
        "baseline_refused": float(is_refusal(baseline_response)),
        "agent_output_tokens": agent_result["output_tokens"],
    }


# 读 train_router.py 训好的权重，没有或不是当前数据集训的就返回 None（冷启动走手工规则）
@lru_cache(maxsize=1)
def load_router():
    if not os.path.exists(config.ROUTER_WEIGHTS_PATH):
        return None
    with open(config.ROUTER_WEIGHTS_PATH, encoding="utf-8") as handle:
        router = json.load(handle)
    return router if router["trained_on"] == config.DATASET else None


# 这道题采信哪条路 --> (用不用 agent, 概率；没有权重时按「检索过才信」判，概率是 None)
def route(features, router):
    if router is None:
        return features["searches"] > 0, None
    # 标准化已折进权重，这里直接线性求和
    total = router["intercept"] + sum(
        weight * features[name] for name, weight in router["weights"].items()
    )
    probability = 1 / (1 + math.exp(-total))
    return probability >= config.ROUTER_THRESHOLD, probability


# 两条路都跑，按路由择一 --> 与 generate_agent_answer 同形状的记录，多带 route 字段
def route_answer(answer, pages, retrieval, pages_by_doc):
    agent_result = generate_agent_answer(answer, pages, retrieval, pages_by_doc)
    baseline = generate_answer(answer["question"], pages)
    features = routing_features(agent_result, baseline["response"])
    use_agent, probability = route(features, load_router())
    return {
        **agent_result,
        "response": agent_result["response"] if use_agent else baseline["response"],
        "cited_pages": agent_result["cited_pages"] if use_agent else [],
        "input_tokens": agent_result["input_tokens"] + baseline["input_tokens"],
        "output_tokens": agent_result["output_tokens"] + baseline["output_tokens"],
        "route": "agent" if use_agent else "baseline",
        "agent_probability": None if probability is None else round(probability, 4),
    }
