import argparse
import json
import os

from docrag import config
from docrag.evaluation import evaluate
from docrag.jsonl import load_jsonl
from docrag.retrieval.dense import image_dense_retrieval, text_dense_retrieval
from docrag.retrieval.fusion import (
    convex_bm25_text_image_retrieval,
    convex_bm25_text_retrieval,
    rrf_bm25_text_retrieval,
)
from docrag.retrieval.rerank import (
    rerank_convex_bm25_text_image_retrieval,
    rerank_convex_bm25_text_retrieval,
)
from docrag.retrieval.sparse import text_bm25_retrieval

# 路线名 --> 造检索器的函数；每个数据集评测哪些路线见 config 登记表
ROUTES = {
    "image_dense": image_dense_retrieval,
    "text_dense": text_dense_retrieval,
    "text_bm25": text_bm25_retrieval,
    "rrf_bm25_text": rrf_bm25_text_retrieval,
    "convex_bm25_text": convex_bm25_text_retrieval,
    "convex_bm25_text_image": convex_bm25_text_image_retrieval,
    "rerank_convex_bm25_text": rerank_convex_bm25_text_retrieval,
    "rerank_convex_bm25_text_image": rerank_convex_bm25_text_image_retrieval,
}

# 菜单编号 --> (评测范围, 说明)
SCOPES = {
    "1": ("closed", "闭域：只在题目所属文档里搜"),
    "2": ("open", "开域：全库检索"),
}


# 一条路线在某个范围下上次的成绩（菜单里显示）
def previous_result(route, scope):
    path = f"{config.EVALUATION_DIR}/{scope}/results_{route}.json"
    if not os.path.exists(path):
        return "未测评"
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    return (
        f"recall@1={result['recall@1']:.3f}  "
        f"@3={result['recall@3']:.3f}  "
        f"@5={result['recall@5']:.3f}  "
        f"@10={result['recall@10']:.3f}  "
        f"MRR={result['mrr']:.3f}"
    )


# 菜单选评测范围 --> "closed" / "open"
def choose_scope():
    print("选择评测范围：")
    for number, (scope, description) in SCOPES.items():
        print(f"  {number}) {description}")
    choice = input("[1/2]: ").strip()
    if choice not in SCOPES:
        raise SystemExit(f"无效选择: {choice!r}")
    return SCOPES[choice][0]


# 菜单选路线（附这个范围下的历史成绩）--> 路线名
def choose_route(scope):
    routes = sorted(config.EVALUATION_ROUTES)
    print(f"\n选择要评测的路线({scope}): ")
    width = max(len(route) for route in routes)
    for i, route in enumerate(routes, 1):
        print(f"  {i}) {route:{width}}   {previous_result(route, scope)}")
    choice = input(f"[1-{len(routes)}]: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(routes):
        raise SystemExit(f"无效选择: {choice!r}")
    return routes[int(choice) - 1]


# 入口：选范围和路线（或命令行传入），造检索器，逐题评测
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=[scope for scope, description in SCOPES.values()])
    parser.add_argument("--route", choices=sorted(config.EVALUATION_ROUTES))
    args = parser.parse_args()
    scope = args.scope or choose_scope()
    route = args.route or choose_route(scope)

    # 造检索器时 dense 路会一次性嵌入全部问题
    questions = [query["question"] for query in load_jsonl(config.QUERIES_PATH)]
    retriever = ROUTES[route](questions)
    evaluate(retriever.retrieve, route, scope)


if __name__ == "__main__":
    main()
