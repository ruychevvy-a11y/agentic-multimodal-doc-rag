import argparse
import json
from functools import partial
from itertools import groupby
from operator import itemgetter

from docrag import config
from docrag.agent.agent_answer import agent_summary, generate_agent_answer
from docrag.generation.answer import generate_answer
from docrag.generation.answer_evaluation import evaluate_answers
from docrag.jsonl import load_jsonl
from docrag.retrieval.rerank import rerank_convex_bm25_text_image_retrieval

# 答案路线名, answer为单次baseline，agent为agent loop
ANSWER_ROUTES = {
    "answer_rerank_convex_bm25_text_image": rerank_convex_bm25_text_image_retrieval,
    "agent_rerank_convex_bm25_text_image": rerank_convex_bm25_text_image_retrieval,
}

# 选模型 / 思考设置实验用的 200 题，agent 的对照组也在这批题上
SELECTION_SAMPLE_PATH = (
    f"{config.EVALUATION_DIR}/answer_experiments/model_selection/samples_200.json"
)


# 基线作答：看前几页页图一次作答
def baseline_generate(answer, pages):
    return generate_answer(answer["question"], pages)


# 入口：解析参数 --> 检查数据集 --> 挑题 --> 造检索器（dense 路一次性嵌入这批题的问题）--> 选作答方式 --> 跑答案评测
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["closed", "open"], required=True)
    parser.add_argument("--route", choices=sorted(ANSWER_ROUTES), required=True)
    parser.add_argument(
        "--sample",
        choices=["selection200"],
        default=None,
        help="只评测选模型用的 200 题",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="只评测前 limit 题，不传表示全部"
    )
    args = parser.parse_args()

    # 答案评测用 MMLongBench 官方抽答案和判分，answers.jsonl 也只有它是答案评测格式
    if config.DATASET != "mmlongbench":
        raise SystemExit(
            f"答案评测目前只支持 mmlongbench，当前 DATASET = {config.DATASET!r}"
        )
    # agent 的工具只在题目所属文档里搜和看
    is_agent = args.route.startswith("agent_")
    if is_agent and args.scope != "closed":
        raise SystemExit("agent 路线只支持 --scope closed")

    answers = load_jsonl(config.ANSWERS_PATH)
    if args.sample:
        with open(SELECTION_SAMPLE_PATH, encoding="utf-8") as f:
            sample_ids = {row["query_id"] for row in json.load(f)}
        answers = [answer for answer in answers if answer["query_id"] in sample_ids]
    answers = answers[: args.limit]
    retriever = ANSWER_ROUTES[args.route]([answer["question"] for answer in answers])

    if is_agent:
        # 同一份文档的页在 pages.jsonl 里连着
        pages = load_jsonl(config.PAGES_PATH)
        pages_by_doc = {
            doc_id: list(doc_pages)
            for doc_id, doc_pages in groupby(pages, key=itemgetter("doc_id"))
        }
        generate_fn = partial(
            generate_agent_answer, retrieval=retriever, pages_by_doc=pages_by_doc
        )
        summary_fn = agent_summary
    else:
        generate_fn = baseline_generate
        summary_fn = None
    evaluate_answers(
        retriever.retrieve, generate_fn, args.route, args.scope, answers, summary_fn
    )


if __name__ == "__main__":
    main()
