import json
import os

from docrag import config
from docrag.jsonl import load_jsonl, write_jsonl

# 报告 recall@k 的 k
K = (1, 3, 5, 10)


# 读金标 --> {query_id: 金标页集合}
def load_qrels():
    qrels = {}
    for row in load_jsonl(config.QRELS_PATH):
        qrels.setdefault(row["query_id"], set()).add(row["page_id"])
    return qrels


# 一道题的检索指标 --> ({k: recall@k}, 倒数排名)
# recall@k = 金标页落进前 k 名的比例；倒数排名 = 1 / 第一个金标页的名次，没有记 0
# 例：ranked=['a','b','c','d'], gold_pages={'b','d','x'} --> recall@3 = 1/3，倒数排名 = 1/2
def retrieval_metrics(ranked, gold_pages):
    recalls = {k: len(gold_pages & set(ranked[:k])) / len(gold_pages) for k in K}
    gold_ranks = [rank for rank, page_id in enumerate(ranked, 1) if page_id in gold_pages]
    reciprocal_rank = 1.0 / gold_ranks[0] if gold_ranks else 0.0
    return recalls, reciprocal_rank


# 用一条路线跑完全部问题并写结果文件 --> 平均指标
# scope：closed 只在题目所属文档里搜，open 全库检索
def evaluate(retrieval_fn, tag, scope):
    queries = load_jsonl(config.QUERIES_PATH)
    qrels = load_qrels()
    recall_sums = {k: 0.0 for k in K}
    reciprocal_rank_sum = 0.0
    run = []
    for i, query in enumerate(queries, 1):
        search_filter = f'doc_id == "{query["doc_id"]}"' if scope == "closed" else ""
        ranked = retrieval_fn(query["question"], search_filter)
        run.append({"query_id": query["query_id"], "ranked": ranked})
        recalls, reciprocal_rank = retrieval_metrics(ranked, qrels[query["query_id"]])
        for k in K:
            recall_sums[k] += recalls[k]
        reciprocal_rank_sum += reciprocal_rank
        if i % 50 == 0:
            print(f"{i}/{len(queries)} evaluated")

    n = len(queries)
    results = {
        "tag": tag,
        "scope": scope,
        "num_queries": n,
        # 路线名里不写模型，模型记在这里
        "text_model": config.TEXT_EMBEDDING_MODEL,
        "image_model": config.MULTIMODAL_EMBEDDING_MODEL,
        **{f"recall@{k}": round(recall_sums[k] / n, 4) for k in K},
        "mrr": round(reciprocal_rank_sum / n, 4),
    }
    save_results(results, run)
    return results


# 写 results_<路线>.json（平均指标）和 run_<路线>.jsonl（逐题排名），并打印汇总
def save_results(results, run):
    out_dir = f"{config.EVALUATION_DIR}/{results['scope']}"
    os.makedirs(out_dir, exist_ok=True)
    results_path = f"{out_dir}/results_{results['tag']}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    run_path = f"{out_dir}/run_{results['tag']}.jsonl"
    write_jsonl(run_path, run)

    print(f"\n=== [{results['tag']}] {results['scope']} n={results['num_queries']} ===")
    print("  ".join(f"recall@{k}={results[f'recall@{k}']}" for k in K) + f"  MRR={results['mrr']}")
    print("written:", results_path, run_path)
