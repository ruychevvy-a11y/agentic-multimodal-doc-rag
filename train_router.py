import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from docrag import config
from docrag.agent.routing import FEATURE_NAMES, routing_features
from docrag.jsonl import load_jsonl


# run 文件 --> {query_id: 记录}
def load_run(path):
    return {record["query_id"]: record for record in load_jsonl(path)}


# 两条路的 run 文件 --> 特征矩阵、两边得分
def build_dataset(agent_run, baseline_run):
    rows = []
    for query_id in sorted(set(agent_run) & set(baseline_run)):
        agent_record = agent_run[query_id]
        # 作答报错的题没有过程字段
        if "searches" not in agent_record:
            continue
        agent_record = {
            **agent_record,
            "output_tokens": agent_record["generation_output_tokens"],
        }
        features = routing_features(agent_record, baseline_run[query_id]["response"])
        rows.append(
            (
                [features[name] for name in FEATURE_NAMES],
                agent_record["score"],
                baseline_run[query_id]["score"],
            )
        )
    return (
        np.array([row[0] for row in rows], dtype=float),
        np.array([row[1] for row in rows]),
        np.array([row[2] for row in rows]),
    )


def new_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            C=config.ROUTER_REGULARISATION,
            class_weight="balanced",
        ),
    )


# 交叉验证：每道题都由没见过它的那一折来判，得到不掺水的路由得分
def cross_validate(features_matrix, agent_scores, baseline_scores, differs):
    folds = StratifiedKFold(n_splits=config.ROUTER_FOLDS, shuffle=True, random_state=0)
    out_of_fold = np.zeros(len(features_matrix))
    for train_index, test_index in folds.split(
        features_matrix, np.sign(agent_scores - baseline_scores)
    ):
        informative = train_index[differs[train_index]]
        model = new_model()
        model.fit(
            features_matrix[informative],
            (agent_scores[informative] > baseline_scores[informative]).astype(int),
        )
        probabilities = model.predict_proba(features_matrix[test_index])[:, 1]
        out_of_fold[test_index] = np.where(
            probabilities >= config.ROUTER_THRESHOLD,
            agent_scores[test_index],
            baseline_scores[test_index],
        )
    return out_of_fold


# 把标准化折进权重：sigmoid(Σ cᵢ(xᵢ-mᵢ)/sᵢ + b₀) = sigmoid(Σ (cᵢ/sᵢ)xᵢ + b)，线上就不用带 sklearn
def fold_scaler(model):
    scaler, classifier = model[0], model[-1]
    weights = classifier.coef_[0] / scaler.scale_
    intercept = float(
        classifier.intercept_[0]
        - np.sum(classifier.coef_[0] * scaler.mean_ / scaler.scale_)
    )
    return dict(zip(FEATURE_NAMES, weights.round(6).tolist())), round(intercept, 6)


# 入口：读两条路的 run 文件 --> 只用分出胜负的题训练 --> 交叉验证 --> 写权重文件
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["closed", "open"], default="closed")
    parser.add_argument("--agent-route", default="agent_rerank_convex_bm25_text_image")
    parser.add_argument(
        "--baseline-route", default="answer_rerank_convex_bm25_text_image"
    )
    args = parser.parse_args()

    run_dir = f"{config.EVALUATION_DIR}/{args.scope}"
    features_matrix, agent_scores, baseline_scores = build_dataset(
        load_run(f"{run_dir}/run_{args.agent_route}.jsonl"),
        load_run(f"{run_dir}/run_{args.baseline_route}.jsonl"),
    )

    # 两条路得分一样的题对「该选谁」没有信息量，放进训练只会稀释信号
    differs = agent_scores != baseline_scores
    if differs.sum() < 20:
        raise SystemExit(f"分出胜负的题只有 {differs.sum()} 道，至少要 20 道")

    out_of_fold = cross_validate(
        features_matrix, agent_scores, baseline_scores, differs
    )
    model = new_model()
    model.fit(
        features_matrix[differs],
        (agent_scores[differs] > baseline_scores[differs]).astype(int),
    )
    weights, intercept = fold_scaler(model)
    router = {
        "trained_on": config.DATASET,
        "scope": args.scope,
        "questions": int(len(features_matrix)),
        "training_samples": int(differs.sum()),
        "out_of_fold_accuracy": round(float(out_of_fold.mean()), 4),
        "weights": weights,
        "intercept": intercept,
    }
    with open(config.ROUTER_WEIGHTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(router, handle, ensure_ascii=False, indent=2)
    print(
        f"一次作答 {baseline_scores.mean():.4f} | agent {agent_scores.mean():.4f} | "
        f"路由 {out_of_fold.mean():.4f}（{differs.sum()} 道题训练）"
    )
    print("written:", config.ROUTER_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
