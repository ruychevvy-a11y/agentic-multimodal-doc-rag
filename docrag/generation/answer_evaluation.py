import contextlib
import io
import json
import os
import time
import requests

from concurrent.futures import ThreadPoolExecutor
from functools import partial


from dashscope import MultiModalConversation
from docrag import config
from docrag.embeddings import call_with_retry
from docrag.generation.mmlongbench_official.eval_score import eval_score
from docrag.jsonl import load_jsonl, write_jsonl

# 抽答案模型：把作答模型的自由回答抽成短答案（官方用 GPT-4o）；关思考、温度 0
EXTRACTION_MODEL = "qwen3.7-plus"

# 作答 + 抽答案的并发线程数
ANSWER_WORKERS = 12

# 官方抽答案指令（few-shot 模板），和官方判分代码放在同一个目录
EXTRACTION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "mmlongbench_official", "prompt_for_answer_extraction.md"
)
with open(EXTRACTION_PROMPT_PATH, encoding="utf-8") as f:
    EXTRACTION_PROMPT = f.read()


# 抽答案：数据集官方消息结构（user 放抽取指令，assistant 放问题 + 模型的回答）
def extract_answer(question, response):
    messages = [
        {"role": "user", "content": [{"text": EXTRACTION_PROMPT}]},
        {
            "role": "assistant",
            "content": [{"text": f"\n\nQuestion:{question}\nAnalysis:{response}\n"}],
        },
    ]
    # 提取答案关闭思考
    resp = call_with_retry(
        lambda: MultiModalConversation.call(
            model=EXTRACTION_MODEL,
            messages=messages,
            enable_thinking=False,
            # 回答的随机性
            temperature=0.0,
            max_tokens=256,
            api_key=config.API_KEY,
        )
    )

    message = resp.output.choices[0].message
    return {
        "extracted": "".join(
            part["text"] for part in message.content if "text" in part
        ),
        "input_tokens": resp.usage["input_tokens"],
        "output_tokens": resp.usage["output_tokens"],
    }


# 抽取结果 --> 预测答案：取「Extracted answer:」和「Answer format:」之间的文字；格式不对记 "Failed to extract"
def predicted_answer(extracted):
    try:
        return (
            extracted.split("Answer format:")[0].split("Extracted answer:")[1].strip()
        )
    except IndexError:
        return "Failed to extract"


# 一题的官方得分（0~1）：answers[0] 是标准答案；官方 List 分支会 eval 预测字符串，格式坏了会抛异常 --> 记 0 分
# 官方函数里有调试 print，用 redirect_stdout 吞掉；它会临时替换全局 stdout，不要在多线程里调用
def answer_score(answers, prediction, answer_format):
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return eval_score(answers[0], prediction, answer_format)
        except Exception:
            return 0.0


# Acc / F1 --> (accuracy, f1)；rows 每项要有 answers / pred / score
def accuracy_and_f1(rows):
    accuracy = sum(row["score"] for row in rows) / len(rows)
    # 看标准答案：能回答的题
    answerable = [row for row in rows if row["answers"][0] != "Not answerable"]
    # 看模型预测：没有拒答的题
    answered = [row for row in rows if row["pred"] != "Not answerable"]
    # 可回答题的得分之和（正确拒答的分不算在内）
    answerable_score = sum(row["score"] for row in answerable)

    recall = answerable_score / len(answerable) if answerable else 0.0
    precision = answerable_score / len(answered) if answered else 0.0
    f1 = (
        2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    )
    return accuracy, f1


# 一题作答 + 抽答案 --> 结果记录；调用失败（如内容审核、超时）记 error，回答 / 抽取记 "Failed"，判分时为 0 分
# generate_fn(answer, pages) --> {response, input_tokens, output_tokens, ...}，多出的字段原样记进结果
def answer_question(answer, pages, generate_fn):
    record = {
        "query_id": answer["query_id"],
        "page_ids": [page["page_id"] for page in pages],
    }
    # 计时
    started = time.perf_counter()
    # 模型作答，计时只算这一步的
    try:
        generation = generate_fn(answer, pages)
    except (RuntimeError, requests.exceptions.ReadTimeout) as error:
        return {
            **record,
            "response": "Failed",
            "extracted": "Failed",
            "error": repr(error),
        }
    record["response"] = generation["response"]
    record["generation_input_tokens"] = generation["input_tokens"]
    record["generation_output_tokens"] = generation["output_tokens"]
    record["generation_seconds"] = round(time.perf_counter() - started, 2)
    # 作答方式自己的字段（如 agent 的步骤记录和过程统计）
    record.update(
        {
            key: value
            for key, value in generation.items()
            if key not in ("response", "input_tokens", "output_tokens")
        }
    )

    # 抽答案
    try:
        extraction = extract_answer(answer["question"], generation["response"])
    except RuntimeError as error:
        return {**record, "extracted": "Failed", "error": repr(error)}
    record["extracted"] = extraction["extracted"]
    record["extraction_input_tokens"] = extraction["input_tokens"]
    record["extraction_output_tokens"] = extraction["output_tokens"]
    return record


# 一批题的汇总：Acc / F1 + 按官方分组（单页 / 跨页 / 不可回答）和证据类型的准确率 + 每题平均 token + 作答延迟 p50 / p90
# answers 和 rows 按位置一一对应
def answer_summary(answers, rows):
    # 一次遍历把每题放进它所属的组；一题可以同时属于多个组
    groups = {}
    for answer, row in zip(answers, rows):
        names = list(answer["evidence_sources"])
        if len(answer["evidence_pages"]) == 1:
            names.append("single_page")
        elif answer["answers"][0] != "Not answerable":
            names.append("cross_page")
        if answer["answers"][0] == "Not answerable":
            names.append("unanswerable")
        for name in names:
            groups.setdefault(name, []).append(row)

    accuracy, f1 = accuracy_and_f1(rows)
    # 作答成功的题才有耗时；分位数取排序后对应位置的值
    seconds = sorted(
        row["generation_seconds"] for row in rows if "generation_seconds" in row
    )
    token_fields = (
        "generation_input_tokens",
        "generation_output_tokens",
        "extraction_input_tokens",
        "extraction_output_tokens",
    )
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "group_accuracy": {
            name: {
                "n": len(group_rows),
                "accuracy": round(accuracy_and_f1(group_rows)[0], 4),
            }
            for name, group_rows in groups.items()
        },
        "errors": sum("error" in row for row in rows),
        # 每题平均 token：失败的题按 0 算，乘单价就是每题平均成本
        **{
            f"{field}_mean": round(sum(row.get(field, 0) for row in rows) / len(rows))
            for field in token_fields
        },
        "generation_seconds_p50": (
            seconds[round(0.5 * (len(seconds) - 1))] if seconds else None
        ),
        "generation_seconds_p90": (
            seconds[round(0.9 * (len(seconds) - 1))] if seconds else None
        ),
    }


# 跑答案评测：检索 --> 并发作答 + 抽答案（逐题追加进 run 文件，已有的题跳过）--> 判分 --> 写结果
# generate_fn 是作答方式（基线一次作答 / agent）；answers 是这次要评测的题；scope：closed 只在题目所属文档里搜，open 全库检索
# summary_fn(rows) 是作答方式自己的汇总（如 agent 过程指标），没有传 None
def evaluate_answers(retrieval_fn, generate_fn, tag, scope, answers, summary_fn=None):
    all_answers = load_jsonl(config.ANSWERS_PATH)
    pages = {page["page_id"]: page for page in load_jsonl(config.PAGES_PATH)}
    out_dir = f"{config.EVALUATION_DIR}/{scope}"
    os.makedirs(out_dir, exist_ok=True)
    run_path = f"{out_dir}/run_{tag}.jsonl"
    results_path = f"{out_dir}/results_{tag}.json"

    # 断点续跑：run 文件里已有的题不再调 API
    done = (
        {record["query_id"]: record for record in load_jsonl(run_path)}
        if os.path.exists(run_path)
        else {}
    )
    pending = [answer for answer in answers if answer["query_id"] not in done]
    print(
        f"{len(answers)} 题：已完成 {len(answers) - len(pending)}，待跑 {len(pending)}"
    )

    # 检索在主线程逐题做，只把调大模型的两步并发；每题取前 ANSWER_PAGES 页的页面记录
    page_lists = []
    for i, answer in enumerate(pending, 1):
        search_filter = f'doc_id == "{answer["doc_id"]}"' if scope == "closed" else ""
        page_ids = retrieval_fn(answer["question"], search_filter)[
            : config.ANSWER_PAGES
        ]
        page_lists.append([pages[page_id] for page_id in page_ids])
        if i % 50 == 0:
            print(f"{i}/{len(pending)} retrieved")

    # 作答 + 抽答案并发；结果回到主线程后一题一行追加进 run 文件并立即落盘
    answer_one = partial(answer_question, generate_fn=generate_fn)
    with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as pool, open(
        run_path, "a", encoding="utf-8"
    ) as f:
        for i, record in enumerate(pool.map(answer_one, pending, page_lists), 1):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[record["query_id"]] = record
            if "error" in record:
                print("失败", record["query_id"], record["error"][:120])
            if i % 50 == 0:
                print(f"{i}/{len(pending)} answered")

    # 判分在主线程做；run 文件里所有已完成的题都补上预测和得分，按题目顺序整份重写（limit 只影响汇总范围，不丢记录）
    scored = {}
    for answer in all_answers:
        if answer["query_id"] not in done:
            continue
        record = done[answer["query_id"]]
        prediction = predicted_answer(record["extracted"])
        scored[answer["query_id"]] = {
            **record,
            "answers": answer["answers"],
            "pred": prediction,
            "score": answer_score(
                answer["answers"], prediction, answer["answer_format"]
            ),
        }
    write_jsonl(run_path, scored.values())
    rows = [scored[answer["query_id"]] for answer in answers]
    results = {
        "tag": tag,
        "scope": scope,
        "num_questions": len(answers),
        # 路线名里不写模型，模型记在这里
        "generation_model": config.GENERATION_MODEL,
        "extraction_model": EXTRACTION_MODEL,
        "answer_pages": config.ANSWER_PAGES,
        **answer_summary(answers, rows),
    }
    # summary_fn(rows) 是作答方式自己的汇总（如 agent 过程指标），没有传 None
    if summary_fn:
        results.update(summary_fn(rows))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(
        f"\n=== [{tag}] {scope} n={len(answers)} ===  Acc={results['accuracy']}  F1={results['f1']}  errors={results['errors']}"
    )
    print("written:", results_path, run_path)
    return results
