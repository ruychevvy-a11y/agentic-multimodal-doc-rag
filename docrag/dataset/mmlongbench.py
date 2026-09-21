import ast
import glob
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pymupdf

from docrag import config
from docrag.jsonl import write_jsonl

# 渲染进程数：按文档并行，每个进程各开各的 PDF
RENDER_WORKERS = 8


# 一份 PDF 逐页渲染成页图 --> 这份 PDF 的 pages 行
def render_document(pdf_path):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    rows = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            page_id = f"{doc_id}_p{page_number}"
            image_path = f"{config.IMAGE_DIR}/{page_id}.jpg"
            # 已渲染的页跳过
            if not os.path.exists(image_path):
                # PDF 是矢量的：长边缩放到 MAX_SIDE 渲染，小页会被放大
                scale = config.MAX_SIDE / max(page.rect.width, page.rect.height)
                page.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).save(
                    image_path, jpg_quality=90
                )
            rows.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "image_path": image_path,
                    "page_number": page_number,
                }
            )
    return rows


# 本地 PDF + 标注 --> 页图 + pages / queries / qrels / answers 四个 jsonl
def build():
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    pdf_paths = sorted(glob.glob(f"{config.PDF_DIR}/*.pdf"))

    # pool.map 按 pdf_paths 的顺序交回，同一份 PDF 的页在 pages 里连着
    pages = []
    with ProcessPoolExecutor(max_workers=RENDER_WORKERS) as pool:
        for i, rows in enumerate(pool.map(render_document, pdf_paths), 1):
            pages.extend(rows)
            print(f"{i}/{len(pdf_paths)} rendered {rows[0]['doc_id']} ({len(rows)} 页)")
    page_counts = Counter(row["doc_id"] for row in pages)

    with open(config.ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    queries = []  # {query_id, question, doc_id, evidence_sources}，只收检索评测用的题
    qrels = []  # {query_id, page_id, relevance}，一个金标页一行
    answers = (
        []
    )  # {query_id, question, doc_id, doc_type, answers, answer_format, evidence_pages, evidence_sources, gold_page_ids}，全部题

    for index, row in enumerate(annotations):
        if config.TARGET_QUERIES is not None and len(queries) >= config.TARGET_QUERIES:
            break

        # 标注没有题号，用行号
        query_id = str(index)
        doc_id = os.path.splitext(row["doc_id"])[0]
        # 字符串化的列表，保留官方原样（单页 / 跨页按它的长度分）："[5, 5]" --> [5, 5]
        evidence_pages = ast.literal_eval(row["evidence_pages"])
        # 金标页：去重排序，只留文档里真实存在的页号
        gold_page_ids = [
            f"{doc_id}_p{page_number}"
            for page_number in sorted(set(evidence_pages))
            if 1 <= page_number <= page_counts[doc_id]
        ]
        evidence_sources = ast.literal_eval(row["evidence_sources"])

        # 答案评测用全部题，含不可回答题
        answers.append(
            {
                "query_id": query_id,
                "question": row["question"],
                "doc_id": doc_id,
                "doc_type": row["doc_type"],
                "answers": [row["answer"]],
                "answer_format": row["answer_format"],
                "evidence_pages": evidence_pages,
                "evidence_sources": evidence_sources,
                "gold_page_ids": gold_page_ids,
            }
        )

        # 检索评测只收有金标页的题：证据页为空 = 不可回答题；页号 0 或超过文档页数 = 标注错误
        if not evidence_pages or len(gold_page_ids) < len(set(evidence_pages)):
            continue
        queries.append(
            {
                "query_id": query_id,
                "question": row["question"],
                "doc_id": doc_id,
                "evidence_sources": evidence_sources,
            }
        )
        qrels.extend(
            {"query_id": query_id, "page_id": page_id, "relevance": 1}
            for page_id in gold_page_ids
        )

    write_jsonl(config.PAGES_PATH, pages)
    write_jsonl(config.QUERIES_PATH, queries)
    write_jsonl(config.QRELS_PATH, qrels)
    write_jsonl(config.ANSWERS_PATH, answers)

    print(f"pages   : {len(pages)}  --> {config.PAGES_PATH}")
    print(
        f"queries : {len(queries)}  --> {config.QUERIES_PATH}（跳过 {len(annotations) - len(queries)} 条）"
    )
    print(f"qrels   : {len(qrels)}  --> {config.QRELS_PATH}")
    print(f"answers : {len(answers)}  --> {config.ANSWERS_PATH}")
    print(f"images  : {config.IMAGE_DIR}")
