import ast
import io
import os

from datasets import Image as ImageFeature
from datasets import load_dataset
from PIL import Image

from docrag import config
from docrag.jsonl import write_jsonl

# 每行最多带 20 张页图：image_1 .. image_20
MAX_DOCUMENT_PAGES = 20


# 流式读 MP-DocVQA val --> 页图 + pages / queries / qrels / answers 四个 jsonl
def build():
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    ds = load_dataset(config.HF_REPO, split="val", streaming=True)
    # 图片列不解码，只在存盘时解码
    for k in range(1, MAX_DOCUMENT_PAGES + 1):
        ds = ds.cast_column(f"image_{k}", ImageFeature(decode=False))
    pages = {}  # page_id --> {page_id, doc_id, image_path}
    queries = []  # {query_id, question, doc_id}
    qrels = []  # {query_id, page_id, relevance}
    answers = []  # {query_id, answers, gold_page_id}

    for row in ds:
        if config.TARGET_QUERIES is not None and len(queries) >= config.TARGET_QUERIES:
            break

        # 字符串化的列表："['txpp0227_p7', 'txpp0227_p8']" --> ['txpp0227_p7', 'txpp0227_p8']
        page_ids = ast.literal_eval(row["page_ids"])
        answers_list = ast.literal_eval(row["answers"])
        images = [row[f"image_{k}"] for k in range(1, MAX_DOCUMENT_PAGES + 1)]
        images = [image for image in images if image is not None]

        for page_id, image in zip(page_ids, images):
            # 同一页被多道题共享，只存一次
            if page_id in pages:
                continue
            image_path = f"{config.IMAGE_DIR}/{page_id}.jpg"
            if not os.path.exists(image_path):
                page_image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                # 只缩不放：长边缩到 MAX_SIDE
                page_image.thumbnail((config.MAX_SIDE, config.MAX_SIDE), Image.Resampling.LANCZOS)
                page_image.save(image_path, "JPEG", quality=90)
            pages[page_id] = {"page_id": page_id, "doc_id": row["doc_id"], "image_path": image_path}

        query_id = row["questionId"]
        queries.append({"query_id": query_id, "question": row["question"], "doc_id": row["doc_id"]})

        gold_page_id = page_ids[int(row["answer_page_idx"])]
        qrels.append({"query_id": query_id, "page_id": gold_page_id, "relevance": 1})
        answers.append({"query_id": query_id, "answers": answers_list, "gold_page_id": gold_page_id})
        if len(queries) % 100 == 0:
            print(f"{len(queries)}/{config.TARGET_QUERIES} queries, {len(pages)} pages")

    write_jsonl(config.PAGES_PATH, pages.values())
    write_jsonl(config.QUERIES_PATH, queries)
    write_jsonl(config.QRELS_PATH, qrels)
    write_jsonl(config.ANSWERS_PATH, answers)

    print(f"pages   : {len(pages)}  --> {config.PAGES_PATH}")
    print(f"queries : {len(queries)}  --> {config.QUERIES_PATH}")
    print(f"qrels   : {len(qrels)}  --> {config.QRELS_PATH}")
    print(f"images  : {config.IMAGE_DIR}")
