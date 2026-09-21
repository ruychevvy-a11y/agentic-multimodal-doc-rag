import os
import unicodedata
from collections import Counter
from itertools import groupby
from operator import itemgetter

import pymupdf
from paddleocr import PaddleOCR
from PIL import Image

from docrag import config
from docrag.jsonl import load_jsonl, write_jsonl

# OCR 识别模型一次处理几个文本行小图（不是页数）
TEXT_RECOGNITION_BATCH_SIZE = 16

# PDF 页文本层少于这么多字符就当图片页，走 OCR
TEXT_LAYER_MIN_CHARS = 50

# 文本层控制字符占比超过这个值就当乱码（字体编码坏了），走 OCR
GARBLED_CONTROL_RATIO = 0.2


# 页图 OCR --> Page 结构 {source, text, blocks, coord_size}
def parse_scanned_page(ocr, image_path):
    # bbox 落在页图的像素坐标系里
    coord_size = Image.open(image_path).size
    res = ocr.predict(image_path)
    if not res:
        return {
            "source": "ocr",
            "text": "",
            "blocks": [],
            "coord_size": list(coord_size),
        }

    result = res[0]
    texts = result.get("rec_texts", [])
    boxes = result.get("rec_boxes", [])
    scores = result.get("rec_scores", [])

    # 一个 block = 一个 OCR 文本行，带 bbox 和置信度
    blocks = [
        {"text": text, "bbox": [int(p) for p in box], "score": round(float(score), 4)}
        for text, box, score in zip(texts, boxes, scores)
    ]
    return {
        "source": "ocr",
        "text": " ".join(texts),
        "blocks": blocks,
        "coord_size": list(coord_size),
    }


# 文本层是否乱码
def is_garbled(text):
    control_chars = sum(
        unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in text
    )
    return control_chars / len(text) > GARBLED_CONTROL_RATIO


# PDF 文本层直接提取 --> Page 结构 {source, text, blocks, coord_size}
def parse_pdf_text_page(page, coord_size):
    # PDF 用 point 坐标，页图用像素坐标，scale 是换算比例
    scale = coord_size[0] / page.rect.width

    # 一个 block = PDF 自带的一个文本块（约一个段落）；sort=True 自上而下排，block_type 0 是文字
    blocks = [
        {
            "text": " ".join(text.split()),
            "bbox": [
                int(x0 * scale),
                int(y0 * scale),
                int(x1 * scale),
                int(y1 * scale),
            ],
        }
        for x0, y0, x1, y1, text, block_number, block_type in page.get_text(
            "blocks", sort=True
        )
        if block_type == 0 and text.strip()
    ]
    return {
        "source": "pdf_text",
        "text": " ".join(block["text"] for block in blocks),
        "blocks": blocks,
        "coord_size": list(coord_size),
    }


# 一页 PDF --> Page 结构：有文本层直接提取，图片型页和乱码页走 OCR
def parse_pdf_page(row, document, ocr):
    # PyMuPDF 下标 0 起，page_number 1 起
    page = document[row["page_number"] - 1]
    text = page.get_text("text")
    if len(text) < TEXT_LAYER_MIN_CHARS or is_garbled(text):
        return parse_scanned_page(ocr, row["image_path"])
    return parse_pdf_text_page(page, Image.open(row["image_path"]).size)


# pages.jsonl 逐页解析 --> corpus.jsonl
def parse_corpus():
    # 扫描件不需要方向分类和弯曲展平，关掉这三个预处理模型
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=TEXT_RECOGNITION_BATCH_SIZE,
    )
    rows = load_jsonl(config.PAGES_PATH)

    if config.DATASET_SOURCE == "pdf":
        # 同一份 PDF 的页在 pages.jsonl 里连着：按文档分组，一份 PDF 只打开一次
        parsed = 0
        for doc_id, doc_rows in groupby(rows, key=itemgetter("doc_id")):
            with pymupdf.open(
                os.path.join(config.PDF_DIR, f"{doc_id}.pdf")
            ) as document:
                for row in doc_rows:
                    row.update(parse_pdf_page(row, document, ocr))
                    parsed += 1
            print(f"{parsed}/{len(rows)} parse {doc_id}")
    else:
        for i, row in enumerate(rows, 1):
            row.update(parse_scanned_page(ocr, row["image_path"]))
            if i % 25 == 0:
                print(f"{i}/{len(rows)} parse")

    write_jsonl(config.CORPUS_PATH, rows)
    print("done -->", config.CORPUS_PATH, dict(Counter(row["source"] for row in rows)))
