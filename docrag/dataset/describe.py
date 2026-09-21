import json
import os
from concurrent.futures import ThreadPoolExecutor

from dashscope import MultiModalConversation

from docrag import config
from docrag.embeddings import call_with_retry
from docrag.generation.answer import message_text
from docrag.jsonl import load_jsonl

# 并发线程数
DESCRIBE_WORKERS = 12

# 视觉描述指令：只写图、表、照片等视觉内容，不抄正文；没有视觉内容回 NONE
DESCRIBE_PROMPT = (
    "Describe the visual content of this document page so that the page can be found by search. "
    "For every chart, figure, diagram, photo, map, table or infographic on the page, write its type, its title or caption, "
    "what it shows (axes, legend categories, key values, the highest and lowest items, trends) and notable colours, people or objects. "
    "Do not copy ordinary paragraphs of text. If the page has no visual content, reply exactly NONE. At most 150 words."
)


# 一页页图 --> {page_id, description, token}；回 NONE 或内容审核报错时 description 记空串
def describe_page(page):
    image_url = "file://" + os.path.abspath(page["image_path"]).replace("\\", "/")
    try:
        resp = call_with_retry(
            lambda: MultiModalConversation.call(
                model=config.DESCRIBE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [{"image": image_url}, {"text": DESCRIBE_PROMPT}],
                    }
                ],
                enable_thinking=False,
                api_key=config.API_KEY,
            )
        )
    except RuntimeError as error:
        return {"page_id": page["page_id"], "description": "", "error": repr(error)}
    text = message_text(resp.output.choices[0].message).strip()
    return {
        "page_id": page["page_id"],
        # "NONE" / "**NONE**" / "NONE." --> ""
        "description": "" if text.strip("*. ").upper() == "NONE" else text,
        "input_tokens": resp.usage["input_tokens"],
        "output_tokens": resp.usage["output_tokens"],
    }


# pages.jsonl 逐页生成视觉描述 --> descriptions.jsonl（逐页追加，已描述的页跳过）
def describe_pages():
    pages = load_jsonl(config.PAGES_PATH)
    done = (
        {row["page_id"] for row in load_jsonl(config.DESCRIPTIONS_PATH)}
        if os.path.exists(config.DESCRIPTIONS_PATH)
        else set()
    )
    pending = [page for page in pages if page["page_id"] not in done]
    print(f"{len(pages)} 页：已描述 {len(done)}，待描述 {len(pending)}")

    # pool.map 按 pending 的顺序交回，逐页落盘，中断后重跑只补没描述的页
    with ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool, open(
        config.DESCRIPTIONS_PATH, "a", encoding="utf-8"
    ) as f:
        for i, row in enumerate(pool.map(describe_page, pending), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 500 == 0:
                print(f"{i}/{len(pending)} described")
    print("done -->", config.DESCRIPTIONS_PATH)
