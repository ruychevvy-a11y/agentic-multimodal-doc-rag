import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dashscope import MultiModalEmbedding, TextEmbedding

from docrag import config

# 各嵌入模型的接口约束：单次最多几条、返回结果里标原顺序的字段
EMBEDDING_MODEL_LIMITS = {
    "tongyi-embedding-vision-plus": {"batch_size": 20, "index_key": "index"},
    "text-embedding-v4": {"batch_size": 10, "index_key": "text_index"},
    "qwen3.7-text-embedding": {"batch_size": 20, "index_key": "text_index"},
}

# 并发线程数
EMBED_WORKERS = 3

# 临时性失败最多尝试几次（含第一次）
API_MAX_ATTEMPTS = 6


# 把输入切成小批并发调用 --> 按原顺序拼回的结果列表
def run_in_batches(inputs, batch_size, call_fn):
    batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
    vectors = []
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        # pool.map 按 batches 的顺序交回结果
        for batch_vectors in pool.map(call_fn, batches):
            vectors.extend(batch_vectors)
    return vectors


# 调一次 dashscope 接口，限流 / 服务端错误 / 网络异常按指数退避重试，其余错误直接抛
def call_with_retry(call_fn):
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            resp = call_fn()
        except requests.exceptions.ReadTimeout:  # 请求已发出、等回复超时
            raise
        except Exception as err:  # 连接重置、超时等网络层异常
            error = repr(err)
        else:
            if resp.status_code == 200:
                return resp
            error = f"{resp.status_code} {resp.code} {resp.message}"
            transient = (
                resp.status_code == 429
                or resp.status_code >= 500
                or str(resp.code).startswith(("Throttling", "InternalError"))
            )
            if not transient:
                raise RuntimeError(f"dashscope failed: {error}")
        # 等 1, 2, 4 ... 秒（最长 30 秒），带随机抖动
        if attempt < API_MAX_ATTEMPTS - 1:
            time.sleep(min(2**attempt, 30) * random.uniform(0.5, 1.0))
    raise RuntimeError(f"dashscope failed after {API_MAX_ATTEMPTS} attempts: {error}")


# =============================================#
#               image embedding                #
# =============================================#


# 调多模态模型：图像和文本编码进同一个向量空间
def call_multimodal_model(inputs):
    limits = EMBEDDING_MODEL_LIMITS[config.MULTIMODAL_EMBEDDING_MODEL]

    # 发一批，结果按下标字段排回原顺序；lambda 让 call_with_retry 失败时能重新发请求
    def call_one(batch):
        resp = call_with_retry(
            lambda: MultiModalEmbedding.call(
                model=config.MULTIMODAL_EMBEDDING_MODEL,
                input=batch,
                api_key=config.API_KEY,
            )
        )
        ordered = sorted(
            resp.output["embeddings"], key=lambda e: e[limits["index_key"]]
        )
        return [e["embedding"] for e in ordered]

    return run_in_batches(inputs, limits["batch_size"], call_one)


# 页图 --> 图像向量
def embed_image_pages(image_paths):
    # 本地文件要写成 file:// 绝对路径
    urls = [
        "file://" + os.path.abspath(path).replace("\\", "/") for path in image_paths
    ]
    return call_multimodal_model([{"image": url} for url in urls])


# 问题 --> 图像路的问题向量（和页图同一个空间）
def embed_image_queries(questions):
    return call_multimodal_model([{"text": question} for question in questions])


# =============================================#
#                text embedding                #
# =============================================#


# 调文本嵌入模型；text_type 区分文档和问题
def call_text_model(texts, text_type):
    limits = EMBEDDING_MODEL_LIMITS[config.TEXT_EMBEDDING_MODEL]

    # 发一批，结果按下标字段排回原顺序
    def call_one(batch):
        resp = call_with_retry(
            lambda: TextEmbedding.call(
                model=config.TEXT_EMBEDDING_MODEL,
                input=batch,
                text_type=text_type,
                dimension=config.TEXT_EMBEDDING_DIM,
                api_key=config.API_KEY,
            )
        )
        ordered = sorted(
            resp.output["embeddings"], key=lambda e: e[limits["index_key"]]
        )
        return [e["embedding"] for e in ordered]

    return run_in_batches(texts, limits["batch_size"], call_one)


# 页面正文 --> 文本向量；空正文换成一个空格（qwen3.7 批里有空串会整批返回错误维度）
def embed_text_pages(texts):
    return call_text_model(
        [text if text.strip() else " " for text in texts], "document"
    )


# 问题 --> 文本向量
def embed_text_queries(questions):
    return call_text_model(questions, "query")
