from dashscope import TextReRank

from docrag import config
from docrag.embeddings import call_with_retry
from docrag.indexing import milvus_client
from docrag.retrieval.fusion import convex_bm25_text_image_retrieval, convex_bm25_text_retrieval


# 重排模型给每页正文打相关分 --> 与 texts 同序的分数；空正文不送接口（接口不接受），分数记 0
def rerank_scores(question, texts):
    scores = [0.0] * len(texts)
    # 非空正文的下标，例：['a', '', 'b'] --> [0, 2]
    kept = [i for i, text in enumerate(texts) if text.strip()]
    if not kept:
        return scores
    resp = call_with_retry(
        lambda: TextReRank.call(
            model=config.RERANK_MODEL,
            query=question,
            documents=[texts[i] for i in kept],
            instruct=config.RERANK_INSTRUCT,
            api_key=config.API_KEY,
        )
    )
    # index 是在送进去的非空列表里的下标，例：kept=[0, 2]，{'index': 1, 'relevance_score': 0.9} --> scores[2] = 0.9
    for result in resp.output["results"]:
        scores[kept[result["index"]]] = result["relevance_score"]
    return scores


# 第一阶段取前 RERANK_DEPTH 页，重排后返回前 RETRIEVE_K 页
class RerankRetrieval:
    def __init__(self, first_stage):
        self.client = milvus_client()
        self.first_stage = first_stage

    # 按 page_id 回 Milvus 取正文 --> {page_id: text}（get 不保证顺序）
    def page_texts(self, page_ids):
        rows = self.client.get(collection_name=config.PAGE_COLLECTION, ids=page_ids, output_fields=["text"])
        return {row["page_id"]: row["text"] for row in rows}

    # 重排检索 --> 前 RETRIEVE_K 个 page_id；同分保持第一阶段名次
    def retrieve(self, question, search_filter):
        candidates = self.first_stage.candidates(question, config.RERANK_DEPTH, search_filter)
        texts = self.page_texts(candidates)
        scores = rerank_scores(question, [texts[page_id] for page_id in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [page_id for page_id, score in ranked[: config.RETRIEVE_K]]


# 两路凸组合出候选再重排
def rerank_convex_bm25_text_retrieval(questions):
    return RerankRetrieval(convex_bm25_text_retrieval(questions))


# 三路凸组合出候选再重排
def rerank_convex_bm25_text_image_retrieval(questions):
    return RerankRetrieval(convex_bm25_text_image_retrieval(questions))
