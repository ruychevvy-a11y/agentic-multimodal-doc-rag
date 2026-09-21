from pymilvus import RRFRanker

from docrag import config
from docrag.indexing import milvus_client
from docrag.retrieval.dense import image_dense_retrieval, text_dense_retrieval
from docrag.retrieval.sparse import text_bm25_retrieval


# 一路分数 min-max 缩放到 0~1 --> {page_id: 缩放分}；分数全相同时都记 1
def minmax(scores):
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    span = high - low
    return {page_id: (score - low) / span if span else 1.0 for page_id, score in scores.items()}


# 各路按名次投票，由 Milvus hybrid_search 一次算完
class RRFRetrieval:
    def __init__(self, retrievals):
        self.client = milvus_client()
        self.retrievals = retrievals
        self.ranker = RRFRanker(config.RRF_K)

    # 融合检索 --> 前 RETRIEVE_K 个 page_id
    def retrieve(self, question, search_filter):
        res = self.client.hybrid_search(
            collection_name=config.PAGE_COLLECTION,
            reqs=[
                retrieval.search_request(question, config.FUSION_DEPTH, search_filter)
                for retrieval in self.retrievals
            ],
            ranker=self.ranker,
            limit=config.RETRIEVE_K,
        )
        return [hit["page_id"] for hit in res[0]]


# 各路原始分数各自 min-max 后按权重相加，没被某路召回按 0 分
class ConvexRetrieval:
    def __init__(self, retrievals, weights):
        # zip 遇到长短不一会静默截断，个数不一致直接报错
        assert len(retrievals) == len(weights), "检索器个数和权重个数对不上"
        self.client = milvus_client()
        self.retrievals = retrievals
        self.weights = weights

    # 跑一路检索 --> {page_id: 原始分数}；搜索参数和过滤条件由这一路的请求带着
    def route_scores(self, retrieval, question, search_filter):
        request = retrieval.search_request(question, config.FUSION_DEPTH, search_filter)
        res = self.client.search(
            collection_name=config.PAGE_COLLECTION,
            data=request.data,
            anns_field=request.anns_field,
            limit=request.limit,
            search_params=request.param,
            filter=request.filter,
        )
        return {hit["page_id"]: hit["distance"] for hit in res[0]}

    # 融合排序 --> 前 k 个 page_id（重排也从这里取候选）
    def candidates(self, question, k, search_filter):
        fused = {}
        for retrieval, weight in zip(self.retrievals, self.weights):
            route = self.route_scores(retrieval, question, search_filter)
            for page_id, score in minmax(route).items():
                fused[page_id] = fused.get(page_id, 0.0) + weight * score
        return sorted(fused, key=fused.get, reverse=True)[:k]

    # 融合检索 --> 前 RETRIEVE_K 个 page_id
    def retrieve(self, question, search_filter):
        return self.candidates(question, config.RETRIEVE_K, search_filter)


# BM25 + 文本 dense 等权 RRF（MP-DocVQA 基线）
def rrf_bm25_text_retrieval(questions):
    return RRFRetrieval([text_bm25_retrieval(questions), text_dense_retrieval(questions)])


# BM25 + 文本 dense 凸组合；先取权重，没选过权重就在嵌入问题之前报错
def convex_bm25_text_retrieval(questions):
    weights = config.CONVEX_WEIGHTS["convex_bm25_text"]
    return ConvexRetrieval([text_bm25_retrieval(questions), text_dense_retrieval(questions)], weights)


# BM25 + 文本 dense + 图像 dense 凸组合
def convex_bm25_text_image_retrieval(questions):
    weights = config.CONVEX_WEIGHTS["convex_bm25_text_image"]
    return ConvexRetrieval(
        [
            text_bm25_retrieval(questions),
            text_dense_retrieval(questions),
            image_dense_retrieval(questions),
        ],
        weights,
    )
