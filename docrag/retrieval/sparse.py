from pymilvus import AnnSearchRequest

from docrag import config
from docrag.indexing import milvus_client


# 问题原文交给 Milvus 内置 BM25 打分
class BM25Retrieval:
    # questions 用不到（BM25 不需要预先嵌入），保留参数是为了和其他检索器的工厂签名一致
    def __init__(self, questions):
        self.client = milvus_client()

    # 单路检索 --> 前 RETRIEVE_K 个 page_id
    def retrieve(self, question, search_filter):
        res = self.client.search(
            collection_name=config.PAGE_COLLECTION,
            data=[question],
            anns_field=config.BM25_VECTOR_FIELD,
            limit=config.RETRIEVE_K,
            filter=search_filter,
        )
        return [hit["page_id"] for hit in res[0]]

    # 给融合层的一路搜索请求
    def search_request(self, question, depth, search_filter):
        return AnnSearchRequest(
            data=[question],
            anns_field=config.BM25_VECTOR_FIELD,
            param={},
            limit=depth,
            filter=search_filter,
        )


# BM25 路
def text_bm25_retrieval(questions):
    return BM25Retrieval(questions)
