from pymilvus import AnnSearchRequest

from docrag import config
from docrag.embeddings import embed_image_queries, embed_text_queries
from docrag.indexing import milvus_client

# dense 路的 HNSW 搜索参数
SEARCH_PARAMS = {"params": {"ef": config.HNSW_SEARCH_EF}}


# 问题向量在指定向量字段里找最近的页
class DenseRetrieval:
    def __init__(self, vector_field, embed_query_fn, questions):
        self.client = milvus_client()
        self.vector_field = vector_field
        self.embed_query_fn = embed_query_fn
        # 去重后一次性嵌入全部问题 --> {question: vector}
        unique_questions = list(dict.fromkeys(questions))
        self.query_embeddings = dict(
            zip(unique_questions, embed_query_fn(unique_questions))
        )

    # 问题 --> 向量；没嵌入过的（如 agent 临时写的查询）现嵌入并记进缓存
    def query_vector(self, question):
        if question not in self.query_embeddings:
            self.query_embeddings[question] = self.embed_query_fn([question])[0]
        return self.query_embeddings[question]

    # 单路检索 --> 前 RETRIEVE_K 个 page_id
    # search_filter："" 全库（开域），'doc_id == "xxx"' 只在这份文档里搜（闭域）
    def retrieve(self, question, search_filter):
        res = self.client.search(
            collection_name=config.PAGE_COLLECTION,
            data=[self.query_vector(question)],
            anns_field=self.vector_field,
            limit=config.RETRIEVE_K,
            search_params=SEARCH_PARAMS,
            filter=search_filter,
        )
        return [hit["page_id"] for hit in res[0]]

    # 给融合层的一路搜索请求
    def search_request(self, question, depth, search_filter):
        return AnnSearchRequest(
            data=[self.query_vector(question)],
            anns_field=self.vector_field,
            param=SEARCH_PARAMS,
            limit=depth,
            filter=search_filter,
        )


# 图像 dense 路
def image_dense_retrieval(questions):
    return DenseRetrieval(config.IMAGE_VECTOR_FIELD, embed_image_queries, questions)


# 文本 dense 路
def text_dense_retrieval(questions):
    return DenseRetrieval(config.TEXT_VECTOR_FIELD, embed_text_queries, questions)
